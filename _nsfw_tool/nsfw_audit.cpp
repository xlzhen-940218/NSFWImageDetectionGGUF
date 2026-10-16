// NSFW 图像审核工具 (C++ 版，对应 nsfw_audit.py)
//
// 功能：扫描目录中的图片 -> stb_image 解码 -> 与 Pillow BILINEAR 对齐的
// 224x224 预处理 -> ViT-B/16 (GGUF) 批量推理，Vulkan 后端原生支持
// NVIDIA / AMD / Intel 显卡，无 GPU 时回退 CPU -> nsfw 概率超阈值的
// 图片移动到 NSFW 文件夹。
//
// 用法：
//   nsfw_audit.exe [--dir 目录] [--threshold 0.9] [--batch 8]
//                  [--device auto|vulkan|cpu] [--model 路径]
//                  [--nsfw-dir 目录] [--recursive] [--dry-run]

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244 4267 4505 4996)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO  // 一律走内存加载，保证中文路径可用
#include "stb_image.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================== 常量 ==============================
static constexpr int IMAGE_SIZE = 224;
static constexpr int PATCH      = 16;
static constexpr int GRID       = IMAGE_SIZE / PATCH;  // 14
static constexpr int N_PATCH    = GRID * GRID;        // 196
static constexpr int HIDDEN     = 768;
static constexpr int N_CLASS    = 2;
static constexpr int FRAME_FLOATS = N_PATCH * HIDDEN;

// ============================== 编码辅助 ==============================
static std::string escape_json(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", (int)(unsigned char)c);
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

static std::string narrow(const std::wstring & w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(),
                                nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(),
                        s.data(), n, nullptr, nullptr);
    return s;
}

static std::wstring widen(const std::string & s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), w.data(), n);
    return w;
}

static fs::path exe_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(std::wstring(buf, n)).parent_path();
}

// ============================== 图像预处理 ==============================
// 与 Pillow 12 的 precompute_coeffs + normalize_coeffs_8bpc 完全一致：
// double 权重归一化后，量化为 PRECISION_BITS=22 位的 int32 系数
static constexpr int PRECISION_BITS = 32 - 8 - 2;  // 22，与 Pillow 相同

struct Axis1D {
    std::vector<std::vector<int>> idx;     // 每个输出位置的采样下标
    std::vector<std::vector<int32_t>> k;   // 量化后的整数系数
};

static Axis1D bilinear_weights(int n_in, int n_out) {
    const double scale = (double) n_in / n_out;
    const double fs = (std::max)(scale, 1.0);
    const double support = fs;  // bilinear support = 1
    Axis1D a;
    a.idx.resize(n_out);
    a.k.resize(n_out);
    for (int o = 0; o < n_out; ++o) {
        const double center = (o + 0.5) * scale;
        int xmin = (int) std::floor(center - support + 0.5);
        int xmax = (int) std::floor(center + support + 0.5);
        xmin = (std::max)(0, (std::min)(xmin, n_in));
        xmax = (std::max)(0, (std::min)(xmax, n_in));
        double ww = 0.0;
        std::vector<double> raw;
        for (int x = xmin; x < xmax; ++x) {
            const double t = (x + 0.5 - center) / fs;
            const double at = std::fabs(t);
            double wv = at < 1.0 ? (1.0 - at) / fs : 0.0;
            raw.push_back(wv);
            ww += wv;
        }
        if (ww != 0.0) {
            for (double & v : raw) v /= ww;
        }
        a.idx[o].reserve(raw.size());
        a.k[o].reserve(raw.size());
        const double q = (double) (1 << PRECISION_BITS);
        for (size_t i = 0; i < raw.size(); ++i) {
            a.idx[o].push_back(xmin + (int) i);
            a.k[o].push_back((int32_t) std::floor(raw[i] * q + 0.5));
        }
    }
    return a;
}

// Pillow 的 clip8: ss 带 2^(22-1) 的四舍五入偏移，算术右移后裁剪到 0..255
static inline uint8_t clip8(int ss) {
    int v = ss >> PRECISION_BITS;  // bilinear 权重非负，算术/逻辑右移等价
    return (uint8_t) (v < 0 ? 0 : (v > 255 ? 255 : v));
}

// 解码 + BILINEAR 224x224 + 归一化 + patch unfold
// 成功返回 true，out 为 N_PATCH*HIDDEN 个 float
static bool preprocess(const fs::path & path, std::vector<float> & out) {
    // 用宽字符路径读入内存，避免 stb 的 ANSI fopen 在中文路径上失败
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
    if (bytes.empty()) return false;

    int W = 0, H = 0, C = 0;
    stbi_uc * img = stbi_load_from_memory(bytes.data(), (int) bytes.size(),
                                          &W, &H, &C, 3);
    if (!img) return false;

    const Axis1D ax = bilinear_weights(W, IMAGE_SIZE);
    const Axis1D ay = bilinear_weights(H, IMAGE_SIZE);

    // 水平阶段（对齐 Pillow ImagingResampleHorizontal_8bpc 的 3-band 路径）：
    // int32 累加，bias=1<<21 四舍五入，clip8 量化回 uint8 中间图 [H,224,3]
    std::vector<uint8_t> tmp((size_t) H * IMAGE_SIZE * 3);
    for (int y = 0; y < H; ++y) {
        const stbi_uc * row = img + (size_t) y * W * 3;
        for (int ox = 0; ox < IMAGE_SIZE; ++ox) {
            const auto & ids = ax.idx[ox];
            const auto & kk  = ax.k[ox];
            int sr = 1 << (PRECISION_BITS - 1);
            int sg = sr, sb = sr;
            for (size_t i = 0; i < ids.size(); ++i) {
                const stbi_uc * px = row + (size_t) ids[i] * 3;
                const int w = kk[i];
                sr += px[0] * w; sg += px[1] * w; sb += px[2] * w;
            }
            uint8_t * dst = tmp.data() + ((size_t) y * IMAGE_SIZE + ox) * 3;
            dst[0] = clip8(sr); dst[1] = clip8(sg); dst[2] = clip8(sb);
        }
    }
    stbi_image_free(img);

    // 垂直阶段（对齐 ImagingResampleVertical_8bpc）：uint8 -> uint8 [224,224,3]
    std::vector<uint8_t> rgb((size_t) IMAGE_SIZE * IMAGE_SIZE * 3);
    for (int oy = 0; oy < IMAGE_SIZE; ++oy) {
        const auto & ids = ay.idx[oy];
        const auto & kk  = ay.k[oy];
        for (int ox = 0; ox < IMAGE_SIZE; ++ox) {
            int sr = 1 << (PRECISION_BITS - 1);
            int sg = sr, sb = sr;
            for (size_t i = 0; i < ids.size(); ++i) {
                const uint8_t * s = tmp.data()
                    + ((size_t) ids[i] * IMAGE_SIZE + ox) * 3;
                const int w = kk[i];
                sr += s[0] * w; sg += s[1] * w; sb += s[2] * w;
            }
            uint8_t * d = rgb.data() + ((size_t) oy * IMAGE_SIZE + ox) * 3;
            d[0] = clip8(sr); d[1] = clip8(sg); d[2] = clip8(sb);
        }
    }

    // 归一化 (/127.5 - 1) 与 patch unfold
    // patch 向量内层顺序 (C, kh, kw)，与 GGUF 中线性化的卷积权重一致
    out.resize(FRAME_FLOATS);
    for (int gy = 0; gy < GRID; ++gy) {
        for (int gx = 0; gx < GRID; ++gx) {
            const int p = gy * GRID + gx;
            for (int ky = 0; ky < PATCH; ++ky) {
                const int oy = gy * PATCH + ky;
                for (int kx = 0; kx < PATCH; ++kx) {
                    const int ox = gx * PATCH + kx;
                    const uint8_t * s = rgb.data()
                        + ((size_t) oy * IMAGE_SIZE + ox) * 3;
                    float * dst = out.data()
                        + ((size_t) p * HIDDEN)
                        + (ky * PATCH + kx);
                    dst[0]               = s[0] / 127.5f - 1.0f;
                    dst[PATCH * PATCH]     = s[1] / 127.5f - 1.0f;
                    dst[2 * PATCH * PATCH] = s[2] / 127.5f - 1.0f;
                }
            }
        }
    }
    return true;
}

// ============================== ViT 引擎 ==============================
struct vit_hparams {
    int n_layer = 12, n_head = 12, hidden = 768, inter = 3072;
    int n_patch = 196, n_class = 2, n_seq = 197;
    float eps = 1e-12f;
};

struct vit_block {
    ggml_tensor *q_w = nullptr, *q_b = nullptr, *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr, *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr, *fc2_w = nullptr, *fc2_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
};

struct vit_model {
    ~vit_model() {
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (backend) ggml_backend_free(backend);
        if (ctx_w) ggml_free(ctx_w);
    }
    vit_hparams hp;
    ggml_tensor *patch_w = nullptr, *patch_b = nullptr;
    ggml_tensor *cls_token = nullptr, *pos_embed = nullptr;
    std::vector<vit_block> blocks;
    ggml_tensor *lnf_w = nullptr, *lnf_b = nullptr;
    ggml_tensor *head_w = nullptr, *head_b = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context *ctx_w = nullptr;
};

static ggml_tensor * need_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) throw std::runtime_error("GGUF 中缺少张量: " + name);
    return t;
}

static int32_t kv_i32(gguf_context * g, const char * k, int32_t def) {
    int i = gguf_find_key(g, k);
    return i >= 0 ? gguf_get_val_i32(g, i) : def;
}

static float kv_f32(gguf_context * g, const char * k, float def) {
    int i = gguf_find_key(g, k);
    return i >= 0 ? gguf_get_val_f32(g, i) : def;
}

static ggml_backend_t make_backend(const std::string & want) {
    ggml_backend_dev_t best = nullptr;
    int best_score = -1;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        std::string name = props.name ? props.name : "";
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        int score = -1;
        if (want == "cpu" && props.type == GGML_BACKEND_DEVICE_TYPE_CPU) score = 1000;
        else if (want == "vulkan" && name.find("vulkan") != std::string::npos) score = 900;
        else if (want == "auto" && name.find("vulkan") != std::string::npos) score = 500;
        else if (want == "auto" && props.type != GGML_BACKEND_DEVICE_TYPE_CPU) score = 300;
        if (score > best_score) { best_score = score; best = dev; }
    }
    if (!best) return ggml_backend_cpu_init();
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(best, &props);
    ggml_backend_t be = ggml_backend_dev_init(best, nullptr);
    if (be) {
        std::printf("[BACKEND] %s - %s\n", props.name,
                    props.description ? props.description : "");
        std::fflush(stdout);
    }
    return be;
}

static bool vit_load(const std::string & fname, const std::string & device,
                     vit_model & m) {
    gguf_init_params gp{true, &m.ctx_w};
    gguf_context * g = gguf_init_from_file(fname.c_str(), gp);
    if (!g) {
        std::fprintf(stderr, "[ERROR] Failed to open GGUF model: %s\n", fname.c_str());
        std::fflush(stderr);
        return false;
    }

    m.hp.n_layer = kv_i32(g, "vit.n_layer", 12);
    m.hp.n_head  = kv_i32(g, "vit.n_head", 12);
    m.hp.hidden  = kv_i32(g, "vit.hidden", 768);
    m.hp.inter   = kv_i32(g, "vit.intermediate", 3072);
    m.hp.n_patch = kv_i32(g, "vit.n_patch", 196);
    m.hp.n_class = kv_i32(g, "vit.n_class", 2);
    m.hp.n_seq   = m.hp.n_patch + 1;
    m.hp.eps     = kv_f32(g, "vit.layer_norm_eps", 1e-12f);

    m.backend = make_backend(device);
    if (!m.backend) { gguf_free(g); return false; }
    m.buf_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    if (!m.buf_w) {
        std::fprintf(stderr, "[ERROR] Failed to allocate device memory for weights\n");
        std::fflush(stderr);
        gguf_free(g);
        return false;
    }

    try {
        m.patch_w   = need_tensor(m.ctx_w, "patch_w");
        m.patch_b   = need_tensor(m.ctx_w, "patch_b");
        m.cls_token = need_tensor(m.ctx_w, "cls_token");
        m.pos_embed = need_tensor(m.ctx_w, "pos_embed");
        m.blocks.resize(m.hp.n_layer);
        char buf[128];
        for (int i = 0; i < m.hp.n_layer; ++i) {
            auto & b = m.blocks[i];
            std::snprintf(buf, sizeof(buf), "blocks.%d.", i);
            std::string p = buf;
            b.q_w = need_tensor(m.ctx_w, p + "attn_q_w");
            b.q_b = need_tensor(m.ctx_w, p + "attn_q_b");
            b.k_w = need_tensor(m.ctx_w, p + "attn_k_w");
            b.k_b = need_tensor(m.ctx_w, p + "attn_k_b");
            b.v_w = need_tensor(m.ctx_w, p + "attn_v_w");
            b.v_b = need_tensor(m.ctx_w, p + "attn_v_b");
            b.o_w = need_tensor(m.ctx_w, p + "attn_o_w");
            b.o_b = need_tensor(m.ctx_w, p + "attn_o_b");
            b.ln1_w = need_tensor(m.ctx_w, p + "ln1_w");
            b.ln1_b = need_tensor(m.ctx_w, p + "ln1_b");
            b.fc1_w = need_tensor(m.ctx_w, p + "mlp_fc1_w");
            b.fc1_b = need_tensor(m.ctx_w, p + "mlp_fc1_b");
            b.fc2_w = need_tensor(m.ctx_w, p + "mlp_fc2_w");
            b.fc2_b = need_tensor(m.ctx_w, p + "mlp_fc2_b");
            b.ln2_w = need_tensor(m.ctx_w, p + "ln2_w");
            b.ln2_b = need_tensor(m.ctx_w, p + "ln2_b");
        }
        m.lnf_w  = need_tensor(m.ctx_w, "lnf_w");
        m.lnf_b  = need_tensor(m.ctx_w, "lnf_b");
        m.head_w = need_tensor(m.ctx_w, "head_w");
        m.head_b = need_tensor(m.ctx_w, "head_b");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        std::fflush(stderr);
        gguf_free(g);
        return false;
    }

    FILE * f = std::fopen(fname.c_str(), "rb");
    if (!f) { gguf_free(g); return false; }
    const int n_tensors = gguf_get_n_tensors(g);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(m.ctx_w, name);
        size_t offs = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
        size_t nb = ggml_nbytes(t);
        std::vector<uint8_t> buf(nb);
        if (std::fseek(f, (long) offs, SEEK_SET) != 0
            || std::fread(buf.data(), 1, nb, f) != nb) {
            std::fclose(f); gguf_free(g); return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nb);
    }
    std::fclose(f);
    gguf_free(g);

    std::printf("[MODEL] layers=%d heads=%d hidden=%d patches=%d classes=%d\n",
                m.hp.n_layer, m.hp.n_head, m.hp.hidden, m.hp.n_patch, m.hp.n_class);
    std::fflush(stdout);
    return true;
}

static ggml_tensor * layernorm(ggml_context * ctx, const vit_hparams & hp,
                               ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, hp.eps);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

static ggml_cgraph * vit_graph(const vit_model & m, int B) {
    const vit_hparams & hp = m.hp;
    const int D = hp.hidden, S = hp.n_seq, H = hp.n_head, Dh = D / H;

    static std::vector<uint8_t> gbuf;
    if (gbuf.empty())
        gbuf.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead());
    ggml_init_params ip{gbuf.size(), gbuf.data(), true};
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, hp.n_patch, B);
    ggml_set_name(x, "input");
    ggml_set_input(x);

    x = ggml_add(ctx, ggml_mul_mat(ctx, m.patch_w, x), m.patch_b);
    x = ggml_reshape_4d(ctx, x, D, hp.n_patch, 1, B);
    ggml_tensor * cls = ggml_reshape_4d(ctx, m.cls_token, D, 1, 1, 1);
    cls = ggml_repeat_4d(ctx, cls, D, 1, 1, B);
    x = ggml_concat(ctx, cls, x, 1);
    x = ggml_reshape_3d(ctx, x, D, S, B);
    x = ggml_add(ctx, x, ggml_reshape_3d(ctx, m.pos_embed, D, S, 1));

    for (int i = 0; i < hp.n_layer; ++i) {
        const vit_block & blk = m.blocks[i];
        ggml_tensor * h = layernorm(ctx, hp, x, blk.ln1_w, blk.ln1_b);
        ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, blk.q_w, h), blk.q_b);
        ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, blk.k_w, h), blk.k_b);
        ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, blk.v_w, h), blk.v_b);
        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, Dh, H, S, B), 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, Dh, H, S, B), 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, Dh, H, S, B), 0, 2, 1, 3));

        ggml_tensor * att = ggml_mul_mat(ctx, k, q);
        att = ggml_scale(ctx, att, 1.0f / std::sqrt((float) Dh));
        att = ggml_soft_max(ctx, att);
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        att = ggml_mul_mat(ctx, vt, att);
        att = ggml_cont(ctx, ggml_permute(ctx, att, 0, 2, 1, 3));
        att = ggml_reshape_3d(ctx, att, D, S, B);
        att = ggml_add(ctx, ggml_mul_mat(ctx, blk.o_w, att), blk.o_b);
        x = ggml_add(ctx, x, att);

        h = layernorm(ctx, hp, x, blk.ln2_w, blk.ln2_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, blk.fc1_w, h), blk.fc1_b);
        h = ggml_gelu(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, blk.fc2_w, h), blk.fc2_b);
        x = ggml_add(ctx, x, h);
    }

    x = layernorm(ctx, hp, x, m.lnf_w, m.lnf_b);
    ggml_tensor * cls_out = ggml_view_3d(ctx, x, D, 1, B, x->nb[1], x->nb[2], 0);
    cls_out = ggml_cont(ctx, cls_out);
    ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, m.head_w, cls_out), m.head_b);
    logits = ggml_reshape_3d(ctx, logits, hp.n_class, 1, B);
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    probs = ggml_reshape_2d(ctx, probs, hp.n_class, B);
    ggml_set_name(probs, "probs");
    ggml_set_output(probs);
    ggml_build_forward_expand(gf, probs);
    return gf;
}

struct vit_runner {
    int batch;
    vit_model model;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * in = nullptr;
    ggml_tensor * probs = nullptr;
    std::vector<float> in_buf, out_buf;

    bool init(const std::string & model_path, int B, const std::string & device) {
        batch = B;
        if (!vit_load(model_path, device, model)) return false;
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        gf = vit_graph(model, batch);
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            std::fprintf(stderr, "[ERROR] Failed to allocate compute graph memory\n");
            std::fflush(stderr);
            return false;
        }
        in = ggml_graph_get_tensor(gf, "input");
        probs = ggml_graph_get_tensor(gf, "probs");
        in_buf.assign((size_t) batch * FRAME_FLOATS, 0.0f);
        out_buf.assign((size_t) batch * N_CLASS, 0.0f);
        ggml_backend_tensor_set(in, in_buf.data(), 0, in_buf.size() * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[ERROR] Warmup inference failed\n");
            std::fflush(stderr);
            return false;
        }
        ggml_backend_synchronize(model.backend);
        return true;
    }

    // frames: 1..batch 个帧；返回 n*2 概率
    std::vector<float> infer(const std::vector<const std::vector<float>*> & frames) {
        const int n = (int) frames.size();
        for (int i = 0; i < batch; ++i) {
            const auto & src = *frames[i < n ? i : n - 1];  // 末批用最后一帧补齐
            std::memcpy(in_buf.data() + (size_t) i * FRAME_FLOATS,
                        src.data(), FRAME_FLOATS * sizeof(float));
        }
        ggml_backend_tensor_set(in, in_buf.data(), 0, in_buf.size() * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Inference execution failed");
        ggml_backend_tensor_get(probs, out_buf.data(), 0, out_buf.size() * sizeof(float));
        return std::vector<float>(out_buf.begin(), out_buf.begin() + (size_t) n * N_CLASS);
    }

    ~vit_runner() { if (alloc) ggml_gallocr_free(alloc); }
};

// ============================== 文件操作 ==============================
static const std::vector<std::wstring> IMAGE_EXTS = {
    L".jpg", L".jpeg", L".png", L".webp", L".bmp", L".gif", L".tif", L".tiff"
};

static bool is_image(const fs::path & p) {
    std::wstring e = p.extension().wstring();
    std::transform(e.begin(), e.end(), e.begin(),
                   [](wchar_t c) { return (wchar_t) std::towlower(c); });
    for (const auto & x : IMAGE_EXTS) if (e == x) return true;
    return false;
}

static std::wstring lower(std::wstring s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](wchar_t c) { return (wchar_t) std::towlower(c); });
    return s;
}

static bool starts_with_ci(std::wstring s, const std::wstring & prefix) {
    s = lower(std::move(s));
    if (s.size() < prefix.size()) return false;
    if (s.compare(0, prefix.size(), lower(prefix)) != 0) return false;
    return s.size() == prefix.size() || s[prefix.size()] == L'\\' || s[prefix.size()] == L'/';
}

static std::vector<fs::path> collect(const fs::path & root, bool recursive,
                                     const fs::path & nsfw_dir) {
    std::vector<fs::path> out;
    std::error_code ec;
    const std::wstring nd = nsfw_dir.wstring();
    if (recursive) {
        for (auto it = fs::recursive_directory_iterator(
                 root, fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec) || !is_image(it->path())) continue;
            if (starts_with_ci(it->path().wstring(), nd)) continue;
            out.push_back(it->path());
        }
    } else {
        for (auto it = fs::directory_iterator(root, ec);
             !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec) || !is_image(it->path())) continue;
            out.push_back(it->path());
        }
    }
    std::sort(out.begin(), out.end(),
              [](const fs::path & a, const fs::path & b) {
                  return lower(a.wstring()) < lower(b.wstring());
              });
    return out;
}

static fs::path unique_dest(const fs::path & dst) {
    if (!fs::exists(dst)) return dst;
    for (int i = 1;; ++i) {
        fs::path cand = dst.parent_path()
            / (dst.stem().wstring() + L" (" + std::to_wstring(i) + L")" + dst.extension().wstring());
        if (!fs::exists(cand)) return cand;
    }
}

static bool move_file(const fs::path & src, const fs::path & dst) {
    std::error_code ec;
    fs::rename(src, dst, ec);
    if (!ec) return true;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) return false;
    fs::remove(src, ec);
    return true;
}

// ============================== 入口 ==============================
static void usage() {
    std::printf(
        "Usage: nsfw_audit.exe [options]\n"
        "  --dir <path>          Directory to audit (default: current directory)\n"
        "  --threshold <float>   NSFW probability threshold (default: 0.90)\n"
        "  --batch <int>         GPU batch size (default: 8, range 1..128)\n"
        "  --device <dev>        Device backend: auto|vulkan|cpu (default: auto)\n"
        "  --model <path>        Path to GGUF model file\n"
        "  --nsfw-dir <path>     Destination directory for NSFW images (default: <dir>/NSFW)\n"
        "  --recursive           Recursively scan subdirectories\n"
        "  --dry-run             Dry run mode (scan only, do not move files)\n"
        "  --format <fmt>        Output format: text|json (default: text)\n"
        "  --json                Shorthand for --format json\n"
        "  -h, --help            Display this help message\n");
    std::fflush(stdout);
}

int wmain(int argc, wchar_t ** argvW) {
    SetConsoleOutputCP(CP_UTF8);

    fs::path root = fs::current_path();
    double threshold = 0.9;
    int batch = 8;
    std::string device = "auto";
    std::string model_arg;
    fs::path nsfw_dir_arg;
    bool recursive = false, dry_run = false;
    std::string format = "text";

    for (int i = 1; i < argc; ++i) {
        std::string a = narrow(argvW[i]);
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing argument for ") + name);
            return narrow(argvW[++i]);
        };
        if (a == "--dir") root = fs::path(widen(need("--dir")));
        else if (a == "--threshold") threshold = std::stod(need("--threshold"));
        else if (a == "--batch") batch = std::stoi(need("--batch"));
        else if (a == "--device") device = need("--device");
        else if (a == "--model") model_arg = need("--model");
        else if (a == "--nsfw-dir") nsfw_dir_arg = fs::path(widen(need("--nsfw-dir")));
        else if (a == "--recursive") recursive = true;
        else if (a == "--dry-run") dry_run = true;
        else if (a == "--format") format = need("--format");
        else if (a == "--json") format = "json";
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else {
            std::fprintf(stderr, "[ERROR] Unknown option: %s\n", a.c_str());
            usage();
            return 1;
        }
    }
    const bool is_json = (format == "json");

    if (batch < 1 || batch > 128) {
        std::fprintf(stderr, "[ERROR] batch must be in range 1..128\n");
        return 1;
    }
    if (device != "auto" && device != "vulkan" && device != "cpu") {
        std::fprintf(stderr, "[ERROR] device must be auto|vulkan|cpu\n");
        return 1;
    }

    fs::path model_path = model_arg.empty()
        ? (exe_dir().parent_path().parent_path() / "models" / "nsfw_image_detection.fp32.gguf")
        : fs::path(widen(model_arg));
    if (!fs::exists(model_path)) {
        std::string mstr = narrow(model_path.wstring());
        if (is_json) {
            std::printf("{\"type\":\"error\",\"message\":\"Model file not found: %s\"}\n",
                        escape_json(mstr).c_str());
        } else {
            std::fprintf(stderr, "[ERROR] Model file not found: %s\n", mstr.c_str());
        }
        std::fflush(stdout);
        std::fflush(stderr);
        return 2;
    }
    std::error_code ec;
    root = fs::weakly_canonical(root, ec);
    fs::path nsfw_dir = nsfw_dir_arg.empty() ? (root / "NSFW")
                                             : fs::weakly_canonical(nsfw_dir_arg, ec);

    std::vector<fs::path> files = collect(root, recursive, nsfw_dir);
    if (files.empty()) {
        if (is_json) {
            std::printf("{\"type\":\"init\",\"total\":0,\"threshold\":%.4f,\"device\":\"%s\",\"dir\":\"%s\"}\n",
                        threshold, device.c_str(), escape_json(narrow(root.wstring())).c_str());
            std::printf("{\"type\":\"summary\",\"total\":0,\"safe\":0,\"nsfw\":0,\"moved\":0,\"failed\":0,\"dry_run\":%s}\n",
                        dry_run ? "true" : "false");
        } else {
            std::printf("[INFO] No images found to audit in target directory: %s\n",
                        narrow(root.wstring()).c_str());
        }
        std::fflush(stdout);
        return 0;
    }

    if (is_json) {
        std::printf("{\"type\":\"init\",\"total\":%zu,\"threshold\":%.4f,\"batch\":%d,\"device\":\"%s\",\"recursive\":%s,\"dry_run\":%s,\"dir\":\"%s\",\"nsfw_dir\":\"%s\"}\n",
                    files.size(), threshold, batch, device.c_str(),
                    recursive ? "true" : "false", dry_run ? "true" : "false",
                    escape_json(narrow(root.wstring())).c_str(),
                    escape_json(narrow(nsfw_dir.wstring())).c_str());
        std::fflush(stdout);
    } else {
        std::printf("[INIT] total=%zu | threshold=%.2f | batch=%d | device=%s | dir=%s\n",
                    files.size(), threshold, batch, device.c_str(), narrow(root.wstring()).c_str());
        std::printf("[INFO] Auditing %zu images | Threshold: %.2f | Device: %s | Batch: %d\n",
                    files.size(), threshold, device.c_str(), batch);
        std::fflush(stdout);
    }

    vit_runner runner;
    if (!runner.init(model_path.string(), batch, device)) {
        if (is_json) {
            std::printf("{\"type\":\"error\",\"message\":\"Failed to initialize model inference engine\"}\n");
            std::fflush(stdout);
        }
        return 2;
    }

    std::vector<fs::path> bad;
    std::vector<float> bad_score;
    std::vector<fs::path> failed;
    size_t safe_count = 0;

    std::vector<fs::path> bp;
    std::vector<std::vector<float>> bf;

    auto flush = [&]() {
        if (bp.empty()) return;
        std::vector<const std::vector<float>*> ptrs;
        for (auto & f : bf) ptrs.push_back(&f);
        std::vector<float> probs = runner.infer(ptrs);
        for (size_t i = 0; i < bp.size(); ++i) {
            float pns = probs[i * N_CLASS + 1];
            bool is_nsfw = (pns >= threshold);
            std::string filename_str = narrow(bp[i].filename().wstring());
            std::string fullpath_str = narrow(bp[i].wstring());

            if (is_nsfw) {
                bad.push_back(bp[i]);
                bad_score.push_back(pns);
            } else {
                safe_count++;
            }

            if (is_json) {
                std::printf("{\"type\":\"item\",\"status\":\"%s\",\"score\":%.4f,\"percent\":%.2f,\"file\":\"%s\",\"path\":\"%s\"}\n",
                            is_nsfw ? "NSFW" : "SAFE", pns, pns * 100.0f,
                            escape_json(filename_str).c_str(),
                            escape_json(fullpath_str).c_str());
            } else {
                std::printf("[ITEM] status=%s | score=%.4f | percent=%6.2f%% | file=%s | path=%s\n",
                            is_nsfw ? "NSFW" : "SAFE", pns, pns * 100.0f,
                            filename_str.c_str(), fullpath_str.c_str());
                std::printf("  [%-4s] %6.2f%%  %s\n",
                            is_nsfw ? "NSFW" : "ok", pns * 100.0f, filename_str.c_str());
            }
            std::fflush(stdout);
        }
        bp.clear();
        bf.clear();
    };

    for (size_t i = 0; i < files.size(); ++i) {
        std::vector<float> frame;
        if (preprocess(files[i], frame)) {
            bp.push_back(files[i]);
            bf.push_back(std::move(frame));
        } else {
            std::string fail_name = narrow(files[i].filename().wstring());
            std::string fail_path = narrow(files[i].wstring());
            if (is_json) {
                std::printf("{\"type\":\"item\",\"status\":\"ERROR\",\"score\":-1.0,\"percent\":0.0,\"file\":\"%s\",\"path\":\"%s\",\"error\":\"Decode failed\"}\n",
                            escape_json(fail_name).c_str(), escape_json(fail_path).c_str());
            } else {
                std::fprintf(stderr, "[ERROR] Failed to read/decode: %s\n", fail_name.c_str());
            }
            std::fflush(stdout);
            std::fflush(stderr);
            failed.push_back(files[i]);
        }
        if ((int) bp.size() == batch) {
            flush();
            size_t current = i + 1;
            double pct = (double) current * 100.0 / files.size();
            if (is_json) {
                std::printf("{\"type\":\"progress\",\"current\":%zu,\"total\":%zu,\"percent\":%.2f}\n",
                            current, files.size(), pct);
            } else {
                std::printf("[PROGRESS] current=%zu | total=%zu | percent=%.2f%%\n",
                            current, files.size(), pct);
                std::printf("  --- Progress %zu/%zu (%.1f%%) ---\n", current, files.size(), pct);
            }
            std::fflush(stdout);
        }
    }
    if (!bp.empty()) {
        flush();
        if (is_json) {
            std::printf("{\"type\":\"progress\",\"current\":%zu,\"total\":%zu,\"percent\":100.00}\n",
                        files.size(), files.size());
        } else {
            std::printf("[PROGRESS] current=%zu | total=%zu | percent=100.00%%\n",
                        files.size(), files.size());
        }
        std::fflush(stdout);
    }

    int moved = 0;
    if (!bad.empty() && !dry_run) fs::create_directories(nsfw_dir, ec);
    for (const auto & src : bad) {
        fs::path target = unique_dest(nsfw_dir / src.filename());
        std::string src_str = narrow(src.wstring());
        std::string tgt_str = narrow(target.wstring());

        if (dry_run) {
            if (is_json) {
                std::printf("{\"type\":\"action\",\"status\":\"DRY_RUN\",\"src\":\"%s\",\"dst\":\"%s\"}\n",
                            escape_json(src_str).c_str(), escape_json(tgt_str).c_str());
            } else {
                std::printf("[ACTION] status=DRY_RUN | src=%s | dst=%s\n",
                            src_str.c_str(), tgt_str.c_str());
                std::printf("  [DRY-RUN] Will move: %s -> %s\n", src_str.c_str(), tgt_str.c_str());
            }
        } else if (move_file(src, target)) {
            ++moved;
            if (is_json) {
                std::printf("{\"type\":\"action\",\"status\":\"MOVED\",\"src\":\"%s\",\"dst\":\"%s\"}\n",
                            escape_json(src_str).c_str(), escape_json(tgt_str).c_str());
            } else {
                std::printf("[ACTION] status=MOVED | src=%s | dst=%s\n",
                            src_str.c_str(), tgt_str.c_str());
            }
        } else {
            if (is_json) {
                std::printf("{\"type\":\"action\",\"status\":\"FAILED\",\"src\":\"%s\",\"dst\":\"%s\"}\n",
                            escape_json(src_str).c_str(), escape_json(tgt_str).c_str());
            } else {
                std::fprintf(stderr, "[ERROR] Failed to move: %s\n", narrow(src.filename().wstring()).c_str());
            }
        }
        std::fflush(stdout);
        std::fflush(stderr);
    }

    if (is_json) {
        std::printf("{\"type\":\"summary\",\"total\":%zu,\"safe\":%zu,\"nsfw\":%zu,\"moved\":%d,\"failed\":%zu,\"dry_run\":%s}\n",
                    files.size(), safe_count, bad.size(), moved, failed.size(),
                    dry_run ? "true" : "false");
    } else {
        std::printf("\n==================================================\n");
        std::printf("[SUMMARY] total=%zu | safe=%zu | nsfw=%zu | moved=%d | failed=%zu | dry_run=%s\n",
                    files.size(), safe_count, bad.size(), moved, failed.size(),
                    dry_run ? "true" : "false");
        std::printf("Audit finished: %zu total, %zu safe, %zu NSFW.\n",
                    files.size(), safe_count, bad.size());
        if (dry_run) {
            std::printf("Dry-run mode: no files were moved.\n");
        } else if (moved > 0) {
            std::printf("Moved %d images to: %s\n", moved, narrow(nsfw_dir.wstring()).c_str());
        }
        if (!failed.empty()) {
            std::printf("Unreadable files: %zu\n", failed.size());
        }
    }
    std::fflush(stdout);
    std::fflush(stderr);
    return 0;
}
