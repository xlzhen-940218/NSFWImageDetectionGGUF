// NSFW 图像识别引擎实现（ViT-B/16 + ggml）
//
// 移植自 _nsfw_tool/nsfw_audit.cpp：
//   - 图像预处理与 Pillow 12 的 BILINEAR 完全一致（PRECISION_BITS=22）
//   - ViT-B/16 前向图与权重加载逻辑保持相同，保证打分与桌面端一致
// Android 侧固定 batch=1，计算图缓冲改为每个 Engine 实例独占（线程安全前提：
// 同一个 Engine 实例串行调用 score_image）。

#define STB_IMAGE_IMPLEMENTATION
#ifndef STBI_NO_STDIO
#define STBI_NO_STDIO  // 一律走内存加载
#endif
#include "stb_image.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "nsfw_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace nsfw {
namespace {

// ============================== 常量 ==============================
constexpr int IMAGE_SIZE = 224;
constexpr int PATCH      = 16;
constexpr int GRID       = IMAGE_SIZE / PATCH;  // 14
constexpr int N_PATCH    = GRID * GRID;         // 196
constexpr int HIDDEN     = 768;
constexpr int N_CLASS    = 2;
constexpr int FRAME_FLOATS = N_PATCH * HIDDEN;

// ============================== 图像预处理 ==============================
// 与 Pillow 12 的 precompute_coeffs + normalize_coeffs_8bpc 完全一致：
// double 权重归一化后，量化为 PRECISION_BITS=22 位的 int32 系数
constexpr int PRECISION_BITS = 32 - 8 - 2;  // 22，与 Pillow 相同

struct Axis1D {
    std::vector<std::vector<int>> idx;    // 每个输出位置的采样下标
    std::vector<std::vector<int32_t>> k;  // 量化后的整数系数
};

Axis1D bilinear_weights(int n_in, int n_out) {
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

// Pillow 的 clip8：ss 带 2^(22-1) 的四舍五入偏移，算术右移后裁剪到 0..255
inline uint8_t clip8(int ss) {
    int v = ss >> PRECISION_BITS;  // bilinear 权重非负，算术/逻辑右移等价
    return (uint8_t) (v < 0 ? 0 : (v > 255 ? 255 : v));
}

// 解码 + BILINEAR 224x224 + 归一化 + patch unfold
// out 为 N_PATCH*HIDDEN 个 float
bool decode_and_preprocess(const uint8_t * data, size_t size,
                           std::vector<float> & out, std::string & err) {
    int W = 0, H = 0, C = 0;
    stbi_uc * img = stbi_load_from_memory(data, (int) size, &W, &H, &C, 3);
    if (!img) {
        const char * reason = stbi_failure_reason();
        err = std::string("图片解码失败: ") + (reason ? reason : "unknown");
        return false;
    }

    const Axis1D ax = bilinear_weights(W, IMAGE_SIZE);
    const Axis1D ay = bilinear_weights(H, IMAGE_SIZE);

    // 水平阶段（对齐 Pillow ImagingResampleHorizontal_8bpc 的 3-band 路径）：
    // int32 累加，四舍五入后 clip8 量化回 uint8 中间图 [H,224,3]
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
                    dst[0]                = s[0] / 127.5f - 1.0f;
                    dst[PATCH * PATCH]     = s[1] / 127.5f - 1.0f;
                    dst[2 * PATCH * PATCH] = s[2] / 127.5f - 1.0f;
                }
            }
        }
    }
    return true;
}

// ============================== ViT 模型 ==============================
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

ggml_tensor * need_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) throw std::runtime_error("GGUF 中缺少张量: " + name);
    return t;
}

int32_t kv_i32(gguf_context * g, const char * k, int32_t def) {
    int i = gguf_find_key(g, k);
    return i >= 0 ? gguf_get_val_i32(g, i) : def;
}

float kv_f32(gguf_context * g, const char * k, float def) {
    int i = gguf_find_key(g, k);
    return i >= 0 ? gguf_get_val_f32(g, i) : def;
}

// Android 上仅编译 CPU 后端；线程数由调用方指定
ggml_backend_t make_cpu_backend(int n_threads) {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (be) {
        ggml_backend_cpu_set_n_threads(be, (std::max)(1, n_threads));
    }
    return be;
}

bool vit_load(const std::string & fname, int n_threads, vit_model & m,
              std::string & err) {
    gguf_init_params gp{true, &m.ctx_w};
    gguf_context * g = gguf_init_from_file(fname.c_str(), gp);
    if (!g) {
        err = "无法打开 GGUF 模型文件（文件不存在或已损坏）: " + fname;
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

    m.backend = make_cpu_backend(n_threads);
    if (!m.backend) {
        gguf_free(g);
        err = "CPU 后端初始化失败";
        return false;
    }
    m.buf_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    if (!m.buf_w) {
        gguf_free(g);
        err = "权重内存分配失败（内存不足）";
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
            b.q_w  = need_tensor(m.ctx_w, p + "attn_q_w");
            b.q_b  = need_tensor(m.ctx_w, p + "attn_q_b");
            b.k_w  = need_tensor(m.ctx_w, p + "attn_k_w");
            b.k_b  = need_tensor(m.ctx_w, p + "attn_k_b");
            b.v_w  = need_tensor(m.ctx_w, p + "attn_v_w");
            b.v_b  = need_tensor(m.ctx_w, p + "attn_v_b");
            b.o_w  = need_tensor(m.ctx_w, p + "attn_o_w");
            b.o_b  = need_tensor(m.ctx_w, p + "attn_o_b");
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
        gguf_free(g);
        err = e.what();
        return false;
    }

    FILE * f = std::fopen(fname.c_str(), "rb");
    if (!f) {
        gguf_free(g);
        err = "模型文件读取失败: " + fname;
        return false;
    }
    const int n_tensors = gguf_get_n_tensors(g);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(m.ctx_w, name);
        size_t offs = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
        size_t nb = ggml_nbytes(t);
        std::vector<uint8_t> buf(nb);
        if (std::fseek(f, (long) offs, SEEK_SET) != 0
            || std::fread(buf.data(), 1, nb, f) != nb) {
            std::fclose(f);
            gguf_free(g);
            err = "模型权重读取不完整（文件可能未下载完整）";
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nb);
    }
    std::fclose(f);
    gguf_free(g);
    return true;
}

ggml_tensor * layernorm(ggml_context * ctx, const vit_hparams & hp,
                        ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, hp.eps);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

// 构建前向图（B=1）。ctx 由调用方提供的缓冲创建，调用方负责 ggml_free
ggml_cgraph * vit_graph(const vit_model & m, int B, std::vector<uint8_t> & gbuf,
                        ggml_context ** ctx_out) {
    const vit_hparams & hp = m.hp;
    const int D = hp.hidden, S = hp.n_seq, H = hp.n_head, Dh = D / H;

    if (gbuf.empty())
        gbuf.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead());
    ggml_init_params ip{gbuf.size(), gbuf.data(), true};
    ggml_context * ctx = ggml_init(ip);
    *ctx_out = ctx;
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

}  // namespace

// ============================== Engine::Impl ==============================
struct Engine::Impl {
    vit_model model;
    ggml_gallocr_t alloc = nullptr;
    ggml_context * ctx_graph = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * in = nullptr;
    ggml_tensor * probs = nullptr;
    std::vector<uint8_t> gbuf;
    std::vector<float> in_buf, out_buf;
    std::string backend_name;
    std::mutex mtx;
    bool ready = false;

    ~Impl() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx_graph) ggml_free(ctx_graph);  // 元数据在 gbuf 中，先释放 context
    }

    bool init(const std::string & model_path, int n_threads, std::string & err) {
        if (!vit_load(model_path, n_threads, model, err)) return false;

        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!alloc) { err = "计算图内存分配器创建失败"; return false; }

        gf = vit_graph(model, 1, gbuf, &ctx_graph);
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            err = "计算图内存分配失败（内存不足）";
            return false;
        }
        in = ggml_graph_get_tensor(gf, "input");
        probs = ggml_graph_get_tensor(gf, "probs");
        if (!in || !probs) { err = "计算图张量缺失"; return false; }

        in_buf.assign((size_t) FRAME_FLOATS, 0.0f);
        out_buf.assign((size_t) N_CLASS, 0.0f);
        ggml_backend_tensor_set(in, in_buf.data(), 0, in_buf.size() * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            err = "热身推理失败";
            return false;
        }
        ggml_backend_synchronize(model.backend);
        backend_name = "CPU";
        ready = true;
        return true;
    }

    bool infer(const std::vector<float> & frame, float & p_nsfw, std::string & err) {
        std::memcpy(in_buf.data(), frame.data(), FRAME_FLOATS * sizeof(float));
        ggml_backend_tensor_set(in, in_buf.data(), 0, in_buf.size() * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            err = "推理执行失败";
            return false;
        }
        ggml_backend_tensor_get(probs, out_buf.data(), 0, out_buf.size() * sizeof(float));
        p_nsfw = out_buf[1];
        return true;
    }
};

// ============================== Engine ==============================
Engine::Engine() : impl_(new Impl()) {}

Engine::~Engine() = default;

bool Engine::load(const std::string & gguf_path, int n_threads, std::string & err) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!impl_->init(gguf_path, n_threads, err)) {
        impl_->ready = false;
        return false;
    }
    return true;
}

bool Engine::score_image(const uint8_t * data, size_t size, float & p_nsfw,
                         std::string & err) {
    if (!data || size == 0) {
        err = "图片数据为空";
        return false;
    }
    std::vector<float> frame;
    if (!decode_and_preprocess(data, size, frame, err)) return false;

    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!impl_->ready) {
        err = "模型尚未加载";
        return false;
    }
    return impl_->infer(frame, p_nsfw, err);
}

bool Engine::loaded() const { return impl_->ready; }

const std::string & Engine::backend_name() const { return impl_->backend_name; }

}  // namespace nsfw
