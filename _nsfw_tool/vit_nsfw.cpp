// ViT-B/16 NSFW 图像分类推理器 (GGUF + ggml)
//
// 输入协议 (stdin/stdout, 全二进制)：
//   启动参数: vit_nsfw <model.gguf> [--batch N] [--device auto|vulkan|cpu]
//   每批输入: batch * 196 * 768 个 float32 (已在外部完成 patch unfold, NCHW 排布)
//   每批输出: batch * 2 个 float32 (softmax 概率: [normal, nsfw] * batch)
//   stdin 到达 EOF 时退出。日志全部写 stderr。

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

struct vit_hparams {
    int n_layer = 12;
    int n_head  = 12;
    int hidden  = 768;
    int inter   = 3072;
    int n_patch = 196;
    int n_class = 2;
    int n_seq   = 197;
    float eps   = 1e-12f;
};

struct vit_block {
    struct ggml_tensor * attn_q_w = nullptr, * attn_q_b = nullptr;
    struct ggml_tensor * attn_k_w = nullptr, * attn_k_b = nullptr;
    struct ggml_tensor * attn_v_w = nullptr, * attn_v_b = nullptr;
    struct ggml_tensor * attn_o_w = nullptr, * attn_o_b = nullptr;
    struct ggml_tensor * ln1_w    = nullptr, * ln1_b    = nullptr;
    struct ggml_tensor * mlp_fc1_w = nullptr, * mlp_fc1_b = nullptr;
    struct ggml_tensor * mlp_fc2_w = nullptr, * mlp_fc2_b = nullptr;
    struct ggml_tensor * ln2_w    = nullptr, * ln2_b    = nullptr;
};

struct vit_model {
    ~vit_model() {
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (backend) ggml_backend_free(backend);
        if (ctx_w) ggml_free(ctx_w);
    }

    vit_hparams hparams;

    struct ggml_tensor * patch_w = nullptr, * patch_b = nullptr;
    struct ggml_tensor * cls_token = nullptr, * pos_embed = nullptr;
    std::vector<vit_block> blocks;
    struct ggml_tensor * lnf_w = nullptr, * lnf_b = nullptr;
    struct ggml_tensor * head_w = nullptr, * head_b = nullptr;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context * ctx_w = nullptr;
};

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        throw std::runtime_error("GGUF 中缺少张量: " + name);
    }
    return t;
}

static int32_t get_i32(struct gguf_context * g, const char * key, int32_t def) {
    int idx = gguf_find_key(g, key);
    return idx >= 0 ? gguf_get_val_i32(g, idx) : def;
}

static float get_f32(struct gguf_context * g, const char * key, float def) {
    int idx = gguf_find_key(g, key);
    return idx >= 0 ? gguf_get_val_f32(g, idx) : def;
}

// 选择计算后端: vulkan(同时支持 NVIDIA/AMD/Intel) > 其他 GPU > CPU
static ggml_backend_t make_backend(const std::string & want) {
    ggml_backend_dev_t best = nullptr;
    int best_score = -1;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        struct ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        std::string name = props.name ? props.name : "";
        for (auto & c : name) { c = (char) tolower((unsigned char) c); }

        int score = -1;
        if (want == "cpu" && props.type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            score = 1000;
        } else if (want == "vulkan" && name.find("vulkan") != std::string::npos) {
            score = 900;
        } else if (want == "auto" && name.find("vulkan") != std::string::npos) {
            score = 500;
        } else if (want == "auto" && props.type != GGML_BACKEND_DEVICE_TYPE_CPU) {
            score = 300;
        }
        if (score > best_score) { best_score = score; best = dev; }
    }

    if (!best) {
        fprintf(stderr, "找不到匹配的后端 (want=%s)，回退到 CPU\n", want.c_str());
        return ggml_backend_cpu_init();
    }

    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(best, &props);
    ggml_backend_t backend = ggml_backend_dev_init(best, nullptr);
    if (backend) {
        fprintf(stderr, "[后端] %s — %s\n", props.name, props.description ? props.description : "");
    }
    return backend;
}

static bool vit_load(const std::string & fname, const std::string & device, vit_model & m) {
    struct gguf_init_params params = { /*.no_alloc=*/true, /*.ctx=*/&m.ctx_w };
    struct gguf_context * g = gguf_init_from_file(fname.c_str(), params);
    if (!g) {
        fprintf(stderr, "无法打开 GGUF 模型: %s\n", fname.c_str());
        return false;
    }

    m.hparams.n_layer = get_i32(g, "vit.n_layer", 12);
    m.hparams.n_head  = get_i32(g, "vit.n_head", 12);
    m.hparams.hidden  = get_i32(g, "vit.hidden", 768);
    m.hparams.inter   = get_i32(g, "vit.intermediate", 3072);
    m.hparams.n_patch = get_i32(g, "vit.n_patch", 196);
    m.hparams.n_class = get_i32(g, "vit.n_class", 2);
    m.hparams.n_seq   = m.hparams.n_patch + 1;
    m.hparams.eps     = get_f32(g, "vit.layer_norm_eps", 1e-12f);

    m.backend = make_backend(device);
    if (!m.backend) {
        gguf_free(g);
        return false;
    }
    m.buf_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    if (!m.buf_w) {
        fprintf(stderr, "权重显存分配失败\n");
        gguf_free(g);
        return false;
    }

    // 取张量句柄
    try {
        m.patch_w   = get_tensor(m.ctx_w, "patch_w");
        m.patch_b   = get_tensor(m.ctx_w, "patch_b");
        m.cls_token = get_tensor(m.ctx_w, "cls_token");
        m.pos_embed = get_tensor(m.ctx_w, "pos_embed");
        m.blocks.resize(m.hparams.n_layer);
        char buf[128];
        for (int i = 0; i < m.hparams.n_layer; ++i) {
            auto & b = m.blocks[i];
            snprintf(buf, sizeof(buf), "blocks.%d.", i);
            std::string p = buf;
            b.attn_q_w = get_tensor(m.ctx_w, p + "attn_q_w");
            b.attn_q_b = get_tensor(m.ctx_w, p + "attn_q_b");
            b.attn_k_w = get_tensor(m.ctx_w, p + "attn_k_w");
            b.attn_k_b = get_tensor(m.ctx_w, p + "attn_k_b");
            b.attn_v_w = get_tensor(m.ctx_w, p + "attn_v_w");
            b.attn_v_b = get_tensor(m.ctx_w, p + "attn_v_b");
            b.attn_o_w = get_tensor(m.ctx_w, p + "attn_o_w");
            b.attn_o_b = get_tensor(m.ctx_w, p + "attn_o_b");
            b.ln1_w    = get_tensor(m.ctx_w, p + "ln1_w");
            b.ln1_b    = get_tensor(m.ctx_w, p + "ln1_b");
            b.mlp_fc1_w = get_tensor(m.ctx_w, p + "mlp_fc1_w");
            b.mlp_fc1_b = get_tensor(m.ctx_w, p + "mlp_fc1_b");
            b.mlp_fc2_w = get_tensor(m.ctx_w, p + "mlp_fc2_w");
            b.mlp_fc2_b = get_tensor(m.ctx_w, p + "mlp_fc2_b");
            b.ln2_w    = get_tensor(m.ctx_w, p + "ln2_w");
            b.ln2_b    = get_tensor(m.ctx_w, p + "ln2_b");
        }
        m.lnf_w  = get_tensor(m.ctx_w, "lnf_w");
        m.lnf_b  = get_tensor(m.ctx_w, "lnf_b");
        m.head_w = get_tensor(m.ctx_w, "head_w");
        m.head_b = get_tensor(m.ctx_w, "head_b");
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        gguf_free(g);
        return false;
    }

    // 拷贝权重数据到后端
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "无法重新打开模型文件读取权重\n");
        gguf_free(g);
        return false;
    }
    const int n_tensors = gguf_get_n_tensors(g);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        struct ggml_tensor * t = ggml_get_tensor(m.ctx_w, name);
        size_t offs = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> buf(nbytes);
        if (fseek(f, (long) offs, SEEK_SET) != 0 || fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "读取权重失败: %s\n", name);
            fclose(f);
            gguf_free(g);
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }
    fclose(f);
    gguf_free(g);

    fprintf(stderr, "[模型] layers=%d heads=%d hidden=%d patches=%d classes=%d\n",
            m.hparams.n_layer, m.hparams.n_head, m.hparams.hidden,
            m.hparams.n_patch, m.hparams.n_class);
    return true;
}

static struct ggml_tensor * layer_norm(
        struct ggml_context * ctx, const vit_hparams & hp,
        struct ggml_tensor * x, struct ggml_tensor * w, struct ggml_tensor * b) {
    struct ggml_tensor * y = ggml_norm(ctx, x, hp.eps);
    y = ggml_mul(ctx, y, w);
    y = ggml_add(ctx, y, b);
    return y;
}

static struct ggml_cgraph * vit_graph(const vit_model & m, int B) {
    const vit_hparams & hp = m.hparams;
    const int D = hp.hidden;
    const int S = hp.n_seq;
    const int H = hp.n_head;
    const int Dh = D / H;

    static std::vector<uint8_t> gbuf;
    if (gbuf.empty()) {
        gbuf.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead());
    }
    struct ggml_init_params ip = { gbuf.size(), gbuf.data(), /*.no_alloc=*/true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // 输入: 已 unfold 的 patch 序列 [768, 196, B]
    struct ggml_tensor * in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, hp.n_patch, B);
    ggml_set_name(in, "input");
    ggml_set_input(in);

    // patch embedding (线性化的卷积) + cls token + 位置编码
    struct ggml_tensor * x = ggml_mul_mat(ctx, m.patch_w, in);
    x = ggml_add(ctx, x, m.patch_b);                                  // [D,196,B]
    x = ggml_reshape_4d(ctx, x, D, hp.n_patch, 1, B);

    struct ggml_tensor * cls = ggml_reshape_4d(ctx, m.cls_token, D, 1, 1, 1);
    cls = ggml_repeat_4d(ctx, cls, D, 1, 1, B);                       // [D,1,1,B]
    x = ggml_concat(ctx, cls, x, 1);                                  // [D,197,1,B]
    x = ggml_reshape_3d(ctx, x, D, S, B);
    x = ggml_add(ctx, x, ggml_reshape_3d(ctx, m.pos_embed, D, S, 1));

    for (int i = 0; i < hp.n_layer; ++i) {
        const vit_block & blk = m.blocks[i];

        // ---- 自注意力 ----
        struct ggml_tensor * h = layer_norm(ctx, hp, x, blk.ln1_w, blk.ln1_b);
        struct ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, blk.attn_q_w, h), blk.attn_q_b);
        struct ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, blk.attn_k_w, h), blk.attn_k_b);
        struct ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, blk.attn_v_w, h), blk.attn_v_b);
        // [Dh,H,S,B] -> permute -> [Dh,S,H,B]，把头维放到 batch 维上
        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, Dh, H, S, B), 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, Dh, H, S, B), 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, Dh, H, S, B), 0, 2, 1, 3));

        // KQ[i,j] = sum_d K[d,i]Q[d,j], softmax 在 key 维(ne0)上
        struct ggml_tensor * att = ggml_mul_mat(ctx, k, q);           // [S_k,S_q,H,B]
        att = ggml_scale(ctx, att, 1.0f / std::sqrt((float) Dh));
        att = ggml_soft_max(ctx, att);
        // V 转置: [Dh,S,H,B] -> [S,Dh,H,B]；KQV = Vt @ att -> [Dh,S_q,H,B]
        struct ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        att = ggml_mul_mat(ctx, vt, att);
        att = ggml_cont(ctx, ggml_permute(ctx, att, 0, 2, 1, 3));     // [Dh,H,S_q,B]
        att = ggml_reshape_3d(ctx, att, D, S, B);
        att = ggml_add(ctx, ggml_mul_mat(ctx, blk.attn_o_w, att), blk.attn_o_b);
        x = ggml_add(ctx, x, att);

        // ---- MLP ----
        h = layer_norm(ctx, hp, x, blk.ln2_w, blk.ln2_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, blk.mlp_fc1_w, h), blk.mlp_fc1_b);
        h = ggml_gelu(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, blk.mlp_fc2_w, h), blk.mlp_fc2_b);
        x = ggml_add(ctx, x, h);
    }

    // 最终 LN，取 cls token (序列第 0 个)
    x = layer_norm(ctx, hp, x, m.lnf_w, m.lnf_b);
    struct ggml_tensor * cls_out = ggml_view_3d(
            ctx, x, D, 1, B,
            x->nb[1], x->nb[2], 0);
    cls_out = ggml_cont(ctx, cls_out);                                // [D,1,B]

    struct ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, m.head_w, cls_out), m.head_b);
    logits = ggml_reshape_3d(ctx, logits, hp.n_class, 1, B);
    struct ggml_tensor * probs = ggml_soft_max(ctx, logits);
    probs = ggml_reshape_2d(ctx, probs, hp.n_class, B);               // [2,B]
    ggml_set_name(probs, "probs");
    ggml_set_output(probs);

    ggml_build_forward_expand(gf, probs);
    return gf;
}

static bool read_exact(FILE * f, void * dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    // stdin/stdout 传输二进制 float，必须关闭文本模式转义 (0x1A/CRLF)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    if (argc < 2) {
        fprintf(stderr, "用法: %s <model.gguf> [--batch N] [--device auto|vulkan|cpu]\n", argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    int batch = 8;
    std::string device = "auto";
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--batch" && i + 1 < argc) { batch = std::stoi(argv[++i]); }
        else if (a == "--device" && i + 1 < argc) { device = argv[++i]; }
    }
    if (batch < 1 || batch > 128) {
        fprintf(stderr, "batch 取值范围 1..128\n");
        return 1;
    }

    setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
    setvbuf(stderr, nullptr, _IONBF, 0);

    vit_model model;
    if (!vit_load(model_path, device, model)) {
        return 2;
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    struct ggml_cgraph * gf = vit_graph(model, batch);
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "计算图显存分配失败\n");
        return 3;
    }
    struct ggml_tensor * in = ggml_graph_get_tensor(gf, "input");

    const size_t frame = 196 * 768;                       // 每张图 patch 向量数
    std::vector<float> in_buf((size_t) batch * frame);
    std::vector<float> out_buf((size_t) batch * model.hparams.n_class);

    // 预热 (首次运行 Vulkan 会即时编译着色器)
    {
        std::fill(in_buf.begin(), in_buf.end(), 0.0f);
        ggml_backend_tensor_set(in, in_buf.data(), 0, in_buf.size() * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "预热推理失败\n");
            return 4;
        }
        ggml_backend_synchronize(model.backend);
    }

    fprintf(stderr, "READY batch=%d\n", batch);

    while (true) {
        if (!read_exact(stdin, in_buf.data(), in_buf.size() * sizeof(float))) {
            break;
        }
        ggml_backend_tensor_set(in, in_buf.data(), 0, in_buf.size() * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "推理失败\n");
            return 5;
        }
        struct ggml_tensor * probs = ggml_graph_get_tensor(gf, "probs");
        ggml_backend_tensor_get(probs, out_buf.data(), 0, out_buf.size() * sizeof(float));
        if (fwrite(out_buf.data(), sizeof(float), out_buf.size(), stdout) != out_buf.size()) {
            break;
        }
        fflush(stdout);
    }

    ggml_gallocr_free(alloc);
    return 0;
}
