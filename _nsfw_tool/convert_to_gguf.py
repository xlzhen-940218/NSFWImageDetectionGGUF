# -*- coding: utf-8 -*-
"""
将 Falconsai/nsfw_image_detection (ViT-B/16 图像分类模型) 转换为 GGUF 格式。

推理端 (vit_nsfw.cpp) 为了最大程度兼容各家 GPU 的 ggml 算子，
不使用 Conv，而是由调用方把输入图片 unfold 成 196 个 patch 向量 (196x768)，
因此这里把 patch embedding 的卷积权重原样 reshape 成线性层权重
(PyTorch 卷积权重 [out,C,kH,kW] 的内存排布与 patch 向量 [C,kH,kW] 完全对齐)。

用法:
  py convert_to_gguf.py [HF模型目录] [输出gguf路径]
"""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
import gguf


def convert(hf_dir: Path, out_path: Path) -> None:
    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "vit", f"不支持的模型类型: {cfg['model_type']}"

    n_layer = cfg["num_hidden_layers"]
    n_head  = cfg["num_attention_heads"]
    hidden  = cfg["hidden_size"]
    inter   = cfg["intermediate_size"]
    patch   = cfg["patch_size"]
    image   = cfg["image_size"]
    n_ch    = cfg["num_channels"]
    n_class = len(cfg["id2label"])
    eps     = cfg["layer_norm_eps"]
    n_patch = (image // patch) ** 2
    assert hidden % n_head == 0

    st_path = hf_dir / "model.safetensors"
    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(st_path), framework="numpy") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    def take(name: str) -> np.ndarray:
        if name not in tensors:
            raise KeyError(f"缺少权重: {name}")
        return tensors[name].astype(np.float32, copy=False)

    writer = gguf.GGUFWriter(str(out_path), "nsfw-vit")
    try:
        # ---- 超参数与标签 ----
        writer.add_name("nsfw_image_detection")
        writer.add_int32("vit.n_layer", n_layer)
        writer.add_int32("vit.n_head", n_head)
        writer.add_int32("vit.hidden", hidden)
        writer.add_int32("vit.intermediate", inter)
        writer.add_int32("vit.patch_size", patch)
        writer.add_int32("vit.image_size", image)
        writer.add_int32("vit.n_channels", n_ch)
        writer.add_int32("vit.n_patch", n_patch)
        writer.add_int32("vit.n_class", n_class)
        writer.add_float32("vit.layer_norm_eps", float(eps))
        id2label = cfg["id2label"]
        for idx_str, label in sorted(id2label.items(), key=lambda kv: int(kv[0])):
            writer.add_string(f"vit.label.{idx_str}", str(label))

        # ---- 输出张量 ----
        # GGUFWriter 按 numpy 形状约定接收，写入时自动反转为 ggml 的 ne 顺序
        def add(name: str, arr: np.ndarray) -> None:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            writer.add_tensor(name, arr)

        # patch embedding: 卷积权重 [out, C, kH, kW] -> [out, C*kH*kW]
        w = take("vit.embeddings.patch_embeddings.projection.weight")
        add("patch_w", w.reshape(hidden, n_ch * patch * patch))
        add("patch_b", take("vit.embeddings.patch_embeddings.projection.bias"))

        # cls token: [1,1,H] -> [H]
        add("cls_token", take("vit.embeddings.cls_token").reshape(hidden))
        # 位置编码: [1,S,H] -> [S,H]
        add("pos_embed", take("vit.embeddings.position_embeddings").reshape(n_patch + 1, hidden))

        for i in range(n_layer):
            p = f"vit.encoder.layer.{i}."
            q = take(p + "attention.attention.query.weight")
            k = take(p + "attention.attention.key.weight")
            v = take(p + "attention.attention.value.weight")
            add(f"blocks.{i}.attn_q_w", q)
            add(f"blocks.{i}.attn_k_w", k)
            add(f"blocks.{i}.attn_v_w", v)
            add(f"blocks.{i}.attn_q_b", take(p + "attention.attention.query.bias"))
            add(f"blocks.{i}.attn_k_b", take(p + "attention.attention.key.bias"))
            add(f"blocks.{i}.attn_v_b", take(p + "attention.attention.value.bias"))
            add(f"blocks.{i}.attn_o_w", take(p + "attention.output.dense.weight"))
            add(f"blocks.{i}.attn_o_b", take(p + "attention.output.dense.bias"))
            add(f"blocks.{i}.ln1_w", take(p + "layernorm_before.weight"))
            add(f"blocks.{i}.ln1_b", take(p + "layernorm_before.bias"))
            add(f"blocks.{i}.mlp_fc1_w", take(p + "intermediate.dense.weight"))
            add(f"blocks.{i}.mlp_fc1_b", take(p + "intermediate.dense.bias"))
            add(f"blocks.{i}.mlp_fc2_w", take(p + "output.dense.weight"))
            add(f"blocks.{i}.mlp_fc2_b", take(p + "output.dense.bias"))
            add(f"blocks.{i}.ln2_w", take(p + "layernorm_after.weight"))
            add(f"blocks.{i}.ln2_b", take(p + "layernorm_after.bias"))

        add("lnf_w", take("vit.layernorm.weight"))
        add("lnf_b", take("vit.layernorm.bias"))
        add("head_w", take("classifier.weight"))
        add("head_b", take("classifier.bias"))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    print(f"转换完成: {out_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    hf = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "model_hf"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "models" / "nsfw_image_detection.fp32.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    convert(hf, out)
