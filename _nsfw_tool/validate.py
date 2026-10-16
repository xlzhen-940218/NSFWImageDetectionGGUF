# -*- coding: utf-8">
"""数值正确性验证: PyTorch/transformers 参考实现 vs GGUF 引擎 (CPU)。"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TOOL = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL.parent))
import nsfw_audit  # noqa: E402

import torch  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

HF_DIR = TOOL / "model_hf"
EXE = TOOL / "build" / "bin" / "vit_nsfw.exe"
GGUF = TOOL / "models" / "nsfw_image_detection.fp32.gguf"

images = sorted(TOOL.parent.glob("*.png"))[:6]

proc = AutoImageProcessor.from_pretrained(HF_DIR)
model = AutoModelForImageClassification.from_pretrained(HF_DIR).eval()

# ---- 参考输出 (transformers 官方预处理 + PyTorch) ----
pil = [Image.open(p).convert("RGB") for p in images]
with torch.no_grad():
    inputs = proc(images=pil, return_tensors="pt")
    ref_logits = model(**inputs).logits
    ref = torch.softmax(ref_logits, dim=-1).numpy().astype(np.float32)
    ref_input = inputs["pixel_values"].numpy()  # NCHW

# ---- 我们的预处理，与官方 pixel_values 对比 ----
our = np.stack([nsfw_audit.preprocess(p) for p in images])  # N,196,768
our_chw = our.reshape(len(images), 14, 14, 3, 16, 16) \
             .transpose(0, 3, 1, 4, 2, 5).reshape(len(images), 3, 224, 224)
pre_err = np.abs(our_chw - ref_input).max()
print(f"预处理与 transformers pixel_values 最大偏差: {pre_err:.2e}")

# ---- GGUF 引擎输出 ----
p = subprocess.Popen([str(EXE), str(GGUF), "--batch", str(len(images)), "--device", "cpu"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
while not p.stderr.readline().startswith(b"READY"):
    pass
p.stdin.write(our.astype(np.float32).tobytes())
p.stdin.flush()
out = p.stdout.read(len(images) * 2 * 4)
p.stdin.close()
p.wait(timeout=60)
gguf_probs = np.frombuffer(out, np.float32).reshape(len(images), 2)

print(f"{'图片':<40s} {'ref_nsfw':>9} {'gguf_nsfw':>10} {'|diff|':>9}")
max_prob_diff = 0.0
for path, r, g in zip(images, ref, gguf_probs):
    d = abs(float(r[1]) - float(g[1]))
    max_prob_diff = max(max_prob_diff, d)
    print(f"{path.name[:40]:<40s} {r[1]:9.5f} {g[1]:10.5f} {d:9.2e}")

# logits 级别对比 (反 softmax，仅差一个常数项，比较 MAE/最大差)
ref_l = ref_logits.numpy().astype(np.float64)
gguf_l = np.log(np.clip(gguf_probs, 1e-12, 1.0)).astype(np.float64)
# 减去各自的均值抵消 softmax 常数平移
logit_diff = np.abs((ref_l - ref_l.mean(axis=1, keepdims=True))
                    - (gguf_l - gguf_l.mean(axis=1, keepdims=True)))
print(f"\n概率最大偏差: {max_prob_diff:.3e}")
print(f"logits 最大偏差: {logit_diff.max():.3e}, 平均: {logit_diff.mean():.3e}")
ok = logit_diff.max() < 5e-2 and pre_err < 1e-4 \
     and np.array_equal(ref.argmax(1), gguf_probs.argmax(1))
print("结论:", "PASS (与 PyTorch 参考实现一致)" if ok else "FAIL")
