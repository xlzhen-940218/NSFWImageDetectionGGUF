# -*- coding: utf-8 -*-
"""验证 nsfw_audit.exe: transformers 参考概率 vs C++ 审核器输出 (预处理新实现)。"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

EXE = TOOL / "build" / "bin" / "nsfw_audit.exe"

# 6 张普通图 + NSFW 文件夹里的 4 张
normal = sorted(ROOT.glob("*.png"))[:6]
nsfw = sorted((ROOT / "NSFW").glob("*.png"))
samples = normal + nsfw

tmp = TOOL / "val_tmp"
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
for p in samples:
    shutil.copy2(p, tmp / p.name)

proc = AutoImageProcessor.from_pretrained(TOOL / "model_hf")
model = AutoModelForImageClassification.from_pretrained(TOOL / "model_hf").eval()
pil = [Image.open(p).convert("RGB") for p in samples]
with torch.no_grad():
    ref = torch.softmax(model(**proc(images=pil, return_tensors="pt")).logits, dim=-1).numpy()
ref_pct = {p.name: float(r[1]) * 100 for p, r in zip(samples, ref)}

def run(device: str):
    out = subprocess.run(
        [str(EXE), "--dir", str(tmp), "--batch", "6", "--device", device,
         "--threshold", "0.9", "--dry-run"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    got = {}
    for line in out.stdout.splitlines():
        m = re.search(r"\[\s*(ok|NSFW)\]\s+([\d.]+)%\s+(\S+)", line)
        if m:
            got[m.group(3)] = (m.group(1), float(m.group(2)))
    return got, out

max_diff = 0.0
ok_all = True
for device in ("cpu", "vulkan"):
    got, out = run(device)
    print(f"===== {device} =====")
    if out.returncode != 0:
        print(out.stdout[-500:]); print(out.stderr[-500:]); ok_all = False; continue
    print(f"{'图片':<40s}{'参考':>9}{'C++':>9}{'差':>8}  判定")
    for p in samples:
        flag, g = got[p.name]
        r = ref_pct[p.name]
        d = abs(r - g)
        max_diff = max(max_diff, d)
        ref_flag = "NSFW" if r >= 90 else "ok"
        match = "✓" if flag == ref_flag else "✗ 判定不一致!"
        if flag != ref_flag:
            ok_all = False
        print(f"{p.name[:40]:<40s}{r:9.2f}{g:9.2f}{d:8.3f}  {match}")

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n概率最大偏差: {max_diff:.3f} 个百分点")
print("结论:", "PASS (C++ 预处理与判定与参考实现一致)" if ok_all and max_diff < 0.5 else "FAIL")
