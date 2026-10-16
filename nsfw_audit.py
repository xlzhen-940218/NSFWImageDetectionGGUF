# -*- coding: utf-8 -*-
"""
NSFW 图像审核工具
=================
使用 Falconsai/nsfw_image_detection 的 GGUF 版本 (ViT-B/16)，
通过 ggml 推理 (Vulkan 后端，同时支持 NVIDIA / AMD / Intel 显卡；
无可用 GPU 时自动回退 CPU)。

不合规图片会被移动到目录下的 NSFW 文件夹。

示例:
  py nsfw_audit.py                         # 审核当前目录
  py nsfw_audit.py --dir D:\\photos --threshold 0.9
  py nsfw_audit.py --recursive --dry-run   # 递归并只打印不移动
  py nsfw_audit.py --device cpu --batch 4  # 强制 CPU
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

# ---- 常量 (ViT-B/16) ----
IMAGE_SIZE = 224
PATCH = 16
GRID = IMAGE_SIZE // PATCH          # 14
N_PATCH = GRID * GRID               # 196
HIDDEN = 768
N_CLASS = 2
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

HERE = Path(__file__).resolve().parent
TOOL_DIR = HERE / "_nsfw_tool"
DEFAULT_EXE = TOOL_DIR / "build" / "bin" / "vit_nsfw.exe"
DEFAULT_MODEL = TOOL_DIR / "models" / "nsfw_image_detection.fp32.gguf"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
# 与 ViTImageProcessor(preprocessor_config.json) 一致: resample=2 (BILINEAR)
RESAMPLE = Image.BILINEAR


def preprocess(path: Path) -> np.ndarray:
    """图片 -> 196x768 的 patch 序列 (float32)。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        # size = {height:224, width:224}：直接缩放到 224x224，然后居中裁剪(同尺寸，no-op)
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE)
        w, h = im.size
        left = (w - IMAGE_SIZE) // 2
        top = (h - IMAGE_SIZE) // 2
        im = im.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        x = np.asarray(im, dtype=np.float32) / 255.0           # HWC
    x = (x - MEAN) / STD                                       # HWC 广播
    x = np.ascontiguousarray(x.transpose(2, 0, 1))             # CHW
    # unfold 成 patch 序列: (3,14,16,14,16) -> (196, 768)
    # 内层顺序 (C, kh, kw) 与卷积权重内存排布一致
    x = x.reshape(3, GRID, PATCH, GRID, PATCH)
    x = np.ascontiguousarray(x.transpose(1, 3, 0, 2, 4))
    return x.reshape(N_PATCH, HIDDEN)


class VitWorker:
    def __init__(self, exe: Path, model: Path, batch: int, device: str):
        self.batch = batch
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [str(exe), str(model), "--batch", str(batch), "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, creationflags=flags,
        )
        self.log_q: queue.Queue[str] = queue.Queue()
        self.err_t = threading.Thread(target=self._pump_stderr, daemon=True)
        self.err_t.start()
        self._wait_ready()

    def _pump_stderr(self) -> None:
        try:
            for raw in self.proc.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self.log_q.put(line)
        except Exception:
            pass

    def _wait_ready(self) -> None:
        while True:
            try:
                line = self.log_q.get(timeout=60)
            except queue.Empty:
                raise RuntimeError("推理引擎启动超时")
            if line.startswith("READY"):
                return
            print(f"  [引擎] {line}", file=sys.stderr)

    def infer(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames 数量 1..batch，返回 (n, 2) 概率。"""
        n = len(frames)
        assert 0 < n <= self.batch
        if n < self.batch:  # 末批不足时用最后一张补齐，结果只取前 n 个
            frames = frames + [frames[-1]] * (self.batch - n)
        payload = np.stack(frames, axis=0).astype(np.float32, copy=False)
        assert payload.shape == (self.batch, N_PATCH, HIDDEN)
        self.proc.stdin.write(payload.tobytes())
        self.proc.stdin.flush()
        out = self._read_exact(self.batch * N_CLASS * 4)
        probs = np.frombuffer(out, dtype=np.float32).reshape(self.batch, N_CLASS)
        return probs[:n].copy()

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("推理引擎提前退出")
            buf.extend(chunk)
            while True:        # 顺手排空 stderr 日志
                try:
                    line = self.log_q.get_nowait()
                    print(f"  [引擎] {line}", file=sys.stderr)
                except queue.Empty:
                    break
        return bytes(buf)

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def unique_destination(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand
        i += 1


def collect_images(root: Path, recursive: bool, nsfw_dir: Path) -> list[Path]:
    nsfw_dir = nsfw_dir.resolve()
    out = []
    iterator = root.rglob("*") if recursive else root.iterdir()
    for p in sorted(iterator):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            p.resolve().relative_to(nsfw_dir)
            continue  # 已在 NSFW 目录中的不处理
        except ValueError:
            pass
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="GGUF ViT NSFW 图像审核 (NVIDIA/AMD/Intel GPU)")
    ap.add_argument("--dir", default=".", help="要审核的目录 (默认当前目录)")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="nsfw 概率阈值，超过即判定不合规 (默认 0.9)")
    ap.add_argument("--batch", type=int, default=8, help="GPU 批大小 (默认 8)")
    ap.add_argument("--device", default="auto", choices=["auto", "vulkan", "cpu"])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--nsfw-dir", default=None, help="不合规图片目标目录 (默认 目录/NSFW)")
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印不移动")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    exe, model = Path(args.exe), Path(args.model)
    if not exe.exists():
        print(f"找不到推理引擎: {exe}", file=sys.stderr)
        return 2
    if not model.exists():
        print(f"找不到 GGUF 模型: {model}", file=sys.stderr)
        return 2

    nsfw_dir = Path(args.nsfw_dir).resolve() if args.nsfw_dir else root / "NSFW"
    files = collect_images(root, args.recursive, nsfw_dir)
    if not files:
        print("目录中没有可审核的图片。")
        return 0

    print(f"待审核图片: {len(files)} 张  |  阈值: {args.threshold:.2f}  |  设备: {args.device}")
    worker = VitWorker(exe, model, args.batch, args.device)

    bad: list[tuple[Path, float]] = []
    failed: list[Path] = []
    try:
        buf_paths: list[Path] = []
        buf_frames: list[np.ndarray] = []

        def flush_batch() -> None:
            if not buf_paths:
                return
            probs = worker.infer(buf_frames)
            for p, pr in zip(buf_paths, probs):
                p_nsfw = float(pr[1])
                flag = "NSFW" if p_nsfw >= args.threshold else "ok"
                print(f"  [{flag:>4}] {p_nsfw * 100:6.2f}%  {p.name}")
                if p_nsfw >= args.threshold:
                    bad.append((p, p_nsfw))
            buf_paths.clear()
            buf_frames.clear()

        for i, p in enumerate(files, 1):
            try:
                buf_frames.append(preprocess(p))
                buf_paths.append(p)
            except Exception as e:
                print(f"  [读取失败] {p.name}: {e}", file=sys.stderr)
                failed.append(p)
                continue
            if len(buf_paths) == args.batch:
                flush_batch()
                print(f"  --- 进度 {i}/{len(files)} ---")
        flush_batch()
    finally:
        worker.close()

    # 移动不合规图片
    moved = 0
    if bad and not args.dry_run:
        nsfw_dir.mkdir(parents=True, exist_ok=True)
    for p, score in bad:
        target = unique_destination(nsfw_dir / p.name)
        if args.dry_run:
            print(f"  [试运行] 将移动: {p} -> {target}")
            continue
        try:
            shutil.move(str(p), str(target))
            moved += 1
        except Exception as e:
            print(f"  [移动失败] {p.name}: {e}", file=sys.stderr)

    print()
    print("=" * 50)
    print(f"审核完成: 共 {len(files)} 张, 不合规 {len(bad)} 张")
    if args.dry_run:
        print("试运行模式，未执行任何移动。")
    elif moved:
        print(f"已移动 {moved} 张到: {nsfw_dir}")
    if failed:
        print(f"无法读取的文件 {len(failed)} 张", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
