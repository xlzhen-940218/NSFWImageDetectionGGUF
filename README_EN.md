# NSFW Image Detection GGUF (ViT-B/16)

<p align="center">
  <a href="README.md"><b>简体中文</b></a> | <a href="README_EN.md"><b>English</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D6?logo=windows" alt="Platform" />
  <img src="https://img.shields.io/badge/.NET-9.0%20%7C%20WPF%20Fluent-512BD4?logo=dotnet" alt="Framework" />
  <img src="https://img.shields.io/badge/ggml-Vulkan%20GPU%20%7C%20CPU-red?logo=vulkan" alt="Backend" />
  <img src="https://img.shields.io/badge/Model-ViT--B%2F16%20GGUF-orange" alt="Model" />
  <img src="https://img.shields.io/badge/GUI%20Size-1.4%20MB-brightgreen" alt="GUI Size" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
</p>

A high-performance, ultra-lightweight, 100% offline local tool for **NSFW image detection and automatic quarantine**.  
Powered by the `Falconsai/nsfw_image_detection` (ViT-B/16 architecture) quantized GGUF model via the `ggml` Vulkan hardware acceleration backend (compatible with NVIDIA, AMD, and Intel GPUs, with automatic CPU fallback). It performs batch image audits in milliseconds and automatically moves detected NSFW images into a designated quarantine directory.

Includes an **ultra-lightweight Windows 11 Fluent desktop GUI (standalone single-file executable of just 1.4 MB)** and a **formatted C++ command-line interface (CLI)**. 100% offline privacy guarantee with zero internet communication.

---

## ✨ Key Features

- 🚀 **Pure Local GPU Inference**:
  Leverages the `ggml` Vulkan unified backend, supporting NVIDIA GeForce/RTX, AMD Radeon, and Intel Arc/Integrated GPUs without requiring bulky Python or PyTorch runtimes. Ultra-low VRAM footprint and instant startup.
- 🖥️ **Stunning Lightweight Desktop Client (WPF + Windows 11 Mica Fluent)**:
  - **Extremely Compact**: Standalone single-file executable is only **1.4 MB** (no WinUI 3 WindowsAppSDK bundle bloat), starts instantly with negligible memory usage.
  - **Windows 11 Fluent Aesthetics**: Native DWM Mica backdrop material, immersive dark title bar, and rounded card styling.
  - **Left Sidebar + Right Canvas Layout**: Clean two-column workbench layout eliminating clutter and redundant controls.
  - **Live Metrics Dashboard**: Real-time progress bar, Total Scanned, Safe Images, NSFW Detected, and Moved Quarantine counters.
  - **Interactive Thumbnail Results Stream**: High-performance virtualized image list with asynchronous background thumbnail decoding (`Freeze()`), high-contrast probability pill badges, and quick actions (view image / show in explorer).
  - **No Console Window Flickers**: Background engine dispatch via anonymous pipes (`CreateNoWindow = true`) ensures a seamless desktop experience.
  - **Adaptive Bilingual Support**: Automatically matches system UI culture (Chinese/English) with an instant manual switch.
  - **Drag-and-Drop Support**: Simply drop folders into the window to immediately begin scanning.
- ⚡ **Formatted C++ Engine (`nsfw_audit.exe`)**:
  - Structured English log tags (`[INIT]`, `[ITEM]`, `[PROGRESS]`, `[ACTION]`, `[SUMMARY]`).
  - Optional `--json` flag for machine-readable NDJSON streaming with proper Windows path escaping.
  - Real-time `fflush(stdout)` preventing pipe buffering latency.
- 🛡️ **Safe Quarantine Pipeline**:
  - Automatically isolates NSFW images into the `NSFW/` folder.
  - Automatic collision resolution: never overwrites existing files (appends `(1)`, `(2)`, etc.).
  - **Dry-Run Simulation**: Scan and inspect without touching or moving any files.

---

## 📂 Project Structure

```text
NSFWImageDetectionGGUF/
├── publish/                    # Published single-file binaries
│   └── NsfwAuditGui.exe        # Portable single-file GUI client (only 1.4 MB)
├── NsfwAuditGui/               # Modern WPF desktop client source (.NET 9 + Fluent)
│   ├── Models/                 # Data entities (AuditItem, AuditConfig)
│   ├── Services/               # Process service (NsfwAuditProcessService) & localization (LocalizationService)
│   ├── ViewModels/             # MVVM ViewModel (MainViewModel)
│   ├── Converters/             # XAML value converters for status badges & visibility
│   ├── MainWindow.xaml         # Main interface (2-column layout + Mica dark theme)
│   └── App.xaml                # Global dark Fluent styles and brushes
├── _nsfw_tool/                 # C++ inference engine source & build files
│   ├── build/bin/              # Compiled engine binaries
│   │   ├── nsfw_audit.exe      # Formatted C++ audit engine
│   │   └── vit_nsfw.exe        # Core ViT inference runner
│   ├── models/                 # Quantized GGUF model storage
│   │   └── nsfw_image_detection.fp32.gguf
│   ├── ggml/                   # ggml tensor library (Vulkan / CPU backends)
│   ├── CMakeLists.txt          # CMake project definition
│   └── nsfw_audit.cpp          # Engine source with formatted tags and NDJSON output
├── image-test/                 # Test images directory
├── nsfw_audit.py               # Python reference prototype
├── launch_gui.bat              # One-click launcher script
├── README.md                   # Chinese documentation
└── README_EN.md                # English documentation
```

---

## 🚀 Quick Start

### 1. Launching the GUI Desktop Client

The easiest method: double-click **`launch_gui.bat`** in the repository root.

Or directly run the portable standalone executable:
```text
publish\NsfwAuditGui.exe
```

Workflow:
1. **Select Target Directory**: Click "📁" or drag-and-drop a folder into the window;
2. **Configure Parameters (Optional)**: Set threshold (default 90%), device (Auto/Vulkan/CPU), batch size, etc.;
3. **Click "Start Audit"**: The client will execute `nsfw_audit.exe` with local GPU acceleration and populate thumbnail cards in real time.

---

### 2. Command Line Interface (CLI)

Run `nsfw_audit.exe` directly from PowerShell or Command Prompt:

```bash
# 1. Audit a target directory (default threshold: 0.90, auto Vulkan GPU)
_nsfw_tool\build\bin\nsfw_audit.exe --dir "D:\Photos"

# 2. Dry-run mode (print scores without moving any files)
_nsfw_tool\build\bin\nsfw_audit.exe --dir "image-test" --dry-run

# 3. Specify custom threshold, batch size, and force Vulkan GPU
_nsfw_tool\build\bin\nsfw_audit.exe --dir "D:\Photos" --threshold 0.85 --batch 8 --device vulkan

# 4. Recursively scan subdirectories and output streaming NDJSON
_nsfw_tool\build\bin\nsfw_audit.exe --dir "D:\Photos" --recursive --json
```

#### Output Stream Examples:

- **Formatted Text (Default)**:
```text
[INIT] total=14 | threshold=0.90 | batch=8 | device=auto | dir=C:\...\image-test
[INFO] Auditing 14 images | Threshold: 0.90 | Device: auto | Batch: 8
[BACKEND] Vulkan0 - AMD Radeon(TM) Graphics
[MODEL] layers=12 heads=12 hidden=768 patches=196 classes=2
[ITEM] status=SAFE | score=0.0001 | percent=  0.01% | file=0001.png | path=C:\...\0001.png
  [ok  ]   0.01%  0001.png
[ITEM] status=NSFW | score=0.9672 | percent= 96.72% | file=0085.png | path=C:\...\0085.png
  [NSFW]  96.72%  0085.png
[PROGRESS] current=8 | total=14 | percent=57.14%
[ACTION] status=MOVED | src=C:\...\0085.png | dst=C:\...\NSFW\0085.png
==================================================
[SUMMARY] total=14 | safe=10 | nsfw=4 | moved=4 | failed=0 | dry_run=false
Audit finished: 14 total, 10 safe, 4 NSFW.
Moved 4 images to: C:\...\NSFW
```

- **JSON Stream (`--json`)**:
```json
{"type":"init","total":14,"threshold":0.9000,"batch":8,"device":"auto","recursive":false,"dry_run":false,"dir":"...","nsfw_dir":"..."}
{"type":"item","status":"SAFE","score":0.0001,"percent":0.01,"file":"0001.png","path":"..."}
{"type":"item","status":"NSFW","score":0.9672,"percent":96.72,"file":"0085.png","path":"..."}
{"type":"progress","current":8,"total":14,"percent":57.14}
{"type":"action","status":"MOVED","src":"...","dst":"..."}
{"type":"summary","total":14,"safe":10,"nsfw":4,"moved":4,"failed":0,"dry_run":false}
```

---

## 🛠️ CLI Options Reference

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--dir <path>` | Current dir | Target directory containing images to audit |
| `--threshold <float>` | `0.90` | NSFW probability threshold (0.0 - 1.0) for quarantine classification |
| `--batch <int>` | `8` | Inference batch size (1 - 128) |
| `--device <auto\|vulkan\|cpu>` | `auto` | Device backend: `auto` selects Vulkan GPU, falls back to CPU |
| `--model <path>` | Default path | Custom absolute or relative path to GGUF model file |
| `--nsfw-dir <path>` | `<dir>\NSFW` | Quarantine destination folder |
| `--recursive` | Disabled | Recursively traverse images in subdirectories |
| `--dry-run` | Disabled | Simulation mode: evaluate and print results without moving files |
| `--format <text\|json>` | `text` | Standard output format |
| `--json` | - | Shorthand for `--format json` (streams machine-readable NDJSON) |
| `-h, --help` | - | Display CLI usage help |

---

## 🔨 Build & Publish Instructions

### 1. Build C++ Engine (`nsfw_audit.exe`)

```cmd
# Open x64 Native Tools Command Prompt for VS 2022
cd /d %~dp0\_nsfw_tool
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja nsfw_audit
```
Binary will be created at `_nsfw_tool\build\bin\nsfw_audit.exe`.

### 2. Build & Publish Standalone GUI Client (`NsfwAuditGui.exe`)

```cmd
cd /d %~dp0
dotnet publish NsfwAuditGui\NsfwAuditGui.csproj -c Release -r win-x64 --self-contained false -p:PublishSingleFile=true -o publish
```
The output executable will be created at `publish\NsfwAuditGui.exe` (only **~1.4 MB**).

---

## 🔒 Privacy & Safety Guarantee

- **100% Offline**: All image decoding and tensor operations occur strictly on your local GPU/CPU. No network requests are made.
- **Anti-Overwrite Protection**: When quarantining files with identical filenames, names are automatically resolved to `filename (1).ext`, `filename (2).ext` to prevent accidental data loss.

---

## 📄 License

This project is licensed under the MIT License. ViT weights and pretrained models originate from the open-source community and belong to their respective authors.
