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

一个高性能、极轻量、纯本地离线运行的 **NSFW 违规图像检测与自动隔离整理工具**。  
基于 `Falconsai/nsfw_image_detection`（ViT-B/16 架构）GGUF 量化模型，通过 `ggml` Vulkan 原生硬件加速后端（兼容 NVIDIA / AMD / Intel 显卡，无 GPU 时自动回退 CPU），在毫秒级内完成批量审核，并将违规图片自动移动至专属隔离目录。

提供 **超轻量 Windows 11 Fluent 桌面端 GUI 软件（单文件绿色版仅 1.4 MB）** 与 **格式化 C++ 命令行工具**。100% 离线隐私保障，绝无任何网络上传。

---

## ✨ 核心特性

- 🚀 **纯本地 GPU 加速推理**：
  采用 `ggml` Vulkan 统一后端，跨平台支持 NVIDIA GeForce/RTX、AMD Radeon 及 Intel Arc/核显，无需安装庞大的 Python/PyTorch 环境，低显存极速推演。
- 🖥️ **精美超轻量桌面端（WPF + Windows 11 Mica Fluent）**：
  - **极致小巧**：单文件绿色版仅 **1.4 MB**（彻底告别 WinUI 3 捆绑 WindowsAppSDK 的 138MB 臃肿体积），秒开无延时，内存占用极低。
  - **Windows 11 Fluent 视觉**：原生 Mica 材质半透明质感背景、深色模式标题栏（DWMAPI）、圆角卡片。
  - **左侧控制工作台 + 右侧内容画布**：清晰的左右分栏布局，消除多余堆叠与重复操作，视野开阔。
  - **实时进度与 4 项数据看板**：动态展示平滑进度条、总扫描数、合规安全数、NSFW 检出数与隔离移动数。
  - **交互式缩略图结果流**：高性能虚拟化列表流畅承载海量图片，异步解码渲染缩略图，提供高对比度红绿概率胶囊标签、一键看图与资源管理器定位。
  - **绝无控制台黑框弹出**：后台管道化异步流式调度引擎进程（`CreateNoWindow = true`），交互体验丝滑。
  - **自适应中英双语（Bilingual）**：自动根据操作系统语言环境匹配中文/英文界面，并支持随时手动一键切换。
  - **文件夹拖拽支持**：直接将文件夹拖入窗口即可迅速选定目标目录。
- ⚡ **格式化输出 C++ 审核引擎（`nsfw_audit.exe`）**：
  - 规范的英文标签输出（`[INIT]`, `[ITEM]`, `[PROGRESS]`, `[ACTION]`, `[SUMMARY]`）。
  - 支持 `--json` 逐行输出（NDJSON），自带 Windows 特殊路径转义，专为前端及自动化工作流设计。
  - 输出即时刷新（`fflush`），消除管道阻塞。
- 🛡️ **安全隔离机制**：
  - 自动创建并移动违规图片至 `NSFW/` 目录；
  - 自动处理重名文件冲突（自动递增序列命名，杜绝文件覆盖）；
  - 提供 **Dry-Run 试运行仿真模式**（只评估打印/显示，不产生实际文件位移）。

---

## 📂 项目结构

```text
NSFWImageDetectionGGUF/
├── publish/                    # 单文件绿色发布版目录
│   └── NsfwAuditGui.exe        # 单文件便携版客户端（仅 1.4 MB，双击即用）
├── NsfwAuditGui/               # 现代化 WPF 桌面客户端源码 (.NET 9 + Fluent)
│   ├── Models/                 # 核心数据实体 (AuditItem, AuditConfig)
│   ├── Services/               # 进程托管服务 (NsfwAuditProcessService) 与中英本地化 (LocalizationService)
│   ├── ViewModels/             # MVVM 视图模型 (MainViewModel)
│   ├── Converters/             # XAML 状态着色与可视绑定转换器
│   ├── MainWindow.xaml         # 现代化主界面 (左右分栏布局 + Mica/深色主题)
│   └── App.xaml                # 全局暗色 Fluent 控件样式与主题资源
├── _nsfw_tool/                 # C++ 核心推理引擎源码与构建脚本
│   ├── build/bin/              # 编译产物输出目录
│   │   ├── nsfw_audit.exe      # 格式化 C++ 审核引擎主程序
│   │   └── vit_nsfw.exe        # 标准 ViT 推理服务器
│   ├── models/                 # GGUF 视觉模型存储目录
│   │   └── nsfw_image_detection.fp32.gguf
│   ├── ggml/                   # ggml 核心张量运算库 (含 Vulkan/CPU 后端)
│   ├── CMakeLists.txt          # CMake 构建工程脚本
│   └── nsfw_audit.cpp          # 审核引擎源码 (支持英文格式化与 JSON 流式输出)
├── image-test/                 # 测试图片文件夹 (含合规壁纸与测试用例)
├── nsfw_audit.py               # Python 原型参考脚本
├── launch_gui.bat              # GUI 桌面程序一键启动脚本
├── README.md                   # 项目中文说明文档
└── README_EN.md                # English Documentation
```

---

## 🚀 快速上手

### 1. 运行图形界面客户端（GUI）

最简便的方式：直接双击根目录下的 **`launch_gui.bat`**。

或者直接双击运行绿色单文件版本：
```text
publish\NsfwAuditGui.exe
```

在界面中：
1. **选择待审核目录**：点击“📁”选择目录，或直接将文件夹拖拽进入窗口；
2. **调整参数（可选）**：设置判定阈值（默认 90%）、推理设备（Auto/Vulkan/CPU）、批处理大小等；
3. **点击“开始审核”**：软件将全自动调用 `nsfw_audit.exe` 进行本地硬件加速检测，并在界面实时列出缩略图及状态。

---

### 2. 命令行调用（CLI）

可以直接在终端中运行 `nsfw_audit.exe`：

```bash
# 1. 审核指定目录（默认阈值 0.90，自动选择 Vulkan GPU 显卡）
_nsfw_tool\build\bin\nsfw_audit.exe --dir "D:\Photos"

# 2. 试运行模式（仅打印检测结果，不移动任何文件）
_nsfw_tool\build\bin\nsfw_audit.exe --dir "image-test" --dry-run

# 3. 指定概率阈值、批大小并强制使用 Vulkan 后端
_nsfw_tool\build\bin\nsfw_audit.exe --dir "D:\Photos" --threshold 0.85 --batch 8 --device vulkan

# 4. 递归扫描所有子目录，并输出 JSON 流（供第三方程序/脚本集成）
_nsfw_tool\build\bin\nsfw_audit.exe --dir "D:\Photos" --recursive --json
```

#### 命令行输出格式示例：

- **普通格式化文本（Formatted Text，默认）**：
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

- **JSON 流格式（`--json`）**：
```json
{"type":"init","total":14,"threshold":0.9000,"batch":8,"device":"auto","recursive":false,"dry_run":false,"dir":"...","nsfw_dir":"..."}
{"type":"item","status":"SAFE","score":0.0001,"percent":0.01,"file":"0001.png","path":"..."}
{"type":"item","status":"NSFW","score":0.9672,"percent":96.72,"file":"0085.png","path":"..."}
{"type":"progress","current":8,"total":14,"percent":57.14}
{"type":"action","status":"MOVED","src":"...","dst":"..."}
{"type":"summary","total":14,"safe":10,"nsfw":4,"moved":4,"failed":0,"dry_run":false}
```

---

## 🛠️ 参数说明

| 参数项 | 默认值 | 作用说明 |
| :--- | :---: | :--- |
| `--dir <path>` | 当前目录 | 待审核的目标图片目录路径 |
| `--threshold <float>` | `0.90` | NSFW 判定概率阈值（0.0 ~ 1.0），达到或超过即判定为违规 |
| `--batch <int>` | `8` | 推理批处理大小（1 ~ 128），显存充裕时调大可显著提升吞吐速度 |
| `--device <auto\|vulkan\|cpu>` | `auto` | 计算设备后端：`auto` 优先探测 Vulkan GPU，无可用时回退 CPU |
| `--model <path>` | 默认模型路径 | 自定义 GGUF 模型文件绝对或相对路径 |
| `--nsfw-dir <path>` | `<dir>\NSFW` | 违规图片隔离移动目标目录 |
| `--recursive` | 否 | 是否递归遍历所有子目录中的图片 |
| `--dry-run` | 否 | 试运行模式，仅检测与打印结果，不执行文件移动 |
| `--format <text\|json>` | `text` | 控制台标准输出格式 |
| `--json` | - | 等价于 `--format json`，输出机器解析友好的 NDJSON 流 |
| `-h, --help` | - | 查看命令行使用帮助 |

---

## 🔨 编译与发布指南

### 1. 编译 C++ 引擎 (`nsfw_audit.exe`)

```cmd
# 打开 x64 Native Tools Command Prompt for VS 2022
cd /d %~dp0\_nsfw_tool
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja nsfw_audit
```
产物将生成在 `_nsfw_tool\build\bin\nsfw_audit.exe`。

### 2. 编译并打包单文件 GUI 桌面端 (`NsfwAuditGui.exe`)

```cmd
cd /d %~dp0
dotnet publish NsfwAuditGui\NsfwAuditGui.csproj -c Release -r win-x64 --self-contained false -p:PublishSingleFile=true -o publish
```
输出产物直接位于 `publish\NsfwAuditGui.exe`，单文件大小仅约 **1.4 MB**。

---

## 🔒 隐私与安全声明

- **本地离线**：本项目所有推理与图像解码均在使用者本地显卡/CPU 完成，不需要也不发起任何互联网网络连接。
- **防止覆盖**：在移动违规文件至隔离目录时，如果遇到同名文件，程序会自动重命名为 `filename (1).ext`、`filename (2).ext`，杜绝误覆盖。

---

## 📄 开源许可证

本项目遵循 MIT 许可证发布。使用的 ViT 权重与预训练模型源自开源社区，归原作者所有。
