using System;
using System.ComponentModel;
using System.Globalization;

namespace NsfwAuditGui.Services;

public class LocalizationService : INotifyPropertyChanged
{
    private static LocalizationService? _instance;
    public static LocalizationService Instance => _instance ??= new LocalizationService();

    public event PropertyChangedEventHandler? PropertyChanged;
    private void NotifyAll() => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(null));

    private string _currentLanguage = "auto";
    public string CurrentLanguage
    {
        get => _currentLanguage;
        set
        {
            if (_currentLanguage != value)
            {
                _currentLanguage = value;
                NotifyAll();
            }
        }
    }

    public bool IsChinese
    {
        get
        {
            if (CurrentLanguage == "zh") return true;
            if (CurrentLanguage == "en") return false;
            // "auto": detect system UI culture
            var culture = CultureInfo.CurrentUICulture;
            return culture.Name.StartsWith("zh", StringComparison.OrdinalIgnoreCase);
        }
    }

    // ================= String Resources =================

    public string AppTitle => IsChinese ? "NSFW 图像审核工具" : "NSFW Image Audit Studio";
    public string AppSubtitle => IsChinese ? "基于 ViT-B/16 GGUF 本地离线模型，支持 Vulkan GPU 显卡加速" : "Local ViT-B/16 GGUF offline model with Vulkan GPU acceleration";
    public string TagVitModel => IsChinese ? "ViT-B/16 GGUF 模型" : "ViT-B/16 GGUF";
    public string TagHardware => IsChinese ? "Vulkan 硬件加速 & CPU" : "Vulkan GPU & CPU";

    // Actions
    public string StartAudit => IsChinese ? "开始审核" : "Start Audit";
    public string StopAudit => IsChinese ? "停止审核" : "Stop Audit";
    public string ClearResults => IsChinese ? "清空结果" : "Clear Results";
    public string ClearLogs => IsChinese ? "清空日志" : "Clear Logs";
    public string CopyLogs => IsChinese ? "复制日志" : "Copy Logs";
    public string Browse => IsChinese ? "浏览..." : "Browse...";
    public string OpenFolder => IsChinese ? "打开文件夹" : "Open Folder";
    public string OpenFile => IsChinese ? "查看原图" : "Open Image";
    public string LocateFile => IsChinese ? "在资源管理器中定位" : "Show in Explorer";

    // Folders & Options
    public string TargetFolderTitle => IsChinese ? "待审核目录" : "Target Directory";
    public string TargetFolderPlaceholder => IsChinese ? "选择文件夹或直接将文件夹拖入窗口..." : "Select folder or drag and drop folder here...";
    public string NsfwFolderTitle => IsChinese ? "NSFW 隔离目录" : "NSFW Quarantine Directory";
    public string NsfwFolderPlaceholder => IsChinese ? "默认：[待审核目录]\\NSFW" : "Default: [Target Directory]\\NSFW";
    public string ThresholdTitle => IsChinese ? "判定阈值 (超过即判定为违规)" : "NSFW Threshold";
    public string DeviceTitle => IsChinese ? "推理设备" : "Hardware Device";
    public string BatchSizeTitle => IsChinese ? "批处理大小" : "Batch Size";
    public string RecursiveTitle => IsChinese ? "递归扫描子目录" : "Scan Subdirectories";
    public string DryRunTitle => IsChinese ? "试运行模式 (仅检测不移动)" : "Dry Run (Scan Only, No Move)";
    public string DryRunTooltip => IsChinese ? "仅扫描并输出判定结果，不移动任何图片文件" : "Simulate audit without moving any image files";

    // Advanced
    public string AdvancedHeader => IsChinese ? "高级引擎与模型配置" : "Advanced Engine & Model Configuration";
    public string EnginePathLabel => IsChinese ? "审核引擎程序:" : "Audit Engine:";
    public string ModelPathLabel => IsChinese ? "GGUF 模型文件:" : "GGUF Model File:";

    // Metrics Dashboard
    public string MetricTotal => IsChinese ? "扫描总数" : "Total Scanned";
    public string MetricSafe => IsChinese ? "合规安全" : "Safe Images";
    public string MetricNsfw => IsChinese ? "违规图片" : "NSFW Detected";
    public string MetricMoved => IsChinese ? "已隔离移动" : "Moved to Quarantine";

    // Tabs & Filters
    public string TabResults => IsChinese ? "审核明细" : "Audit Results";
    public string TabLogs => IsChinese ? "运行日志" : "Engine Output Logs";
    public string FilterAll => IsChinese ? "全部图片" : "All Images";
    public string FilterNsfw => IsChinese ? "仅违规 (NSFW)" : "NSFW Only";
    public string FilterSafe => IsChinese ? "仅安全 (Safe)" : "Safe Only";
    public string ShowingItems => IsChinese ? "当前显示" : "Showing";
    public string ItemsSuffix => IsChinese ? "项" : "items";

    // Status Texts
    public string StatusReady => IsChinese ? "就绪，请选择待审核目录后点击开始。" : "Ready. Select a target directory and click Start.";
    public string StatusInitializing => IsChinese ? "正在加载 GGUF 模型与预热推理引擎..." : "Initializing inference engine and loading model...";
    public string StatusScanning => IsChinese ? "正在审核" : "Auditing";
    public string StatusCompleted => IsChinese ? "审核完成" : "Audit completed";
    public string StatusCancelled => IsChinese ? "审核已由用户取消。" : "Audit cancelled by user.";
    public string StatusDryRunMoved => IsChinese ? "拟移动 (试运行)" : "Would move (Dry Run)";
    public string StatusMoved => IsChinese ? "已移至 NSFW 隔离文件夹" : "Moved to NSFW folder";

    // Notifications
    public string MsgFolderPicked => IsChinese ? "目录已更新" : "Folder Selected";
    public string MsgFolderInvalid => IsChinese ? "目录无效，请重新选择。" : "Please select a valid directory.";
    public string MsgEngineMissing => IsChinese ? "未找到 nsfw_audit.exe 引擎程序。" : "Cannot find nsfw_audit.exe executable.";
    public string MsgModelMissing => IsChinese ? "未找到 GGUF 模型文件。" : "Cannot find GGUF model file.";
    public string MsgLogsCopied => IsChinese ? "日志已复制到剪贴板。" : "Logs copied to clipboard.";
    public string MsgAuditStarted => IsChinese ? "正在执行本地图像安全审核..." : "Analyzing images with GGUF ViT model...";

    // Language Selector
    public string LanguageLabel => IsChinese ? "界面语言" : "Language";
    public string LangAuto => IsChinese ? "跟随系统" : "System Auto";
    public string LangZh => "简体中文";
    public string LangEn => "English";
}
