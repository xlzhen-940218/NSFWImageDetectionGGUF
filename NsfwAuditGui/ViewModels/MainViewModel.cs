using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using NsfwAuditGui.Models;
using NsfwAuditGui.Services;

namespace NsfwAuditGui.ViewModels;

public enum InfoSeverity
{
    Informational,
    Success,
    Warning,
    Error
}

public partial class MainViewModel : ObservableObject
{
    private readonly NsfwAuditProcessService _processService = new();
    private CancellationTokenSource? _cts;

    public MainViewModel()
    {
        TargetDirectory = NsfwAuditProcessService.FindDefaultImageDir();
        ExePath = NsfwAuditProcessService.FindDefaultExePath();
        ModelPath = NsfwAuditProcessService.FindDefaultModelPath();
        StatusMessage = L10n.StatusReady;

        L10n.PropertyChanged += (_, _) =>
        {
            OnPropertyChanged(nameof(L10n));
            if (!IsScanning) StatusMessage = L10n.StatusReady;
            ApplyFilter();
        };

        _processService.LogReceived += OnLogReceived;
        _processService.InitReceived += OnInitReceived;
        _processService.ItemReceived += OnItemReceived;
        _processService.ProgressReceived += OnProgressReceived;
        _processService.ActionReceived += OnActionReceived;
        _processService.SummaryReceived += OnSummaryReceived;
        _processService.ErrorReceived += OnErrorReceived;
    }

    public LocalizationService L10n => LocalizationService.Instance;

    [ObservableProperty]
    private string _selectedLanguage = "auto";

    partial void OnSelectedLanguageChanged(string value)
    {
        L10n.CurrentLanguage = value;
    }

    [ObservableProperty]
    private string _targetDirectory = string.Empty;

    [ObservableProperty]
    private string _nsfwDirectory = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ThresholdPercentText))]
    private double _threshold = 0.90;

    public string ThresholdPercentText => $"{(int)(Threshold * 100)}%";

    [ObservableProperty]
    private string _selectedDevice = "auto";

    public ObservableCollection<string> AvailableDevices { get; } = ["auto", "vulkan", "cpu"];

    [ObservableProperty]
    private int _selectedBatchSize = 8;

    public ObservableCollection<int> AvailableBatchSizes { get; } = [1, 2, 4, 8, 16, 32];

    [ObservableProperty]
    private bool _isRecursive = false;

    [ObservableProperty]
    private bool _isDryRun = false;

    [ObservableProperty]
    private string _exePath = string.Empty;

    [ObservableProperty]
    private string _modelPath = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanScan))]
    [NotifyPropertyChangedFor(nameof(CanCancel))]
    private bool _isScanning = false;

    public bool CanScan => !IsScanning;
    public bool CanCancel => IsScanning;

    [ObservableProperty]
    private bool _isIndeterminate = false;

    [ObservableProperty]
    private int _totalImages = 0;

    [ObservableProperty]
    private int _currentIndex = 0;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ProgressPercentText))]
    private double _progressPercent = 0.0;

    public string ProgressPercentText => $"{ProgressPercent:F1}%";

    [ObservableProperty]
    private int _safeCount = 0;

    [ObservableProperty]
    private int _nsfwCount = 0;

    [ObservableProperty]
    private int _movedCount = 0;

    [ObservableProperty]
    private int _failedCount = 0;

    [ObservableProperty]
    private string _statusMessage = "Ready to audit. Select a directory and start.";

    [ObservableProperty]
    private string _selectedFilter = "All";

    [ObservableProperty]
    private bool _infoBarIsOpen = false;

    [ObservableProperty]
    private string _infoBarTitle = string.Empty;

    [ObservableProperty]
    private string _infoBarMessage = string.Empty;

    [ObservableProperty]
    private InfoSeverity _infoBarSeverity = InfoSeverity.Informational;

    public ObservableCollection<AuditItem> AuditItems { get; } = [];
    public ObservableCollection<AuditItem> FilteredItems { get; } = [];
    public ObservableCollection<string> EngineLogs { get; } = [];

    [RelayCommand]
    public async Task StartAuditAsync()
    {
        if (IsScanning) return;

        if (string.IsNullOrWhiteSpace(TargetDirectory) || !Directory.Exists(TargetDirectory))
        {
            ShowInfoBar(L10n.TargetFolderTitle, L10n.MsgFolderInvalid, InfoSeverity.Warning);
            return;
        }

        if (string.IsNullOrWhiteSpace(ExePath) || !File.Exists(ExePath))
        {
            ShowInfoBar("Error", $"{L10n.MsgEngineMissing} ({ExePath})", InfoSeverity.Error);
            return;
        }

        if (string.IsNullOrWhiteSpace(ModelPath) || !File.Exists(ModelPath))
        {
            ShowInfoBar("Error", $"{L10n.MsgModelMissing} ({ModelPath})", InfoSeverity.Error);
            return;
        }

        ClearResults();
        IsScanning = true;
        IsIndeterminate = true;
        StatusMessage = L10n.StatusInitializing;
        ShowInfoBar(L10n.AppTitle, L10n.MsgAuditStarted, InfoSeverity.Informational);

        _cts = new CancellationTokenSource();

        var config = new AuditConfig
        {
            TargetDirectory = TargetDirectory,
            NsfwDirectory = NsfwDirectory,
            Threshold = Threshold,
            BatchSize = SelectedBatchSize,
            Device = SelectedDevice,
            Recursive = IsRecursive,
            DryRun = IsDryRun,
            ExePath = ExePath,
            ModelPath = ModelPath
        };

        try
        {
            int exitCode = await Task.Run(() => _processService.RunAuditAsync(config, _cts.Token));
            if (exitCode == 0)
            {
                string summaryText = L10n.IsChinese
                    ? $"审核完成：共扫描 {TotalImages} 张，发现违规 {NsfwCount} 张，移动隔离 {MovedCount} 张。"
                    : $"Audit finished: {TotalImages} images scanned, {NsfwCount} NSFW detected, {MovedCount} moved.";
                StatusMessage = summaryText;
                ShowInfoBar(L10n.StatusCompleted, summaryText, InfoSeverity.Success);
            }
            else
            {
                StatusMessage = $"Engine exited with code {exitCode}. Check logs for details.";
                ShowInfoBar("Warning", StatusMessage, InfoSeverity.Warning);
            }
        }
        catch (OperationCanceledException)
        {
            StatusMessage = L10n.StatusCancelled;
            ShowInfoBar(L10n.StatusCancelled, StatusMessage, InfoSeverity.Informational);
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
            ShowInfoBar("Error", StatusMessage, InfoSeverity.Error);
        }
        finally
        {
            IsScanning = false;
            IsIndeterminate = false;
        }
    }

    [RelayCommand]
    public void CancelAudit()
    {
        if (!IsScanning) return;
        StatusMessage = L10n.StatusCancelled;
        _cts?.Cancel();
        _processService.Cancel();
    }

    [RelayCommand]
    public void ClearResults()
    {
        AuditItems.Clear();
        FilteredItems.Clear();
        TotalImages = 0;
        CurrentIndex = 0;
        ProgressPercent = 0.0;
        SafeCount = 0;
        NsfwCount = 0;
        MovedCount = 0;
        FailedCount = 0;
        StatusMessage = L10n.StatusReady;
        InfoBarIsOpen = false;
    }

    [RelayCommand]
    public void ClearLogs()
    {
        EngineLogs.Clear();
    }

    [RelayCommand]
    public void OpenTargetFolder()
    {
        try
        {
            if (Directory.Exists(TargetDirectory))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = TargetDirectory,
                    UseShellExecute = true
                });
            }
        }
        catch { }
    }

    [RelayCommand]
    public void OpenNsfwFolder()
    {
        try
        {
            string nsfwDir = string.IsNullOrWhiteSpace(NsfwDirectory)
                ? Path.Combine(TargetDirectory, "NSFW")
                : NsfwDirectory;

            if (!Directory.Exists(nsfwDir))
            {
                Directory.CreateDirectory(nsfwDir);
            }

            Process.Start(new ProcessStartInfo
            {
                FileName = nsfwDir,
                UseShellExecute = true
            });
        }
        catch { }
    }

    [RelayCommand]
    public void SetFilter(string filter)
    {
        SelectedFilter = filter;
        ApplyFilter();
    }

    public void ApplyFilter()
    {
        FilteredItems.Clear();
        var items = SelectedFilter switch
        {
            "NSFW" => AuditItems.Where(x => x.IsNsfw),
            "Safe" => AuditItems.Where(x => x.IsSafe),
            _ => AuditItems
        };

        foreach (var item in items)
        {
            FilteredItems.Add(item);
        }
    }

    public void ShowInfoBar(string title, string message, InfoSeverity severity)
    {
        RunOnUi(() =>
        {
            InfoBarTitle = title;
            InfoBarMessage = message;
            InfoBarSeverity = severity;
            InfoBarIsOpen = true;
        });
    }

    private void RunOnUi(Action action)
    {
        var dispatcher = Application.Current?.Dispatcher;
        if (dispatcher != null && !dispatcher.CheckAccess())
        {
            dispatcher.BeginInvoke(action);
        }
        else
        {
            action();
        }
    }

    private void OnLogReceived(string log)
    {
        RunOnUi(() =>
        {
            EngineLogs.Add($"[{DateTime.Now:HH:mm:ss}] {log}");
            if (EngineLogs.Count > 1000) EngineLogs.RemoveAt(0);
        });
    }

    private void OnInitReceived(int total, double threshold, string device)
    {
        RunOnUi(() =>
        {
            TotalImages = total;
            IsIndeterminate = false;
            StatusMessage = $"Auditing {total} images on {device}...";
        });
    }

    private void OnItemReceived(string status, double score, double percent, string file, string path)
    {
        RunOnUi(() =>
        {
            var auditStatus = status switch
            {
                "NSFW" => AuditStatus.Nsfw,
                "SAFE" => AuditStatus.Safe,
                "ERROR" => AuditStatus.Error,
                _ => AuditStatus.Pending
            };

            var item = new AuditItem
            {
                FilePath = path,
                Score = score,
                Status = auditStatus
            };

            if (auditStatus == AuditStatus.Nsfw) NsfwCount++;
            else if (auditStatus == AuditStatus.Safe) SafeCount++;
            else if (auditStatus == AuditStatus.Error) FailedCount++;

            AuditItems.Add(item);

            // Thumbnail loaded asynchronously in background
            _ = item.LoadThumbnailAsync();

            if (SelectedFilter == "All" ||
                (SelectedFilter == "NSFW" && item.IsNsfw) ||
                (SelectedFilter == "Safe" && item.IsSafe))
            {
                FilteredItems.Add(item);
            }
        });
    }

    private void OnProgressReceived(int current, int total, double percent)
    {
        RunOnUi(() =>
        {
            CurrentIndex = current;
            TotalImages = total;
            ProgressPercent = percent;
            StatusMessage = $"Auditing: {current}/{total} ({percent:F1}%)";
        });
    }

    private void OnActionReceived(string status, string src, string dst)
    {
        RunOnUi(() =>
        {
            var match = AuditItems.FirstOrDefault(x => string.Equals(x.FilePath, src, StringComparison.OrdinalIgnoreCase));
            if (match != null)
            {
                if (status == "MOVED")
                {
                    match.Status = AuditStatus.Moved;
                    match.DestinationPath = dst;
                    match.ActionText = "Moved to NSFW";
                    MovedCount++;
                }
                else if (status == "DRY_RUN")
                {
                    match.Status = AuditStatus.DryRun;
                    match.DestinationPath = dst;
                    match.ActionText = "Would move (Dry Run)";
                }
            }
        });
    }

    private void OnSummaryReceived(int total, int safe, int nsfw, int moved, int failed, bool dryRun)
    {
        RunOnUi(() =>
        {
            TotalImages = total;
            SafeCount = safe;
            NsfwCount = nsfw;
            MovedCount = moved;
            FailedCount = failed;
            ProgressPercent = 100.0;
        });
    }

    private void OnErrorReceived(string error)
    {
        RunOnUi(() =>
        {
            ShowInfoBar("Engine Error", error, InfoSeverity.Error);
        });
    }
}
