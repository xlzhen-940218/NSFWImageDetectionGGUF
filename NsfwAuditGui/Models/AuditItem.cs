using System;
using System.ComponentModel;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace NsfwAuditGui.Models;

public enum AuditStatus
{
    Pending,
    Safe,
    Nsfw,
    Error,
    Moved,
    DryRun
}

public class AuditItem : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;
    protected void OnPropertyChanged(string prop) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(prop));

    public string FilePath { get; set; } = string.Empty;
    public string FileName => Path.GetFileName(FilePath);
    public string DirectoryName => Path.GetDirectoryName(FilePath) ?? string.Empty;
    public string Extension => Path.GetExtension(FilePath).ToUpperInvariant();

    private double _score;
    public double Score
    {
        get => _score;
        set
        {
            if (Math.Abs(_score - value) > 0.00001)
            {
                _score = value;
                OnPropertyChanged(nameof(Score));
                OnPropertyChanged(nameof(ScorePercentText));
            }
        }
    }

    public string ScorePercentText => $"{Score * 100:F2}%";

    private AuditStatus _status = AuditStatus.Pending;
    public AuditStatus Status
    {
        get => _status;
        set
        {
            if (_status != value)
            {
                _status = value;
                OnPropertyChanged(nameof(Status));
                OnPropertyChanged(nameof(StatusBadgeText));
                OnPropertyChanged(nameof(IsNsfw));
                OnPropertyChanged(nameof(IsSafe));
                OnPropertyChanged(nameof(IsError));
            }
        }
    }

    public bool IsNsfw => Status == AuditStatus.Nsfw || Status == AuditStatus.Moved || Status == AuditStatus.DryRun;
    public bool IsSafe => Status == AuditStatus.Safe;
    public bool IsError => Status == AuditStatus.Error;

    public string StatusBadgeText
    {
        get
        {
            bool zh = Services.LocalizationService.Instance.IsChinese;
            return Status switch
            {
                AuditStatus.Pending => zh ? "待处理" : "Pending",
                AuditStatus.Safe => zh ? "安全 (SAFE)" : "SAFE",
                AuditStatus.Nsfw => zh ? "违规 (NSFW)" : "NSFW",
                AuditStatus.Moved => zh ? "已隔离" : "MOVED",
                AuditStatus.DryRun => zh ? "试运行" : "DRY RUN",
                AuditStatus.Error => zh ? "错误" : "ERROR",
                _ => "UNKNOWN"
            };
        }
    }

    private string _actionText = string.Empty;
    public string ActionText
    {
        get => _actionText;
        set { _actionText = value; OnPropertyChanged(nameof(ActionText)); }
    }

    private string _destinationPath = string.Empty;
    public string DestinationPath
    {
        get => _destinationPath;
        set { _destinationPath = value; OnPropertyChanged(nameof(DestinationPath)); }
    }

    public string FileSizeText
    {
        get
        {
            try
            {
                if (File.Exists(FilePath))
                {
                    long bytes = new FileInfo(FilePath).Length;
                    if (bytes >= 1024 * 1024) return $"{bytes / (1024.0 * 1024.0):F2} MB";
                    if (bytes >= 1024) return $"{bytes / 1024.0:F1} KB";
                    return $"{bytes} B";
                }
            }
            catch { }
            return string.Empty;
        }
    }

    private BitmapImage? _thumbnail;
    public BitmapImage? Thumbnail
    {
        get => _thumbnail;
        set { _thumbnail = value; OnPropertyChanged(nameof(Thumbnail)); }
    }

    public async Task LoadThumbnailAsync()
    {
        try
        {
            if (!File.Exists(FilePath)) return;
            await Task.Run(() =>
            {
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(FilePath);
                bitmap.DecodePixelWidth = 180;
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();
                bitmap.Freeze();
                Thumbnail = bitmap;
            });
        }
        catch
        {
            // Ignore decode failures for corrupted images
        }
    }
}
