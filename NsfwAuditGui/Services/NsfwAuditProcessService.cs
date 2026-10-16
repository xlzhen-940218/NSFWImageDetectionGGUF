using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using NsfwAuditGui.Models;

namespace NsfwAuditGui.Services;

public class NsfwAuditProcessService
{
    private Process? _currentProcess;
    private readonly object _lock = new();

    public event Action<string>? LogReceived;
    public event Action<int, double, string>? InitReceived; // total, threshold, device
    public event Action<string, double, double, string, string>? ItemReceived; // status, score, percent, file, path
    public event Action<int, int, double>? ProgressReceived; // current, total, percent
    public event Action<string, string, string>? ActionReceived; // status, src, dst
    public event Action<int, int, int, int, int, bool>? SummaryReceived; // total, safe, nsfw, moved, failed, dryRun
    public event Action<string>? ErrorReceived;

    public bool IsRunning
    {
        get
        {
            lock (_lock)
            {
                return _currentProcess is { HasExited: false };
            }
        }
    }

    public static string FindDefaultExePath()
    {
        string baseDir = AppContext.BaseDirectory;
        string[] candidates =
        [
            Path.Combine(baseDir, "nsfw_audit.exe"),
            Path.Combine(baseDir, "_nsfw_tool", "build", "bin", "nsfw_audit.exe"),
            Path.Combine(baseDir, "..", "_nsfw_tool", "build", "bin", "nsfw_audit.exe"),
            Path.Combine(baseDir, "..", "..", "..", "..", "_nsfw_tool", "build", "bin", "nsfw_audit.exe"),
            Path.Combine(baseDir, "..", "..", "..", "..", "..", "_nsfw_tool", "build", "bin", "nsfw_audit.exe"),
            Path.Combine(Environment.CurrentDirectory, "nsfw_audit.exe"),
            Path.Combine(Environment.CurrentDirectory, "_nsfw_tool", "build", "bin", "nsfw_audit.exe"),
            Path.Combine(Environment.CurrentDirectory, "..", "_nsfw_tool", "build", "bin", "nsfw_audit.exe")
        ];

        foreach (var path in candidates)
        {
            try
            {
                string full = Path.GetFullPath(path);
                if (File.Exists(full)) return full;
            }
            catch { }
        }

        return candidates[2];
    }

    public static string FindDefaultModelPath()
    {
        string baseDir = AppContext.BaseDirectory;
        string[] candidates =
        [
            Path.Combine(baseDir, "models", "nsfw_image_detection.fp32.gguf"),
            Path.Combine(baseDir, "_nsfw_tool", "models", "nsfw_image_detection.fp32.gguf"),
            Path.Combine(baseDir, "..", "_nsfw_tool", "models", "nsfw_image_detection.fp32.gguf"),
            Path.Combine(baseDir, "..", "..", "..", "..", "_nsfw_tool", "models", "nsfw_image_detection.fp32.gguf"),
            Path.Combine(baseDir, "..", "..", "..", "..", "..", "_nsfw_tool", "models", "nsfw_image_detection.fp32.gguf"),
            Path.Combine(Environment.CurrentDirectory, "models", "nsfw_image_detection.fp32.gguf"),
            Path.Combine(Environment.CurrentDirectory, "_nsfw_tool", "models", "nsfw_image_detection.fp32.gguf"),
            Path.Combine(Environment.CurrentDirectory, "..", "_nsfw_tool", "models", "nsfw_image_detection.fp32.gguf")
        ];

        foreach (var path in candidates)
        {
            try
            {
                string full = Path.GetFullPath(path);
                if (File.Exists(full)) return full;
            }
            catch { }
        }

        return candidates[2];
    }

    public static string FindDefaultImageDir()
    {
        string baseDir = AppContext.BaseDirectory;
        string[] candidates =
        [
            Path.Combine(baseDir, "image-test"),
            Path.Combine(baseDir, "..", "image-test"),
            Path.Combine(baseDir, "..", "..", "..", "..", "image-test"),
            Path.Combine(baseDir, "..", "..", "..", "..", "..", "image-test"),
            Path.Combine(Environment.CurrentDirectory, "image-test"),
            Path.Combine(Environment.CurrentDirectory, "..", "image-test")
        ];

        foreach (var path in candidates)
        {
            try
            {
                string full = Path.GetFullPath(path);
                if (Directory.Exists(full)) return full;
            }
            catch { }
        }

        return Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
    }

    public async Task<int> RunAuditAsync(AuditConfig config, CancellationToken cancellationToken)
    {
        string exe = string.IsNullOrWhiteSpace(config.ExePath) ? FindDefaultExePath() : config.ExePath;
        if (!File.Exists(exe))
        {
            throw new FileNotFoundException($"Cannot find nsfw_audit.exe at: {exe}");
        }

        var sb = new StringBuilder();
        sb.Append($"--dir \"{config.TargetDirectory}\" ");
        sb.Append($"--threshold {config.Threshold:F2} ");
        sb.Append($"--batch {config.BatchSize} ");
        sb.Append($"--device {config.Device} ");
        sb.Append("--json ");

        if (!string.IsNullOrWhiteSpace(config.ModelPath) && File.Exists(config.ModelPath))
        {
            sb.Append($"--model \"{config.ModelPath}\" ");
        }

        if (!string.IsNullOrWhiteSpace(config.NsfwDirectory))
        {
            sb.Append($"--nsfw-dir \"{config.NsfwDirectory}\" ");
        }

        if (config.Recursive)
        {
            sb.Append("--recursive ");
        }

        if (config.DryRun)
        {
            sb.Append("--dry-run ");
        }

        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = sb.ToString().Trim(),
            CreateNoWindow = true,
            UseShellExecute = false,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };

        var process = new Process { StartInfo = psi, EnableRaisingEvents = true };

        lock (_lock)
        {
            _currentProcess = process;
        }

        var tcs = new TaskCompletionSource<int>();

        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            string line = e.Data.Trim();
            if (string.IsNullOrEmpty(line)) return;

            LogReceived?.Invoke(line);
            ProcessJsonLine(line);
        };

        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            string line = e.Data.Trim();
            if (string.IsNullOrEmpty(line)) return;

            LogReceived?.Invoke($"[STDERR] {line}");
            if (line.StartsWith("[ERROR]"))
            {
                ErrorReceived?.Invoke(line);
            }
        };

        process.Exited += (_, _) =>
        {
            tcs.TrySetResult(process.ExitCode);
        };

        try
        {
            LogReceived?.Invoke($"[HOST] Launching: {psi.FileName} {psi.Arguments}");
            if (!process.Start())
            {
                throw new InvalidOperationException("Failed to launch nsfw_audit.exe process.");
            }

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            using (cancellationToken.Register(() =>
            {
                try
                {
                    LogReceived?.Invoke("[HOST] Audit cancellation requested by user. Terminating engine process...");
                    process.Kill(entireProcessTree: true);
                }
                catch { }
                tcs.TrySetCanceled();
            }))
            {
                return await tcs.Task;
            }
        }
        finally
        {
            lock (_lock)
            {
                _currentProcess = null;
            }
            process.Dispose();
        }
    }

    public void Cancel()
    {
        lock (_lock)
        {
            if (_currentProcess is { HasExited: false })
            {
                try
                {
                    _currentProcess.Kill(entireProcessTree: true);
                }
                catch { }
            }
        }
    }

    private void ProcessJsonLine(string line)
    {
        if (!line.StartsWith('{') || !line.EndsWith('}'))
        {
            // Non-JSON line (e.g. Vulkan device info or backend log)
            return;
        }

        try
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            if (!root.TryGetProperty("type", out var typeProp)) return;

            string type = typeProp.GetString() ?? string.Empty;
            switch (type)
            {
                case "init":
                {
                    int total = root.TryGetProperty("total", out var t) ? t.GetInt32() : 0;
                    double th = root.TryGetProperty("threshold", out var thP) ? thP.GetDouble() : 0.9;
                    string dev = root.TryGetProperty("device", out var d) ? d.GetString() ?? "auto" : "auto";
                    InitReceived?.Invoke(total, th, dev);
                    break;
                }
                case "item":
                {
                    string status = root.TryGetProperty("status", out var s) ? s.GetString() ?? "" : "";
                    double score = root.TryGetProperty("score", out var sc) ? sc.GetDouble() : 0.0;
                    double pct = root.TryGetProperty("percent", out var pc) ? pc.GetDouble() : 0.0;
                    string file = root.TryGetProperty("file", out var f) ? f.GetString() ?? "" : "";
                    string path = root.TryGetProperty("path", out var p) ? p.GetString() ?? "" : "";
                    ItemReceived?.Invoke(status, score, pct, file, path);
                    break;
                }
                case "progress":
                {
                    int cur = root.TryGetProperty("current", out var c) ? c.GetInt32() : 0;
                    int tot = root.TryGetProperty("total", out var tt) ? tt.GetInt32() : 0;
                    double pct = root.TryGetProperty("percent", out var pt) ? pt.GetDouble() : 0.0;
                    ProgressReceived?.Invoke(cur, tot, pct);
                    break;
                }
                case "action":
                {
                    string actStatus = root.TryGetProperty("status", out var asProp) ? asProp.GetString() ?? "" : "";
                    string src = root.TryGetProperty("src", out var sProp) ? sProp.GetString() ?? "" : "";
                    string dst = root.TryGetProperty("dst", out var dProp) ? dProp.GetString() ?? "" : "";
                    ActionReceived?.Invoke(actStatus, src, dst);
                    break;
                }
                case "summary":
                {
                    int tot = root.TryGetProperty("total", out var tt) ? tt.GetInt32() : 0;
                    int safe = root.TryGetProperty("safe", out var sf) ? sf.GetInt32() : 0;
                    int nsfw = root.TryGetProperty("nsfw", out var nf) ? nf.GetInt32() : 0;
                    int moved = root.TryGetProperty("moved", out var mv) ? mv.GetInt32() : 0;
                    int fail = root.TryGetProperty("failed", out var fl) ? fl.GetInt32() : 0;
                    bool dry = root.TryGetProperty("dry_run", out var dr) && dr.GetBoolean();
                    SummaryReceived?.Invoke(tot, safe, nsfw, moved, fail, dry);
                    break;
                }
                case "error":
                {
                    string msg = root.TryGetProperty("message", out var m) ? m.GetString() ?? "" : "";
                    ErrorReceived?.Invoke(msg);
                    break;
                }
            }
        }
        catch
        {
            // Not JSON or parse error, ignored
        }
    }
}
