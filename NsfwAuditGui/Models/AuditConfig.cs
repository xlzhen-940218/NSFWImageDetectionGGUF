namespace NsfwAuditGui.Models;

public class AuditConfig
{
    public string TargetDirectory { get; set; } = string.Empty;
    public string NsfwDirectory { get; set; } = string.Empty;
    public double Threshold { get; set; } = 0.90;
    public int BatchSize { get; set; } = 8;
    public string Device { get; set; } = "auto";
    public bool Recursive { get; set; } = false;
    public bool DryRun { get; set; } = false;
    public string ExePath { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
}
