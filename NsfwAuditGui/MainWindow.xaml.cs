using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using Microsoft.Win32;
using NsfwAuditGui.Models;
using NsfwAuditGui.ViewModels;

namespace NsfwAuditGui;

public partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; } = new();

    [StructLayout(LayoutKind.Sequential)]
    private struct MARGINS
    {
        public int cxLeftWidth;
        public int cxRightWidth;
        public int cyTopHeight;
        public int cyBottomHeight;
    }

    [DllImport("dwmapi.dll")]
    private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);

    [DllImport("dwmapi.dll")]
    private static extern int DwmExtendFrameIntoClientArea(IntPtr hWnd, ref MARGINS pMarInset);

    private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;
    private const int DWMWA_MICA_EFFECT = 1029;
    private const int DWMWA_SYSTEMBACKDROP_TYPE = 38;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = ViewModel;

        ViewModel.L10n.PropertyChanged += (_, _) => UpdateTitle();
        UpdateTitle();
    }

    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);
        EnableMicaAndDarkMode();
    }

    private void EnableMicaAndDarkMode()
    {
        try
        {
            var helper = new WindowInteropHelper(this);
            IntPtr hwnd = helper.Handle;

            // 1. Enable immersive dark mode for title bar & window chrome
            int darkMode = 1;
            DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ref darkMode, sizeof(int));

            // 2. Set Windows 11 Mica backdrop (2 = Mica, 4 = MicaAlt, 3 = Acrylic)
            int backdropType = 2; // Mica
            int hr = DwmSetWindowAttribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, ref backdropType, sizeof(int));
            if (hr != 0)
            {
                // Fallback for Windows 11 21H2 builds
                int micaOn = 1;
                DwmSetWindowAttribute(hwnd, DWMWA_MICA_EFFECT, ref micaOn, sizeof(int));
            }

            // 3. Extend glass frame into client area so DWM renders Mica over entire window
            var margins = new MARGINS { cxLeftWidth = -1, cxRightWidth = -1, cyTopHeight = -1, cyBottomHeight = -1 };
            DwmExtendFrameIntoClientArea(hwnd, ref margins);

            // 4. Set WPF HwndSource transparent background so WPF does not paint opaque black or white
            var source = HwndSource.FromHwnd(hwnd);
            if (source?.CompositionTarget != null)
            {
                source.CompositionTarget.BackgroundColor = Colors.Transparent;
            }
        }
        catch
        {
            // Fallback for older OS versions
        }
    }

    private void UpdateTitle()
    {
        Title = $"{ViewModel.L10n.AppTitle} (GGUF ViT-B/16)";
    }

    private void OnBrowseTargetFolderClicked(object sender, RoutedEventArgs e)
    {
        try
        {
            var dialog = new OpenFolderDialog
            {
                Title = ViewModel.L10n.TargetFolderTitle,
                InitialDirectory = Directory.Exists(ViewModel.TargetDirectory) ? ViewModel.TargetDirectory : string.Empty
            };

            if (dialog.ShowDialog() == true)
            {
                ViewModel.TargetDirectory = dialog.FolderName;
            }
        }
        catch (Exception ex)
        {
            ViewModel.ShowInfoBar("Folder Picker Error", ex.Message, InfoSeverity.Error);
        }
    }

    private void OnBrowseNsfwFolderClicked(object sender, RoutedEventArgs e)
    {
        try
        {
            var dialog = new OpenFolderDialog
            {
                Title = ViewModel.L10n.NsfwFolderTitle,
                InitialDirectory = Directory.Exists(ViewModel.NsfwDirectory) ? ViewModel.NsfwDirectory : string.Empty
            };

            if (dialog.ShowDialog() == true)
            {
                ViewModel.NsfwDirectory = dialog.FolderName;
            }
        }
        catch (Exception ex)
        {
            ViewModel.ShowInfoBar("Folder Picker Error", ex.Message, InfoSeverity.Error);
        }
    }

    private void OnBrowseExeClicked(object sender, RoutedEventArgs e)
    {
        try
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select nsfw_audit.exe",
                Filter = "Executable files (*.exe)|*.exe|All files (*.*)|*.*",
                InitialDirectory = File.Exists(ViewModel.ExePath) ? Path.GetDirectoryName(ViewModel.ExePath) : string.Empty
            };

            if (dialog.ShowDialog() == true)
            {
                ViewModel.ExePath = dialog.FileName;
            }
        }
        catch (Exception ex)
        {
            ViewModel.ShowInfoBar("File Picker Error", ex.Message, InfoSeverity.Error);
        }
    }

    private void OnBrowseModelClicked(object sender, RoutedEventArgs e)
    {
        try
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select GGUF Model",
                Filter = "GGUF Model (*.gguf)|*.gguf|All files (*.*)|*.*",
                InitialDirectory = File.Exists(ViewModel.ModelPath) ? Path.GetDirectoryName(ViewModel.ModelPath) : string.Empty
            };

            if (dialog.ShowDialog() == true)
            {
                ViewModel.ModelPath = dialog.FileName;
            }
        }
        catch (Exception ex)
        {
            ViewModel.ShowInfoBar("File Picker Error", ex.Message, InfoSeverity.Error);
        }
    }

    private void OnOpenFileClicked(object sender, RoutedEventArgs e)
    {
        if (sender is Button btn && btn.Tag is string path && File.Exists(path))
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = path,
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                ViewModel.ShowInfoBar("Open File Error", ex.Message, InfoSeverity.Error);
            }
        }
    }

    private void OnShowInExplorerClicked(object sender, RoutedEventArgs e)
    {
        if (sender is Button btn && btn.Tag is string path && File.Exists(path))
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = "explorer.exe",
                    Arguments = $"/select,\"{path}\"",
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                ViewModel.ShowInfoBar("Explorer Error", ex.Message, InfoSeverity.Error);
            }
        }
    }

    private void OnCopyLogsClicked(object sender, RoutedEventArgs e)
    {
        try
        {
            string allLogs = string.Join(Environment.NewLine, ViewModel.EngineLogs);
            if (!string.IsNullOrWhiteSpace(allLogs))
            {
                Clipboard.SetText(allLogs);
                ViewModel.ShowInfoBar(ViewModel.L10n.TabLogs, ViewModel.L10n.MsgLogsCopied, InfoSeverity.Success);
            }
        }
        catch (Exception ex)
        {
            ViewModel.ShowInfoBar("Clipboard Error", ex.Message, InfoSeverity.Warning);
        }
    }

    private void OnWindowDragOver(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            e.Effects = DragDropEffects.Copy;
            e.Handled = true;
        }
    }

    private void OnWindowDrop(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            if (e.Data.GetData(DataFormats.FileDrop) is string[] files && files.Length > 0)
            {
                string first = files[0];
                if (Directory.Exists(first))
                {
                    ViewModel.TargetDirectory = first;
                    ViewModel.ShowInfoBar(ViewModel.L10n.AppTitle, $"{ViewModel.L10n.MsgFolderPicked}: {first}", InfoSeverity.Informational);
                }
                else if (File.Exists(first))
                {
                    string dir = Path.GetDirectoryName(first) ?? string.Empty;
                    if (Directory.Exists(dir))
                    {
                        ViewModel.TargetDirectory = dir;
                        ViewModel.ShowInfoBar(ViewModel.L10n.AppTitle, $"{ViewModel.L10n.MsgFolderPicked}: {dir}", InfoSeverity.Informational);
                    }
                }
            }
        }
    }

    private void OnCloseInfoBarClicked(object sender, RoutedEventArgs e)
    {
        ViewModel.InfoBarIsOpen = false;
    }
}
