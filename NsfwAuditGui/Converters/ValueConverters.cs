using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using NsfwAuditGui.Models;

namespace NsfwAuditGui.Converters;

public class StatusToBrushConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is AuditStatus status)
        {
            return status switch
            {
                AuditStatus.Safe => new SolidColorBrush(Color.FromRgb(34, 197, 94)),    // #22c55e Emerald
                AuditStatus.Nsfw => new SolidColorBrush(Color.FromRgb(239, 68, 68)),    // #ef4444 Red
                AuditStatus.Moved => new SolidColorBrush(Color.FromRgb(168, 85, 247)),  // #a855f7 Purple
                AuditStatus.DryRun => new SolidColorBrush(Color.FromRgb(245, 158, 11)), // #f59e0b Amber
                AuditStatus.Error => new SolidColorBrush(Color.FromRgb(244, 63, 94)),   // #f43f5e Rose
                _ => new SolidColorBrush(Color.FromRgb(156, 163, 175))                  // #9ca3af Gray
            };
        }
        return new SolidColorBrush(Colors.Gray);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
}

public class StatusToBackgroundBrushConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is AuditStatus status)
        {
            return status switch
            {
                AuditStatus.Safe => new SolidColorBrush(Color.FromArgb(38, 34, 197, 94)),    // 15% opacity green
                AuditStatus.Nsfw => new SolidColorBrush(Color.FromArgb(38, 239, 68, 68)),    // 15% opacity red
                AuditStatus.Moved => new SolidColorBrush(Color.FromArgb(38, 168, 85, 247)),  // 15% opacity purple
                AuditStatus.DryRun => new SolidColorBrush(Color.FromArgb(38, 245, 158, 11)), // 15% opacity amber
                AuditStatus.Error => new SolidColorBrush(Color.FromArgb(38, 244, 63, 94)),   // 15% opacity rose
                _ => new SolidColorBrush(Color.FromArgb(25, 156, 163, 175))
            };
        }
        return new SolidColorBrush(Color.FromArgb(20, 128, 128, 128));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
}

public class BoolToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool b = value is bool flag && flag;
        if (parameter is string paramStr && paramStr.Equals("Inverse", StringComparison.OrdinalIgnoreCase))
        {
            b = !b;
        }
        return b ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
}

public class NullToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool hasValue = value != null;
        if (parameter is string paramStr && paramStr.Equals("Inverse", StringComparison.OrdinalIgnoreCase))
        {
            hasValue = !hasValue;
        }
        return hasValue ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
}
