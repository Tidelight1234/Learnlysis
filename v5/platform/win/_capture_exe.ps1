Add-Type -AssemblyName System.Drawing
Add-Type -ReferencedAssemblies System.Drawing, System.Collections -TypeDefinition @"
using System;using System.Text;using System.Runtime.InteropServices;using System.Collections.Generic;
public class Cap2 {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc e, IntPtr l);
  [DllImport("user32.dll")] public static extern int  GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern int  GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  public struct RECT { public int Left, Top, Right, Bottom; }

  public static List<IntPtr> TopLevel(uint target) {
    var res = new List<IntPtr>();
    EnumWindows((h, l) => { uint pid; GetWindowThreadProcessId(h, out pid);
      if (pid == target) res.Add(h); return true; }, IntPtr.Zero);
    return res;
  }

  public static string Info(IntPtr h) {
    var t = new StringBuilder(512); GetWindowText(h, t, 512);
    var c = new StringBuilder(256); GetClassName(h, c, 256);
    RECT r; GetWindowRect(h, out r);
    return string.Format("hwnd={0} cls={1} visible={2} rect={3}x{4}+{5}+{6} title='{7}'",
      h, c, IsWindowVisible(h), (r.Right - r.Left), (r.Bottom - r.Top), r.Left, r.Top, t);
  }

  public static bool Grab(IntPtr h, string path, int w, int ht) {
    var bmp = new System.Drawing.Bitmap(w, ht);
    var g = System.Drawing.Graphics.FromImage(bmp);
    var hdc = g.GetHdc();
    bool ok = PrintWindow(h, hdc, 2);   // PW_RENDERFULLCONTENT
    g.ReleaseHdc(hdc); g.Dispose();
    if (ok) bmp.Save(path, System.Drawing.Imaging.ImageFormat.Png);
    bmp.Dispose();
    return ok;
  }
}
"@

$proc = Get-Process -Name Learnlysis -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $proc) { Write-Host "NOT RUNNING"; exit 1 }
$pid32 = [uint32]$proc.Id
Write-Host ("PID {0} top-level windows:" -f $pid32)
$wins = [Cap2]::TopLevel($pid32)
$best = $null; $bestArea = 0
foreach ($h in $wins) {
  $info = [Cap2]::Info($h)
  Write-Host ("  " + $info)
  if ($info -match "visible=True" -and $info -match "rect=(\d+)x(\d+)") {
    $area = [int]$Matches[1] * [int]$Matches[2]
    if ($area -gt $bestArea) { $bestArea = $area; $best = $h; $bw = [int]$Matches[1]; $bh = [int]$Matches[2] }
  }
}
if (-not $best) { Write-Host "no visible window found"; exit 1 }
# 截图落在脚本自己的目录（version\v5\platform\win\_exe_shot.png）：它是 gitignore 的 `_exe_shot.png`，
# 不写死仓库绝对路径，仓库换地方也不会拍不到。
$out = Join-Path $PSScriptRoot "_exe_shot.png"
$ok = [Cap2]::Grab($best, $out, $bw, $bh)
Write-Host ("PrintWindow ok={0}  saved={1}  ({2:N0} KB)" -f $ok, $out, ((Get-Item $out).Length / 1KB))
