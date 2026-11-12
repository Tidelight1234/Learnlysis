# ============================================================================
# ENCODING CONTRACT (keep this block ASCII-only):
#   This file MUST be saved as UTF-8 WITH BOM. Windows PowerShell 5.1 reads a
#   BOM-less .ps1 as the ANSI code page (cp936 here); the Chinese comments then
#   decode to mojibake that unbalances quotes and kills the script with a
#   ParserError before anything runs. If an editor strips the BOM, restore it:
#     $f='build_exe.ps1'; $t=[IO.File]::ReadAllText($f,[Text.Encoding]::UTF8);
#     [IO.File]::WriteAllText($f,$t,(New-Object Text.UTF8Encoding($true)))
#   Verify: [Management.Automation.Language.Parser]::ParseFile($f,[ref]$null,[ref]$e)
# ============================================================================
<#
.SYNOPSIS
    Learnlysis 本地 PyInstaller 打包脚本。

.DESCRIPTION
    与 .github/workflows/build-windows.yml 的 "Build with PyInstaller" 步骤逐字对齐。
    改任何一处参数，两边都要改 —— 否则会出现「本地好好的、线上不一样」。

    默认打的是**加固过的包**：先把 version\v5\src 复制成一份 stage 副本，在副本里
    剥 docstring、写构建水印、把算法族与加固模块 Cython 编成 .pyd，再拿这份副本
    跑 PyInstaller（详见 version\v5\src\_protect_stage.py）。源码树一个字节不动。
    -NoHarden 只给排障用（看「不加固是不是就能跑」），产物会带醒目警告。

    刻意在 .build-venv 里打包，不用 .conda：
    .conda 装着 torch（506 MB），而 analysis_dl.py 在模块顶层写了 `import torch`。
    try/except ImportError 是运行时概念，PyInstaller 的静态分析只看有没有 import 语句，
    所以在 .conda 里打包会把 torch_cpu.dll / c10.dll / fbgemm.dll 连带 sympy 一起拖进 exe。
    --exclude-module=torch 能挡住，但洁净环境更省心，也让本地产物和 CI 产物可以直接比体积。

.PARAMETER Layout
    onedir（默认，启动快、易排错）或 onefile（单文件，冷启动 5-15 秒）。

.PARAMETER Console
    加上就带控制台黑窗。--windowed 的 exe 启动崩溃是没有任何输出的，
    排查启动问题时用 -Console 打一次，异常栈会直接打在黑窗里。

.PARAMETER Setup
    先（重）建 .build-venv 再打包。会自动挑一个「非 conda」的解释器，理由见下。

.PARAMETER NoHarden
    跳过构建加固（不剥 docstring、不写水印、不编 .pyd），直接在源码树上打包。
    只用于排障：这种包里的 docstring 会被 pyinstxtractor + pycdc 原样翻出来。

.PARAMETER SkipStageTests
    跳过 stage 里那一趟 pytest（加固后产物的就地回归）。只在赶时间时用，
    代价是「编出来的 .pyd 行为是否仍与源码一致」这件事没人验过。

.NOTES
    为什么不能用 conda 的 python 建构建 venv：
    venv 不复制 stdlib，它直接用基准解释器的那一份。用 .conda\python.exe 建出来的
    venv，stdlib 和扩展模块仍然落在 .conda\Lib 与 .conda\DLLs，于是：
      1. PyInstaller 会去收 .conda\DLLs\pyexpat.pyd，而它动态依赖 libexpat.dll；
         那个 dll 在 .conda\Library\bin，只有 conda activate 才会把它加进 PATH，
         构建时没激活 → "Library not found: could not resolve 'libexpat.dll'"。
         pyexpat 是 xml.etree 的后端，openpyxl 读写 xlsx 全靠它 —— 这不是无害警告。
      2. GitHub actions/setup-python 给的是 python.org 独立构建，它的 pyexpat.pyd
         把 expat 静态链接进去了，根本没有这个依赖。
    所以「用 conda python 建 venv」会让本地产物和 CI 产物在二进制层面就不是一回事。
    本机可用的独立解释器：C:\Users\dell\AppData\Local\Programs\Python\Python311\python.exe

.EXAMPLE
    cd version\v5\platform\win
    .\build_exe.ps1
    .\build_exe.ps1 -Layout onefile
    .\build_exe.ps1 -Console
    .\build_exe.ps1 -Setup
    .\build_exe.ps1 -NoHarden          # 排障用，这种产物不该发出去
    .\build_exe.ps1 -RepoRoot D:\ci\inputs   # version\ 被单独拷走、上面没有仓库根时
#>
[CmdletBinding()]
param(
    [ValidateSet("onedir", "onefile")]
    [string] $Layout = "onedir",

    [switch] $Console,

    [switch] $Setup,

    [switch] $NoHarden,

    [switch] $SkipStageTests,

    # 仓库根所在处：.build-venv 建在它下面，依赖清单与法律材料也从它那里取。
    # 留空 = 从本脚本向上找第一个真有 requirements.txt
    # 的目录。只有把 version\ 单独拷到别的机器上、上面没有仓库根时才需要显式指。
    [string] $RepoRoot
)

$ErrorActionPreference = "Stop"
# 方案 B：本脚本住在 version\v5\platform\win，打包对象是同一棵树里的 src\，
# 而 _stage / dist / build 这些产物留在版本树根 version\v5（现场层，与 data\ 同级）。
# 仓库级文件仍在仓库根：.build-venv / _piptmp / requirements*.txt / LICENSE / NOTICE / LGPL 文本。
# $RepoRoot 默认从脚本所在目录向上找第一个真有 requirements.txt 的目录 —— 不把层数算死：
# 同一份脚本在 windows-version 里数三级到仓库根，在 mac 仓库的 version\ 下数四级，
# 写死一个数就必然在另一个仓库指错。单独把 version\ 文件夹搬到别的机器上时上面什么都没有，
# 那时用 -RepoRoot 显式指过去。
$PlatformDir = $PSScriptRoot
$TreeRoot    = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SrcDir      = Join-Path $TreeRoot "src"
if ($RepoRoot) {
    # 给了 -RepoRoot 就不猜：猜错会把另一棵树的输入当本树的，装出版本不对的
    # .build-venv，而且要等到 pip 装完才露出来 —— 这里当场验一样最明显的。
    if (-not (Test-Path (Join-Path $RepoRoot "requirements.txt"))) {
        throw "-RepoRoot $RepoRoot 下没有 requirements.txt —— 它必须是放了五份打包输入的那个目录"
    }
} else {
    $RepoRoot = $PlatformDir
    while ($RepoRoot -and -not (Test-Path (Join-Path $RepoRoot "requirements.txt"))) {
        $parent = Split-Path $RepoRoot -Parent
        if ($parent -eq $RepoRoot) { break }
        $RepoRoot = $parent
    }
}
if (-not (Test-Path (Join-Path $RepoRoot "requirements.txt"))) {
    throw "仓库根 $RepoRoot 里没有 requirements.txt —— 从 $PlatformDir 往上没找到；把 version\ 单独拷出来跑时用 -RepoRoot 指定输入目录"
}
if (-not (Test-Path (Join-Path $SrcDir "main.py"))) {
    throw "src\main.py 不在 $SrcDir —— 这棵树还没做方案 B 拆分，别去改本脚本的 --add-data 硬凑"
}

# 打包输入或许可材料缺失就不要往下跑：requirements* 决定 .build-venv 里装什么，
# LICENSE / NOTICE / LGPL 文本与第三方许可目录要随包分发。
$RequiredRootFiles = @("requirements.txt", "requirements-build.txt", "requirements-ai.txt",
                       "LICENSE", "NOTICE", "LGPL-3.0.txt", "LGPL-2.1.txt",
                       "PYTHON-LICENSE.txt", "THIRD_PARTY_LICENSES")
$missingRoot = @($RequiredRootFiles | Where-Object { -not (Test-Path (Join-Path $RepoRoot $_)) })
if ($missingRoot.Count) {
    throw @"
仓库根 $RepoRoot 缺打包输入：$($missingRoot -join ', ')
补齐后重跑。别为了绕开它去删 --add-data / -r 参数：法律文本不进包，
关于页让用户「保留安装包目录下的 LICENSE 与 NOTICE」就成了假话。
"@
}
$VenvDir  = Join-Path $RepoRoot ".build-venv"
$VenvPy   = Join-Path $VenvDir "Scripts\python.exe"
# $BuildDir 在加固那一步之后被指向 stage 副本；-NoHarden 时就是源码树。
$BuildDir = $SrcDir
$StageDir = Join-Path $TreeRoot "_stage"
# 被 Cython 编掉的那八个模块：gate 与 --hidden-import 两边共用同一份名单。
$HardenedModules = @("analysis_stats_sequence", "analysis_stats_group",
                     "analysis_stats_network", "analysis_features",
                     "interaction_model", "build_stamp", "secure_store", "integrity")

# ---------------------------------------------------------------- 建构建环境
if ($Setup -or -not (Test-Path $VenvPy)) {
    # 优先找非 conda 的 3.11：py 启动器登记的、或 python.org 默认安装位置。
    $candidates = @()
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $candidates += (& py -0p 2>$null | ForEach-Object {
            if ($_ -match "^\s*-V:(3\.\d+)\S*\s+(.+?python\.exe)\s*$") { $Matches[2] }
        })
    }
    $candidates += "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    $candidates += "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"

    $basePy = $candidates | Where-Object { $_ -and (Test-Path $_) } | Where-Object {
        -not (& $_ -c "import os,sys;print(os.path.exists(os.path.join(sys.base_prefix,'conda-meta')))" 2>$null).Trim().EndsWith("True")
    } | Select-Object -First 1

    if (-not $basePy) {
        throw @"
没找到「非 conda」的 Python 3.11+。别用 .conda\python.exe 建构建 venv（见脚本 .NOTES）。
装一个 python.org 的独立解释器，或把它的路径写进本脚本的 `$candidates。
"@
    }

    Write-Host "== setup build venv ==" -ForegroundColor Cyan
    Write-Host "  base python : $basePy"
    & $basePy -m venv --clear $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

    # TEMP 挪到仓库本地：pip 把 wheel 先解到 %TEMP%\pip-unpack-*，
    # C 盘 Temp 被杀软或清理任务插手时会报 [Errno 2] No such file or directory。
    $pipTmp = Join-Path $RepoRoot "_piptmp"
    New-Item -ItemType Directory -Force $pipTmp | Out-Null
    $env:TEMP = $pipTmp
    $env:TMP  = $pipTmp

    # 网络抖就重试：每轮都有更多 wheel 落进 pip 的 http 缓存，会逐步收敛。
    $installed = $false
    for ($attempt = 1; $attempt -le 5 -and -not $installed; $attempt++) {
        Write-Host "  pip install (attempt $attempt) ..."
        & $VenvPy -m pip install --upgrade pip --timeout 900 --retries 15 2>&1 | Out-Null
        & $VenvPy -m pip install --timeout 900 --retries 15 `
            -r (Join-Path $RepoRoot "requirements.txt") `
            -r (Join-Path $RepoRoot "requirements-build.txt") `
            -r (Join-Path $RepoRoot "requirements-ai.txt")
        if ($LASTEXITCODE -eq 0) { $installed = $true } else { Start-Sleep -Seconds 10 }
    }
    if (-not $installed) { throw "pip install failed after 5 attempts" }
    Write-Host "  venv ready" -ForegroundColor Green
}

if (-not (Test-Path $VenvPy)) {
    throw "找不到 .build-venv。先跑一次：.\build_exe.ps1 -Setup"
}

Set-Location $SrcDir
Write-Host "`n== Learnlysis local build ==" -ForegroundColor Cyan
Write-Host ("  venv   : {0}" -f $VenvPy)
Write-Host ("  cwd    : {0}" -f (Get-Location).Path)
Write-Host ("  layout : {0}" -f $Layout)
Write-Host ("  console: {0}" -f [bool]$Console)

# 清掉上一次的残留。--clean 只清 build 缓存，不清 dist；
# dist 里剩下的文件被杀软扫住时，COLLECT 阶段会报 PermissionError。
foreach ($stale in @("dist", "build")) {
    $p = Join-Path $TreeRoot $stale
    if (Test-Path $p) {
        Write-Host "  clearing stale $stale/"
        Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path $p) { throw "删不掉 $p —— 是不是有 Learnlysis.exe 还在跑，或者杀软正占着？" }
    }
}
Remove-Item (Join-Path $SrcDir "Learnlysis.spec") -Force -ErrorAction SilentlyContinue

# 打包前的 30 秒闸门：模块导不全就别浪费几分钟去跑 PyInstaller。
# 钉的是 analysis_methods.all_methods() 的动态派发结果 —— IND12–IND15 / GRP10
# 全部来自 importlib 加载的 analysis_ml / analysis_dl，少一个就说明下面的
# --hidden-import 被改掉了，而 all_methods() 对 ImportError 是 continue，不会报错。
Write-Host "`n-- import smoke gate --" -ForegroundColor Cyan
$env:QT_QPA_PLATFORM = "offscreen"
& $VenvPy -X utf8 -c @"
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
import theme, icon_provider, config, interaction_model, timeline_axis
import analysis_core, analysis_methods, analysis_params, analysis_features
import analysis_plots, analysis_export, analysis_r_bridge
import analysis_stats_sequence, analysis_stats_group, analysis_stats_network
import analysis_ml, analysis_dl, analysis_workbench
import individual_analysis, group_analysis
import video_player, data_encoder, dimension_panel, group_encoder, summary_tab
import about_dialog, statistic_visualization, statistic_visualization_data
import statistic_visualization_methods, statistic_visualization_stats
import statistic_visualization_plots, statistic_visualization_export, main_window
# AI 辅助编码一族（与 CI 闸门逐字同步）：user_state/coding_assist 被主链顶层 import
# 已经间接验到，但 auto_coding / auto_coding_train 只是 data_encoder 里的函数级导入，
# 不显式拉一遍，依赖断裂要等到用户在 exe 里点开弹窗才暴露。
import user_state, video_probe, coding_assist
import ai_client, ai_settings, ai_worker, ai_context, ai_assistant
import framework_schema, framework_library, framework_quality, framework_prompt
import framework_store, framework_export, framework_designer
import auto_coding, auto_coding_train
# 加固/校验三件套：它们都被 Cython 编进 .pyd，在源码树里先验一遍导入链，
# 免得「只能在源码里跑得通」的接线问题到产物里才发现。
import build_stamp, secure_store, integrity
theme.load_fonts(); theme.load_mode(); theme.apply_theme(app)
ids = {s.id for s in analysis_methods.all_methods()}
missing = {'IND12', 'IND13', 'IND14', 'IND15', 'GRP10'} - ids
assert not missing, 'dynamic-dispatch methods missing: %s' % sorted(missing)
assert len(ids) >= 25, 'method registry shrank to %d' % len(ids)
print('SMOKE_IMPORT_OK, methods =', len(ids))
"@
if ($LASTEXITCODE -ne 0) { throw "import smoke gate failed (exit $LASTEXITCODE)" }

# ---------------------------------------------------------------- 构建加固 staging
# 把 version\v5 复制成一份「可以发给别人」的副本，在副本里：
#   1. AST 定点剥 docstring（源码树不动；本项目的 docstring 写满了公式口径与设计理由）；
#   2. 写构建水印（build_id / build_time / git_commit / hardening）；
#   3. 把 5 个算法族+契约与 3 个加固/校验模块 Cython 编成 .pyd，并删同名 .py。
# 后面的 PyInstaller 就跑在这份副本上，$BuildDir 随之改指。
# 没有 MSVC 时脚本会降级成「只剥文档串 + 写水印」并把原因写进 manifest，
# 但 --require-compiled 会把这种降级当失败：发布包缺 .pyd 就是防护档没落地。
if ($NoHarden) {
    Write-Host "`n-- hardening SKIPPED (-NoHarden) --" -ForegroundColor Yellow
    Write-Host "   这种产物里 docstring 完整保留，pyinstxtractor + pycdc 能还原成接近原样的源码。"
    Write-Host "   仅用于排障，不要把它发出去。"
} else {
    Write-Host "`n-- hardening build stage --" -ForegroundColor Cyan
    & $VenvPy -c "import Cython" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw @"
.build-venv 里没有 Cython，加固跑不了。两种选择：
  1. 装上：& '$VenvPy' -m pip install -r '$(Join-Path $RepoRoot 'requirements-build.txt')'
  2. 真的要出不加固的排障包：重跑一次带 -NoHarden（这种包不该发出去）
"@
    }
    $stageArgs = @("_protect_stage.py", "--out", $StageDir, "--require-compiled")
    if (-not $SkipStageTests) { $stageArgs += "--verify" }
    Push-Location $SrcDir
    try { & $VenvPy -X utf8 @stageArgs } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "harden stage exited with $LASTEXITCODE" }
    if (-not (Test-Path (Join-Path $StageDir "_ly_manifest.json"))) {
        throw "stage 没有产出 _ly_manifest.json，加固证据丢失"
    }
    $BuildDir = $StageDir
}
$Hardened = -not $NoHarden

# ---------------------------------------------------------------- 打包参数
# 与 workflow 的 $piArgs 逐字一致，只有 layout / console 由本脚本的参数决定。
# 产物落点定成绝对路径：本地可能在源码树、也可能在 stage 里跑，而两边都应落到
# version\v5\dist（CI 在 stage 里用 ../dist，指向同一个位置）。
$distOut = Join-Path $TreeRoot "dist"
$workOut = Join-Path $TreeRoot "build"

# PyInstaller 的相对路径都按当前目录解：进 $BuildDir，--add-data 与图标才从同一份副本里取。
Set-Location $BuildDir
Write-Host ("  build  : {0}" -f (Get-Location).Path)
# --add-data / --icon 必须给绝对路径：写了 --specpath/--workpath 之后，PyInstaller
# 把「相对源路径」按 spec 所在目录（build/）解，而不是按 cwd——症状是
# `Unable to find '...\build\assets\fonts\MiSans-Regular.ttf'`（本地与 CI 同命）。
# 目标侧（分号右半边）仍写相对路径，那是包内的布局。
$Here = (Get-Location).Path

$piArgs = @(
    "--$Layout",
    "--clean",              # 清掉上一次的 build 缓存
    "--noconfirm",
    "--noupx",              # UPX 压 Qt DLL 经常压出启动崩溃，本地/CI 一律不用
    "--name=Learnlysis",
    "--log-level=WARN",
    "--distpath=$distOut",
    "--workpath=$workOut",
    "--specpath=$workOut",

    # ---- 必须显式声明的隐藏导入 ----
    # analysis_methods.all_methods() 用 importlib.import_module(name) 遍历
    # _FAMILY_MODULES 做动态派发，PyInstaller 的静态分析看不见变量里的模块名。
    # 漏了不会报错，只是 IND12–IND15 / GRP10 在 exe 的方法下拉里静默消失。
    "--hidden-import=analysis_stats_sequence",
    "--hidden-import=analysis_stats_group",
    "--hidden-import=analysis_stats_network",
    # 下面五个在加固模式下**没有 .py**（只剩 .pyd），静态分析看不见任何 import 语句；
    # 不显式列全就是「exe 里根本没有这个模块」，而它缺席时应用只记一条日志，不报错。
    # 与 CI 的 $piArgs 逐字同步。名单与脚本开头的 $HardenedModules 是同一份。
    "--hidden-import=analysis_features",
    "--hidden-import=interaction_model",
    "--hidden-import=build_stamp",
    "--hidden-import=secure_store",
    "--hidden-import=integrity",
    "--hidden-import=analysis_ml",
    "--hidden-import=analysis_dl",
    "--collect-submodules=hmmlearn",   # IND08 的 Cython 扩展

    # ---- 资源 ----
    # 只装 theme._FONT_FILES 里真正用到的三个字重（约 24 MB）。
    # assets/fonts 下另外 7 个字重共约 57 MB，运行时读不到，不进包。
    # 改了 _FONT_FILES 就要同步改这里，否则 exe 会回退到微软雅黑。
    "--add-data=$Here\assets\fonts\MiSans-Regular.ttf;assets/fonts",
    "--add-data=$Here\assets\fonts\MiSans-Medium.ttf;assets/fonts",
    "--add-data=$Here\assets\fonts\MiSans-Semibold.ttf;assets/fonts",
    "--add-data=$PlatformDir\logo.ico;.",
    # 不需要 --add-data data：config.get_app_dir() 在 frozen 下取 exe 同级目录，
    # data/ 由 config.py 的 os.makedirs 现场创建，打进包里反而落到只读的 _MEIPASS。

    # ---- hook 噪声抑制（与 CI 逐字同步）----
    # scipy 1.15.3 删掉了 scipy.special._cdflib（功能并入 _ufuncs 的 C 扩展），而
    # PyInstaller 6.22.2 自带的 hook-scipy.special._ufuncs.py 仍把它写死进 hiddenimports，
    # 于是每次构建都刷一条 `WARNING: Hidden import "scipy.special._cdflib" not found!`。
    # 无害（项目代码 0 处引用，chi2_contingency / ppf 实测正常）但误导人；
    # 显式 exclude 后最小构建验证警告归零。上游 hook 适配 scipy 1.15+ 后可删这段。
    "--exclude-module=scipy.special._cdflib",

    # ---- 体积闸门：可选后端一律不进包 ----
    "--exclude-module=torch",
    "--exclude-module=xgboost",
    "--exclude-module=shap",
    "--exclude-module=sympy",          # torch 的依赖，本项目不用
    "--exclude-module=mpmath",         # sympy 的依赖
    "--exclude-module=tkinter",        # 绘图走 QtAgg，不需要 Tk 后端
    "--exclude-module=IPython",
    "--exclude-module=pytest",

    # ---- 用不到的 PySide6 子模块 ----
    # 代码只 import QtCore / QtGui / QtWidgets / QtSvg / QtMultimedia / QtMultimediaWidgets。
    # 不要顺手排掉 QtNetwork、QtOpenGL、QtPrintSupport：
    # Multimedia 依赖 QtNetwork，QVideoWidget 依赖 QtOpenGL，matplotlib 打印走 QtPrintSupport。
    "--exclude-module=PySide6.QtWebEngineCore",
    "--exclude-module=PySide6.QtWebEngineWidgets",
    "--exclude-module=PySide6.QtWebChannel",
    "--exclude-module=PySide6.QtQml",
    "--exclude-module=PySide6.QtQuick",
    "--exclude-module=PySide6.QtQuickWidgets",
    "--exclude-module=PySide6.Qt3DCore",
    "--exclude-module=PySide6.Qt3DRender",
    "--exclude-module=PySide6.Qt3DInput",
    "--exclude-module=PySide6.Qt3DLogic",
    "--exclude-module=PySide6.Qt3DAnimation",
    "--exclude-module=PySide6.Qt3DExtras",
    "--exclude-module=PySide6.QtBluetooth",
    "--exclude-module=PySide6.QtNfc",
    "--exclude-module=PySide6.QtPositioning",
    "--exclude-module=PySide6.QtLocation",
    "--exclude-module=PySide6.QtSensors",
    "--exclude-module=PySide6.QtSerialPort",
    "--exclude-module=PySide6.QtSerialBus",
    "--exclude-module=PySide6.QtTest",
    "--exclude-module=PySide6.QtSql",
    "--exclude-module=PySide6.QtDesigner",
    "--exclude-module=PySide6.QtHelp",
    "--exclude-module=PySide6.QtCharts",
    "--exclude-module=PySide6.QtDataVisualization",
    # Qt Virtual Keyboard は Community 版では GPLv3-only。应用没有导入或调用它；
    # 不要让 PyInstaller 的插件收集把 GPL-only 组件混进 MIT/LGPL 分发包。
    "--exclude-module=PySide6.QtVirtualKeyboard",
    "--exclude-module=PySide6.QtPdf",
    "--exclude-module=PySide6.QtTextToSpeech",
    "--exclude-module=PySide6.QtRemoteObjects",
    "--exclude-module=PySide6.QtScxml",
    "--exclude-module=PySide6.QtGamepad"
)

# 控制台开关放在这里而不是数组里：CI 恒定 --windowed，本地排查时才要黑窗。
if ($Console) { $piArgs += "--console" } else { $piArgs += "--windowed" }

# exe 属性页（右键 → 详细信息）：版本号与 build_id 从 stage 的 manifest 来，
# 由 _protect_stage.py 生成（它才知道 build_id）。-NoHarden 时没有这份文件，直接不带。
if ($Hardened) {
    $vinfo = Join-Path $BuildDir "_ly_version_info.txt"
    if (-not (Test-Path $vinfo)) { throw "加固了却没有 _ly_version_info.txt，水印进不了 exe 属性页" }
    $piArgs += "--version-file=$vinfo"
}

# 图标住在 platform\win（方案 B：.ico 是平台资产，不进 src、也不跟着 stage 副本走）。
# 兜底那两条是给旧包形状与临时改图标留的，命中哪条都打印出来，别让人猜。
$ico = @((Join-Path $PlatformDir "logo.ico"), (Join-Path $Here "logo.ico"),
        (Join-Path $Here "assets\logo.ico")) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($ico) { $piArgs += "--icon=$ico"; Write-Host "  icon   : $ico" }
else      { Write-Host "  icon   : (none found, building without)" }

Write-Host "`n-- pyinstaller ($($piArgs.Count) args) --" -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $VenvPy -m PyInstaller @piArgs main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller exited with $LASTEXITCODE" }
$sw.Stop()

# Qt Virtual Keyboard is GPLv3-only in the Community edition. Learnlysis does not
# use it; PyInstaller may still collect its Qt DLL/plugin with other Qt plugins.
# Remove those exact unused artifacts from onedir bundles and fail closed if any remain.
if ($Layout -eq "onedir") {
    $virtualKeyboard = @(Get-ChildItem $distOut -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(Qt6VirtualKeyboard\.dll|qtvirtualkeyboardplugin\.dll)$' })
    foreach ($file in $virtualKeyboard) {
        Remove-Item -LiteralPath $file.FullName -Force
        Write-Host ("  removed GPL-only unused Qt component: {0}" -f $file.Name) -ForegroundColor Yellow
    }
    $remainingVirtualKeyboard = @(Get-ChildItem $distOut -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(Qt6VirtualKeyboard\.dll|qtvirtualkeyboardplugin\.dll)$' })
    if ($remainingVirtualKeyboard.Count) {
        throw "GPL-only Qt Virtual Keyboard files remain in the onedir distribution"
    }
}

# ---------------------------------------------------------------- 体积报告
Write-Host "`n-- size report --" -ForegroundColor Cyan
$dist = $distOut
$bytes = (Get-ChildItem $dist -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("  layout = {0,-8} dist total = {1:N1} MB   ({2:N0}s)" -f $Layout, ($bytes / 1MB), $sw.Elapsed.TotalSeconds)

$exe = Get-ChildItem $dist -Recurse -Filter *.exe | Sort-Object Length -Descending | Select-Object -First 1
if ($exe) { Write-Host ("  main exe : {0}  {1:N1} MB" -f $exe.Name, ($exe.Length / 1MB)) }

Write-Host "  top files:"
Get-ChildItem $dist -Recurse -File | Sort-Object Length -Descending | Select-Object -First 8 |
    ForEach-Object { Write-Host ("    {0,8:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Substring($dist.Length + 1)) }

# --------------------------------------------------------------- 加固产物 gate
# manifest 说「编出来了」只是脚本的自述；真正的证据是 dist 里躺着这些文件，
# 而且它们对应的 .py 不在副本里（留着 .py 等于白编：源码 import 优先级更高）。
Write-Host "`n-- hardening gate --" -ForegroundColor Cyan
if (-not $Hardened) {
    Write-Host "  SKIPPED (-NoHarden)：这份包没有 .pyd、没有水印、docstring 全在，别发出去" -ForegroundColor Yellow
} else {
    $manifest = Get-Content (Join-Path $StageDir "_ly_manifest.json") -Raw | ConvertFrom-Json
    if (@($manifest.compiled).Count -lt $HardenedModules.Count) {
        throw "加固清单里只编出 $(@($manifest.compiled).Count) 个模块（要求 $($HardenedModules.Count)）：$($manifest.compiled -join ', ')"
    }
    if ($manifest.strip.docstrings_left -ne 0) {
        throw "stage 里仍残留 $($manifest.strip.docstrings_left) 条 docstring"
    }
    if (-not $manifest.build_id) { throw "manifest 里没有 build_id，水印没写进去" }
    $distFiles = @(Get-ChildItem $dist -Recurse -File)
    $missing = @()
    foreach ($m in $HardenedModules) {
        $pyd = @($distFiles | Where-Object { $_.Name -match "^$m(\..+)?\.pyd$" })
        $src = @($distFiles | Where-Object { $_.Name -eq "$m.py" })
        if (-not $pyd) { $missing += "$m (缺 .pyd)" }
        if ($src)      { $missing += "$m (发行目录里还留着 .py)" }
    }
    if ($missing.Count) { throw "加固未落到产物：$($missing -join '; ')" }
    Write-Host ("  hardening OK: {0} 个模块以 .pyd 进包；build_id = {1}" -f `
                $HardenedModules.Count, $manifest.build_id) -ForegroundColor Green
}

# 可选后端必须真的没进包 —— 这是「体积闸门」有没有生效的唯一硬证据。
# ⚠️ 变量名必须写成 ${probe}_cpu：写成 $probe_cpu 会被当成一个未定义变量名，
#    展开成空串后正则退化成 `\.dll`，会把 dist 里每一个 DLL 都报成泄漏（踩过）。
Write-Host "`n-- optional-backend leak check --" -ForegroundColor Cyan
$leaked = @()
foreach ($probe in @("torch", "xgboost", "shap")) {
    $hits = Get-ChildItem $dist -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "^${probe}([.\-_]|$)" -or
                           $_.Name -match "^${probe}_cpu\.dll$" -or
                           $_.Name -match "^lib${probe}" }
    if ($hits) { $leaked += $probe; Write-Host ("  LEAK {0}: {1}" -f $probe, ($hits.Name -join ", ")) -ForegroundColor Red }
}
if (-not $leaked) { Write-Host "  clean: torch / xgboost / shap absent from dist" -ForegroundColor Green }
else { throw "optional backend leaked into dist: $($leaked -join ', ')" }

# --------------------------------------------------------------- 法律文本随包
# MIT、LGPL、Python 与第三方许可文本必须随包；-NoHarden 也照拷。
Write-Host "`n-- legal files --" -ForegroundColor Cyan
$pkgDir = if ($Layout -eq "onedir") {
    (Get-ChildItem $dist -Directory | Select-Object -First 1).FullName
} else { $dist }
foreach ($doc in @("LICENSE", "NOTICE", "LGPL-3.0.txt", "LGPL-2.1.txt", "PYTHON-LICENSE.txt")) {
    $from = Join-Path $RepoRoot $doc
    if (-not (Test-Path $from)) { throw "仓库根缺许可文件 $doc，包里不能没有" }
    $targetName = if ($doc -in @("LICENSE", "NOTICE")) { "$doc.txt" } else { $doc }
    Copy-Item $from (Join-Path $pkgDir $targetName) -Force
    Write-Host "  + $targetName"
}
$thirdPartySource = Join-Path $RepoRoot "THIRD_PARTY_LICENSES"
if (-not (Test-Path $thirdPartySource -PathType Container)) { throw "仓库根缺 THIRD_PARTY_LICENSES 目录" }
$thirdPartyTarget = Join-Path $pkgDir "THIRD_PARTY_LICENSES"
New-Item -ItemType Directory -Path $thirdPartyTarget -Force | Out-Null
Copy-Item (Join-Path $thirdPartySource "*") $thirdPartyTarget -Recurse -Force
Write-Host "  + THIRD_PARTY_LICENSES"
if ($Hardened) {
    # 加固证据（build_id / 编译清单）跟着包走一份：用户报障时不用翻构建机就能定位现场。
    Copy-Item (Join-Path $StageDir "_ly_manifest.json") (Join-Path $pkgDir "BUILD_INFO.json") -Force
    Write-Host "  + BUILD_INFO.json"
}

# ---------------------------------------------------------------- 代码签名（可选）
# 与 CI 的 sign-windows job 同一口径：没证书就诚实跳过，不拿「没签」当成失败。
# 签的只是主 exe——第三方 DLL 的签名属于上游，本项目不去动它们。
# 本机用法：
#     $env:LEARNLYSIS_SIGN_PFX = 'C:\secrets\codesign.pfx'
#     $env:LEARNLYSIS_SIGN_PFX_PASSWORD = '...'
#     .\build_exe.ps1
Write-Host "`n-- code signing --" -ForegroundColor Cyan
if ($env:LEARNLYSIS_SIGN_PFX) {
    if (-not (Test-Path $env:LEARNLYSIS_SIGN_PFX)) { throw "LEARNLYSIS_SIGN_PFX 指向的文件不存在" }
    $pw = if ($env:LEARNLYSIS_SIGN_PFX_PASSWORD) {
        ConvertTo-SecureString $env:LEARNLYSIS_SIGN_PFX_PASSWORD -AsPlainText -Force
    } else { $null }
    $cert = if ($pw) { New-Object Security.Cryptography.X509Certificates.X509Certificate2 ($env:LEARNLYSIS_SIGN_PFX, $pw) }
            else     { New-Object Security.Cryptography.X509Certificates.X509Certificate2 ($env:LEARNLYSIS_SIGN_PFX) }
    Set-AuthenticodeSignature -FilePath $exe.FullName -Certificate $cert `
        -HashAlgorithm SHA256 -TimestampServer 'http://timestamp.digicert.com' |
        Select-Object Status, Path | Format-List | Out-String -Width 200 | Write-Host
    # 签完立刻验一次：状态不是 Valid 就直接失败，不要交出一份「以为签好了」的包。
    # 也是 integrity.py 能报 signed 的前置条件（两者走同一个 WinVerifyTrust）。
    $st = (Get-AuthenticodeSignature $exe.FullName).Status
    if ($st -ne 'Valid') { throw "签名后校验为 $st（要求 Valid）" }
    Write-Host "  signed & verified" -ForegroundColor Green
} else {
    Write-Host "  未配置 LEARNLYSIS_SIGN_PFX —— 本次产物不签名（integrity.py 会显示「未签名构建」）" -ForegroundColor Yellow
}

Write-Host ("`nDONE -> {0}" -f $exe.FullName) -ForegroundColor Green
Write-Host "  run it:  & '$($exe.FullName)'"
