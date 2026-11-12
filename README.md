# Learnlysis

**PySide6 v5 · Collaborative Learning Behavior Coding and Analysis**

Learnlysis is a desktop application for coding and analyzing collaborative learning behavior from video. It supports frame-based individual coding, group interaction event logging, data summaries, and a collection of descriptive and sequence analysis methods. Optional AI features provide coding suggestions for human review.

## Features

- Import video and code individual behavior frame by frame with a configurable coding framework.
- Record group interaction events separately from individual behavior states.
- Summarize coding data and export analysis-ready spreadsheets.
- Explore descriptive statistics (M01–M19), individual pattern analyses (IND01–IND15), and group pattern analyses (GRP01–GRP10).
- Use optional AI-assisted coding and analysis planning. AI-generated codes are suggestions that require human confirmation; they do not silently replace manually entered codes.
- Save the current workspace locally and restore it when the application starts.

## Getting started

The source application uses Python 3.11 and PySide6 6.8.3. From the repository root on Windows, create an environment, install the core dependencies, and launch v5:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\v5\src\main.py
```

The optional video-frame backend can be installed with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

Build-only dependencies are listed in `requirements-build.txt`.

## Windows release

A ready-to-run Windows onedir build is available in `v5/dist/Learnlysis/`. Run `Learnlysis.exe` and keep the neighboring `_internal` directory and license files together with it. The published package includes the Qt/PySide6 runtime and is larger than the source-only repository. ZIP archives in `v5/dist/` are excluded from Git; publish them as GitHub Release assets.

## Project layout

```text
LICENSE                    MIT license for Learnlysis original code
NOTICE                     Third-party notices and Qt/PySide6 source information
THIRD_PARTY_LICENSES/      Third-party license texts and font agreement
requirements*.txt          Runtime, AI, and build dependencies
v5/src/                    Application and analysis source
v5/platform/win/           Windows run and packaging scripts
v5/dist/Learnlysis/        Windows onedir application package
```

This repository publishes the PySide6 v5 application. Local virtual environments, build intermediates, and user runtime data are excluded from version control.

## Licensing

Learnlysis original code is licensed under the MIT License; see [LICENSE](LICENSE). Third-party libraries, Qt/PySide6, Python, fonts, and other bundled materials are **not** relicensed under MIT and retain their own terms. See [NOTICE](NOTICE), `LGPL-3.0.txt`, `LGPL-2.1.txt`, `PYTHON-LICENSE.txt`, and `THIRD_PARTY_LICENSES/`. The Windows package also carries these notices and license texts so it can be distributed independently of this source repository.

---

# Learnlysis（中文）

**PySide6 v5 · 协作学习行为编码与分析工具**

Learnlysis 是一款用于协作学习行为编码与分析的桌面软件。它支持导入视频并逐帧记录个体行为、录入小组交互事件、汇总数据，并提供描述统计和序列模式分析。可选 AI 功能用于生成待人工确认的编码建议。

## 功能

- 导入视频，使用可配置的编码框架逐帧记录个体行为。
- 独立记录小组交互事件，保留事件数据与个体状态数据的区别。
- 汇总编码数据，并导出适用于后续分析的表格。
- 提供描述统计（M01–M19）、个体模式分析（IND01–IND15）和小组模式分析（GRP01–GRP10）。
- 提供可选的 AI 辅助编码和分析规划。AI 生成的编码仅作为建议，须由研究者确认，不会静默覆盖人工编码。
- 在本地自动保存当前工作现场，并在程序启动时恢复。

## 安装与启动

源码版本使用 Python 3.11 和 PySide6 6.8.3。Windows PowerShell 下，在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\v5\src\main.py
```

如需安装可选的视频抽帧后端：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

打包所需的依赖列在 `requirements-build.txt` 中。

## Windows 程序包

可直接运行的 Windows onedir 程序位于 `v5/dist/Learnlysis/`。启动 `Learnlysis.exe` 时，请保留旁边的 `_internal` 目录和许可文件。程序包包含 Qt/PySide6 运行库，可以独立于源码仓库分发。`v5/dist/` 中的 ZIP 压缩包不纳入 Git，应作为 GitHub Release 附件发布。

## 项目结构

```text
LICENSE                    Learnlysis 原创代码的 MIT 许可证
NOTICE                     第三方声明及 Qt/PySide6 源码信息
THIRD_PARTY_LICENSES/      第三方许可证文本和字体授权协议
requirements*.txt          运行、AI 和打包依赖
v5/src/                    应用程序与分析源码
v5/platform/win/           Windows 启动与打包脚本
v5/dist/Learnlysis/        Windows onedir 程序包
```

本仓库发布 PySide6 v5。虚拟环境、构建中间产物和用户运行数据不会纳入版本控制。

## 许可证

Learnlysis 原创代码采用 MIT 许可证，详见 [LICENSE](LICENSE)。第三方库、Qt/PySide6、Python、字体及其他随包材料不受本项目 MIT 许可证重新授权，仍适用各自的许可证或协议。详见 [NOTICE](NOTICE)、`LGPL-3.0.txt`、`LGPL-2.1.txt`、`PYTHON-LICENSE.txt` 和 `THIRD_PARTY_LICENSES/`。Windows 程序包也附带相应声明和许可证文本，可独立于源码仓库分发。
