"""pytest 全局配置：离屏平台 + QApplication 会话 fixture。"""

import os
import sys

# ── 离屏平台必须在 QApplication 构造前设置 ──
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ── 确保 version/v5 在 sys.path 中（模块间用相对导入） ──
_V5_DIR = os.path.dirname(os.path.abspath(__file__))
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """整个测试会话共享一个 QApplication 实例。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="session", autouse=True)
def chart_fonts(qapp):
    """把图表字体注册成和真应用一致（口径同 `_smoke_modes.configure_chart_fonts`）。

    产品链路里 `theme.apply_theme` 会走 `_apply_matplotlib`；测试进程不建主窗口，
    不补这一步 matplotlib 会退回 DejaVu Sans——中文全画成方框，绘图层「按量到的
    文字矩形决定画布尺寸」的断言也就量的不是真字宽。
    """
    from matplotlib import font_manager, rcParams

    import theme
    for fname in theme._FONT_FILES:
        path = os.path.join(theme.FONTS_DIR, fname)
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:                     # noqa: BLE001
                pass
    known = {entry.name for entry in font_manager.fontManager.ttflist}
    chosen = [family for family in ("MiSans", theme.FONT_FALLBACK, "Microsoft YaHei", "SimHei")
              if family in known]
    if not chosen:
        return
    rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.unicode_minus"] = False
