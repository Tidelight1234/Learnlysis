# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""程序入口：启动学习行为编码与分析系统。"""

import os
import sys
import datetime
import threading
import traceback

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory
from PySide6.QtGui import QIcon, QGuiApplication

import config
from main_window import MainWindow
import icon_provider
import theme
from language import install_language_filter


# --------------------------------------------------------------------------
# 全局异常兜底（低成本防护档第 4 项：减少信息泄露面 + 不让应用无声闪退）
#
# 为什么需要它：PySide6 ≥ 5.5 的默认行为是——槽函数 / 重写 C++ 方法里逃逸出来的
# Python 异常直接触发 qFatal()→abort()，应用当场闪退且不给任何提示。但只要把
# sys.excepthook 替换成非默认值，PySide6 就改为调用它、不再 abort。所以这一个 hook
# 同时兜住了「主线程未捕获异常」与「Qt 槽里的异常」两类。
#
# 安全口径：对用户只给一句人话，绝不把 traceback 甩到弹窗里（那等于主动交出内部
# 结构）；详细信息只落在本机 data/crash.log 供用户按需反馈，程序不主动外发。
# --------------------------------------------------------------------------

#: crash.log 的滚动上限：超过就只保留最近一半，避免长跑机器上无限增长
_CRASHLOG_MAX_BYTES = 256 * 1024


def _append_crash_log(text):
    """把一条崩溃记录追加到 data/crash.log。本函数自身绝不许再抛异常。"""
    try:
        path = os.path.join(config.data_dir, "crash.log")
        os.makedirs(config.data_dir, exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > _CRASHLOG_MAX_BYTES:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                tail = fh.read()[-(_CRASHLOG_MAX_BYTES // 2):]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("...（更早的记录已截断）...\n" + tail)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
    except Exception:                              # noqa: BLE001  日志失败绝不能连累主流程
        pass


def _friendly_hook(exc_type, exc_value, exc_tb):
    """sys.excepthook 的替代实现：本地留痕 + 友好弹窗（offscreen 下只留痕不弹窗）。"""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _append_crash_log("\n[%s] %s: %s\n%s" % (
        stamp, getattr(exc_type, "__name__", exc_type), exc_value, detail))
    # -Console 打包 / 开发运行时，控制台里仍留一份，方便当场排查
    try:
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    except Exception:                              # noqa: BLE001
        pass

    app = QApplication.instance()
    if app is None:
        return
    # 冒烟测试与 pytest 强制 offscreen；此时弹模态框会阻塞整个测试进程，必须跳过
    try:
        if app.platformName() == "offscreen":
            return
    except Exception:                              # noqa: BLE001
        pass
    try:
        box = QMessageBox()
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("出了点小问题")
        box.setText("这一步操作没能完成，但程序可以继续运行。")
        box.setInformativeText("如果反复出现，请把 data/crash.log 一并发给开发者。")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()
    except Exception:                              # noqa: BLE001  兜底本身不能再崩
        pass


def _install_crash_guard():
    """安装全局异常兜底：主线程走 sys.excepthook，Python 线程走 threading.excepthook。"""
    sys.excepthook = _friendly_hook

    def _thread_hook(args):
        # QThread 里的计算由 ComputeWorker 自己 try 住；这里兜的是裸 threading 线程
        _friendly_hook(args.exc_type, args.exc_value, args.exc_traceback)

    try:
        threading.excepthook = _thread_hook
    except Exception:                              # noqa: BLE001  极老的解释器没有这个钩子
        pass


def _application_icon():
    """窗口图标：开发态在同树的 platform/<os>/ 里，打包后随包落在 _MEIPASS。

    theme.APP_DIR 现在是 src/，而 .icns/.ico 是平台资产、不在 src/ —— 这就是
    候选清单要同时认两处的前半截；后半截是 frozen 包，见 build_macos.sh 与
    platform/win/build_exe.ps1 里的 --add-data ...:.
    """
    is_mac = sys.platform == "darwin"
    name = "logo.icns" if is_mac else "logo.ico"
    tree_root = os.path.dirname(theme.APP_DIR)
    candidates = (
        os.path.join(theme.APP_DIR, name),                       # frozen：随包进 _MEIPASS 根
        os.path.join(tree_root, "platform", "mac" if is_mac else "win", name),  # 开发态覆盖层
        os.path.join(theme.APP_DIR, "logo.ico"),                 # 旧包形状的兜底
    )
    return QIcon(next(path for path in candidates if os.path.exists(path)))


if __name__ == '__main__':
    # Qt 6 默认按屏幕缩放比例使用小数 DPR；Round 让 125%/150% 的逻辑像素
    # 与原版 Qt 5 的取整策略一致，避免字号、控件和布局产生系统性放大。
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.Round)
    # Qt 6.8 在 Windows 上默认使用 DirectWrite；Qt 5 使用 GDI。保留 GDI
    # 能使同机 MiSans 字体栅格化效果与原版更接近，同时尊重用户显式设置的平台参数。
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_PLATFORM", "windows:fontengine=gdi")
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setApplicationDisplayName(config.APP_NAME)
    install_language_filter(app)
    _install_crash_guard()
    app.setStyle(QStyleFactory.create("Fusion"))
    theme.load_fonts()
    theme.load_mode()
    theme.apply_theme(app)
    icon = _application_icon()
    app.setWindowIcon(icon)
    window = MainWindow()
    # 任务栏图标是品牌标识，不随深色模式反转
    window.setWindowIcon(icon)
    window.check_and_restore_archive()  # 新增
    window.show()
    sys.exit(app.exec())
