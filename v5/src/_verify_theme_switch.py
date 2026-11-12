"""真屏验收：走界面同款路径（菜单「深色模式」）切换主题，检查视频占位底色是否立即同步。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_MULTIMEDIA_PREFERRED_PLUGINS", "windowsmediafoundation")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

import theme

app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app, "light")

from main_window import MainWindow

w = MainWindow()
w.move(60, 60)
w.resize(1200, 800)
w.show()


def pump(ms):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


pump(1200)
screen = QGuiApplication.primaryScreen()
vw = w.video_player.video_widget
FAILED = []


def sample(tag):
    """抓取视频区中心像素，同时给出期望色（当前主题 token）。"""
    pos = vw.mapToGlobal(QPoint(vw.width() // 2, vw.height() // 2))
    pm = screen.grabWindow(0)
    dpr = pm.devicePixelRatio()
    got = pm.toImage().pixelColor(int(pos.x() * dpr), int(pos.y() * dpr)).name().lower()
    want = theme.tokens()["video_bg"].lower()
    ok = got == want
    if not ok:
        FAILED.append(tag)
    print("%-38s screen=%s want=%s %s" % (tag, got, want, "OK" if ok else "FAIL"), flush=True)


# 1) 深色：切换调用返回后（仍在 _on_dark_toggled 内）与事件循环空转后各测一次
w.dark_action.trigger()
sample("dark：切换调用刚返回")
pump(800)
sample("dark：事件循环空转后")

# 2) 浅色：同上
w.dark_action.trigger()
sample("light：切换调用刚返回")
pump(800)
sample("light：事件循环空转后")

# 3) 连续快速来回切，确认不会残留旧色
for i in range(3):
    w.dark_action.trigger()
    pump(120)
    sample("round%d 快速切换后" % i)

w.close()
print("FAILURES=%s" % (FAILED or "none"))
print("VERIFY_DONE")
