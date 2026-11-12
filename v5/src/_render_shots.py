"""离屏渲染截图：把主窗口四个标签页在浅/深两套主题下导出 PNG，供人工目视验收。"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication

import theme

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "..", "ui-mockups", "shots")
os.makedirs(OUT, exist_ok=True)

settings = theme.theme.settings_path()
backup = open(settings, encoding="utf-8").read() if os.path.exists(settings) else None

app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app, "light")

from main_window import MainWindow

w = MainWindow()
w.resize(1600, 900)
w.show()


def pump(ms):
    # offscreen 下只有 processEvents 不会推进 QTimer，fade_in 会停在 opacity=0
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


pump(300)


def shoot(tag):
    for i in range(w.tab_widget.count()):
        w.tab_widget.setCurrentIndex(i)
        pump(400)
        name = "%s-%d-%s.png" % (tag, i, w.tab_widget.tabText(i))
        path = os.path.join(OUT, name)
        w.grab().save(path)
        print("SAVED", path)


shoot("light")
w.dark_action.setChecked(True)
pump(400)
shoot("dark")
w.dark_action.setChecked(False)
pump(400)

if backup is not None:
    open(settings, "w", encoding="utf-8").write(backup)
elif os.path.exists(settings):
    os.remove(settings)
print("SHOTS_DONE")
