"""离屏渲染视频面板：浅色/深色 + 速度下拉展开态，供人工目视验收。"""
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


pump(400)

vp = w.video_player
vp.file_list.addItems([
    r"D:\课堂录像\课堂互动_01.avi",
    r"D:\课堂录像\课堂互动_02.avi",
    r"D:\课堂录像\小组讨论_03_long_file_name_for_elide.avi",
])
vp.current_file_path = r"D:\课堂录像\课堂互动_02.avi"
vp.highlight_current_file()
vp.current_time_label.setText("04:12 / 45:00")
vp.progress_slider.setRange(0, 2700000)
vp.progress_slider.setValue(252000)
vp.duration_slider.setValue(15)
pump(300)

for tag in ("light", "dark"):
    vp.grab().save(os.path.join(OUT, "video-%s.png" % tag))
    print("SAVED video-%s.png" % tag)
    if tag == "light":
        w.dark_action.setChecked(True)
        pump(400)

w.dark_action.setChecked(False)
pump(400)

vp.speed_combo.showPopup()
pump(250)
popup = vp.speed_combo.view().window()
popup.grab().save(os.path.join(OUT, "video-speed-popup.png"))
print("SAVED video-speed-popup.png", popup.width(), popup.height())
vp.speed_combo.hidePopup()

if backup is not None:
    open(settings, "w", encoding="utf-8").write(backup)
elif os.path.exists(settings):
    os.remove(settings)
print("VIDEO_SHOTS_DONE")
