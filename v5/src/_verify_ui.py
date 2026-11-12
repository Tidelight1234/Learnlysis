"""临时验证：真实桌面截图检查标题栏、菜单、关于弹窗、倍速下拉、最大化与深色态。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

import theme

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "..", "ui-mockups", "verify")
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app)

from main_window import MainWindow
from about_dialog import AboutDialog

w = MainWindow()
w.move(120, 60)
w.resize(1500, 800)
w.show()


def pump(ms):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


pump(500)

screen = QGuiApplication.primaryScreen()


def shoot_screen(tag, rect):
    pm = screen.grabWindow(0)
    dpr = pm.devicePixelRatio()
    region = pm.copy(int(rect.x() * dpr), int(rect.y() * dpr),
                     int(rect.width() * dpr), int(rect.height() * dpr))
    path = os.path.join(OUT, tag + ".png")
    region.save(path)
    log("SAVED", path)


def shoot_widget(tag, widget, rect=None):
    pm = widget.grab()
    if rect is not None:
        pm = pm.copy(rect)
    path = os.path.join(OUT, tag + ".png")
    pm.save(path)
    log("SAVED", path)


# 1 标题栏（浅色）
shoot_widget("1_titlebar_light", w, QRect(0, 0, w.width(), 46))

# 2 文件菜单
btn = w.title_bar.menu_buttons["file"]
w.file_menu.popup(btn.mapToGlobal(QPoint(0, btn.height() + 4)))
pump(500)
shoot_screen("2_file_menu", w.file_menu.frameGeometry().adjusted(-30, -50, 30, 30))
w.file_menu.hide()
pump(300)

# 3 视图菜单
btn = w.title_bar.menu_buttons["view"]
w.view_menu.popup(btn.mapToGlobal(QPoint(0, btn.height() + 4)))
pump(500)
shoot_screen("3_view_menu", w.view_menu.frameGeometry().adjusted(-30, -50, 30, 30))
w.view_menu.hide()
pump(300)

# 4 关于弹窗
dlg = AboutDialog(w)
dlg.show()
pump(500)
shoot_screen("4_about", dlg.frameGeometry().adjusted(-40, -40, 40, 48))
dlg.close()
pump(300)

# 5 倍速下拉
combo = w.video_player.speed_combo
combo.showPopup()
pump(500)
container = combo.view().parentWidget().window()
shoot_screen("5_speed_popup", container.frameGeometry().adjusted(-30, -60, 30, 30))
combo.hidePopup()
pump(300)

# 6 最大化态标题栏
w.showMaximized()
pump(500)
shoot_screen("6_maximized_top", QRect(0, 0, 900, 120))
w.showNormal()
pump(500)

# 7 深色标题栏 + 文件菜单
w.dark_action.setChecked(True)
pump(500)
shoot_widget("7_titlebar_dark", w, QRect(0, 0, w.width(), 46))
btn = w.title_bar.menu_buttons["file"]
w.file_menu.popup(btn.mapToGlobal(QPoint(0, btn.height() + 4)))
pump(500)
shoot_screen("8_file_menu_dark", w.file_menu.frameGeometry().adjusted(-30, -50, 30, 30))
w.file_menu.hide()
pump(300)
dlg = AboutDialog(w)
dlg.show()
pump(500)
shoot_screen("9_about_dark", dlg.frameGeometry().adjusted(-40, -40, 40, 48))
dlg.close()
pump(200)
w.dark_action.setChecked(False)
pump(300)

log("VERIFY_DONE")
os._exit(0)
