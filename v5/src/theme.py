# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""统一主题层：设计 token、全局 QSS、字体加载、主题切换与卡片阴影。

唯一样式来源：业务代码通过动态属性（btn/card/codeRow/section/notice）
与本模块提供的 helper 表达样式，禁止再写内联 setStyleSheet。
"""

import json
import os
import sys
import weakref

from PySide6.QtCore import QEvent, QObject, Qt, QRectF, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetrics, QPalette, QPainter,
                         QPainterPath, QRadialGradient, QRegion)
from PySide6.QtWidgets import (QApplication, QGraphicsDropShadowEffect, QLabel, QWidget,
                             QPushButton, QTabBar, QToolButton)

import icon_provider


def coding_font_pixels(size):
    if isinstance(size, dict):
        return size["custom"]
    return {"small": 12, "medium": 13, "normal": 14}.get(size, 14)


def coding_density_value(size, small, normal):
    """以 12px 小档和 14px 原档为锚点，给自定义字号同步缩放布局。"""
    return max(1, round(small + (coding_font_pixels(size) - 12) * (normal - small) / 2))


def coding_dimension_style(size):
    """数据编码页局部字号；颜色和交互态仍由全局主题管理。"""
    if size == "normal":
        return ""
    base = coding_font_pixels(size)
    section = base + 1
    height = coding_density_value(size, 20, 24)
    vertical = coding_density_value(size, 3, 5)
    return (f'QLabel, QLineEdit, QPushButton, QCheckBox {{ font-size: {base}px; }} '
            f'QLabel[section="true"] {{ font-size: {section}px; }} '
            f'QLineEdit {{ min-height: {height}px; padding: {vertical}px 6px; }} '
            f'QPushButton {{ padding: {vertical}px 6px; }}')


def coding_table_style(size):
    if size == "normal":
        return ""
    base = coding_font_pixels(size)
    header = base - 1
    vertical = coding_density_value(size, 4, 6)
    horizontal = coding_density_value(size, 7, 11)
    return (f'QTableWidget {{ font-size: {base}px; }} '
            f'QTableWidget::item {{ padding: {vertical}px {horizontal}px; }} '
            f'QHeaderView::section {{ font-size: {header}px; padding: {vertical}px {horizontal}px; }} '
            f'QComboBox, QLineEdit, QListView {{ font-size: {base}px; }}')


def coding_action_button_style(size):
    """编码页维度面板下方的操作按钮，与面板按钮使用同一字号阶梯。"""
    if size == "normal":
        return ""
    base = coding_font_pixels(size)
    vertical = coding_density_value(size, 3, 5)
    horizontal = coding_density_value(size, 10, 14)
    return (f'QPushButton {{ font-size: {base}px; '
            f'padding: {vertical}px {horizontal}px; }}')


def coding_input_style(size):
    """时间胶囊与注释输入区跟随编码字号和控件密度。"""
    if size == "normal":
        return ""
    base = coding_font_pixels(size)
    hint = max(9, base - 1)
    vertical = coding_density_value(size, 3, 5)
    horizontal = coding_density_value(size, 7, 10)
    return (f'QLineEdit {{ font-size: {base}px; padding: {vertical}px {horizontal}px; }} '
            f'QLabel {{ font-size: {hint}px; }}')

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_fonts_dir():
    """定位内嵌 MiSans 字体目录，兼容源码运行与三种打包布局。

    Text Coder / Text Trainer 的 spec 把资产放在 _MEIPASS/v5/assets（两个 app 的
    main.py 也按 BUNDLE_DIR/v5 找 theme），而本模块作为顶层模块导入时 __file__
    在 _MEIPASS 根——只按 __file__ 拼路径会在打包后找不到字体、静默回退雅黑。
    因此按候选目录依次探测，取第一个真正含有字体文件的。
    """
    candidates = [os.path.join(APP_DIR, "assets", "fonts")]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        for sub in (("v5", "assets", "fonts"), ("assets", "fonts")):
            candidates.append(os.path.join(meipass, *sub))
    for directory in candidates:
        if os.path.isfile(os.path.join(directory, _FONT_FILES[0])):
            return directory
    return candidates[0]


FONT_FALLBACK = "Microsoft YaHei UI"
_FONT_FILES = ("MiSans-Regular.ttf", "MiSans-Medium.ttf", "MiSans-Semibold.ttf")

FONTS_DIR = _resolve_fonts_dir()

# 树状图连线用：紫色系渐进，装饰性而非分类
CHART_COLORS = ["#6D5CE7", "#8B7CF0", "#AAA0F4", "#C9C2F8", "#3BA55D", "#E3A008", "#E5484D"]
# 分类色板：与下方 QSS 中 chip 的 tone="0"~"7" 同色，图表与标签共用一套色相
CATEGORY_COLORS = ["#6D5CE7", "#3BA55D", "#E3A008", "#E5484D",
                   "#0EA5E9", "#EC4899", "#14B8A6", "#9AA0AE"]

TOKENS = {
    "light": {
        "canvas": "#F5F5F8", "card": "#FFFFFF", "card_sub": "#FAFAFC",
        "line": "#E7E7EE", "line_strong": "#D9D9E3",
        "ink1": "#20222B", "ink2": "#5C6070", "ink3": "#9AA0AE",
        "purple": "#6D5CE7", "purple_ink": "#5346C9",
        "tint": "#EFECFD", "tint2": "#E4DEFC",
        # 标签栏滚动按钮的悬停/按下底：浅色下 tint 本就是不透明色，直接沿用。
        "tab_btn_hover": "#EFECFD", "tab_btn_press": "#E4DEFC",
        "track": "#ECECF2", "hover_row": "#F7F7FB",
        "dim_head": "#ECECF2", "dim_row": "#F4F4F7",
        "stage": "#17181D", "video_bg": "#000000",
        "scroll_thumb": "#D9D9E3", "scroll_thumb_hover": "#C4C4D2",
        "disabled_bg": "#F2F2F6",
        "success": "#3BA55D", "warn": "#E3A008", "danger": "#E5484D",
        "atmo_inner": "#ECE9FB", "atmo_outer": "#F5F5F8",
        "shadow_rgb": "24,24,46", "shadow_alpha": 26,
    },
    "dark": {
        "canvas": "#17171C", "card": "#1F1F26", "card_sub": "#24242C",
        "line": "#2E2E38", "line_strong": "#3A3A46",
        "ink1": "#ECECF2", "ink2": "#A6A6B4", "ink3": "#6E6E7C",
        "purple": "#8B7CF0", "purple_ink": "#A99DF5",
        "tint": "rgba(139,124,240,0.16)", "tint2": "rgba(139,124,240,0.28)",
        # tint/tint2 是半透明的，只能叠在不透明底上用。标签栏本身是透明的
        # （为了那条笔直下划线），滚动按钮一旦拿半透明 tint 当 :hover/:pressed
        # 底，Qt 是「替换」而不是「叠加」背景 —— 于是悬停时按钮只剩 16% 不透明度，
        # 底下的标签字母直接透上来。这里给它们预先混好的不透明等效色
        # （tint@0.16 / tint2@0.28 over card #1F1F26）。
        "tab_btn_hover": "#302E46", "tab_btn_press": "#3D395F",
        "track": "#24242C", "hover_row": "rgba(255,255,255,0.04)",
        "dim_head": "#2A2A33", "dim_row": "#24242C",
        # 深色下视频占位底是白的，舞台边框若还是近黑就会在白块外圈出一圈黑边。
        # 取同色让两者连成一整块白（视频画面自身的信箱边本来也填 video_bg）。
        "stage": "#FFFFFF", "video_bg": "#FFFFFF",
        "scroll_thumb": "#3A3A46", "scroll_thumb_hover": "#4A4A58",
        "disabled_bg": "#24242C",
        "success": "#4CC179", "warn": "#F0B429", "danger": "#F2555A",
        "atmo_inner": "#221E33", "atmo_outer": "#17171C",
        "shadow_rgb": "0,0,0", "shadow_alpha": 110,
    },
}

# 字号阶梯（px）：整体上调一档，但保持既有层级——顶栏菜单/标签页名最大，区块标题
# 次之（不超过前两者），正文与辅助文字小步跟随。QSS 里所有 font-size 都取这里的
# 变量，配套的内边距同步收紧，使各控件 sizeHint 与调整前基本一致（放大字号≠放大布局）。
FONT_SIZES = {
    "fs_nav": 16,        # 顶栏菜单按钮 + 标签页名（原 13）
    "fs_section": 15,    # 区块标题：维度编辑 / 视频列表 / 维度名（原 14）
    "fs_time": 16,       # 编码区时间胶囊（原 15）
    "fs_base": 14,       # 正文：输入框 / 按钮 / 表格 / 列表 / 菜单项（原 13）
    "fs_small": 13,      # 辅助：表头 / 提示 / 芯片 / 时间码（原 12）
    "fs_title": 20,      # 关于弹窗软件名（原 18）
}

# 字重阶梯（QSS 数值）。MiSans 各字面自带的 usWeightClass 比名字低一档，实测
# （QSS 数值 → QFont 权重 → 实际命中的字面，用中文墨量覆盖率区分三个真字面）：
#   css 300 → qt 37 → Regular(0.0996)     css 400 → qt 50 → Medium(0.1143)
#   css 500 → qt 62 → Semibold(0.1507)    css 600 → qt 75 → 没有 Bold 字面，Qt 合成加粗(0.2040)
# 也就是说按字面意思写 600/700 会得到「算法描粗」的假 Bold，小字号下笔画糊成一团；
# 而写 400 的正文其实落在 Medium 上，整站看起来一律偏重。故三档取 300/400/500，
# 让每一档都命中真实字面，最重也只到 SemiBold，不再有合成加粗。
FONT_WEIGHTS = {
    "fw_regular": 300,    # → MiSans Regular：正文、列表项、菜单项
    "fw_medium": 400,     # → MiSans Medium：表头、时间码、单元格下拉
    "fw_semibold": 500,   # → MiSans Semibold：区块标题、选中页签、主按钮
}
# Qt 6 的 QFont 权重采用 100–900 标尺；300 对应 MiSans Regular。
APP_FONT_WEIGHT = 300

# 弹层圆角的唯一来源：QSS 的 border-radius 与 rounded_popup_mask 的裁剪半径必须是同一个数。
# 写成两个字面量就会各改各的 —— 画圆 12、裁圆 10（或反过来）时角上必然露出方形像素，
# 那正是这一族修复要根除的缺陷。下拉列表与 QMenu 本来就不同值，所以也不能共用一个默认参数。
POPUP_RADIUS = 12       # QListView#comboPopup / #speedPopup / #cellPopup
MENU_RADIUS = 10        # QMenu（菜单栏下拉、右键菜单、子菜单）

_QSS = """
* {{ outline: none; }}
QWidget {{ color: {ink1}; font-family: "{font}"; font-size: {fs_base}px; font-weight: {fw_regular}; }}
QMainWindow, QDialog {{ background: {canvas}; }}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {card}; border: 1px solid {line}; border-radius: 10px;
    padding: 7px 16px; color: {ink2}; font-size: {fs_base}px; min-height: 20px;
}}
QPushButton:hover {{ border-color: {line_strong}; background: {tint}; color: {purple_ink}; }}
QPushButton:pressed {{ background: {tint2}; color: {purple_ink}; }}
QPushButton:disabled {{ background: {disabled_bg}; border-color: {line}; color: {ink3}; }}
QPushButton[btn="primary"] {{ background: {purple}; border-color: transparent; color: #FFFFFF; font-weight: {fw_semibold}; }}
QPushButton[btn="primary"]:hover {{ background: {purple_ink}; }}
QPushButton[btn="primary"]:pressed {{ background: {purple}; }}
QPushButton[btn="primary"]:disabled {{ background: {disabled_bg}; color: {ink3}; }}
QPushButton[btn="ghost"] {{ background: transparent; border: none; border-radius: 8px; padding: 4px 8px; }}
QPushButton[btn="ghost"]:hover {{ background: {tint}; color: {purple_ink}; }}
QPushButton[btn="ghost"]:pressed {{ background: {tint2}; }}
/* 顶栏菜单项（文件/视图/说明）：字号最大的一档，纵向内边距收紧以保持 30px 按钮高 */
QPushButton[btn="ghost"][nav="true"] {{ padding: 2px 10px; font-size: {fs_nav}px; }}
QPushButton[btn="round"] {{ border-radius: 15px; padding: 0px; color: {ink2}; }}
QPushButton[btn="round"]:hover {{ background: {tint}; border-color: {tint2}; color: {purple_ink}; }}
/* 圆形按钮的圆角必须正好是边长的一半，超出会被 Qt 判为非法并画成直角 */
QPushButton[btn="roundSm"] {{ border-radius: 13px; padding: 0px; color: {ink2}; }}
QPushButton[btn="roundSm"]:hover {{ background: {tint}; border-color: {tint2}; color: {purple_ink}; }}
/* 可切换的圆钮（标签栏上的布局切换）：按下态用紫色底提示当前处于该布局 */
QPushButton[btn="roundSm"]:checked {{ background: {tint}; border-color: {tint2}; color: {purple_ink}; }}
QPushButton[btn="roundSm"]:disabled {{ color: {ink3}; }}
/* 编码区横/纵切换只保留图标，静止、悬停、按下和选中均融入标签栏背景。 */
QPushButton[layoutToggle="true"],
QPushButton[layoutToggle="true"]:hover,
QPushButton[layoutToggle="true"]:pressed,
QPushButton[layoutToggle="true"]:checked {{
    background: transparent; border: none; border-radius: 0px; padding: 0px;
}}
/* 虚线添加按钮（+ 添加维度 / + 添加编码） */
QPushButton[btn="dashed"] {{
    background: transparent; border: 1px dashed {line_strong}; border-radius: 10px;
    padding: 7px 16px; color: {purple_ink}; font-weight: {fw_semibold};
}}
QPushButton[btn="dashed"]:hover {{ background: {tint}; border-color: {purple}; }}
QPushButton[btn="dashed"]:pressed {{ background: {tint2}; }}
QPushButton[btn="dashed"]:disabled {{ background: transparent; border: 1px dashed {line}; color: {ink3}; }}
QPushButton[btn="ghost"]:disabled {{ background: transparent; border: none; color: {ink3}; }}
QPushButton[btn="play"] {{ background: {purple}; border-color: transparent; border-radius: 18px; padding: 0px; }}
QPushButton[btn="play"]:hover {{ background: {purple_ink}; }}
QPushButton[btn="play"]:pressed {{ background: {purple}; }}

/* ---------- 输入 ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
    background: {card}; border: 1px solid {line}; border-radius: 10px;
    padding: 7px 10px; selection-background-color: {tint2}; selection-color: {ink1};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {purple}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ background: {disabled_bg}; color: {ink3}; }}
QLineEdit[invalid="true"], QSpinBox[invalid="true"] {{ border-color: {danger}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url("{arrow}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background: {card}; border: 1px solid {line}; border-radius: 8px;
    padding: 4px; selection-background-color: {tint}; selection-color: {ink1};
}}
/* 药丸形下拉（播放速度）：收起时像 chip，展开时是圆角菜单 */
QComboBox[pill="true"] {{
    background: {tint}; border: 1px solid {tint2}; border-radius: 15px;
    padding: 0px 6px 0px 12px; color: {purple_ink}; font-size: {fs_small}px; font-weight: {fw_semibold};
}}
QComboBox[pill="true"]:hover {{ background: {tint2}; }}
QComboBox[pill="true"]:focus {{ border-color: {purple}; }}
QComboBox[pill="true"]::drop-down {{ border: none; width: 20px; }}
QComboBox[pill="true"]::down-arrow {{ image: url("{arrow_accent}"); width: 12px; height: 12px; }}
/* 下拉弹层统一沿用菜单样式：靠 view 的 objectName 命中（经 QComboBox 后代选择器写 ::item 不生效） */
QListView#speedPopup, QListView#cellPopup, QListView#comboPopup {{
    background: {card}; border: 1px solid {line}; border-radius: {popup_radius}px;
    padding: 4px; outline: none;
}}
/* 下拉的顶层容器默认带方形边框/填充，会在圆角列表外露出一圈直角矩形，抹掉 */
QComboBoxPrivateContainer {{ background: transparent; border: none; }}
QListView#speedPopup::item, QListView#cellPopup::item, QListView#comboPopup::item {{
    border: none; border-radius: 8px; padding: 6px 12px;
    min-height: 16px; color: {ink2}; font-weight: {fw_regular};
}}
QListView#speedPopup::item:hover, QListView#cellPopup::item:hover, QListView#comboPopup::item:hover {{ background: {hover_row}; color: {ink1}; }}
QListView#speedPopup::item:selected, QListView#cellPopup::item:selected, QListView#comboPopup::item:selected {{
    background: {tint}; color: {purple_ink}; font-weight: {fw_semibold};
}}
/* 表格单元格里的编码下拉：贴着单元格尺寸，避免默认内边距撑高行 */
QComboBox[cellEditor="true"] {{ border-radius: 8px; padding: 1px 4px; font-weight: {fw_medium}; }}
QComboBox[cellEditor="true"]::drop-down {{ border: none; width: 18px; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent; border: none; width: 18px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{arrow_up}"); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{arrow}"); width: 10px; height: 10px; }}

/* ---------- 标签页 ---------- */
QTabWidget::pane {{ border: none; background: transparent; }}
QTabWidget#outerTab QTabBar {{ background: {track}; border-radius: 12px; padding: 4px; }}
QTabWidget#outerTab QTabBar::tab {{
    background: transparent; border: 1px solid transparent; border-radius: 9px;
    padding: 7px 18px; margin-right: 4px; color: {ink2}; font-size: {fs_nav}px;
    min-width: {tab_min_width}px; min-height: 20px;
}}
QTabWidget#outerTab QTabBar::tab:hover:!selected {{ color: {ink1}; }}
QTabWidget#outerTab QTabBar::tab:selected {{ background: {card}; border-color: {line}; color: {ink1}; font-weight: {fw_semibold}; }}
QTabWidget#innerTab QTabBar {{ background: transparent; border-bottom: 1px solid {line}; }}
QTabWidget#innerTab QTabBar::tab {{
    background: transparent; border: none;
    padding: 8px 10px 8px 14px; color: {ink2}; font-size: {fs_nav}px; min-height: 20px;
}}
QTabWidget#innerTab QTabBar::tab:hover:!selected {{ color: {ink1}; }}
QTabWidget#innerTab QTabBar::tab:selected {{ color: {ink1}; font-weight: {fw_semibold}; border-bottom: 3px solid {purple}; }}
/* 编码页标签栏（flatUnderline）自己 paint 一条笔直下划线：QSS 的 border-bottom 两端会被拐出小钩 */
QTabWidget#innerTab QTabBar[flatUnderline="true"]::tab {{ border-bottom: none; }}
QTabWidget#innerTab QTabBar[flatUnderline="true"]::tab:selected {{ border-bottom: none; }}
/* 标签页关闭叉：与维度/编码行的删除叉同一观感（透明底、hover 淡紫） */
QPushButton[btn="tabClose"] {{ background: transparent; border: none; border-radius: 10px; padding: 0px; }}
QPushButton[btn="tabClose"]:hover {{ background: {tab_btn_hover}; }}
QPushButton[btn="tabClose"]:pressed {{ background: {tab_btn_press}; }}
/* 标签栏溢出时右侧的两个滚动按钮：背景与箭头都必须显式写。
   不写的话 Qt 会按当前 palette 的 Text（= ink1）整块填色：浅色下发黑、
   深色下发白，箭头淹在里面看不见（主题切换后才重现，故只能靠样式表兜住）。
   悬停/按下底必须用不透明的 tab_btn_hover/tab_btn_press，不能用 tint/tint2：
   深色下 tint 是半透明 rgba，而标签栏本身透明，Qt 换态时是「替换」而非「叠加」
   背景，于是鼠标一悬停按钮就只剩 16% 不透明度，底下的标签字母直接透上来。
   圆角仍取 0：样式表 border-radius 是「裁掉不画」，圆角外像素属未绘制区也会透底。 */
QTabBar QToolButton {{ background: {card}; border: 1px solid {line}; border-radius: 0px; }}
QTabBar QToolButton:hover {{ background: {tab_btn_hover}; border-color: {tab_btn_press}; }}
QTabBar QToolButton:pressed {{ background: {tab_btn_press}; border-color: {tab_btn_press}; }}
QTabBar QToolButton:disabled {{ background: {disabled_bg}; border-color: {line}; }}
QTabBar QToolButton::left-arrow {{ image: url("{arrow_left}"); width: 10px; height: 10px; }}
QTabBar QToolButton::right-arrow {{ image: url("{arrow_right}"); width: 10px; height: 10px; }}
QTabBar QToolButton::left-arrow:disabled, QTabBar QToolButton::right-arrow:disabled {{
    image: none; width: 0px; height: 0px;
}}

/* ---------- 表格 ---------- */
QTableWidget, QTableView {{
    background: {card}; border: none; gridline-color: {line};
    selection-background-color: {tint}; selection-color: {ink1};
}}
/* 分析结果表要四条边都收住：Qt 的 gridline 只画在数据单元格的上沿与左沿，
   行表头左侧、末列右侧本就不属于网格，而上面的 border: none 又把 QFrame 外框
   去了 —— 于是整张表左右两头都是开口的。挂 gridTable 属性补一圈与网格同色的
   外框，只给分析结果表用，编码表仍保持无边观感。 */
QTableWidget[gridTable="true"], QTableView[gridTable="true"] {{ border: 1px solid {line}; }}
QTableWidget::item, QTableView::item {{ border: none; padding: 7px 12px; }}
QTableWidget::item:hover, QTableView::item:hover {{ background: {hover_row}; }}
QTableWidget::item:selected, QTableView::item:selected {{ background: {tint}; color: {ink1}; }}
/* 表头最后一个分区之后的空白区沿用表头底色，否则深色模式会露出默认白底 */
QHeaderView {{ background: {card_sub}; border: none; }}
QHeaderView::section {{
    background: {card_sub}; color: {ink3}; font-size: {fs_small}px; font-weight: {fw_medium};
    padding: 7px 12px; border: none; border-bottom: 1px solid {line};
}}
QTableCornerButton::section {{ background: {card_sub}; border: none; }}

/* ---------- 列表 ---------- */
QListWidget {{ background: transparent; border: none; }}
QListWidget::item {{
    border: 1px solid transparent; border-radius: 8px;
    padding: 7px 10px; color: {ink2};
}}
QListWidget::item:hover {{ background: {hover_row}; }}
QListWidget::item:selected {{ background: {tint}; color: {purple_ink}; font-weight: {fw_semibold}; }}

/* ---------- 语义容器 ---------- */
QWidget[card="true"] {{ background: {card}; border: 1px solid {line}; border-radius: 14px; }}
QWidget[card="true"]:hover {{ border-color: {line_strong}; }}
QWidget[videoStage="true"] {{ background: {stage}; border-radius: 10px; }}
/* 视频占位背景（无视频时 QVideoWidget 的填充色）故意不走 QSS：
   它是带独立 HWND 的原生窗体，样式表要等延迟的 polish 才落画，且 Qt 的重绘不会让
   Win32 客户区失效，主题切换后旧色会一直残留。改由 video_player 直写 palette(Window)
   并强制失效客户区刷新，token 仍为 video_bg。 */
QLabel[timecode="true"] {{ background: transparent; border: none; color: {ink2}; font-size: {fs_small}px; font-weight: {fw_medium}; }}
/* 编码区时间胶囊（起始/当前时间、时间间隔）：白底描边圆角、居中可编辑 */
QLineEdit[timeField="true"] {{
    background: {card}; border: 1px solid {line}; border-radius: 12px;
    padding: 7px 10px; color: {ink1}; font-size: {fs_time}px; font-weight: {fw_semibold};
}}
QLineEdit[timeField="true"]:focus {{ border-color: {purple}; }}
/* 当前时间被手动编辑：紫底紫框提示「本次记录会插入/覆盖表格行」 */
QLineEdit[timeField="true"][edited="true"] {{ border-color: {purple}; background: {tint}; }}
QLineEdit[timeField="true"][invalid="true"] {{ border-color: {danger}; }}
QWidget[codeRow="true"] {{ background: {dim_row}; border: 1px solid transparent; border-radius: 10px; }}
QWidget[codeRow="true"]:hover {{ background: {card}; border-color: {purple}; }}
QWidget[codeRow="true"][selectedCode="true"] {{ background: {tint2}; border-color: {purple}; }}
QWidget[codeRow="true"] QLineEdit {{ background: transparent; border: none; padding: 2px 0px; }}
QWidget[codeRow="true"] QLineEdit:focus {{ background: transparent; border: none; }}
/* 编码维度面板：圆角白底容器；维度名灰底圆角条 */
QWidget[dimPanel="true"] {{ background: {card}; border: 1px solid {line}; border-radius: 14px; }}
QWidget[dimHead="true"] {{ background: {dim_head}; border: none; border-radius: 10px; }}
QLabel[section="true"] {{ background: transparent; color: {ink1}; font-size: {fs_section}px; font-weight: {fw_semibold}; }}
QLabel[notice="true"] {{
    background: {tint}; border: 1px solid {tint2}; border-radius: 10px;
    padding: 7px 12px; color: {ink2};
}}
QLabel[hint="true"] {{ background: transparent; border: none; color: {ink3}; font-size: {fs_small}px; }}
QLabel[chip="true"] {{
    border: 1px solid transparent; border-radius: 8px;
    padding: 5px 10px; color: {ink1}; font-size: {fs_small}px;
}}
QLabel[chip="true"][tone="0"] {{ background: rgba(109,92,231,0.14); border-color: rgba(109,92,231,0.30); }}
QLabel[chip="true"][tone="1"] {{ background: rgba(59,165,93,0.14); border-color: rgba(59,165,93,0.30); }}
QLabel[chip="true"][tone="2"] {{ background: rgba(227,160,8,0.16); border-color: rgba(227,160,8,0.32); }}
QLabel[chip="true"][tone="3"] {{ background: rgba(229,72,77,0.14); border-color: rgba(229,72,77,0.30); }}
QLabel[chip="true"][tone="4"] {{ background: rgba(14,165,233,0.14); border-color: rgba(14,165,233,0.30); }}
QLabel[chip="true"][tone="5"] {{ background: rgba(236,72,153,0.14); border-color: rgba(236,72,153,0.30); }}
QLabel[chip="true"][tone="6"] {{ background: rgba(20,184,166,0.14); border-color: rgba(20,184,166,0.30); }}
QLabel[chip="true"][tone="7"] {{ background: rgba(154,160,174,0.18); border-color: rgba(154,160,174,0.34); }}
/* ---------- AI 助手：对话气泡与角色标签（不走内联 setStyleSheet） ---------- */
QWidget[aiBubble="assistant"] {{ background: {card}; border: 1px solid {line}; border-radius: 12px; }}
QWidget[aiBubble="user"] {{ background: {tint}; border: 1px solid {tint2}; border-radius: 12px; }}
QWidget[aiBubble="error"] {{ background: {card}; border: 1px solid {danger}; border-radius: 12px; }}
QLabel[aiRole="true"] {{ background: transparent; border: none; color: {ink3}; font-size: {fs_small}px; font-weight: {fw_medium}; }}
QLabel[aiBody="true"] {{ background: transparent; border: none; color: {ink1}; font-size: {fs_base}px; }}
QLabel[aiUsage="true"] {{ background: transparent; border: none; color: {ink3}; font-size: {fs_small}px; }}
QGroupBox {{
    background: {card}; border: 1px solid {line}; border-radius: 14px;
    margin-top: 14px; padding: 12px; font-size: {fs_base}px; font-weight: {fw_semibold}; color: {ink1};
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 14px; top: 2px; padding: 0 6px; color: {ink2};
}}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* ---------- 滚动条 / 分割器 ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px 4px; }}
QScrollBar::handle:vertical {{ background: {scroll_thumb}; border-radius: 4px; min-height: 32px; }}
QScrollBar::handle:horizontal {{ background: {scroll_thumb}; border-radius: 4px; min-width: 32px; }}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {scroll_thumb_hover}; }}
/* 两端的箭头按钮整块关掉：除了收尺寸，还要写死 border/background ——
   只给 width/height: 0 时 Qt 仍会画 Fusion 的下沉边框，按下去就在
   透明滚动条上露出一小块带边框的箭头，底下内容透上来。 */
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; border: none; background: transparent; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: {line}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {purple}; }}
/* 编码四格：保留 8px 拖拽热区，静止及悬停透明，按住拖动才显示。 */
QSplitter[encoderGrid="true"]::handle,
QSplitter[encoderGrid="true"]::handle:hover {{ background: transparent; border: none; }}
QSplitter[encoderGrid="true"]::handle:horizontal:pressed {{
    background: {purple}; margin-left: 3px; margin-right: 4px; border: none;
}}
QSplitter[encoderGrid="true"]::handle:vertical:pressed {{
    background: {purple}; margin-top: 3px; margin-bottom: 4px; border: none;
}}
/* 卡片堆叠用的分割器：分割条只是拖拽热区，不该是一条可见的灰带——卡片自带
   向下阴影，灰带紧贴阴影边缘就会被读成「卡片底下压了一块灰色矩形」。
   挂 cardGap 属性后整条热区透明，露出背后的画布底色。 */
QSplitter[cardGap="true"]::handle, QSplitter[cardGap="true"]::handle:hover {{
    background: transparent;
}}

/* ---------- 滑块 ---------- */
QSlider::groove:horizontal {{ background: {track}; height: 6px; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {purple}; height: 6px; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {card}; border: 2px solid {purple};
    width: 14px; height: 14px; margin: -5px 0; border-radius: 8px;
}}
QSlider::groove:vertical {{ background: {track}; width: 6px; border-radius: 3px; }}
QSlider::sub-page:vertical {{ background: {purple}; width: 6px; border-radius: 3px; }}
QSlider::handle:vertical {{
    background: {card}; border: 2px solid {purple};
    width: 14px; height: 14px; margin: 0 -5px; border-radius: 8px;
}}
QSlider::handle:hover {{ background: {tint}; }}
QSlider::handle:pressed {{ background: {tint2}; }}
/* macOS/Fusion 的默认 slider 高度偏窄，圆点会被控件边界裁掉；播放器只给
   两个滑块挂 macOptimized 属性，留足 24px 绘制区并稍微放大手柄。 */
QSlider[macOptimized="true"] {{ min-height: 24px; }}
QSlider[macOptimized="true"]::handle:horizontal {{
    width: 16px; height: 16px; margin: -6px 0; border-radius: 9px;
}}

/* ---------- 菜单 / 提示 ---------- */
QMenuBar {{ background: transparent; }}
QMenuBar::item {{ background: transparent; border-radius: 8px; padding: 6px 10px; }}
QMenuBar::item:selected {{ background: {tint}; color: {purple_ink}; }}
QMenu {{ background: {card}; border: 1px solid {line}; border-radius: {menu_radius}px; padding: 6px; }}
QMenu::item {{ border-radius: 7px; padding: 6px 24px 6px 12px; }}
QMenu::item:selected {{ background: {tint}; color: {purple_ink}; }}
QMenu::separator {{ height: 1px; background: {line}; margin: 6px 8px; }}
/* 标题栏关闭按钮：hover 用语义危险色 */
QPushButton[btn="titleClose"] {{ background: transparent; border: none; border-radius: 8px; padding: 4px 8px; color: {ink2}; }}
QPushButton[btn="titleClose"]:hover {{ background: {danger}; border-color: transparent; color: #FFFFFF; }}
QPushButton[btn="titleClose"]:pressed {{ background: {danger}; border-color: transparent; color: #FFFFFF; }}
/* 关于弹窗 */
QDialog[about="true"] {{ background: transparent; }}
QWidget[aboutLogo="true"] {{ background: {tint}; border: 1px solid {tint2}; border-radius: 22px; }}
QLabel[aboutName="true"] {{ background: transparent; border: none; color: {ink1}; font-size: {fs_title}px; font-weight: {fw_semibold}; }}
QLabel[aboutTag="true"] {{ background: transparent; border: none; color: {ink2}; font-size: {fs_base}px; }}
QLabel[aboutKey="true"] {{ background: transparent; border: none; color: {ink3}; font-size: {fs_small}px; }}
QLabel[aboutVal="true"] {{ background: transparent; border: none; color: {ink1}; font-size: {fs_small}px; font-weight: {fw_medium}; }}
QLabel[aboutCopy="true"] {{ background: transparent; border: none; color: {ink3}; font-size: {fs_small}px; }}
QToolTip {{
    background: {ink1}; color: {canvas}; border: none;
    border-radius: 6px; padding: 6px 8px;
}}
QStatusBar {{ background: transparent; border-top: 1px solid {line}; color: {ink3}; }}
QStatusBar::item {{ border: none; }}
/* 状态栏右端的自动保存指示：saveState 属性由主窗口按保存情况设置，字号比临时消息小一档 */
QLabel[saveState] {{ padding: 0 6px; font-size: {fs_small}px; }}
QLabel[saveState="idle"] {{ color: {ink3}; }}
QLabel[saveState="off"] {{ color: {ink3}; }}
QLabel[saveState="pending"] {{ color: {warn}; }}
QLabel[saveState="saving"] {{ color: {ink2}; }}
QLabel[saveState="saved"] {{ color: {success}; }}
QLabel[saveState="error"] {{ color: {danger}; }}

/* ---------- 勾选 ---------- */
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px; border: 1px solid {line_strong};
    background: {card}; border-radius: 5px;
}}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {purple}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {purple}; border-color: {purple};
}}
"""


class _Theme(QObject):
    themeChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.mode = "light"
        self.font_family = FONT_FALLBACK

    def settings_path(self):
        from config import data_dir
        return os.path.join(data_dir, "ui_settings.json")

    def load_mode(self):
        try:
            with open(self.settings_path(), encoding="utf-8") as f:
                self.mode = json.load(f).get("theme", "light")
        except Exception:
            self.mode = "light"
        if self.mode not in TOKENS:
            self.mode = "light"

    def save_mode(self):
        try:
            with open(self.settings_path(), "w", encoding="utf-8") as f:
                json.dump({"theme": self.mode}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


theme = _Theme()

# 模块级代理：业务代码统一用 theme.xxx 访问，无需关心单例
themeChanged = theme.themeChanged


def load_mode():
    theme.load_mode()


def save_mode():
    theme.save_mode()


def get_mode():
    return theme.mode


def load_fonts():
    """加载内嵌 MiSans，失败回退微软雅黑；返回实际 family 名。"""
    db = QFontDatabase()
    family = FONT_FALLBACK
    present = []
    for fname in _FONT_FILES:
        path = os.path.join(FONTS_DIR, fname)
        if not os.path.exists(path):
            continue
        present.append(path)
        fid = db.addApplicationFont(path)
        if fid != -1 and fname == "MiSans-Regular.ttf":
            families = db.applicationFontFamilies(fid)
            if families:
                family = families[0]
    _register_matplotlib_fonts(present)
    theme.font_family = family
    return family


def font_check():
    """Return the font information used by packaged-app build checks."""
    regular = os.path.join(FONTS_DIR, "MiSans-Regular.ttf")
    return {"family": theme.font_family, "font_file": regular,
            "font_file_exists": os.path.isfile(regular)}


def _register_matplotlib_fonts(paths):
    # QFontDatabase 只对 Qt 生效；matplotlib 有独立的字体表，不注册就找不到 MiSans
    try:
        from matplotlib import font_manager
    except Exception:
        return
    for path in paths:
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass


def tokens():
    return TOKENS[theme.mode]


def build_qss():
    t = dict(tokens())
    t.update(FONT_SIZES)
    from user_state import state
    t["tab_min_width"] = 200 if state.language() == "en-US" else 96
    if state.language() == "en-US":
        for key in ("fs_nav", "fs_section", "fs_base", "fs_small"):
            t[key] -= 1
    t.update(FONT_WEIGHTS)
    # 弹层半径从这里注入样式表，与 rounded_popup_mask / bind_menu_popup 的默认参数同源
    t["popup_radius"] = POPUP_RADIUS
    t["menu_radius"] = MENU_RADIUS
    t["font"] = theme.font_family
    t.update(_arrow_urls())
    return _QSS.format(**t)


_ARROW_SPECS = (
    ("arrow", "chevron-down", "ink3"),
    ("arrow_accent", "chevron-down", "purple_ink"),
    ("arrow_up", "chevron-up", "ink3"),
    ("arrow_left", "chevron-left", "ink3"),
    ("arrow_right", "chevron-right", "ink3"),
)
_arrow_cache = {}


def _arrow_urls():
    """下拉/步进箭头用位图：border 拼三角形在完整样式表下会退化成实心色块。"""
    urls = _arrow_cache.get(theme.mode)
    if urls is None:
        urls = {}
        for name, icon_name, token in _ARROW_SPECS:
            path = os.path.join(APP_DIR, "assets", "%s-%s.png" % (name, theme.mode))
            icon_provider.icon(icon_name, TOKENS[theme.mode][token], 12).pixmap(24, 24).save(path)
            urls[name] = path.replace(os.sep, "/")
        _arrow_cache[theme.mode] = urls
    return urls


def _apply_matplotlib(t):
    try:
        from matplotlib import pyplot as plt
    except Exception:
        return
    plt.rcParams.update({
        "font.sans-serif": [theme.font_family, FONT_FALLBACK, "Microsoft YaHei", "SimHei"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "text.color": t["ink1"],
        "axes.edgecolor": t["line"],
        "axes.labelcolor": t["ink2"],
        "xtick.color": t["ink3"],
        "ytick.color": t["ink3"],
        "grid.color": t["line"],
        "grid.alpha": 0.6,
        "figure.facecolor": t["card"],
        "axes.facecolor": t["card"],
        "savefig.facecolor": t["card"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        # 图表文字与 UI 同步上调一档（matplotlib 默认 10pt → 11pt）。
        # 画布尺寸由布局决定、与字号无关，且两处图表（序列图 / 树状图）都走
        # tight_layout 且标签短（编码码、G1…），放大只会小幅压缩绘图区，不会顶破版面。
        "font.size": 11,
    })


def sync_figure(fig):
    """同步长驻画布的底色到当前主题。

    rcParams 只在 Figure 构造时生效，clear() 也不重置 figure.facecolor，
    所以主题切换后必须显式改，否则深色模式下图表仍是白底。
    坐标区一起改：长驻画布上「只换主题不重建坐标轴」的路径（占位图）里，
    axes 的底色是上一次主题下从 rcParams 取的，不改就会在图中央留一块旧底色。
    """
    card = tokens()["card"]
    fig.set_facecolor(card)
    for axes in fig.axes:
        axes.set_facecolor(card)


def apply_theme(app=None, mode=None):
    app = app or QApplication.instance()
    if mode and mode in TOKENS:
        theme.mode = mode
        theme.save_mode()
    t = tokens()
    app.setStyleSheet(build_qss())
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(t["canvas"]))
    pal.setColor(QPalette.WindowText, QColor(t["ink1"]))
    pal.setColor(QPalette.Base, QColor(t["card"]))
    pal.setColor(QPalette.AlternateBase, QColor(t["card_sub"]))
    pal.setColor(QPalette.Text, QColor(t["ink1"]))
    pal.setColor(QPalette.Button, QColor(t["card"]))
    pal.setColor(QPalette.ButtonText, QColor(t["ink2"]))
    pal.setColor(QPalette.Highlight, QColor(t["tint2"]))
    pal.setColor(QPalette.HighlightedText, QColor(t["ink1"]))
    pal.setColor(QPalette.Link, QColor(t["purple"]))
    pal.setColor(QPalette.PlaceholderText, QColor(t["ink3"]))
    pal.setColor(QPalette.ToolTipBase, QColor(t["ink1"]))
    pal.setColor(QPalette.ToolTipText, QColor(t["canvas"]))
    app.setPalette(pal)
    # 10pt ≈ 13.3px，与 QSS 的 fs_base（14px）同一量级：未被样式表命中的
    # 原生绘制文字（部分代理、对话框默认按钮等）不会明显小于界面正文。
    # 权重显式给 APP_FONT_WEIGHT：不设就是 QFont.Normal(50)，在 MiSans 上会落到
    # Medium 字面（见 FONT_WEIGHTS 注释），正文会莫名偏粗。
    app.setFont(QFont(theme.font_family, 10, APP_FONT_WEIGHT))
    # QComboBox 下拉动画 (UI_AnimateCombo) 与半透明容器叠加时，Windows 上会在
    # 弹出瞬间留下黑色矩形拖影（菜单无此动画故不受影响）；社区验证的直接解法是禁用该动画效应。
    app.setEffectEnabled(Qt.UI_AnimateCombo, False)
    _apply_matplotlib(t)
    theme.themeChanged.emit(theme.mode)


def refresh_style(widget):
    """动态属性变更后调用，使 QSS 重新生效。"""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def no_popup_shadow(window):
    """让弹出窗体（QMenu / QComboBoxPrivateContainer）变成干净的圆角浮层。

    社区验证过的标准配方（Qt5 on Windows）：
    1) FramelessWindowHint 去掉窗体系统原生直角边框；
    2) NoDropShadowWindowHint 禁用 Qt/系统为 Popup 附加的阴影伴随窗体
       （否则圆角外露出一圈直角方形与黑块）；
    3) WA_TranslucentBackground 把窗体改成分层透明，窗体自身不再用
       palette Window 填方形底色，仅保留 QSS 绘制的圆角内容层；
    4) palette.Window 置透明，避免 autoFill 用不透明色填四角。
    必须在原生窗体创建前（首次 showPopup 之前）调用才生效。
    """
    window.setWindowFlags(window.windowFlags()
                          | Qt.FramelessWindowHint
                          | Qt.NoDropShadowWindowHint)
    window.setAttribute(Qt.WA_TranslucentBackground)
    pal = window.palette()
    pal.setColor(QPalette.Window, QColor(0, 0, 0, 0))
    window.setPalette(pal)
    return window


class _RoundedMaskFilter(QObject):
    """把弹层窗体的可见区域物理裁成圆角（兜底手段）。

    不依赖 DPI 缩放、DWM 合成或弹出动画：窗体区域本身就是圆角矩形，
    圆角外不可能再出现任何方形/黑色像素。Show/Resize 时同步更新裁剪。
    """

    def __init__(self, window, radius):
        super().__init__(window)
        self._window = window
        self._radius = radius

    def eventFilter(self, obj, ev):
        if obj is self._window and ev.type() in (QEvent.Show, QEvent.Resize):
            self._apply()
        return False

    def _apply(self):
        rect = self._window.rect()
        if rect.isEmpty():
            return
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), self._radius, self._radius)
        self._window.setMask(QRegion(path.toFillPolygon().toPolygon()))


def rounded_popup_mask(window, radius=POPUP_RADIUS):
    """兜底：把弹层容器窗体区域裁成圆角，彻底杜绝圆角外的方形/黑线残留。

    默认半径取 POPUP_RADIUS，与样式表里弹层 border-radius 用的是同一个常量，
    使裁剪边正好落在圆角描边外缘。QMenu 的圆角与此不同，走 bind_menu_popup。
    """
    filt = _RoundedMaskFilter(window, radius)
    window.installEventFilter(filt)
    # installEventFilter 不持有 Python 端引用；包装对象被 GC 后 eventFilter 重写会静默失效，
    # 必须在窗体上留强引用。
    window._rounded_mask_filter = filt
    return window


def bind_menu_popup(menu):
    """QMenu 的弹层配方：去原生阴影 + 按 MENU_RADIUS 裁圆，绘制与裁剪同源。

    菜单圆角与下拉列表本就不同值，所以配对收在这里，调用点不传半径 ——
    全仓库的弹层半径只有 POPUP_RADIUS 与 MENU_RADIUS 两个数。
    """
    return rounded_popup_mask(no_popup_shadow(menu), MENU_RADIUS)


class _ComboPopupBinder(QObject):
    """把弹层配方绑定到 QComboBox 当前容器，容器被重建时自动重新施加。

    QComboBox 的父窗体变化（如组装进布局/页签）时 Qt 会重建
    QComboBoxPrivateContainer，初始化时施加的 no_popup_shadow / mask 会随旧容器
    一起丢失——这是“菜单干净、唯独下拉残留”的根因。这里挂在稳定的 combo 上，
    在容器原生窗体创建前（combo Show/ParentChange）重新解析并施加。
    """

    def __init__(self, combo, radius):
        super().__init__(combo)
        self._combo = combo
        self._radius = radius
        self._container = None
        combo.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self._combo and ev.type() in (QEvent.Show, QEvent.Polish,
                                                QEvent.ParentChange):
            self.rebind()
        return False

    def rebind(self):
        view = self._combo.view()
        if view is None:
            return
        container = view.parentWidget().window()
        if container is self._container:
            return
        self._container = container
        no_popup_shadow(container)
        rounded_popup_mask(container, self._radius)


def bind_combo_popup(combo, radius=POPUP_RADIUS):
    """对 QComboBox 下拉容器持续施加圆角弹层配方（容器重建后自动重新施加）。"""
    view = combo.view()
    if view is not None and not view.objectName():
        # QComboBox 的弹出列表不是收起态控件的普通子项；给它稳定名称，才能让
        # QSS 命中项目统一的圆角菜单样式，而不是回退到带直角黑框的原生列表。
        view.setObjectName("comboPopup")
    binder = _ComboPopupBinder(combo, radius)
    binder.rebind()
    # 同 rounded_popup_mask：留住 Python 端引用，防 GC 使 eventFilter 失效
    combo._popup_binder = binder
    return combo


def set_invalid(widget, invalid):
    widget.setProperty("invalid", bool(invalid))
    refresh_style(widget)


_icon_bindings = []
_shadow_bindings = []


def bind_icon(widget, name, size=16, color="ink2"):
    """设置图标并登记，主题切换时自动按新配色重绘。

    color 可以是 token 名（默认 ink2，随主题变化）或固定色值（如主按钮上的 #FFFFFF）。
    """
    entry = (weakref.ref(widget), (name, size, color))
    _icon_bindings.append(entry)
    _paint_icon(entry)


def _paint_icon(entry):
    ref, (name, size, color) = entry
    widget = ref()
    if widget is None:
        return False
    resolved = color if color.startswith("#") else tokens()[color]
    icon = icon_provider.icon(name, resolved, size)
    if isinstance(widget, QLabel):
        widget.setPixmap(icon.pixmap(size, size))
    else:
        widget.setIcon(icon)
    return True


def set_icon(widget, name, size=16, color="ink2"):
    """切换已登记控件的图标（播放/暂停这类运行时变化）。

    必须原地替换绑定条目：若追加新条目，主题切换时旧条目会把控件重绘回旧图标。
    """
    for i, (ref, _args) in enumerate(_icon_bindings):
        if ref() is widget:
            entry = (ref, (name, size, color))
            _icon_bindings[i] = entry
            _paint_icon(entry)
            return
    bind_icon(widget, name, size, color)


def _repaint_bindings(_mode=None):
    _icon_bindings[:] = [e for e in _icon_bindings if _paint_icon(e)]
    refs = _shadow_bindings[:]
    _shadow_bindings.clear()
    for ref in refs:
        widget = ref()
        if widget is not None:
            apply_card_shadow(widget)


def apply_card_shadow(widget):
    """卡片弱阴影；仅顶层卡片调用（全窗口 <=10 个）。"""
    t = tokens()
    r, g, b = (int(x) for x in t["shadow_rgb"].split(","))
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(28)
    eff.setOffset(0, 6)
    eff.setColor(QColor(r, g, b, t["shadow_alpha"]))
    widget.setGraphicsEffect(eff)
    _shadow_bindings.append(weakref.ref(widget))


def paint_card_shadow(host, card, radius=14, blur=12, offset=4):
    """在宿主控件的 paintEvent 里，为它的一张子卡片手绘柔和投影。

    含原生 QVideoWidget 的卡片不能用 apply_card_shadow：QGraphicsEffect 会把整棵子树
    画进位图再合成，HWND 进不了那张位图（与 fade_in 里记录的是同一个坑）。改由父件在
    卡片外圈逐层描半透明圆角矩形，不引入任何图形效果，视频画面照常直出。
    投影只能画在 host 布局留出的空白边距内（超出被父件裁掉），故 blur 须小于边距。
    """
    if card is None or not card.isVisible():
        return
    t = tokens()
    r, g, b = (int(x) for x in t["shadow_rgb"].split(","))
    rect = QRectF(card.geometry())
    painter = QPainter(host)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    steps = 6
    # 叠加是 1-∏(1-a) 而非相加，每层取 token 浓度的 1.8/steps，叠完才刚好到 shadow_alpha
    base = t["shadow_alpha"] * 1.8 / steps
    for i in range(steps, 0, -1):
        grow = blur * i / steps
        painter.setBrush(QColor(r, g, b, max(1, int(round(base * (1.0 - 0.6 * i / steps))))))
        painter.drawRoundedRect(
            QRectF(rect.left() - grow, rect.top() - grow + offset,
                   rect.width() + 2 * grow, rect.height() + 2 * grow),
            radius + grow, radius + grow)
    painter.end()


class Canvas(QWidget):
    """带顶部氛围光的画布容器，替代硬编码背景色。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        ref = weakref.ref(self)
        theme.themeChanged.connect(lambda _mode: ref() and ref().update())

    def paintEvent(self, event):
        t = tokens()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(t["canvas"]))
        g = QRadialGradient(self.width() * 0.85, -self.height() * 0.15,
                            max(self.width(), 600) * 0.9)
        g.setColorAt(0.0, QColor(t["atmo_inner"]))
        g.setColorAt(1.0, QColor(t["atmo_outer"]))
        p.fillRect(self.rect(), g)
        p.end()


# ---------- 下划线式标签栏 ----------
TAB_CLOSE_SIZE = 20    # 标签页关闭叉边长，与维度/编码行的删除叉同一观感
TAB_ADD_SIZE = 26      # “+ 新建页”按钮边长（roundSm 的圆角正好是它的一半）
TAB_ADD_GAP = 10       # 最后一个标签与“+”的间距
TAB_UNDERLINE = 3      # 选中页下划线粗细
TAB_TEXT_PAD = 34      # fit_text 标签左右留白：QSS 的 padding(14+10) 再加缓冲
TAB_QSS_WEIGHT = 500   # Qt 6 QFont 使用 100–900 标尺；500 对应 MiSans Medium


def tab_font(widget):
    """按 QSS 给 tab 定的字体量宽度（供 tabSizeHint 与回归脚本共用同一把尺子）。

    tab 上写的 font-size: {fs_nav}px / font-weight: {fw_semibold} 只在绘制时生效，
    widget.font() 仍是全局的 fs_base/Regular，直接拿它量会低估十几个像素，
    配合 ElideNone 就是两头一起切字。
    """
    font = QFont(widget.font())
    try:
        font.setPixelSize(int(FONT_SIZES["fs_nav"]))
        font.setWeight(TAB_QSS_WEIGHT)
    except Exception:
        pass        # 虚拟函数里抛异常会直接 abort 进程，量不准也比崩掉好
    return font


class UnderlineTabBar(QTabBar):
    """标签栏：选中页下方是一条笔直的下划线；with_add 时把“+ 新建页”贴在最后一个标签右侧。

    两条 Qt 怪癖:
    1) QSS 的 border-bottom 会在 tab 两端拐出小钩（原生 tab 形状所致），故关掉 drawBase
       并自己 paint，同时靠 [flatUnderline="true"] 规则把 QSS 的 border-bottom 清掉；
    2) QTabWidget.resizeEvent 按标签栏 sizeHint 的宽度摆放它（不会撑满整行），所以有“+”
       时要在 sizeHint 里预留位置，否则按钮只能盖在最后一个标签上。
    """

    addRequested = Signal()
    tabClicked = Signal(int)   # 左键在某个标签上松开（选中已由基类完成），供上层挂点击后的选项

    def __init__(self, parent=None, with_add=False, fit_text=False):
        super().__init__(parent)
        self.setProperty("flatUnderline", True)
        self.fit_text = fit_text
        self.add_button = None
        if with_add:
            # 先建按钮再改旗标：setExpanding() 会回调 tabLayoutChange()，那里要用到
            # add_button（虚拟函数里访问未初始化属性会直接崩掉进程）
            self.add_button = QPushButton(self)
            self.add_button.setFixedSize(TAB_ADD_SIZE, TAB_ADD_SIZE)
            self.add_button.setProperty("btn", "roundSm")
            self.add_button.setToolTip("新建编码个体")
            bind_icon(self.add_button, "plus", 14)
            self.add_button.clicked.connect(lambda _=False: self.addRequested.emit())
            self.add_button.raise_()
        self.setDrawBase(False)
        self.setExpanding(False)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # 叉按钮、“+”按钮是标签栏的子控件，点它们时事件落不到这里，不会误报点击标签
        if event.button() == Qt.LeftButton:
            index = self.tabAt(event.pos())
            if index >= 0:
                self.tabClicked.emit(index)

    def tabLayoutChange(self):
        self.place_add_button()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.place_add_button()

    def tabSizeHint(self, index):
        """fit_text：标签宽至少容得下全文。

        QSS 下的默认 tabSizeHint 会把标签压成等宽，长名称（如“时间窗口频次 (105)”）
        两头都会被截掉；这里按 QSS 实际生效的字体量出宽度兜底，配合 ElideNone 保证完整显示。
        """
        size = super().tabSizeHint(index)
        # getattr 而非直读：tabSizeHint 可能在 __init__ 完成前就被 Qt 回调，
        # 虚拟函数里抛 Python 异常会直接 abort 整个进程
        if getattr(self, "fit_text", False) and 0 <= index < self.count():
            need = QFontMetrics(tab_font(self)).horizontalAdvance(self.tabText(index))
            size.setWidth(max(size.width(), need + TAB_TEXT_PAD))
        return size

    def sizeHint(self):
        hint = super().sizeHint()
        if self.add_button is not None:
            hint.setWidth(hint.width() + TAB_ADD_SIZE + TAB_ADD_GAP)
        return hint

    def place_add_button(self):
        """把“+”放到最后一个标签右侧，垂直方向与标签对齐"""
        if self.add_button is None:
            return
        last = self.count() - 1
        if last >= 0:
            rect = self.tabRect(last)
            x = rect.right() + 1 + TAB_ADD_GAP // 2
            y = rect.center().y() - TAB_ADD_SIZE // 2
        else:
            x = 6
            y = (self.height() - TAB_ADD_SIZE) // 2
        scroll_buttons = [button for button in self.findChildren(QToolButton)
                          if button.isVisible() and button.width() > 0]
        right_edge = min((button.geometry().left() for button in scroll_buttons),
                         default=self.width() - 2)
        x = max(0, min(x, right_edge - TAB_ADD_SIZE - TAB_ADD_GAP // 2))
        self.add_button.move(x, max(0, y))
        self.add_button.raise_()

    def paintEvent(self, event):
        super().paintEvent(event)
        index = self.currentIndex()
        if index < 0:
            return
        rect = self.tabRect(index)
        if rect.width() <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(rect.left(), rect.bottom() - TAB_UNDERLINE + 1,
                         rect.width(), TAB_UNDERLINE, QColor(tokens()["purple"]))
        painter.end()


def install_underline_tab_bar(tabs, with_add=False, fit_text=False):
    """把 QTabWidget 的标签栏换成 UnderlineTabBar，返回新标签栏。

    必须在加入任何页之前调用；QTabWidget::setTabBar 会按样式提示重设 expanding，
    不在这里再关掉一次的话标签会被拉伸填满整行。
    """
    bar = UnderlineTabBar(with_add=with_add, fit_text=fit_text)
    tabs.setTabBar(bar)
    bar.setExpanding(False)
    if fit_text:
        # 宁可用滚动按钮也不要把名字缩成“…”
        bar.setElideMode(Qt.ElideNone)
        bar.setUsesScrollButtons(True)
    return bar


def has_video_surface(widget):
    """子树里是否含自带窗口的视频控件（QVideoWidget）。

    只认 QVideoWidget 而不看 WA_NativeWindow：那个属性会被“祖先也是原生”机制连带
    标上（子窗体一创建，一路往上的祖先全变 True），拿它做判条件会把四个页面一起
    误杀；而 matplotlib 画布只是挂了个属性、并没有真的 HWND，淡入对它们正常。
    """
    if widget.inherits("QVideoWidget"):
        return True
    return any(c.inherits("QVideoWidget") for c in widget.findChildren(QWidget))


def fade_in(widget, msec=140):
    """标签切换淡入微动效；含视频窗体的页面直接跳过。

    QGraphicsOpacityEffect 是把控件画进位图再按透明度合成，原生 HWND 画不进去。
    实测（真屏拓屏，量编码区相邻像素灰度差）：对含视频窗体的页面做动效后，整块编码
    区从稳定态的 (亮度 251.6, 锐度 0.67) 掉到 (245.3, 0.00)——就是一张画布底色的空白，
    一直持续到 250ms 开外（动效设计时长只有 140ms）；同期视频窗体一直满亮度正常显示。
    一边空着一边亮着，内容到动效结束才“一下长回来”，就是用户看到的“视频区在缩放”。
    切页本身已够醒目，这一页不为它保留半秒不到的动效。
    """
    if has_video_surface(widget):
        return
    from PySide6.QtCore import QPropertyAnimation
    from PySide6.QtWidgets import QGraphicsOpacityEffect
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(msec)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start()
    widget.setProperty("_fade_anim", anim)


theme.themeChanged.connect(_repaint_bindings)
