# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""关于弹窗 AboutDialog：仿 Qoder 关于窗的居中卡片式软件信息展示。"""

import datetime
import os
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

import build_stamp
import icon_provider
import integrity
import theme
from config import (APP_COPYRIGHT, APP_LEGAL_NOTE, APP_NAME, APP_TAGLINE,
                    APP_VERSION, get_app_dir)


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=get_app_dir(), capture_output=True,
                             timeout=2, text=True)
        return out.stdout.strip() or "dev"
    except Exception:
        return "dev"


def _build_date():
    app_dir = get_app_dir()
    newest = 0.0
    for name in os.listdir(app_dir):
        if name.endswith(".py"):
            newest = max(newest, os.path.getmtime(os.path.join(app_dir, name)))
    return datetime.date.fromtimestamp(newest).isoformat()


def _info_rows():
    """关于页的信息行：发布构建报水印，源码运行报本地现场。

    分成两套而不是同一套里填「dev」，是因为这两组东西的受众不同：拿到 exe 的人
    需要的是能反查到某一次发布的 build_id，而在源码里跑的开发者看到的 build_id
    永远是空串，摆三行空白不如把 git 提交与文件时间摊开。

    完整性行只在发布态出现：dev 下 `self_result()` 恒为 `dev`，摆出来是噪声。
    这一行也是「只标记不拦启动」立场的落点——校验失败照样开窗口，只是把状态写脸上。
    """
    if build_stamp.is_release():
        return (("版本", APP_VERSION),
                ("构建标识", build_stamp.BUILD_ID),
                ("构建时间", build_stamp.BUILD_TIME),
                ("完整性", integrity.label(integrity.self_result())))
    return (("版本", APP_VERSION),
            ("提交", _git_commit()),
            ("构建日期", _build_date()))


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("about", True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedSize(420, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 36)
        card = QWidget(self)
        card.setProperty("card", True)
        theme.apply_card_shadow(card)
        outer.addWidget(card)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 16, 24, 20)
        lay.setSpacing(0)

        top = QHBoxLayout()
        top.addStretch(1)
        close_btn = QPushButton()
        close_btn.setProperty("btn", "round")
        close_btn.setFixedSize(30, 30)
        close_btn.setToolTip("关闭")
        theme.bind_icon(close_btn, "close", 14)
        close_btn.clicked.connect(self.accept)
        top.addWidget(close_btn)
        lay.addLayout(top)

        logo_tile = QWidget()
        logo_tile.setProperty("aboutLogo", True)
        logo_tile.setFixedSize(88, 88)
        tile_lay = QVBoxLayout(logo_tile)
        tile_lay.setContentsMargins(0, 0, 0, 0)
        tile_lay.setSpacing(0)
        tile_lay.setAlignment(Qt.AlignCenter)
        logo = QLabel()
        theme.bind_icon(logo, "tab-data", 44, "purple")
        tile_lay.addWidget(logo, 0, Qt.AlignCenter)
        lay.addWidget(logo_tile, 0, Qt.AlignHCenter)
        lay.addSpacing(18)

        name = QLabel(APP_NAME)
        name.setProperty("aboutName", True)
        name.setAlignment(Qt.AlignCenter)
        lay.addWidget(name)
        lay.addSpacing(6)
        tag = QLabel(APP_TAGLINE)
        tag.setProperty("aboutTag", True)
        tag.setAlignment(Qt.AlignCenter)
        lay.addWidget(tag)
        lay.addSpacing(36)

        info = QGridLayout()
        info.setContentsMargins(40, 0, 40, 0)
        info.setHorizontalSpacing(16)
        info.setVerticalSpacing(14)
        rows = _info_rows()
        for r, (key, val) in enumerate(rows):
            k = QLabel(key)
            k.setProperty("aboutKey", True)
            k.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v = QLabel(val)
            v.setProperty("aboutVal", True)
            v.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            info.addWidget(k, r, 0)
            info.addWidget(v, r, 1)
        lay.addLayout(info)
        lay.addStretch(1)

        # 版权声明与根目录 LICENSE/NOTICE 同口径（两个串都住在 config.py）；打包时
        # 这两个文件会被拷进发行目录，少了这句提示，用户不知道手上那份应该留着什么。
        copy = QLabel("%s\n%s" % (APP_COPYRIGHT, APP_LEGAL_NOTE))
        copy.setProperty("aboutCopy", True)
        copy.setWordWrap(True)
        copy.setAlignment(Qt.AlignCenter)
        lay.addWidget(copy)
