# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""线性图标提供器：内嵌 SVG 路径，按颜色/尺寸渲染为 QIcon 并缓存。"""

import os

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_SVG_WRAP = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">{body}</svg>'
)

ICON_PATHS = {
    "play": '<path d="M8 5.5v13l10-6.5z" fill="currentColor" stroke="none"/>',
    "pause": '<rect x="7" y="5" width="3.5" height="14" rx="1.2" fill="currentColor" stroke="none"/>'
             '<rect x="13.5" y="5" width="3.5" height="14" rx="1.2" fill="currentColor" stroke="none"/>',
    "skip-back": '<path d="M11 6.5v11L3.5 12z" fill="currentColor" stroke="none"/>'
                 '<path d="M20.5 6.5v11L13 12z" fill="currentColor" stroke="none"/>',
    "skip-forward": '<path d="M13 6.5v11l7.5-5.5z" fill="currentColor" stroke="none"/>'
                    '<path d="M3.5 6.5v11L11 12z" fill="currentColor" stroke="none"/>',
    "replay": '<path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
    "jump": '<path d="M4 12h11"/><path d="M11.5 8.5L15 12l-3.5 3.5"/><path d="M19.5 5.5v13"/>',
    "plus": '<path d="M12 5v14"/><path d="M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "close": '<path d="M6 6l12 12"/><path d="M18 6L6 18"/>',
    "edit": '<path d="M4 20h4l10.5-10.5a2.828 2.828 0 1 0-4-4L4 16z"/>'
            '<path d="M13.5 6.5l4 4"/>',
    "maximize": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    "restore": '<rect x="9" y="9" width="11" height="11" rx="1.5"/>'
               '<path d="M5 15H4.5A1.5 1.5 0 0 1 3 13.5v-9A1.5 1.5 0 0 1 4.5 3h9A1.5 1.5 0 0 1 15 4.5V5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.5h.01"/>',
    "chevron-down": '<path d="M6 9l6 6 6-6"/>',
    "chevron-up": '<path d="M6 15l6-6 6 6"/>',
    # 左右向：标签栏溢出时的滚动按钮要用，与上下向同一套笔画
    "chevron-left": '<path d="M15 6l-6 6 6 6"/>',
    "chevron-right": '<path d="M9 6l6 6-6 6"/>',
    "import": '<path d="M12 3v10"/><path d="M8 9l4 4 4-4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "export": '<path d="M12 13V3"/><path d="M8 7l4-4 4 4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>'
            '<path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/>',
    # 复制：两块错位的圆角矩形，给「复制命令」这类剪贴板动作用
    "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/>'
            '<path d="M15 6.5V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h1.5"/>',
    "calculate": '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 7h8"/>'
                 '<path d="M8 12h2"/><path d="M14 12h2"/><path d="M8 16h2"/><path d="M14 16h2"/>',
    "chart-bar": '<path d="M5 20v-6"/><path d="M11 20V6"/><path d="M17 20v-9"/><path d="M3 20h18"/>',
    "chart-line": '<path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/>',
    "cluster": '<circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="19" r="2.5"/>'
               '<circle cx="19" cy="19" r="2.5"/><path d="M10.8 7.2L6.2 16.8"/>'
               '<path d="M13.2 7.2l4.6 9.6"/><path d="M7.5 19h9"/>',
    "sequence": '<path d="M3 7h13"/><path d="M13 4l3 3-3 3"/><path d="M21 17H8"/>'
                '<path d="M11 14l-3 3 3 3"/>',
    "video": '<rect x="3" y="6" width="12" height="12" rx="2"/><path d="M15 10l6-3v10l-6-3"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
            '<path d="M14 2v6h6"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "tab-group": '<circle cx="9" cy="8" r="3"/><path d="M4 20v-1a5 5 0 0 1 10 0v1"/>'
                 '<circle cx="17" cy="9" r="2.5"/><path d="M16 14.6A4.5 4.5 0 0 1 21 19v1"/>',
    "tab-data": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18"/>'
                '<path d="M9 10v10"/>',
    # 个体模式：单人与群体同一画风（一个头 + 一个身体）
    "individual": '<circle cx="12" cy="8" r="3.4"/>'
                  '<path d="M5.5 20v-1a6.5 6.5 0 0 1 13 0v1"/>',
    # 网络：四个节点三条边，给交互网络 / ENA / ONA 类入口用
    "network": '<circle cx="5.5" cy="6" r="2.2"/><circle cx="18.5" cy="6" r="2.2"/>'
               '<circle cx="12" cy="18" r="2.2"/><path d="M7.6 6.9l8.8 0"/>'
               '<path d="M6.6 8l4.3 8"/><path d="M17.4 8l-4.3 8"/>',
    # 进度：三个依次变亮的点，训练 / 长任务的就地反馈
    "progress": '<circle cx="5.5" cy="12" r="1.8" fill="currentColor" stroke="none"/>'
                '<circle cx="12" cy="12" r="1.8" fill="currentColor" stroke="none" opacity="0.65"/>'
                '<circle cx="18.5" cy="12" r="1.8" fill="currentColor" stroke="none" opacity="0.35"/>',
    # 编码区布局：左右两栏 / 上下两栏
    "layout-h": '<rect x="3" y="5" width="7.5" height="14" rx="1.5"/>'
                '<rect x="13.5" y="5" width="7.5" height="14" rx="1.5"/>',
    "layout-v": '<rect x="5" y="3" width="14" height="7.5" rx="1.5"/>'
                '<rect x="5" y="13.5" width="14" height="7.5" rx="1.5"/>',
    # AI 助手：机器人头像（天线 + 双眼 + 耳），给外层标签页与档案区用
    "robot": '<rect x="4" y="8" width="16" height="12" rx="3"/>'
             '<path d="M12 4.2V8"/><circle cx="12" cy="3" r="1.3"/>'
             '<circle cx="9" cy="13" r="1.2" fill="currentColor" stroke="none"/>'
             '<circle cx="15" cy="13" r="1.2" fill="currentColor" stroke="none"/>'
             '<path d="M9.5 16.8h5"/><path d="M2 12v3"/><path d="M22 12v3"/>',
    # 发送：纸飞机（对话区发送按钮）
    "send": '<path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4z"/>',
    # 停止：实心方块（流式生成时中止）
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
    # 闪光：四角星 + 小星，给「AI 建议 / 智能」类动作点缀
    "sparkle": '<path d="M11 3l1.7 4.8L17.5 9.5l-4.8 1.7L11 16l-1.7-4.8L4.5 9.5l4.8-1.7z"/>'
               '<path d="M18 15l.8 2.2L21 18l-2.2.8L18 21l-.8-2.2L15 18l2.2-.8z"/>',
    # 密钥：给 API key 输入框（设置区）
    "key": '<circle cx="7.5" cy="15.5" r="3.5"/><path d="M10 13L20 3"/>'
           '<path d="M16.5 6.5l2 2"/><path d="M14 9l2 2"/>',
    # 连接：链环，给「测试连接」按钮
    "link": '<path d="M9.5 14.5l5-5"/>'
            '<path d="M11 6.5l1.6-1.6a4 4 0 0 1 5.7 5.7L16.5 12.4"/>'
            '<path d="M13 17.5l-1.6 1.6a4 4 0 0 1-5.7-5.7L7.5 11.6"/>',
    # 删除：垃圾桶（删除智能体档案）
    "trash": '<path d="M4 7h16"/><path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>'
             '<path d="M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13"/>'
             '<path d="M10 11v6"/><path d="M14 11v6"/>',
    # 魔杖：新建智能体向导
    "wand": '<path d="M4 20l9.5-9.5"/><path d="M13 4l.9 2.1L16 7l-2.1.9L13 10l-.9-2.1L10 7l2.1-.9z"/>'
            '<path d="M19 10l.6 1.4L21 12l-1.4.6L19 14l-.6-1.4L17 12l1.4-.6z"/>',
}

_cache = {}


def icon(name, color="#5C6070", size=20):
    key = (name, color, size)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    body = ICON_PATHS.get(name)
    if body is None:
        return QIcon()
    svg = _SVG_WRAP.format(body=body).replace("currentColor", QColor(color).name())
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    px = size * 2  # @2x 保证高 DPI 清晰
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    renderer.render(painter)
    painter.end()
    result = QIcon(QPixmap.fromImage(img))
    _cache[key] = result
    return result


def clear_cache():
    _cache.clear()
