# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""时间轴共用件：三个时间胶囊 + 「骨架 + 格位」行时间模型 + 表格量宽工具。

数据编码页与「小组交互」页都要求同一套录入语义：起始时间/当前时间/时间间隔三个
胶囊、顺序记录永远接在时间轴骨架尾部、手动改当前时间后可插入或覆盖某一行、删掉一
个顺序行只是在骨架上留一个隐藏格。两套面板各写一遍这种带不变式的代码迟早分叉，
所以骨架算法、胶囊控件与量宽三件事集中在这里维护一次。

- ``TimeControls``：三个时间胶囊 + 播放头跟随 + 非法态描红（两页观感一致）。
- ``TimeAxis``   ：行时间 / 格位两个平行表的算法层，只读写 host 上的三个属性，
                   host 换掉行列表（读档、测试直接赋值）也立刻生效。
- ``apply_column_widths`` 等：表格定宽列 + 内容量宽，两页共用同一口径。

``row_times[i] is None`` = 顺序行（时间由格位推算）；``row_positions[i] is None``
= 固定时间点行（时间钉死、不占格）。这条不变式只由 ``align_row_states`` 维护。
"""

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QIntValidator, QRegularExpressionValidator
from PySide6.QtWidgets import (QGridLayout, QHeaderView, QLineEdit, QLabel, QTableWidget,
                             QWidget)
from PySide6.QtCore import Signal

import theme

# 时间间隔（秒）上限：留出到 1 小时，兼容长视频的粗粒度编码
INTERVAL_MAX = 3600
# 时间/计时列宽度下限：实际宽还看字体量出来多少（取两者的大值）
TIME_COL_MIN_WIDTH = 100
ELAPSED_COL_MIN_WIDTH = 60
# 两列的排版基准文本：时间列要容下 HH:MM:SS，计时列按四位数预留
TIME_COL_SAMPLE = "00:00:00"
ELAPSED_COL_SAMPLE = "0000"
# 单元格左右内边距（QSS 里是 12px×2）加选中框/省略号余量
CELL_H_PADDING = 32
# 表格左侧固定列数：时间 + 计时（取代原来的序号列）
FIXED_COL_COUNT = 2
# 编码/注释列最小宽度：再窄也要能看清两位编码与几个汉字（不足则表格横向滚动）
CODE_COL_MIN_WIDTH = 96
# 编码/注释列按内容放宽的上限：长注释也只撑到这里，再多留给横向滚动
CONTENT_COL_MAX = 200
# 单元格下拉弹窗最小宽度（弹窗可比单元格宽，否则编码项挤成一列竖条）
POPUP_MIN_WIDTH = 160
# 撤销栈深度上限：一次快照是一份行数据，堆到底会吃掉不少内存
HISTORY_LIMIT = 100


# ------------------------------------------------------------------ 时间文本 ↔ 秒数

def parse_time(text):
    """解析 时:分:秒 / 分:秒 / 秒 文本为总秒数，非法返回 None"""
    text = str(text).strip()
    if not text:
        return None
    parts = text.split(':')
    if len(parts) > 3 or any(not p.isdigit() for p in parts):
        return None
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return total


def format_seconds(seconds):
    """总秒数 → HH:MM:SS：三个时间胶囊与表格时间列共用这一种写法（支持超过一小时）"""
    h, rem = divmod(max(int(seconds), 0), 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def make_time_field(text="00:00:00", parent=None):
    """时间胶囊输入框：只放行 时:分:秒 形状的输入，居中对齐"""
    edit = QLineEdit(text, parent)
    edit.setProperty("timeField", True)
    edit.setAlignment(Qt.AlignCenter)
    edit.setValidator(QRegularExpressionValidator(
        QRegularExpression(r"\d{0,2}(:\d{0,2}){0,2}"), edit))
    return edit


# ------------------------------------------------------------------ 三个时间胶囊

class TimeControls(QWidget):
    """起始时间 / 当前时间 / 时间间隔 三个胶囊：录入非法只描红，不牵动表格。

    「当前时间」默认跟随视频播放头；用户一旦编辑过就冻结跟随（``edited`` 置真、
    描紫色描边）。手动时间记录后推进一个间隔并继续冻结；需要回到播放头时由宿主
    调 ``reset_to_playback()`` 交还默认态。
    非法或输入中时 ``start_seconds()`` / ``interval()`` 回退到上一次合法值，避免
    表格随半个输入抖动——这条回退策略两页都要，故收在这里。
    """

    #: 起始时间/间隔文本变化（宿主据此重画表格）
    params_changed = Signal()
    #: 用户开始手动编辑当前时间
    timepoint_edited = Signal()

    def __init__(self, parent=None, captions=("起始时间", "当前时间", "时间间隔")):
        super().__init__(parent)
        self._start_fallback = 0
        self._unit_fallback = 1
        self._edited = False
        self.video_player = None
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.start_edit = make_time_field("00:00:00")
        self.timepoint_edit = make_time_field("00:00:00")
        self.timepoint_edit.textEdited.connect(self.mark_timepoint_edited)
        self.unit_edit = QLineEdit("1")
        self.unit_edit.setProperty("timeField", True)
        self.unit_edit.setAlignment(Qt.AlignCenter)
        self.unit_edit.setValidator(QIntValidator(1, INTERVAL_MAX, self))
        for col, (cap, edit) in enumerate(zip(captions,
                (self.start_edit, self.timepoint_edit, self.unit_edit))):
            lbl = QLabel(cap)
            lbl.setProperty("hint", True)
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, col)
            grid.addWidget(edit, 1, col)
            grid.setColumnStretch(col, 1)
        self.unit_edit.textChanged.connect(self.validate_unit)
        self.unit_edit.textChanged.connect(self.params_changed.emit)
        self.start_edit.textChanged.connect(self.validate_start)
        self.start_edit.textChanged.connect(self.params_changed.emit)

    # ---------- 起始时间 ----------
    def parse_start_seconds(self):
        """起始时间胶囊 → 总秒数；空或半截输入返回 None"""
        return parse_time(self.start_edit.text())

    def start_seconds(self):
        """时间轴原点：非法或输入中时沿用上一次的合法值，不让表格跳回 0"""
        t = self.parse_start_seconds()
        if t is not None:
            self._start_fallback = t
        return self._start_fallback

    def set_start_seconds(self, seconds):
        self.start_edit.setText(format_seconds(seconds))

    def validate_start(self):
        theme.set_invalid(self.start_edit, self.parse_start_seconds() is None)

    # ---------- 时间间隔 ----------
    def parse_interval_seconds(self):
        """时间间隔胶囊 → 秒数；空/0/超上限返回 None（0 会让全表挤在同一时刻）"""
        text = self.unit_edit.text()
        if text.isdigit() and 1 <= int(text) <= INTERVAL_MAX:
            return int(text)
        return None

    def interval(self):
        """时间轴步长：非法或输入中时沿用上一次的合法值，所以恒在 1..INTERVAL_MAX"""
        val = self.parse_interval_seconds()
        if val is not None:
            self._unit_fallback = val
        return self._unit_fallback

    def validate_unit(self):
        theme.set_invalid(self.unit_edit, self.parse_interval_seconds() is None)

    # ---------- 当前时间（播放头 / 手动编辑） ----------
    @property
    def edited(self):
        return self._edited

    def set_edited(self, value):
        """外部（读档 / 撤销恢复）直接设定手动编辑态：同步描边，不触发信号"""
        self._edited = bool(value)
        if not self._edited:
            theme.set_invalid(self.timepoint_edit, False)
        self._paint_state()

    def set_video_player(self, player):
        """接入视频模块：当前时间默认跟随播放头动态更新，不影响表格。"""
        self.video_player = player
        player.segment_start_changed.connect(self._on_video_seconds)
        player.media_player.positionChanged.connect(self._on_video_position)
        self._on_video_seconds(player.current_position)

    def _on_video_position(self, ms):
        self._on_video_seconds(ms // 1000)

    def _on_video_seconds(self, seconds):
        # 用户编辑过当前时间后冻结跟随，直到完成一次记录
        if not self._edited:
            self.show_timepoint(seconds)

    def show_timepoint(self, seconds):
        self.timepoint_edit.setText(format_seconds(seconds))

    def parse_timepoint(self):
        """手动编辑的当前时间 → 秒；非法返回 None（非法时描红）"""
        t = parse_time(self.timepoint_edit.text())
        theme.set_invalid(self.timepoint_edit, t is None)
        return t

    def mark_timepoint_edited(self):
        """用户开始手动编辑当前时间：冻结播放头跟随，记录改为插入/覆盖"""
        self._edited = True
        theme.set_invalid(self.timepoint_edit, False)
        self._paint_state()
        self.timepoint_edited.emit()

    def _paint_state(self):
        """手动编辑态用紫色描边提示：此时记录会插入/覆盖表格行"""
        self.timepoint_edit.setProperty("edited", self._edited)
        theme.refresh_style(self.timepoint_edit)

    def reset_to_playback(self):
        """记录后当前时间回到默认（跟随视频）状态，不再影响表格"""
        self._edited = False
        theme.set_invalid(self.timepoint_edit, False)
        self._paint_state()
        if self.video_player is not None:
            pos = self.video_player.media_player.position()
            if pos >= 0:
                self.show_timepoint(pos // 1000)
            else:
                self._on_video_seconds(self.video_player.current_position)

    def advance_after_record(self, recorded_seconds):
        """手动时间记录后推进一个间隔，并保持编辑态供下一次记录。"""
        self._edited = True
        self.show_timepoint(recorded_seconds + self.interval())
        theme.set_invalid(self.timepoint_edit, False)
        self._paint_state()


# ------------------------------------------------------------------ 时间轴骨架

class TimeAxis:
    """「骨架 + 格位」行时间模型的算法层。

    不持有数据：三个平行表始终按属性名从 host 现读现写，所以宿主（或测试、读档）
    整体替换 ``selected_data`` / ``row_times`` / ``row_positions`` 之后不需要再通
    报一声。时间原点与步长由 host 上的取值器提供（一般是 ``TimeControls``）。
    """

    def __init__(self, host, rows_attr="selected_data", times_attr="row_times",
                 positions_attr="row_positions", start_getter=None,
                 interval_getter=None):
        self.host = host
        self.rows_attr = rows_attr
        self.times_attr = times_attr
        self.positions_attr = positions_attr
        self._start_getter = start_getter or getattr(host, "get_start_seconds")
        self._interval_getter = interval_getter or getattr(host, "interval_seconds")

    # ---------- 三个平行表 ----------
    @property
    def rows(self):
        return getattr(self.host, self.rows_attr)

    @rows.setter
    def rows(self, value):
        setattr(self.host, self.rows_attr, value)

    @property
    def times(self):
        return getattr(self.host, self.times_attr)

    @times.setter
    def times(self, value):
        setattr(self.host, self.times_attr, list(value))

    @property
    def positions(self):
        return getattr(self.host, self.positions_attr)

    @positions.setter
    def positions(self, value):
        setattr(self.host, self.positions_attr, list(value))

    def start_seconds(self):
        return self._start_getter()

    def interval(self):
        return self._interval_getter()

    # ---------- 不变式与骨架 ----------
    def align_row_states(self):
        """行时间/格位两个平行表补齐到行数，并维持“顺序行各占一格、固定时间点行不占”的不变式

        外部直接换掉行表（测试、读档、表格回同步）时会走到这里的兜底：
        缺格位的顺序行按行序接到骨架尾部，固定时间点行的格位一律抹成 None。
        """
        row_times, row_positions = self.times, self.positions
        n = len(self.rows)
        if len(row_times) < n:
            row_times += [None] * (n - len(row_times))
        del row_times[n:]
        if len(row_positions) < n:
            row_positions += [None] * (n - len(row_positions))
        del row_positions[n:]
        taken = {pos for pos, tm in zip(row_positions, row_times)
                 if tm is None and pos is not None}
        seen = set()
        tail = (max(taken) + 1) if taken else 0
        for i in range(n):
            if row_times[i] is not None:
                row_positions[i] = None  # 固定时间点行不在骨架上，删它不留痕迹
                continue
            pos = row_positions[i]
            if pos is None or pos in seen:
                pos = tail
                tail += 1
            row_positions[i] = pos
            seen.add(pos)

    def skeleton(self):
        """时间轴骨架：当前被顺序行占住的格位列表（固定时间点行不在内）"""
        return [pos for tm, pos in zip(self.times, self.positions)
                if tm is None and pos is not None]

    def head_slot(self):
        """骨架表头格位（最早的一格）：起始时间恒落在这一格上；无顺序行时返回 None"""
        seq = self.skeleton()
        return min(seq) if seq else None

    def tail_grid_slot(self):
        """接在骨架尾部的下一个格位（骨架未建立时从 0 起算）"""
        seq = self.skeleton()
        return max(seq) + 1 if seq else 0

    def computed_row_times(self):
        """逐行计算表格时间：顺序行 = 起始时间 +（格位 - 表头格位）× 间隔，固定时间点行保持原值

        格位是时间轴骨架上的第几格：删掉一个顺序行只是在骨架上留个隐藏格，所以
        改间隔时其余行仍按各自的原始格距展开，不会被压缩回相邻；而表头取骨架上最
        靠前的一格，删掉原第一行后起始时间照样落在当前第一行上。
        """
        self.align_row_states()
        starttime = self.start_seconds()
        unit = self.interval()
        # 格位恒不为负（记录不得早于起始时间），骨架未建立时表里也没有顺序行会用到它
        base = self.head_slot() or 0
        return [starttime + (pos - base) * unit if t is None else t
                for t, pos in zip(self.times, self.positions)]

    def header_row_index(self):
        """起始时间行的行号：占用骨架表头那一格的顺序行（它的时间就是起始时间）

        没建骨架（表里只有固定时间点行）时返回 None，此时没有受保护的行。
        """
        self.align_row_states()
        head = self.head_slot()
        if head is None:
            return None
        # 固定时间点行的格位已被抹成 None，而顺序行之间格位唯一，直接按表头格位定位
        return self.positions.index(head)

    def header_row_locked(self, rows):
        """起始时间行的删除保护：它是时间轴原点，删掉会让其余行整体重锚

        唯一放行的情况是选区覆盖全表（等价于清空表格数据）；部分选区里带着表头行就不删。
        """
        head = self.header_row_index()
        if head is None or not self.rows:
            return False
        return head in rows and len(rows) < len(self.rows)

    def grid_slot_for(self, t):
        """时间 t 是否与间隔格点重合：是则返回该档格位（按顺序行入表），否则 None

        格点 = 起始时间 + 整数格距 × 间隔，所以“与间隔重合”的记录一律是顺序行：
        夹在中间的空白格点是被删行留下的隐藏格，补回去即回到最原始的时间轴；
        表尾延长线上的空格点也算（新行接在骨架尾上，跟着以后的间隔改动一起走）。
        记录时间不早于起始时间、起始时间行又受删除保护，所以格位不会走到表头之前；
        骨架还没建立（表里一条顺序行都没有）时无从推格距，只能落成固定时间点行。
        """
        head = self.head_slot()
        if head is None:
            return None
        unit = self.interval()  # 恒 ≥ 1，不会拿 0 去求余
        delta = t - self.start_seconds()
        if delta < 0 or delta % unit:
            return None
        pos = head + delta // unit
        return pos if pos not in self.skeleton() else None

    def grid_insert_index(self, times, t):
        """按时间序给新行找插入位置：落在第一个比它晚的行之前"""
        for i, tt in enumerate(times):
            if tt > t:
                return i
        return len(times)

    # ---------- 记录 ----------
    def can_record_at(self, t):
        """记录时间不得早于起始时间：它是时间轴原点，格位不往负方向走"""
        return t >= self.start_seconds()

    def append(self, values):
        """默认追加：永远接在时间轴骨架尾部，从此跟着起始时间/间隔一起走"""
        times = self.computed_row_times()  # 顺带把行时间/格位两个平行表对齐到行数
        if not times:
            grid, t = 0, None                      # 第一行就是骨架表头，时间等于起始时间
        elif self.skeleton():
            grid, t = self.tail_grid_slot(), None  # 接在骨架尾上，跟着间隔走
        else:
            # 骨架未建立（表里只有固定时间点行，只能来自读档）：无格距可依，只能钉在末行之后
            grid, t = None, times[-1] + self.interval()
        self.rows.append(values)
        self.times.append(t)
        self.positions.append(grid)

    def record(self, values, t, overwrite=True):
        """指定时间点记录：与某行时间重合则覆盖、踩在格点上则按顺序行入骨架，否则插固定时间点行

        同一时刻上既有顺序行又有固定时间点行时优先覆盖后者，把格位留给时间轴骨架；
        早于起始时间一律拒收（返回 False），规则由 ``can_record_at`` 一处定义。
        ``overwrite=False`` 供「同一时刻允许多条记录」的事件表使用（小组交互页）：
        同刻不覆盖，只做时间序插入。
        """
        if not self.can_record_at(t):
            return False
        times = self.computed_row_times()
        if overwrite:
            same = [i for i, tt in enumerate(times) if tt == t]
            if same:
                idx = next((i for i in same if self.times[i] is not None), same[0])
                self.rows[idx] = values
                return True
        pos = self.grid_insert_index(times, t)
        slot = self.grid_slot_for(t)
        self.rows.insert(pos, values)
        # 顺序行：时间由格位推算，从此跟着起始时间/间隔走；固定时间点行钉死、不占格
        self.times.insert(pos, None if slot is not None else t)
        self.positions.insert(pos, slot)
        return True

    def drop_rows(self, rows):
        """删行：两种行的“删”后果不同，但都由调用方负责先行压栈

        删顺序行只是在骨架上留个隐藏格：其余行的时间与计时原样不动，改间隔时那一格
        照样跟着走，在那个时间上重新记录即补回；固定时间点行本来就不在骨架上，抹掉即
        回到没记过它的样子。起始时间行的保护判定在 ``header_row_locked``。
        """
        for row in sorted(rows, reverse=True):
            del self.rows[row], self.times[row], self.positions[row]

    def sort_rows_by_time(self):
        """行按当前计算时间排列：内容随行一起移动，顺序行之间相对次序不变

        当前时间点插入的行位置固定于其时间，起始时间/时间间隔改变后也要重新排座次；
        行时间与格位是同一行的两个侧面，必须随行一起搬。
        """
        row_times, row_positions = self.times, self.positions
        times = self.computed_row_times()
        order = sorted(range(len(self.rows)), key=lambda i: (times[i], i))
        if order == list(range(len(self.rows))):
            return
        self.rows = [self.rows[i] for i in order]
        self.times = [row_times[i] for i in order]
        self.positions = [row_positions[i] for i in order]

    def display_rows(self):
        """逐行的 (时间文本, 距起始时间的秒数, 行数据)：重建表格与导出都从这里取"""
        times = self.computed_row_times()
        start = self.start_seconds()
        return [(format_seconds(t), t - start, row)
                for t, row in zip(times, self.rows)]

    def archive(self):
        """存档用三件套快照（行数据由宿主自己深拷贝）"""
        return list(self.times), list(self.positions)


# ------------------------------------------------------------------ 表格量宽

def fixed_column_widths(table, fixed_count=FIXED_COL_COUNT):
    """前 ``fixed_count`` 列（时间/计时）的宽度：按当前字体量样本文本，再加单元格内边距

    写死像素在高分屏放大、字体回退（MiSans 装不上转微软雅黑时数字宽得多）以
    及用户改字号时都会不够（时间列被省略号截成 00:00:0），所以取“量出来的宽度”
    与“下限”里的较大者。
    """
    fm = table.fontMetrics()
    density = table.property("codingDensity")
    padding = theme.coding_density_value(density, 20, CELL_H_PADDING)
    return [max(theme.coding_density_value(density, 80, TIME_COL_MIN_WIDTH),
                fm.horizontalAdvance(TIME_COL_SAMPLE) + padding),
            max(theme.coding_density_value(density, 48, ELAPSED_COL_MIN_WIDTH),
                fm.horizontalAdvance(ELAPSED_COL_SAMPLE) + padding)]


def content_column_width(table, col):
    """单列按内容需要的宽度：量这一列最宽的文本（含表头维度名）再加内边距

    用 fontMetrics 而不是字符数×固定宽度：后者在中英文混排、字体回退时都算不
    准；封顶 CONTENT_COL_MAX（长注释不把相邻列挤到看不下），一到顶就不再往下扫，
    数千行的表上量宽只占重建耗时的一小部分。
    """
    fm = table.fontMetrics()
    padding = theme.coding_density_value(table.property("codingDensity"), 20, CELL_H_PADDING)
    header = table.horizontalHeaderItem(col)
    widest = fm.horizontalAdvance(header.text()) if header else 0
    # 同一列的文本高度重复（编码就是那几个值），量过的进缓存，不再重复测量；
    # 一旦顶到上限就收工，数千行也不会把重建时间堆在量宽上
    measured = {}
    for row in range(table.rowCount()):
        item = table.item(row, col)
        if not item:
            continue
        text = item.text()
        width = measured.get(text)
        if width is None:
            width = measured[text] = fm.horizontalAdvance(text)
        if width > widest:
            widest = width
            if widest + padding >= CONTENT_COL_MAX:
                return CONTENT_COL_MAX
    return min(CONTENT_COL_MAX, widest + padding)


def apply_column_widths(table, fixed_count=FIXED_COL_COUNT):
    """时间/计时定宽，其余列先均分剩余空间，内容装不下时再按内容放宽。

    均分份额是地板（不低于 CODE_COL_MIN_WIDTH，也保证宽表下不留右侧空白），
    内容宽是天花板（到 CONTENT_COL_MAX 为止）；可编辑下拉列要承载单元格编辑器，
    不能因为表格区被压窄就缩成几十像素，列多到放不下时由表格横向滚动接手。
    """
    header = table.horizontalHeader()
    cols = table.columnCount()
    widths = fixed_column_widths(table, fixed_count)
    for col in range(min(fixed_count, cols)):
        header.setSectionResizeMode(col, QHeaderView.Fixed)
        table.setColumnWidth(col, widths[col])
    for col in range(fixed_count, cols):
        header.setSectionResizeMode(col, QHeaderView.Interactive)
    flexible = max(cols - fixed_count, 1)
    spare = table.viewport().width() - sum(widths[:min(fixed_count, len(widths))])
    each = max(theme.coding_density_value(
        table.property("codingDensity"), 76, CODE_COL_MIN_WIDTH), spare // flexible)
    body = [max(each, content_column_width(table, col))
            for col in range(fixed_count, cols)]
    if body:
        # 整数除法剩下的尾差（1~N 像素）交给末列吃掉，否则右缘总留着一条空白缝
        body[-1] = max(body[-1], spare - sum(body[:-1]))
    for col, width in zip(range(fixed_count, cols), body):
        table.setColumnWidth(col, width)
