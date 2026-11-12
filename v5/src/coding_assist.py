# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""Phase 1 编码辅助：从「当前编码页自己已录的序列」给下一步候选码。

**这不是视频识别**，而是序列自动补全：按每个维度已记录的一阶转移（上一个码 → 下一个
码）给出 top-k 候选，用户点选即填入对应维度卡片的勾选态，**绝不自动写表**——真正的写入
仍走编码页现有的「记录」按钮（``continue_``）。因此建议码只是把某个编码块勾上（与用户
亲手点它完全等价），撤销栈与脏标记口径原样不变（见 ``DimensionCard.select_value``）。

数据取自 data_encoder 的 ``selected_data``（行 = ``[各维度码…, 注释]``）与
``dim_panel.framework()``（``[(维度名, [编码…])]``，与维度列一一对应）。样本极少时如实
降级为「仅供参考」，不假装有把握；没有任何历史时给出引导文案而非空板。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

import theme

#: 每个维度给出的候选码上限
TOP_K = 3
#: 转移证据少于此数时只当「仅供参考」（小样本下诚实降级）
MIN_SUPPORT_CONFIDENT = 6


# ------------------------------------------------------------------ 纯逻辑

def _row_codes(row, n_dims):
    """一行 ``[各维度码…, 注释]`` → 前 n_dims 个维度码（去空白、缺失补空串）。"""
    out = []
    for index in range(n_dims):
        value = row[index] if index < len(row) else ""
        out.append(str(value or "").strip())
    return out


def build_transition_counts(rows, n_dims):
    """扫一遍已录行，得到每维度的转移计数与整体频次。

    返回 ``(trans, uni)``：``trans[d] = {prev_code: {next_code: count}}``、
    ``uni[d] = {code: count}``。只统计相邻两行在该维度上**都非空**的转移对。
    """
    trans = [dict() for _ in range(n_dims)]
    uni = [dict() for _ in range(n_dims)]
    prev = None
    for row in rows or []:
        codes = _row_codes(row, n_dims)
        for dim in range(n_dims):
            if codes[dim]:
                uni[dim][codes[dim]] = uni[dim].get(codes[dim], 0) + 1
        if prev is not None:
            for dim in range(n_dims):
                before, after = prev[dim], codes[dim]
                if before and after:
                    bucket = trans[dim].setdefault(before, {})
                    bucket[after] = bucket.get(after, 0) + 1
        prev = codes
    return trans, uni


def _last_codes(rows, n_dims):
    """每维度「最后一个非空码」：一阶预测的起点。整列都空则留空串。"""
    last = ["" for _ in range(n_dims)]
    for row in reversed(rows or []):
        codes = _row_codes(row, n_dims)
        for dim in range(n_dims):
            if not last[dim] and codes[dim]:
                last[dim] = codes[dim]
        if all(last):
            break
    return last


def _rank(dist, vocab, k):
    """按计数降序（同数按码名稳定排序）取前 k，可选只保留框架词表内的码。"""
    total = sum(dist.values())
    if total <= 0:
        return [], 0
    ranked = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for code, count in ranked:
        if vocab and code not in vocab:
            continue
        out.append((code, count / total))
        if len(out) >= k:
            break
    return out, total


def suggest_for_dimension(trans_dim, uni_dim, last_code, vocab=None, k=TOP_K):
    """单维度的下一步候选。

    返回 ``(candidates, mode, support)``：``candidates = [(code, prob)]``，
    ``mode`` 为 ``"transition"``（有上一个码的转移证据）/ ``"frequency"``（退回整体频次）
    / ``"none"``（无任何证据），``support`` 是该判断背后的样本次数。
    """
    vocab = [str(item).strip() for item in (vocab or []) if str(item).strip()]
    if last_code:
        dist = trans_dim.get(last_code)
        if dist:
            candidates, total = _rank(dist, vocab, k)
            if candidates:
                return candidates, "transition", total
    if uni_dim:
        candidates, total = _rank(uni_dim, vocab, k)
        if candidates:
            return candidates, "frequency", total
    return [], "none", 0


def suggest_next(rows, framework, k=TOP_K):
    """整页的下一步建议：``framework`` 与 ``selected_data`` 的维度列一一对应。

    返回 ``[{"dimension", "last_code", "candidates", "mode", "support", "confident"}]``，
    每个维度一条；没有维度时返回空列表。``confident`` 为假表示证据不足、仅供参考。
    """
    n_dims = len(framework or [])
    if n_dims == 0:
        return []
    trans, uni = build_transition_counts(rows, n_dims)
    last_codes = _last_codes(rows, n_dims)
    out = []
    for dim, (name, codes) in enumerate(framework):
        candidates, mode, support = suggest_for_dimension(
            trans[dim], uni[dim], last_codes[dim], codes, k)
        out.append({
            "dimension": str(name),
            "last_code": last_codes[dim],
            "candidates": candidates,
            "mode": mode,
            "support": support,
            "confident": support >= MIN_SUPPORT_CONFIDENT,
        })
    return out


# ------------------------------------------------------------------ 建议条 UI

class CodingSuggestBar(QWidget):
    """编码页左列的「下一步建议」条：逐维度列出可点选的候选码。

    点一个候选码即把它填入对应维度卡片的勾选态（``encoder.apply_suggestion``），用户仍需
    亲手点「记录」才写表——建议本身不碰数据、不进撤销栈、不触发脏标记。
    """

    def __init__(self, encoder, parent=None):
        super().__init__(encoder if parent is None else parent)
        self.encoder = encoder
        self.setProperty("card", True)
        self._rows = []
        self._build()
        self.refresh()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.icon_label = QLabel()
        theme.bind_icon(self.icon_label, "sparkle", 15, color="purple")
        head.addWidget(self.icon_label)
        title = QLabel("下一步建议")
        title.setProperty("section", True)
        head.addWidget(title)
        head.addStretch(1)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setProperty("btn", "ghost")
        self.refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self.refresh_btn)
        outer.addLayout(head)

        self.note_label = QLabel("序列自动补全，非视频识别；点选即填入，仍需你确认后「记录」。")
        self.note_label.setProperty("hint", True)
        self.note_label.setWordWrap(True)
        outer.addWidget(self.note_label)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(4)
        outer.addWidget(self.body)

    # ---------- 刷新 ----------
    def refresh(self):
        """按编码页当前已录序列重算建议并重画各行；无历史时给引导文案。"""
        self._clear_rows()
        try:
            framework = self.encoder.dim_panel.framework()
            rows = list(self.encoder.selected_data)
        except Exception:                       # noqa: BLE001 面板还没建好时静默
            framework, rows = [], []
        suggestions = suggest_next(rows, framework)
        usable = [item for item in suggestions if item["candidates"]]
        if not rows:
            self._add_message("记录几行编码后，这里会按你的历史给出下一步候选码。")
            return
        if not usable:
            self._add_message("暂无可给的候选：先在下方勾选编码并「记录」几行。")
            return
        for index, item in enumerate(suggestions):
            if not item["candidates"]:
                continue
            self._add_suggestion_row(index, item)

    def _clear_rows(self):
        """清掉上一轮的建议行。

        ``deleteLater`` 是异步的：只调它的话，旧行在事件循环回收前仍是 ``body`` 的子控件，
        ``findChildren`` / 主题图标重绑定都还会查到它们（连点到的可能是待删的旧按钮）。故先
        ``hide`` + ``setParent(None)`` 立即脱离父级，再交给 ``deleteLater`` 回收。
        """
        while self.body_layout.count():
            widget = self.body_layout.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _add_message(self, text):
        label = QLabel(text)
        label.setProperty("hint", True)
        label.setWordWrap(True)
        self.body_layout.addWidget(label)

    def _add_suggestion_row(self, dim_index, item):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        name = QLabel(item["dimension"] or ("维度%d" % (dim_index + 1)))
        name.setProperty("hint", True)
        name.setMinimumWidth(64)
        name.setMaximumWidth(110)
        lay.addWidget(name)
        base = "上一个码：%s" % (item["last_code"] or "（空）")
        for code, prob in item["candidates"]:
            button = QPushButton(code)
            button.setProperty("btn", "ghost")
            button.setToolTip("%s · 历史概率约 %d%%%s"
                              % (base, round(prob * 100),
                                 "" if item["confident"] else " · 样本较少，仅供参考"))
            button.clicked.connect(
                lambda _checked=False, index=dim_index, value=code: self._on_pick(index, value))
            lay.addWidget(button)
        if item["mode"] == "frequency":
            tail = QLabel("（按频次）")
            tail.setProperty("hint", True)
            lay.addWidget(tail)
        elif not item["confident"]:
            tail = QLabel("（仅供参考）")
            tail.setProperty("hint", True)
            lay.addWidget(tail)
        lay.addStretch(1)
        self.body_layout.addWidget(row)

    def _on_pick(self, dim_index, code):
        apply = getattr(self.encoder, "apply_suggestion", None)
        if callable(apply):
            apply(dim_index, code)
