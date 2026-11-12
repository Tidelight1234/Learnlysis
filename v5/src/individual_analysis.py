# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""个体模式分析：IND01–IND15 的工作台入口。

外壳全在 `analysis_workbench`，本文件只钉两件事：模式 = individual，以及这个模式
的独立分析单元口径（`student_uid = 组号__学生`）如何在界面上说清楚 —— 个体模式
最常踩的坑是「把四个组各自的 S1 当成同一个人」，所以数据卡下方固定给一行提醒。
"""

from __future__ import annotations

import analysis_methods as am
from analysis_workbench import ModeAnalysisWorkbench


class IndividualAnalysisModule(ModeAnalysisWorkbench):
    def __init__(self, tabmodule=None, parent=None):
        super().__init__(mode=am.MODE_INDIVIDUAL, parent=parent)
        # tabmodule 仅为兼容旧调用点保留：个体模式不再持有编码框架面板
        self.tabmodule = tabmodule
        self.data_hint.setToolTip(
            "独立分析单元 = 组号__学生：不同组的 S1 是不同的人，绝不合并")

    def _refresh_data_hint(self, ctx):
        super()._refresh_data_hint(ctx)
        if not ctx.usable or ctx.n_groups < 2:
            return
        self.data_hint.setText(
            "%s\n注意：多组同名学生已按「组号__学生」拆分，当前共 %d 个独立单元"
            % (self.data_hint.text(), ctx.n_students))
