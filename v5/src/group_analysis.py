# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""小组模式分析：GRP01–GRP10 的工作台入口。

与个体模式只差三点，都在「组是独立分析单元」这件事上：

1. 数据卡的组清单多一列「事件来源」—— 小组方法里 GRP08/GRP09 吃的是 Interaction_Event，
   看得到事件从哪来（会话录入 / 文件 sheet / 无）才知道为什么某张表是空的；
2. 批量上传天然按「一文件一组」组织，跨组比较需要 ≥ 2 组，不足时按注册表口径降级；
3. 取数口默认取小组交互页登记的 group_id，而不是写死 group_1。
"""

from __future__ import annotations

import analysis_core as ac
import analysis_methods as am
from analysis_workbench import ModeAnalysisWorkbench


class GroupAnalysisModule(ModeAnalysisWorkbench):
    EXTRA_GROUP_COLUMNS = ("事件来源",)

    def __init__(self, tabmodule=None, parent=None):
        super().__init__(mode=am.MODE_GROUP, parent=parent)
        # tabmodule 仅为兼容旧调用点保留：小组模式不再持有编码框架面板
        self.tabmodule = tabmodule

    def _extra_group_values(self, bundle):
        return (self._event_source(bundle),)

    @staticmethod
    def _event_source(bundle):
        """事件从哪来：只影响 GRP08/GRP09 能否出表，不影响状态表类方法。"""
        if not bundle.has_events:
            return "无"
        origin = str(bundle.meta.get("event_source") or "").strip()
        if origin:
            return origin
        if str(bundle.meta.get("source") or "") == "session":
            return "会话录入"
        if bundle.source_file:
            return "文件 sheet"
        return "未知"

    def _refresh_data_hint(self, ctx):
        super()._refresh_data_hint(ctx)
        if not ctx.usable:
            return
        downgrade = ac.group_count_note(ctx.bundles)
        if downgrade:
            self.data_hint.setText("%s\n%s" % (self.data_hint.text(), downgrade))
