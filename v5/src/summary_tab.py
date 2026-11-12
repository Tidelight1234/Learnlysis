# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""数据汇总标签页 SummaryTab。

表格把所有编码页的列数据（不含注释）按标签顺序并排陈列，
另可导出 Time / S1Encoding_1 … 的标准分析格式与多 sheet 小组数据包。
"""

import pandas as pd
from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                             QLabel, QTableWidget, QTableWidgetItem, QAbstractItemView,
                             QMessageBox)
from openpyxl.styles import Alignment, Font

import interaction_model as im
import theme
import user_state
from language import source_text, translate_widget_tree
from analysis_core import members_from_columns
from data_encoder import DataEncoderModule

# 不参与“编码维度”汇总的列名：时间类固定列（序号列已取消，兼容旧存档）
TIME_HEADERS = ("时间", "计时", "序号")
# 注释列按列名包含判定，兼容“注释/备注”等不同写法
COMMENT_KEY = "注释"
# 可选的组别列：只在导出小组数据包与显式勾选时出现，默认不写以免打碎旧列名契约
GROUP_KEY = "group_id"
# 汇总表格的列头分隔符：维度名在各页之间会重名，冠上标签名才分得开
GROUP_SEP = "·"


class SummaryTab(QWidget):
    def __init__(self, parent=None, tabmodule=None):
        super().__init__(parent)
        if tabmodule is None:
            raise ValueError("必须传入 tabmodule 参数")
        self.tabmodule = tabmodule  # 保存对内层标签页的引用
        # 小组页取数入口由主窗口注入：没有小组页时只导出状态表，不报错
        self.group_provider = None
        # 初始化UI
        self.initUI()
        # 初始化时加载数据
        self.generate_summary_data()
        # 监听标签页变化
        self.tabmodule.currentChanged.connect(self.generate_summary_data)

    def set_group_source(self, provider):
        """注入小组交互页取数器 provider() -> dict（组号/成员/事件表/主体字典）。"""
        self.group_provider = provider

    def initUI(self):
        """初始化汇总页面UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        time_notice_label = QLabel("注意：编码个体的时间需要保持一致。"
                                   "下表按标签页顺序并排陈列各页数据（不含注释列），行号一一对应。")
        time_notice_label.setProperty("notice", True)
        layout.addWidget(time_notice_label)
        # 汇总表只作展示与核对，改动会随切换标签页被重算，故锁住编辑
        self.summary_table = QTableWidget()
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 创建输出按钮
        self.analysis_btn = QPushButton("输出分析格式文件")
        self.analysis_btn.setProperty("btn", "primary")
        self.analysis_btn.setToolTip(f"导出 Time、S1Encoding_1 等标准列名，可直接用于后续数据分析（不含{COMMENT_KEY}）")
        theme.bind_icon(self.analysis_btn, "export", 16)
        self.analysis_note_btn = QPushButton("输出分析格式文件（含注释）")
        self.analysis_note_btn.setProperty("btn", "secondary")
        self.summary_btn = QPushButton("输出汇总表格")
        self.summary_btn.setProperty("btn", "secondary")
        self.group_btn = QPushButton("输出小组数据包")
        self.group_btn.setProperty("btn", "secondary")
        self.group_btn.setToolTip("导出 Group_Info / Students / Behavior_State / "
                                  "Interaction_Event / Codebook / Enums 多 sheet 文件")
        self.analysis_btn.clicked.connect(lambda _=False: self.output_analysis_data())
        self.analysis_note_btn.clicked.connect(lambda _=False: self.output_analysis_data(with_comments=True))
        self.summary_btn.clicked.connect(self.output_summary_data)
        self.group_btn.clicked.connect(self.output_group_pack)
        output_btn_layout = QHBoxLayout()
        for btn in (self.analysis_btn, self.analysis_note_btn, self.summary_btn, self.group_btn):
            output_btn_layout.addWidget(btn)
        output_btn_layout.addStretch(1)
        # 布局组装
        layout.addWidget(self.summary_table)
        layout.addLayout(output_btn_layout)
        self.setLayout(layout)

    # ---------- 取数 ----------
    def encoder_pages(self):
        """按标签顺序返回 (标签名, 编码页)，数据汇总页自身不在其中"""
        pages = []
        for i in range(self.tabmodule.count()):
            widget = self.tabmodule.widget(i)
            if isinstance(widget, DataEncoderModule):
                pages.append((source_text(self.tabmodule.tabText(i)), widget))
        return pages

    @staticmethod
    def header_at(table, col):
        item = table.horizontalHeaderItem(col)
        return source_text(item.text().strip()) if item else ""

    @staticmethod
    def cell_text(table, row, col):
        item = table.item(row, col)
        return item.text().strip() if item else ""

    @staticmethod
    def column_by_header(table, names, default=0):
        """按列名定位列号（列可能被用户挪动，故不写死下标）"""
        for col in range(table.columnCount()):
            if SummaryTab.header_at(table, col) in names:
                return col
        return default

    def body_columns(self, table):
        """汇总表格要陈列的列：该页所有列去掉注释"""
        return [col for col in range(table.columnCount())
                if COMMENT_KEY not in self.header_at(table, col)]

    def dimension_columns(self, table):
        """编码维度列：去掉时间类固定列与注释列（按列名精确匹配，允许维度叫“时间分配”）"""
        return [col for col in range(table.columnCount())
                if self.header_at(table, col)
                and self.header_at(table, col) not in TIME_HEADERS
                and COMMENT_KEY not in self.header_at(table, col)]

    def comment_column(self, table):
        """注释列号，没有则返回 None"""
        for col in range(table.columnCount()):
            if COMMENT_KEY in self.header_at(table, col):
                return col
        return None

    @staticmethod
    def to_seconds(text):
        """HH:MM:SS / MM:SS / 秒数 → 秒，无法解析返回 None"""
        text = text.strip()
        if not text:
            return None
        try:
            total = 0
            for part in text.split(':'):
                total = total * 60 + int(part)
            return total
        except ValueError:
            return None

    def page_time_column(self, table):
        """按行取该页时间列（换算为秒），空格保留为 ''，行号不与表格错位"""
        # 编码页表头固定是“时间”在前、“计时”在后，按列序取第一个命中的列即可
        col = self.column_by_header(table, TIME_HEADERS)
        values = []
        for row in range(table.rowCount()):
            seconds = self.to_seconds(self.cell_text(table, row, col))
            values.append("" if seconds is None else str(seconds))
        return values

    def summary_time_column(self, pages):
        """时间列取非空时间最多的那一页，其余页只为空缺行补位；长度按各页最大行数补齐

        各页时间本应一致，标准分析格式里因此仍用单一 Time 列代表全部个体。
        """
        candidates = [self.page_time_column(page.result_table) for _, page in pages]
        if not candidates:
            return []
        times = list(max(candidates, key=lambda values: len([v for v in values if v])))
        times += [""] * (max(len(values) for values in candidates) - len(times))
        for values in candidates:
            for i, value in enumerate(values):
                if not times[i] and value:
                    times[i] = value
        return times

    # ---------- 汇总表格 ----------
    def generate_summary_data(self):
        """把所有编码页的列数据（不含注释）按标签顺序并排摆进汇总表格"""
        columns, rows = self.build_summary_view()
        self.update_summary_table(columns, rows)

    def build_summary_view(self):
        """返回 (列头列表, 行列表)：每页贡献一组列，行号跨页对齐"""
        groups = []
        columns = []
        for tab_text, page in self.encoder_pages():
            table = page.result_table
            picked = self.body_columns(table)
            groups.append((table, picked))
            columns += [f"{tab_text}{GROUP_SEP}{self.header_at(table, col)}" for col in picked]
        row_count = max([table.rowCount() for table, _ in groups], default=0)
        rows = []
        for row in range(row_count):
            cells = []
            for table, picked in groups:
                cells += [self.cell_text(table, row, col) if row < table.rowCount() else ""
                          for col in picked]
            rows.append(cells)
        return columns, rows

    def update_table_with_data(self, data, headers=None):
        """存档恢复用：直接按行列表（可选列头）填充汇总表格"""
        if not data:
            self.summary_table.clear()
            self.summary_table.setRowCount(0)
            self.summary_table.setColumnCount(0)
            return
        self.summary_table.setColumnCount(len(data[0]))
        if headers and len(headers) == len(data[0]):
            self.summary_table.setHorizontalHeaderLabels([str(h) for h in headers])
        self.summary_table.setRowCount(len(data))
        for row_idx, row in enumerate(data):
            for col_idx, value in enumerate(row):
                self.summary_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()
        translate_widget_tree(self.summary_table)

    def update_summary_table(self, columns, rows):
        """更新汇总表格：列头与单元格一一对应，不用字典以免跨页同名列相互覆盖"""
        self.summary_table.setColumnCount(len(columns))
        self.summary_table.setHorizontalHeaderLabels(columns)
        translate_widget_tree(self.summary_table)
        self.summary_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx in range(len(columns)):
                value = row[col_idx] if col_idx < len(row) else ""
                self.summary_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.summary_table.resizeColumnsToContents()
        self.summary_table.resizeRowsToContents()

    # ---------- 标准分析格式 ----------
    def build_analysis_data(self, with_comments=False, with_group=False):
        """生成 Time + S{个体序号}Encoding_{维度序号} 的标准格式（含注释时再附 S{序号}注释）

        ``with_group`` 只在首列前加一列 ``group_id``：默认关，避免改变既有列名契约。
        """
        pages = self.encoder_pages()
        if not pages:
            return [], []
        dims = []          # [(table, col)] 按个体、维度顺序排列
        comments = []      # [(table, col)] 与 dims 同页序，供含注释导出
        columns = [GROUP_KEY] if with_group else []
        columns += ["Time"]
        for page_idx, (_, page) in enumerate(pages, start=1):
            table = page.result_table
            dim_cols = self.dimension_columns(table)
            dims += [(table, col) for col in dim_cols]
            columns += [f"S{page_idx}Encoding_{dim_idx}" for dim_idx in range(1, len(dim_cols) + 1)]
            if with_comments:
                col = self.comment_column(table)
                comments.append((table, col if col is not None else -1))
        # 注释列排在所有编码列之后，与每行单元格的先后顺序保持一致
        if with_comments:
            columns += [f"S{i}{COMMENT_KEY}" for i in range(1, len(comments) + 1)]
        times = self.summary_time_column(pages)
        row_count = max([table.rowCount() for table, _ in dims] + [len(times)])
        group_id = self.group_id() if with_group else ""
        rows = []
        for row in range(row_count):
            cells = [group_id] if with_group else []
            cells.append(times[row] if row < len(times) else "")
            for table, col in dims:
                cells.append(self.cell_text(table, row, col) if row < table.rowCount() else "")
            for table, col in comments:
                cells.append(self.cell_text(table, row, col)
                             if col >= 0 and row < table.rowCount() else "")
            rows.append(cells)
        return columns, rows

    def output_analysis_data(self, with_comments=False):
        """输出标准分析格式的 Excel 文件"""
        columns, rows = self.build_analysis_data(with_comments)
        if not columns or not rows:
            QMessageBox.warning(self, "警告", "没有可输出的数据！")
            return
        suffix = "（含注释）" if with_comments else ""
        file_path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, f"保存分析格式文件{suffix}",
            f"分析格式数据{suffix}.xlsx", "Excel 文件 (*.xlsx);;所有文件 (*)")
        if not file_path:
            return
        try:
            self.export_dataframe(pd.DataFrame(rows, columns=columns),
                                  file_path, "分析格式")
            QMessageBox.information(
                self, "成功",
                f"分析格式文件已导出（{len(rows)} 行 × {len(columns)} 列）！\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    # ---------- 小组数据包 ----------
    def group_info(self):
        """小组页现场：没注入或没内容时返回空 dict（导出时只给状态表）。"""
        return (self.group_provider() if self.group_provider else None) or {}

    def group_id(self):
        return str(self.group_info().get("group_id") or "group_1")

    def output_group_pack(self):
        """导出多 sheet 小组数据包（Group_Info / Students / Behavior_State /
        Interaction_Event / Codebook / Enums）。

        成员优先取小组页现场（包含只有页标题没有编码列的情况），
        退回从列名解析；Codebook 交给 ``im.codebook_rows`` 展平。
        """
        info = self.group_info()
        columns, rows = self.build_analysis_data()
        state = pd.DataFrame(rows, columns=columns) if columns and rows else None
        members = [str(value).strip() for value in info.get("members", []) if str(value).strip()]
        members = members or (members_from_columns(columns) if columns else [])
        sheets = im.template_sheets(
            group_id=info.get("group_id") or "group_1",
            group_name=info.get("group_name") or "",
            members=members,
            interaction_types=info.get("interaction_types"),
            codebook=im.codebook_rows(info.get("codebook")),
            behavior_state=state, events=info.get("events"))
        file_path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "保存小组数据包",
            "%s_数据包.xlsx" % (info.get("group_id") or "group_1"),
            "Excel 文件 (*.xlsx);;所有文件 (*)")
        if not file_path:
            return
        try:
            im.write_workbook(sheets, file_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")
            return
        QMessageBox.information(self, "成功", f"小组数据包已导出！\n{file_path}")

    # ---------- 导出汇总表格 ----------
    def output_summary_data(self):
        """输出当前汇总表格（各页并排、不含注释）到 Excel"""
        if self.summary_table.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有可输出的数据！")
            return
        file_path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "保存汇总表格", "汇总表格.xlsx",
            "Excel 文件 (*.xlsx);;所有文件 (*)")
        if not file_path:
            return
        try:
            self.export_dataframe(self.table_dataframe(), file_path, "汇总")
            QMessageBox.information(self, "成功", f"数据已成功导出！\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def table_dataframe(self):
        """把汇总表格原样转成 DataFrame（列头即表头）"""
        table = self.summary_table
        columns = [self.header_at(table, col) or f"列{col + 1}"
                   for col in range(table.columnCount())]
        rows = [[self.cell_text(table, row, col) for col in range(table.columnCount())]
                for row in range(table.rowCount())]
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def export_dataframe(df, file_path, sheet_name):
        """统一的 Excel 落盘：自动补扩展名、列宽自适应、内容居中、表头加粗"""
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except Exception:
                        pass
                worksheet.column_dimensions[column].width = max_length + 4
                for cell in col:
                    cell.alignment = Alignment(horizontal="center")
            for cell in worksheet["1:1"]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
