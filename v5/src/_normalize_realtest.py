"""把老导出规范化成符合列名契约的副本，供 `_smoke_modes.py` 的真实数据通道使用。

老数据的列名是 `S1Eoding_1`（Encoding 少了一个 n），按 `analysis_core.STRICT_COLUMN_RE`
装不出成员。本脚本只改表头这一处：单元格内容、行序、列序原样搬到 `_testdata_real/`
（已 gitignore），**受保护的 `Results/testfile/` 只读不写**。

跑法（PowerShell）：
    cd version\\v5; ..\\.conda\\python.exe -X utf8 _normalize_realtest.py

退出码 0 = 副本已生成；1 = 源目录一个文件都没有（说明仓库不完整，不是脚本失败）。
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd

SRC_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "Results", "testfile", "group_*.xlsx")
DST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_testdata_real")

#: 老导出里的拼写错误列名：只认这一种变体，别的形状一律不猜
TYPO_COLUMN_RE = re.compile(r"^(S\d+)Eoding_(\d+)$")
#: 契约列名：副本改完还得能被这条正则认出来，否则等于白搬一次
STRICT_COLUMN_RE = re.compile(r"^(S\d+)Encoding_(\d+)$")


def normalize_columns(frame):
    """返回 (新表, 改名数)：只把 `Eoding` 补成 `Encoding`，其余列名一字不动。"""
    renames = {}
    for column in frame.columns:
        match = TYPO_COLUMN_RE.match(str(column).strip())
        if match:
            renames[column] = "%sEncoding_%s" % (match.group(1), match.group(2))
    return frame.rename(columns=renames), len(renames)


def main():
    import glob
    paths = sorted(glob.glob(SRC_GLOB))
    if not paths:
        print("找不到源文件：%s" % os.path.normpath(SRC_GLOB))
        return 1
    os.makedirs(DST_DIR, exist_ok=True)
    fixed, copied, already, unmatched = 0, 0, [], []
    for path in paths:
        frame = pd.read_excel(path, sheet_name=0, dtype=object)
        frame, renamed = normalize_columns(frame)
        if renamed:
            fixed += 1
        elif any(STRICT_COLUMN_RE.match(str(column).strip()) for column in frame.columns):
            already.append(os.path.basename(path))
        else:
            unmatched.append(os.path.basename(path))
        target = os.path.join(DST_DIR, os.path.basename(path))
        frame.to_excel(target, index=False, sheet_name="Behavior_State")
        copied += 1
    print("NORMALIZE_OK 复制=%d 改表头=%d 本来就合规=%d" % (copied, fixed, len(already)))
    if unmatched:
        # 表头形状根本不是 `S{i}Encoding_{j}`（老版中文表头导出），不猜语义，原样带着
        print("下列文件表头不符合契约，副本仍不可解析：%s" % "、".join(unmatched))
    print("输出目录：%s" % DST_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
