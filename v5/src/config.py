# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""全局路径配置：应用根目录、data 目录、存档文件路径、应用元信息。"""

import sys
import os

# 应用元信息：关于弹窗与任务栏展示用
APP_NAME = "Learnlysis"
APP_TAGLINE = "学习行为编码与分析系统"
APP_VERSION = "5.1.0"

# 版权口径的唯一来源：根目录 LICENSE 的版权行与这里必须一致，
# 关于弹窗只拼这两个串，不在界面里写第二种说法（改一处就漏一处）。APP_NAME
# 已在上面，所以版权行里的名字仍跟着产品名走。
APP_COPYRIGHT = "Copyright © 2026 %s. Licensed under the MIT License." % APP_NAME
APP_LEGAL_NOTE = "使用 PySide6/Qt（LGPLv3）；界面字体 MiSans（小米）。许可见 LICENSE/NOTICE。"

def get_app_dir():
    """获取应用程序运行时的根目录"""
    if getattr(sys, 'frozen', False):  # 打包后
        return os.path.dirname(sys.executable)
    else:  # 开发环境：产品码在 src/ 里，而 data/ 一直落在版本树根（方案 B）。
        # 多剥一层等于承认「src 是码、树根是现场」这条分界；改这里就是改老用户
        # 存档的位置，动之前先想清楚 archive.json 与 projects/ 要不要跟着搬。
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 构建 data 目录路径（可写目录）
data_dir = os.path.join(get_app_dir(), "data")

# 创建目录（如果不存在）
os.makedirs(data_dir, exist_ok=True)

# 构建 archive.json 文件路径
ARCHIVE_FILE = os.path.join(data_dir, "archive.json")
# 自动保存的原子写中转文件（写完即被替换掉，不会留下半份存档）
ARCHIVE_TMP_FILE = ARCHIVE_FILE + ".tmp"
# 用户运行时状态：文件对话框记忆目录、最近打开清单、自动保存开关
USER_STATE_FILE = os.path.join(data_dir, "user_state.json")
# R 参考层的手工指定路径：探测失败时用户可在里面写 Rscript 绝对路径
R_SETTINGS_FILE = os.path.join(data_dir, "r_settings.json")
# AI 助手设置：OpenAI 兼容端点 / 模型 / API key / 智能体档案（见 ai_settings.py）。
# 与 archive.json 分开存——它由用户在 AI 助手页显式改动时落盘，不进 1.5 秒防抖的自动保存，
# 里面的 api_key 也绝不进存档/导出/日志（其键名不作为对外数据契约）。
AI_SETTINGS_FILE = os.path.join(data_dir, "ai_settings.json")
# AI「框架设计师」的研究简报 / 草稿版本 / 参考卡快照。它与 API key 设置分开，
# 也不进入 archive.json；只有研究者确认应用后的轻量框架才随编码页进入存档。
AI_FRAMEWORK_PROJECTS_FILE = os.path.join(data_dir, "ai_framework_projects.json")

