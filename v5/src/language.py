# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""Persistent Chinese/English UI labels shared by the v5 shell and text tools."""

from user_state import state

LANGUAGES = (("zh-CN", "中文"), ("en-US", "English"))

_EN = {
    "文件": "File", "视图": "View", "说明": "Help", "最小化": "Minimize", "最大化": "Maximize", "关闭": "Close",
    "深色模式": "Dark Mode", "语言": "Language", "中文": "Chinese", "就绪": "Ready", "确定": "OK", "取消": "Cancel",
    "显示 AI 编码工具": "Show AI Coding Tools", "数据编码字号": "Coding Font Size",
    "正常": "Normal", "小一号": "Smaller",
    "小": "Small", "中": "Medium", "大（正常）": "Large (Normal)",
    "自定义字号…": "Custom Size…", "自定义字号": "Custom Font Size",
    "字号（像素，10–20）": "Font size (pixels, 10–20)",
    "数据编码": "Coding", "字号": "Font Size", "编码区布局": "Coding Layout",
    "维度编辑显示模式": "Dimension Display Mode", "标准模式": "Standard",
    "简洁模式": "Compact", "统计及可视化": "Statistics & Visualization",
    "个体模式分析": "Individual Analysis", "小组模式分析": "Group Analysis", "AI 助手": "AI Assistant",
    "最近打开": "Recent Files", "导入": "Import", "导出": "Export", "保存": "Save", "另存为": "Save As", "打开": "Open",
    "新建": "New", "删除": "Delete", "编辑": "Edit", "应用": "Apply", "刷新": "Refresh", "搜索": "Search", "选择": "Select",
    "视频": "Video", "统计数据": "Statistics Data", "个体分析数据": "Individual Analysis Data", "小组分析数据": "Group Analysis Data",
    "模型与密钥": "Model & API Key", "智能体": "Assistant Profile", "对话": "Chat", "测试连接": "Test Connection", "保存设置": "Save Settings",
    "任务模板（双击填入对话框）": "Prompt templates (double-click to insert)", "附带上下文：": "Context: ", "无": "None",
    "进入框架设计工作台": "Open Framework Designer", "框架设计师": "Framework Designer", "分析规划师": "Analysis Planner",
    "编码助手": "Coding Assistant", "结果解读": "Results Interpreter", "报告撰写": "Report Writer",
    "新项目": "New Project", "新项目 · 尚未保存": "New Project · Unsaved", "项目": "Project", "项目名称": "Project Name",
    "小组 ID": "Group ID", "任务 ID": "Task ID", "阶段 ID": "Phase ID", "编码员 ID": "Annotator ID", "编码本版本": "Codebook Version",
    "工作进度": "Progress", "媒体与转写": "Media & Transcript", "编码数据": "Coding Data", "质检与导出": "Validation & Export",
    "项目概览": "Project Overview", "导入数据": "Import Data", "数据检查": "Data Validation", "拆分与防泄漏": "Splitting & Leakage Prevention",
    "训练模型": "Train Model", "评估与预测": "Evaluation & Prediction", "开始训练": "Train", "上一步": "Back", "继续": "Continue",
    "选择 CSV…": "Choose CSV…", "加载内置样本": "Load Demo Sample", "训练集": "Training Set", "验证集": "Development Set", "冻结测试集": "Held-out Test Set",
    "数据": "Data", "方法": "Method", "参数": "Parameters", "质检": "Quality Check", "导出": "Export", "结果": "Results",
    "加载数据": "Load Data", "运行分析": "Run Analysis", "分析完成": "Analysis complete", "分析失败": "Analysis failed",
    "编码维度": "Coding Dimensions", "维度": "Dimension", "编码": "Code", "名称": "Name", "定义": "Definition", "示例": "Example",
    "添加维度": "Add Dimension", "添加编码": "Add Code", "删除维度": "Delete Dimension", "删除编码": "Delete Code",
    "项目已保存": "Project Saved", "项目已打开": "Project Opened", "保存失败": "Save Failed", "无法打开项目": "Unable to Open Project",
    "导入媒体": "Import Media", "语音识别": "Transcribe Audio", "导入识别结果": "Import Transcript", "导出识别模板": "Export Transcript Template",
    "播放 / 暂停": "Play / Pause", "拆分片段": "Split Utterance", "合并片段": "Merge Utterances", "删除片段": "Delete Utterances", "新增片段": "Add Utterance",
    "多维话语编码": "Multidimensional Utterance Coding", "管理编码框架": "Manage Codebook", "生成 AI 编码建议": "Generate AI Coding Suggestions",
    "采用选中行的建议": "Accept Suggestions for Selected Rows", "当前多维编码框架": "Current Multidimensional Codebook",
    "导出到 Text Trainer": "Export to Text Trainer", "运行导出质检": "Validate Export", "导出 Text Trainer 数据包": "Export Text Trainer Package",
    "媒体播放器": "Media Player", "当前结果": "Current Results", "编码框架": "Codebook", "统计数据": "Statistics Data",
    "本次不保存 API Key（仅内存使用，退出即丢）": "Do not save API key (memory only; cleared on exit)",
    "未测试": "Not tested", "设置已保存": "Settings saved", "设置已保存（本次未保存 Key）": "Settings saved (API key not saved)",
    "Base URL": "Base URL", "模型": "Model", "API Key": "API Key", "温度": "Temperature", "Max tokens": "Max tokens", "显示": "Show", "隐藏": "Hide",
    "清空": "Clear", "停止": "Stop", "发送": "Send", "向智能体提问……（Ctrl+Enter 发送）": "Ask the assistant… (Ctrl+Enter to send)",
    "正在生成……": "Generating…", "没有待处理文本": "No Pending Text", "导出完成": "Export Complete", "任务进行中": "Task in Progress",
    "需要媒体文件": "Media Required", "请先导入视频或音频文件。": "Import a video or audio file first.",
    "需要训练与验证数据": "Training and Development Data Required", "请加载内置样本或分别导入 train.csv 和 dev.csv。": "Load the demo sample or import train.csv and dev.csv separately.",
    "训练完成": "Training complete", "训练失败": "Training failed", "尚未加载数据": "No data loaded", "未加载": "Not loaded",
    "MacBERT-base：环境入口暂未启用。后续需要安装 PyTorch / Transformers 并缓存 hfl/chinese-macbert-base 模型；当前版本提供可运行的 CPU 基线。":
        "MacBERT-base is not available yet. It requires PyTorch, Transformers, and the hfl/chinese-macbert-base model. This version provides a working CPU baseline.",
    "文本分析 · 本地训练工作台": "Text Analytics · Local Training Workbench", "切换浅色 / 深色": "Toggle Light / Dark Theme",
    "训练向导  /  ": "Training Wizard  /  ", "欢迎使用 Text Trainer": "Welcome to Text Trainer", "数据检查": "Data Validation",
    "项目描述": "Project Description", "准备数据": "Prepare Data", "检查质量": "Check Data Quality", "运行基线": "Run Baseline", "查看结果": "View Results",
    "导入 CSV，确认话语、人工裁定标签和实体字段。": "Import CSV files and verify utterances, adjudicated labels, and entity fields.",
    "发现缺失、重复、未定义标签和跨集合泄漏。": "Find missing values, duplicate records, undefined labels, and split leakage.",
    "使用字符 TF-IDF 与 LR 或 Linear SVM 训练。": "Train with character-level TF-IDF and Logistic Regression or Linear SVM.",
    "查看 Macro F1、分类报告和单条文本预测。": "Review Macro F1, the classification report, and single-text predictions.",
    "样本包是合成演示数据。它用于跑通界面和训练流程，不能作为科研性能证据。测试集保持冻结；训练与模型选择使用 train/dev。":
        "The sample package contains synthetic demo data for exercising the interface and training flow. It is not evidence of research performance. Keep the test set frozen; use train/dev for training and model selection.",
    "导入训练数据": "Import Training Data", "选择 CSV 文件，或加载随软件提供的演示样本。CSV 使用 UTF-8 编码。": "Choose CSV files or load the included demo sample. CSV files must use UTF-8 encoding.",
    "选择 train.csv / dev.csv / test.csv …": "Choose train.csv / dev.csv / test.csv…", "选择 CSV…": "Choose CSV…", "导入 CSV": "Import CSV", "CSV 文件 (*.csv)": "CSV Files (*.csv)",
    "加载内置样本": "Load Demo Sample", "训练集": "Training Set", "验证集": "Development Set", "冻结测试集": "Held-out Test Set", "未加载": "Not loaded",
    "数据质量检查": "Data Quality Check", "先解决阻断项。数据保留在本机；此页不上传任何内容。": "Resolve blocking issues first. Data stays on this computer and is not uploaded from this page.",
    "运行检查": "Run Checks", "算法": "Algorithm", "内置样本已预先隔离：train、dev、test 使用不同 group_id 和 session_id。重新拆分和比例约束将在后续版本完善。": "The bundled sample keeps train, dev, and test separate by group_id and session_id. Custom splitting and ratio controls are planned for a later version.",
    "评估与单条预测": "Evaluation & Single-Text Prediction", "Macro F1 为主指标。小样本结果仅用于验证流程，请勿作为正式研究结论。": "Macro F1 is the primary metric. Results from small samples validate the workflow only and should not be used as research findings.",
    "尚未训练模型": "No model trained", "输入一条对话文本": "Enter one utterance", "预测": "Predict", "模型建议：%s  ·  请由研究者复核，不会写回原始人工编码。": "Model suggestion: %s · Review it manually. It will not overwrite human-coded data.",
    "请先导入数据。": "Import data first.", "演示数据已就绪": "Demo data is ready", "已加载 %d 条": "Loaded %d records", "检查完成": "Validation complete", "检查有阻断项": "Blocking issues found",
    "训练完成": "Training complete", "训练失败": "Training failed", "请先完成一次训练。": "Train a model first.", "请输入文本。": "Enter some text.",
    "数据未通过检查": "Data Validation Failed", "需要训练与验证数据": "Training and Development Data Required", "请加载内置样本或分别导入 train.csv 和 dev.csv。": "Load the demo sample or import train.csv and dev.csv separately.",
    "取消当前任务": "Cancel Current Task", "导入一段协作学习音视频，校订 ASR 转写并逐条编码，完成后导出 Text Trainer 可读取的数据。": "Import collaborative learning audio or video, review the ASR transcript, code each utterance, and export data for Text Trainer.",
    "项目": "Project", "尚未导入媒体": "No media loaded", "转写校订表 · 双击时间定位媒体，原始转写保留在项目记录中": "Transcript review · Double-click a time to seek; the raw transcript remains in the project.",
    "媒体与转写": "Media & Transcript", "编码数据": "Coding Data", "质检与导出": "Validation & Export", "AI 编码": "AI Coding",
    "导入媒体": "Import Media", "语音识别": "Transcribe Audio", "导入识别结果": "Import Transcript", "导出识别模板": "Export Transcript Template",
    "新增片段": "Add Utterance", "拆分片段": "Split Utterance", "合并片段": "Merge Utterances", "删除片段": "Delete Utterances", "播放 / 暂停": "Play / Pause",
    "导入视频或音频": "Import Video or Audio", "媒体文件 (*.mp4 *.mov *.mkv *.avi *.wav *.mp3 *.m4a *.flac *.aac *.ogg *.webm);;所有文件 (*)": "Media Files (*.mp4 *.mov *.mkv *.avi *.wav *.mp3 *.m4a *.flac *.aac *.ogg *.webm);;All Files (*)",
    "保存识别数据导入模板": "Save Transcript Import Template", "导入语音识别结果": "Import Transcription Results", "识别结果 (*.xlsx *.xlsm *.csv *.json)": "Transcription Results (*.xlsx *.xlsm *.csv *.json)", "Excel (*.xlsx)": "Excel (*.xlsx)",
    "打开 Text Coder 项目": "Open Text Coder Project", "保存 Text Coder 项目": "Save Text Coder Project", "Text Coder 项目 (*.json)": "Text Coder Project (*.json)",
    "无法打开项目": "Unable to Open Project", "媒体导入失败": "Media Import Failed", "模板保存失败": "Template Save Failed", "设置保存失败": "Settings Save Failed",
    "追加导入": "Append Import", "当前已有 %d 条话语，是否追加 %d 条导入记录？": "There are already %d utterances. Append %d imported records?",
    "需要媒体文件": "Media Required", "请先导入视频或音频文件。": "Import a video or audio file first.", "需要配置语音识别": "Transcription Setup Required", "请在“模型设置”中补齐当前服务商的必填字段。": "Complete the required fields for the selected provider in Model Settings.",
    "追加转写结果": "Append Transcript Results", "当前项目已有 %d 条话语。新识别结果会追加到现有片段，不会覆盖它们。继续吗？": "This project already contains %d utterances. New transcription results will be appended without replacing them. Continue?",
    "任务进行中": "Task in Progress", "请等待当前后台任务完成。": "Wait for the current background task to finish.", "任务失败": "Task Failed",
    "选择片段": "Select Utterances", "请先选择已经听取并校对的片段。": "Select utterances that have been listened to and reviewed first.", "文本为空": "Empty Text", "有选中片段没有文本，不能标记为已校订。": "Some selected utterances have no text and cannot be marked as reviewed.",
    "时间范围太短": "Time Range Too Short", "请先修正片段起止时间，再执行拆分。": "Correct the utterance start and end times before splitting.", "请按住 Ctrl 选择至少两个话语片段。": "Hold Ctrl and select at least two utterances.",
    "无法合并": "Cannot Merge", "选中片段的说话人或人工标签不同。请先统一这些字段，避免合并时丢失编码信息。": "The selected utterances have different speakers or human labels. Align these fields first to avoid losing coding information.",
    "需要配置 AI 编码": "AI Coding Setup Required", "请先在模型设置中填写编码模型 API。": "Enter the coding model API details in Model Settings first.", "请先校订转写": "Review Transcript First",
    "%d 条 ASR 片段尚未标记为已校订。请先检查文本与时间，再生成 AI 编码建议。": "%d ASR utterance(s) are not marked as reviewed. Check the text and timing before generating AI coding suggestions.", "没有待处理文本": "No Pending Text", "请先完成转写校订。": "Review the transcript first.",
    "存在阻断项": "Blocking Issues Found", "请先修复阻断项后再导出。提醒项可在查看后继续。": "Resolve blocking issues before exporting. You can review warnings and continue.",
    "存在待补充信息": "Additional Information Needed", "%d 项提醒：包括未编码或缺少说话人。仍导出草稿供 Text Trainer 继续处理？": "%d warning(s), including uncoded items or missing speakers. Export a draft for Text Trainer to continue?",
    "选择 Text Trainer 导出目录": "Choose Text Trainer Export Folder", "导出完成": "Export Complete", "Text Trainer 数据包已写入：\n%s\n\n项目存档仍保留原始转写、建议和修订记录。": "The Text Trainer package was written to:\n%s\n\nThe project archive still contains the original transcript, suggestions, and revision history.",
    "后台任务进行中": "Background Task Running", "请等待语音识别或 AI 编码请求结束后再关闭。": "Wait for transcription or AI coding to finish before closing.",
    "语音识别与 AI 编码设置": "Transcription & AI Coding Settings", "分别配置转写服务与编码建议模型。密钥仅用于发送请求，不写入项目和导出文件。": "Configure transcription and coding suggestion services. API keys are only used for requests and are not stored in project or export files.",
    "直接上传本地音频，支持 100 MB、最长 2 小时；返回句段和词级时间戳。": "Upload local audio directly (up to 100 MB and 2 hours). Returns utterance and word-level timestamps.",
    "同步 multipart 上传本地文件；单文件上限 25 MB。": "Upload the local file synchronously using multipart; maximum file size is 25 MB.", "兼容 /chat/completions 的对话模型。模型只能给出编码建议，需人工复核。": "A /chat/completions-compatible model. It can only suggest codes, which must be reviewed by a person.",
    "当前环境没有 DPAPI。若保存 key，将以明文写入本机设置文件。": "DPAPI is unavailable in this environment. If saved, the key will be stored in plain text in the local settings file.",
    "新项目 · 尚未保存": "New Project · Unsaved", "未命名项目": "Untitled Project", "未保存": "Unsaved", "已自动保存": "Saved automatically", "已保存": "Saved",
    "自动保存失败：%s": "Automatic save failed: %s", "已创建新项目": "New project created", "媒体已导入；正在后台计算文件指纹": "Media imported; calculating file fingerprint in the background",
    "识别数据导入模板已保存": "Transcript import template saved", "已导入 %d 条；%d 条待补说话人，%d 条待复核时间。": "Imported %d item(s); %d need a speaker and %d need timing review.",
    "多维编码框架已更新": "Multidimensional codebook updated", "模型设置已保存": "Model settings saved", "新增空白片段，请补充文本和标签": "Blank utterance added. Enter text and labels.",
    "正在请求语音识别…": "Transcription in progress…", "正在计算媒体文件指纹…": "Calculating media fingerprint…", "正在生成编码建议…": "Generating coding suggestions…",
    "已请求取消；当前 HTTPS 请求结束后会丢弃结果。": "Cancellation requested. The result will be discarded when the current HTTPS request ends.",
    "转写完成：%d 段。请校订文本、补充说话人并完成编码。": "Transcription complete: %d utterances. Review the text, add speakers, and finish coding.", "媒体指纹已计算": "Media fingerprint calculated", "后台任务失败：": "Background task failed: ",
    "已标记 %d 段转写为已校订。": "Marked %d utterance(s) as reviewed.", "收到 %d 条有效建议；请选择记录后点击“采用选中行的建议”。": "Received %d valid suggestion(s). Select records and choose “Accept Suggestions for Selected Rows”.",
    "已人工确认 %d 条建议。": "Manually confirmed %d suggestion(s).", "质检发现 %d 个阻断项、%d 个提醒项。": "Validation found %d blocking issue(s) and %d warning(s).", "已导出 %s": "Exported to %s",
    "项目名称": "Project Name", "小组 ID": "Group ID", "任务 ID": "Task ID", "阶段 ID": "Phase ID", "编码员 ID": "Annotator ID", "编码本版本": "Codebook Version",
    "%d 条": "%d records", "【%s】%d 条": "[%s] %d records", "  阻断：": "  Blocking: ", "  提醒：": "  Warning: ",
    "  通过：字段、文本、标签与实体边界检查完成。": "  Passed: field, text, label, and entity boundary checks complete.",
    "【跨集合隔离】": "[Cross-split isolation]", "【跨集合隔离】group_id、session_id 与 event_id 未发现交叉。": "[Cross-split isolation] No overlap found in group_id, session_id, or event_id.",
    "训练集摘要": "Training summary", "记录数：%d\n组数：%d\n会话数：%d\n数据指纹：%s\n标签分布：\n%s": "Records: %d\nGroups: %d\nSessions: %d\nData fingerprint: %s\nLabel distribution:\n%s",
    "导入失败": "Import Failed", "先前往“数据检查”查看必需字段、空文本、重复 ID 或泄漏问题。": "Review required fields, empty text, duplicate IDs, or leakage on the Data Validation page.",
    "内置演示样本": "Built-in demo sample", "输入一条对话文本": "Enter one utterance", "预测": "Predict", "已停止": "Stopped", "完成": "Complete", "出错": "Error",
    "尚未装载数据。": "No data loaded.", "组清单（改「组号」列可覆写 group_id，改完点应用）": "Groups (edit the Group ID column to override group_id, then click Apply)",
    "选择文件…": "Choose Files…", "从当前会话取数": "Load from Current Session", "导入编码框架 Excel": "Import Codebook from Excel",
    "框架：自动解析": "Codebook: auto-detected", "应用组号覆写": "Apply Group ID Overrides", "生成结果": "Generate Results", "取消计算": "Cancel Computation",
    "装载数据后这里会列出阻断、警告与提示。": "Blocking issues, warnings, and notes will appear here after data is loaded.",
    "导出统计表": "Export Tables", "导出图形": "Export Figures", "统计表 / 图 / config / audit / 模型卡共用同一 analysis_id 前缀。": "Tables, figures, config, audit, and model card share the same analysis_id prefix.",
    "R 交叉校验": "R Cross-check", "训练日志：逐 epoch 输出会打在这里（深度学习类方法）。": "Training log: epoch-by-epoch output for deep-learning methods appears here.",
    "用 R 复算并对照": "Recompute and Compare with R", "R 参考层用于佐证 Python 实现，缺席不影响本工作台。": "The R reference layer cross-checks the Python implementation; its absence does not affect this workbench.",
    "复制安装命令": "Copy Install Command", "安装 R 参考包": "Install R Reference Packages", "R 参考层：尚未探测。": "R reference layer: not detected yet.",
    "统计": "Statistics", "个体": "Individual", "小组": "Group", "编码个体1": "Coded Individual 1", "编码个体2": "Coded Individual 2",
    "数据汇总": "Data Summary", "时间": "Time", "说话人": "Speaker", "注释": "Notes", "输入注释...": "Enter notes…",
    "撤销": "Undo", "输出": "Output", "无效时间": "Invalid Time", "当前时间请按 时:分:秒 格式输入。": "Enter the current time in HH:MM:SS format.",
    "起始时间行不可删除": "The start-time row cannot be deleted", "删除选中行": "Delete Selected Rows", "成功": "Success", "数据已成功导出！": "Data exported successfully!", "错误": "Error",
    "AI 辅助编码（视频多模态）": "AI-Assisted Coding (Video + Multimodal)", "本地训练辅助编码（离线）": "Local Training-Assisted Coding (Offline)",
    "暂无打开记录": "No recent files", "清空最近打开记录": "Clear Recent Files", "关于 Learnlysis": "About Learnlysis",
    "编辑智能体": "Edit Assistant", "新建智能体": "New Assistant", "例如：分析规划师": "e.g. Analysis Planner", "定义这个智能体的身份、任务与回答风格……": "Define this assistant's role, tasks, and response style…",
    "点击即可填入对话框的常用提问": "Common prompts (click to insert)", "尚未接入编码页。": "The coding page is not connected yet.",
    "取数失败": "Data Load Failed", "导入框架失败": "Codebook Import Failed", "只有批量上传的文件可以改组号。": "Group IDs can only be changed for files uploaded in batches.",
    "未解析出可用的编码列，请检查列名是否为 Time + S1Encoding_1。": "No usable coding columns found. Check that column names follow the pattern Time + S1Encoding_1.",
    "编码会话": "Coding Session", "组号覆写": "Group ID Override", "请先选择一个可用方法并生成结果。": "Select an available method and generate results first.",
    "尚未生成结果": "No results yet", "取消计算": "Cancel Computation", "计算中…": "Computing…", "未探测到 R 桥接层：无安装命令可复制。": "R bridge not detected; no install command is available to copy.",
    "导入识别数据": "Import Transcript Data", "下载导入模板": "Download Import Template", "新增话语": "Add Utterance", "合并选中": "Merge Selected", "标记已校订": "Mark as Reviewed",
    "开始 ms": "Start (ms)", "结束 ms": "End (ms)", "转写文本": "Transcript", "状态": "Status", "校订文本": "Reviewed Text", "AI 多维建议": "AI Suggestions by Dimension",
    "未编码": "Uncoded", "待建议": "Awaiting Suggestion", "未导入": "Not imported", "置信度 %s": "Confidence %s",
    "媒体：%s\n\n话语片段：%d\n已人工确认：%d\nAI 待复核：%d\n缺说话人：%d\n\nsession_id：%s\ngroup_id：%s\n编码本：%s":
        "Media: %s\n\nUtterances: %d\nManually confirmed: %d\nAI pending review: %d\nMissing speakers: %d\n\nsession_id: %s\ngroup_id: %s\nCodebook: %s",
    "用逐步向导完成协作学习对话文本分类的本地训练。先检查数据，再训练可复现的基线模型。": "Use this guided workflow to train a local classifier for collaborative learning dialogue. Check the data, then train a reproducible baseline model.",
    "以 group_id / session_id 作为实体边界。同一实体不可同时进入训练、验证或测试。": "Use group_id and session_id as entity boundaries. An entity cannot appear in training, development, and test sets at the same time.",
    "训练基线模型": "Train a Baseline Model", "训练仅使用 train.csv；dev.csv 用于模型比较。test.csv 不参与训练或调参。": "Only train.csv is used for training; dev.csv is used to compare models. test.csv is excluded from training and tuning.",
    "v0.1  ·  离线优先": "v0.1 · Offline First", "切换主题": "Toggle Theme", "模型设置": "Model Settings",
    "导出会生成 dataset.csv、labels.csv、dataset_manifest.json 和 README.md。每个自定义维度导出为 code_{dimension_id} 列；AI 建议不会写入人工编码列。":
        "Export creates dataset.csv, labels.csv, dataset_manifest.json, and README.md. Each custom dimension becomes a code_{dimension_id} column; AI suggestions do not enter human-coded columns.",
}


def current_language():
    return state.language()


def set_language(value):
    if value == state.language():
        return
    state.set_language(value)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QAction
    app = QApplication.instance()
    if app is not None:
        import theme
        app.setStyleSheet(theme.build_qss())
        for action in app.findChildren(QAction):
            if action.data() in ("zh-CN", "en-US"):
                action.setChecked(action.data() == value)
        for window in app.topLevelWidgets():
            translate_widget_tree(window)
            refresh = getattr(window, "retranslate_ui", None)
            if callable(refresh):
                refresh()


def tr(text):
    return _EN.get(text, text) if current_language() == "en-US" else text


_EN.update({
    "图例位置": "Legend Position", "图形正下方": "Below Figure", "图形右侧": "Right of Figure", "图内空白处": "Inside Empty Area",
    "图内右上": "Inside Top Right", "图内左下": "Inside Bottom Left", "不显示": "Hide", "图形尺寸": "Figure Size",
    "适应预览区": "Fit Preview", "放大 1.4 倍": "Scale 1.4×", "放大 1.8 倍": "Scale 1.8×", "放大 2.4 倍": "Scale 2.4×",
    "分析方法": "Analysis Method", "参数设置": "Parameter Settings", "自动质检": "Automatic Checks", "图形预览": "Figure Preview",
    "结果统计表": "Result Tables", "提示": "Notes", "来源文件": "Source File", "组号": "Group ID", "成员数": "Members",
    "时间点": "Time Point", "事件表": "Event Table", "事件来源": "Event Source", "平滑窗长(时点)": "Smoothing Window (Time Points)",
    "尚未装载可解析的编码数据": "No parseable coding data loaded", "尚未装载数据：选择 Group Excel 文件，或从当前编码会话取数。": "No data loaded. Choose a Group Excel file or load the current coding session.",
    "导出配置 / 质检 / 模型卡": "Export Config / Checks / Model Card", "交叉校验与训练日志": "Cross-check & Training Log",
    "训练日志": "Training Log", "量": "Measure", "绝对差": "Absolute Difference", "相对差": "Relative Difference", "判定": "Result",
    "最大滞后期": "Maximum Lag", "口径": "Definition", "同维（前状态→后状态）": "Within Dimension (Previous → Next)",
    "跨维（维度A_t → 维度B_t+k）": "Across Dimensions (A_t → B_t+k)", "两种都要": "Both", "|Z| 阈值": "|Z| Threshold",
    "选择文件": "Choose File", "从数据汇总取数": "Load from Data Summary", "时间列": "Time Column", "源列名": "Source Column",
    "学生 ID": "Student ID", "应用时间列与列映射": "Apply Time and Column Mapping", "列 → 学生 / 维度映射（可直接改单元格，留空即剔除该列）": "Column → Student / Dimension Mapping (edit cells; blank excludes a column)",
    "尚未装载数据：选择 Excel / CSV，或直接从数据汇总页取数。": "No data loaded. Choose Excel / CSV or load from Data Summary.",
    "编码排序": "Code Order", "频次降序": "Descending Frequency", "字母序": "Alphabetical", "时间序": "Time Order",
    "显示数值标签": "Show Value Labels", "另附跨学生合计表": "Include Cross-student Totals", "请先装载数据": "Load data first",
    "导出配置与质检": "Export Config and Checks", "文件名统一为 类型_方法_分析名_analysis_id，四类产物同前缀。": "Files use type_method_analysis_analysis_id; all four exports share the same prefix.",
    "输出分析格式文件": "Export Analysis File", "输出分析格式文件（含注释）": "Export Analysis File (with Notes)", "输出汇总表格": "Export Summary Table",
    "输出小组数据包": "Export Group Data Package", "计时": "Timer", "维度编辑": "Edit Dimensions", "导入框架": "Import Codebook",
    "起始时间": "Start Time", "当前时间": "Current Time", "时间间隔": "Time Interval", "下一步建议": "Next-step Suggestions",
    "记录几行编码后，这里会按你的历史给出下一步候选码。": "After you code a few rows, candidate codes based on your history will appear here.",
    "序列自动补全，非视频识别；点选即填入，仍需你确认后「记录」。": "Sequence completion, not video recognition. Select a suggestion, then confirm it before recording.",
    "片段时长 10 秒": "Clip Length: 10 Seconds", "视频列表": "Video List", "编码与汇总": "Coding & Summary", "分析取数": "Analysis Data",
    "结果导出": "Export Results", "立即保存存档": "Save Archive Now", "自动保存": "Auto-save", "重新载入存档": "Reload Archive",
    "清除存档文件": "Clear Archive File", "打开存档所在文件夹": "Open Archive Folder", "● 自动保存已停用（测试环境）": "● Auto-save disabled (test environment)",
    "A 状态与轨迹（1）": "A States & Trajectories (1)", "A 状态与轨迹（3）": "A States & Trajectories (3)",
    "B 转移与序列（2）": "B Transitions & Sequences (2)", "B 转移与序列（4）": "B Transitions & Sequences (4)",
    "C 距离与聚类（3）": "C Distance & Clustering (3)", "D 网络（2）": "D Networks (2)", "E 概率模型（1）": "E Probabilistic Models (1)",
    "F 预测与机器学习（1）": "F Prediction & Machine Learning (1)", "F 预测与机器学习（4）": "F Prediction & Machine Learning (4)",
    "G 组间比较（1）": "G Between-group Comparison (1)", "A 基础频次（3）": "A Basic Frequency (3)", "B 时间分布（2）": "B Time Distribution (2)",
    "C 时间轨迹（6）": "C Temporal Trajectories (6)", "D 序列描述（3）": "D Sequence Descriptions (3)",
    "E 共现描述（1）": "E Co-occurrence (1)", "F 路径呈现（1）": "F Pathways (1)", "G 熵与多样性（3）": "G Entropy & Diversity (3)",
    "组状态分布": "Group State Distribution", "组内熵": "Within-group Entropy", "同步度": "Synchrony", "领先-滞后": "Lead-Lag",
    "组级转移 + LSA": "Group Transitions + LSA", "组内异质性": "Within-group Heterogeneity", "组级 ENA / ONA": "Group ENA / ONA",
    "交互网络 SNA": "Interaction Network SNA", "角色发现（轨迹 + 网络聚类）": "Role Discovery (Trajectories + Network Clustering)",
    "跨组比较 / 组聚类 / 组结果预测": "Between-group Comparison / Clustering / Outcome Prediction",
    "滞后序列分析 LSA": "Lag Sequential Analysis (LSA)", "一阶马尔可夫链": "First-order Markov Chain",
    "高阶 / 变阶马尔可夫": "Higher / Variable-order Markov", "n-gram / 频繁子序列": "n-gram / Frequent Subsequences",
    "序列距离 OM/HAM/DHD": "Sequence Distance OM/HAM/DHD", "序列聚类 AHC / k-medoids": "Sequence Clustering AHC / k-medoids",
    "DTW 序列距离": "DTW Sequence Distance", "隐马尔可夫模型 HMM / MHMM": "Hidden Markov Model HMM / MHMM",
    "变点 / 分段": "Change Points / Segmentation", "共现网络 ENA / 网络坐标": "Co-occurrence Network ENA / Coordinates",
    "有向时序网络 ONA": "Directed Temporal Network ONA", "特征型机器学习（分类）": "Feature-based Machine Learning (Classification)",
    "循环/卷积网络 next-state": "Recurrent / Convolutional Network: Next State", "Transformer 自监督表征": "Transformer Self-supervised Representation",
    "时序图注意力网络（GAT）": "Temporal Graph Attention Network (GAT)",
    "一键导出全部": "Export All", "BH-FDR 校正": "BH-FDR Correction", "记录": "Record", "导入视频": "Import Video",
    "当前编码页": "Current Coding Page", "输出编码表": "Export Coding Table", "小组交互页": "Group Interaction Page",
    "导出 AI 对话 / 分析规划": "Export AI Conversation / Analysis Plan", "导入编码框架": "Import Codebook",
    "附带上下文：数据规模、编码框架、方法清单": "Context: data overview, codebook, method list",
    "Key 经 Windows DPAPI 加密后存于 data/ai_settings.json，只有当前 Windows 账户能解开；绝不进存档/导出/日志。共用电脑请勾选「本次不保存」。":
        "On Windows, the key is encrypted with DPAPI in data/ai_settings.json for this account only. It is excluded from archives, exports, and logs. On a shared computer, choose 'Do not save'.",
    "你好，我是 AI 助手。先在左侧「模型与密钥」填入你的 API 信息并测试连接，再选一个智能体开始对话。所有调用走你自己的 key、由你自付费用。":
        "Hello, I'm the AI assistant. Enter your API settings and test the connection on the left, then choose an assistant. Requests use your own API key and billing account.",
    "你是协作学习观察研究的编码框架设计顾问。先检查研究问题、研究现场、可用证据与分析单位，再建议适用的文献框架和操作化方案。每条建议必须区分文献概念、针对当前研究的改编与 AI 新建议；优先给出可观察定义、纳入/排除规则和例证，不得杜撰文献、DOI 或原文定义。所有输出都是待研究者确认的草稿，不得声称已经验证，也不得直接覆盖编码数据。":
        "You advise on coding frameworks for observational research in collaborative learning. Review the research question, setting, available evidence, and unit of analysis before recommending literature-based frameworks and operationalizations. Distinguish literature concepts, adaptations for this study, and new AI suggestions. Give observable definitions, inclusion and exclusion rules, and examples. Do not invent sources, DOIs, or original definitions. All output is a draft for researcher review; do not claim validation or overwrite coded data.",
    "本方法没有 R 参考结果（注册表未声明 r_reference）：点「用 R 复算并对照」现跑，或以 Python 为准。":
        "No R reference result is registered for this method. Choose 'Recompute and Compare with R', or use the Python result.",
    "本方法没有 R 参考结果（注册表声明 lsa_residual.R）：点「用 R 复算并对照」现跑，或以 Python 为准。":
        "No R reference result is available yet for lsa_residual.R. Choose 'Recompute and Compare with R', or use the Python result.",
    "注意：编码个体的时间需要保持一致。下表按标签页顺序并排陈列各页数据（不含注释列），行号一一对应。":
        "Note: coded individuals must use aligned time points. The table places pages side by side in tab order, excluding notes, with matching row numbers.",
    "M01 Frequency 频次": "M01 Frequency", "M02 Percentage 比例": "M02 Percentage", "M03 Duration 持续时长": "M03 Duration",
    "M04 Time-bin Frequency 时间窗口频次": "M04 Time-bin Frequency", "M05 Time-bin Percentage 时间窗口构成": "M05 Time-bin Percentage",
    "M06 Timeline 时间轴": "M06 Timeline", "M07 Multi-dimensional Timeline 多维同步时间轴": "M07 Multidimensional Timeline",
    "M08 Heatmap 时间窗口热图": "M08 Time-bin Heatmap", "M09 State Distribution 状态分布": "M09 State Distribution",
    "M10 Modal State 主导状态": "M10 Modal State", "M11 Sequence Index 序列索引图": "M11 Sequence Index",
    "M12 Sequence Frequency 序列频次": "M12 Sequence Frequency", "M13 Mean Time 平均状态占用时间": "M13 Mean State Time",
    "M14 Spell / Run 连续段": "M14 Spell / Run", "M15 Co-occurrence 共现 / 同步": "M15 Co-occurrence / Synchrony",
    "M16 Alluvial / Sankey 阶段路径": "M16 Alluvial / Sankey Pathways", "M17 Entropy 状态熵": "M17 State Entropy",
    "M18 Switch & Retention 切换率/保持率": "M18 Switching & Retention", "M19 Transition Entropy 条件熵": "M19 Conditional Transition Entropy",
    "每一时刻全组在做什么？\n口径：group_state_share(t) = 该时刻取该码的成员数 / 该时刻有码成员数\n当前不可用：尚未装载可解析的编码数据":
        "What is the group doing at each time point?\nDefinition: group_state_share(t) = members with this code / coded members at that time\nUnavailable: no parseable coding data loaded",
    "哪些编码之间的先后衔接不偶然？\n口径：E_ij = row_i·col_j/N（N 不含对角）；Z = (O−E)/√(E(1−row/N)(1−col/N))\n当前不可用：尚未装载可解析的编码数据":
        "Which code transitions are unlikely to occur by chance?\nDefinition: E_ij = row_i·col_j/N (excluding diagonal); Z = (O−E)/√(E(1−row/N)(1−col/N))\nUnavailable: no parseable coding data loaded",
    "每个编码在该学生该维度下一共出现了多少次？\n当前不可用：尚未装载可解析的编码数据":
        "How often does each code occur for this student and dimension?\nUnavailable: no parseable coding data loaded",
    "AI 标签仅作为建议。未填写人工裁定标签的记录会在质检中标出。":
        "AI labels are suggestions only. Records without human-adjudicated labels are flagged during validation.",
    "每个编码维度独立选择；AI 建议按维度显示，采用建议前仍需人工确认。":
        "Choose a code for each dimension separately. AI suggestions are shown by dimension and require human review before acceptance.",
    "话语功能": "Discourse Function",
    "服务商": "Provider", "模型名称": "Model Name", "识别引擎": "Recognition Engine", "语言代码": "Language Code", "音频格式（留空自动识别）": "Audio Format (blank for auto-detect)",
    "腾讯云录音文件识别极速版": "Tencent Cloud Flash ASR", "OpenAI 兼容接口": "OpenAI-compatible API",
    "中文普通话通用": "Mandarin Chinese (General)", "中英粤及多方言大模型（推荐）": "Multilingual Model: Mandarin, English, Cantonese & Dialects (Recommended)",
    "开启说话人分离（中文普通话、单声道）": "Enable Speaker Diarization (Mandarin, Mono Audio)",
    "在本机保存 API key（Windows 使用当前账户 DPAPI 加密）": "Save API key locally (Windows encrypts it with the current account's DPAPI)",
    "多维编码框架": "Multidimensional Codebook", "导入 Excel": "Import Excel", "导出模板": "Export Template", "保存框架": "Save Codebook",
    "删除选中编码": "Delete Selected Codes", "维度 ID": "Dimension ID", "维度名称": "Dimension Name", "维度解释": "Dimension Description",
    "解释": "Description", "正例": "Positive Example",
    "学习行为编码与分析系统": "Collaborative Learning Coding and Analysis", "兼容工作现场": "Legacy Workspace",
    "小组交互": "Group Interaction",
})

# Distinct English captions keep an English-first session reversible to Chinese.
_EN.update({"组号": "Group Number", "口径": "Computation Rule", "新增话语": "Add New Utterance",
            "不显示": "Do Not Show", "出错": "An Error Occurred", "提示": "Notice", "导入框架": "Import Framework"})

_EN.update({
    "0 = 不平滑；>1 时按该窗长做移动平均": "0 = no smoothing; values above 1 use a moving average of that window length",
    "1.96 ⇔ 双侧 α=0.05": "1.96 ⇔ two-sided α=0.05", "CRAN 镜像（可留空），如 https://mirrors.tuna.tsinghua.edu.cn/CRAN/": "CRAN mirror (optional), e.g. https://mirrors.tuna.tsinghua.edu.cn/CRAN/",
    "GRP 系列方法，按 A 状态与轨迹 … G 组间比较分组；不可用的方法会写明原因": "GRP methods are grouped from A States & Trajectories to G Between-group Comparison. Unavailable methods explain why.",
    "IND 系列方法，按 A 状态与轨迹 … G 组间比较分组；不可用的方法会写明原因": "IND methods are grouped from A States & Trajectories to G Between-group Comparison. Unavailable methods explain why.",
    "M01–M16，按 A 基础频次 … F 路径呈现分组": "M01–M16, grouped from A Basic Frequency to F Pathways",
    "lag=k 含义：前一个状态取第 t 个有效编码，后一个取第 t+k 个": "lag=k compares the valid code at time t with the code at time t+k",
    "仅对 PNG 生效": "PNG only", "分": "min", "秒": "sec", "时": "hr",
    "双击单元格可修改编码/注释；选中行后按 Delete 键或右键菜单可删除该行": "Double-click a cell to edit codes or notes. Select a row, then press Delete or use the context menu to remove it.",
    "双击播放": "Double-click to play", "图形按内容自动放大，超出预览区时可滚动查看完整内容": "Figures grow to fit their content; scroll the preview to see the full figure.",
    "图按内容自动放大，超出预览区时可滚动查看完整内容": "The figure grows to fit its content; scroll the preview to see it all.",
    "在新窗口中全屏查看图形（Ctrl+滚轮缩放，Esc 关闭）": "Open the figure full-screen (Ctrl + wheel to zoom, Esc to close)",
    "导出 Group_Info / Students / Behavior_State / Interaction_Event / Codebook / Enums 多 sheet 文件": "Export a multi-sheet Group_Info / Students / Behavior_State / Interaction_Event / Codebook / Enums file",
    "导出 Time、S1Encoding_1 等标准列名，可直接用于后续数据分析（不含注释）": "Export standard columns such as Time and S1Encoding_1 for analysis (without notes)",
    "快进 5 秒": "Forward 5 seconds", "快退 5 秒": "Back 5 seconds", "打开视频文件": "Open video file", "播放进度": "Playback position", "播放速度": "Playback speed",
    "改为上下排列": "Switch to vertical layout", "新建一个维度，会先让你输入维度名": "Add a dimension; you will be asked for its name first",
    "新建编码个体": "Add coded individual", "最近使用过的文件": "Recently used files", "最近装载过的批次": "Recently loaded batches",
    "本方法输出的图不止一张时在此定位": "Select among this method's figures here",
    "标签放不下时图本来就会自己撑大（预览区滚动看）；这一档再整体放大，字号与格子都会更松": "Figures already expand when labels need room. This setting scales the entire figure further, including text and cells.",
    "每个片段自动播放的时长": "Automatic playback duration for each clip", "深度学习类方法逐 epoch 检查取消标志，已完成的折仍会报出": "Deep learning checks cancellation each epoch; completed folds are still reported",
    "清空编码框架与已记录的编码数据": "Clear the codebook and recorded coding data",
    "独立分析单元 = 组号__学生：不同组的 S1 是不同的人，绝不合并": "Independent analysis unit = group ID__student. S1 in different groups refers to different people and is never merged.",
    "用已录编码在本地训练分类头（不联网）；视听重后端 mediapipe/faster-whisper 缺席时对话框会写明安装命令，仍可用序列特征训练": "Train a local classifier from recorded codes without network access. If mediapipe or faster-whisper is missing, the dialog gives installation instructions; sequence features still work.",
    "该口径把 S1/S2 当作同类观测合并计数，不代表任何单个学生": "This definition pools S1/S2 as observations and does not represent one individual student",
    "跳转到指定时间": "Go to a time", "跳转到的分钟": "Minute to seek", "跳转到的小时（留空视为 0）": "Hour to seek (blank means 0)", "跳转到的秒": "Second to seek",
    "选“图形正下方 / 图形右侧”时会在图上单独留一条边距，不会压住坐标区": "Choosing Below Figure or Right of Figure reserves a separate margin and does not cover the plot area",
    "重播上一片段": "Replay previous clip", "需在「AI 助手」页配置 OpenAI 兼容 API；抽帧需可选依赖 opencv-python 或 ffmpeg，缺席时对话框会写明降级原因": "Configure an OpenAI-compatible API in AI Assistant. Frame extraction requires opencv-python or ffmpeg; the dialog explains any missing backend.",
    "本次状态记录所属的 Context": "Context for this state record",
    "播放": "Play",
})

_SOURCE_BY_EN = {english: chinese for chinese, english in _EN.items()}


def source_text(value):
    """Return the language-independent Chinese UI source for archive contracts."""
    return _source_text(value)


def _source_text(value):
    import re
    value = value.replace("&&", "&")
    if value.startswith("Archive file: "):
        return "存档文件：" + value[len("Archive file: "):]
    if value in _EN:
        return value
    if value in _SOURCE_BY_EN:
        return _SOURCE_BY_EN[value]
    individual = re.fullmatch(r"Coded Individual (\d+)( · (Time|Timer))?", value)
    if individual:
        suffix = "·时间" if individual.group(3) == "Time" else "·计时" if individual.group(3) == "Timer" else ""
        return "编码个体" + individual.group(1) + suffix
    separated = re.split(r"(\s*—\s*)", value, maxsplit=1)
    if len(separated) == 3:
        title, separator, detail = separated
        original_title = _source_text(title)
        original_detail = _source_text(detail)
        if (original_title, original_detail) != (title, detail):
            return original_title + separator + original_detail
    coded = re.fullmatch(r"((?:IND|GRP|M)\d{2} )(.+)", value)
    if coded:
        original = _source_text(coded.group(2))
        if original != coded.group(2):
            return coded.group(1) + original
    numbered = re.fullmatch(r"(\d{2}\s{2,})(.+)", value)
    if numbered:
        original = _source_text(numbered.group(2))
        if original != numbered.group(2):
            return numbered.group(1) + original
    return value


def _translated(value):
    import re
    if current_language() != "en-US":
        return value
    if value.startswith("存档文件："):
        return "Archive file: " + value[len("存档文件："):]
    if value in _EN:
        return _EN[value]
    individual = re.fullmatch(r"编码个体(\d+)(·(时间|计时))?", value)
    if individual:
        suffix = " · Time" if individual.group(3) == "时间" else " · Timer" if individual.group(3) == "计时" else ""
        return "Coded Individual " + individual.group(1) + suffix
    separated = re.split(r"(\s*—\s*)", value, maxsplit=1)
    if len(separated) == 3:
        title, separator, detail = separated
        return _translated(title) + separator + _translated(detail)
    coded = re.fullmatch(r"((?:IND|GRP|M)\d{2} )(.+)", value)
    if coded:
        return coded.group(1) + _EN.get(coded.group(2), coded.group(2))
    numbered = re.fullmatch(r"(\d{2}\s{2,})(.+)", value)
    if numbered:
        return numbered.group(1) + _EN.get(numbered.group(2), numbered.group(2))
    return value


def _replace(widget, key, current, setter, mnemonic=False):
    slots = widget.property("_ly_i18n_slots") or {}
    old = slots.get(key)
    source = old[0] if old and current == old[1] else _source_text(current)
    rendered = _translated(source)
    if mnemonic and current_language() == "en-US":
        rendered = rendered.replace("&", "&&")
    if rendered != current:
        setter(rendered)
    if source != current or source in _EN or old:
        slots[key] = (source, rendered)
        widget.setProperty("_ly_i18n_slots", slots)
    return rendered


def _fit_control(widget, text):
    """Keep translated controls readable at the shared MiSans font size."""
    from PySide6.QtWidgets import QAbstractButton, QComboBox
    from PySide6.QtGui import QFontMetrics
    if not isinstance(widget, (QAbstractButton, QComboBox)) or not text:
        return
    original = widget.property("_ly_i18n_min_width")
    if original is None:
        original = widget.minimumWidth()
        widget.setProperty("_ly_i18n_min_width", original)
        widget.setProperty("_ly_i18n_max_width", widget.maximumWidth())
    if current_language() == "zh-CN":
        widget.setMinimumWidth(original)
        widget.setMaximumWidth(widget.property("_ly_i18n_max_width"))
        return
    longest = text
    if isinstance(widget, QComboBox):
        longest = max((widget.itemText(i) for i in range(widget.count())), key=len, default=text)
    padding = 44 if isinstance(widget, QComboBox) else 30
    needed = QFontMetrics(widget.font()).horizontalAdvance(longest.replace("&", "")) + padding
    widget.setMinimumWidth(max(original, needed))
    if widget.maximumWidth() < needed:
        widget.setMaximumWidth(needed)


def translate_widget_tree(root):
    """Switch visible text in place and update geometry without rebuilding the window."""
    from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QTabWidget,
                                 QGroupBox, QAbstractButton, QTableWidget, QTreeWidget)
    from PySide6.QtCore import QObject
    from PySide6.QtGui import QAction
    for widget in [root, *root.findChildren(QObject)]:
        if isinstance(widget, QWidget):
            if widget.isWindow() and widget.windowTitle():
                _replace(widget, "window_title", widget.windowTitle(), widget.setWindowTitle)
            if widget.toolTip():
                _replace(widget, "tooltip", widget.toolTip(), widget.setToolTip)
        if isinstance(widget, (QLabel, QAbstractButton, QAction)) and widget.text():
            shown = _replace(widget, "text", widget.text(), widget.setText,
                             mnemonic=isinstance(widget, (QAbstractButton, QAction)))
            _fit_control(widget, shown)
            if isinstance(widget, QLabel):
                original_wrap = widget.property("_ly_i18n_word_wrap")
                if original_wrap is None:
                    original_wrap = widget.wordWrap()
                    widget.setProperty("_ly_i18n_word_wrap", original_wrap)
                widget.setWordWrap(True if current_language() == "en-US" and len(shown) > 42 else original_wrap)
        if isinstance(widget, QGroupBox) and widget.title():
            _replace(widget, "title", widget.title(), widget.setTitle)
        if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)) and widget.placeholderText():
            _replace(widget, "placeholder", widget.placeholderText(), widget.setPlaceholderText)
        if isinstance(widget, QComboBox):
            if not widget.property("i18nData"):
                blocked = widget.blockSignals(True)
                try:
                    for index in range(widget.count()):
                        _replace(widget, f"item:{index}", widget.itemText(index),
                                 lambda value, i=index, w=widget: w.setItemText(i, value))
                finally:
                    widget.blockSignals(blocked)
            _fit_control(widget, widget.currentText())
        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                _replace(widget, f"tab:{index}", widget.tabText(index),
                         lambda value, i=index, w=widget: w.setTabText(i, value), mnemonic=True)
            widget.tabBar().updateGeometry()
        if isinstance(widget, QTableWidget):
            blocked = widget.blockSignals(True)
            try:
                for index in range(widget.columnCount()):
                    item = widget.horizontalHeaderItem(index)
                    if item is not None:
                        shown = _replace(widget, f"column:{index}", item.text(), item.setText)
                        width = widget.horizontalHeader().fontMetrics().horizontalAdvance(shown) + 26
                        if widget.columnWidth(index) < width:
                            widget.setColumnWidth(index, width)
            finally:
                widget.blockSignals(blocked)
        if isinstance(widget, QTreeWidget):
            for index in range(widget.columnCount()):
                item = widget.headerItem()
                if item is not None:
                    _replace(widget, f"tree-column:{index}", item.text(index),
                             lambda value, i=index, h=item: h.setText(i, value))
        if hasattr(widget, "updateGeometry"):
            widget.updateGeometry()


def install_language_filter(app):
    """Catch dialogs and secondary windows created after the main window."""
    from PySide6.QtCore import QObject, QEvent
    class LanguageFilter(QObject):
        def eventFilter(self, watched, event):
            if event.type() == QEvent.Show and getattr(watched, "isWindow", lambda: False)():
                translate_widget_tree(watched)
            return False
    language_filter = LanguageFilter(app)
    app.installEventFilter(language_filter)
    app._learnlysis_language_filter = language_filter


def attach_language_menu(button, parent, title="Learnlysis"):
    """Attach a language menu that updates all open windows immediately."""
    from PySide6.QtWidgets import QMenu
    from PySide6.QtGui import QActionGroup
    menu = QMenu(parent)
    group = QActionGroup(menu)
    group.setExclusive(True)
    for code, label in LANGUAGES:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(current_language() == code)
        action.setData(code)
        group.addAction(action)
        action.triggered.connect(
            lambda _checked=False, value=code: _select_language(value, parent, title))
    button.setMenu(menu)
    return menu


def _select_language(value, parent, title):
    if value == current_language():
        return
    set_language(value)
