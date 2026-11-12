"""Smoke tests：覆盖各核心模块的导入和基础实例化。

目标：
- 每个模块能成功 import（无 ImportError / SyntaxError）
- 关键类能用 QApplication 实例化（不崩溃）
- 纯数据/配置模块能在无 GUI 的情况下导入

运行方式：
    cd version/v5
    pytest test_smoke_imports.py -v
"""

import importlib
import sys
import os
import pytest

# 确保 version/v5 在路径中
_V5_DIR = os.path.dirname(os.path.abspath(__file__))
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一层：纯配置/数据模块（不需要 QApplication）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConfigModules:
    """配置和纯数据模块应能在无 GUI 环境下导入。"""

    def test_import_config(self):
        """config.py 提供全局路径和元信息。"""
        import config
        assert hasattr(config, "APP_NAME")
        assert hasattr(config, "APP_VERSION")
        assert hasattr(config, "ARCHIVE_FILE")
        assert config.APP_NAME == "Learnlysis"

    def test_import_statistic_visualization_methods(self):
        """方法注册表是纯数据声明，不依赖 GUI。"""
        mod = importlib.import_module("statistic_visualization_methods")
        # 注册表应有 METHOD_SPECS 或类似导出
        assert hasattr(mod, "METHOD_SPECS") or hasattr(mod, "MethodSpec")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二层：主题/图标模块（需要 QApplication）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThemeModules:
    """主题和图标模块需要 QApplication 存在。"""

    def test_import_icon_provider(self, qapp):
        """icon_provider 提供 SVG 图标渲染。"""
        import icon_provider
        assert hasattr(icon_provider, "icon")
        assert hasattr(icon_provider, "ICON_PATHS")

    def test_import_theme(self, qapp):
        """theme.py 是统一样式层。"""
        import theme
        assert hasattr(theme, "TOKENS")
        assert hasattr(theme, "apply_theme")
        assert hasattr(theme, "load_fonts")

    def test_theme_load_fonts(self, qapp):
        """字体加载不应崩溃。"""
        import theme
        theme.load_fonts()

    def test_theme_load_mode(self, qapp):
        """模式加载（浅色/深色）不应崩溃。"""
        import theme
        theme.load_mode()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三层：功能模块（需要 QApplication + 主题初始化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFeatureModules:
    """功能模块的导入和基础实例化。"""

    @pytest.fixture(autouse=True)
    def _init_theme(self, qapp):
        """确保主题在模块实例化前加载。"""
        import theme
        theme.load_fonts()
        theme.load_mode()
        theme.apply_theme(qapp)

    def test_import_about_dialog(self):
        """about_dialog 应能导入。"""
        mod = importlib.import_module("about_dialog")
        assert hasattr(mod, "AboutDialog")

    def test_import_dimension_panel(self):
        """dimension_panel 是三个模块的共用基础。"""
        mod = importlib.import_module("dimension_panel")
        assert hasattr(mod, "DimensionPanel")
        assert hasattr(mod, "DimensionCard")

    def test_import_video_player(self):
        """video_player 导入时会设置多媒体后端环境变量。"""
        mod = importlib.import_module("video_player")
        assert hasattr(mod, "VideoPlayerModule")
        assert hasattr(mod, "QAudioOutput")
        # 验证环境变量已被设置
        assert os.environ.get("QT_MULTIMEDIA_PREFERRED_PLUGINS") == "windowsmediafoundation"

    def test_import_data_encoder(self):
        """data_encoder 依赖 dimension_panel。"""
        mod = importlib.import_module("data_encoder")
        assert hasattr(mod, "DataEncoderModule")

    def test_import_summary_tab(self):
        """summary_tab 依赖 data_encoder。"""
        mod = importlib.import_module("summary_tab")
        assert hasattr(mod, "SummaryTab")

    def test_import_analysis_workbench(self):
        """analysis_workbench 是两个模式共用五卡工作台。"""
        mod = importlib.import_module("analysis_workbench")
        assert hasattr(mod, "ModeAnalysisWorkbench")

    def test_import_individual_analysis(self):
        """个体模式分析（原聚类/滞后序列的替代者之一）。"""
        mod = importlib.import_module("individual_analysis")
        assert hasattr(mod, "IndividualAnalysisModule")

    def test_import_group_analysis(self):
        """小组模式分析。"""
        mod = importlib.import_module("group_analysis")
        assert hasattr(mod, "GroupAnalysisModule")

    def test_import_analysis_export(self):
        """IND/GRP 导出层（统计表 + 模型卡 + R 校验）。"""
        mod = importlib.import_module("analysis_export")
        assert hasattr(mod, "export_all")
        assert hasattr(mod, "build_model_card")

    def test_old_modules_removed(self):
        """cluster_analysis / lag_sequence 已删除，不留兼容壳。"""
        for name in ("cluster_analysis", "lag_sequence"):
            assert not os.path.exists(os.path.join(_V5_DIR, name + ".py")), name
            sys.modules.pop(name, None)
            with pytest.raises(ImportError):
                importlib.import_module(name)

    def test_import_statistic_visualization(self):
        """statistic_visualization 是统计模块主入口。"""
        mod = importlib.import_module("statistic_visualization")
        assert hasattr(mod, "StatisticVisualizationModule")

    def test_import_statistic_visualization_data(self):
        """统计模块的数据模型。"""
        mod = importlib.import_module("statistic_visualization_data")
        assert hasattr(mod, "CodingDataset")

    def test_import_statistic_visualization_stats(self):
        """统计模块的计算函数。"""
        mod = importlib.import_module("statistic_visualization_stats")

    def test_import_statistic_visualization_plots(self):
        """统计模块的绘图函数。"""
        mod = importlib.import_module("statistic_visualization_plots")

    def test_import_statistic_visualization_export(self):
        """统计模块的导出函数。"""
        mod = importlib.import_module("statistic_visualization_export")

    def test_import_main_window(self):
        """main_window 是最后导入的模块，依赖所有其他模块。"""
        mod = importlib.import_module("main_window")
        assert hasattr(mod, "MainWindow")

    # ---------- 功能二：AI 智能体模块（OpenAI 兼容、BYO key） ----------
    def test_import_ai_client(self):
        """ai_client：OpenAI 兼容 HTTP/SSE 客户端（纯标准库 + 可注入 transport）。"""
        mod = importlib.import_module("ai_client")
        assert hasattr(mod, "OpenAICompatClient") and hasattr(mod, "AiError")
        assert hasattr(mod, "system_message") and hasattr(mod, "user_multimodal")
        assert isinstance(mod.optional_backends(), dict)

    def test_import_ai_worker(self):
        """ai_worker：把一次对话搬到 QThread（网络绝不进 UI 线程）。"""
        mod = importlib.import_module("ai_worker")
        assert hasattr(mod, "AiWorker")

    def test_import_ai_settings(self):
        """ai_settings：data/ai_settings.json 读写 + 智能体档案（原子写）。"""
        mod = importlib.import_module("ai_settings")
        assert hasattr(mod, "AiSettings") and hasattr(mod, "settings")
        assert hasattr(mod.settings, "provider")

    def test_import_ai_context(self):
        """ai_context：把工作现场 + 方法注册表组装成 LLM 上下文。"""
        mod = importlib.import_module("ai_context")
        assert hasattr(mod, "build_messages") and hasattr(mod, "context_block")

    def test_import_ai_assistant(self):
        """ai_assistant：AI 助手外层标签页（档案/对话/设置三区）。"""
        mod = importlib.import_module("ai_assistant")
        assert hasattr(mod, "AiAssistantModule") and hasattr(mod, "AgentWizardDialog")

    def test_import_framework_designer(self):
        """framework_designer：研究问题 → 文献 → 草稿 → 质检 → 导出/应用闭环。"""
        mod = importlib.import_module("framework_designer")
        assert hasattr(mod, "FrameworkDesignerDialog")
        for name in ("framework_schema", "framework_library", "framework_quality",
                     "framework_prompt", "framework_store", "framework_export"):
            assert importlib.import_module(name) is not None

    # ---------- 功能一：视频自动编码（Phase 1–3） ----------
    def test_import_coding_assist(self):
        """coding_assist：Phase 1 序列辅助自动补全（一阶转移模型，零新依赖）。"""
        mod = importlib.import_module("coding_assist")
        assert hasattr(mod, "CodingSuggestBar") and hasattr(mod, "suggest_next")
        assert hasattr(mod, "build_transition_counts")

    def test_import_video_probe(self):
        """video_probe：Phase 2 独立离线抽帧（绝不动直播播放器）。"""
        mod = importlib.import_module("video_probe")
        assert hasattr(mod, "extract_frame") and hasattr(mod, "qimage_to_data_uri")
        assert hasattr(mod, "frames_available") and isinstance(mod.optional_backends(), dict)

    def test_import_auto_coding(self):
        """auto_coding：Phase 2 多模态建议码 + 人在环写回。"""
        mod = importlib.import_module("auto_coding")
        assert hasattr(mod, "AutoCodingDialog") and hasattr(mod, "apply_suggestions")
        assert hasattr(mod, "parse_suggestions") and hasattr(mod, "build_prompt")

    def test_import_auto_coding_train(self):
        """auto_coding_train：Phase 3 本地训练脚手架（sklearn + QThread + 优雅降级）。"""
        mod = importlib.import_module("auto_coding_train")
        assert hasattr(mod, "LocalTrainDialog") and hasattr(mod, "CodingTrainWorker")
        assert hasattr(mod, "build_dataset") and hasattr(mod, "train_coding_model")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四层：关键类实例化（验证 __init__ 不崩溃）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBasicInstantiation:
    """验证关键类能在离屏环境下实例化。

    注意：离屏平台下模态对话框 + 框架 churn 有已知的访问违例风险，
    因此这里只做最小实例化，不触发复杂交互。
    """

    @pytest.fixture(autouse=True)
    def _init_theme(self, qapp):
        import theme
        theme.load_fonts()
        theme.load_mode()
        theme.apply_theme(qapp)

    def test_instantiate_dimension_panel(self):
        """DimensionPanel 是最基础的共用组件。"""
        from dimension_panel import DimensionPanel
        panel = DimensionPanel()
        assert panel is not None

    def test_instantiate_about_dialog(self):
        """AboutDialog 应能构造。"""
        from about_dialog import AboutDialog
        dialog = AboutDialog()
        assert dialog is not None

    def test_instantiate_mode_workbenches(self):
        """两个工作台构造后应各自锁定模式，且未装载数据时不崩。"""
        import analysis_methods as am
        from individual_analysis import IndividualAnalysisModule
        from group_analysis import GroupAnalysisModule
        ind = IndividualAnalysisModule()
        grp = GroupAnalysisModule()
        assert ind.mode == am.MODE_INDIVIDUAL and grp.mode == am.MODE_GROUP
        assert ind.ctx is None and ind.result is None
        # 方法下拉必须按模式填充，且分组标题行不可选
        for module in (ind, grp):
            ids = [module.method_combo.itemData(i)
                   for i in range(module.method_combo.count())
                   if module.method_combo.itemData(i) is not None]
            assert ids, module.mode
            assert all(am.method_by_id(i) for i in ids)
            assert all(spec.mode == module.mode for spec in am.all_methods()
                       if spec.id in ids)

    def test_workbench_export_without_result_is_guarded(self):
        """没有结果时 `*_to` 系列只给空列表，不建目录也不写文件。"""
        from analysis_workbench import ModeAnalysisWorkbench
        directory = os.path.join(_V5_DIR, "_should_not_be_written")
        module = ModeAnalysisWorkbench()
        assert module.export_all_to(directory) == []
        assert module.export_statistics_to(os.path.join(directory, "x.xlsx")) == []
        assert module.export_chart_to(os.path.join(directory, "x.png")) == []
        assert module.export_metadata_to(os.path.join(directory, "x.json")) == []
        assert not os.path.exists(directory)
