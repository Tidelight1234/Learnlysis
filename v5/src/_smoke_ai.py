"""离屏冒烟：AI 助手全链路（mock transport，不连真网、不写工程 data/）。

覆盖：设置读写 / 档案 CRUD / 上下文组装 / 客户端解析与错误分支 / worker 启动-流式-取消-错误 /
AI 助手页发送闭环 / main_window 第 5 个标签页集成。全部用注入的假传输，绝不发真实请求；
offscreen 下 ai_settings.save() 被 _disk_writes_blocked 挡住，不碰工程 data/ai_settings.json。
"""
import io
import os
import re
import sys
import json
import time
import urllib.error

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QEventLoop, QRect, QTimer, Qt
from PySide6.QtWidgets import QApplication, QComboBox

import theme
import ai_client
import ai_context
import ai_worker
from ai_settings import settings
from config import AI_SETTINGS_FILE

failures = []


def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(label)


# 备份主题设置，跑完还原（apply_theme 会写 ui_settings.json）
_theme_path = theme.theme.settings_path()
_theme_backup = None
if os.path.exists(_theme_path):
    with open(_theme_path, encoding="utf-8") as handle:
        _theme_backup = handle.read()
_ai_existed_before = os.path.exists(AI_SETTINGS_FILE)

app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app, "light")


# ------------------------------------------------------------------ 假传输
def make_sse(tokens, usage=None, done=True):
    """产出假 SSE 行的传输：逐 token 一帧，可选 usage 帧与 [DONE]。"""
    def transport(payload):
        for piece in tokens:
            frame = {"choices": [{"delta": {"content": piece}}]}
            yield "data: %s\n\n" % json.dumps(frame, ensure_ascii=False)
        if usage:
            yield "data: %s\n\n" % json.dumps({"choices": [], "usage": usage})
        if done:
            yield "data: [DONE]\n\n"
    return transport


def error_body_transport(payload):
    yield 'data: {"error":{"message":"rate limited"}}\n\n'


def endless_transport(payload):
    for _ in range(5000):
        yield 'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'


def http_error(code, body=""):
    return urllib.error.HTTPError("http://x", code, "err", {},
                                  io.BytesIO(body.encode("utf-8")))


def run_worker(client, messages, cancel=False, stream=True, **overrides):
    """在事件循环里跑一轮 worker，返回 (holder, tokens)。"""
    holder, tokens = {}, []
    loop = QEventLoop()
    worker = ai_worker.AiWorker(client, messages, stream=stream, **overrides)
    worker.token.connect(lambda piece: tokens.append(piece))
    worker.done.connect(lambda result: (holder.setdefault("done", result), loop.quit()))
    worker.error.connect(lambda message: (holder.setdefault("error", message), loop.quit()))
    worker.start()
    if cancel:
        worker.cancel()
    QTimer.singleShot(4000, loop.quit)
    loop.exec()
    worker.wait(2000)
    return holder, tokens


# ------------------------------------------------------------------ 1. 设置读写 + 档案 CRUD
settings.set_provider(base_url="https://api.test/v1", model="test-model",
                      api_key="sk-test", temperature=0.5, max_tokens=256)
check("provider base_url 落内存", settings.base_url() == "https://api.test/v1", settings.base_url())
check("provider model 落内存", settings.model() == "test-model")
check("provider api_key 落内存", settings.api_key() == "sk-test")
check("provider temperature", abs(settings.temperature() - 0.5) < 1e-9)
check("provider max_tokens", settings.max_tokens() == 256)

count0 = len(settings.agents())
agent = settings.add_agent("测试体", "你是测试角色", {"data": True, "framework": True},
                           ["模板一", "模板二"])
check("add_agent 数量 +1", len(settings.agents()) == count0 + 1)
check("add_agent 后成为激活档案", settings.active_agent_id() == agent["id"])
check("档案 templates 落库", settings.agent(agent["id"])["templates"] == ["模板一", "模板二"])
settings.update_agent(agent["id"], name="改名体")
check("update_agent 改名", settings.agent(agent["id"])["name"] == "改名体")
settings.remove_agent(agent["id"])
check("remove_agent 数量还原", len(settings.agents()) == count0)
check("remove 后激活档案仍有效", settings.active_agent() is not None)

# ------------------------------------------------------------------ 2. 客户端：校验 / 解析 / 错误分支
blank = ai_client.OpenAICompatClient("", "", "")
ok, note = blank.ready()
check("ready 缺 base_url 报人话", not ok and "base_url" in note, note)
nokey = ai_client.OpenAICompatClient("https://x/v1", "", "m")
ok, note = nokey.ready()
check("ready 缺 key 报人话", not ok and "key" in note, note)
nomodel = ai_client.OpenAICompatClient("https://x/v1", "k", "")
ok, note = nomodel.ready()
check("ready 缺 model 报人话", not ok and "模型" in note, note)

client = ai_client.OpenAICompatClient("https://x/v1", "k", "m")
check("endpoint 拼接", client.endpoint() == "https://x/v1/chat/completions", client.endpoint())
check("explain 401 无效 key", "无效" in client._explain_http(http_error(401, '{"error":{"message":"bad"}}')))
check("explain 429 限流", "限流" in client._explain_http(http_error(429)))
check("explain 404 不存在", "不存在" in client._explain_http(http_error(404)))
check("explain 500 服务端", "服务端" in client._explain_http(http_error(503)))

piece, usage, done = ai_client.OpenAICompatClient._parse_line("data: [DONE]")
check("parse [DONE]", piece is None and done is True)
piece, usage, done = ai_client.OpenAICompatClient._parse_line(
    'data: {"choices":[{"delta":{"content":"hi"}}]}')
check("parse delta.content", piece == "hi" and not done, piece)
piece, usage, done = ai_client.OpenAICompatClient._parse_line(
    'data: {"choices":[{"message":{"content":"yo"}}]}')
check("parse message.content（非流式）", piece == "yo", piece)
piece, usage, done = ai_client.OpenAICompatClient._parse_line("")
check("parse 空行", piece is None and not done)

check("rough_tokens CJK", ai_client.rough_tokens("你好") == 2)
check("rough_tokens ASCII", ai_client.rough_tokens("abcd") == 1)
mm = ai_client.user_multimodal("看图", ["data:image/jpeg;base64,AAA"])
check("user_multimodal 拼图片块",
      isinstance(mm["content"], list) and mm["content"][1]["type"] == "image_url")

# chat 直接跑（mock transport）
stream_client = ai_client.OpenAICompatClient("https://x/v1", "k", "m",
                                             transport=make_sse(["你", "好", "世", "界"],
                                                                usage={"prompt_tokens": 5,
                                                                       "completion_tokens": 4,
                                                                       "total_tokens": 9}))
result = stream_client.chat([ai_client.user_message("hi")])
check("chat 流式拼全文", result["text"] == "你好世界", result["text"])
check("chat 回传 usage", result["usage"].get("total_tokens") == 9)
check("chat 未取消", result["cancelled"] is False)

err_client = ai_client.OpenAICompatClient("https://x/v1", "k", "m",
                                          transport=error_body_transport)
try:
    err_client.chat([ai_client.user_message("hi")])
    check("200 内嵌 error 抛 AiError", False)
except ai_client.AiError as exc:
    check("200 内嵌 error 抛 AiError", "rate limited" in str(exc), str(exc))

# ------------------------------------------------------------------ 3. worker：流式 / 取消 / 错误
holder, tokens = run_worker(stream_client, [ai_client.user_message("hi")])
check("worker done 交货", holder.get("done", {}).get("text") == "你好世界", holder)
check("worker 逐块 token", "".join(tokens) == "你好世界", tokens)

cancel_client = ai_client.OpenAICompatClient("https://x/v1", "k", "m",
                                             transport=endless_transport)
holder, tokens = run_worker(cancel_client, [ai_client.user_message("hi")], cancel=True)
check("worker 取消标记", holder.get("done", {}).get("cancelled") is True, holder.get("done"))
check("worker 取消后只收部分 token", len(tokens) < 5000, len(tokens))

err_worker_client = ai_client.OpenAICompatClient("https://x/v1", "k", "m",
                                                 transport=error_body_transport)
holder, tokens = run_worker(err_worker_client, [ai_client.user_message("hi")])
check("worker error 信号", "rate limited" in holder.get("error", ""), holder)

# ------------------------------------------------------------------ 4. 上下文组装
class FakeSummary:
    def build_analysis_data(self):
        return (["Time", "S1Encoding_1", "S2Encoding_1"],
                [[0, "A", "B"], [1, "A", "A"], [2, "C", "B"]])


class FakePanel:
    @staticmethod
    def framework():
        return [("认知投入", ["A", "B", "C"])]


class FakePage:
    dim_panel = FakePanel()


snapshot = {"summary_tab": FakeSummary(),
            "encoder_pages": [("S1", FakePage()), ("S2", FakePage())],
            "group_id": "session",
            "result_text": "方法：IND01\n编码 频次\nA 3"}
full_agent = {"role": "你是顾问",
              "context": {"data": True, "framework": True, "methods": True, "results": True}}
messages = ai_context.build_messages(
    full_agent, "帮我规划", snapshot,
    history=[{"role": "user", "content": "上一轮"}, {"role": "assistant", "content": "上轮答复"}])
check("messages 首条为 system", messages[0]["role"] == "system")
system_text = messages[0]["content"]
check("上下文含数据概况", "数据概况" in system_text)
check("上下文含编码框架", "认知投入" in system_text)
check("上下文含方法清单", "M01" in system_text)
check("上下文含结果表", "IND01" in system_text)
check("messages 含两轮历史 + 本轮", len(messages) == 4, len(messages))
check("messages 末条为本轮用户消息",
      messages[-1]["role"] == "user" and messages[-1]["content"] == "帮我规划")
no_ctx = ai_context.build_messages({"role": "R", "context": {}}, "问题", snapshot)
check("无上下文开关时 system 只有角色",
      no_ctx[0]["content"] == "R" and len(no_ctx) == 2, no_ctx[0]["content"])

# ------------------------------------------------------------------ 5. AI 助手页发送闭环
import ai_assistant as aa

module = aa.AiAssistantModule()
module.set_transport(make_sse(["规", "划", "如", "下"],
                              usage={"prompt_tokens": 3, "completion_tokens": 4,
                                     "total_tokens": 7}))
module.base_url_edit.setText("https://api.test/v1")
module.model_edit.setText("test-model")
module.key_edit.setText("sk-test")
module.input.setPlainText("帮我规划分析方案")
module._on_send()
check("发送后 worker 起来了", module._worker is not None)
deadline = time.time() + 4
while module._worker is not None and time.time() < deadline:
    app.processEvents()
check("助手历史两轮", len(module._history) == 2, len(module._history))
check("助手答复文本", module._history[-1]["content"] == "规划如下", module._history[-1])
check("发送后输入框清空", module.input.toPlainText() == "")

# 缺 key 时给人话错误气泡，不起 worker
module2 = aa.AiAssistantModule()
module2.base_url_edit.setText("")
module2.model_edit.setText("")
module2.key_edit.setText("")
bubbles_before = module2.msg_lay.count()
module2.input.setPlainText("在吗")
module2._on_send()
check("缺配置时不起 worker", module2._worker is None)
check("缺配置时加错误气泡", module2.msg_lay.count() == bubbles_before + 1)

# ------------------------------------------------------------------ 5b. 框架设计工作台（离线闭环）
from framework_designer import FrameworkDesignerDialog

designer = FrameworkDesignerDialog()
designer.brief_widgets["research_question"].setPlainText("小组如何共同调节任务进度？")
designer.brief_widgets["task_context"].setPlainText("四人组科学探究")
for index in range(designer.reference_list.count()):
    item = designer.reference_list.item(index)
    if item.data(Qt.UserRole) in ("icap_2014", "ssrl_family"):
        item.setCheckState(Qt.Checked)
designer._offline_draft()
check("框架设计师可离线生成起始草稿", designer.draft_table.rowCount() > 0,
      designer.draft_table.rowCount())
designer.draft_table.selectAll()
designer._set_decision("accepted")
_project, _issues, _stats = designer._run_quality(silent=True)
check("已确认草稿通过阻断级质检", _stats["passed"], _issues)
check("状态/事件分流后只投影状态维度",
      [name for name, _codes in
       __import__("framework_schema").project_to_state_framework(_project)] == ["认知投入方式"])
# 下拉弹层：未施加圆角配方时，容器窗体的方形原生边框会在弹层底部露出一圈黑线框矩形。
check("框架设计工作台的常驻下拉都施加了弹层配方",
      all(getattr(combo, "_popup_binder", None) is not None
          for combo in (designer.project_combo, designer.version_combo, designer.target_combo)))
check("研究简报的选项型字段下拉同样施加",
      any(isinstance(widget, QComboBox) for widget in designer.brief_widgets.values())
      and all(getattr(widget, "_popup_binder", None) is not None
              for widget in designer.brief_widgets.values()
              if isinstance(widget, QComboBox)))
# 弹层半径只有一个来源：样式表画多少，容器窗体就裁多少。分成两个字面量时
# 「画圆 12、裁圆 8」必然在角上露出方形像素 —— 这一族修复的全部意义就是不让它俩分家。
_qss = theme.build_qss()
_popup_rule = re.search(r"QListView#speedPopup[^{]*\{(.*?)\}", _qss, re.S)
_menu_rule = re.search(r"^QMenu \{(.*?)\}", _qss, re.M)
check("下拉样式表的圆角等于 POPUP_RADIUS",
      bool(_popup_rule)
      and re.findall(r"border-radius: (\d+)px", _popup_rule.group(1)) == [str(theme.POPUP_RADIUS)],
      _popup_rule and re.findall(r"border-radius: (\d+)px", _popup_rule.group(1)))
check("QMenu 样式表的圆角等于 MENU_RADIUS",
      bool(_menu_rule)
      and re.findall(r"border-radius: (\d+)px", _menu_rule.group(1)) == [str(theme.MENU_RADIUS)],
      _menu_rule and re.findall(r"border-radius: (\d+)px", _menu_rule.group(1)))
check("已施加配方的下拉，实际裁剪半径就是 POPUP_RADIUS",
      all(getattr(combo._popup_binder, "_radius", None) == theme.POPUP_RADIUS
          for combo in (designer.project_combo, designer.version_combo, designer.target_combo)))
from PySide6.QtWidgets import QMenu
_menu_probe = QMenu()
theme.bind_menu_popup(_menu_probe)
check("菜单容器的裁剪半径就是 MENU_RADIUS（不与下拉共用默认值）",
      getattr(_menu_probe._rounded_mask_filter, "_radius", None) == theme.MENU_RADIUS,
      getattr(_menu_probe._rounded_mask_filter, "_radius", None))
_menu_probe.deleteLater()
# 半径不再由调用点各写一份：除 theme.py 外，整棵源码树不许出现显式半径实参。
_stray = []
_src_dir = os.path.dirname(os.path.abspath(__file__))
for _name in sorted(os.listdir(_src_dir)):
    if not _name.endswith(".py") or _name == "theme.py":
        continue
    with open(os.path.join(_src_dir, _name), encoding="utf-8") as _handle:
        _stray += [f"{_name}:{_no}" for _no, _line in enumerate(_handle, 1)
                   if re.search(r"(?:bind_combo_popup|rounded_popup_mask)\([^)]*,\s*\d", _line)]
check("业务模块不再手传弹层半径", not _stray, _stray[:5])
designer.close()

# ------------------------------------------------------------------ 6. main_window 集成
from main_window import MainWindow

window = MainWindow()
window.show()
app.processEvents()
check("外层标签页 == 5", window.tab_widget.count() == 5, window.tab_widget.count())
check("第 5 页标题为 AI 助手", window.tab_widget.tabText(4) == "AI 助手",
      window.tab_widget.tabText(4))
check("主窗口持有 ai_assistant", hasattr(window, "ai_assistant"))
snap = window._ai_context_snapshot()
check("快照为 dict 且含取数口",
      isinstance(snap, dict) and "summary_tab" in snap and "encoder_pages" in snap)
check("快照 result_text 为字符串", isinstance(snap.get("result_text"), str))
window.tab_widget.setCurrentIndex(4)
app.processEvents()
check("可切到 AI 助手页", window.tab_widget.currentIndex() == 4)
check("AI 页图标已挂", not window.tab_widget.tabIcon(4).isNull())

# ------------------------------------------------------------------ 7. AI 页布局：文字完整 + 有富余才钉底
def pump_events(times=30):
    for _ in range(times):
        app.processEvents()


def bottom_in_module(widget):
    """下边缘在模块坐标里的 y：跨滚动区比齐平只能换算到同一坐标。"""
    top = widget.mapTo(module, widget.rect().topLeft())
    return top.y() + widget.height()


agent_card = module.role_preview.parentWidget()
settings_card = module.status_label.parentWidget()
right_card = module.msg_scroll.parentWidget()
left_scroll = module.left_scroll
module.show()
pump_events()

for tag, page_h in (("富余态", 1200), ("矮窗口", 700)):
    module.resize(1200, page_h)
    pump_events()
    for index in range(module.agent_combo.count()):
        module.agent_combo.setCurrentIndex(index)
        pump_events()
        name = module.agent_combo.currentText()
        role_label, chips = module.role_preview, module.context_chips
        need_role = role_label.heightForWidth(role_label.width())
        need_chips = chips.heightForWidth(chips.width())
        check("%s %s：角色说明文字完整" % (tag, name),
              need_role > 0 and role_label.height() >= need_role,
              (role_label.height(), need_role))
        check("%s %s：附带上下文文字完整" % (tag, name),
              need_chips > 0 and chips.height() >= need_chips,
              (chips.height(), need_chips))
        check("%s %s：左栏不开横向滚动" % (tag, name),
              left_scroll.horizontalScrollBar().maximum() == 0)
        inner = left_scroll.widget()
        for button in (module.test_btn, module.save_settings_btn):
            top_left = button.mapTo(inner, button.rect().topLeft())
            check("%s %s：「%s」完整可滚到" % (tag, name, button.text()),
                  inner.rect().contains(QRect(top_left, button.rect().size())),
                  (top_left.y(), inner.height()))
        delta = bottom_in_module(settings_card) - bottom_in_module(right_card)
        if tag == "富余态":
            check("富余态 %s：「模型与密钥」底边与对话区齐平" % name, delta == 0, delta)
        else:
            check("矮窗口 %s：放不下就滚动，绝不允许既不滚又不齐平" % name,
                  left_scroll.verticalScrollBar().maximum() > 0 or delta == 0,
                  (left_scroll.verticalScrollBar().maximum(), delta))

module.agent_combo.setCurrentIndex(module.agent_combo.findData("framework_designer"))
pump_events()
card_lay = agent_card.layout()
last_item = card_lay.itemAt(card_lay.count() - 1).widget()
natural_with_btn = agent_card.sizeHint().height()
check("框架设计师：「进入框架设计工作台」在任务模板清单正下方（卡内最后一行）",
      last_item is module.framework_designer_btn and module.framework_designer_btn.isVisible(),
      last_item.text() if last_item is not None else None)
module.agent_combo.setCurrentIndex(module.agent_combo.findData("planner"))
pump_events()
check("非框架设计师：按钮隐藏且不占高",
      not module.framework_designer_btn.isVisible()
      and agent_card.sizeHint().height() < natural_with_btn,
      (natural_with_btn, agent_card.sizeHint().height()))
check("智能体下拉施加了弹层配方（不留方形原生边框）",
      getattr(module.agent_combo, "_popup_binder", None) is not None)
module.hide()

# ------------------------------------------------------------------ 8. 不落盘断言
check("offscreen 未写工程 ai_settings.json",
      os.path.exists(AI_SETTINGS_FILE) == _ai_existed_before,
      (_ai_existed_before, os.path.exists(AI_SETTINGS_FILE)))

# 还原主题设置
if _theme_backup is not None:
    with open(_theme_path, "w", encoding="utf-8") as handle:
        handle.write(_theme_backup)
elif os.path.exists(_theme_path):
    os.remove(_theme_path)

print()
if failures:
    print("SMOKE_FAILED:", failures)
    sys.exit(1)
print("SMOKE_OK  ai assistant 全链路通过（未连真网、未写 data/）")
