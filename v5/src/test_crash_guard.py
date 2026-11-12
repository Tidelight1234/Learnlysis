"""全局异常兜底（`main._install_crash_guard`）的行为契约。

这一层同时兜两件事，缺一个都会在最难受的时候暴露：

1. **不无声闪退**——PySide6 ≥ 5.5 对槽函数里逃逸的 Python 异常默认走 qFatal()→abort()，
   应用当场没；只要 `sys.excepthook` 被换成非默认值，PySide6 就改调它而不 abort。
2. **不泄露内部结构**——traceback 只落本机 `data/crash.log`，弹窗里一个字都不出现。
   这条是「低成本防护档」的立场，写成断言才不会被后来人顺手改成 `setDetailedText(tb)`。

`config.data_dir` 一律 monkeypatch 到 tmp_path：源码树里的 `data/` 是开发现场，
测试不许往那里写 crash.log，更不许因为写盘失败而把兜底本身测成红。
"""

import sys
import threading
import traceback

import pytest

import main


@pytest.fixture
def crash_dir(tmp_path, monkeypatch):
    """把 data 目录整个搬到 tmp_path 下。"""
    target = tmp_path / "data"
    monkeypatch.setattr(main.config, "data_dir", str(target))
    return target


def _boom():
    try:
        raise ValueError("fixture 崩了")
    except ValueError:
        return sys.exc_info()


# ------------------------------------------------------------------ 接线

def test_兜底装上后两个钩子都不是默认值(monkeypatch):
    """主线程 + Python 线程各有一个钩子；只装一个就会漏掉 QThread 之外的裸线程。"""
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    main._install_crash_guard()
    assert sys.excepthook is main._friendly_hook
    assert threading.excepthook is not threading.__excepthook__
    assert callable(threading.excepthook)


def test_线程里抛的异常真的会被兜到(monkeypatch, crash_dir):
    """不手造 `ExceptHookArgs`（它是 structseq，构造器不接受关键字），而是真开一个
    会抛异常的线程：解释器自己调 `threading.excepthook`，接线与字段顺序一起验。"""
    calls = []
    monkeypatch.setattr(main, "_friendly_hook", lambda *a: calls.append(a))
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    main._install_crash_guard()

    def worker():
        raise ValueError("fixture 崩了")

    t = threading.Thread(target=worker, name="fixture-thread")
    t.start()
    t.join()

    # 三个位置参数与 sys.excepthook 一致：少传一个，traceback 就拼不出来
    assert len(calls) == 1, calls
    got_type, got_value, got_tb = calls[0]
    assert got_type is ValueError and isinstance(got_value, ValueError)
    assert "".join(traceback.format_exception(got_type, got_value, got_tb))


# ------------------------------------------------------------------ 落盘

def test_traceback_只进_crashlog(crash_dir):
    exc_type, exc_value, exc_tb = _boom()
    main._friendly_hook(exc_type, exc_value, exc_tb)
    path = crash_dir / "crash.log"
    assert path.is_file(), "崩溃记录没落盘，用户拿什么反馈"
    text = path.read_text(encoding="utf-8")
    assert "ValueError" in text and "fixture 崩了" in text
    assert "test_crash_guard.py" in text, "traceback 细节应完整留在本地文件里"


def test_日志目录不存在时现场创建(crash_dir):
    assert not crash_dir.exists()
    main._append_crash_log("first\n")
    assert (crash_dir / "crash.log").read_text(encoding="utf-8") == "first\n"


def test_写日志失败绝不连累主流程(monkeypatch):
    """data 目录被只读盘/权限挡住时，兜底不能再抛一个异常把应用带崩。"""
    monkeypatch.setattr(main.config, "data_dir", "\x00 非法路径")
    main._append_crash_log("ignored\n")     # 不抛即通过


def test_超过上限只保留最近一半(crash_dir, monkeypatch):
    monkeypatch.setattr(main, "_CRASHLOG_MAX_BYTES", 200)
    main._append_crash_log("X" * 400 + "\n")
    main._append_crash_log("tail-marker\n")
    text = (crash_dir / "crash.log").read_text(encoding="utf-8")
    assert "更早的记录已截断" in text
    assert "tail-marker" in text
    assert len(text) < 400, len(text)


# ------------------------------------------------------------------ 弹窗口径

class _FakeBox:
    """记录 QMessageBox 的用法；`exec_()` 一旦被调就说明测试环境下弹了模态。

    两个枚举占位不是凑数：`_friendly_hook` 里写的是 `QMessageBox.Warning` /
    `QMessageBox.Ok`，这两个名字不在替身类上就会 AttributeError，而被兜底自己的
    `except Exception` 吞掉——症状是“弹窗没记账”而不是报错，测试会静悄悄地绿。
    """

    shown = []
    Warning = 0x80        # QMessageBox.Warning
    Ok = 0x400            # QMessageBox.Ok

    def __init__(self, *a, **k):
        self._texts = []
        type(self).shown.append(self)

    def setIcon(self, *a): pass
    def setWindowTitle(self, t): self._texts.append(("title", t))
    def setText(self, t): self._texts.append(("text", t))
    def setInformativeText(self, t): self._texts.append(("info", t))
    def setDetailedText(self, t): self._texts.append(("detail", t))
    def setStandardButtons(self, *a): pass
    def exec_(self): raise AssertionError("offscreen 下不得弹模态")

    def flat(self):
        return "\n".join(t for _, t in self._texts)


def test_offscreen_只留痕不弹模态(monkeypatch, crash_dir, qapp):
    exc_type, exc_value, exc_tb = _boom()
    _FakeBox.shown = []
    monkeypatch.setattr(main, "QMessageBox", _FakeBox)
    assert qapp.platformName() == "offscreen", "本项断言依赖 conftest 把平台钉成 offscreen"
    main._friendly_hook(exc_type, exc_value, exc_tb)
    assert _FakeBox.shown == [], "offscreen 弹模态会阻塞整个测试进程"
    assert (crash_dir / "crash.log").is_file()


def test_弹窗文案里不含_traceback(monkeypatch, crash_dir, qapp):
    """非 offscreen 分支：给的是人话，内部结构一个字都不给。

    offscreen 下 `_friendly_hook` 会提前 return，所以这里把 `platformName` 也换掉，
    让代码真走到弹窗那一段；`exec_()` 被替成记账，不阻塞。
    """
    class _Box(_FakeBox):
        def exec_(self):
            return 0

    exc_type, exc_value, exc_tb = _boom()
    _Box.shown = []
    monkeypatch.setattr(main, "QMessageBox", _Box)
    monkeypatch.setattr(qapp, "platformName", lambda: "windows")
    main._friendly_hook(exc_type, exc_value, exc_tb)

    assert len(_Box.shown) == 1, _Box.shown
    box = _Box.shown[0]
    flat = box.flat()
    assert "可以继续运行" in flat, flat
    assert "crash.log" in flat, "要告诉用户反馈时带上哪个文件"
    for key, text in box._texts:
        assert "Traceback" not in text and 'File "' not in text, (key, text)
    assert not any(k == "detail" for k, _ in box._texts), "setDetailedText 等于交出栈"
