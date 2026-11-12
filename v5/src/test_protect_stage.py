"""构建加固 staging（_protect_stage.py）的单元回归。

这里钉的是「加固动作本身」的正确性，不是分析口径：
1. 剥 docstring 是 AST 定点手术——剥完必须还能 import，且**不能**碰语义
   （唯一语句是 docstring 的作用域要补 pass，前后有代码的行必须放过）；
2. 水印只写 stage 那份副本，源树的 build_stamp.py 必须一个字节不动；
3. 编译产物判定必须认带 CPython tag 的文件名（`.cp311-win_amd64.pyd`），
   判错就会既不删 .py（等于白编）又把成功报成失败；
4. 生成的 exe 版本资源得能被 PyInstaller 自己的解析器读回去；
5. 复制副本不许自嵌套（输出目录就在源码树里，这是本项目的实际布局）。

本模块测的是**开发工具本身**，而 `_protect_stage.py` / `_set_source_header.py` 按设计
不进 stage 副本（`_*.py` 一律排除），所以拿不到它们时整块 skip——包括那条
「源码树里的水印恒为 dev」：在已写过水印的副本里它必然反。编译产物自己的行为
由 `test_build_watermark.py` / `test_integrity.py` 与全部公式回归在 stage 里验。
"""

import ast
import types

import pytest

ps = pytest.importorskip("_protect_stage",
                         reason="构建脚本不进 stage 副本（_*.py 被排除）")


# ---------------------------------------------------------------- 目标清单

def test_编译目标覆盖算法族与加固模块():
    assert set(ps.CORE_TARGETS) == {"analysis_stats_sequence", "analysis_stats_group",
                                    "analysis_stats_network", "analysis_features",
                                    "interaction_model"}
    assert set(ps.SECURITY_TARGETS) == {"build_stamp", "secure_store", "integrity"}
    # 水印携带者单独一档：它必须最后编（HARDENING 要写编译结果）
    assert ps.WATERMARK_MODULE in ps.SECURITY_TARGETS
    assert ps.CythonTargets == ps.CORE_TARGETS + ps.SECURITY_TARGETS


def test_开发期文件不进_stage():
    assert ps._keep_file("main.py") and ps._keep_file("analysis_core.py")
    assert ps._keep_file("logo.ico") and ps._keep_file("MiSans-Regular.ttf")
    for name in ("_smoke_ai.py", "_protect_stage.py", "test_analysis_core.py",
                 "conftest.py", "Learnlysis.spec", "stale.pyd", "x.pyc"):
        assert not ps._keep_file(name), name
    for name in ("data", "dist", "build", "__pycache__", "_testdata", "_stage"):
        assert ps._excluded_dir(name), name
    assert not ps._excluded_dir("assets") and not ps._excluded_dir("r_refs")


def test_cython_directives_里不含_docstrings():
    """Cython 3.x 没有 `docstrings` 这个 compiler directive，塞进去当场 ValueError。

    剥文档串走 ``Cython.Compiler.Options.docstrings`` 这个模块级全局（setup 模板里设），
    这条断言钉的是「别把它当指令名写回字典」——写回去的症状是整趟编译直接失败。
    """
    assert "docstrings" not in ps.CYTHON_DIRECTIVES
    assert ps.CYTHON_DIRECTIVES["binding"] is True
    assert 'docstrings = False' in ps._SETUP_TEMPLATE


# ---------------------------------------------------------------- 副本复制

def test_copy_tree_不会把输出目录卷进副本(tmp_path):
    src = tmp_path / "v5"
    (src / "assets").mkdir(parents=True)
    (src / "main.py").write_text("print(1)\n", encoding="utf-8")
    (src / "assets" / "a.ttf").write_text("x", encoding="utf-8")
    (src / "_smoke_tool.py").write_text("print(2)\n", encoding="utf-8")
    (src / "data").mkdir()
    (src / "data" / "archive.json").write_text("{}", encoding="utf-8")
    dst = src / "_stage"                      # 输出目录就在源码树里（本项目实况）
    copied = ps.copy_tree(src, dst)
    assert copied == 2                        # main.py + a.ttf
    assert (dst / "main.py").is_file() and (dst / "assets" / "a.ttf").is_file()
    assert not (dst / "_smoke_tool.py").exists()
    assert not (dst / "data").exists()
    assert not (dst / "_stage").exists(), "输出目录被当成源目录复制了（自嵌套）"


# ---------------------------------------------------------------- 剥 docstring

_SOURCE = '''"""模块文档串。"""


class Greeter:
    """类文档串。"""

    def hello(self):
        """方法文档串。"""
        return "hi"

    def only_doc(self):
        """这个作用域只有一句文档串。"""

    def tagged(self):
        """带 f-string 与引号嵌套的文档串："x" / \'y\'。"""
        value = f"{'ok'}"
        return value


assignment = """docstring"""
'''


def test_strip_one_剥干净但保留语义(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(_SOURCE, encoding="utf-8")
    stripped = ps.strip_one(path)
    text = path.read_text(encoding="utf-8")

    assert stripped >= 5
    tree = ast.parse(text)                       # 语法必须还立得住
    for phrase in ("模块文档串", "类文档串", "方法文档串", "只有一句文档串", "引号嵌套"):
        assert phrase not in text, phrase        # 真 docstring 逐个消失
    # 唯一的语句被换成 pass，不是整块删掉
    assert "pass" in text
    # `assignment = """docstring"""` 前后都有代码，必须放过（放了就不是 docstring）
    assert "assignment =" in text and "docstring" in text

    module = _blank_module("ly_stripped")
    exec(compile(text, str(path), "exec"), module.__dict__)
    assert module.Greeter.__doc__ is None
    assert module.Greeter.hello.__doc__ is None
    assert module.Greeter().hello() == "hi"      # 行为一个字节没变
    assert module.Greeter().only_doc() is None
    assert module.Greeter().tagged() == "ok"


def test_strip_docstrings_统计与残留归零(tmp_path):
    (tmp_path / "a.py").write_text('"""doc a"""\n\nX = 1\n', encoding="utf-8")
    (tmp_path / "b.py").write_text("def f():\n    '''doc b'''\n    return 2\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def g(:\n", encoding="utf-8")   # 读不懂就原样留着
    info = ps.strip_docstrings(tmp_path)
    assert info["py_files"] == 3
    assert info["files_stripped"] == 2
    assert info["docstrings_left"] == 0, info
    assert info["ast_unreadable"] == 1, info
    assert (tmp_path / "broken.py").read_text(encoding="utf-8") == "def g(:\n"


def test_strip_one_剪坏语法就整文件回退(tmp_path):
    """前后可见性守卫之外的最后一道：compile() 不过就当没剥过，绝不交出一份坏副本。"""
    body = "if True:\n    '''doc'''\nelse:\n    pass\n"
    path = tmp_path / "weird.py"
    path.write_text(body, encoding="utf-8")
    ps.strip_one(path)
    assert path.read_text(encoding="utf-8") == body or _parses(path)


def _parses(path):
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        return True
    except SyntaxError:
        return False


def test_剥完文档串后版权头还在(tmp_path):
    """归属声明走注释，就是为了让加固这一刀砍不到它（docstring 会被剥光）。"""
    header = pytest.importorskip("_set_source_header")
    path = tmp_path / "mod.py"
    path.write_text("\n".join(header.HEADER_LINES) + "\n" + _SOURCE, encoding="utf-8")
    assert ps.strip_one(path) >= 5
    text = path.read_text(encoding="utf-8")
    assert header.has_header(text), "版权头被剥掉了"
    assert "模块文档串" not in text, "docstring 没剥干净"


def _blank_module(name):
    """拿一个干净的模块对象当 exec 容器（不用 importlib，不污染 sys.modules）。"""
    return types.ModuleType(name)


# ---------------------------------------------------------------- 写水印

_STAMP = '''"""build_stamp 替身。"""
BUILD_ID = ""
BUILD_TIME = ""
GIT_COMMIT = ""
HARDENING = ""
'''


def test_stamp_只改副本不碰源树(tmp_path):
    (tmp_path / "build_stamp.py").write_text(_STAMP, encoding="utf-8")
    ps.stamp(tmp_path, {"BUILD_ID": "5.1.0+20260915-abcd", "HARDENING":
                        "docstrings=stripped;compiled=integrity"})
    text = (tmp_path / "build_stamp.py").read_text(encoding="utf-8")
    assert 'BUILD_ID = "5.1.0+20260915-abcd"' in text
    assert 'HARDENING = "docstrings=stripped;compiled=integrity"' in text
    assert 'BUILD_TIME = ""' in text           # 没点名的常量原样留着
    with pytest.raises(RuntimeError):
        ps.stamp(tmp_path, {"NO_SUCH_CONSTANT": "x"})


def test_源树里的水印仍是dev():
    """跑完上面这些文件级手术，工作区那份 build_stamp.py 不能被人顺手写了。"""
    import build_stamp
    assert not build_stamp.is_release(), build_stamp.audit_dict()


# ---------------------------------------------------------------- 编译产物判定

def test_extension_files_认带_tag_的产物名(tmp_path):
    (tmp_path / "integrity.cp311-win_amd64.pyd").write_text("x", encoding="utf-8")
    hits = ps._extension_files(tmp_path, "integrity")
    assert [p.name for p in hits] == ["integrity.cp311-win_amd64.pyd"]
    assert ps._extension_files(tmp_path, "analysis_features") == []


# ---------------------------------------------------------------- exe 版本资源

def test_version_file_能被_pyinstaller_读回(tmp_path):
    ver = pytest.importorskip("PyInstaller.utils.win32.versioninfo")
    name = ps.write_version_file(tmp_path, "5.1.0", "5.1.0+20260915-abcd")
    path = tmp_path / name
    info = ver.load_version_info_from_text_file(str(path))
    # 四段版本号被打包成 MS/LS 两个 DWORD：(5,1,0,0) → 0x50001 / 0x0
    assert info.ffi.fileVersionMS == (5 << 16) | 1
    assert info.ffi.fileVersionLS == 0
    strings = {}
    for block in info.kids:                      # StringFileInfo / VarFileInfo
        for table in getattr(block, "kids", []) or []:
            for struct in getattr(table, "kids", []) or []:
                if getattr(struct, "name", None) and getattr(struct, "val", None):
                    strings[struct.name] = struct.val
    assert strings["FileVersion"] == "5.1.0+20260915-abcd"
    assert strings["ProductVersion"] == "5.1.0+20260915-abcd"
    assert "MIT License" in strings["LegalCopyright"]


def test_版本号数值化():
    assert ps._numeric("5.1.0") == (5, 1, 0, 0)
    assert ps._numeric("5.10.2+build7") == (5, 10, 2, 0)
    assert ps._numeric("") == (0, 0, 0, 0)


def test_app_version_与config一致():
    """staging 用文本抓版本号（不 import config，那会拉起 Qt）——抓错的后果是水印与实物不符。"""
    import config
    assert ps.app_version() == config.APP_VERSION


def test_config_literals_能算出拼接式常量():
    """``APP_COPYRIGHT = "..." % APP_NAME`` 不是字面量，只会正则抓行的写法拿不到它。

    拿不到不报错、只会静静地回落到脚本里的默认值——那就是「改了 config 的年份，
    exe 属性页还是旧年份」的实际发生路径，所以单独钉一条。
    """
    import config
    got = ps.config_literals(ps.CONFIG_KEYS)
    for key in ps.CONFIG_KEYS:
        assert got[key] == getattr(config, key), key
    assert ps.config_literals(("NO_SUCH_KEY",))["NO_SUCH_KEY"] == ""
