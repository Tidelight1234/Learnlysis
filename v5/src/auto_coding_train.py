# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""Phase 3 本地训练脚手架（可选重依赖，优雅降级）：在用户已标注的编码上训练轻量分类头。

**现实预期**：小样本（真实副本约 23 组 / 55 学生 / 每组数百步）下精度有限，本模块定位为
「离线可控的辅助」，**始终人在环**——它的预测与 Phase 2 一样只是「建议码、待确认」，写回
口径完全复用 ``auto_coding.apply_suggestions``（只填空、不覆盖人工码、可撤销、注释标注）。

特征分两层：
- **序列特征（始终可得，纯 numpy）**：上一时刻各维度的编码 one-hot + 时间，是不依赖任何重
  后端的诚实基线（本质是比 Phase 1 一阶转移更强的判别式模型）。
- **视听特征（可选重后端）**：``mediapipe`` 姿态 / ``faster-whisper`` ASR / ``opencv`` 抽帧的
  画面统计。缺席时明确降级并给安装命令，绝不假装具备视听识别能力。

训练复用 ``analysis_ml`` 的 sklearn RF/HistGB + ``class_balanced_weights``（n/(k·n_c) 同口径）
+ ``grouped_split``（按组键分组 K 折，防跨组泄漏）；跑在 ``CodingTrainWorker``（QThread）里，
每折检查 ``ctx.cancelled``，取消走单一通路。
"""

from __future__ import annotations

import logging
import pickle
import warnings

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QSpinBox, QProgressBar, QTextEdit)

import analysis_ml
import theme
import user_state

try:                                        # sklearn 是核心依赖，兜底只为「导入失败」的实话
    from sklearn.metrics import balanced_accuracy_score, f1_score
except Exception:                           # noqa: BLE001  pragma: no cover
    balanced_accuracy_score = f1_score = None

#: 训练所需的最少标注样本数（低于此不装模作样，直接阻断并说明）
MIN_SAMPLES = 4


# ------------------------------------------------------------------ 可选后端

def optional_backends():
    """视听重后端 + 抽帧 + sklearn 的在场判定（照 analysis_ml.optional_backends 范式）。"""
    import importlib.util
    import shutil
    return {
        "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
        "mediapipe": importlib.util.find_spec("mediapipe") is not None,
        "opencv": importlib.util.find_spec("cv2") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
    }


def av_backends_present():
    """有没有任一「视听特征」后端（mediapipe / faster-whisper）；没有就只用序列特征。"""
    backends = optional_backends()
    return any(backends.get(key) for key in ("mediapipe", "faster_whisper")), backends


def install_hint():
    """视听后端缺席时的人话 + 安装命令（Phase 3 最重、始终可选）。"""
    return ("本地视听特征需要可选重后端（未随核心安装，Phase 3 始终可选）：\n"
            "  · pip install mediapipe        （人体姿态 → 行为特征）\n"
            "  · pip install faster-whisper   （语音转写 → 话语特征）\n"
            "未装时本模块只用「序列特征」训练轻量分类头，仍可离线给出建议码（精度有限，人在环）。")


def train_available(n_samples, n_classes):
    """``(是否可训练, 原因)``：sklearn 在场 + 样本够 + 标签不止一类。"""
    if not optional_backends().get("sklearn"):
        return False, "未检测到 scikit-learn（核心依赖缺失），无法训练"
    if n_samples < MIN_SAMPLES:
        return False, "标注样本太少（%d < %d），先在编码页多录几行有标注的编码" % (n_samples, MIN_SAMPLES)
    if n_classes < 2:
        return False, "该维度的标注只有单一类别，无法训练判别式分类头"
    return True, ""


# ------------------------------------------------------------------ 特征工程

def sequence_feature_names(framework):
    """序列特征名（与 ``sequence_feature_vector`` 一一对应，供重要度/调试）。"""
    names = []
    for name, codes in framework or []:
        for code in codes or []:
            names.append("lag1_%s=%s" % (name, code))
        names.append("lag1_%s=<空>" % name)
    return names


def sequence_feature_vector(rows, framework, index):
    """第 ``index`` 行的序列特征：上一时刻各维度编码 one-hot。只用 i-1 之前信息，
    不把当前行的目标码喂进特征（避免泄漏）。列顺序与 ``sequence_feature_names`` 一一对应：
    每维度按词表逐码 one-hot，末尾补一列 ``<空>``（上一行该维度无码/越界时置 1）。

    刻意不拼「绝对时间」特征：它对预测类别码是弱且易泄漏的信号，还会让「预测下一行」
    （index == len(rows)）的特征长度与训练时不一致。维度长度恒定 = 无错配隐患。"""
    feats = []
    prev = rows[index - 1] if index - 1 >= 0 else None
    for dim, (_name, codes) in enumerate(framework or []):
        vocab = [str(code) for code in (codes or [])]
        prev_code = ""
        if prev is not None and dim < len(prev):
            prev_code = str(prev[dim] or "").strip()
        for code in vocab:
            feats.append(1.0 if prev_code == code else 0.0)
        feats.append(1.0 if not prev_code else 0.0)   # <空>
    return feats


def visual_features(image, bins=6):
    """从 QImage 抽极简画面统计（亮度均值/标准差 + 灰度直方图），纯 numpy。

    需要 video_probe 抽到的帧；无帧时返回空列表（调用方据此判断是否并入特征矩阵）。
    这只是视听特征的「脚手架级」实现——真正的姿态/动作识别要 mediapipe 等重后端。
    """
    if image is None or image.isNull():
        return []
    try:
        gray = image.convertToFormat(QImage.Format_Grayscale8)
        w, h = gray.width(), gray.height()
        plane = np.frombuffer(gray.constBits(), dtype=np.uint8,
                              count=gray.sizeInBytes()).reshape(h, gray.bytesPerLine())
        arr = plane[:, :w].astype(float)   # Grayscale8：每像素 1 字节，剔掉行末对齐 padding
    except Exception as exc:                      # noqa: BLE001 取不到像素就退化为空特征
        logging.warning("visual_features 读取像素失败：%s", exc)
        return []
    hist, _edges = np.histogram(arr, bins=bins, range=(0, 255))
    hist = hist.astype(float)
    hist = hist / hist.sum() if hist.sum() else hist
    return [float(arr.mean()), float(arr.std())] + hist.tolist()


def build_dataset(rows, framework, dim_index, groups=None, frames=None):
    """从编码页已录行里，为某维度攒训练集。

    只收「该维度落在词表内的非空标注」行；``groups`` 给分组切分用（防跨组泄漏），缺省时
    每行自成一组（近似留一）。``frames`` = ``{row_index: QImage}`` 可选：**仅当每个标注行都
    有帧、且视觉特征长度一致时**才并入 ``visual_features``（视听特征），否则诚实退回纯序列
    特征（不拼半拉子、不造参差矩阵）。返回 ``{"X", "y", "groups", "feature_names", "n", "visual"}``。
    """
    framework = list(framework or [])
    if not (0 <= dim_index < len(framework)):
        return {"X": np.zeros((0, 0)), "y": [], "groups": [], "feature_names": [], "n": 0,
                "visual": False, "reason": "维度号越界"}
    vocab = [str(code) for code in framework[dim_index][1]]
    labeled = [index for index, row in enumerate(rows or [])
               if dim_index < len(row) and str(row[dim_index]).strip()
               and str(row[dim_index]).strip() in vocab]
    # 视听特征：全有或全无——每个标注行都有帧且向量长度一致才并入，否则只用序列特征
    use_visual, visual_vecs, n_visual = False, {}, 0
    if frames and labeled:
        candidate, consistent = {}, True
        for index in labeled:
            image = frames.get(index)
            vec = visual_features(image) if image is not None else []
            if not vec:
                consistent = False
                break
            if n_visual == 0:
                n_visual = len(vec)
            elif len(vec) != n_visual:
                consistent = False
                break
            candidate[index] = vec
        if consistent and candidate:
            use_visual, visual_vecs = True, candidate
    seq_names = sequence_feature_names(framework)
    names = seq_names + (["vis%d" % k for k in range(n_visual)] if use_visual else [])
    X, y, g = [], [], []
    for index in labeled:
        feats = sequence_feature_vector(rows, framework, index)
        if use_visual:
            feats = feats + visual_vecs[index]
        X.append(feats)
        y.append(str(rows[index][dim_index]).strip())
        g.append(str(groups[index]) if groups is not None and index < len(groups) else str(index))
    matrix = np.asarray(X, dtype=float) if X else np.zeros((0, len(names)))
    return {"X": matrix, "y": y, "groups": g, "feature_names": names, "n": len(y),
            "visual": use_visual}


def time_block_groups(n_rows, target_blocks=5):
    """把 ``n_rows`` 行按时间顺序切成 ``target_blocks`` 段连续块，返回每行的块键。

    顺序编码数据有强自相关（相邻时刻高度相关），逐行独立切分会泄漏。用「连续时间块」
    当组键交给 ``analysis_ml.grouped_split``，整块全进全出 = block CV，是时序数据的诚实口径。
    """
    n_rows = int(n_rows)
    if n_rows <= 0:
        return []
    target_blocks = int(max(2, min(int(target_blocks), n_rows)))
    size = max(1, n_rows // target_blocks)
    return ["blk%d" % min(index // size, target_blocks - 1) for index in range(n_rows)]


# ------------------------------------------------------------------ 训练

class TrainContext:
    """训练上下文：取消标志（单一通路）+ 进度回调（镜像 analysis_methods.AnalysisContext）。"""

    def __init__(self, cancelled=None, progress=None):
        self._cancelled = cancelled or (lambda: False)
        self._progress = progress

    def cancelled(self):
        return bool(self._cancelled())

    def report(self, step, total, message=""):
        if self._progress is not None:
            self._progress(int(step), int(total), str(message))


def train_coding_model(dataset, model_name="rf", folds=5, seed=None, ctx=None):
    """在数据集上训练轻量分类头：分组 K 折给诚实指标，再全量拟合一个部署模型。

    返回 ``{"ok", "reason"?, "engine", "model", "classes", "feature_names", "per_fold",
    "summary", "n_samples", "note"}``。样本不足 / 单类 / 取消都 ``ok=False`` 并写明原因。
    """
    X = dataset.get("X")
    y = list(dataset.get("y") or [])
    groups = list(dataset.get("groups") or [])
    seed = analysis_ml.DEFAULT_SEED if seed is None else int(seed)
    n = int(getattr(X, "shape", (0,))[0])
    ok, reason = train_available(n, len(set(y)))
    if not ok:
        return {"ok": False, "reason": reason}
    if balanced_accuracy_score is None:
        return {"ok": False, "reason": "scikit-learn 指标不可用，无法训练"}

    probe, engine = analysis_ml._make_model(model_name, seed)
    if probe is None:
        return {"ok": False, "reason": engine}

    per_fold = []
    note = ""
    split = analysis_ml.grouped_split(groups, folds=folds, seed=seed)
    if split:
        for position, (train_idx, test_idx) in enumerate(split):
            if ctx is not None and ctx.cancelled():
                return {"ok": False, "cancelled": True, "reason": "已取消"}
            estimator, _engine = analysis_ml._make_model(model_name, seed)
            y_train = [y[i] for i in train_idx]
            weights = analysis_ml.class_balanced_weights(y_train)
            fit_kwargs = {"sample_weight": weights} if weights is not None and len(weights) else {}
            estimator.fit(X[train_idx], y_train, **fit_kwargs)
            pred = [str(value) for value in estimator.predict(X[test_idx])]
            truth = [y[i] for i in test_idx]
            # 小样本下某折可能只剩单一类别：sklearn 会 warn，但指标仍正确（抑制噪音）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_fold.append({
                    "fold": position + 1,
                    "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
                    "f1_macro": float(f1_score(truth, pred, average="macro", zero_division=0)),
                    "n_test": len(truth),
                })
            if ctx is not None:
                ctx.report(position + 1, len(split), "第 %d/%d 折完成" % (position + 1, len(split)))
    else:
        note = "组键不足 2 个，无法做分组交叉验证：只给全量拟合的模型，不报指标以免泄漏虚高。"

    if ctx is not None and ctx.cancelled():
        return {"ok": False, "cancelled": True, "reason": "已取消"}
    final, engine = analysis_ml._make_model(model_name, seed)
    weights = analysis_ml.class_balanced_weights(y)
    fit_kwargs = {"sample_weight": weights} if weights is not None and len(weights) else {}
    final.fit(X, y, **fit_kwargs)
    classes = [str(item) for item in getattr(final, "classes_", sorted(set(y)))]
    summary = {}
    if per_fold:
        summary = {
            "balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in per_fold])),
            "f1_macro": float(np.mean([f["f1_macro"] for f in per_fold])),
            "folds": len(per_fold),
        }
    return {"ok": True, "engine": engine, "model": final, "classes": classes,
            "feature_names": dataset.get("feature_names", []), "per_fold": per_fold,
            "summary": summary, "n_samples": n, "note": note}


def predict_dim_code(bundle, feature_vector):
    """用训练好的模型预测某维度的建议码：返回 ``(code, proba or None)``。"""
    model = (bundle or {}).get("model")
    if model is None:
        return "", None
    x = np.asarray([list(feature_vector)], dtype=float)
    code = str(model.predict(x)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(np.max(model.predict_proba(x)[0]))
        except Exception:                       # noqa: BLE001 概率拿不到不影响给码
            proba = None
    return code, proba


# ------------------------------------------------------------------ 持久化

def save_bundle(bundle, path):
    """把部署模型 + 元信息落盘（pickle；joblib 在场则优先）。返回是否成功。"""
    payload = {key: bundle.get(key) for key in
               ("engine", "classes", "feature_names", "model", "n_samples", "summary")}
    try:
        import joblib
        joblib.dump(payload, path)
        return True
    except Exception:                           # noqa: BLE001 没有 joblib 就退回 pickle
        pass
    try:
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
        return True
    except Exception as exc:                    # noqa: BLE001
        logging.warning("保存本地编码模型失败：%s", exc)
        return False


def load_bundle(path):
    """读回 ``save_bundle`` 落盘的模型；失败返回 None。"""
    try:
        import joblib
        return joblib.load(path)
    except Exception:                           # noqa: BLE001
        pass
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:                    # noqa: BLE001
        logging.warning("读取本地编码模型失败：%s", exc)
        return None


# ------------------------------------------------------------------ 训练线程

class CodingTrainWorker(QThread):
    """把一次本地训练搬到子线程：进度 / 完成 / 错误信号回主线程，取消走单一通路。"""

    progress = Signal(int, int, str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, dataset, model_name="rf", folds=5, seed=None, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.model_name = model_name
        self.folds = folds
        self.seed = seed
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        ctx = TrainContext(cancelled=lambda: self._cancelled,
                           progress=lambda s, t, m: self.progress.emit(s, t, m))
        try:
            result = train_coding_model(self.dataset, self.model_name, self.folds,
                                        self.seed, ctx)
        except Exception as exc:                # noqa: BLE001 线程里抛不出去，只能回传
            self.error.emit("训练失败：%s" % exc)
            return
        self.done.emit(result)


#: 写回时的注释标记：区别于 Phase 2 的「AI 建议」（那是多模态 API），这是本地模型产物
SUGGEST_MARKER_LOCAL = "本地模型建议、待确认"


class LocalTrainDialog(QDialog):
    """Phase 3「本地训练辅助编码」：在你已标注的编码上离线训练轻量分类头，预测仍作「建议码」写回。

    - 视听重后端（mediapipe/faster-whisper）缺席时如实标注 + 给安装命令，但仍可用「序列特征」训练；
    - 训练跑在 ``CodingTrainWorker``（QThread）：进度条 + 可取消，绝不卡 UI 线程；
    - 连续时间块分组 K 折给诚实指标（block CV，防时序泄漏）；
    - 预测当前时刻 → 经 ``auto_coding.apply_suggestions`` 写回，只填空、不覆盖人工码、可撤销、标注待确认。
    """

    def __init__(self, encoder, parent=None):
        super().__init__(parent or encoder)
        self.encoder = encoder
        self._worker = None
        self._bundle = None
        self.setWindowTitle("本地训练辅助编码（Phase 3 · 离线，人在环）")
        self.resize(680, 600)
        self._build_ui()
        self._refresh_backends()
        self._refresh_dims()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        intro = QLabel("用你已录的编码在本地训练一个轻量分类头（离线、不联网、不上传）；"
                       "它给的仍是「建议码、待确认」，逐条写回且绝不覆盖你已录的人工码。")
        intro.setProperty("hint", True)
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.backend_notice = QLabel()
        self.backend_notice.setProperty("hint", True)
        self.backend_notice.setWordWrap(True)
        root.addWidget(self.backend_notice)

        dim_row = QHBoxLayout()
        dim_row.addWidget(self._section_label("训练维度"))
        self.dim_combo = QComboBox()
        self.dim_combo.setMinimumWidth(220)
        self.dim_combo.currentIndexChanged.connect(self._refresh_sample_label)
        # 与主窗口各页下拉同一套弹层配方，避免圆角外露出方形原生边框
        theme.bind_combo_popup(self.dim_combo)
        dim_row.addWidget(self.dim_combo, 1)
        root.addLayout(dim_row)

        opt_row = QHBoxLayout()
        opt_row.addWidget(self._section_label("模型"))
        self.model_combo = QComboBox()
        for key, text in analysis_ml.available_models():
            self.model_combo.addItem(text, key)
        theme.bind_combo_popup(self.model_combo)
        opt_row.addWidget(self.model_combo)
        opt_row.addSpacing(12)
        opt_row.addWidget(self._section_label("分组折数"))
        self.folds_spin = QSpinBox()
        self.folds_spin.setRange(2, 10)
        self.folds_spin.setValue(5)
        opt_row.addWidget(self.folds_spin)
        opt_row.addStretch(1)
        root.addLayout(opt_row)

        self.sample_label = QLabel()
        self.sample_label.setProperty("hint", True)
        self.sample_label.setWordWrap(True)
        root.addWidget(self.sample_label)

        train_row = QHBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setProperty("btn", "primary")
        theme.bind_icon(self.train_btn, "robot", 16)
        self.train_btn.clicked.connect(self._on_train)
        self.stop_btn = QPushButton("取消")
        self.stop_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.stop_btn, "stop", 16)
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        train_row.addWidget(self.train_btn)
        train_row.addWidget(self.stop_btn)
        train_row.addStretch(1)
        root.addLayout(train_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        root.addWidget(self._section_label("训练指标（分组 K 折 = 诚实的跨块泛化，非训练集自评）"))
        self.metrics = QTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setFixedHeight(150)
        self.metrics.setPlaceholderText("训练后在此显示每折与汇总指标……")
        root.addWidget(self.metrics, 1)

        act_row = QHBoxLayout()
        self.predict_btn = QPushButton("预测当前时刻并写回建议")
        self.predict_btn.setProperty("btn", "primary")
        theme.bind_icon(self.predict_btn, "sparkle", 16)
        self.predict_btn.clicked.connect(self._on_predict)
        self.predict_btn.setEnabled(False)
        self.save_btn = QPushButton("保存模型…")
        self.save_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.save_btn, "save", 16)
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        act_row.addWidget(self.predict_btn)
        act_row.addWidget(self.save_btn)
        act_row.addStretch(1)
        root.addLayout(act_row)

        self.status_label = QLabel("就绪")
        self.status_label.setProperty("hint", True)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("btn", "secondary")
        self.close_btn.clicked.connect(self.accept)
        close_row.addWidget(self.close_btn)
        root.addLayout(close_row)

    def _section_label(self, text):
        label = QLabel(text)
        label.setProperty("section", True)
        return label

    # ---------- 数据面 ----------
    def _framework(self):
        try:
            return self.encoder.dim_panel.framework()
        except Exception:                        # noqa: BLE001
            return []

    def _rows(self):
        try:
            return [list(row) for row in (self.encoder.selected_data or [])]
        except Exception:                        # noqa: BLE001
            return []

    def _current_seconds(self):
        try:
            value = self.encoder.time_ctrl.parse_timepoint()
            return float(value) if value is not None else None
        except Exception:                        # noqa: BLE001
            return None

    def _refresh_backends(self):
        present, backends = av_backends_present()
        if present:
            have = "、".join(name for name in ("mediapipe", "faster_whisper") if backends.get(name))
            self.backend_notice.setText("视听后端：%s（就绪，可并入视听特征）" % have)
        else:
            self.backend_notice.setText("视听后端：未安装 —— " + install_hint())

    def _refresh_dims(self):
        self.dim_combo.blockSignals(True)
        self.dim_combo.clear()
        for index, (name, codes) in enumerate(self._framework()):
            self.dim_combo.addItem("%s（%d 个码）" % (name, len(codes or [])), index)
        self.dim_combo.blockSignals(False)
        self._refresh_sample_label()

    def _dim_index(self):
        value = self.dim_combo.currentData()
        return int(value) if value is not None else -1

    def _collect(self):
        rows = self._rows()
        groups = time_block_groups(len(rows), target_blocks=self.folds_spin.value())
        return build_dataset(rows, self._framework(), self._dim_index(), groups=groups)

    def _refresh_sample_label(self):
        framework = self._framework()
        dim_index = self._dim_index()
        if not (0 <= dim_index < len(framework)):
            self.sample_label.setText("编码页还没有维度框架，先导入/添加维度。")
            self.train_btn.setEnabled(False)
            return
        dataset = self._collect()
        n = dataset["n"]
        n_classes = len(set(dataset["y"]))
        ok, reason = train_available(n, n_classes)
        self.sample_label.setText("可用标注：%d 行，%d 个类别。%s"
                                  % (n, n_classes, "" if ok else reason))
        self.train_btn.setEnabled(ok)

    # ---------- 训练 ----------
    def _on_train(self):
        dataset = self._collect()
        ok, reason = train_available(dataset["n"], len(set(dataset["y"])))
        if not ok:
            self.status_label.setText(reason)
            return
        self._set_busy(True)
        self.progress.setValue(0)
        self.metrics.clear()
        self.status_label.setText("训练中……（%d 行，%s）" % (dataset["n"], self.model_combo.currentText()))
        model_name = self.model_combo.currentData() or "rf"
        self._worker = CodingTrainWorker(dataset, model_name=model_name,
                                         folds=self.folds_spin.value(), parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _set_busy(self, busy):
        self.train_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.dim_combo.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.folds_spin.setEnabled(not busy)

    def _on_progress(self, step, total, message):
        if total > 0:
            self.progress.setValue(int(step * 100 / total))
        if message:
            self.status_label.setText(message)

    def _on_done(self, result):
        self._set_busy(False)
        self.progress.setValue(100)
        self.metrics.setPlainText(self._format_metrics(result))
        if not result.get("ok"):
            self._bundle = None
            self.predict_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.status_label.setText("未训练：%s" % result.get("reason", "未知原因"))
            return
        self._bundle = result
        self.predict_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        summary = result.get("summary") or {}
        if summary:
            self.status_label.setText(
                "训练完成：%s，%d 折平均 balanced-acc=%.3f / macro-F1=%.3f（小样本仅供参考）"
                % (result.get("engine", ""), summary.get("folds", 0),
                   summary.get("balanced_accuracy", 0.0), summary.get("f1_macro", 0.0)))
        else:
            self.status_label.setText("训练完成：%s（%s）"
                                      % (result.get("engine", ""), result.get("note", "")))

    def _on_error(self, message):
        self._set_busy(False)
        self.status_label.setText(message)

    def _on_finished(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _on_stop(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("正在取消……")

    def _format_metrics(self, result):
        if not result.get("ok"):
            return "未训练：%s" % result.get("reason", "")
        lines = ["引擎：%s" % result.get("engine", ""),
                 "样本：%d 行；类别：%s"
                 % (result.get("n_samples", 0), "、".join(result.get("classes", [])))]
        per_fold = result.get("per_fold") or []
        if per_fold:
            lines.append("分组 K 折（连续时间块，防泄漏）：")
            for fold in per_fold:
                lines.append("  第 %d 折：balanced-acc=%.3f  macro-F1=%.3f  (n_test=%d)"
                             % (fold["fold"], fold["balanced_accuracy"],
                                fold["f1_macro"], fold["n_test"]))
            summary = result.get("summary") or {}
            lines.append("汇总：%d 折平均 balanced-acc=%.3f  macro-F1=%.3f"
                         % (summary.get("folds", len(per_fold)),
                            summary.get("balanced_accuracy", 0.0),
                            summary.get("f1_macro", 0.0)))
        if result.get("note"):
            lines.append("说明：%s" % result["note"])
        lines.append("现实预期：小样本下精度有限，仅作离线辅助，始终人在环。")
        return "\n".join(lines)

    # ---------- 预测写回 / 保存 ----------
    def _on_predict(self):
        if not self._bundle or not self._bundle.get("model"):
            self.status_label.setText("还没有训练好的模型。")
            return
        dim_index = self._dim_index()
        if dim_index < 0:
            self.status_label.setText("没有可预测的维度。")
            return
        try:
            from auto_coding import apply_suggestions, find_row_at_time
        except Exception as exc:                 # noqa: BLE001 写回不可用不该崩
            self.status_label.setText("写回不可用：%s" % exc)
            return
        framework = self._framework()
        rows = self._rows()
        seconds = self._current_seconds()
        target = seconds if seconds is not None else 0.0
        row_index = find_row_at_time(self.encoder, target)
        feat_index = row_index if row_index is not None else len(rows)
        code, proba = predict_dim_code(self._bundle, sequence_feature_vector(rows, framework, feat_index))
        dim_name = framework[dim_index][0] if dim_index < len(framework) else "维度%d" % dim_index
        if not code:
            self.status_label.setText("模型没给出建议码。")
            return
        proba_text = ("，置信度 %.0f%%" % (proba * 100)) if proba is not None else ""
        out = apply_suggestions(self.encoder, target, [(dim_index, code)],
                                marker=SUGGEST_MARKER_LOCAL)
        if out.get("written"):
            self.status_label.setText("已写回建议：%s=%s%s（待你确认；只填空，未覆盖人工码）"
                                      % (dim_name, code, proba_text))
        else:
            self.status_label.setText("未写回：%s（建议 %s=%s%s）"
                                      % (out.get("note", ""), dim_name, code, proba_text))

    def _on_save(self):
        if not self._bundle or not self._bundle.get("model"):
            self.status_label.setText("还没有训练好的模型可保存。")
            return
        dim_index = self._dim_index()
        framework = self._framework()
        dim_name = framework[dim_index][0] if 0 <= dim_index < len(framework) else "dim%d" % dim_index
        path = user_state.save_file_name(self, user_state.BUCKET_EXPORT, "保存本地编码模型",
                                         "本地编码模型_%s.pkl" % dim_name,
                                         "模型文件 (*.pkl *.joblib)")
        if not path:
            return
        if save_bundle(self._bundle, path):
            self.status_label.setText("模型已保存：%s" % path)
        else:
            self.status_label.setText("保存失败（详见日志）。")
