# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""IND12：特征型机器学习（计划 §2 ``analysis_ml`` / §3 IND12）。

五条口径写死在这里，别的方法不许各写一份：

1. **特征只有一个出处**：全部列来自 ``analysis_features``，本模块只负责选族与丢常数列；
2. **切分绝不按行随机**：同一 ``student_uid``（伪标签）或 ``group_id``（outcome 是组级）
   只能整个儿落在某一边，否则同一个人的相邻时刻会同时出现在训练与测试里，指标虚高；
3. **标签来源如实标注**：数据带 ``outcome`` 才用真值，否则退回 IND07 的簇标签并在结果、
   导出、质检三处写「伪标签」；
4. **指标固定三件**：Macro-F1、balanced accuracy、log-loss，另给混淆矩阵与逐单元预测；
   log-loss 只在**该折测试集出现的类别**上算（训练未见类会让概率为 0、loss 炸成 inf）；
5. **可复现**：切分、森林、重排都吃同一个 ``seed``；XGBoost 缺席时自动只跑 sklearn
   两个模型，``meta.engines`` 里写清实际用的构造参数。

类别不平衡统一用 ``sample_weight = n / (k · n_c)``（sklearn ``class_weight="balanced"``
同口径），三个模型一套做法，免得 HistGB 不支持 ``class_weight`` 时偷偷变成另一种口径。
IsolationForest 只作异常提示（样本天然少，不当结论用）。
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pandas as pd

import analysis_features as af
import analysis_methods as am
import analysis_stats_sequence as sq
from statistic_visualization_data import QAItem, fmt_round

try:                                        # sklearn 是核心依赖，兜底只为「导入失败」的实话
    from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                                  RandomForestClassifier)
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                                 log_loss)
except ImportError:                         # pragma: no cover
    HistGradientBoostingClassifier = IsolationForest = RandomForestClassifier = None
    permutation_importance = None
    balanced_accuracy_score = confusion_matrix = f1_score = log_loss = None

METHOD_ID = "IND12"
METHOD_NAME = "特征型机器学习（分类）"
GROUP_PREDICT = am.GROUP_PREDICT
MODE_INDIVIDUAL = am.MODE_INDIVIDUAL
MIN_ROWS = 8                                 # 少于 8 行不配谈泛化
_BASELINE = "多数类基线"
DEFAULT_SEED = 20240517


def _start():
    return time.perf_counter()


def _ms(marker):
    return int((time.perf_counter() - marker) * 1000)


def blocked(note):
    """阻断结果：本地构造，不反向查注册表（方法声明在本文件末尾，查了会踩循环导入）。"""
    return am.MethodSpec(id=METHOD_ID, name=METHOD_NAME, group="", question="") \
        .blocked_result(None, note)


# ------------------------------------------------------------------ 可选后端与模型

_MODELS = (("rf", "随机森林"), ("histgb", "HistGradientBoosting"), ("xgb", "XGBoost（可选）"))


def optional_backends():
    """可选依赖在场判定：torch / xgboost / shap（对应 requirements-optional.txt）。"""
    return {name: importlib.util.find_spec(name) is not None
            for name in ("torch", "xgboost", "shap")}


def available_models():
    """下拉里该出现的模型：XGBoost 未安装时整条剔除，不留「选了再报错」。"""
    has_xgb = optional_backends().get("xgboost", False)
    return tuple((key, text) for key, text in _MODELS if key != "xgb" or has_xgb)


def _make_model(name, seed):
    """按名字造分类器；返回 ``(estimator, engine 说明)``，造不出来给 ``(None, 原因)``。"""
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=1,
                                      random_state=seed, n_jobs=1), \
            "RandomForestClassifier(n_estimators=200)"
    if name == "histgb":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1,
                                              max_leaf_nodes=15, random_state=seed), \
            "HistGradientBoostingClassifier(max_iter=120)"
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None, "未安装 xgboost（见 requirements-optional.txt）"
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                             random_state=seed, n_jobs=1, eval_metric="mlogloss"), \
            "XGBClassifier(n_estimators=200, max_depth=4)"
    return None, "未知模型：%s" % name


def class_balanced_weights(labels):
    """``n / (k · n_c)`` 逐行权重（sklearn class_weight="balanced" 同口径）。"""
    series = pd.Series([str(value) for value in labels], dtype=str)
    if not len(series):
        return np.zeros(0, dtype=float)
    counts = series.value_counts()
    lookup = {str(index): float(len(series)) / (float(len(counts)) * float(value))
              for index, value in counts.items()}
    return np.asarray([lookup[str(value)] for value in series], dtype=float)


# ------------------------------------------------------------------ 特征与标签

_FAMILY_MATCHERS = {
    "prop": lambda name: name.startswith("prop_"),
    "freq": lambda name: name.startswith("freq_"),
    "entropy": lambda name: name in ("entropy", "entropy_norm", "n_states"),
    "switch": lambda name: name in ("switch_rate", "self_transition", "run_count"),
    "run": lambda name: name in ("run_mean", "run_median", "run_max"),
    "transition": lambda name: name in ("transition_entropy", "top_transition_share"),
    "distance": lambda name: name.startswith(("om_to_centroid", "dtw_to_centroid")),
    "pattern": lambda name: name.startswith("pattern_support_"),
    "model": lambda name: name.startswith(("hmm_state_entropy", "embedding_")),
}

_ID_COLUMNS = ("student_uid", "group_id", "student", "dimension")
_SIZE_COLUMNS = ("n_steps", "n_valid", "n_missing")   # 样本量不是行为特征，进模型必泄答案


def feature_columns(frame, families=None):
    """按特征族挑列：只留数值列，文本列与常数列一律不进模型。

    ``families`` 留空 = 全部数值特征（含 IND07/IND08 注册的距离/模型族）。
    """
    picked = [name for name in (families or []) if name in _FAMILY_MATCHERS]
    out = []
    for column in frame.columns:
        if column in _ID_COLUMNS or column in _SIZE_COLUMNS:
            continue
        if af.is_text_feature(column):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            continue
        if picked and not any(match(column) for match in
                              [_FAMILY_MATCHERS[name] for name in picked]):
            continue
        if float(values.std(ddof=0) or 0.0) <= 1e-12:      # 常数列对分类没有信息量
            continue
        out.append(column)
    return out


def _cluster_labels(ctx, params):
    """IND07 簇标签：``({"uid / dimension": 簇}, {uid: 多数簇})``；跑不动时给原因。

    IND07 的标签行是 ``build_distance_matrix`` 的产物 —— 单通道是 ``uid / D1``，
    多通道（MCO/MCM）只有 ``uid``，所以两级兜底都要备着。
    """
    spec = am.method_by_id("IND07")
    base = {item.key: item.default for item in (spec.params if spec else ())}
    base.update({"students": params.get("students"), "dimensions": params.get("dimensions"),
                 "groups": params.get("groups")})
    for key in ("method", "compress", "cluster_method", "linkage", "k"):
        if params.get(key) not in (None, ""):
            base[key] = params[key]
    out = sq.sequence_clustering(ctx, base)
    if getattr(out, "blocked", False):
        return None, str(getattr(out, "blocked_reason", "") or "IND07 未出结果"), base
    frame = (out.tables or {}).get("clusters")
    if frame is None or not len(frame):
        return None, "IND07 的 clusters 表为空", base
    per_row, per_uid = {}, {}
    for record in frame.itertuples(index=False):
        label = str(getattr(record, "序列", "") or "")
        cluster = int(record.簇)
        per_row[label] = cluster
        per_uid.setdefault(label.split(" / ")[0], []).append(cluster)
    majority = {uid: int(pd.Series(values, dtype=int).value_counts().idxmax())
                for uid, values in per_uid.items()}
    return (per_row, majority), "", base


def design_matrix(ctx, params):
    """IND12 的设计矩阵：行 = student×dimension，列 = 选中特征族，y = outcome 或簇标签。

    返回 ``(frame, columns, label_source, notes)``；不满足条件时 frame 为空、notes 说明原因。
    """
    uids = set(am.chosen(ctx.students, params.get("students")))
    dimensions = [ctx.dimension_key(item) for item in
                  (params.get("dimensions") or ctx.dimensions)]
    sequences = [item for item in ctx.sequences(min_length=am.MIN_SEQUENCE_LENGTH)
                 if item.uid in uids and ctx.dimension_key(item.dimension) in dimensions]
    if not sequences:
        return pd.DataFrame(), [], "", ["没有可建模的序列：长度都要 ≥ %d 步，且落在选中的学生/维度里"
                                        % am.MIN_SEQUENCE_LENGTH]
    base = af.feature_frame(sequences, {item: ctx.codebook_of(item) for item in dimensions})
    columns = feature_columns(base, params.get("families"))
    if not columns:
        return pd.DataFrame(), [], "", ["选中特征族里没有非常数数值列（当前 %d 列全被剔除）"
                                        % len(base.columns)]
    wanted = str(params.get("label_source") or "auto")
    notes, labels, source = [], [], ""
    if wanted in ("auto", "outcome"):
        values = {str(bundle.group_id): str(bundle.outcome)
                  for bundle in ctx.bundles if bundle.outcome is not None}
        if len(values) >= 2 and len(set(values.values())) >= 2:
            labels = [values.get(str(group), "") for group in base["group_id"]]
            source = "outcome"
        elif wanted == "outcome":
            return pd.DataFrame(), [], "", ["数据里的 outcome 不能作标签：带值的组 %d 个、"
                                            "取值种类 %d 类（至少各需 2）"
                                            % (len(values), len(set(values.values())))]
    if not source and wanted in ("auto", "pseudo"):
        found, reason, used = _cluster_labels(ctx, params)
        if found is None:
            if wanted == "pseudo":
                return pd.DataFrame(), [], "", ["IND07 未能出簇标签（%s），伪标签无从取" % reason]
        else:
            per_row, majority = found
            labels = [per_row.get("%s / %s" % (uid, dimension),
                                  majority.get(str(uid), -1))
                      for uid, dimension in zip(base["student_uid"], base["dimension"])]
            source = "伪标签（IND07 簇）"
            notes.append("伪标签取自 IND07（method=%s、k=%s），改这两个参数就会换一套标签"
                         % (str(used.get("method") or "OM").upper(), used.get("k")))
    if not source:
        return pd.DataFrame(), [], "", ["没有可用标签：数据缺 outcome，且未取到 IND07 簇标签"]
    frame = base.copy()
    frame["label"] = [str(value) for value in labels]
    frame = frame[frame["label"] != ""].reset_index(drop=True)
    unknown = int((frame["label"] == "-1").sum())
    if unknown:
        notes.append("%d 行没在 IND07 的标签里出现（该序列被长度/筛选挡在外面），标为 -1 仍参与训练"
                     % unknown)
    notes.append("标签来源：%s；行 = student×dimension，切分单元 = %s"
                 % (source, "group_id（outcome 是组级量）" if source == "outcome"
                    else "student_uid"))
    if len(dimensions) > 1:
        notes.append("跨维度建模：%d 个维度的同名码列共用（``prop_A`` 在 D1 与 D2 里含义不同），"
                     "想按维度分别建模请只勾一个维度" % len(dimensions))
    return frame, columns, source, notes


# ------------------------------------------------------------------ 切分（绝不按行随机）

def grouped_split(keys, folds=5, seed=DEFAULT_SEED):
    """按单元的组键做分组 K 折：同一组键全进全出，折数不超过组键数。

    返回 ``[(train_index, test_index)]``，下标指向 ``keys`` 的行位置。
    """
    texts = [str(key) for key in keys]
    unique = list(dict.fromkeys(texts))
    if len(unique) < 2:
        return []
    folds = int(max(2, min(int(folds), len(unique))))
    order = np.random.default_rng(int(seed)).permutation(len(unique)).tolist()
    buckets = [[] for _ in range(folds)]
    for position, index in enumerate(order):
        buckets[position % folds].append(unique[index])
    out = []
    for fold in range(folds):
        held = set(buckets[fold])
        train = [row for row, key in enumerate(texts) if key not in held]
        test = [row for row, key in enumerate(texts) if key in held]
        if train and test:
            out.append((np.asarray(train, dtype=int), np.asarray(test, dtype=int)))
    return out


def leave_one_group_out(keys):
    """留一组出：每个组键当一次测试集（LOYO-CV，跨组预测的标准口径）。"""
    texts = [str(key) for key in keys]
    out = []
    for held in dict.fromkeys(texts):
        train = [row for row, key in enumerate(texts) if key != held]
        test = [row for row, key in enumerate(texts) if key == held]
        if train and test:
            out.append((np.asarray(train, dtype=int), np.asarray(test, dtype=int)))
    return out


def holdout_split(keys, ratio=0.25, seed=DEFAULT_SEED):
    """单次留出：按组键整块划分，取最后一折当测试（组键数 < 2 时没有折可给）。"""
    ratio = min(max(float(ratio or 0.25), 0.05), 0.5)
    return grouped_split(keys, folds=max(2, int(round(1.0 / ratio))), seed=seed)[-1:]


def make_folds(frame, params, source):
    """按参数选切分方式：返回 ``(切分名, [(train, test)], 切分单元列名)``。"""
    unit_column = "group_id" if source == "outcome" else "student_uid"
    keys = [str(value) for value in frame[unit_column]]
    mode = str(params.get("cv") or "kfold")
    seed = int(params.get("seed") or DEFAULT_SEED)
    if mode == "loyo":
        return "留一组出 LOYO", leave_one_group_out(keys), unit_column
    if mode == "holdout":
        return ("单次留出（按 %s 整块划分）" % unit_column), \
            holdout_split(keys, params.get("test_size"), seed), unit_column
    folds = max(int(params.get("folds") or 5), 2)
    return ("分组 %d 折（同一 %s 全进全出）" % (folds, unit_column)), \
        grouped_split(keys, folds, seed), unit_column


# ------------------------------------------------------------------ 指标与训练

def _align_proba(estimator, x_test, classes, predicted):
    """把 ``predict_proba`` 的列对到本折类别全集上：测试集含训练未见类时不崩。"""
    columns = len(classes)
    try:
        raw = estimator.predict_proba(x_test)
    except AttributeError:
        raw = None
    out = np.zeros((len(predicted), columns), dtype=float)
    if raw is None:
        for row, value in enumerate(predicted):
            if value in classes:
                out[row, classes.index(value)] = 1.0
        return out
    seen = [str(value) for value in getattr(estimator, "classes_", [])]
    for position, label in enumerate(seen):
        if label in classes and position < raw.shape[1]:
            out[:, classes.index(label)] = np.asarray(raw[:, position], dtype=float)
    totals = out.sum(axis=1)
    uniform = 1.0 / max(columns, 1)
    out[totals <= 1e-12, :] = uniform
    return out


def metrics(y_true, y_pred, proba, classes):
    """三件固定指标 + accuracy。log-loss 只在 y_true 实际出现的类别上算（见模块 docstring）。

    IND13/IND14/GRP10 的预测类方法共用这一份，保证「同一指标名在哪个方法里都是同一个算法」。
    """
    probabilities = np.asarray(proba, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    present = sorted({str(value) for value in y_true})
    kept = [classes.index(label) for label in present if label in classes]
    subset = probabilities[:, kept] if kept else probabilities
    subset = subset / np.clip(subset.sum(axis=1, keepdims=True), 1e-12, None)
    loss = float(log_loss([str(value) for value in y_true], subset, labels=present)) \
        if len(present) >= 2 else 0.0
    return {
        "Macro-F1": fmt_round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4),
        "balanced_acc": fmt_round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "log_loss": fmt_round(loss, 4),
        "accuracy": fmt_round(float(np.mean([str(left) == str(right)
                                             for left, right in zip(y_true, y_pred)])), 4),
    }


_metrics = metrics                           # 旧名兼容：单测与 analysis_dl 两个名字都能用


def majority_baseline(y, folds):
    """永远要报的地板：每折都用训练侧多数类预测测试侧。"""
    rows = []
    for fold, (train, test) in enumerate(folds, start=1):
        y_train = [str(y[position]) for position in train]
        y_test = [str(y[position]) for position in test]
        counts = pd.Series(y_train, dtype=str).value_counts()
        top = str(counts.index[0]) if len(counts) else ""
        classes = sorted(set(y_train) | set(y_test))
        uniform = np.tile([1.0 / max(len(classes), 1)], (len(y_test), max(len(classes), 1)))
        row = metrics(y_test, [top] * len(y_test), uniform, classes)
        row.update({"模型": _BASELINE, "折": fold, "n_train": int(len(train)),
                    "n_test": int(len(test)), "备注": "常数预测器（多数类）"})
        rows.append(row)
    return rows


def cross_validate(x, y, columns, folds, models, seed, class_weight, n_repeats, ctx=None):
    """逐折逐模型跑：返回逐折指标行、混淆行、预测行、重要度行、附加说明。"""
    rows, confusion_rows, prediction_rows, importance_rows, notes = [], [], [], [], []
    for model_name, model_text in models:
        for fold, (train, test) in enumerate(folds, start=1):
            if ctx is not None and getattr(ctx, "cancelled", False):
                notes.append("用户取消：IND12 在 %s 第 %d 折前停止" % (model_text, fold))
                return rows, confusion_rows, prediction_rows, importance_rows, notes
            if ctx is not None and callable(getattr(ctx, "progress", None)):
                ctx.progress("%s：第 %d/%d 折（训练 %d 行 / 测试 %d 行）"
                             % (model_text, fold, len(folds), len(train), len(test)))
            estimator, engine = _make_model(model_name, seed + fold)
            if estimator is None:
                rows.append({"模型": model_text, "折": fold, "Macro-F1": np.nan,
                             "balanced_acc": np.nan, "log_loss": np.nan, "accuracy": np.nan,
                             "n_train": int(len(train)), "n_test": int(len(test)),
                             "备注": engine})
                notes.append("%s 不可用：%s" % (model_text, engine))
                continue
            y_train = [str(y[position]) for position in train]
            y_test = [str(y[position]) for position in test]
            classes = sorted(set(y_train) | set(y_test))
            weights = class_balanced_weights(y_train) if class_weight else None
            estimator.fit(x[train], y_train,
                          **({"sample_weight": weights} if weights is not None else {}))
            predicted = [str(value) for value in estimator.predict(x[test])]
            proba = _align_proba(estimator, x[test], classes, predicted)
            unseen = sorted(set(y_test) - set(y_train))
            if unseen:
                notes.append("%s 第 %d 折的测试集含训练未见类 %s：该类 recall 必为 0，"
                             "指标按全集平均" % (model_text, fold, "、".join(unseen)))
            row = metrics(y_test, predicted, proba, classes)
            row.update({"模型": model_text, "折": fold, "n_train": int(len(train)),
                        "n_test": int(len(test)), "备注": engine})
            rows.append(row)
            table = confusion_matrix(y_test, predicted, labels=classes)
            for row_index, true_label in enumerate(classes):
                for column_index, pred_label in enumerate(classes):
                    if int(table[row_index, column_index]):
                        confusion_rows.append({"模型": model_text, "折": fold,
                                               "真实": str(true_label),
                                               "预测": str(pred_label),
                                               "计数": int(table[row_index, column_index])})
            for position, index in enumerate(test):
                record = {"行号": int(index) + 1, "折": fold, "模型": model_text,
                          "真实": y_test[position], "预测": predicted[position],
                          "预测正确": int(y_test[position] == predicted[position])}
                for class_index, label in enumerate(classes):
                    record["置信_%s" % label] = fmt_round(float(proba[position, class_index]), 4)
                prediction_rows.append(record)
            if n_repeats and permutation_importance is not None and len(test) >= 2:
                scored = permutation_importance(
                    estimator, x[test], y_test, scoring="f1_macro",
                    n_repeats=int(n_repeats), random_state=seed + fold, n_jobs=1)
                for position, value in enumerate(scored.importances_mean):
                    importance_rows.append({"模型": model_text, "特征": columns[position],
                                            "重要度(ΔMacro-F1)": fmt_round(float(value), 5),
                                            "标准差": fmt_round(
                                                float(scored.importances_std[position]), 5)})
    return rows, confusion_rows, prediction_rows, importance_rows, notes


def anomaly_scores(x, contamination, seed):
    """IsolationForest：逐行的「反常程度」提示，顺序与输入一致，不参与任何统计结论。"""
    if IsolationForest is None or not len(x):
        return pd.DataFrame()
    forest = IsolationForest(n_estimators=200,
                             contamination=float(min(max(float(contamination or 0.1), 0.01), 0.5)),
                             random_state=seed)
    flags = forest.fit_predict(x)
    scores = forest.decision_function(x)
    return pd.DataFrame({"异常": [int(value == -1) for value in flags],
                         "孤立森林分": [fmt_round(float(value), 4) for value in scores]})


# ------------------------------------------------------------------ IND12

def ml_analysis(ctx, params):
    """IND12：特征 → 分类器（RF / HistGB / XGBoost-可选）+ 重要度 + 异常提示。"""
    marker = _start()
    if RandomForestClassifier is None or f1_score is None:
        return blocked("未安装 scikit-learn，IND12 无法运行")
    if getattr(ctx, "cancelled", False):
        # 工作台取消后再算一次会把标记清掉，所以这里为真就是「本次一开始就被取消」
        return blocked("已取消：计算在训练日志卡被中止，未产生任何折结果")
    frame, columns, source, notes = design_matrix(ctx, params)
    if not len(frame):
        return blocked("；".join(notes) or "没有可建模的特征表：换一组特征族或补数据")
    classes = sorted(set(frame["label"]))
    if len(classes) < 2:
        return blocked("标签只有 1 类（%s）：分类没有可比对象，改用 IND07 / GRP07 这类无监督方法"
                       % classes[0])
    wanted = [str(item) for item in (params.get("models") or [])] or ["rf", "histgb"]
    picked = [(key, text) for key, text in available_models() if key in wanted]
    if not picked:
        picked = list(available_models()[:2])
    seed = int(params.get("seed") or DEFAULT_SEED)
    cv_name, folds, unit_column = make_folds(frame, params, source)
    if not folds:
        return blocked("切不出同时含训练与测试的折：独立单元只有 %d 个（%s），"
                       "请减少筛选或改用分组 K 折" % (frame[unit_column].nunique(), unit_column))
    x = np.nan_to_num(frame[columns].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = list(frame["label"])
    rows, confusion_rows, prediction_rows, importance_rows, extra = cross_validate(
        x, y, columns, folds, picked, seed, bool(params.get("class_weight", True)),
        int(params.get("perm_repeats") or 0), ctx)
    notes += extra
    if not rows:
        return blocked("没有任何一折完成：" + ("；".join(extra) or "检查模型选择与切分"))
    baseline_rows = majority_baseline(y, folds)
    summary_rows = []
    for model_text in [text for _key, text in picked] + [_BASELINE]:
        part = [row for row in rows + baseline_rows
                if row["模型"] == model_text and np.isfinite(float(row["Macro-F1"]))]
        if not part:
            continue
        values = [float(row["Macro-F1"]) for row in part]
        summary_rows.append({"模型": model_text, "折数": len(part),
                             "Macro-F1 均值": fmt_round(float(np.mean(values)), 4),
                             "Macro-F1 标准差": fmt_round(float(np.std(values, ddof=0)), 4),
                             "balanced_acc 均值": fmt_round(float(np.mean(
                                 [row["balanced_acc"] for row in part])), 4),
                             "log_loss 均值": fmt_round(float(np.mean(
                                 [row["log_loss"] for row in part])), 4)})
    summary = pd.DataFrame(summary_rows).sort_values("Macro-F1 均值", ascending=False) \
        if summary_rows else pd.DataFrame()
    tables = {
        "features": frame[list(_ID_COLUMNS) + ["label"] + columns].copy(),
        "cv_plan": pd.DataFrame([{"切分方式": cv_name, "切分单元": unit_column, "折": fold,
                                  "训练行数": int(len(train)), "测试行数": int(len(test)),
                                  "测试含类数": len({str(y[position]) for position in test})}
                                 for fold, (train, test) in enumerate(folds, start=1)]),
        "metrics": pd.DataFrame(rows + baseline_rows),
        "model_selection": summary,
        "confusion": pd.DataFrame(confusion_rows),
        "predictions": pd.DataFrame(prediction_rows),
    }
    if importance_rows:
        tables["importance"] = pd.DataFrame(importance_rows).sort_values(
            "重要度(ΔMacro-F1)", ascending=False).reset_index(drop=True)
    qa = [QAItem("warn", "IND12.pseudo_label",
                 "标签是 IND07 簇标签（伪标签）：模型学到的是「聚类器怎么分」，不是外部真值")
          if source.startswith("伪标签") else
          QAItem("info", "IND12.outcome_label",
                 "标签来自数据里的 outcome（组级），切分已按 group_id 整块划分以防泄漏")]
    if not len(tables.get("importance", [])):
        qa.append(QAItem("info", "IND12.no_importance",
                         "重要度重排次数为 0（或测试行不足），本次没有 permutation importance 表"))
    if len(frame) < MIN_ROWS:
        qa.append(QAItem("warn", "IND12.small_sample",
                         "只有 %d 行样本（< %d），分类指标波动极大，只作流程演示"
                         % (len(frame), MIN_ROWS)))
    units = int(frame[unit_column].nunique())
    if len(classes) > units:
        qa.append(QAItem("warn", "IND12.few_units",
                         "类别数 %d 多于独立单元数 %d：至少一类永远学不到" % (len(classes), units)))
    counts = frame.groupby("label")[unit_column].nunique()
    thin = [str(label) for label, value in counts.items() if int(value) < 2]
    if thin:
        qa.append(QAItem("warn", "IND12.thin_class",
                         "类别 %s 只有 1 个（或 0 个）独立单元，全进全出切分下无法同时出现在"
                         "训练与测试" % "、".join(thin)))
    if getattr(ctx, "cancelled", False):
        qa.append(QAItem("warn", "IND12.cancelled",
                         "计算中途取消：只完成 %d 行折指标（计划 %d 模型 × %d 折），"
                         "汇总数不可与完整运行比较" % (len(rows), len(picked), len(folds))))
    if bool(params.get("anomaly", True)):
        scores = anomaly_scores(x, params.get("contamination"), seed)
        if len(scores):
            flagged = pd.concat([frame[list(_ID_COLUMNS)].reset_index(drop=True), scores], axis=1)
            tables["anomalies"] = flagged.sort_values("孤立森林分").reset_index(drop=True)
            qa.append(QAItem("info", "IND12.anomalies",
                             "IsolationForest 标出 %d/%d 行为最反常（预期比例 %s）：只作复核线索"
                             % (int(scores["异常"].sum()), len(scores),
                                fmt_round(float(params.get("contamination") or 0.1), 2))))
    engines = sorted({str(row.get("备注")) for row in rows if row.get("备注")})
    meta = {"label_source": source, "n_rows": int(len(frame)), "n_features": len(columns),
            "n_classes": len(classes), "classes": classes, "cv": cv_name, "cv_unit": unit_column,
            "folds": len(folds), "seed": seed,
            "models": [text for _key, text in picked], "engines": engines,
            "optional_backends": optional_backends(), "feature_columns": columns,
            "n_units": units}
    notes += ["特征族：%s（进模型 %d 列 / 全表 %d 列）"
              % ("、".join(str(item) for item in (params.get("families") or [])) or "全部数值列",
                 len(columns), len(frame.columns)),
              "切分：%s，单元 = %s；同一单元不同时出现在训练与测试" % (cv_name, unit_column),
              "重要度 = 重排 %d 次后的 ΔMacro-F1 均值（scoring=f1_macro）"
              % int(params.get("perm_repeats") or 0)]
    if source.startswith("伪标签"):
        notes.append("伪标签实验的解释口径：指标高于多数类基线只说明「这些特征足以复现簇」，"
                     "不代表对学习效果有预测力")
    return am.result(METHOD_ID, METHOD_NAME, tables, meta=meta, qa=qa, notes=notes,
                     elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ 参数与方法声明

P_LABEL_SOURCE = am.ParamSpec("label_source", "标签来源", "combo",
                              options=(("auto", "自动（有 outcome 用 outcome，否则伪标签）"),
                                       ("pseudo", "只用 IND07 簇标签（伪标签）"),
                                       ("outcome", "只用数据里的 outcome")),
                              default="auto",
                              tooltip="outcome 是组级监督目标；缺省时退回伪标签并全程标注")
P_FAMILIES = am.ParamSpec("families", "特征族", "checklist",
                          options=am.FEATURE_FAMILY_CHOICES,
                          default=("prop", "entropy", "switch"),
                          tooltip="留空 = 全部数值特征；口径见 analysis_features 的特征字典")
P_MODELS = am.ParamSpec("models", "模型", "checklist", options=available_models(),
                        default=("rf", "histgb"),
                        tooltip="XGBoost 需另装 requirements-optional.txt，未装时整条不出现")
P_CV = am.ParamSpec("cv", "切分方式", "combo",
                    options=(("kfold", "分组 K 折"), ("holdout", "单次留出"),
                             ("loyo", "留一组出 LOYO")), default="kfold",
                    tooltip="三种都按独立单元整块切，绝不按行随机")
P_FOLDS = am.ParamSpec("folds", "折数", "spin", default=5, minimum=2, maximum=10,
                       shows_for=("cv", "kfold"))
P_TEST_SIZE = am.ParamSpec("test_size", "测试集比例", "dspin", default=0.25, minimum=0.1,
                           maximum=0.5, shows_for=("cv", "holdout"))
P_SEED = am.ParamSpec("seed", "随机种子", "spin", default=DEFAULT_SEED, minimum=0,
                      maximum=999999)
P_CLASS_WEIGHT = am.ParamSpec("class_weight", "类别不平衡加权", "check", default=True,
                              tooltip="按 n/(k·n_c) 给训练样本加权，三个模型同一口径")
P_PERM_REPEATS = am.ParamSpec("perm_repeats", "重要度重排次数", "spin", default=20,
                              minimum=0, maximum=200,
                              tooltip="0 = 不算 permutation importance（大表时省时间）")
P_ANOMALY = am.ParamSpec("anomaly", "附异常检测（IsolationForest）", "check", default=True)
P_CONTAMINATION = am.ParamSpec("contamination", "预期异常比例", "dspin", default=0.1,
                               minimum=0.01, maximum=0.3, shows_for=("anomaly", True))

SPECS = [
    am.MethodSpec(
        id=METHOD_ID, name=METHOD_NAME, group=GROUP_PREDICT,
        question="这些行为特征能把人分到不同的簇 / 结果里去吗？哪几列最有信息量？",
        mode=MODE_INDIVIDUAL, compute=ml_analysis,
        params=(P_LABEL_SOURCE, P_FAMILIES, P_MODELS, P_CV, P_FOLDS, P_TEST_SIZE, P_SEED,
                P_CLASS_WEIGHT, P_PERM_REPEATS, P_ANOMALY, P_CONTAMINATION),
        requires=("students>=4",), unit="student", engine=am.ENGINE_PYTHON,
        formula_note="同一 student_uid/group_id 全进全出（grouped split / LOYO）；指标 = "
                     "Macro-F1、balanced accuracy、log-loss（log-loss 只在本折出现类上算）；"
                     "重要度 = 重排后 ΔMacro-F1 均值；伪标签实验只报「可复现簇」程度"),
]
