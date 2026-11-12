"""统计公式手算对照（计划 §9）。

每一件被检的算法都在测试里手写期望值：小样本、能用心算或草稿纸复核，必要时再拿
scipy 当第二个独立实现交叉验证。目的是"数字对不对"，不是"跑没跑通"——后者由
``_smoke_modes.py`` 负责。

运行方式：
    cd version/v5
    pytest test_stats_formulas.py -v
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_V5_DIR = os.path.dirname(os.path.abspath(__file__))
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)

import analysis_core as ac                                    # noqa: E402
import analysis_dl as dl                                      # noqa: E402
import analysis_features as af                                # noqa: E402
import analysis_methods as am                                 # noqa: E402
import analysis_ml as ml                                      # noqa: E402
import analysis_stats_group as gp                             # noqa: E402
import analysis_stats_network as nt                           # noqa: E402
import analysis_stats_sequence as sq                          # noqa: E402
import interaction_model as im                                # noqa: E402
import statistic_visualization_data as svd                     # noqa: E402
import statistic_visualization_methods as svm                  # noqa: E402


def seq(codes, student="S1", group="g1", dimension="D1"):
    """最小可用的 Sequence：时间戳按 1 步递增，只关心码表本身。"""
    return ac.Sequence(group, student, dimension,
                       [float(step) for step in range(len(codes))], codes)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND01 LSA：O / E / 调整残差 Z
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLsaResiduals:
    def test_two_by_two_expected_and_z(self):
        """[[4,2],[3,1]] → 去对角后 total=5，E/Z 全部草稿纸可核。"""
        observed = [[4, 2], [3, 1]]
        off, expected, z = sq.lsa_residuals(observed)
        assert off.tolist() == [[0, 2], [3, 0]]
        # 行合计 [2, 3]、列合计 [3, 2]、total 5
        assert expected[0, 1] == pytest.approx(2 * 2 / 5)
        assert expected[1, 0] == pytest.approx(3 * 3 / 5)
        # Z01 = (2 - 0.8) / sqrt(0.8 * (1-2/5) * (1-2/5)) = 1.2 / sqrt(0.288)
        assert z[0, 1] == pytest.approx(1.2 / math.sqrt(0.288), abs=1e-9)
        # Z10 = (3 - 1.8) / sqrt(1.8 * (1-3/5) * (1-3/5))：与 Z01 同值（2×2 的对称性）
        assert z[1, 0] == pytest.approx(1.2 / math.sqrt(0.288), abs=1e-9)
        assert z[0, 0] == 0.0 and z[1, 1] == 0.0

    def test_three_by_three_row_column_margins(self):
        observed = [[2, 3, 1], [4, 1, 2], [1, 5, 3]]
        off, expected, z = sq.lsa_residuals(observed)
        total = off.sum()
        assert total == 16.0
        assert off.sum(axis=1).tolist() == [4.0, 6.0, 6.0]
        assert off.sum(axis=0).tolist() == [5.0, 8.0, 3.0]
        assert expected[0, 1] == pytest.approx(4 * 8 / 16)
        assert expected[2, 0] == pytest.approx(6 * 5 / 16)
        assert z[0, 1] == pytest.approx(1.0 / math.sqrt(2.0 * 0.375), abs=1e-9)
        assert np.all(np.diag(z) == 0.0)

    def test_degenerate_table_gives_all_zero(self):
        _off, expected, z = sq.lsa_residuals([[0, 0], [0, 0]])
        assert not expected.any() and not z.any()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND01 Yule's Q
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestYule:
    def test_q_basic(self):
        assert am.yules_q([[3, 7], [4, 6]]) == pytest.approx(-10 / 46, abs=1e-6)

    def test_q_accepts_flat_table(self):
        assert am.yules_q([3, 7, 4, 6]) == am.yules_q([[3, 7], [4, 6]])

    def test_q_extremes_and_degenerate(self):
        assert am.yules_q([[10, 0], [0, 10]]) == pytest.approx(1.0)
        assert am.yules_q([[0, 10], [10, 0]]) == pytest.approx(-1.0)
        assert am.yules_q([[0, 0], [0, 0]]) == 0.0

    def test_q_rejects_wrong_size(self):
        with pytest.raises(ValueError):
            am.yules_q([[1, 2, 3], [4, 5, 6]])

    def test_yule_matrix_uses_own_pair_blocks(self):
        observed = [[2, 3, 1], [4, 1, 2], [1, 5, 3]]
        matrix = sq.yule_matrix(observed)
        # (0,1) 取 [[O00,O01],[O10,O11]] = [[2,3],[4,1]] → (2-12)/(2+12)
        assert matrix[0, 1] == pytest.approx(-10 / 14, abs=1e-6)
        assert matrix[1, 0] == matrix[0, 1]
        assert np.all(np.diag(matrix) == 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND02 一阶 Markov
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMarkov:
    def test_row_normalisation(self):
        counts = [[2, 2, 0], [1, 3, 6], [0, 0, 4]]
        probabilities, rows = sq.markov_probabilities(counts)
        assert rows.tolist() == [4.0, 10.0, 4.0]
        assert probabilities[0].tolist() == pytest.approx([0.5, 0.5, 0.0])
        assert probabilities[1].tolist() == pytest.approx([0.1, 0.3, 0.6])
        assert probabilities.sum(axis=1).tolist() == pytest.approx([1, 1, 1])

    def test_absorbing_row_stays_zero(self):
        probabilities, rows = sq.markov_probabilities([[1, 1], [0, 0]])
        assert probabilities[1].tolist() == [0.0, 0.0]
        assert rows[1] == 0.0

    def test_stationary_two_state_chain(self):
        pi = sq.stationary_distribution([[0.5, 0.5], [0.25, 0.75]])
        assert pi[0] == pytest.approx(1 / 3, abs=1e-6)
        assert pi[1] == pytest.approx(2 / 3, abs=1e-6)

    def test_stationary_is_left_eigenvector(self):
        matrix = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.0, 0.4, 0.6]])
        pi = sq.stationary_distribution(matrix)
        assert (pi @ matrix).tolist() == pytest.approx(pi.tolist(), abs=1e-6)
        assert pi.sum() == pytest.approx(1.0, abs=1e-9)

    def test_entropy_rate_and_initial_entropy(self):
        """纯交替 A↔B：转移确定 → 熵率 0，初始熵 ln2，ratio 0。"""
        stats = sq.markov_statistics([[0, 2], [2, 0]])
        assert stats["entropy_rate"] == pytest.approx(0.0, abs=1e-12)
        assert stats["initial_entropy"] == pytest.approx(math.log(2), abs=1e-9)
        assert stats["ratio"] == pytest.approx(0.0)
        assert stats["stationary"].tolist() == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_loglik_order_zero_and_one(self):
        sequences = [seq(["A", "A", "B", "B"])]
        loglik, parameters, transitions, contexts = sq.loglik_order(sequences, 1)
        assert transitions == 3                                  # A→A, A→B, B→B
        assert parameters == 1                                   # (2-1) + (1-1)
        assert loglik == pytest.approx(2 * math.log(0.5), abs=1e-9)
        assert contexts == {("A",): {"A": 1, "B": 1}, ("B",): {"B": 1}}

    def test_loglik_zero_order_is_marginal(self):
        loglik, parameters, transitions, _contexts = sq.loglik_order(
            [seq(["A", "A", "B", "B"])], 0)
        assert transitions == 4 and parameters == 1
        assert loglik == pytest.approx(4 * math.log(0.5), abs=1e-9)

    def test_loglik_never_pools_across_sequences(self):
        """两条独立序列内部的转移数 = 各自 len-1 之和，首尾不得相连。"""
        _l, _p, transitions, _c = sq.loglik_order(
            [seq(["A", "B"]), seq(["A", "B"])], 1)
        assert transitions == 2

    def test_pooled_counts_respects_lag(self):
        states, counts = sq.pooled_counts([seq(["A", "B", "A", "B"])], ["A", "B"], lag=1)
        assert states == ["A", "B"]
        assert counts.tolist() == [[0.0, 2.0], [1.0, 0.0]]
        _states, lag2 = sq.pooled_counts([seq(["A", "B", "A", "B"])], ["A", "B"], lag=2)
        # 隔一步：(A,A) 与 (B,B) 各一次
        assert lag2.tolist() == [[1.0, 0.0], [0.0, 1.0]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND04 n-gram / PrefixSpan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPatterns:
    def test_token_stream_raw_and_compressed(self):
        sequence = seq(["A", "A", "", "B", "B"])
        assert sq.valid_codes(sequence) == ["A", "A", "B", "B"]
        assert sq.token_stream(sequence, "raw") == ["A", "A", "B", "B"]
        assert sq.token_stream(sequence, "compressed") == ["A", "B"]

    def test_ngram_support_counts_students_not_windows(self):
        sequences = [seq(["A", "B", "A", "B"]), seq(["A", "B", "C", "C"])]
        frame = sq.ngram_table(sequences, n=2, min_sup=0.5)
        row = frame[frame["gram"] == "A→B"].iloc[0]
        assert row["occurrences"] == 3                           # 2 + 1 个窗口
        assert row["序列数"] == 2 and row["总序列"] == 2
        assert row["support"] == pytest.approx(1.0)               # 2/2，不是 3/5
        assert row["confidence"] == pytest.approx(1.0)            # 前缀 A 也覆盖两条

    def test_prefix_span_grows_by_one_item(self):
        frame = sq.prefix_span([seq(["A", "B", "C"]), seq(["A", "B", "C"])],
                               min_sup=1.0, max_length=2)
        assert set(frame["子序列"]) == {"A", "B", "C", "A→B", "A→C", "B→C"}
        assert (frame["support"] == 1.0).all()
        assert set(frame["长度"]) == {1, 2}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND05 OM / HAM / DHD / 多通道
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDistances:
    def test_om_ab_vs_ba_takes_cheaper_of_two_routes(self):
        """AB vs BA：要么两次替换（2s），要么一次删除一次插入（2d）→ 取小者。"""
        states = ["A", "B"]
        index = {state: position for position, state in enumerate(states)}
        unit = sq.substitution_costs([], states, "constant", 1.0)
        assert sq.om_distance(["A", "B"], ["B", "A"], unit, index, indel=1.0) == 2.0
        assert sq.om_distance(["A", "B"], ["B", "A"], unit, index, indel=0.5) == 1.0
        cheap = sq.substitution_costs([], states, "constant", 0.4)
        assert sq.om_distance(["A", "B"], ["B", "A"], cheap, index, indel=1.0) == 0.8

    def test_om_identity_and_single_insertion(self):
        states = ["A", "B", "C"]
        index = {state: position for position, state in enumerate(states)}
        cost = sq.substitution_costs([], states, "constant", 1.0)
        assert sq.om_distance(["A", "B"], ["A", "B"], cost, index) == 0.0
        assert sq.om_distance(["A", "B"], ["A", "B", "C"], cost, index) == 1.0

    def test_hamming_counts_mismatching_positions(self):
        states = ["A", "B"]
        index = {state: position for position, state in enumerate(states)}
        cost = sq.substitution_costs([], states, "constant", 1.0)
        assert sq.hamming_distance(["A", "B"], ["B", "A"], cost, index) == 2.0
        assert sq.hamming_distance(["A", "B"], ["A", "B"], cost, index) == 0.0
        # 不等长：只比公共前缀
        assert sq.hamming_distance(["A", "B", "A"], ["B", "A"], cost, index) == 2.0

    def test_trate_costs(self):
        """TRATE：d(i,j) = 2 − p(i→j) − p(j→i)。A→A/A→B/B→A → 0.5。"""
        states = ["A", "B"]
        cost = sq.transition_rate_costs([seq(["A", "A", "B", "A"])], states)
        assert cost[0, 1] == pytest.approx(0.5)
        assert cost[1, 0] == pytest.approx(0.5)
        assert cost[0, 0] == 0.0 and cost[1, 1] == 0.0

    def test_custom_costs_parsing_is_symmetric_and_defaulted(self):
        states = ["A", "B", "C"]
        cost = sq.parse_custom_costs("A,B=1.5; B,C=2", states)
        index = {state: position for position, state in enumerate(states)}
        assert cost[index["A"], index["B"]] == 1.5 == cost[index["B"], index["A"]]
        assert cost[index["B"], index["C"]] == 2.0
        assert cost[index["A"], index["C"]] == 1.0               # 未给出 → 默认 1
        assert np.all(np.diag(cost) == 0.0)

    def test_index_distance_channels_of_state(self):
        # 同通道不同状态：2/k
        assert sq.index_distance([("D1", "A", 1)], [("D1", "B", 1)],
                                 {"A": 1, "B": 1}) == pytest.approx(2.0)
        assert sq.index_distance([("D1", "A", 1)], [("D1", "B", 1)],
                                 {"A": 2, "B": 2}) == pytest.approx(1.0)
        # 不同通道：1/k + 1/k'
        assert sq.index_distance([("D1", "A", 1)], [("D2", "B", 1)],
                                 {"A": 2, "B": 2}) == pytest.approx(1.0)
        # 同一元素 / indel 兜底
        assert sq.index_distance([("D1", "A", 1)], [("D1", "A", 1)], {"A": 1}) == 0.0
        assert sq.index_distance([], [("D1", "A", 1)], {"A": 5}) == 1.0

    def test_multichannel_distance_averages_channels(self):
        assert sq.multichannel_distance([["A"], ["B"]], [["A"], ["B"]]) == 0.0
        # 两通道只有一致性差异：pair_cost = (0 + 1)/2 = 0.5
        assert sq.multichannel_distance([["A"], ["B"]], [["A"], ["C"]]) == 0.5

    def test_build_distance_matrix_indel_ratio(self):
        sequences = [seq(["A", "B"], student="S1"), seq(["B", "A"], student="S2")]
        params = {"method": "OM", "compress": False, "indel_ratio": 0.5,
                  "subst": "constant", "cost_value": 1.0}
        labels, matrix, note = sq.build_distance_matrix(sequences, params)
        assert labels[0].startswith("g1__S1") and "D1" in labels[0]
        assert matrix[0, 1] == pytest.approx(1.0)                # min(2*0.5, 2*1)
        assert "OM" in note

    def test_dhd_equals_unit_cost_optimal_match(self):
        sequences = [seq(["A", "B"], student="S1"), seq(["B", "A"], student="S2")]
        params = {"method": "DHD", "compress": False, "indel_ratio": 0.5,
                  "subst": "constant", "cost_value": 1.0}
        _labels, matrix, note = sq.build_distance_matrix(sequences, params)
        assert matrix[0, 1] == pytest.approx(2.0)                # 单位代价 + indel 1
        assert "DHD" in note

    def test_matrix_frame_layout(self):
        frame = sq.matrix_frame(["A", "B"], [[1, 2], [3, 4]])
        assert list(frame.columns) == ["状态", "A", "B"]
        assert frame["状态"].tolist() == ["A", "B"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND07 聚类
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClustering:
    def test_silhouette_point_by_point(self):
        dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        values, mean = sq.silhouette_scores(dist, [0, 0, 1])
        # 点0：a=1, b=2 → 0.5；点1：a=1, b=1 → 0；点2 独苗 → 0
        assert values == pytest.approx([0.5, 0.0, 0.0])
        assert mean == pytest.approx(1 / 6)

    def test_silhouette_requires_two_clusters(self):
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert sq.silhouette_scores(dist, [0, 0]) == ([0.0, 0.0], 0.0)

    def test_adjusted_rand_index_hand_cases(self):
        assert sq.adjusted_rand_index([0, 0, 1, 1], [2, 2, 3, 3]) == pytest.approx(1.0)
        # 交叉划分：index 0, expected 2*2/6, max 2 → −0.5
        assert sq.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)

    def test_adjusted_rand_index_matches_sklearn(self):
        from sklearn.metrics import adjusted_rand_score
        first = [0, 0, 1, 1, 2, 2, 2]
        second = [1, 1, 1, 0, 0, 2, 2]
        assert sq.adjusted_rand_index(first, second) == pytest.approx(
            adjusted_rand_score(first, second), abs=1e-9)

    def test_co_clustering_jaccard(self):
        assert sq.co_clustering_jaccard([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
        # 交叉划分：同簇对集不相交 → 交集 0、并集 4
        assert sq.co_clustering_jaccard([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)

    def test_pam_build_and_assign(self):
        dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
        labels, medoids = sq.pam_clusters(dist, k=2)
        assert labels[0] == labels[1] != labels[2]
        assert len(set(labels)) == 2
        assert medoids == [0, 2]                                 # Build 阶段即选到最优

    def test_ahc_two_clusters_on_chain(self):
        dist = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]])
        labels, tree = sq.ahc_clusters(dist, linkage="average", k=2)
        assert labels[0] == labels[1] != labels[2]
        assert len(tree) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 熵族（M17–M19 与 IND/GRP 共用口径）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEntropyFamily:
    def test_shannon_ln_and_normalized(self):
        values = ["A", "A", "B", "B"]
        assert af.entropy_of(values) == pytest.approx(math.log(2), abs=1e-9)
        assert af.normalized_entropy_of(values) == pytest.approx(1.0, abs=1e-9)
        assert am.entropy(values) == pytest.approx(math.log(2), abs=1e-9)
        assert am.normalized_entropy(values) == pytest.approx(1.0, abs=1e-9)

    def test_entropy_base_switch(self):
        values = ["A", "A", "B", "B"]
        assert am.entropy(values, base=2.0) == pytest.approx(1.0, abs=1e-9)

    def test_single_state_and_empty(self):
        assert af.entropy_of(["A", "A"]) == pytest.approx(0.0)
        assert af.normalized_entropy_of(["A", "A"]) == 0.0
        assert af.entropy_of([]) == 0.0

    def test_four_way_uniform(self):
        assert af.entropy_of(["A", "B", "C", "D"]) == pytest.approx(math.log(4), abs=1e-9)

    def test_transition_entropy_conditional_form(self):
        """H(Y|X) = −Σ p(x) Σ p(y|x) ln p(y|x)。"""
        assert af.transition_entropy(["A", "A", "B"]) == pytest.approx(
            -(0.5 * math.log(0.5) + 0.5 * math.log(0.5)), abs=1e-9)
        # 确定性交替 → 条件熵 0
        assert af.transition_entropy(["A", "B", "A", "B"]) == pytest.approx(0.0, abs=1e-12)
        assert af.transition_entropy(["A"]) == 0.0

    def test_switch_retention_and_runs(self):
        values = ["A", "A", "B", "B", "B", "C"]
        assert af.switch_rate(values) == pytest.approx(2 / 5)
        assert af.self_transition(values) == pytest.approx(3 / 5)
        assert af.run_lengths(values) == [2, 3, 1]
        assert af.switch_rate(values) + af.self_transition(values) == pytest.approx(1.0)
        assert af.top_transition(values) == ("B→B", pytest.approx(2 / 5))

    def test_sequence_features_columns(self):
        frame = af.feature_frame([seq(["A", "A", "B", "C", ""], student="S3")])
        row = frame.iloc[0]
        assert row["student_uid"] == "g1__S3" and row["n_valid"] == 4
        assert row["n_missing"] == 1 and row["run_count"] == 3
        assert row["n_states"] == 3
        for column in ("freq_", "prop_", "entropy", "entropy_norm", "switch_rate",
                       "self_transition", "transition_entropy", "run_mean"):
            assert any(name.startswith(column) or name == column for name in frame.columns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRP03 同步度 / GRP04 领先-滞后
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSynchrony:
    def test_perfect_agreement(self):
        track = ["A", "B", "A", "B", "A", "B"]
        value, matched = gp.agreement(track, list(track))
        assert value == 1.0 and matched == 6

    def test_lagged_agreement_shifts_one_step(self):
        first = ["A", "B", "A", "B", "A", "B"]
        second = ["B", "A", "B", "A", "B", "A"]
        assert gp.agreement(first, second)[0] == 0.0
        value, matched = gp.agreement(first, second, lag=1)
        assert value == 1.0 and matched == 5

    def test_missing_cells_are_out_of_denominator(self):
        first = ["A", "", "A", "B", "C", "B"]
        second = ["A", "B", "A", "B", "A", "B"]
        value, matched = gp.agreement(first, second)
        assert matched == 5                                      # 空格不参与，分母不是 6
        assert value == pytest.approx(4 / 5)

    def test_chance_agreement_uses_marginals(self):
        first = ["A", "B", "A", "B", "A", "B"]
        assert gp.chance_agreement(first, list(first)) == pytest.approx(0.5)
        assert gp.chance_agreement(["A", "A", "A", "A", "A"],
                                   ["A", "A", "A", "A", "A"]) == pytest.approx(1.0)

    def test_cross_agreement_curve_peaks_at_true_shift(self):
        first = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
        second = ["B", "A", "B", "A", "B", "A", "B", "A", "B", "A"]
        rows = {row["lag"]: row for row in gp.cross_agreement_curve(first, second, 2)}
        assert set(rows) == {-2, -1, 0, 1, 2}
        assert rows[0]["agreement"] == pytest.approx(0.0)
        assert rows[0]["chance"] == pytest.approx(0.5)
        assert rows[0]["excess"] == pytest.approx(-0.5)
        assert rows[1]["agreement"] == pytest.approx(1.0)
        assert rows[-1]["agreement"] == pytest.approx(1.0)
        assert rows[2]["agreement"] == pytest.approx(0.0)
        assert rows[2]["重叠点数"] == 8

    def test_lagged_mutual_information_peaks_at_true_shift(self):
        first = ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B"]
        second = ["X", "A", "A", "B", "B", "C", "C", "A", "A", "B"]
        late = gp.lagged_mutual_information(first, second, 1)
        none = gp.lagged_mutual_information(first, second, 0)
        assert late > none


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRP10 跨组非参数比较内核
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNonparametric:
    def test_ranks_average_ties(self):
        ranks, tie = gp._ranks([10, 20, 20, 30])
        assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]
        assert tie == 2 ** 3 - 2                                 # 一组 2 个并列

    def test_kruskal_hand_case(self):
        result = gp.kruskal_wallis([[1, 2], [3, 4], [5, 6]])
        # R = [3, 7, 11]，n_i = 2，n = 6 → H = 12/42 * 89.5 − 21
        assert result["H"] == pytest.approx(12.0 / 42.0 * 89.5 - 21.0, abs=1e-6)
        assert result["df"] == 2
        assert result["p"] == pytest.approx(math.exp(-result["H"] / 2.0), abs=1e-6)
        assert result["groups"] == [2, 2, 2]

    def test_kruskal_matches_scipy_with_ties(self):
        from scipy.stats import kruskal
        samples = [[1, 2, 2, 4], [5, 5, 8], [3, 6]]
        ours = gp.kruskal_wallis(samples)
        statistic, p_value = kruskal(*samples)
        assert ours["H"] == pytest.approx(statistic, abs=1e-5)
        assert ours["p"] == pytest.approx(p_value, abs=1e-5)

    def test_kruskal_needs_two_groups_and_four_points(self):
        assert gp.kruskal_wallis([[1, 2, 3]]) is None
        assert gp.kruskal_wallis([[1], [2]]) is None

    def test_mann_whitney_hand_case(self):
        result = gp.mann_whitney_u([1, 2], [3, 4])
        assert result["U"] == 0.0 and result["U_a"] == 0.0
        assert result["n1"] == 2 and result["n2"] == 2
        assert result["rank_biserial"] == pytest.approx(-1.0)
        sigma = math.sqrt(2 * 2 / 12.0 * 5)
        z_exact = -2.0 / sigma                                   # p 用未取整的 z 算
        assert result["z"] == pytest.approx(z_exact, abs=1e-4)
        assert result["p"] == pytest.approx(math.erfc(abs(z_exact) / math.sqrt(2)),
                                            abs=1e-9)

    def test_mann_whitney_matches_scipy_approximation(self):
        from scipy.stats import mannwhitneyu
        a, b = [1, 3, 3, 7], [2, 4, 5, 6, 9]
        ours = gp.mann_whitney_u(a, b)
        ref = mannwhitneyu(a, b, method="asymptotic", use_continuity=False)
        assert ours["U_a"] == pytest.approx(ref.statistic, abs=1e-6)
        assert ours["p"] == pytest.approx(ref.pvalue, abs=1e-4)

    def test_cliffs_delta_bounds(self):
        assert gp.cliffs_delta([3, 4], [1, 2]) == pytest.approx(1.0)
        assert gp.cliffs_delta([1, 2], [3, 4]) == pytest.approx(-1.0)
        assert gp.cliffs_delta([1, 4], [2, 3]) == pytest.approx(0.0)

    def test_bh_fdr_monotone(self):
        assert am.bh_fdr([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.03, 0.03])
        assert am.bh_fdr([0.001, 0.3, 0.4]) == pytest.approx([0.003, 0.4, 0.4])
        assert am.bh_fdr([]) == []
        assert am.bh_fdr([0.6, 0.9]) == pytest.approx([0.9, 0.9])   # 单调化：1.2→0.9

    def test_fisher_exact_hand_and_scipy(self):
        from scipy.stats import fisher_exact
        assert am.fisher_exact_two_by_two([[2, 0], [0, 2]]) == pytest.approx(1 / 3,
                                                                            abs=1e-6)
        for table in ([[1, 9], [9, 1]], [[5, 1], [1, 5]], [[3, 7], [4, 6]]):
            assert am.fisher_exact_two_by_two(table) == pytest.approx(
                fisher_exact(table).pvalue, abs=1e-4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRP06 异质性：Gini 与成员间分布熵
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDistributionShape:
    def test_gini_extremes(self):
        assert gp.gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-12)
        assert gp.gini([1, 0, 0, 0]) == pytest.approx(0.75)
        assert gp.gini([5, 5]) == pytest.approx(0.0, abs=1e-12)
        assert gp.gini([0, 0]) == 0.0

    def test_distribution_entropy_scale(self):
        assert gp.distribution_entropy([1, 1, 1, 1]) == pytest.approx(1.0, abs=1e-12)
        assert gp.distribution_entropy([4, 0, 0, 0]) == pytest.approx(0.0, abs=1e-12)
        shares = [0.75, 0.25]
        raw = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        assert gp.distribution_entropy([3, 1]) == pytest.approx(raw / math.log(2))
        assert gp.distribution_entropy([shares[0] * 4, shares[1] * 4]) == \
            pytest.approx(raw / math.log(2))

    def test_gini_and_entropy_agree_on_ordering(self):
        even, skewed = [1, 1, 1, 1], [7, 1, 1, 1]
        assert gp.gini(skewed) > gp.gini(even)
        assert gp.distribution_entropy(skewed) < gp.distribution_entropy(even)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRP08 / GRP09 网络度量
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNetworkMetrics:
    def test_density_reciprocity_and_degree(self):
        edges = pd.DataFrame([
            {"source": "A", "target": "B", "权重": 1.0},
            {"source": "B", "target": "A", "权重": 2.0},
            {"source": "B", "target": "C", "权重": 1.0},
        ])
        out = gp.network_metrics(edges, directed=True)
        assert out["节点数"] == 3 and out["边数"] == 3
        assert out["密度"] == pytest.approx(3 / 6, abs=5e-5)        # n(n−1) = 6，报 4 位
        assert out["互惠率"] == pytest.approx(2 / 3, abs=5e-5)      # A↔B 两条都算
        assert out["无向边对数"] == 2
        assert out["出度均值"] == pytest.approx((1.0 + 3.0) / 2)
        assert out["出度最大"] == pytest.approx(3.0)

    def test_undirected_density(self):
        edges = pd.DataFrame([{"节点A": "A", "节点B": "B"}, {"节点A": "B", "节点B": "C"}])
        out = gp.network_metrics(edges, directed=False)
        assert out["节点数"] == 3
        assert out["密度"] == pytest.approx(2 / 3, abs=5e-5)        # n(n−1)/2 = 3
        assert "互惠率" not in out

    def test_empty_edges(self):
        assert gp.network_metrics(pd.DataFrame(columns=["source", "target"])) == {}

    def test_cross_dimension_counts_lag_window(self):
        """跨维共现：A_t × B_{t+lag} 只数同一学生、同一栅格。"""
        tracks = {"g1__S1": {"D1": ["A", "B"], "D2": ["X", "X"]}}
        states_a, states_b, counts = sq.cross_dimension_counts(tracks, "D1", "D2", 0)
        assert (states_a, states_b) == (["A", "B"], ["X"])
        assert counts.tolist() == [[1.0], [1.0]]
        _sa, _sb, lag1 = sq.cross_dimension_counts(tracks, "D1", "D2", 1)
        assert lag1.tolist() == [[1.0], [0.0]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 契约与工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHelpers:
    def test_normal_and_chi2_tails_match_scipy(self):
        from scipy.stats import chi2, norm
        for z in (-2.5, -0.5, 0.0, 1.96):
            assert sq.normal_two_sided_p(z) == pytest.approx(
                2 * norm.sf(abs(z)), abs=1e-9)
        for x, df in ((0.5, 1), (3.84, 1), (5.99, 2), (4.57, 2)):
            assert sq.chi2_sf(x, df) == pytest.approx(chi2.sf(x, df), abs=1e-6)

    def test_transition_counts_helper(self):
        assert am.transition_counts(["A", "B", "A"], lag=1) == {"A": {"B": 1},
                                                                "B": {"A": 1}}
        assert am.transition_counts(["A", "B", "A"], lag=2) == {"A": {"A": 1}}

    def test_sequence_compressed_and_uid(self):
        sequence = seq(["A", "A", "B"], student="S2")
        assert sequence.compressed() == ["A", "B"]
        assert sequence.uid == "g1__S2"
        assert ac.split_uid(sequence.uid) == ("g1", "S2")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND06 DTW / soft-DTW / DBA 重心
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _onehot(codes, states=("A", "B")):
    """简写：码流 → one-hot 矩阵（DTW 家族吃的是矩阵，不是码流）。"""
    return sq.one_hot_matrix(list(codes), list(states))


class TestDtw:
    def test_one_hot_matrix(self):
        assert sq.one_hot_matrix(["A", "B", "A"], ["A", "B"]).tolist() == \
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        # 字母表外的码：整行全零（局部代价按“谁都不像”算）
        assert sq.one_hot_matrix(["A", "Z"], ["A", "B"]).tolist() == \
            [[1.0, 0.0], [0.0, 0.0]]

    def test_local_cost_is_euclidean_of_one_hot_rows(self):
        cost = sq.local_cost_matrix(_onehot("AB"), _onehot("BA"))
        assert cost.tolist() == [[pytest.approx(math.sqrt(2)), 0.0],
                                 [0.0, pytest.approx(math.sqrt(2))]]

    def test_classic_dtw_hand_cases(self):
        # [A,B] ↔ [B,A]：只能走 3 格阶梯（(0,0)(0,1)(1,1) 或 (0,0)(1,0)(1,1)），2×√2
        assert sq.dtw_distance(_onehot("AB"), _onehot("BA")) == pytest.approx(2 * math.sqrt(2))
        # [A,B] ↔ [A,A,B]：重复的 A 可免费压格，代价 0
        assert sq.dtw_distance(_onehot("AB"), _onehot("AAB")) == pytest.approx(0.0)
        assert sq.dtw_distance(_onehot("A"), _onehot("B")) == pytest.approx(math.sqrt(2))
        # 等长全异码：逐格 1-1 对齐最省，总代价 n·√2 —— 距离不做长度归一
        assert sq.dtw_distance(_onehot("AAAA"), _onehot("BBBB")) == pytest.approx(4 * math.sqrt(2))

    def test_dtw_path_and_empty_inputs(self):
        table = sq.dtw_cost_matrix(_onehot("ABA"), _onehot("A"))
        assert sq.dtw_path(table) == [(0, 0), (1, 0), (2, 0)]
        assert table[-1, -1] == pytest.approx(math.sqrt(2))
        assert math.isnan(sq.dtw_distance(_onehot(""), _onehot("AB")))
        assert sq.dtw_path(np.full((0, 3), np.inf)) == []

    def test_soft_min_is_smooth_underestimate(self):
        assert sq.soft_min([2.0, 4.0], 0.0) == 2.0                      # γ≤0 → 硬 min
        assert sq.soft_min([2.0, 4.0], 1e-6) == pytest.approx(2.0, abs=1e-5)
        assert sq.soft_min([2.0, 4.0], 1.0) == pytest.approx(
            -math.log(math.exp(-2.0) + math.exp(-4.0)))
        assert sq.soft_min([float("inf"), 3.0], 1.0) == pytest.approx(3.0)
        assert math.isinf(sq.soft_min([float("inf")], 1.0))

    def test_soft_dtw_bounded_by_classic(self):
        left, right = _onehot("AABBB"), _onehot("BBBAA")
        classic = sq.dtw_distance(left, right)
        assert classic == pytest.approx(4 * math.sqrt(2))            # 对角 1-1，仅第 3 格同码
        assert sq.dtw_distance(left, right, 0.0) == pytest.approx(classic)
        error = {gamma: classic - sq.dtw_distance(left, right, gamma)
                 for gamma in (0.05, 0.2, 1.0)}
        assert all(value >= 0.0 for value in error.values())          # soft ≤ 经典
        assert error[0.05] < error[0.2] < error[1.0]                  # γ→0 收回硬 min
        # 每格偏差 ≤ γ·ln3（三邻域），整条对角路径共 5 格（宽算）
        assert error[0.05] <= 0.05 * 5 * math.log(3.0)
        assert error[0.05] == pytest.approx(0.19756, abs=1e-4)

    def test_barycenter_degenerate_cases(self):
        matrix = _onehot("AAB")
        assert sq.dtw_barycenter([matrix], ["A", "B"]).tolist() == matrix.tolist()
        assert sq.dtw_barycenter([], ["A", "B"]).shape == (0, 2)
        center = sq.dtw_barycenter([matrix, matrix], ["A", "B"])
        assert center.shape == matrix.shape
        # 重心是 one-hot 行的逐格平均：每行和为 1，元素落在 [0,1]
        assert np.allclose(center.sum(axis=1), 1.0)
        assert float(center.min()) >= 0.0 and float(center.max()) <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND08 HMM：发射、前向后向 / Viterbi 推断、信忌准则
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHmmInference:
    PI = [0.5, 0.5]
    A = [[0.9, 0.1], [0.3, 0.7]]
    B = [[0.8, 0.2], [0.3, 0.7]]            # 行 = 隐状态，列 = 可见码
    SYMBOLS = ["A", "B"]

    def test_emission_matrix_orientation(self):
        emit = sq.hmm_emission(["A", "B"], self.SYMBOLS, self.B)
        assert emit.tolist() == [[0.8, 0.3], [0.2, 0.7]]   # emit[t, s] = P(o_t | s)
        assert sq.hmm_emission(["Z"], self.SYMBOLS, self.B).shape == (0, 2)

    def test_forward_backward_scaled_probabilities(self):
        emit = sq.hmm_emission(["A", "B"], self.SYMBOLS, self.B)
        out = sq.hmm_inference(emit, self.PI, self.A)
        # α0 未归一 = [0.5·0.8, 0.5·0.3] = [0.40, 0.15] → 缩放因子 0.55
        assert out["scales"][0] == pytest.approx(0.55)
        # α1 未归一 = [0.405, 0.145]·[0.2, 0.7] = [0.0810, 0.1015] → 缩放 0.1825/0.55
        assert out["scales"][1] == pytest.approx(0.1825 / 0.55)
        assert out["loglik"] == pytest.approx(math.log(0.55) + math.log(0.1825 / 0.55))
        assert out["gamma"][0].tolist() == [pytest.approx(2.0 / 3.65),
                                            pytest.approx(1.65 / 3.65)]
        assert out["gamma"][1].tolist() == [pytest.approx(1.62 / 3.65),
                                            pytest.approx(2.03 / 3.65)]
        assert np.allclose(out["gamma"].sum(axis=1), 1.0)

    def test_viterbi_hand_path(self):
        emit = sq.hmm_emission(["A", "B"], self.SYMBOLS, self.B)
        out = sq.hmm_inference(emit, self.PI, self.A)
        # δ1: 从状态 1 过来最大（0.105·0.7 > 0.36·0.2 的对数比较），回溯得 [1, 1]
        assert out["path"] == [1, 1]
        assert not out["degenerate"]

    def test_zero_first_step_is_reported_not_faked(self):
        out = sq.hmm_inference(np.zeros((3, 2)), self.PI, self.A)
        assert out["degenerate"] and out["loglik"] == float("-inf")
        assert out["path"] == []
        assert sq.hmm_inference(np.zeros((0, 2)), self.PI, self.A) is None

    def test_criteria_free_parameters_and_penalties(self):
        free, aic, bic = sq.hmm_criteria(-3.0, 2, [2], 10)
        # (S−1) + S(S−1) + Σ S(V−1) = 1 + 2 + 2 = 5
        assert free == 5
        assert aic == pytest.approx(6.0 + 2.0 * 5)
        assert bic == pytest.approx(6.0 + 5.0 * math.log(10))
        two_channel, _, _ = sq.hmm_criteria(-3.0, 2, [2, 3], 10)
        assert two_channel == 1 + 2 + (2 * 1 + 2 * 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND09 变点分段：滑窗 Hellinger 与 BIC 二分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChangepoints:
    STATES = ["A", "B"]

    def test_code_distribution(self):
        assert sq.code_distribution(["A", "A", "B"], ["A", "B", "C"]).tolist() == \
            [pytest.approx(2 / 3), pytest.approx(1 / 3), 0.0]
        assert sq.code_distribution([], self.STATES).tolist() == [0.5, 0.5]

    def test_hellinger_bounds(self):
        assert sq.hellinger_distance([1, 0], [0, 1]) == pytest.approx(1.0)
        assert sq.hellinger_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
        # 半距离形式：H² = ½[(√0.5−√1)² + (√0.5−√0)²] …草稿纸可核
        assert sq.hellinger_distance([0.5, 0.5], [1, 0]) == pytest.approx(
            math.sqrt(0.5 * ((math.sqrt(0.5) - 1.0) ** 2 + 0.5)))

    def test_window_hellinger_tracks_the_true_break(self):
        values = ["A"] * 10 + ["B"] * 10
        frame = sq.window_hellinger(values, self.STATES, window=5)
        assert len(frame) == len(values) - 2 * 5 + 1            # 滑窗个数 11
        assert frame["Hellinger"].tolist()[0] == pytest.approx(0.0)
        peak = int(frame.loc[frame["Hellinger"].idxmax(), "位置"])
        assert peak == 10                                        # 真变点在 10

    def test_multinomial_loglik_and_bic_guard(self):
        assert sq.multinomial_loglik(["A", "A", "B"], self.STATES) == pytest.approx(
            2 * math.log(2 / 3) + 1 * math.log(1 / 3))
        assert sq.multinomial_loglik([], self.STATES) == 0.0
        values = ["A"] * 4 + ["B"] * 4
        gain = sq.bic_split_gain(values, self.STATES, 4)
        raw = 0.0 + 0.0 - sq.multinomial_loglik(values, self.STATES)
        assert gain == pytest.approx(raw - 0.5 * 1 * math.log(8))
        assert sq.bic_split_gain(values, self.STATES, 0) == -np.inf
        assert sq.bic_split_gain(values, self.STATES, 8) == -np.inf

    def test_find_changepoints_single_break(self):
        values = ["A"] * 10 + ["B"] * 10
        assert sq.find_changepoints(values, self.STATES, min_segment=5) == [10]
        # 无变点：常数序列怎么切都是负增益
        assert sq.find_changepoints(["A"] * 20, self.STATES, min_segment=5) == []
        # 段太短直接不搜：长度 8 < 2×min_segment
        assert sq.find_changepoints(values[:8], self.STATES, min_segment=5) == []
        assert sq.find_changepoints(values, self.STATES, min_segment=5,
                                   penalty=50.0) == []            # 重罚 → 不分段

    def test_segmentation_plot_draws_changepoint_lines(self):
        """变点非空才走得到的绘图分支：拿手造样本把它钉住，不靠真实数据撞上。"""
        codes = ["A", "B"] * 10 + ["B"] * 20          # 前 20 步交替，后 20 步只剩 B
        ctx = am.AnalysisContext([_bundle("g1", {"S1": codes}, dimensions=1)],
                                 mode=am.MODE_INDIVIDUAL)
        params = {"students": list(ctx.students), "dimensions": list(ctx.dimensions),
                  "window": 4, "min_segment": 5, "penalty": 1.0, "compress": False}
        out = sq.segmentation_analysis(ctx, params)
        assert not out.blocked, out.blocked_reason
        assert len(out.tables["changepoints"]), "样本没切出变点，测不到竖线那一段"
        import analysis_plots as ap
        figures = ap.render("IND09", out, ctx, params, strict=True)
        assert figures and all(fig.axes for fig in figures.values())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND10 / IND11 / GRP07 网络与 TCA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bundle(group_id, codes_by_member, dimensions=2):
    """宽表一行一个时间点：列名契约 `Time + S{i}Encoding_{j}` 不可改。"""
    rows = len(list(codes_by_member.values())[0])
    data = {"Time": [step * 10.0 for step in range(rows)]}
    for member, codes in codes_by_member.items():
        data["%sEncoding_1" % member] = list(codes)
    if dimensions > 1:
        for member, codes in codes_by_member.items():
            data["%sEncoding_2" % member] = [str(code).lower() for code in codes]
    return ac.GroupBundle(group_id, wide=pd.DataFrame(data))


def _net_ctx():
    return am.AnalysisContext([_bundle("g1", {"S1": "AABBAABBCCAA", "S2": "BBAABBAACCBB"},
                                              dimensions=2)],
                              mode=am.MODE_INDIVIDUAL)


class TestTcaAndNetwork:
    def _edges(self):
        return pd.DataFrame([
            {"uid": "u1", "node_a": "X", "node_b": "Y", "weight": 2.0},
            {"uid": "u1", "node_a": "Y", "node_b": "Z", "weight": 1.0},
            {"uid": "u2", "node_a": "X", "node_b": "Y", "weight": 1.0},
        ])

    def test_profile_matrix_counts_both_endpoints(self):
        rows, nodes, matrix = nt.profile_matrix(self._edges())
        assert rows == ["u1", "u2"] and nodes == ["X", "Y", "Z"]
        assert matrix.tolist() == [[2.0, 3.0, 1.0], [1.0, 1.0, 0.0]]
        normed = nt.row_normalize(matrix)
        assert normed[0].tolist() == [pytest.approx(1 / 3), pytest.approx(0.5),
                                      pytest.approx(1 / 6)]
        # 全零行不参与归一，也不报错
        assert nt.row_normalize(np.zeros((1, 3))).tolist() == [[0.0, 0.0, 0.0]]

    def test_tca_coordinates_hand_case(self):
        out = nt.tca_coordinates([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], axes=2)
        # 中心化后 CᵀC = [[2,-2],[-2,2]] → 单非零奇异值 √4 = 2，解释率 100%
        assert out["values"][0] == pytest.approx(2.0)
        assert out["explained"][0] == pytest.approx(1.0)
        coords = out["units"][:, 0]
        assert np.allclose(np.abs(coords), [math.sqrt(2), 0.0, math.sqrt(2)])
        assert coords[0] == pytest.approx(-coords[2])           # 两头的单元反号
        assert np.allclose(out["nodes"][:, 0], [-math.sqrt(2), math.sqrt(2)],
                           atol=1e-9) or np.allclose(
            out["nodes"][:, 0], [math.sqrt(2), -math.sqrt(2)], atol=1e-9)

    def test_tca_reference_centering(self):
        profile = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        out = nt.tca_coordinates(profile,
                                 reference=np.asarray([False, False, True]), axes=2)
        # 减去参照行 [3,1] → C = [[-2,2],[-1,1],[0,0]]，CᵀC 非零特征值 10
        assert out["values"][0] == pytest.approx(math.sqrt(10.0))
        assert np.allclose(np.abs(out["units"][:, 0]),
                           [2 * math.sqrt(2), math.sqrt(2), 0.0])
        assert np.allclose(np.abs(out["nodes"][:, 0]), [math.sqrt(5), math.sqrt(5)])
        assert nt.tca_coordinates(np.zeros((0, 0)))["units"].shape == (0, 0)

    def test_welch_contrast(self):
        out = nt.welch_contrast([1.0, 3.0], [2.0, 4.0])
        assert out["差"] == pytest.approx(-1.0)
        assert out["z"] == pytest.approx(-1.0 / math.sqrt(2.0))   # var/2 + var/2 = 2
        assert out["p"] == pytest.approx(sq.normal_two_sided_p(-1.0 / math.sqrt(2.0)))
        thin = nt.welch_contrast([1.0], [2.0, 4.0])
        assert thin["z"] == 0.0 and thin["p"] == 1.0 and thin["n_focal"] == 1

    def test_graph_metrics_and_degrees(self):
        edges = pd.DataFrame([
            {"source": "A", "target": "B", "weight": 1.0},
            {"source": "B", "target": "A", "weight": 2.0},
            {"source": "B", "target": "C", "weight": 1.0},
            {"source": "C", "target": "C", "weight": 9.0},     # 自环：不算关系
        ])
        graph = nt.build_graph(edges)
        assert graph.number_of_edges() == 3 and not graph.has_edge("C", "C")
        out = nt.graph_metrics(graph)
        assert out["节点数"] == 3 and out["边数"] == 3
        assert out["密度"] == pytest.approx(3 / 6, abs=5e-5)
        assert out["平均度"] == pytest.approx(2.0)
        assert out["加权互惠率"] == pytest.approx(2.0 / 4.0)   # 2·min(1,2) / (1+2+1)
        assert out["弱连通分量数"] == 1
        degrees = nt.degree_frame(graph).set_index("节点")
        assert degrees.loc["A", "出度"] == 1 and degrees.loc["A", "入度"] == 1
        assert degrees.loc["A", "出强度"] == pytest.approx(1.0)
        assert degrees.loc["A", "入强度"] == pytest.approx(2.0)
        assert degrees.loc["A", "度中心性"] == pytest.approx(1.0)   # (1+1)/(3−1)
        assert degrees.loc["A", "互惠伙伴数"] == 1
        assert nt.graph_metrics(nt.build_graph(edges.iloc[0:0])) == {}

    def test_unweighted_reciprocity_definition(self):
        edges = pd.DataFrame([{"source": "A", "target": "B", "weight": 1.0},
                              {"source": "B", "target": "C", "weight": 1.0}])
        assert nt.weighted_reciprocity(nt.build_graph(edges)) == 0.0

    def test_member_grid_and_cooccurrence_counts(self):
        """逐成员共现：每人 3 对跳维组合，S1 的 A/B/C = 6/4/2，S2 的 A/B/C = 4/6/2。"""
        ctx = _net_ctx()
        params = {"groups": ["g1"], "students": list(ctx.students),
                  "dimensions": list(ctx.dimensions)}
        grid, dims = nt.member_grid(ctx, params)
        assert dims == ["D1", "D2"]
        assert len(grid) == 12 * 2 * 2                        # 时刻 × 成员 × 维度
        edges = nt.cooccurrence_rows(grid, ctx)
        assert sum(edges.values()) == 12 * 2                  # 每时刻恰好 1 对跳维组合
        per_member = {}
        for (group_id, member, node_a, node_b), weight in edges.items():
            assert group_id == "g1" and node_a != node_b
            assert node_a.startswith("%s:" % ctx.label_of("D1"))
            assert node_b.startswith("%s:" % ctx.label_of("D2"))
            assert node_b.split(":")[1] == node_a.split(":")[1].lower()
            per_member.setdefault(member, {})[node_a] = weight
        assert per_member["S1"] == {"%s:A" % ctx.label_of("D1"): 6,
                                    "%s:B" % ctx.label_of("D1"): 4,
                                    "%s:C" % ctx.label_of("D1"): 2}
        assert per_member["S2"] == {"%s:A" % ctx.label_of("D1"): 4,
                                    "%s:B" % ctx.label_of("D1"): 6,
                                    "%s:C" % ctx.label_of("D1"): 2}

    def test_cooccurrence_frame_min_edge(self):
        ctx = _net_ctx()
        base = {"groups": ["g1"], "students": list(ctx.students),
                "dimensions": list(ctx.dimensions)}
        frame = nt.cooccurrence_frame(ctx, dict(base, min_edge=5))
        assert len(frame) == 2                                 # 只剩 S1 的 A:a = 6 与 S2 的 B:b = 6
        assert frame["weight"].tolist() == [6, 6]
        assert sorted(zip(frame["member"], frame["node_a"])) == [("S1", "D1:A"), ("S2", "D1:B")]
        full = nt.cooccurrence_frame(ctx, dict(base, min_edge=0))
        assert len(full) == 6 and int(full["weight"].sum()) == 24


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND12 特征型 ML：切分不泄漏 + 指标手算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FAST = "ABABABABABAB"
_SLOW = "AABBCCAABBCC"


def _ml_bundle(group_id, codes_by_member, outcome=None):
    """两个维度的宽表（D2 取小写），可带组级 outcome。"""
    rows = len(next(iter(codes_by_member.values())))
    data = {"Time": [step * 10.0 for step in range(rows)]}
    for member, codes in codes_by_member.items():
        data["%sEncoding_1" % member] = list(codes)
        data["%sEncoding_2" % member] = [str(code).lower() for code in codes]
    return ac.GroupBundle(group_id, wide=pd.DataFrame(data),
                          meta={"outcome": outcome} if outcome else {})


def _ml_ctx():
    """2 组 × 2 学生 × 2 维 = 8 条序列、4 个独立单元：两种节奏足够分簇。"""
    return am.AnalysisContext([_ml_bundle("g1", {"S1": _FAST, "S2": _SLOW}),
                               _ml_bundle("g2", {"S1": _FAST, "S2": _SLOW})],
                              mode=am.MODE_INDIVIDUAL)


def _ml_params(ctx, **over):
    spec = am.MethodSpec(id="IND12", name="", group="", question="", params=ml.SPECS[0].params)
    params = {item.key: item.default for item in spec.params}
    params.update({"students": list(ctx.students), "groups": list(ctx.group_ids),
                   "dimensions": list(ctx.dimensions)})
    params.update(over)
    return params


class TestMlSplits:
    def test_grouped_split_never_leaks_units(self):
        keys = ["A", "A", "B", "B", "C", "C", "D", "D"]
        folds = ml.grouped_split(keys, folds=4, seed=7)
        assert len(folds) == 4
        covered = []
        for train, test in folds:
            train_keys = {keys[index] for index in train}
            test_keys = {keys[index] for index in test}
            assert not (train_keys & test_keys)               # 同一单元全进全出
            assert len(test_keys) == 1                        # 4 单元 4 折 → 每折恰好 1 单元
            covered.extend(test.tolist())
        assert sorted(covered) == list(range(8))              # 每行恰好被测试一次

    def test_grouped_split_clamps_folds_to_units(self):
        keys = ["A", "A", "B", "B", "C"]
        assert len(ml.grouped_split(keys, folds=10, seed=3)) == 3
        assert ml.grouped_split(["A", "A", "A"], folds=5) == []   # 单单元切不出折

    def test_leave_one_group_out_covers_every_unit(self):
        keys = ["A", "B", "C", "D", "D"]
        folds = ml.leave_one_group_out(keys)
        assert len(folds) == 4
        for train, test in folds:
            held = {keys[index] for index in test}
            assert len(held) == 1 and held.isdisjoint({keys[index] for index in train})
            assert len(train) + len(test) == len(keys)
        # “D” 有 2 行：它当测试集的那一折测试行数就是 2
        assert sorted(len(test) for _train, test in folds) == [1, 1, 1, 2]

    def test_holdout_split_ratio_becomes_fold_count(self):
        keys = ["u%d" % index for index in range(8)]
        folds = ml.holdout_split(keys, ratio=0.25, seed=11)
        assert len(folds) == 1
        _train, test = folds[0]
        assert len(test) == 2 and len(_train) == 6            # 1/0.25 = 4 折，末折 2 单元

    def test_unit_column_follows_label_source(self):
        outcome_ctx = am.AnalysisContext(
            [_ml_bundle("g1", {"S1": _FAST, "S2": _SLOW}, outcome="high"),
             _ml_bundle("g2", {"S1": _FAST, "S2": _SLOW}, outcome="low")],
            mode=am.MODE_INDIVIDUAL)
        frame, columns, source, _notes = ml.design_matrix(
            outcome_ctx, _ml_params(outcome_ctx))
        assert source == "outcome" and len(frame) == 8 and columns
        name, folds, unit = ml.make_folds(frame, _ml_params(outcome_ctx), source)
        assert unit == "group_id" and len(folds) == 2 and "group_id" in name
        ctx = _ml_ctx()
        frame, columns, source, _notes = ml.design_matrix(ctx, _ml_params(ctx))
        assert source.startswith("伪标签")
        _name, _folds, unit = ml.make_folds(frame, _ml_params(ctx), source)
        assert unit == "student_uid"                          # 伪标签是学生级，不能按组切


class TestMlMetrics:
    def test_class_balanced_weights_hand_computed(self):
        weights = ml.class_balanced_weights(["A", "A", "A", "B"])
        assert weights.tolist() == [pytest.approx(4 / (2 * 3)), pytest.approx(4 / (2 * 3)),
                                    pytest.approx(4 / (2 * 3)), pytest.approx(4 / (2 * 1))]

    def test_three_fixed_metrics(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "B", "B", "B"]
        proba = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]]
        out = ml._metrics(y_true, y_pred, proba, ["A", "B"])
        # F1(A)：precision=1/1, recall=1/2 → 2/3；F1(B)：precision=2/3, recall=1 → 0.8
        # → macro = (2/3 + 0.8) / 2；balanced_acc = mean(recall A, recall B)
        assert out["Macro-F1"] == pytest.approx((2 / 3 + 0.8) / 2, abs=1e-4)
        assert out["balanced_acc"] == pytest.approx((0.5 + 1.0) / 2, abs=1e-9)
        assert out["accuracy"] == pytest.approx(0.75)
        expected = -(math.log(0.9) + math.log(0.4) + math.log(0.8) + math.log(0.7)) / 4
        assert out["log_loss"] == pytest.approx(expected, abs=1e-4)

    def test_log_loss_renormalises_on_present_classes(self):
        """测试集只见一类时不能把概率为 0 的列塞进 log-loss（否则炸成 inf）。"""
        out = ml._metrics(["B", "B"], ["A", "B"], [[0.6, 0.4], [0.5, 0.5]], ["A", "B"])
        assert out["log_loss"] == 0.0                        # 单类 → 不报 loss
        assert out["accuracy"] == pytest.approx(0.5)

    def test_majority_baseline_hand_computed(self):
        y = ["A", "A", "A", "B"]
        folds = [(np.asarray([0, 1, 2]), np.asarray([3]))]
        rows = ml.majority_baseline(y, folds)
        assert len(rows) == 1 and rows[0]["模型"] == ml._BASELINE
        assert rows[0]["Macro-F1"] == 0.0 and rows[0]["balanced_acc"] == 0.0
        assert rows[0]["accuracy"] == 0.0 and rows[0]["n_train"] == 3 and rows[0]["n_test"] == 1

    def test_align_proba_maps_unseen_classes_without_crashing(self):
        class Stub:
            classes_ = np.array(["A"])

            def predict_proba(self, x):                        # noqa: ANN001
                return np.array([[0.7], [0.4]])

        out = ml._align_proba(Stub(), None, ["A", "B"], ["A", "B"])
        assert out.tolist() == [[0.7, 0.0], [0.4, 0.0]]
        no_proba = ml._align_proba(object(), None, ["A", "B"], ["A", "B"])
        assert no_proba.tolist() == [[1.0, 0.0], [0.0, 1.0]]   # 没有 predict_proba → 退成一热

    def test_anomaly_scores_keep_input_order(self):
        x = np.vstack([np.tile([0.0, 0.0], (6, 1)), np.tile([0.0, 1.0], (6, 1)),
                       [[50.0, 50.0]]])
        frame = ml.anomaly_scores(x, 0.1, 20240517)
        assert len(frame) == 13                               # 一行不进不丢
        assert int(frame["异常"].sum()) >= 1
        assert int(frame["孤立森林分"].idxmin()) == 12         # 离群行得分最低


class TestMlPipeline:
    def test_feature_columns_family_filter(self):
        frame = pd.DataFrame([
            {"student_uid": "g1__S1", "group_id": "g1", "dimension": "D1", "student": "S1",
             "n_steps": 12, "n_valid": 12, "n_missing": 0, "n_states": 2,
             "entropy": 0.9, "entropy_norm": 0.9, "switch_rate": 0.1, "self_transition": 0.2,
             "run_count": 3, "run_mean": 4.0, "run_median": 4.0, "run_max": 6,
             "transition_entropy": 0.3, "top_transition": "A→B", "top_transition_share": 0.4,
             "prop_A": 0.5, "prop_B": 0.5, "freq_A": 6, "freq_B": 6, "const": 1.0},
            {"student_uid": "g2__S1", "group_id": "g2", "dimension": "D1", "student": "S1",
             "n_steps": 12, "n_valid": 12, "n_missing": 0, "n_states": 3,
             "entropy": 1.4, "entropy_norm": 1.0, "switch_rate": 0.5, "self_transition": 0.5,
             "run_count": 7, "run_mean": 2.0, "run_median": 1.0, "run_max": 3,
             "transition_entropy": 0.8, "top_transition": "A→A", "top_transition_share": 0.3,
             "prop_A": 0.25, "prop_B": 0.25, "freq_A": 3, "freq_B": 3, "const": 1.0},
        ])
        assert ml.feature_columns(frame, ["prop", "entropy"]) == ["n_states", "entropy",
                                                                  "entropy_norm", "prop_A",
                                                                  "prop_B"]
        picked = ml.feature_columns(frame, None)
        assert "const" not in picked and "top_transition" not in picked
        assert "n_valid" not in picked and "freq_A" in picked
        assert ml.feature_columns(frame, ["switch"]) == ["switch_rate", "self_transition",
                                                         "run_count"]

    def test_design_matrix_pseudo_labels(self):
        ctx = _ml_ctx()
        frame, columns, source, notes = ml.design_matrix(ctx, _ml_params(ctx))
        assert source.startswith("伪标签")
        assert len(frame) == 8 and columns                    # 4 单元 × 2 维
        assert set(frame["label"]) != {""}
        assert any("切分单元 = student_uid" in note for note in notes)
        assert any("伪标签" in note for note in notes)

    def test_design_matrix_outcome_is_group_level(self):
        ctx = am.AnalysisContext([_ml_bundle("g1", {"S1": _FAST, "S2": _SLOW}, outcome="high"),
                                  _ml_bundle("g2", {"S1": _FAST, "S2": _SLOW}, outcome="low")],
                                 mode=am.MODE_INDIVIDUAL)
        frame, _columns, source, _notes = ml.design_matrix(ctx, _ml_params(ctx))
        assert source == "outcome"
        assert sorted(set(frame["label"])) == ["high", "low"]
        assert frame["label"].tolist() == ["high"] * 4 + ["low"] * 4

    def test_design_matrix_without_two_outcome_classes_is_blocked(self):
        ctx = am.AnalysisContext([_ml_bundle("g1", {"S1": _FAST, "S2": _SLOW}),
                                  _ml_bundle("g2", {"S1": _FAST, "S2": _SLOW})],
                                 mode=am.MODE_INDIVIDUAL)
        frame, columns, source, notes = ml.design_matrix(
            ctx, _ml_params(ctx, label_source="outcome"))
        assert frame is not None and len(frame) == 0 and not columns and not source
        assert "outcome" in "；".join(notes)

    def test_ml_analysis_end_to_end_pseudo(self):
        ctx = _ml_ctx()
        params = _ml_params(ctx, perm_repeats=5, anomaly=True)
        out = ml.ml_analysis(ctx, params)
        assert not out.blocked, out.blocked_reason
        for name in ("features", "cv_plan", "metrics", "model_selection", "confusion",
                     "predictions", "importance", "anomalies"):
            assert out.tables.get(name) is not None and len(out.tables[name]), name
        assert out.meta["label_source"].startswith("伪标签")
        assert out.meta["cv_unit"] == "student_uid" and out.meta["n_classes"] >= 2
        assert any(item.code == "IND12.pseudo_label" for item in out.qa)
        assert set(out.tables["importance"]["特征"]) <= set(out.meta["feature_columns"])
        assert not out.tables["metrics"]["Macro-F1"].isna().any()

    def test_ml_analysis_is_seed_reproducible(self):
        ctx = _ml_ctx()
        params = _ml_params(ctx, perm_repeats=3, anomaly=False)
        first = ml.ml_analysis(ctx, params).tables["metrics"].to_dict("records")
        second = ml.ml_analysis(ctx, params).tables["metrics"].to_dict("records")
        assert first == second

    def test_ml_analysis_cross_validate_cancellation(self):
        ctx = _ml_ctx()
        ctx.cancelled = True
        out = ml.ml_analysis(ctx, _ml_params(ctx, anomaly=False))
        assert out.blocked and "取消" in out.blocked_reason

    def test_registry_plotter_and_requires(self):
        spec = am.method_by_id("IND12")
        assert spec is not None and spec.mode == am.MODE_INDIVIDUAL
        assert spec.group == am.GROUP_PREDICT
        assert spec.requires == ("students>=4",)
        assert ml.available_models()[0][0] == "rf"
        assert "xgb" in {key for key, _text in ml._MODELS}
        import analysis_plots as ap
        assert ap.PLOTTERS["IND12"] is ap.plot_ml and "IND12" in ap.TITLES
        figs = ap.render("IND12", ml.ml_analysis(_ml_ctx(),
                                                 _ml_params(_ml_ctx(), perm_repeats=3)),
                         None, {}, strict=True)
        assert figs and all(fig.axes for fig in figs.values())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# M17–M19 描述统计熵族（statviz 侧：逐单元口径，与 af/am 的 IND 侧实现不共用代码）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sv_dataset(**columns):
    """列名按 S1Encoding_1 约定→ student × dimension；时间固定 10 s 一步。"""
    steps = max(len(values) for values in columns.values())
    frame = pd.DataFrame({"Time": [float(step * 10) for step in range(steps)]})
    for name, values in columns.items():
        frame[name] = list(values) + [None] * (steps - len(values))
    return svd.CodingDataset.build(frame, source="pytest-m17-m19")


def _sv(method_id, ds, **override):
    spec = svm.method_spec(method_id)
    params = dict(svm.DEFAULT_PARAMS[method_id])
    params.update(override)
    return spec.compute(ds, params)


class TestDescriptiveEntropyM17M19:
    def test_m17_entropy_and_normalized_hand_case(self):
        table = _sv("M17", _sv_dataset(S1Encoding_1=list("AABB"))).tables["状态熵"]
        row = table.iloc[0]
        assert row["N"] == 4 and row["n_states"] == 2 and row["missing_N"] == 0
        assert row["entropy"] == pytest.approx(math.log(2), abs=1e-4)
        assert row["H_max"] == pytest.approx(math.log(2), abs=1e-4)
        assert row["H_norm"] == pytest.approx(1.0, abs=1e-9)

    def test_m17_k_is_observed_states_not_framework(self):
        """另一学生带来第三种码：K 仍按本单元实际出现的两种算。"""
        ds = _sv_dataset(S1Encoding_1=list("AABB"), S2Encoding_1=list("ABCC"))
        assert len(ds.codes_by_dim["D1"]) == 3
        row = _sv("M17", ds).tables["状态熵"].query("student_id=='S1'").iloc[0]
        assert row["n_states"] == 2
        assert row["H_max"] == pytest.approx(math.log(2), abs=1e-4)

    def test_m17_single_state_is_zero_not_nan(self):
        row = _sv("M17", _sv_dataset(S1Encoding_1=list("AAAA"))).tables["状态熵"].iloc[0]
        assert row["entropy"] == 0.0 and row["H_norm"] == 0.0    # 0/0 必须记 0
        assert math.isfinite(row["H_norm"])

    def test_m17_contributions_add_up_to_entropy(self):
        result = _sv("M17", _sv_dataset(S1Encoding_1=list("AABBB")))
        detail = result.tables["各状态概率贡献"]
        assert float(detail["contribution"].sum()) == pytest.approx(
            result.tables["状态熵"].iloc[0]["entropy"], abs=3e-4)
        assert float(detail["probability"].sum()) == pytest.approx(1.0, abs=3e-4)

    def test_m17_missing_cells_stay_out_of_denominator(self):
        row = _sv("M17", _sv_dataset(S1Encoding_1=["A", "A", None, "B", "", "B"]) ).tables[
            "状态熵"].iloc[0]
        assert (row["N"], row["missing_N"]) == (4, 2)
        assert row["entropy"] == pytest.approx(math.log(2), abs=1e-4)

    def test_m18_switch_retention_hand_case(self):
        row = _sv("M18", _sv_dataset(S1Encoding_1=["A", "A", None, "B", "", "B"]) ).tables[
            "切换与保持"].iloc[0]
        assert row["T_valid"] == 4 and row["n_transitions"] == 3
        assert row["switch_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert row["self_transition"] == pytest.approx(2 / 3, abs=1e-4)
        assert row["n_runs"] == 2 and row["mean_run_length"] == pytest.approx(2.0)
        assert row["longest_run"] == 2

    def test_m18_alternating_sequence_is_all_switches(self):
        row = _sv("M18", _sv_dataset(S1Encoding_1=list("ABABAB"))).tables["切换与保持"].iloc[0]
        assert row["switch_rate"] == pytest.approx(1.0, abs=1e-9)
        assert row["self_transition"] == 0.0
        assert row["n_runs"] == 6 and row["mean_run_length"] == pytest.approx(1.0)

    def test_m18_orders_by_time_not_row_index(self):
        """行序打乱但时间列跟着走：序列口径按时间排序，切换率不能变。"""
        ordered = _sv("M18", _sv_dataset(S1Encoding_1=list("AABB"))).tables["切换与保持"]
        # 与上行同一份数据（0→A、10→A、20→B、30→B），只是把行写成乱序：
        # 若错误地按行索引读，得到的会是 B→A→A→B（switch 2/3），与时间口径可区分
        frame = pd.DataFrame({"Time": [20.0, 0.0, 10.0, 30.0],
                              "S1Encoding_1": ["B", "A", "A", "B"]})
        shuffled = _sv("M18", svd.CodingDataset.build(frame, source="pytest-shuffled")).tables[
            "切换与保持"]
        assert ordered.iloc[0]["switch_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert shuffled.iloc[0]["switch_rate"] == ordered.iloc[0]["switch_rate"]
        assert shuffled.iloc[0]["n_switch"] == ordered.iloc[0]["n_switch"] == 1

    def test_m18_blocks_and_warns_on_single_observation(self):
        """只剩一格：没有转移可对，不能假装切换率是 0。"""
        assert _sv("M18", _sv_dataset(S1Encoding_1=["A", None, None, None])).blocked
        result = _sv("M18", _sv_dataset(S1Encoding_1=list("AABB"),
                                        S2Encoding_1=["A", None, None, None]))
        assert not result.blocked and len(result.tables["切换与保持"]) == 1
        assert any(item.code == "insufficient_transitions" for item in result.qa)

    def test_m19_conditional_entropy_hand_case(self):
        """[A,A,B,B] → A→A、A→B、B→B：H(Y|X) = (2/3)ln2。"""
        row = _sv("M19", _sv_dataset(S1Encoding_1=list("AABB"))).tables["条件熵"].iloc[0]
        hand_conditional = (2 / 3) * math.log(2)
        hand_marginal = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
        assert row["n_transitions"] == 3 and row["n_states"] == 2
        assert row["H_Y_given_X"] == pytest.approx(hand_conditional, abs=2e-4)
        assert row["H_Y"] == pytest.approx(hand_marginal, abs=2e-4)
        assert row["reduction"] == pytest.approx(hand_marginal - hand_conditional, abs=3e-4)
        assert row["H_norm"] == pytest.approx(hand_conditional / math.log(2), abs=3e-4)

    def test_m19_deterministic_chain_has_zero_conditional_entropy(self):
        """严格 A→B→A→B：知道前态就完全决定后态，reduction 恰为 H(Y)。"""
        row = _sv("M19", _sv_dataset(S1Encoding_1=list("ABABAB"))).tables["条件熵"].iloc[0]
        assert row["H_Y_given_X"] == pytest.approx(0.0, abs=1e-9)
        assert row["reduction"] == pytest.approx(row["H_Y"], abs=2e-4)
        assert row["predictability"] == pytest.approx(1.0, abs=1e-3)

    def test_m19_breakdown_and_row_normalisation(self):
        result = _sv("M19", _sv_dataset(S1Encoding_1=list("AABBB")))
        row = result.tables["条件熵"].iloc[0]
        breakdown = result.tables["前态熵分解"]
        assert float(breakdown["contribution"].sum()) == pytest.approx(
            row["H_Y_given_X"], abs=3e-4)
        assert float(breakdown["p_from"].sum()) == pytest.approx(1.0, abs=3e-4)
        condition = result.tables["转移条件分布"]
        rows_sum = condition.groupby("from_code")["p_to_given_from"].sum()
        assert bool(rows_sum.round(4).eq(1.0).all())

    def test_m19_skips_single_state_units(self):
        assert _sv("M19", _sv_dataset(S1Encoding_1=list("AAAA"))).blocked
        result = _sv("M19", _sv_dataset(S1Encoding_1=list("AABB"),
                                        S2Encoding_1=list("AAAA")))
        assert not result.blocked and len(result.tables["条件熵"]) == 1
        assert any(item.code == "insufficient_transitions" for item in result.qa)

    def test_registry_has_entropy_group_with_three_methods(self):
        assert {spec.id for spec in svm.METHODS if spec.group == svm.GROUP_ENTROPY} == {
            "M17", "M18", "M19"}
        assert svm.GROUP_ORDER[-1] == svm.GROUP_ENTROPY
        assert len(svm.METHODS) == 19
        assert svm.method_spec("M18").requires_time and svm.method_spec("M19").requires_time
        assert not svm.method_spec("M17").requires_time      # 边际分布不需要时间顺序
        assert svm.DEFAULT_PARAMS["M19"]["view"] == "compare"
        for method_id in ("M17", "M18", "M19"):
            assert svm.method_spec(method_id).availability(None)[0] is False

    def test_unparsable_table_blocks_instead_of_key_error(self):
        """空长表上 is_missing 会退成 object dtype，选列直接 KeyError：必须走阻断通道。"""
        ds = svd.CodingDataset.build(pd.DataFrame({"Time": [0, 10], "S1D1": ["A", "B"]}),
                                     source="pytest-no-columns")
        assert len(ds.long_df) == 0
        assert list(ds.slice().columns) == svd.LONG_COLUMNS
        for method_id in ("M01", "M02", "M03", "M17", "M18", "M19"):
            result = _sv(method_id, ds)
            assert result.blocked and not result.tables, method_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IND13–IND15 / GRP10：深度学习层（数据件手算 + 网络件形状/因果性）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

needs_torch = pytest.mark.skipif(not dl.TORCH, reason="未安装 torch（可选后端）")

_DL_CODES = ("ABABABABABAB", "AABBCCAABBCC", "ABCABCABCABC", "AABBBBCCCCDD",
             "BBBAAACCCDDD", "ABCBCABCABCA", "AAAABBBBCCCC", "DDCCBBAADDCC")


def _dl_ctx(groups=1):
    """每组 8 名学生 × 12 步 × 2 维：刚好过 DL 的独立单元下限。"""
    bundles = []
    for index in range(groups):
        members = {"S%d" % (position + 1 + 8 * index): codes for position, codes
                   in enumerate(_DL_CODES)}
        bundles.append(_ml_bundle("g%d" % (index + 1), members,
                                  outcome=("高" if index % 2 == 0 else "低")))
    return am.AnalysisContext(bundles, mode=am.MODE_INDIVIDUAL)


def _dl_params(spec_index, ctx, **over):
    params = {item.key: item.default for item in dl.SPECS[spec_index].params}
    params.update({"students": list(ctx.students), "groups": list(ctx.group_ids),
                   "dimensions": list(ctx.dimensions)})
    params.update(over)
    return params


class TestDlDataPieces:
    """不靠 torch 的那半：词表、滑窗、基线、遮码、分桶全部手算。"""

    def test_state_vocab_and_index_codes(self):
        assert dl.state_vocab([seq("ABBA"), seq("BAC")]) == ["A", "B", "C"]
        assert dl.index_codes("ABX", ["A", "B", "C"]) == [0, 1, -1]
        assert dl.state_vocab([seq(["A", "", "B"])]) == ["A", "B"]   # 空格不入词表
        assert dl.state_vocab([]) == []

    def test_window_samples_never_sees_the_target(self):
        samples = dl.window_samples(list(range(6)), window=3, stride=1, min_context=2)
        assert [target for _context, target in samples] == [2, 3, 4, 5]
        assert samples[0] == ((0, 1), 2)
        for context, target in samples:
            assert target not in context and max(context) < target and len(context) <= 3
        assert [target for _c, target in dl.window_samples(list(range(6)), window=2,
                                                           stride=2)] == [2, 4]

    def test_build_next_state_dataset_pads_left_and_drops_unknown(self):
        frame = dl.build_next_state_dataset([seq("AABB", student="S1")], ["A", "B"],
                                            window=3, min_context=2)
        assert frame["unit"].tolist() == ["g1__S1", "g1__S1"]
        assert frame["context"].tolist() == [[-1, 0, 0], [0, 0, 1]]      # 左侧补 pad
        assert frame["target"].tolist() == [1, 1] and frame["last"].tolist() == [0, 1]
        # 词表外的码：既不能当答案，也不能进上下文（只能靠左补 pad 凑定长）
        part = dl.build_next_state_dataset([seq("AABCA", student="S1")], ["A", "B"],
                                          window=3, min_context=2)
        assert part["target"].tolist() == [1, 0]
        assert part["context"].tolist() == [[-1, 0, 0], [-1, 0, 1]]

    def test_baseline_predictors_hand_case(self):
        markov = dl.markov_predictor([(0, 1), (0, 1), (1, 0)], vocab_size=2)
        assert np.allclose(markov.sum(axis=1), 1.0)
        assert markov[0].tolist() == [pytest.approx(1 / 4), pytest.approx(3 / 4)]   # +1 平滑
        assert markov[1].tolist() == [pytest.approx(2 / 3), pytest.approx(1 / 3)]
        assert dl.markov_predictor([], 2).tolist() == [[0.5, 0.5], [0.5, 0.5]]   # 平滑后均匀
        # 边际分布同样带拉普拉斯平滑：[1+2, 1+1]/5
        assert dl.marginal_distribution([0, 0, 1], 2).tolist() == [pytest.approx(0.6),
                                                                   pytest.approx(0.4)]

    def test_context_feature_matrix_layout(self):
        frame = pd.DataFrame({"context": [[-1, 0, 0], [1, 1, 0]], "last": [0, 1]})
        matrix = dl.context_feature_matrix(frame, vocab_size=2)
        assert matrix[:, :2].tolist() == [[1.0, 0.0], [pytest.approx(1 / 3),
                                                       pytest.approx(2 / 3)]]
        assert matrix[:, 2:].tolist() == [[1.0, 0.0], [0.0, 1.0]]        # 末码一热

    def test_two_dimension_projection_is_centered(self):
        coords, explained = dl.two_dimension_projection([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        assert np.allclose(coords.sum(axis=0), 0.0, atol=1e-9)
        assert explained[0] == pytest.approx(1.0, abs=1e-6)
        assert sum(explained) <= 1.0 + 1e-6
        empty, empty_explained = dl.two_dimension_projection(np.zeros((0, 3)))
        assert empty.shape == (0, 2) and empty_explained == [0.0, 0.0]

    def test_split_by_unit_keeps_units_intact(self):
        frame = pd.DataFrame({"unit": ["a", "a", "b", "b", "c", "c", "d", "d"]})
        folds = dl.split_by_unit(frame, folds=2, seed=7, mode="kfold")
        assert len(folds) == 2
        for train, test in folds:
            assert not (set(train.tolist()) & set(test.tolist()))
            left = {frame["unit"][row] for row in train}
            assert not (left & {frame["unit"][row] for row in test})
        assert len(dl.split_by_unit(frame, 2, 7, mode="loyo")) == 4
        assert len(dl.split_by_unit(frame.iloc[0:2], 2, 7, mode="kfold")) == 0

    def test_ssl_targets_masked_and_autoencoder(self):
        tokens = [0, 1, 2, 3]
        assert dl.ssl_targets(tokens, 9, 0.5, np.random.default_rng(1), mode="ae") == \
            (tokens, tokens)
        full_x, full_y = dl.ssl_targets(tokens, 9, 1.0, np.random.default_rng(1))
        assert full_x == [9, 9, 9, 9] and full_y == tokens
        thin_x, thin_y = dl.ssl_targets(tokens, 9, 0.0, np.random.default_rng(1))
        assert sum(value == 9 for value in thin_x) == 1                   # 至少遮一个
        assert all((target >= 0) == (token == 9) for token, target in zip(thin_x, thin_y))

    def test_ssl_padded_and_majority_floor(self):
        x, y = dl.ssl_padded([[0, 1, 2], [1, 0]], mask_id=9, pad_id=8, ratio=1.0, seed=5)
        assert x.shape == y.shape == (2, 3)
        assert x[0].tolist() == [9, 9, 9] and y[0].tolist() == [0, 1, 2]
        assert x[1].tolist() == [9, 9, 8] and y[1].tolist() == [1, 0, -1]  # pad 位不算损失
        assert dl.ssl_padded([[0, 1, 2, 3, 4]], 9, 8, 1.0, 5, max_len=3)[0].shape == (1, 3)
        assert dl.token_majority_share([[0, 0, 1], [0]]) == pytest.approx(3 / 4)
        assert dl.token_majority_share([]) == 0.0

    def test_masked_training_pairs_shapes(self):
        x, y = dl.masked_training_pairs([0, 1, 1, 0], 2, mask_id=2, ratio=0.5, seed=11)
        assert x.shape == (1, 4) and y.shape == (4,)
        assert all((target >= 0) == (token == 2) for token, target in zip(x[0], y))
        assert dl.masked_training_pairs([0], 2, 2)[0].shape == (0, 1)      # 单位置不产样
        assert dl.masked_training_pairs([0, 5, 1], 2, 2, ratio=1.0)[0].shape == (1, 2)

    def test_streams_merge_and_stream_vocab(self):
        items = [seq("AAB", student="S1", dimension="D1"),
                 seq("ABB", student="S1", dimension="D2")]
        merged = dl._streams(items, merge=True)
        assert list(merged) == ["合并词表"]
        assert [codes for _item, codes in merged["合并词表"]][0] == ["D1:A", "D1:A", "D1:B"]
        assert sorted(dl._streams(items, merge=False)) == ["D1", "D2"]
        assert dl.stream_vocab(merged["合并词表"]) == ["D1:A", "D1:B", "D2:A", "D2:B"]

    def test_group_helpers_hand_case(self):
        assert dl.euclidean_matrix([[0.0, 0.0], [3.0, 4.0]])[0, 1] == pytest.approx(5.0)
        assert dl.euclidean_matrix([]).shape == (0, 0)
        means = dl.group_mean_matrix(np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                                     ["a", "b", "a"], ["a", "b", "gone"])
        assert means.tolist() == [[3.0, 4.0], [3.0, 4.0], [0.0, 0.0]]
        assert dl.tcn_receptive_field(3, 2) == 15 and dl.tcn_receptive_field(1, 2) == 3
        assert dl.tcn_receptive_field(4, 3) == 1 + 4 * 15

    def test_majority_prediction_uses_train_side_only(self):
        labels = ["a", "a", "a", "b"]
        folds = ml.leave_one_group_out(labels)
        row, predicted = dl.majority_prediction(labels, folds)
        assert predicted == ["b", "b", "b", "a"]          # 测试侧自己不参与投票
        assert row["模型"] == "多数类基线"
        # 极不平衡下 LOYO 的多数类地板可以是 0 命中：这正是它该被如实报出来的样子
        assert row["accuracy"] == pytest.approx(0.0) and row["n_test"] == 2

    def test_temporal_edge_buckets_normalises_per_group(self):
        nodes = {ac.student_uid("g1", "S1"): 0, ac.student_uid("g1", "S2"): 1}
        edges = pd.DataFrame([{"group_id": "g1", "actor_id": "S1", "target_id": "S2",
                               "timestamp": 0.0},
                              {"group_id": "g1", "actor_id": "S2", "target_id": "S1",
                               "timestamp": 20.0},
                              {"group_id": "g1", "actor_id": "S1", "target_id": "S9",
                               "timestamp": 10.0}])       # 未登记端点：丢
        buckets, stats = dl.temporal_edge_buckets(edges, nodes, buckets=2)
        assert [array.tolist() for array in buckets] == [[[0], [1]], [[1], [0]]]
        assert stats["次数"].tolist() == [1, 1] and len(stats) == 2
        blank, blank_stats = dl.temporal_edge_buckets(edges.iloc[0:0], nodes, buckets=3)
        assert len(blank) == 3 and all(item.shape == (2, 0) for item in blank)
        assert blank_stats.empty
        unparseable = edges.assign(timestamp=["x", "y", "z"])
        empty_buckets, empty_stats = dl.temporal_edge_buckets(unparseable, nodes, buckets=2)
        assert empty_stats.empty and all(item.shape == (2, 0) for item in empty_buckets)


class TestDlGating:
    """三道闸：可选后端缺席、用户取消、独立单元不足都得阻断并说清原因。"""

    def test_helpers_are_honest_about_state(self):
        assert dl.lr_of({"lr": "0.01"}, 1e-3) == 0.01
        assert dl.lr_of({"lr": ""}, 1e-3) == 1e-3
        assert dl.lr_of({"lr": "abc"}, 0.005) == 0.005
        assert dl.lr_of({"lr": "-1"}, 0.1) == 0.1
        assert set(dl.tidy_tables({"a": pd.DataFrame(), "b": pd.DataFrame({"x": [1]})})) == {"b"}
        out = dl.torch_missing("IND13", "名称")
        assert out.blocked and "pip install torch" in out.blocked_reason
        assert dl.cancelled(type("Fake", (), {"cancelled": True})()) and not dl.cancelled(None)

    @needs_torch                                # torch 缺席时先被「未装 torch」挡下，
    def test_too_few_independent_units_blocks(self):   # 单元数闸只有装了才观测得到
        ctx = _ml_ctx()                                   # 4 个单元 < MIN_STUDENTS_DL
        out = dl.next_state_analysis(ctx, _dl_params(0, ctx))
        assert out.blocked and str(dl.MIN_STUDENTS_DL) in out.blocked_reason
        assert not out.tables

    @needs_torch
    def test_cancellation_short_circuits_before_training(self):
        ctx = _dl_ctx()
        ctx.cancelled = True
        for index, method in ((0, dl.next_state_analysis), (1, dl.transformer_ssl_analysis),
                              (2, dl.temporal_gat_analysis), (3, dl.group_prediction_analysis)):
            out = method(ctx, _dl_params(index, ctx))
            assert out.blocked and "取消" in out.blocked_reason, index

    def test_torch_absence_message(self, monkeypatch):
        monkeypatch.setattr(dl, "TORCH", False)
        ctx = _dl_ctx()
        out = dl.next_state_analysis(ctx, _dl_params(0, ctx))
        assert out.blocked and "pip install torch" in out.blocked_reason

    def test_ind15_needs_person_edges(self):
        ctx = _dl_ctx()                                   # 无事件表
        out = dl.temporal_gat_analysis(ctx, _dl_params(2, ctx))
        assert out.blocked
        assert ("PERSON" in out.blocked_reason) or (not dl.TORCH)

    def test_grp10_keeps_two_sections_without_torch(self, monkeypatch):
        """未装 torch 只该跳过 DeepSets，跨组检验与组聚类照常输出。"""
        monkeypatch.setattr(dl, "TORCH", False)
        ctx = am.AnalysisContext([_ml_bundle("g%d" % (index + 1),
                                            {"S1": _DL_CODES[index], "S2": _DL_CODES[index + 2],
                                             "S3": _DL_CODES[index + 4]},
                                            outcome=("高" if index < 2 else "低"))
                                  for index in range(3)], mode=am.MODE_GROUP)
        out = dl.group_prediction_analysis(ctx, _dl_params(3, ctx, max_features=6))
        assert not out.blocked, out.blocked_reason
        assert {"group_features", "clusters", "metrics"} <= set(out.tables)
        assert any("DeepSets" in note and "torch" in note for note in out.notes)
        assert "DeepSets" not in " ".join(out.tables["metrics"]["模型"].astype(str))

    def test_registry_declares_optional_engine_and_plots(self):
        assert [spec.id for spec in dl.SPECS] == ["IND13", "IND14", "IND15", "GRP10"]
        for spec in dl.SPECS:
            assert spec.engine == am.ENGINE_OPTIONAL, spec.id
            assert spec.formula_note and spec.question and spec.params, spec.id
            assert am.method_by_id(spec.id) is not None and am.method_by_id(spec.id).name \
                == spec.name, spec.id
        assert [spec.requires for spec in dl.SPECS[:2]] == [("torch", "students>=8")] * 2
        assert dl.SPECS[2].requires == ("torch", "person_edges", "outcome", "groups>=3")
        assert dl.SPECS[3].requires == ("groups>=3",)      # GRP10 前两段不依赖 torch
        import analysis_plots as ap
        for method_id in ("IND13", "IND14", "IND15", "GRP10"):
            assert method_id in ap.PLOTTERS and method_id in ap.TITLES, method_id


class TestDlNetworks:
    """torch 件：形状、因果性、多头宽度、遮码专符——都是能拿笔算的不变量。"""

    @needs_torch
    def test_next_state_net_reserves_pad_row(self):
        net = dl._networks()["NextStateNet"](3, 4, 6, 1, kind="gru")
        assert net.embedding.num_embeddings == 4 and net.embedding.padding_idx == 3
        out = net(dl._tensors([[3, 0, 1], [3, 3, 2]]))
        assert out.shape == (2, 3) and bool(dl.torch.isfinite(out).all())
        assert dl._tensors([[3, 0, 1]]).dtype == dl.torch.long

    @needs_torch
    def test_tcn_is_causal_and_keeps_length(self):
        net = dl._networks()["TcnNet"](3, 4, 4, stages=3, kernel_size=2)
        net.eval()
        assert net.receptive_field == dl.tcn_receptive_field(3, 2) == 15
        x = dl._tensors([[0, 1, 2, 0, 1, 2, 0, 1]])
        later = dl._tensors([[0, 1, 2, 0, 1, 2, 0, 2]])
        with dl.torch.no_grad():
            base = net.body(net.embedding(x).transpose(1, 2))[0].numpy().T
            moved = net.body(net.embedding(later).transpose(1, 2))[0].numpy().T
        assert base.shape == moved.shape == (8, 4)          # chomp 正好吃掉右侧 pad
        assert np.allclose(base[:7], moved[:7], atol=1e-6)  # 改末尾不影响早期位置
        assert not np.allclose(base[7], moved[7], atol=1e-6)

    @needs_torch
    def test_gat_heads_width_and_isolated_node(self):
        nets = dl._networks()
        layer = nets["GatLayer"](5, 3, heads=4)
        layer.eval()
        nodes = dl.torch.as_tensor(np.ones((4, 5), dtype=np.float32))
        edges = dl.torch.as_tensor(np.asarray([[0, 1, 2], [1, 1, 0]], dtype=np.int64))
        with dl.torch.no_grad():
            out, weight = layer(nodes, edges)
        assert out.shape == (4, 12) and weight.shape == (3, 4)
        # 按 dst 归一：dst=1 有两条入边（两半），dst=0 只有一条（独占 1）
        totals = np.zeros((4, 4))
        for row, dst in enumerate([1, 1, 0]):
            totals[dst] += weight.numpy()[row]
        assert np.allclose(totals, np.asarray([[1.0] * 4, [1.0] * 4, [0.0] * 4, [0.0] * 4]))
        assert float(out[2].abs().sum()) == 0.0 and float(out[3].abs().sum()) == 0.0
        with dl.torch.no_grad():
            empty, empty_weight = layer(nodes, dl.torch.zeros(2, 0, dtype=dl.torch.long))
        assert empty.shape == (4, 12) and float(empty.abs().sum()) == 0.0
        assert empty_weight.shape == (0, 4)
        model = nets["TemporalGAT"](5, 4, heads=3, layers=2, classes=2)
        model.eval()
        buckets = [edges, dl.torch.zeros(2, 0, dtype=dl.torch.long)]
        with dl.torch.no_grad():
            logits, readout, weights = model(nodes, buckets, [[0, 1], [2], [3]])
        assert logits.shape == (3, 2) and readout.shape == (3, 4)
        assert len(weights) == 4 and weights[0].shape == (3, 3)

    @needs_torch
    def test_deepsets_pooling_is_permutation_invariant(self):
        net = dl._networks()["DeepSetsNet"](3, 6, 4, pool="mean", classes=2)
        net.eval()
        values = np.random.default_rng(3).normal(size=(1, 3, 3)).astype("float32")
        features = dl.torch.as_tensor(values)
        mask = dl.torch.ones(1, 3)
        order = dl.torch.as_tensor(np.asarray([2, 0, 1], dtype=np.int64))
        with dl.torch.no_grad():
            _logits, first = net(features, mask)
            _l2, shuffled = net(features.index_select(1, order), mask.index_select(1, order))
            # mask 得真的吞掉 pad：多堆一行垃圾但 mask=0，池化结果不能变
            extra = dl.torch.cat([features, features[:, :1] * 99.0], dim=1)
            _l3, padded = net(extra, dl.torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
        assert dl.torch.allclose(first, shuffled, atol=1e-6)
        assert dl.torch.allclose(first, padded, atol=1e-6)
        assert first.shape == (1, 4)

    @needs_torch
    def test_masked_transformer_special_tokens(self):
        net = dl._networks()["MaskedTransformer"](4, 8, 2, 1, max_len=16, mode="masked")
        net.eval()
        assert net.special == 4 and net.pool_token == 5 and net.embedding.num_embeddings == 6
        x = dl._tensors([[0, 1, net.special, 2], [1, 2, 3, net.special]])
        with dl.torch.no_grad():
            assert net.embed(x).shape == (2, 8)
            assert net.logits(x).shape == (2, 4, 4)
            # 整行都是 MASK → 无有效位可池化，不报错也不出 NaN
            assert np.allclose(net.embed(dl._tensors([[net.special, net.special]])).numpy(),
                               0.0, atol=1e-6)
        zero = dl.torch.as_tensor(np.full((2, 4), -1, dtype=np.int64))
        assert float(net.loss(x, zero)) == 0.0              # 没遮位 → 无目标，不除 0
        ae = dl._networks()["MaskedTransformer"](4, 8, 2, 1, max_len=16, mode="ae")
        ae.eval()
        with dl.torch.no_grad():
            assert ae.decode(x).shape == (2, 4, 4)

    @needs_torch
    def test_weighted_ce_matches_sklearn_balanced_semantics(self):
        """逐样本权重加权均值必须等于 sklearn 的 class_weight="balanced"。"""
        logits = dl.torch.as_tensor(np.asarray([[2.0, 0.0], [0.5, 1.5], [-1.0, 0.2],
                                               [0.0, 0.0], [1.0, -0.5]], dtype=np.float32))
        targets = dl.torch.as_tensor(np.asarray([0, 1, 0, 1, 0], dtype=np.int64))
        labels = ["a", "b", "a", "b", "a"]                    # 与 targets 同序
        per_sample = ml.class_balanced_weights(labels)        # n/(k·n_c) 逐样本
        assert per_sample.shape == (5,)
        assert np.allclose(per_sample, [5 / (2 * 3), 5 / (2 * 2), 5 / (2 * 3),
                                        5 / (2 * 2), 5 / (2 * 3)])
        point = dl.nn.functional.cross_entropy(logits, targets, reduction="none").numpy()
        expected = float((point * per_sample).sum() / per_sample.sum())
        got = float(dl._weighted_ce(logits, targets,
                                   dl.torch.as_tensor(per_sample.astype("float32"))))
        assert got == pytest.approx(expected, abs=1e-5)
        by_class = dl.torch.as_tensor(np.asarray([5 / (2 * 3), 5 / (2 * 2)], dtype=np.float32))
        assert float(dl.nn.functional.cross_entropy(logits, targets, weight=by_class)) \
            == pytest.approx(expected, abs=1e-5)
        assert float(dl._weighted_ce(logits, targets, None)) == pytest.approx(
            float(dl.nn.functional.cross_entropy(logits, targets)), abs=1e-6)
        # 反向坐实注释里的坑：把「逐样本」长度当逐类向量传给 torch 会直接报错
        with pytest.raises(RuntimeError):
            dl.nn.functional.cross_entropy(logits, targets,
                                           weight=dl.torch.as_tensor(
                                               per_sample.astype("float32")))

    @needs_torch
    def test_train_loop_honours_cancellation_flag(self):
        class Fake:
            cancelled = True
            progress = None

        net = dl._networks()["NextStateNet"](2, 4, 4, 1, kind="gru")
        batches = dl._batches([[0, 1]], [[1]], 2, 7)
        history, stopped = dl._train(net, batches, 3, 0.01, Fake(), "测试")
        assert stopped and history == []

    @needs_torch
    def test_ind13_end_to_end_micro(self):
        ctx = _dl_ctx()
        params = _dl_params(0, ctx, archs=["gru"], window=3, epochs=2, folds=2,
                            batch_size=32, max_samples=200, embed_dim=8, hidden_size=8)
        out = dl.next_state_analysis(ctx, params)
        assert not out.blocked, out.blocked_reason
        metrics = out.tables["metrics"]
        assert "GRU" in set(metrics["模型"])
        assert any("地板" in text for text in metrics["备注"])
        for column in ("accuracy", "balanced_acc", "Macro-F1", "log_loss"):
            values = metrics[column].astype(float)
            if column != "log_loss":
                assert bool(((values >= 0.0) & (values <= 1.0)).all()), column
        assert (metrics["n_test"] > 0).all()
        assert out.tables["model_selection"]["模型"].nunique() >= 2
        assert any(item.code == "IND13.vs_baseline" for item in out.qa)
        assert out.meta["archs"] == ["GRU"] and float(out.meta["floor_macro_f1"]) >= 0.0
        figures = __import__("analysis_plots").render("IND13", out, ctx, params, strict=True)
        assert figures and all(fig.axes for fig in figures.values())


