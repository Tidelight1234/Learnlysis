# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""IND13–IND15 / GRP10：深度学习与组级预测（计划 §2 ``analysis_dl`` / §3）。

**torch 是可选后端**：本文件在未装 torch 的环境里也必须能 import —— 工作台要靠 ``SPECS``
把方法列出来并写明「未安装 torch」，所以网络类全部放在惰性工厂 ``_networks()`` 里，
模块级不出现任何 ``nn.Module`` 子类。

四条硬口径：

1. **切分沿用 IND12**：next-state 任务的同一学生（组级任务的同一组）只能整个儿落在
   某一边，``analysis_ml.grouped_split / leave_one_group_out`` 是唯一实现；
2. **DL 必须和地板一起报**：多数类、一阶 Markov、窗口特征 RF（IND12 口径）与网络在同一
   测试集上并列，谁赢一目了然 —— 小样本上「深度模型没赢」本身就是结论；
3. **可复现 + 可中断**：``torch.manual_seed(seed)`` 在每次训练前调用，epoch 回调走
   ``ctx.progress``，每轮开始处检查 ``ctx.cancelled``；
4. **不硬扛**：词表太小、样本太少、缺 outcome、缺事件边都直接给阻断说明，不静默降级。
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

import analysis_core as ac
import analysis_features as af
import analysis_methods as am
import analysis_ml as ml
import analysis_stats_group as gp
import analysis_stats_network as nt
import analysis_stats_sequence as sq
from statistic_visualization_data import QAItem, fmt_round

try:                                        # 可选后端：缺席时 TORCH=False，方法卡只写原因
    import torch
    from torch import nn
    TORCH = True
except ImportError:                         # pragma: no cover
    torch = None
    nn = None
    TORCH = False

MODE_INDIVIDUAL = am.MODE_INDIVIDUAL
MODE_GROUP = am.MODE_GROUP
GROUP_MODEL = am.GROUP_MODEL
GROUP_PREDICT = am.GROUP_PREDICT
MIN_STUDENTS_DL = 8                          # 计划 §3：DL 方法的独立学生下限
_NETWORK_CACHE = None


def _start():
    return time.perf_counter()


def _ms(marker):
    return int((time.perf_counter() - marker) * 1000)


def blocked(method_id, name, note):
    """阻断结果：本地构造 MethodSpec，不反向查注册表（声明在本文件末尾）。"""
    return am.MethodSpec(id=method_id, name=name, group="", question="") \
        .blocked_result(None, note)


def tidy_tables(tables):
    """空表一律不落结果：工作台不给用户一页空表格，没跑成的分支宁可不出现。"""
    return {name: frame for name, frame in (tables or {}).items()
            if frame is not None and len(getattr(frame, "index", []))}


def torch_missing(method_id, name):
    """没装 torch 时的统一说明：给命令，不含糊。"""
    return blocked(method_id, name,
                   "未安装 torch：pip install torch --index-url "
                   "https://download.pytorch.org/whl/cpu（索引不可达时直接 pip install torch；"
                   "详见 requirements-optional.txt）")


def log_progress(ctx, *messages):
    """训练回调：工作台把 ``ctx.progress`` 接到「训练日志」卡，脚本里为 None 就静默。"""
    if ctx is not None and callable(getattr(ctx, "progress", None)):
        ctx.progress(*messages)


def cancelled(ctx):
    return bool(ctx is not None and getattr(ctx, "cancelled", False))


def lr_of(params, default):
    """学习率读回：参数卡上是一个文本框（dspin 只有两位小数，表达不了 1e-3）。

    空串、非数字、负数一律退回该方法的默认值，不让一个乱输入的字符弄挂训练。
    """
    try:
        value = float(str(params.get("lr") or "").strip())
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0.0 else float(default)


# ------------------------------------------------------------------ 词表与样本（纯 numpy，可单测）

def state_vocab(sequences):
    """码表（按首次出现顺序）：IND13/IND14 的 embedding 与 softmax 都按这个下标。"""
    out = []
    for sequence in sequences:
        for code in sequence.codes:
            if code and code not in out:
                out.append(code)
    return out


def index_codes(codes, vocab):
    """码流 → 下标序列：词表外的码给 -1（调用方自行丢弃）。"""
    lookup = {code: position for position, code in enumerate(vocab)}
    return [lookup.get(code, -1) for code in codes]


def window_samples(codes, window=8, stride=1, min_context=2):
    """一条码流 → ``[(上下文位置元组, 目标位置)]``：目标是第 i 步，上下文是它前 ≤window 步。

    上下文永远不含目标本身（那是答案），所以 ``min_context`` 之前的位置不出样。
    """
    out = []
    start_position = max(int(min_context), 1)
    for position in range(start_position, len(codes), max(int(stride), 1)):
        out.append((tuple(range(max(0, position - int(window)), position)), position))
    return out


def build_next_state_dataset(sequences, vocab, window=8, stride=1, min_context=2):
    """多条序列 → 一张样本表：列 = 上下文下标（定长，左补 -1）、target、unit、seq。

    ``unit`` 是切分单元（student_uid），``seq`` 用来把同一序列的相邻样本认出来。
    """
    rows = []
    for position, sequence in enumerate(sequences):
        indexed = index_codes(list(sequence.codes), vocab)
        for context, target in window_samples(indexed, window, stride, min_context):
            if indexed[target] < 0:
                continue                                  # 目标码不在词表：没有可比的答案
            picked = [indexed[item] for item in context if indexed[item] >= 0]
            if len(picked) < max(1, int(min_context)):
                continue
            padded = [-1] * max(int(window) - len(picked), 0) + picked
            rows.append({"unit": str(sequence.uid), "seq": position,
                         "target": int(indexed[target]), "last": int(picked[-1]),
                         "context": padded})
    return pd.DataFrame(rows)


def split_by_unit(frame, folds, seed, mode="kfold", ratio=0.25):
    """按 ``unit`` 列整块切分：返回 ``[(train_index, test_index)]``（模式同 IND12）。"""
    keys = [str(value) for value in frame["unit"]]
    if mode == "loyo":
        return ml.leave_one_group_out(keys)
    if mode == "holdout":
        return ml.holdout_split(keys, ratio, seed)
    return ml.grouped_split(keys, folds, seed)


def markov_predictor(pairs, vocab_size):
    """一阶 Markov：从 (末码 → 目标码) 计数建 P(next|last)，+1 拉普拉斯平滑。"""
    counts = np.ones((int(vocab_size), int(vocab_size)), dtype=float)
    for last, target in pairs:
        counts[int(last), int(target)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def marginal_distribution(targets, vocab_size):
    """目标码的边际分布（多数类基线的概率版）。"""
    out = np.ones(int(vocab_size), dtype=float)
    for target in targets:
        out[int(target)] += 1.0
    return out / out.sum()


def context_feature_matrix(frame, vocab_size):
    """窗口特征 RF（IND12 口径的「特征型基线」）：上下文码占比 + 末码一热。"""
    matrix = np.zeros((len(frame), int(vocab_size) * 2), dtype=float)
    for row, record in enumerate(frame.itertuples(index=False)):
        counts = np.zeros(int(vocab_size), dtype=float)
        for code in record.context:
            if code >= 0:
                counts[int(code)] += 1.0
        live = counts.sum()
        matrix[row, :int(vocab_size)] = counts / live if live else 0.0
        matrix[row, int(vocab_size) + int(record.last)] = 1.0
    return matrix


def two_dimension_projection(matrix, seed=0):
    """PCA 到两维（numpy SVD，不引 sklearn 之外的东西）：返回 (坐标, 解释率列表)。"""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or not data.size:
        return np.zeros((len(data), 2), dtype=float), [0.0, 0.0]
    data = data - data.mean(axis=0, keepdims=True)
    rows = min(data.shape[0] - 1, data.shape[1])
    if rows < 1:
        return np.zeros((data.shape[0], 2), dtype=float), [1.0, 0.0]
    _u, singular, vh = np.linalg.svd(data, full_matrices=False)
    total = float((singular ** 2).sum())
    components = min(2, vh.shape[0])
    coords = np.zeros((data.shape[0], 2), dtype=float)
    coords[:, :components] = data @ vh[:components].T
    explained = [float((singular[index] ** 2) / total) if total > 0 else 0.0
                 for index in range(vh.shape[0])][:components]
    while len(explained) < 2:
        explained.append(0.0)
    return coords, explained


def masked_training_pairs(codes, vocab_size, mask_id, ratio=0.15, seed=20240517):
    """SSL 预训练样本：按 ratio 随机遮码，目标只在被遮位置算（未遮位置给 -1 = ignore）。

    至少保证一个位置被遮：全码流只有一个位置时不产样。
    """
    values = [int(code) for code in codes if 0 <= int(code) < int(vocab_size)]
    if len(values) < 2:
        return np.zeros((0, len(values)), dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    picks = rng.random(len(values)) < float(ratio)
    if not picks.any():
        picks[int(rng.integers(0, len(values)))] = True
    x = np.array(values, dtype=np.int64)
    y = np.full(len(values), -1, dtype=np.int64)
    y[picks] = x[picks]
    x = np.where(picks, int(mask_id), x)
    return x.reshape(1, -1), y


# ------------------------------------------------------------------ 网络类（惰性构造，torch 缺席不炸）

def _networks():
    """返回本模块的 ``nn`` 网络类字典；torch 不在就给 None。"""
    global _NETWORK_CACHE
    if not TORCH:
        return None
    if _NETWORK_CACHE:
        return _NETWORK_CACHE

    class _Chomp1d(nn.Module):
        """因果卷积裁剪：把右侧未来padding 掉，保证 t 只看 ≤ t。"""

        def __init__(self, width):
            super().__init__()
            self.width = int(width)

        def forward(self, value):                           # noqa: D102
            if self.width <= 0:
                return value
            return value[:, :, :-self.width] if value.size(2) > self.width else value

    class _TcnBlock(nn.Module):
        """单个膨胀因果残差块：两层 weight-norm 空洞卷积 + 跳跃投影。

        因果性靠「两侧补 pad、右侧再切掉 pad」实现，所以 Conv1d 必须真的带上 dilation——
        漏了它，感受野会按 pad 长出多余长度（第二层起残差就对不上），
        报给结果表的 receptive_field 也会变成假数字。
        """

        def __init__(self, channels, kernel_size, dilation, dropout):
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.conv1 = nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                                        padding=pad, dilation=dilation))
            self.conv2 = nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                                        padding=pad, dilation=dilation))
            self.chomp1, self.chomp2 = _Chomp1d(pad), _Chomp1d(pad)
            self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
            self.dropout = nn.Dropout(float(dropout))
            self.projection = nn.Identity() if kernel_size == 1 else \
                nn.Conv1d(channels, channels, 1)

        def forward(self, value):                           # noqa: D102
            hidden = self.chomp1(self.conv1(value))
            hidden = self.norm1(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = self.dropout(torch.relu(hidden))
            hidden = self.chomp2(self.conv2(hidden))
            hidden = self.norm2(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = self.dropout(hidden)
            return torch.relu(hidden + self.projection(value))

    class NextStateNet(nn.Module):
        """embedding → LSTM / GRU 栈 → 末位 softmax：next-code 预测的主力网络。"""

        def __init__(self, vocab_size, embed_dim, hidden_size, layers, kind="gru",
                     dropout=0.1):
            super().__init__()
            self.kind = str(kind)
            self.embedding = nn.Embedding(int(vocab_size) + 1, int(embed_dim),
                                          padding_idx=int(vocab_size))
            recurrent = nn.LSTM if self.kind == "lstm" else nn.GRU
            self.recurrent = recurrent(int(embed_dim), int(hidden_size),
                                       num_layers=max(int(layers), 1), batch_first=True,
                                       dropout=0.0 if int(layers) < 2 else float(dropout))
            self.head = nn.Sequential(nn.Dropout(float(dropout)),
                                      nn.Linear(int(hidden_size), int(vocab_size)))

        def forward(self, x):                               # noqa: D102
            hidden = self.embedding(x)
            out, _state = self.recurrent(hidden)
            return self.head(out[:, -1, :])

        def loss(self, x, y):                               # noqa: D102
            return nn.functional.cross_entropy(self.forward(x), y)

    class TcnNet(nn.Module):
        """TCN：堆叠膨胀因果卷积，感受野 = 1 + 2·(k−1)·Σ2^i，报给结果表核对。"""

        def __init__(self, vocab_size, embed_dim, hidden_size, stages, kernel_size=2,
                     dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(int(vocab_size) + 1, int(embed_dim),
                                          padding_idx=int(vocab_size))
            blocks, dilation = [], 1
            for _stage in range(max(int(stages), 1)):
                blocks.append(_TcnBlock(int(embed_dim), int(kernel_size), dilation, dropout))
                dilation *= 2
            self.body = nn.Sequential(*blocks)
            self.head = nn.Linear(int(embed_dim), int(vocab_size))
            self.stages, self.kernel_size = max(int(stages), 1), int(kernel_size)
            self.receptive_field = tcn_receptive_field(self.stages, self.kernel_size)

        def forward(self, x):                               # noqa: D102
            hidden = self.embedding(x).transpose(1, 2)
            hidden = self.body(hidden).transpose(1, 2)
            return self.head(hidden[:, -1, :])

        def loss(self, x, y):                               # noqa: D102
            return nn.functional.cross_entropy(self.forward(x), y)

    class MaskedTransformer(nn.Module):
        """双向 self-attention：masked-code 预训练 → 池化 embedding；可选自编码版。"""

        def __init__(self, vocab_size, embed_dim, heads, layers, max_len=512,
                     dropout=0.1, mode="masked"):
            super().__init__()
            self.mode = str(mode)
            self.special = int(vocab_size)                   # MASK
            self.pool_token = int(vocab_size) + 1            # 自编码版用的 [CLS]
            self.embedding = nn.Embedding(int(vocab_size) + 2, int(embed_dim),
                                          padding_idx=None)
            self.position = nn.Embedding(int(max_len), int(embed_dim))
            layer = nn.TransformerEncoderLayer(d_model=int(embed_dim), nhead=int(heads),
                                               dim_feedforward=int(embed_dim) * 4,
                                               dropout=float(dropout), batch_first=True,
                                               activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(layers), 1))
            self.head = nn.Linear(int(embed_dim), int(vocab_size))
            self.decoder = nn.GRU(int(embed_dim), int(embed_dim), batch_first=True) \
                if self.mode == "ae" else None

        def _encode(self, x, mask=None):                      # noqa: D102
            positions = torch.arange(x.size(1), device=x.device).clamp(
                max=self.position.num_embeddings - 1)
            hidden = self.embedding(x) + self.position(positions)[None, :, :]
            return self.encoder(hidden, mask=mask) if mask is not None else self.encoder(hidden)

        def forward(self, x, causal=False):                   # noqa: D102
            """causal=True 时用下三角掩码（变体：自回归预训练），默认双向（masked-code 预训练）。"""
            mask = torch.triu(torch.ones(x.size(1), x.size(1), dtype=torch.bool,
                                         device=x.device), diagonal=1) if causal else None
            return self._encode(x, mask)

        def logits(self, x):                                  # noqa: D102
            return self.head(self._encode(x))

        def decode(self, x):                                 # noqa: D102
            """自编码版的重建 logits：embed → GRU 解到原长 → 逐位分类（训练/评估同一路径）。"""
            z = self.embed(x)
            length = x.size(1)
            seed = z.unsqueeze(1).repeat(1, length, 1)
            decoded, _state = self.decoder(seed)
            return self.head(decoded)

        def loss(self, x, y):                                 # noqa: D102
            if self.mode == "ae":
                # 自编码版：重建目标 = 调用给的 y（pad 位 -1）
                return nn.functional.cross_entropy(self.decode(x).transpose(1, 2), y,
                                                   ignore_index=-1)
            keep = y >= 0
            if not bool(keep.any()):
                return torch.zeros((), dtype=torch.float32)
            return nn.functional.cross_entropy(self.logits(x)[keep], y[keep])

        def embed(self, x):                                   # noqa: D102
            """池化 embedding：真实码位（非 MASK/特殊符）的 encoder 输出均值。"""
            hidden = self._encode(x)
            valid = (x >= 0) & (x < self.special)
            weights = valid.to(hidden.dtype).unsqueeze(-1)
            summed = (hidden * weights).sum(dim=1)
            counts = weights.sum(dim=1).clamp(min=1.0)
            return summed / counts

    class FeedForward(nn.Module):
        """DeepSets 的 φ / ρ 与 GAT 的头共用的多层感知机。"""

        def __init__(self, sizes, dropout=0.1):
            super().__init__()
            layers = []
            for width, nxt in zip(list(sizes)[:-1], list(sizes)[1:]):
                layers += [nn.Linear(int(width), int(nxt)), nn.GELU(),
                           nn.Dropout(float(dropout))]
            self.body = nn.Sequential(*layers[:-2]) if len(layers) > 2 else \
                nn.Sequential(*layers[:2])

        def forward(self, x):                                # noqa: D102
            return self.body(x)

    class DeepSetsNet(nn.Module):
        """任意成员数 → 定长组 embedding：φ(成员) → 池化 → ρ；pool 可选 mean/sum/attention。"""

        def __init__(self, in_dim, width, out_dim, pool="mean", classes=0, dropout=0.1):
            super().__init__()
            self.pool = str(pool)
            self.phi = FeedForward([int(in_dim), int(width), int(width)], dropout)
            self.rho = FeedForward([int(width), int(width), int(out_dim)], dropout)
            self.attention = nn.Linear(int(width), 1) if self.pool == "attention" else None
            self.head = nn.Linear(int(out_dim), int(classes)) if int(classes) > 1 else None

        def forward(self, features, mask=None):               # noqa: D102
            hidden = self.phi(features)
            if self.attention is not None:
                scores = self.attention(hidden).squeeze(-1)
                if mask is not None:
                    scores = scores.masked_fill(~mask.bool(), float("-inf"))
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1)
            else:
                if mask is not None:
                    hidden = hidden * mask.to(hidden.dtype).unsqueeze(-1)
                    counts = mask.to(hidden.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
                    pooled = hidden.sum(dim=1) / counts if self.pool != "sum" \
                        else hidden.sum(dim=1)
                else:
                    pooled = hidden.mean(dim=1) if self.pool != "sum" else hidden.sum(dim=1)
            out = self.rho(pooled)
            return (self.head(out), out) if self.head is not None else (None, out)

        def loss(self, features, y, mask=None):                # noqa: D102
            logits, _z = self.forward(features, mask)
            if logits is None:
                return torch.zeros((), dtype=torch.float32)
            return nn.functional.cross_entropy(logits, y)

    class GatLayer(nn.Module):
        """单层图注意力（稀疏边表版）：e_ij = LeakyReLU(a·[Wh_i ‖ Wh_j])，按 dst softmax。

        ``out_dim`` 是**每头**宽度，整层输出 = out_dim × heads。
        """

        def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
            super().__init__()
            self.heads = int(heads)
            self.out_dim = int(out_dim)
            self.linear = nn.Linear(int(in_dim), int(out_dim) * self.heads, bias=False)
            self.source = nn.Parameter(torch.empty(self.heads, int(out_dim)))
            self.target = nn.Parameter(torch.empty(self.heads, int(out_dim)))
            nn.init.xavier_uniform_(self.source)
            nn.init.xavier_uniform_(self.target)
            self.dropout = nn.Dropout(float(dropout))

        def forward(self, nodes, edge_index):                  # noqa: D102
            if edge_index is None or edge_index.numel() == 0:
                return nodes.new_zeros((nodes.size(0), self.out_dim * self.heads)), \
                    nodes.new_zeros((0, self.heads))
            src = edge_index[0].long()
            dst = edge_index[1].long()
            value = self.linear(nodes).view(-1, self.heads, self.out_dim)
            left = value.index_select(0, src)                  # (E, H, F)
            right = value.index_select(0, dst)
            score = F_leaky_relu((left * self.source[None]).sum(-1)
                                 + (right * self.target[None]).sum(-1))   # (E, H)
            weight = _softmax_by_index(score, dst, nodes.size(0))
            weight = self.dropout(weight)
            message = value.new_zeros((nodes.size(0), self.heads, self.out_dim))
            message = message.index_add_(0, dst, value.index_select(0, src) * weight[:, :, None])
            return message.reshape(message.size(0), -1), weight

    class TemporalGAT(nn.Module):
        """时间片上的 GAT：每个时刻桶做一轮图注意力，再用 GRUCell 沿时间更新节点状态。

        节点集对所有时间桶固定（= 全部 ``组__成员``），空桶给全零消息，所以节点数不会跳变。
        """

        def __init__(self, in_dim, hidden, heads=4, layers=1, classes=0, dropout=0.1):
            super().__init__()
            # hidden 是「每个头」的宽度：一层 GAT 输出 hidden × heads，下一层的输入就是它
            self.hidden, self.heads = int(hidden), max(int(heads), 1)
            width = self.hidden * self.heads
            ins = [int(in_dim)] + [width] * max(int(layers), 1)
            self.layers = nn.ModuleList([GatLayer(ins[index], self.hidden, self.heads, dropout)
                                         for index in range(max(int(layers), 1))])
            # 逐时刻先把多头均值掉（→ hidden 宽）再进 GRU：状态只负责“上一条”，不参与层间传导
            self.cell = nn.GRUCell(self.hidden, self.hidden)
            self.readout = nn.Linear(self.hidden, int(classes)) if int(classes) > 1 else None

        def embed(self, nodes, buckets):
            """``buckets`` = [edge_index]（同一节点集的逐时刻边表）。

            返回 (逐节点末态, 逐桶注意力)；注意力张量形状 (E, H)，调用方按边求均值上报。
            """
            state = torch.zeros(nodes.size(0), self.hidden, dtype=torch.float32,
                                device=nodes.device)
            weights = []
            for edge_index in buckets:
                hidden = nodes
                for layer in self.layers:
                    hidden, attention = layer(hidden, edge_index)
                    hidden = torch.relu(hidden)
                    weights.append(attention)
                pooled = hidden.view(hidden.size(0), self.heads, self.hidden).mean(dim=1)
                state = self.cell(pooled, state)
            return state, weights

        def forward(self, nodes, buckets, groups):
            """``groups`` = [节点下标列表]（每组一行）：readout = 组内节点末态均值。"""
            state, weights = self.embed(nodes, buckets)
            rows = [state[torch.as_tensor(members, dtype=torch.long,
                                          device=state.device)].mean(dim=0)
                    for members in groups if len(members)]
            readout = torch.stack(rows) if rows else \
                torch.zeros(0, self.hidden, device=state.device)
            logits = self.readout(readout) if self.readout is not None else None
            return logits, readout, weights

    networks = {
        "NextStateNet": NextStateNet, "TcnNet": TcnNet, "MaskedTransformer": MaskedTransformer,
        "DeepSetsNet": DeepSetsNet, "TemporalGAT": TemporalGAT, "GatLayer": GatLayer,
        "FeedForward": FeedForward,
    }
    _NETWORK_CACHE = networks
    return networks


def F_leaky_relu(value):
    """LeakyReLU(0.2)：GAT 打分用的固定斜率，写死以免参数表里再长一个旋钮。"""
    return nn.functional.leaky_relu(value, 0.2)


def _softmax_by_index(score, index, num_rows):
    """按 dst 分组的 softmax（scatter 版，不依赖 scatter_add 的高版本语义）。"""
    out = torch.zeros_like(score)
    maximum = score.new_full((int(num_rows), score.size(1)), float("-inf"))
    maximum = maximum.index_reduce(0, index, score, "amax", include_self=True)
    shifted = score - maximum.index_select(0, index)
    exponent = torch.exp(shifted)
    totals = exponent.new_zeros((int(num_rows), exponent.size(1)))
    totals = totals.index_add_(0, index, exponent)
    out = exponent / totals.index_select(0, index).clamp(min=1e-12)
    return out


def set_seed(seed):
    """固定随机种子：numpy 一定设，torch 在场时一起设。"""
    seed = int(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    if TORCH:
        torch.manual_seed(seed)
    return seed


# ------------------------------------------------------------------ 训练器（epoch 回调 + 取消）

def _tensors(rows):
    return torch.as_tensor(np.asarray(rows, dtype=np.int64), dtype=torch.long)


def _batches(x, y, batch_size, seed, shuffle=True):
    """定长整型批：训练扰乱用固定 rng，测试集不扰（预测行顺序稳定可追）。"""
    size = max(int(batch_size), 2)
    if shuffle and len(x) > size:
        order = np.random.default_rng(int(seed)).permutation(len(x)).tolist()
    else:
        order = list(range(len(x)))
    out = []
    for start in range(0, len(order), size):
        picked = order[start:start + size]
        out.append((_tensors([x[row] for row in picked]), _tensors([y[row] for row in picked])))
    return out


def _row_count(batches):
    return int(sum(int(x.size(0)) for x, _y in batches))


def _weighted_ce(logits, targets, weights):
    """类不均衡 = **逐样本** 权重加权均值（sklearn ``class_weight="balanced"`` 口径）。

    ``ml.class_balanced_weights`` 返的是逐样本权重（IND12 就靠它传 ``sample_weight``），
    而 ``cross_entropy(weight=)`` 要的是逐类向量，两者不能直接接：组数少于类数时
     sklearn 会递上一个长度 = 样本数的张量，torch 直接报「weight tensor of shape [n]」。
    """
    if weights is None:
        return nn.functional.cross_entropy(logits, targets)
    pointwise = nn.functional.cross_entropy(logits, targets, reduction="none")
    return (pointwise * weights).sum() / weights.sum().clamp(min=1e-12)


def _train(model, batches, epochs, lr, ctx, stage, val=None):
    """Adam + 梯度裁剪 + 逐 epoch 回调：返回 ``(history 行, 是否被取消)``。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    history, stopped = [], False
    epochs = max(int(epochs), 1)
    for epoch in range(1, epochs + 1):
        if cancelled(ctx):
            stopped = True
            break
        model.train()
        total, seen = 0.0, 0
        for x_batch, y_batch in batches:
            loss = model.loss(x_batch, y_batch)
            value = float(loss.item())
            if not math.isfinite(value):                    # NaN 批直接跳过，不污染权重
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += value * int(x_batch.size(0))
            seen += int(x_batch.size(0))
        row = {"阶段": stage, "epoch": epoch, "train_loss": fmt_round(total / max(seen, 1), 5)}
        if val:
            model.eval()
            with torch.no_grad():
                checked = 0.0
                for x_batch, y_batch in val:
                    item = float(model.loss(x_batch, y_batch).item())
                    checked += item * int(x_batch.size(0))
                row["val_loss"] = fmt_round(checked / max(_row_count(val), 1), 5)
        history.append(row)
        log_progress(ctx, "%s epoch %d/%d：train_loss=%.5f%s"
                     % (stage, epoch, epochs, row["train_loss"],
                        "" if "val_loss" not in row else "  val_loss=%.5f" % row["val_loss"]))
    return history, stopped


def _softmax(matrix):
    values = np.asarray(matrix, dtype=float)
    if not values.size:
        return values.reshape(len(values), 0)
    exponent = np.exp(values - values.max(axis=1, keepdims=True))
    return exponent / np.clip(exponent.sum(axis=1, keepdims=True), 1e-12, None)


def _infer_logits(model, x, batch_size=256):
    """整批前向 → numpy logits（评估用，不进梯度）。"""
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(x), max(int(batch_size), 8)):
            part = _tensors(x[start:start + int(batch_size)])
            rows.append(np.asarray(model(part).detach().cpu().numpy(), dtype=float))
    return np.vstack(rows) if rows else np.zeros((0, 0), dtype=float)


def _dl_gate(method_id, name, ctx, min_students=0):
    """三道通用闸：torch / 取消 / 独立单元数。返回 None 表示放行。"""
    if not TORCH:
        return torch_missing(method_id, name)
    if cancelled(ctx):
        return blocked(method_id, name, "已取消：计算在训练日志卡被中止，未产生任何结果")
    if min_students and ctx is not None and int(ctx.n_students) < int(min_students):
        return blocked(method_id, name,
                       "独立学生只有 %d 名，不足 %d 名：深度模型的折内训练集会被切到不可用"
                       % (int(ctx.n_students), int(min_students)))
    return None


def _wanted_sequences(ctx, params):
    """参数卡选序列（长度闸同其他方法）。"""
    return [item for item in ctx.sequences_for(params)
            if len([code for code in item.codes if code]) >= am.MIN_SEQUENCE_LENGTH]


def _streams(sequences, merge=True):
    """``{流名: [(序列, 码流)]}``：merge=True 把维度键前缀进码名（D1:A）合成一个词表。

    合并后不同维度的同名单元不会撞车，而每条码流仍然只属于一个 (学生 × 维度)，
    转移绝不跳维。
    """
    out = {}
    if merge:
        out["合并词表"] = [(item, ["%s:%s" % (item.dimension, code) for code in item.codes
                                   if code])
                           for item in sequences]
        return out
    for item in sequences:
        out.setdefault(str(item.dimension), []).append((item, [code for code in item.codes
                                                              if code]))
    return out


# ------------------------------------------------------------------ IND13 next-state

def _pad_matrix(contexts, pad_id):
    """上下文里的 -1（左侧不足窗口）→ pad 下标，embedding 用 padding_idx 吞掉。"""
    return [[pad_id if int(value) < 0 else int(value) for value in row] for row in contexts]


def tcn_receptive_field(stages, kernel_size=2):
    """TCN 感受野（Bai 等式 (3)）：1 + 2·(k−1)·Σ2^i = 1 + 2·(k−1)·(2^stages−1)。"""
    return 1 + 2 * (int(kernel_size) - 1) * (2 ** max(int(stages), 1) - 1)


def _baseline_predictors(dataset, vocab, train, seed):
    """三条地板在同一测试集上的预测概率矩阵：多数类 / 一阶 Markov / 窗口特征 RF。"""
    size = len(vocab)
    targets = [int(dataset["target"][row]) for row in train]
    marginal = marginal_distribution(targets, size)
    pairs = [(int(dataset["last"][row]), int(dataset["target"][row])) for row in train]
    markov = markov_predictor(pairs, size)
    out = {"多数类（训练边际分布）": np.tile(marginal, (len(dataset), 1)),
           "一阶 Markov（IND02 口径）": markov[[int(dataset["last"][row]) for row
                                                in range(len(dataset))]]}
    try:
        matrix = context_feature_matrix(dataset, size)
        labels = [str(int(dataset["target"][row])) for row in train]
        estimator, engine = ml._make_model("rf", seed)
        if estimator is not None:
            estimator.fit(matrix[train], labels)
            probabilities = np.zeros((len(dataset), size), dtype=float)
            for row, values in enumerate(estimator.predict_proba(matrix)):
                for label, value in zip(estimator.classes_, values):
                    probabilities[row, int(label)] = value
            out["窗口特征 RF（IND12 口径）"] = probabilities
    except Exception:                                       # noqa: BLE001 基线坏了不能拖死主结果
        pass
    return out


def next_state_analysis(ctx, params):
    """IND13：LSTM / GRU / TCN 预测下一个码，与多数类 / 一阶 Markov / 特征 RF 并列报。"""
    marker = _start()
    name = "循环/卷积网络 next-state"
    gate = _dl_gate("IND13", name, ctx, MIN_STUDENTS_DL)
    if gate is not None:
        return gate
    networks = _networks()
    sequences = _wanted_sequences(ctx, params)
    if len(sequences) < 2:
        return blocked("IND13", name, "可用序列不足 2 条（需长度 ≥ %d 步）" % am.MIN_SEQUENCE_LENGTH)
    seed = int(params.get("seed") or ml.DEFAULT_SEED)
    set_seed(seed)
    window = max(int(params.get("window") or 8), 2)
    merge = bool(params.get("merge_vocab", True))
    archs = [item for item in (params.get("archs") or ["gru", "tcn"])] or ["gru"]
    folds_wanted = max(int(params.get("folds") or 3), 2)
    epochs = max(int(params.get("epochs") or 30), 1)
    lr = lr_of(params, 1e-3)
    batch_size = max(int(params.get("batch_size") or 64), 2)
    max_samples = max(int(params.get("max_samples") or 4000), 100)
    metric_rows, prediction_rows, history_rows, data_rows, vocab_rows = [], [], [], [], []
    notes = []
    for stream, items in _streams(sequences, merge).items():
        vocab = state_vocab([item for item, _codes in items])
        if len(vocab) < 2:
            notes.append("流「%s」词表只有 %d 个码，next-state 没有可预测的跳变，跳过"
                         % (stream, len(vocab)))
            continue
        dataset = build_next_state_dataset([item for item, _codes in items], vocab,
                                           window=window, stride=params.get("stride") or 1)
        units = sorted({str(value) for value in dataset["unit"]}) if len(dataset) else []
        folds = split_by_unit(dataset, folds_wanted, seed, str(params.get("cv") or "kfold"),
                              float(params.get("test_size") or 0.25)) \
            if len(dataset) else []
        if not len(dataset) or not folds:
            notes.append("流「%s」只有 %d 个样本 / %d 个单元，切不出折，跳过"
                         % (stream, len(dataset), len(units)))
            continue
        vocab_rows += [{"流": stream, "下标": position, "码": code,
                        "频次": int(sum(1 for _item, codes in items for value in codes
                                        if value == code))}
                       for position, code in enumerate(vocab)]
        pad_id = len(vocab)
        contexts = _pad_matrix(list(dataset["context"]), pad_id)
        targets = [int(value) for value in dataset["target"]]
        if len(dataset) > max_samples:                      # 大表下采样（固定种子，可复现）
            keep = set(np.random.default_rng(seed).choice(
                len(dataset), size=max_samples, replace=False).tolist())
        else:
            keep = set(range(len(dataset)))
        baselines = _baseline_predictors(dataset, vocab, sorted(keep), seed)
        models = {key: networks["NextStateNet"] for key in ("gru", "lstm")}
        for arch in archs:
            for fold, (train, test) in enumerate(folds, start=1):
                train_rows = [row for row in train.tolist() if row in keep]
                test_rows = test.tolist()
                x_train = [contexts[row] for row in train_rows]
                y_train = [targets[row] for row in train_rows]
                x_test = [contexts[row] for row in test_rows]
                y_test = [targets[row] for row in test_rows]
                if len(vocab) < 2 or not x_train or not x_test:
                    continue
                if arch in models:
                    # 词表大小只传真实码数：padding 位由网络自己往 embedding 上加一位
                    model = models[arch](len(vocab), int(params.get("embed_dim") or 32),
                                         int(params.get("hidden_size") or 64),
                                         int(params.get("layers") or 1), kind=arch)
                else:
                    model = networks["TcnNet"](
                        len(vocab), int(params.get("embed_dim") or 32),
                        int(params.get("hidden_size") or 64),
                        int(params.get("stages") or 4),
                        kernel_size=int(params.get("kernel_size") or 2))
                set_seed(seed + fold)
                history, stopped = _train(
                    model, _batches(x_train, y_train, batch_size, seed + fold),
                    epochs, lr, ctx, "%s/%s 第%d折" % (stream, arch, fold),
                    val=_batches(x_test, y_test, batch_size, seed, shuffle=False))
                history_rows += [dict(row, 流=stream, 模型=arch.upper(), 折=fold)
                                 for row in history]
                logits = _infer_logits(model, x_test)
                if stopped or not len(logits):
                    notes.append("%s %s 第 %d 折中途取消，只报已完成的 epoch"
                                 % (stream, arch.upper(), fold))
                    continue
                probabilities = _softmax(logits)
                predicted = [int(value) for value in probabilities.argmax(axis=1)]
                row = ml.metrics([str(value) for value in y_test], [str(value) for value in predicted],
                                 probabilities, sorted({str(value) for value in
                                                        list(y_train) + list(y_test)}))
                row.update({"模型": arch.upper(), "流": stream, "折": fold,
                            "n_train": len(train_rows), "n_test": len(test_rows),
                            "备注": "window=%d、词表=%d" % (window, len(vocab))})
                metric_rows.append(row)
                for position, index in enumerate(test_rows):
                    prediction_rows.append({"流": stream, "模型": arch.upper(), "折": fold,
                                            "样本号": int(index) + 1,
                                            "单元素": str(dataset["unit"][index]),
                                            "末码": vocab[int(dataset["last"][index])],
                                            "真实": vocab[int(targets[index])],
                                            "预测": vocab[int(predicted[position])],
                                            "预测正确": int(predicted[position] == targets[index]),
                                            "真实概率": fmt_round(
                                                float(probabilities[position, targets[index]]), 4)})
        for label, matrix in baselines.items():
            for fold, (train, test) in enumerate(folds, start=1):
                train_rows = [row for row in train.tolist() if row in keep]
                test_rows = test.tolist()
                y_train = [targets[row] for row in train_rows]
                y_test = [targets[row] for row in test_rows]
                part = matrix[test_rows]
                predicted = [int(value) for value in part.argmax(axis=1)]
                row = ml.metrics([str(value) for value in y_test],
                                 [str(value) for value in predicted], part,
                                 sorted({str(value) for value in list(y_train) + list(y_test)}))
                row.update({"模型": label, "流": stream, "折": fold,
                            "n_train": len(train_rows), "n_test": len(test_rows),
                            "备注": "地板（不走深度模型）"})
                metric_rows.append(row)
        data_rows.append({"流": stream, "词表大小": len(vocab), "序列数": len(items),
                          "样本数": int(len(dataset)), "入模型样本": len(keep),
                          "独立单元数": len(units), "折数": len(folds),
                          "窗口": window, "平均有效长度": fmt_round(
                              float(np.mean([len(codes) for _item, codes in items])), 2)})
    if not metric_rows:
        return blocked("IND13", name, "没有任何一个流能完成训练：" +
                       ("；".join(notes) or "检查窗口与词表"))
    frame = pd.DataFrame(metric_rows)
    summary = frame.groupby("模型").agg(折数=("折", "size"),
                                        **{"Macro-F1 均值": ("Macro-F1", "mean"),
                                           "Macro-F1 标准差": ("Macro-F1", lambda values:
                                                               float(np.std(values, ddof=0))),
                                           "accuracy 均值": ("accuracy", "mean"),
                                           "balanced_acc 均值": ("balanced_acc", "mean"),
                                           "log_loss 均值": ("log_loss", "mean")}) \
        .reset_index().sort_values("Macro-F1 均值", ascending=False)
    for column in ("Macro-F1 均值", "Macro-F1 标准差", "accuracy 均值",
                   "balanced_acc 均值", "log_loss 均值"):
        summary[column] = fmt_round(summary[column].astype(float), 4)
    floors = [text for text in summary["模型"] if "地板" in
              str(frame[frame["模型"] == text]["备注"].iloc[0])]
    best_model = str(summary["模型"].iloc[0])
    best_score = float(summary["Macro-F1 均值"].iloc[0])
    floor_score = max([float(summary[summary["模型"] == text]["Macro-F1 均值"].iloc[0])
                       for text in floors], default=0.0)
    qa = [QAItem("info" if best_score > floor_score else "warn", "IND13.vs_baseline",
                 "最佳模型 %s 的 Macro-F1 均值 %.4f，地板（多数类 / 一阶 Markov / 特征 RF）最高 %.4f"
                 "：%s" % (best_model, best_score, floor_score,
                           "深度模型有增量" if best_score > floor_score else
                           "没跑过传统基线，小样本上这不是意外，别拿它当结论"))]
    if int(frame["n_test"].max()) < 200:
        qa.append(QAItem("warn", "IND13.small_test",
                         "单折测试样本最多 %d 条，指标波动大于模型差异" % int(frame["n_test"].max())))
    if cancelled(ctx):
        qa.append(QAItem("warn", "IND13.cancelled", "计算中途取消：只完成 %d 行折指标"
                         % len(frame)))
    tables = {"metrics": frame,
              "model_selection": summary,
              "predictions": pd.DataFrame(prediction_rows),
              "training": pd.DataFrame(history_rows),
              "data": pd.DataFrame(data_rows),
              "vocab": pd.DataFrame(vocab_rows)}
    meta = {"window": window, "archs": [str(item).upper() for item in archs],
            "cv": str(params.get("cv") or "kfold"), "folds": folds_wanted, "seed": seed,
            "epochs": epochs, "merge_vocab": merge, "best_model": best_model,
            "floor_macro_f1": fmt_round(floor_score, 4), "torch": _torch_version(),
            "receptive_field": {str(item).upper(): tcn_receptive_field(
                params.get("stages") or 4, params.get("kernel_size") or 2)
                for item in archs if str(item).lower() == "tcn"}}
    notes += ["词表：%s；上下文取前 %d 步（目标不参与上下文），切分单元 = student_uid，"
              "同一学生不跨集" % ("合并（码名前缀维度键）" if merge else "逐维度独立", window),
              "指标与 IND12 同一实现（analysis_ml.metrics），log-loss 只在本折出现类上算"]
    return am.result("IND13", name, tidy_tables(tables), meta=meta, qa=qa, notes=notes,
                     elapsed_ms=_ms(marker))


def _torch_version():
    return str(getattr(torch, "__version__", "")) if TORCH else ""


# ------------------------------------------------------------------ IND14 Transformer + SSL

MAX_SEQ_LEN = 256                 # 位置编码表上限，也是单条序列的截断长
_EVAL_SEED = 7919                 # 验证/评估遮位专用偏移：与训练遮位错开，不拿同位置自证


def stream_vocab(items):
    """流内码名表（首次出现序）：IND14 的 embedding 下标按它排，码名已由 ``_streams`` 加维前缀。"""
    out = []
    for _sequence, codes in items:
        for code in codes:
            if code and str(code) not in out:
                out.append(str(code))
    return out


def ssl_targets(tokens, mask_id, ratio, rng, mode="masked"):
    """一条码流 → ``(x, y)``：masked 模式按 ratio 遮码（至少遮一个），ae 模式原码重建。

    未遮（与 pad）的位置 y = -1 不计损失，所以 masked-code 目标只在被遮位算。
    """
    values = [int(value) for value in tokens]
    if mode == "ae":
        return list(values), list(values)
    picks = rng.random(len(values)) < float(ratio)
    if len(values) and not picks.any():
        picks[int(rng.integers(0, len(values)))] = True
    y = [value if keep else -1 for value, keep in zip(values, picks.tolist())]
    x = [int(mask_id) if keep else value for value, keep in zip(values, picks.tolist())]
    return x, y


def ssl_padded(records, mask_id, pad_id, ratio, seed, mode="masked", max_len=MAX_SEQ_LEN):
    """一批码流 → 定长 ``(x, y)`` 阵：右补 pad_id、目标补 -1；每条序列的遮位由 seed+序号定。"""
    rows, width = [], 1
    for position, tokens in enumerate(records):
        picked = [int(value) for value in list(tokens)[:int(max_len)]]
        x, y = ssl_targets(picked, mask_id, ratio,
                           np.random.default_rng(int(seed) + position), mode)
        width = max(width, len(x))
        rows.append((x, y))
    x_out = [list(x) + [int(pad_id)] * (width - len(x)) for x, _y in rows]
    y_out = [list(y) + [-1] * (width - len(y)) for _x, y in rows]
    return np.asarray(x_out, dtype=np.int64), np.asarray(y_out, dtype=np.int64)


def token_majority_share(records):
    """地板基准：「永远猜这批评样本里最常见的码」的命中率 —— 衡量 masked 命中率的含金量。"""
    counts = {}
    for tokens in records:
        for value in tokens:
            counts[int(value)] = counts.get(int(value), 0) + 1
    total = int(sum(counts.values()))
    return float(max(counts.values()) / total) if total else 0.0


def _ssl_evaluate(model, token_rows, mask_id, pad_id, seed, ratio, mode, batch_size=64):
    """折外评估：逐序列返 ``(遮码数, 命中数, 平均 NLL, 池化 embedding)``。"""
    x, y = ssl_padded(token_rows, mask_id, pad_id, ratio, seed, mode)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), max(int(batch_size), 4)):
            part_x = _tensors(x[start:start + int(batch_size)].tolist())
            truth = y[start:start + int(batch_size)]
            logits = model.decode(part_x) if mode == "ae" else model.logits(part_x)
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            pred = np.asarray(logits.argmax(dim=-1).cpu().numpy())
            vectors = np.asarray(model.embed(part_x).cpu().numpy(), dtype=float)
            for offset in range(len(truth)):
                keep = np.flatnonzero(truth[offset] >= 0)
                hit = int(sum(1 for index in keep
                              if int(pred[offset, index]) == int(truth[offset, index])))
                nll = float(np.mean([-float(log_probs[offset, index, int(truth[offset, index])])
                                     for index in keep])) if len(keep) else 0.0
                out.append({"masked": int(len(keep)), "hit": hit, "nll": nll,
                            "embedding": vectors[offset]})
    return out


def _label_map(ctx, params):
    """拿 IND12 的标签口径（outcome 优先，否则 IND07 伪标签）→ ``{(uid, dimension): 标签}``。"""
    frame, _columns, source, notes = ml.design_matrix(ctx, params)
    if not len(frame):
        return {}, "", notes
    mapping = {(str(row.student_uid), str(row.dimension)): str(row.label)
               for row in frame.itertuples(index=False)}
    return mapping, source, notes


def _embedding_probe(frame, seed, params):
    """embedding 探针：折外表征 + 同一单元不跨折 → RF 分类标签（指标口径同 IND12）。

    没标签的行（缺 outcome 且 IND07 未出簇）直接不进探针：把空串当成一个类别喂给
    log_loss，只会让折外评估整块炸掉。
    """
    all_labels = [str(value) for value in frame["标签"]]
    keep = [position for position, value in enumerate(all_labels) if value]
    if len(keep) < 2:
        return []
    labels = [all_labels[position] for position in keep]
    keys = [str(frame["uid"].iloc[position]) for position in keep]
    if len(set(keys)) < 2:
        return []
    folds = ml.grouped_split(keys, max(int(params.get("folds") or 3), 2), seed)
    if not folds:
        return []
    matrix = np.stack([np.asarray(frame["embedding"].iloc[position], dtype=float)
                       for position in keep])
    classes = sorted(set(labels))
    estimator, engine = ml._make_model("rf", seed)
    rows = ml.majority_baseline(labels, folds)
    if estimator is None:
        return rows
    for fold, (train, test) in enumerate(folds, start=1):
        y_train = [labels[row] for row in train]
        y_test = [labels[row] for row in test]
        estimator.fit(matrix[train], y_train)
        predicted = [str(value) for value in estimator.predict(matrix[test])]
        raw = estimator.predict_proba(matrix[test])
        aligned = np.zeros((len(test), len(classes)), dtype=float)
        for position, name in enumerate(estimator.classes_):
            if str(name) in classes:
                aligned[:, classes.index(str(name))] = raw[:, position]
        aligned = aligned / np.clip(aligned.sum(axis=1, keepdims=True), 1e-12, None)
        row = ml.metrics(y_test, predicted, aligned, classes)
        row.update({"模型": "折外 embedding + RF", "折": fold, "n_train": int(len(train)),
                    "n_test": int(len(test)), "备注": "特征换成 SSL 表征，其余同 IND12（%s）"
                    % engine})
        rows.append(row)
    return rows


def transformer_ssl_analysis(ctx, params):
    """IND14：masked-code / 自编码 Transformer 预训练 → 折外 embedding + 重建误差异常分。"""
    marker = _start()
    name = "Transformer 自监督表征"
    gate = _dl_gate("IND14", name, ctx, MIN_STUDENTS_DL)
    if gate is not None:
        return gate
    networks = _networks()
    sequences = _wanted_sequences(ctx, params)
    if len(sequences) < 4:
        return blocked("IND14", name,
                       "可用序列不足 4 条（长度 ≥ %d 步）：self-attention 没足够上下文可学"
                       % am.MIN_SEQUENCE_LENGTH)
    seed = int(params.get("seed") or ml.DEFAULT_SEED)
    set_seed(seed)
    mode = "ae" if str(params.get("objective") or "masked") == "ae" else "masked"
    merge = bool(params.get("merge_vocab", True))
    folds_wanted = max(int(params.get("folds") or 3), 2)
    epochs = max(int(params.get("epochs") or 40), 1)
    lr = lr_of(params, 1e-3)
    batch_size = max(int(params.get("batch_size") or 32), 2)
    max_len = max(int(params.get("max_len") or MAX_SEQ_LEN), 8)
    ratio = min(max(float(params.get("mask_ratio") or 0.15), 0.05), 0.8)
    contamination = float(params.get("contamination") or 0.1)
    embed_rows, history_rows, data_rows, notes = [], [], [], []
    for stream, items in _streams(sequences, merge).items():
        vocab = stream_vocab(items)
        size = len(vocab)
        if size < 3:
            notes.append("流「%s」词表只有 %d 个码，masked 任务太简单，跳过" % (stream, size))
            continue
        records = []
        for position, (sequence, codes) in enumerate(items):
            indexed = [value for value in index_codes(list(codes), vocab) if value >= 0][:max_len]
            if len(indexed) < 3:
                continue
            records.append({"uid": str(sequence.uid), "seq": int(position),
                            "dimension": str(sequence.dimension),
                            "student": str(sequence.student),
                            "group_id": str(sequence.group_id), "tokens": indexed,
                            "流": stream})
        if len(records) < 4:
            notes.append("流「%s」只有 %d 条可用序列，不足以做折外预训练，跳过"
                         % (stream, len(records)))
            continue
        mask_id, pad_id = size, size + 1
        folds = ml.grouped_split([item["uid"] for item in records], folds_wanted, seed)
        if not folds:
            notes.append("流「%s」切不出折（独立单元 %d），跳过"
                         % (stream, len({item["uid"] for item in records})))
            continue
        floor = token_majority_share([item["tokens"] for item in records])
        stopped_any = False
        for fold, (train, test) in enumerate(folds, start=1):
            train_records = [records[row] for row in train.tolist()]
            test_records = [records[row] for row in test.tolist()]
            x_train, y_train = ssl_padded([item["tokens"] for item in train_records],
                                           mask_id, pad_id, ratio, seed + fold, mode)
            x_val, y_val = ssl_padded([item["tokens"] for item in train_records],
                                       mask_id, pad_id, ratio, seed + _EVAL_SEED + fold, mode)
            model = networks["MaskedTransformer"](
                size, int(params.get("embed_dim") or 64), int(params.get("heads") or 4),
                int(params.get("layers") or 2), max_len=max_len, mode=mode)
            set_seed(seed + fold)
            history, stopped = _train(
                model, _batches(list(x_train), list(y_train), batch_size, seed + fold),
                epochs, lr, ctx, "%s/%s 第%d折" % (stream, mode, fold),
                val=_batches(list(x_val), list(y_val), batch_size, seed, shuffle=False))
            history_rows += [dict(row, 流=stream, 折=fold, 目标=mode) for row in history]
            stopped_any = stopped_any or stopped
            evaluated = _ssl_evaluate(model, [item["tokens"] for item in test_records],
                                      mask_id, pad_id, seed + _EVAL_SEED + fold, ratio, mode)
            for item, score in zip(test_records, evaluated):
                embed_rows.append(dict(item, 折=int(fold), masked=int(score["masked"]),
                                       hit=int(score["hit"]), nll=fmt_round(score["nll"], 5),
                                       地板=fmt_round(floor, 4), embedding=score["embedding"]))
        data_rows.append({"流": stream, "目标模式": mode, "词表大小": size,
                          "序列数": len(records),
                          "独立单元数": len({item["uid"] for item in records}),
                          "折数": len(folds), "遮码比例": fmt_round(ratio, 3),
                          "平均长度": fmt_round(float(np.mean([len(item["tokens"])
                                                               for item in records])), 2)})
    if stopped_any and cancelled(ctx):
        notes.append("计算在训练日志卡被取消：只报已完成的折")
    if not embed_rows:
        return blocked("IND14", name, "没有任何一个流完成预训练：" +
                       ("；".join(notes) or "检查序列长度与词表"))
    frame = pd.DataFrame(embed_rows)
    matrix = np.stack([np.asarray(row, dtype=float) for row in frame["embedding"]])
    coords, explained = two_dimension_projection(matrix, seed)
    frame["PC1"] = fmt_round(coords[:, 0], 4)
    frame["PC2"] = fmt_round(coords[:, 1], 4)
    frame["命中率"] = fmt_round(np.where(frame["masked"] > 0,
                                         frame["hit"] / frame["masked"].replace(0, np.nan), 0.0), 4)
    cutoff = float(np.quantile(frame["nll"].astype(float),
                               max(0.0, min(1.0 - contamination, 0.999)))) \
        if contamination > 0 else float("inf")
    frame["异常"] = [int(bool(value >= cutoff)) for value in frame["nll"].astype(float)]
    labels, source, label_notes = _label_map(ctx, params)
    frame["标签"] = [labels.get((str(uid), str(dimension)), "")
                    for uid, dimension in zip(frame["uid"], frame["dimension"])]
    notes += label_notes
    probe_rows = _embedding_probe(frame, seed, params)
    mean_acc = float(np.mean([float(value) for value in frame["命中率"]]))
    floor = float(frame["地板"].max())
    qa = [QAItem("info" if mean_acc > floor else "warn", "IND14.masked_acc",
                 "折外 masked 命中率均值 %.4f，「永远猜最常见的码」地板 %.4f：%s"
                 % (mean_acc, floor,
                    "预训练确实学到了上下文" if mean_acc > floor else
                    "没跑过地板，embedding 很可能只是噪声，别拿它下游建模"))]
    if int(frame["uid"].nunique()) < MIN_STUDENTS_DL:
        qa.append(QAItem("warn", "IND14.small",
                         "只有 %d 名独立学生参与预训练，折外 embedding 的方差主要来自折本身"
                         % int(frame["uid"].nunique())))
    if contamination > 0:
        qa.append(QAItem("info", "IND14.anomalies",
                         "按预测异常率 %.0f%% 标出 %d 条重建误差最高的序列（NLL 阈值 %.4f）"
                         % (contamination * 100.0, int(frame["异常"].sum()), cutoff)))
    if cancelled(ctx):
        qa.append(QAItem("warn", "IND14.cancelled", "计算中途取消：只完成 %d 条序列的折外评估"
                         % len(frame)))
    if not probe_rows:
        qa.append(QAItem("warn", "IND14.no_probe",
                         "没有可用标签（缺 outcome 且 IND07 未出簇），embedding 探针未跑"))
    embeddings = frame[["uid", "student", "group_id", "dimension", "seq", "流", "折", "PC1",
                        "PC2", "标签", "masked", "命中率", "nll", "异常"]].rename(
        columns={"uid": "单元", "seq": "序列号", "masked": "遮码数", "nll": "重建NLL"})
    anomalies = embeddings.sort_values("重建NLL", ascending=False) \
        [["单元", "序列号", "流", "标签", "遮码数", "命中率", "重建NLL", "异常"]]
    tables = {"embeddings": embeddings.reset_index(drop=True),
              "pca": pd.DataFrame([{"主成分": "PC1", "解释率": fmt_round(explained[0], 4)},
                                   {"主成分": "PC2", "解释率": fmt_round(explained[1], 4)}]),
              "anomalies": anomalies.reset_index(drop=True),
              "metrics": pd.DataFrame(probe_rows),
              "pretraining": pd.DataFrame(history_rows),
              "data": pd.DataFrame(data_rows)}
    meta = {"objective": mode, "folds": folds_wanted, "seed": seed, "epochs": epochs,
            "max_len": max_len, "mask_ratio": fmt_round(ratio, 3), "torch": _torch_version(),
            "pca_explained": [fmt_round(explained[0], 4), fmt_round(explained[1], 4)],
            "masked_acc": fmt_round(mean_acc, 4), "floor": fmt_round(floor, 4),
            "label_source": source, "n_units": int(frame["uid"].nunique()),
            "n_sequences": int(len(frame))}
    notes += ["预训练目标：%s；embedding = 真实码位的 encoder 均值，按 student_uid 分折取折外"
              % ("自编码重建（ae）" if mode == "ae" else "masked-code 遮码重建"),
              "两维坐标由 numpy SVD（PCA）得到，解释率列在 pca 表；PC1/PC2 只供看图，不进入指标",
              "异常分 = 折外遮码位的平均负对数似然（越大越不像样本里的模式）"]
    return am.result("IND14", name, tidy_tables(tables), meta=meta, qa=qa, notes=notes,
                     elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ 组级公共件（IND15 / GRP10）

def euclidean_matrix(matrix):
    """行间的欧氏距离阵（numpy，不引 scipy.spatial）：组级特征聚类的输入。"""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not len(values):
        return np.zeros((len(values), len(values)), dtype=float)
    squares = (values ** 2).sum(axis=1)
    inner = squares[:, None] + squares[None, :] - 2.0 * values @ values.T
    return np.sqrt(np.clip(inner, 0.0, None))


def person_events(ctx, params):
    """事件表里可用的 PERSON→PERSON 行（与 GRP08 同一口径：去空端点、去自环）。"""
    frame = ctx.events
    if not len(frame):
        return pd.DataFrame()
    picked = {str(value) for value in gp.wanted_groups(ctx, params)}
    part = frame[frame["group_id"].astype(str).isin(picked)]
    part = part[(part["target_type"].astype(str) == "PERSON")
                & (part["actor_id"].astype(str) != "")
                & (part["target_id"].astype(str) != "")]
    part = part[part["actor_id"].astype(str) != part["target_id"].astype(str)]
    return part.copy()


def group_feature_table(ctx, params):
    """组级设计矩阵：成员特征按组求均值/标准差 + 规模 + 事件密度；标签 = ``bundle.outcome``。

    返回 ``(frame, 特征列, notes)``：每行一个组，列 ``label`` 是组结果（可为空串）。
    """
    table = nt.role_feature_table(ctx, params)
    notes = []
    if not len(table):
        return pd.DataFrame(), [], ["成员特征表为空（没有可解析的序列），组级建模无从下手"]
    numeric = ml.feature_columns(table, params.get("families"))
    if not numeric:
        return pd.DataFrame(), [], ["选中特征族里没有非常数数值列，组级表无从构造"]
    rows = []
    for group_id in gp.wanted_groups(ctx, params):
        part = table[table["group_id"].astype(str) == str(group_id)]
        if not len(part):
            continue
        row = {"group_id": str(group_id), "n_members": int(len(part))}
        for column in numeric:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            row["%s_均值" % column] = fmt_round(float(values.mean()), 4) if len(values) else 0.0
            row["%s_标准差" % column] = fmt_round(float(values.std(ddof=0)), 4) \
                if len(values) > 1 else 0.0
        bundle = ctx.bundle(group_id)
        row["label"] = "" if (bundle is None or bundle.outcome is None) else str(bundle.outcome)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not len(frame):
        return pd.DataFrame(), [], ["选中的组里没有成员特征，组级表为空"]
    events = person_events(ctx, params)
    if len(events):
        counts = events.groupby("group_id").size()
        frame["事件数"] = frame["group_id"].map(counts).fillna(0).astype(int)
        frame["事件密度"] = [fmt_round(float(count) / max(int(size) * (int(size) - 1), 1), 4)
                         for count, size in zip(frame["事件数"], frame["n_members"])]
    else:
        frame["事件数"] = 0
        frame["事件密度"] = 0.0
        notes.append("没有 PERSON→PERSON 事件：事件数/事件密度按 0 入表")
    columns = [column for column in frame.columns if column not in ("group_id", "label")]
    columns = [column for column in columns
               if float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).std(ddof=0)) > 0]
    notes.append("组级特征 = 成员轨迹/网络特征（%s）的组内均值与标准差 + 规模与事件密度；"
                 "常数被剔除（当前 %d 列）" % ("、".join(numeric[:6]) + ("…" if len(numeric) > 6 else ""),
                                            len(columns)))
    return frame, columns, notes


def member_feature_tensor(ctx, params, table=None):
    """DeepSets 用的成员级特征：返回 ``(组列表, {组: (行下标阵, 标签)}, 特征列, 矩阵)``。"""
    frame = table if table is not None else nt.role_feature_table(ctx, params)
    if not len(frame):
        return [], {}, [], np.zeros((0, 0), dtype=float)
    columns = ml.feature_columns(frame, params.get("families"))
    matrix, kept = nt.zscore_matrix(frame, columns)
    groups, index = [], {}
    for position, group_id in enumerate(frame["group_id"].astype(str)):
        if group_id not in index:
            index[group_id] = [position]
            groups.append(group_id)
        else:
            index[group_id].append(position)
    labels = {}
    for group_id in groups:
        bundle = ctx.bundle(group_id)
        rows = np.asarray(index[group_id], dtype=int)
        labels[group_id] = (rows, "" if (bundle is None or bundle.outcome is None)
                            else str(bundle.outcome))
    return groups, labels, list(kept), matrix


def loyo_feature_predictions(matrix, labels, folds, seed, class_weight=True):
    """特征地板（RF / HistGB / XGBoost）的 LOYO 折外预测：返回 ``(预测行, 汇总行)``。"""
    texts = [str(value) for value in labels]
    classes = sorted({item for item in texts if item})
    if len(classes) < 2 or not folds:
        return [], []
    weights = ml.class_balanced_weights(texts)
    prediction_rows, summary_rows = [], []
    for key, name in (("rf", "组特征 RF"), ("histgb", "组特征 HistGB"), ("xgb", "组特征 XGBoost")):
        estimator, engine = ml._make_model(key, seed)
        if estimator is None:
            continue
        proba = np.zeros((len(texts), len(classes)), dtype=float)
        predicted = [""] * len(texts)
        broken = ""
        for train, test in folds:
            y_train = [texts[row] for row in train]
            try:
                estimator.fit(matrix[train], y_train,
                              sample_weight=weights[train] if class_weight else None)
            except (TypeError, ValueError):               # 不支持 sample_weight 的模型退一步
                try:
                    estimator.fit(matrix[train], y_train)
                except Exception as error:                 # noqa: BLE001
                    broken = str(error)
                    break
            except Exception as error:                     # noqa: BLE001
                broken = str(error)
                break
            raw = estimator.predict_proba(matrix[test])
            for position, label in enumerate(estimator.classes_):
                if str(label) in classes:
                    proba[test, classes.index(str(label))] = raw[:, position]
            for position, row in enumerate(test.tolist()):
                predicted[row] = str(estimator.predict(matrix[test][position:position + 1])[0])
        if broken:
            continue
        blank = [row for row, value in enumerate(predicted) if not value]
        for row in blank:
            predicted[row] = classes[int(np.argmax(proba[row]))] if proba[row].any() else classes[0]
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        row = ml.metrics(texts, predicted, proba, classes)
        row.update({"模型": name, "折": "折外汇总", "n_train": int(len(texts) -
                                                              len(folds)),
                    "n_test": int(len(folds)), "备注": "LOYO 折外（每组恰好预测一次，%s）" % engine})
        summary_rows.append(row)
        for position, text in enumerate(texts):
            prediction_rows.append({"模型": name, "组": text, "真实": texts[position],
                                    "预测": predicted[position],
                                    "预测正确": int(predicted[position] == texts[position]),
                                    "真实概率": fmt_round(float(proba[position,
                                                              classes.index(texts[position])]), 4)})
    return prediction_rows, summary_rows


def majority_prediction(labels, folds):
    """地板的折外版：每折用训练侧多数类预测测试侧，汇总成唯一一行指标。"""
    texts = [str(value) for value in labels]
    classes = sorted({item for item in texts if item})
    predicted = [""] * len(texts)
    for train, test in folds:
        counts = pd.Series([texts[row] for row in train], dtype=str).value_counts()
        top = str(counts.index[0]) if len(counts) else (classes[0] if classes else "")
        for row in test:
            predicted[int(row)] = top
    uniform = np.tile([1.0 / max(len(classes), 1)], (len(texts), max(len(classes), 1)))
    row = ml.metrics(texts, predicted, uniform, classes)
    row.update({"模型": "多数类基线", "折": "折外汇总", "n_train": int(len(texts)),
                "n_test": int(len(folds)), "备注": "常数预测器（不建模，只看类别先验）"})
    return row, predicted


def group_mean_matrix(matrix, group_keys, groups):
    """成员级特征阵 → 组均值阵（行序跟 ``groups`` 一致）：GAT 的特征地板输入。"""
    keys = [str(value) for value in group_keys]
    values = np.asarray(matrix, dtype=float)
    width = int(values.shape[1]) if values.ndim == 2 and values.size else 0
    out = np.zeros((len(groups), width), dtype=float)
    if not width:
        return out
    for position, group in enumerate(groups):
        rows = [index for index, key in enumerate(keys) if key == str(group)]
        if rows:
            out[position] = values[rows].mean(axis=0)
    return out


# ------------------------------------------------------------------ IND15 Temporal GAT

def temporal_edge_buckets(frame, nodes, buckets=6):
    """事件行 → 逐时间桶的边下标阵：节点集固定，组内把时刻归一到 [0,1) 再切桶。

    返回 ``(edge_index 列表, 边统计 frame)``；edge_index 为 (2, E) 的 int64 阵（第一行 src）。
    端点不在成员表里（外部主体、未登记名字）的事件不建节点，直接丢弃。
    """
    count = max(int(buckets), 1)
    blank = pd.DataFrame(columns=["桶", "source", "target", "次数"])
    if not len(frame):
        return [np.zeros((2, 0), dtype=np.int64) for _ in range(count)], blank
    part = frame.copy()
    part["moment"] = pd.to_numeric(part["timestamp"], errors="coerce")
    part = part[part["moment"].notna()]
    if not len(part):
        return [np.zeros((2, 0), dtype=np.int64) for _ in range(count)], blank
    span = part.groupby("group_id")["moment"].agg(["min", "max"])
    edges = {}
    for record in part.itertuples(index=False):
        group = str(record.group_id)
        limits = span.loc[group]
        low, high = float(limits["min"]), float(limits["max"])
        ratio = 0.5 if high <= low else (float(record.moment) - low) / (high - low)
        bucket = min(max(int(ratio * count), 0), count - 1)
        source = nodes.get(ac.student_uid(group, str(record.actor_id)))
        target = nodes.get(ac.student_uid(group, str(record.target_id)))
        if source is None or target is None:
            continue
        key = (bucket, int(source), int(target))
        edges[key] = edges.get(key, 0) + 1
    lookup = {position: uid for uid, position in nodes.items()}
    out, rows = [], []
    for bucket in range(count):
        pairs = [(src, dst) for (index, src, dst), _hit in sorted(edges.items())
                 if index == bucket]
        out.append(np.asarray(pairs, dtype=np.int64).T if pairs
                   else np.zeros((2, 0), dtype=np.int64))
    for (bucket, source, target), hits in sorted(edges.items()):
        rows.append({"桶": int(bucket), "source": lookup.get(source, ""),
                     "target": lookup.get(target, ""), "次数": int(hits)})
    return out, pd.DataFrame(rows)


def _gat_tensors(matrix, buckets):
    """节点特征 + 逐桶边表 → torch 张量（训练与评估共用一套口径）。"""
    nodes = torch.as_tensor(np.asarray(matrix, dtype=np.float32))
    tensors = [torch.as_tensor(np.asarray(array, dtype=np.int64), dtype=torch.long)
               for array in buckets]
    return nodes, tensors


def _train_gat(model, nodes, tensors, groups, targets, train_rows, epochs, lr, ctx, stage,
               val_rows=(), weights=None):
    """全批训练（组数很少，逐批没意义）：返回 ``(history, 是否被取消)``。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    labels = torch.as_tensor(np.asarray(targets, dtype=np.int64))
    weight = torch.as_tensor(np.asarray(weights, dtype=np.float32)) \
        if weights is not None else None
    history, stopped = [], False
    for epoch in range(1, max(int(epochs), 1) + 1):
        if cancelled(ctx):
            stopped = True
            break
        model.train()
        picked = list(train_rows)
        logits, _readout, _attention = model(nodes, tensors, [groups[row] for row in picked])
        loss = _weighted_ce(logits, labels[picked],
                            None if weight is None else weight[picked])
        value = float(loss.item())
        if math.isfinite(value):
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        row = {"阶段": stage, "epoch": epoch, "train_loss": fmt_round(value, 5)}
        if len(val_rows):
            model.eval()
            with torch.no_grad():
                checked, _r, _a = model(nodes, tensors, [groups[row] for row in val_rows])
                row["val_loss"] = fmt_round(float(nn.functional.cross_entropy(
                    checked, labels[list(val_rows)]).item()), 5)
        history.append(row)
        log_progress(ctx, "%s epoch %d/%d：train_loss=%.5f%s"
                     % (stage, epoch, epochs, row["train_loss"],
                        "" if "val_loss" not in row else "  val_loss=%.5f" % row["val_loss"]))
    return history, stopped


def gat_attention_weights(model, nodes, tensors, names):
    """逐桶逐边的平均注意力（head 与层都取均）：``{(桶, src名, dst名): 权重}``。"""
    model.eval()
    out = {}
    layers = max(len(model.layers), 1)
    with torch.no_grad():
        _state, weights = model.embed(nodes, tensors)
    for position, weight in enumerate(weights):
        bucket = position // layers
        array = np.asarray(weight.detach().cpu().numpy(), dtype=float)
        if not array.shape[0]:
            continue
        edge = tensors[bucket].cpu().numpy()
        mean = array.mean(axis=1)
        for index in range(edge.shape[1]):
            key = (int(bucket), names[int(edge[0, index])], names[int(edge[1, index])])
            out[key] = out.get(key, 0.0) + float(mean[index]) / layers
    return out


def temporal_gat_analysis(ctx, params):
    """IND15：时序图注意力 —— 成员为节点、逐时刻事件为边，读组结果分类（LOYO）。"""
    marker = _start()
    name = "时序图注意力网络（GAT）"
    gate = _dl_gate("IND15", name, ctx)
    if gate is not None:
        return gate
    networks = _networks()
    if not ctx.has_person_edges:
        return blocked("IND15", name,
                       "没有 PERSON→PERSON 事件边：先在「小组交互」页录入，或随文件上传事件表")
    table = nt.role_feature_table(ctx, params)
    if not len(table):
        return blocked("IND15", name, "成员角色特征表为空：图模型没有节点特征可用")
    labels = {str(bundle.group_id): ("" if bundle.outcome is None else str(bundle.outcome))
              for bundle in ctx.bundles}
    seed = int(params.get("seed") or ml.DEFAULT_SEED)
    set_seed(seed)
    columns = ml.feature_columns(table)
    matrix, kept = nt.zscore_matrix(table, columns)
    names = [str(value) for value in table["uid"]]
    nodes = {uid: position for position, uid in enumerate(names)}
    present = {str(value) for value in table["group_id"]}
    wanted = [group for group in gp.wanted_groups(ctx, params)
              if str(group) in present and labels.get(str(group))]
    classes = sorted({labels[str(group)] for group in wanted})
    if len(classes) < 2:
        return blocked("IND15", name,
                       "能算节点特征且有 outcome 的组只有 %d 个，且取值不足 2 类：图分类没监督信号"
                       % len(wanted))
    if len(wanted) < 3:
        return blocked("IND15", name, "带 outcome 且能算特征的组只有 %d 个，不足 3 个："
                                   "LOYO 至少需要三组" % len(wanted))
    group_keys = [str(value) for value in table["group_id"]]
    members = {group: [names.index(uid) for uid, key in zip(names, group_keys) if key == str(group)]
               for group in wanted}
    buckets_wanted = max(int(params.get("buckets") or 6), 1)
    edge_buckets, edge_stats = temporal_edge_buckets(person_events(ctx, params), nodes,
                                                    buckets_wanted)
    total_edges = int(edge_stats["次数"].sum()) if len(edge_stats) else 0
    if not total_edges:
        return blocked("IND15", name,
                       "切桶后一条边也没剩下：检查事件时间戳能否解析、端点名是否在成员表里")
    groups = [members[str(group)] for group in wanted]
    targets = [classes.index(labels[str(group)]) for group in wanted]
    folds = ml.leave_one_group_out([str(group) for group in wanted])
    weights = ml.class_balanced_weights([labels[str(group)] for group in wanted]) \
        if params.get("class_weight", True) else None
    nodes_t, tensors = _gat_tensors(matrix, edge_buckets)
    history_rows, prediction_rows, notes = [], [], []
    attention_totals = {}
    final_model = None
    for fold, (train, test) in enumerate(folds, start=1):
        model = networks["TemporalGAT"](
            int(np.asarray(matrix).shape[1]), int(params.get("hidden_size") or 32),
            heads=int(params.get("heads") or 4), layers=int(params.get("layers") or 1),
            classes=len(classes))
        set_seed(seed + fold)
        history, stopped = _train_gat(model, nodes_t, tensors, groups, targets, train,
                                      max(int(params.get("epochs") or 60), 1),
                                      lr_of(params, 5e-3), ctx,
                                      "temporal-GAT 第%d折" % fold, val_rows=test,
                                      weights=weights)
        history_rows += [dict(row, 折=fold) for row in history]
        if stopped:
            notes.append("第 %d 折中途取消：该折不报预测" % fold)
            continue
        final_model = model
        model.eval()
        with torch.no_grad():
            logits, _readout, _attention = model(nodes_t, tensors,
                                                [groups[row] for row in test])
        probabilities = _softmax(np.asarray(logits.detach().cpu().numpy(), dtype=float))
        for position, row in enumerate(test.tolist()):
            top = int(np.argmax(probabilities[position]))
            prediction_rows.append({"组": wanted[row], "折": fold,
                                    "真实": classes[targets[row]], "预测": classes[top],
                                    "预测正确": int(top == targets[row]),
                                    "真实概率": fmt_round(
                                        float(probabilities[position, targets[row]]), 4)})
        for key, value in gat_attention_weights(model, nodes_t, tensors, names).items():
            attention_totals[key] = attention_totals.get(key, 0.0) + value / max(len(folds), 1)
    if not prediction_rows:
        return blocked("IND15", name, "一折也没跑完：" + ("；".join(notes) or "检查训练参数"))
    frame = pd.DataFrame(prediction_rows)
    texts = [str(value) for value in frame["真实"]]
    predicted = [str(value) for value in frame["预测"]]
    one_hot = np.zeros((len(texts), len(classes)), dtype=float)
    for position, text in enumerate(predicted):
        if text in classes:
            one_hot[position, classes.index(text)] = 1.0
    row = ml.metrics(texts, predicted, one_hot, classes)
    row.update({"模型": "Temporal GAT", "折": "折外汇总", "n_train": len(texts) - len(folds),
                "n_test": len(folds),
                "备注": "LOYO 折外（每组恰好预测一次）：逐折只有 1 个测试组，指标只看汇总"})
    baseline, _base = majority_prediction(texts, folds)
    feature_rows, feature_summary = loyo_feature_predictions(
        group_mean_matrix(matrix, group_keys, wanted), texts, folds, seed,
        bool(params.get("class_weight", True)))
    metric_rows = [row] + feature_summary + [baseline]
    attention_rows = [{"桶": int(key[0]), "source": key[1], "target": key[2],
                       "平均注意力": fmt_round(value, 5)}
                      for key, value in sorted(attention_totals.items())]
    attention = pd.DataFrame(attention_rows)
    if len(attention):
        attention = attention.merge(edge_stats, on=["桶", "source", "target"], how="left")
    else:
        attention = pd.DataFrame(columns=["桶", "source", "target", "平均注意力", "次数"])
    embeddings, explained = pd.DataFrame(columns=["节点", "组", "PC1", "PC2"]), [0.0, 0.0]
    if final_model is not None:
        final_model.eval()
        with torch.no_grad():
            state, _weights = final_model.embed(nodes_t, tensors)
        coords, explained = two_dimension_projection(
            np.asarray(state.detach().cpu().numpy(), dtype=float), seed)
        labeled = {str(group) for group in wanted}
        embeddings = pd.DataFrame({"节点": names,
                                   "组": group_keys,
                                   "所属组有标签": [int(key in labeled) for key in group_keys],
                                   "PC1": fmt_round(coords[:, 0], 4),
                                   "PC2": fmt_round(coords[:, 1], 4)})
    qa = [QAItem("info", "IND15.loyo",
                 "跨组预测只跑了 %d 折（每组一次）：折外 Macro-F1 的置信区间宽到不可比，"
                 "结论只能当线索" % len(folds))]
    gat_score = float(row["Macro-F1"])
    if gat_score <= float(baseline["Macro-F1"]):
        qa.append(QAItem("warn", "IND15.vs_baseline",
                         "GAT 折外 Macro-F1 %.4f ≤ 多数类地板 %.4f：小样本图上深度模型没优势，"
                         "建议改报组特征模型或描述性结果" % (gat_score, float(baseline["Macro-F1"]))))
    if feature_summary:
        best_feature = max(float(item["Macro-F1"]) for item in feature_summary)
        qa.append(QAItem("info" if gat_score > best_feature else "warn", "IND15.vs_features",
                         "组特征模型最高 Macro-F1 %.4f，GAT %.4f：%s"
                         % (best_feature, gat_score,
                            "图结构有增量" if gat_score > best_feature else "图结构没增量")))
    else:
        notes.append("组特征地板没跑成（特征列全为常数或 sklearn 拟合失败），只报了 GAT 与多数类")
    if len(texts) <= 8:
        qa.append(QAItem("warn", "IND15.small",
                         "只有 %d 组样本、%d 类：每类不足 2 组的折外预测儿戏不得" % (
                             len(texts), len(classes))))
    if cancelled(ctx):
        qa.append(QAItem("warn", "IND15.cancelled", "计算中途取消：只完成 %d 折" % len(folds)))
    notes += ["节点 = 全体「组__成员」（无边的成员也保留，消息为零），特征 = 成员角色特征 z 分数；"
              "逐时刻桶共享同一节点集，时间推进用 GRUCell",
              "折外预测：每组恰好被预测一次；注意力为各折均值（跨折可比），节点坐标取自最后一折"
              "模型（折间不可比，只供看图）"]
    tables = {"predictions": frame.reset_index(drop=True),
              "metrics": pd.DataFrame(metric_rows),
              "nodes": embeddings,
              "attention": attention.reset_index(drop=True),
              "edges": edge_stats,
              "training": pd.DataFrame(history_rows),
              "data": pd.DataFrame([{"节点数": len(names), "组数": len(wanted),
                                      "特征列": len(kept), "时间桶数": buckets_wanted,
                                      "边次总数": total_edges,
                                      "不同边数": int(len(edge_stats)),
                                      "类别": "、".join(classes)}])}
    meta = {"buckets": buckets_wanted, "folds": len(folds), "seed": seed, "classes": classes,
            "heads": int(params.get("heads") or 4), "layers": int(params.get("layers") or 1),
            "hidden_size": int(params.get("hidden_size") or 32),
            "epochs": max(int(params.get("epochs") or 60), 1),
            "class_weight": bool(params.get("class_weight", True)),
            "macro_f1": row["Macro-F1"], "floor_macro_f1": baseline["Macro-F1"],
            "pca_explained": [fmt_round(explained[0], 4), fmt_round(explained[1], 4)],
            "torch": _torch_version(), "n_groups": len(wanted), "n_nodes": len(names)}
    return am.result("IND15", name, tidy_tables(tables), meta=meta, qa=qa, notes=notes,
                     elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ GRP10 跨组比较 / 聚类 / 预测

def _stack_groups(matrix, index, groups):
    """成员阵按组堆成 ``(n_groups, max_members, in_dim)`` + mask：DeepSets 的输入。"""
    values = np.asarray(matrix, dtype=np.float32)
    height = max([len(index[group]) for group in groups] or [1])
    width = int(values.shape[1]) if values.ndim == 2 and values.size else 0
    out = np.zeros((len(groups), height, width), dtype=np.float32)
    mask = np.zeros((len(groups), height), dtype=np.float32)
    for row, group in enumerate(groups):
        rows = list(index[group])
        out[row, :len(rows)] = values[rows]
        mask[row, :len(rows)] = 1.0
    return out, mask


def _train_deepsets(model, features, mask, targets, train, epochs, lr, ctx, stage, val_rows=(),
                    weights=None):
    """全批训练 DeepSets（组数少，逐批没意义）：返回 ``(history, 是否被取消)``。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    x = torch.as_tensor(np.asarray(features, dtype=np.float32))
    m = torch.as_tensor(np.asarray(mask, dtype=np.float32))
    y = torch.as_tensor(np.asarray(targets, dtype=np.int64))
    weight = torch.as_tensor(np.asarray(weights, dtype=np.float32)) \
        if weights is not None else None
    history, stopped = [], False
    for epoch in range(1, max(int(epochs), 1) + 1):
        if cancelled(ctx):
            stopped = True
            break
        model.train()
        picked = list(train)
        logits, _z = model(x[picked], m[picked])
        loss = _weighted_ce(logits, y[picked], None if weight is None else weight[picked])
        value = float(loss.item())
        if math.isfinite(value):
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        row = {"阶段": stage, "epoch": epoch, "train_loss": fmt_round(value, 5)}
        if len(val_rows):
            model.eval()
            with torch.no_grad():
                checked, _r = model(x[list(val_rows)], m[list(val_rows)])
                row["val_loss"] = fmt_round(float(nn.functional.cross_entropy(
                    checked, y[list(val_rows)]).item()), 5)
        history.append(row)
        log_progress(ctx, "%s epoch %d/%d：train_loss=%.5f%s"
                     % (stage, epoch, epochs, row["train_loss"],
                        "" if "val_loss" not in row else "  val_loss=%.5f" % row["val_loss"]))
    return history, stopped


def group_prediction_analysis(ctx, params):
    """GRP10：跨组非参数检验（BH + 效应量）+ 组聚类 + 组结果预测（含 DeepSets）。"""
    marker = _start()
    name = "跨组比较 / 组聚类 / 组结果预测"
    if cancelled(ctx):
        return blocked("GRP10", name, "已取消：计算在训练日志卡被中止，未产生任何结果")
    frame, columns, notes = group_feature_table(ctx, params)
    if not len(frame) or not columns:
        return blocked("GRP10", name, (notes or ["组级特征表为空"])[0])
    groups = [str(value) for value in frame["group_id"]]
    labels = [str(value) for value in frame["label"]]
    if len(groups) < 3:
        return blocked("GRP10", name,
                       "可用组只有 %d 个，不足 3 个：跨组检验与 LOYO 预测都跑不动" % len(groups))
    seed = int(params.get("seed") or ml.DEFAULT_SEED)
    set_seed(seed)
    wanted = list(columns)[:max(int(params.get("max_features") or 40), 1)]
    qa = []
    # ---------------- A 跨组比较
    omnibus_rows, cross_parts = [], []
    for column in wanted:
        part = gp.cross_group_compare(frame, column)
        if not len(part):
            continue
        cross_parts.append(part)
        found = part.attrs.get("kruskal")
        if found:
            height = float(found["H"])
            degrees = int(found["df"])
            p_value = float(found["p"])
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            omnibus_rows.append({"特征": column, "组数": int(frame["group_id"].nunique()),
                                 "n": int(len(values)), "H": fmt_round(height, 4),
                                 "df": int(degrees), "p": fmt_round(p_value, 6),
                                 "中位数": fmt_round(float(values.median()), 4)})
    omnibus = pd.DataFrame(omnibus_rows)
    if len(omnibus):
        q_values = am.bh_fdr([float(value) for value in omnibus["p"]])
        omnibus["q(BH)"] = fmt_round(pd.Series(q_values, dtype=float), 6)
        omnibus = omnibus.sort_values("p").reset_index(drop=True)
        hits = int((pd.to_numeric(omnibus["q(BH)"], errors="coerce") <= 0.05).sum())
        qa.append(QAItem("info" if hits else "warn", "GRP10.omnibus",
                         "%d 个特征的 Kruskal–Wallis 全局检验里，BH 校正后 q ≤ 0.05 的有 %d 个：%s"
                         % (len(omnibus), hits, "差异不是个例，可逐特征看两两表" if hits
                          else "组间差异没超出抽样噪声，后面的两两比较别拿着单个 p<0.05 下结论")))
    cross = pd.concat(cross_parts, ignore_index=True) if cross_parts else pd.DataFrame()
    # ---------------- B 组聚类
    matrix_z, kept = nt.zscore_matrix(frame, wanted)
    dist = euclidean_matrix(matrix_z)
    cluster_labels, tree = [], None
    k = max(2, min(int(params.get("k") or 3), len(groups) - 1))
    mean_silhouette = None
    try:
        if str(params.get("cluster_method") or "ahc") == "pam":
            cluster_labels, _medoids = sq.pam_clusters(dist, k)
        else:
            cluster_labels, tree = sq.ahc_clusters(dist, params.get("linkage") or "average", k)
    except ValueError as error:
        notes.append("组聚类没跑成：%s" % error)
    cluster_rows = []
    if cluster_labels:
        scores, mean_silhouette = sq.silhouette_scores(dist, cluster_labels)
        for position, group in enumerate(groups):
            cluster_rows.append({"组": group, "簇": int(cluster_labels[position]),
                                 "silhouette": fmt_round(scores[position], 4),
                                 "组结果": labels[position],
                                 "成员数": int(frame["n_members"].iloc[position])})
        if float(mean_silhouette) < 0.1:
            qa.append(QAItem("warn", "GRP10.weak_silhouette",
                             "组级聚类平均 silhouette = %.3f：组之间在选中的特征上分不开" %
                             float(mean_silhouette)))
    clusters = pd.DataFrame(cluster_rows)
    # ---------------- C 组结果预测
    keep_rows = [position for position, text in enumerate(labels) if text]
    classes = sorted({labels[position] for position in keep_rows})
    prediction_rows, metric_rows, training_rows = [], [], []
    embedding_frame = pd.DataFrame()
    deepsets = ""
    if len(keep_rows) < 3 or len(classes) < 2:
        qa.append(QAItem("warn", "GRP10.no_label",
                         "带 outcome 的组只有 %d 个（类 %d 个）：组结果预测跳过，"
                         "只报跨组比较与聚类" % (len(keep_rows), len(classes))))
        notes.append("组结果预测需要 ≥ 3 个带 outcome 的组且至少 2 类；现在的标签分布不够")
    else:
        sub = frame.iloc[keep_rows].reset_index(drop=True)
        sub_labels = [labels[position] for position in keep_rows]
        folds = ml.leave_one_group_out([str(value) for value in sub["group_id"]])
        x = sub[wanted].apply(lambda column: pd.to_numeric(column, errors="coerce")) \
            .fillna(0.0).to_numpy(dtype=float)
        feature_rows, feature_summary = loyo_feature_predictions(
            x, sub_labels, folds, seed, bool(params.get("class_weight", True)))
        baseline, _base = majority_prediction(sub_labels, folds)
        metric_rows = feature_summary + [baseline]
        prediction_rows = feature_rows
        if not feature_summary:
            notes.append("组特征模型一个也没拟合成（特征全常数或样本太少），只报了多数类地板")
        if TORCH and len(folds) >= 2:
            networks = _networks()
            _groups, mapping, member_columns, member_matrix = member_feature_tensor(ctx, params)
            usable = [group for group in _groups if mapping.get(group, (None, ""))[1]
                      and len(mapping[group][0])]
            if len(usable) >= 3:
                names = sorted({mapping[group][1] for group in usable})
                class_of = {text: position for position, text in enumerate(names)}
                targets = [class_of[mapping[group][1]] for group in usable]
                index = {group: mapping[group][0] for group in usable}
                features, mask = _stack_groups(member_matrix, index, usable)
                # 成员堆叠一次性上张量：逐折返numpy → as_tensor 会让 dtype 参数反复错用
                features_t = torch.as_tensor(np.asarray(features, dtype=np.float32))
                mask_t = torch.as_tensor(np.asarray(mask, dtype=np.float32))
                ds_folds = ml.leave_one_group_out(usable)
                weights = ml.class_balanced_weights([mapping[group][1] for group in usable]) \
                    if params.get("class_weight", True) else None
                pool = str(params.get("pool") or "attention")
                proba = np.zeros((len(usable), len(class_of)), dtype=float)
                predicted = [""] * len(usable)
                final_model = None
                for fold, (train, test) in enumerate(ds_folds, start=1):
                    model = networks["DeepSetsNet"](
                        int(member_matrix.shape[1]), int(params.get("width") or 64),
                        int(params.get("out_dim") or 16), pool=pool, classes=len(class_of))
                    set_seed(seed + fold)
                    history, stopped = _train_deepsets(
                        model, features, mask, targets, train,
                        max(int(params.get("epochs") or 80), 1),
                        lr_of(params, 5e-3), ctx, "DeepSets 第%d折" % fold,
                        val_rows=test, weights=weights)
                    training_rows += [dict(row, 折=fold) for row in history]
                    if stopped:
                        continue
                    final_model = model
                    model.eval()
                    with torch.no_grad():
                        logits, _z = model(features_t[list(test)], mask_t[list(test)])
                    part = _softmax(np.asarray(logits.detach().cpu().numpy(), dtype=float))
                    for position, row in enumerate(test.tolist()):
                        top = int(np.argmax(part[position]))
                        proba[row] = part[position]
                        predicted[row] = names[top]
                scored = [position for position, text in enumerate(predicted) if text]
                if scored:
                    # 只拿真的拿到折外预测的组算指标（取消的折留空串，不是类别）
                    row = ml.metrics([mapping[usable[position]][1] for position in scored],
                                     [predicted[position] for position in scored],
                                     proba[scored], names)
                    row.update({"模型": "DeepSets 组 embedding（%s 池化）" % pool,
                                "折": "折外汇总", "n_train": len(usable) - len(ds_folds),
                                "n_test": len(scored),
                                "备注": "LOYO 折外：成员变长→组 embedding，不要求同组成员数相同"})
                    metric_rows.append(row)
                    deepsets = row["Macro-F1"]
                if final_model is not None:
                    final_model.eval()
                    with torch.no_grad():
                        _logits, vectors = final_model(features_t, mask_t)
                    coords, _explained = two_dimension_projection(
                        np.asarray(vectors.detach().cpu().numpy(), dtype=float), seed)
                    embedding_frame = pd.DataFrame({
                        "组": usable, "组结果": [mapping[group][1] for group in usable],
                        "成员数": [len(index[group]) for group in usable],
                        "PC1": fmt_round(coords[:, 0], 4), "PC2": fmt_round(coords[:, 1], 4)})
            else:
                notes.append("DeepSets 分支跳过：能对上 outcome 且成员可取特征不足 3 组")
        elif not TORCH:
            notes.append("未安装 torch：DeepSets 组 embedding 分支跳过（安装见 "
                         "requirements-optional.txt），组结果预测只报特征地板")
        else:
            notes.append("DeepSets 分支跳过：LOYO 折数不足 2，折外评估无意义")
    tables = {"group_features": frame,
              "omnibus": omnibus,
              "cross_tests": cross,
              "clusters": clusters,
              "metrics": pd.DataFrame(metric_rows),
              "predictions": pd.DataFrame(prediction_rows),
              "embeddings": embedding_frame,
              "training": pd.DataFrame(training_rows)}
    if tree is not None:
        tables["dendrogram"] = pd.DataFrame(tree,
                                           columns=["合并a", "合并b", "高度", "样本数"])
    meta = {"n_groups": len(groups), "features": len(wanted), "k": k,
            "cluster_method": str(params.get("cluster_method") or "ahc"),
            "silhouette": fmt_round(float(mean_silhouette), 4) if mean_silhouette is not None else "",
            "classes": "、".join(classes), "labeled_groups": len(keep_rows),
            "deepsets_macro_f1": deepsets, "torch": _torch_version(), "seed": seed,
            "member_features": len(kept)}
    notes += ["跨组检验：逐特征 Kruskal–Wallis 全局 + Mann–Whitney 两两 + BH 校正（Cliff's δ 与 "
              "rank-biserial 同 GRP06 口径）",
              "组聚类在 z 分数化的组级特征上用欧氏距离；组级特征表见 group_features"]
    return am.result("GRP10", name, tidy_tables(tables), meta=meta, qa=qa, notes=notes,
                     elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ 参数与方法声明

P_SEED = am.ParamSpec("seed", "随机种子", "spin", default=ml.DEFAULT_SEED, minimum=0,
                      maximum=999999, tooltip="进 torch.manual_seed 与 sklearn，同一颗种子结果可复现")
P_EPOCHS = am.ParamSpec("epochs", "训练轮数", "spin", default=40, minimum=1, maximum=2000,
                        tooltip="逐 epoch 上报训练日志，中途可取消；小样本上 30–80 轮足够")
P_LR = am.ParamSpec("lr", "学习率", "text", default="0.001",
                    tooltip="可填 1e-3 这样的写法；留空或非数字按默认 0.001")
P_LR_FAST = am.ParamSpec("lr", "学习率", "text", default="0.005",
                         tooltip="图模型 / DeepSets 默认 0.005（层少、样本是组，得走得快一点）")
P_BATCH = am.ParamSpec("batch_size", "批大小", "spin", default=64, minimum=2, maximum=1024)
P_EMBED = am.ParamSpec("embed_dim", "词嵌入维度", "spin", default=32, minimum=4, maximum=256)
P_HIDDEN = am.ParamSpec("hidden_size", "隐层宽度", "spin", default=64, minimum=8, maximum=512)
P_GAT_HIDDEN = am.ParamSpec("hidden_size", "隐层宽度（每头）", "spin", default=32, minimum=4,
                            maximum=256, tooltip="实际宽度 = 该值 × 注意力头数")
P_LAYERS = am.ParamSpec("layers", "层数", "spin", default=1, minimum=1, maximum=6)
P_HEADS = am.ParamSpec("heads", "注意力头数", "spin", default=4, minimum=1, maximum=16,
                       tooltip="嵌入维度须能被头数整除时最自然，不整除时模型内部自行补齐")
P_FOLDS = am.ParamSpec("folds", "折数", "spin", default=3, minimum=2, maximum=10,
                       tooltip="按独立单元（学生 / 组）整块切，同一单元不跨折")
P_CV = am.ParamSpec("cv", "切分方式", "combo",
                    options=(("kfold", "分组 K 折"), ("holdout", "单次留出"),
                             ("loyo", "留一组出 LOYO")), default="kfold",
                    tooltip="三种都按 unit 列整块切，绝不按行随机")
P_TEST_SIZE = am.ParamSpec("test_size", "测试集比例", "dspin", default=0.25, minimum=0.1,
                           maximum=0.5, shows_for=("cv", "holdout"))
P_MERGE_VOCAB = am.ParamSpec("merge_vocab", "先压缩连续同码", "check", default=True,
                             tooltip="压缩后词表更小、序列更短；不压缩则保留自转移（与 IND06 同口径）")
P_ARCHS = am.ParamSpec("archs", "网络结构", "checklist",
                       options=(("gru", "GRU"), ("lstm", "LSTM"), ("tcn", "TCN（膨胀卷积）")),
                       default=("gru", "tcn"), tooltip="选几个就训几遍，结果表里并列对比")
P_WINDOW = am.ParamSpec("window", "上下文窗口（步）", "spin", default=8, minimum=2, maximum=40,
                       tooltip="用前 window 步预测第 window+1 步；不足左侧 pad")
P_STRIDE = am.ParamSpec("stride", "滑窗步幅", "spin", default=1, minimum=1, maximum=10,
                        tooltip=">1 会丢样本，但窗口总数不爆")
P_STAGES = am.ParamSpec("stages", "TCN 膨胀层数", "spin", default=4, minimum=1, maximum=10,
                        tooltip="感受野 = 1 + 2·(kernel−1)·(2^stages−1)，写进结果表")
P_KERNEL = am.ParamSpec("kernel_size", "TCN 卷积核", "spin", default=2, minimum=2, maximum=7)
P_MAX_SAMPLES = am.ParamSpec("max_samples", "每条流最多样本数", "spin", default=4000, minimum=100,
                             maximum=50000, tooltip="超出部分按固定种子下采样，保证可复现")
P_OBJECTIVE = am.ParamSpec("objective", "预训练目标", "combo",
                           options=(("masked", "遮码预测（masked）"), ("ae", "序列自编码（AE）")),
                           default="masked",
                           tooltip="masked 考“能不能补回被遮的行为”；ae 考“能不能压住整条轨迹”")
P_MAX_LEN = am.ParamSpec("max_len", "序列最长位数（步）", "spin", default=MAX_SEQ_LEN, minimum=8,
                         maximum=1024, tooltip="位置编码表到此为限，超长序列截断")
P_MASK_RATIO = am.ParamSpec("mask_ratio", "遮码比例", "dspin", default=0.15, minimum=0.05,
                            maximum=0.8, tooltip="训练遮一批、评估另遮一批（不拿同位置自证）")
P_CONTAMINATION = am.ParamSpec("contamination", "预期异常比例", "dspin", default=0.1, minimum=0.01,
                               maximum=0.3, tooltip="折外 NLL 按此分位数划异常线")
P_BUCKETS = am.ParamSpec("buckets", "时间分桶数", "spin", default=6, minimum=1, maximum=24,
                         tooltip="组内时刻 min-max 归一后切桶，每桶一张邻接阵（时序图的“时”）")
P_CLASS_WEIGHT = am.ParamSpec("class_weight", "类别不平衡加权", "check", default=True,
                              tooltip="按 n/(k·n_c) 给训练样本加权，网络与地板同口径")
P_FAMILIES = am.ParamSpec("families", "特征族", "checklist",
                          options=am.FEATURE_FAMILY_CHOICES, default=(),
                          tooltip="留空 = 全部非常数数值特征（含网络类特征）；组内均值与标准差各一列")
P_MAX_FEATURES = am.ParamSpec("max_features", "最多进表特征数", "spin", default=40, minimum=1,
                              maximum=200, tooltip="组数少时特征越多越容易过拟合，先用这一刀")
P_POOL = am.ParamSpec("pool", "成员→组池化", "combo",
                      options=(("attention", "注意力池化"), ("mean", "均值池化"),
                               ("sum", "求和池化")), default="attention",
                      tooltip="DeepSets 用：变长成员集→定长组 embedding")
P_WIDTH = am.ParamSpec("width", "φ 网络宽度", "spin", default=64, minimum=8, maximum=256)
P_OUT_DIM = am.ParamSpec("out_dim", "组 embedding 维度", "spin", default=16, minimum=2, maximum=128)

SPECS = [
    am.MethodSpec(
        id="IND13", name="循环/卷积网络 next-state", group=am.GROUP_PREDICT,
        question="看过前几步之后，下一个行为码最可能是什么？网络比一阶 Markov 强吗？",
        mode=MODE_INDIVIDUAL, compute=next_state_analysis,
        params=(P_ARCHS, P_WINDOW, P_STRIDE, P_MERGE_VOCAB, P_CV, P_FOLDS, P_TEST_SIZE,
                P_EPOCHS, P_LR, P_BATCH, P_EMBED, P_HIDDEN, P_LAYERS, P_STAGES, P_KERNEL,
                P_MAX_SAMPLES, P_SEED),
        requires=("torch", "students>=8"), unit="student", engine=am.ENGINE_OPTIONAL,
        formula_note="窗口 = 前 window 步（不足左侧 pad），目标 = 下一步码；同一 student_uid 的"
                     "窗口全进全出；地板 = 多数类 / 一阶 Markov / IND12 特征 RF，与网络同一测试集"
                     "并列；指标口径完全同 IND12（log-loss 只在本折出现类上重归一）；"
                     "TCN 感受野 = 1 + 2·(kernel−1)·(2^stages−1)"),
    am.MethodSpec(
        id="IND14", name="Transformer 自监督表征", group=am.GROUP_PREDICT,
        question="自监督学出来的表征能把人分开吗？谁的轨迹最反常？",
        mode=MODE_INDIVIDUAL, compute=transformer_ssl_analysis,
        params=(P_OBJECTIVE, P_MASK_RATIO, P_MAX_LEN, P_MERGE_VOCAB, P_FOLDS, P_EPOCHS,
                P_LR, P_BATCH, P_EMBED, P_LAYERS, P_HEADS, P_CONTAMINATION, P_SEED),
        requires=("torch", "students>=8"), unit="student", engine=am.ENGINE_OPTIONAL,
        formula_note="masked：随机遮 mask_ratio 的位置预测原码（训练遮位与评估遮位错开）；"
                     "ae：整条序列自编码，NLL = 逐位重建损失；embedding = 序列位置池化向量，"
                     "PC1/PC2 = z 分数后 SVD 投影；异常分 = 折外 NLL 按 contamination 分位数划阈值；"
                     "探针分类只用折外 embedding，同一学生不跨折；地板 = “永远猜最常见码”"),
    am.MethodSpec(
        id="IND15", name="时序图注意力网络（GAT）", group=am.GROUP_PREDICT,
        question="把互动当成图来学：哪几条边真正在抬升组结果的预测？",
        mode=MODE_INDIVIDUAL, compute=temporal_gat_analysis,
        params=(P_BUCKETS, P_GAT_HIDDEN, P_HEADS, P_LAYERS, P_EPOCHS, P_LR_FAST,
                P_CLASS_WEIGHT, P_SEED),
        units=("groups", "dimensions"),
        requires=("torch", "person_edges", "outcome", "groups>=3"),
        unit="group", engine=am.ENGINE_OPTIONAL,
        formula_note="节点 = 组成员（特征同 GRP07 角色表的 z 分数），边 = 同组同时间桶内的"
                     "PERSON→PERSON 事件计数（自环剔除）；时间轴按组内 min-max 归一后切 buckets 桶；"
                     "读出 = 节点均值池化 → 组分类；LOYO 整组全进全出，地板 = 多数类 + "
                     "组均值特征 RF；注意力权重 = 各层各头均值的跨折平均"),
    am.MethodSpec(
        id="GRP10", name="跨组比较 / 组聚类 / 组结果预测", group=GROUP_PREDICT,
        question="组与组差在哪、能不能按组级特征聚类？成员行为又能预测组结果吗？",
        mode=MODE_GROUP, compute=group_prediction_analysis,
        params=(nt.P_LAG, P_FAMILIES, P_MAX_FEATURES, nt.P_CLUSTER_METHOD, nt.P_LINKAGE, nt.P_K,
                P_POOL, P_WIDTH, P_OUT_DIM, P_EPOCHS, P_LR_FAST, P_CLASS_WEIGHT, P_SEED),
        units=("groups", "dimensions"), requires=("groups>=3",),
        unit="group", engine=am.ENGINE_OPTIONAL,
        formula_note="跨组比较：逐特征 Kruskal–Wallis 全局 + Mann–Whitney 两两 + BH 校正"
                     "（效应量同 GRP06）；组聚类：z 分数化的组级特征→欧氏距离→AHC/PAM + silhouette；"
                     "组结果预测：LOYO 整组全进全出，特征地板（RF/HistGB/XGBoost）与 DeepSets"
                     "并列，再加多数类地板；未装 torch 时只跳过 DeepSets 分支，前两段照常输出"),
]


