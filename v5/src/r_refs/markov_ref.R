## ---------------------------------------------------------------------------
## markov_ref.R —— 一阶马尔可夫链（IND02）
##
## 复算：频数 N_ij、转移概率 P_ij、平稳分布 pi、转移熵率 / 初始熵 / 可预测性比值。
## 口径与 analysis_stats_sequence.markov_probabilities / stationary_distribution /
## markov_statistics 一一对应：
##   1) 计数只在单条序列内部滑窗（lag=1），不跨序列；
##   2) self_transition=FALSE 时先抹掉主对角线再算（P 的行合也随之变小）；
##   3) 整行为 0 的状态视为吸收态：该行 P 全 0；
##   4) 平稳分布 = 幂迭代，先加 1e-9 的均匀扰动保证不可约/非周期，tol=1e-12、上限 4000 步；
##   5) 熵一律 ln；转移熵率按行频次加权：Σ_i (row_i/total) * H(P_i·)。
## 初始分布 = 各序列**首个有效码**的计数（不是状态频次）。
## ---------------------------------------------------------------------------

## ---- prolog：由 bridge 传入 <workdir> <refsdir>，共用件在 refsdir ----------
.ly_argv <- commandArgs(trailingOnly = TRUE)
source(file.path(if (length(.ly_argv) >= 2L) .ly_argv[2] else ".", "_common.R"),
       encoding = "UTF-8")

ly_init()

ly_guard(function() {
  data <- ly_sequences("codes")
  if (is.null(data)) {
    ly_input_missing("sequences")
    return(invisible(NULL))
  }

  include_self <- ly_flag("self_transition", TRUE)
  LY$engine[["markov"]] <- "base-R power iteration"

  buckets <- split(seq_along(data$dimensions), data$dimensions)

  freq_rows <- list()
  prob_rows <- list()
  pi_rows <- list()
  value_rows <- list()

  for (key in names(buckets)) {
    index <- buckets[[key]]
    streams <- data$streams[index]
    dimension <- data$dimensions[index[1]]

    states <- character(0)
    for (stream in streams) {
      for (code in stream) if (!(code %in% states)) states <- c(states, code)
    }
    k <- length(states)
    if (k < 2L) next

    counts <- matrix(0, k, k)
    for (stream in streams) {
      n <- length(stream)
      if (n < 2L) next
      for (i in seq_len(n - 1L)) {
        counts[match(stream[i], states), match(stream[i + 1L], states)] <-
          counts[match(stream[i], states), match(stream[i + 1L], states)] + 1
      }
    }
    if (!include_self) diag(counts) <- 0

    rows <- rowSums(counts)
    total <- sum(counts)
    probabilities <- matrix(0, k, k)
    for (i in seq_len(k)) if (rows[i] > 0) probabilities[i, ] <- counts[i, ] / rows[i]

    ## 平稳分布：P*(1-1e-9) + 1e-9/k，行归一后从均匀分布开始幂迭代
    aperiodic <- probabilities * (1 - 1e-9) + 1e-9 / k
    row_totals <- rowSums(aperiodic)
    for (i in seq_len(k)) if (row_totals[i] > 0) aperiodic[i, ] <- aperiodic[i, ] / row_totals[i]
    vector <- rep(1 / k, k)
    for (step in seq_len(4000L)) {
      nxt <- as.vector(vector %*% aperiodic)
      converged <- max(abs(nxt - vector)) < 1e-12
      vector <- nxt
      if (converged) break
    }
    s <- sum(vector)
    pi <- if (s > 0) vector / s else rep(1 / k, k)

    entropy_rate <- 0
    if (total > 0) {
      for (i in seq_len(k)) {
        if (rows[i] <= 0) next
        inner <- 0
        for (j in seq_len(k)) {
          value <- probabilities[i, j]
          if (value > 0) inner <- inner - value * log(value)
        }
        entropy_rate <- entropy_rate + (rows[i] / total) * inner
      }
    }

    initial <- numeric(k)
    for (stream in streams) {
      if (!length(stream)) next
      hit <- match(stream[1], states)
      if (!is.na(hit)) initial[hit] <- initial[hit] + 1
    }
    initial_total <- sum(initial)
    initial_entropy <- 0
    if (initial_total > 0 && total > 0) {
      for (count in initial) if (count > 0) {
        p0 <- count / initial_total
        initial_entropy <- initial_entropy - p0 * log(p0)
      }
    }

    for (i in seq_len(k)) {
      for (j in seq_len(k)) {
        freq_rows[[length(freq_rows) + 1L]] <-
          ly_record(dimension = dimension, from = states[i], to = states[j],
                    n = counts[i, j])
        prob_rows[[length(prob_rows) + 1L]] <-
          ly_record(dimension = dimension, from = states[i], to = states[j],
                    p = probabilities[i, j])
      }
    }
    for (i in seq_len(k)) {
      pi_rows[[length(pi_rows) + 1L]] <-
        ly_record(dimension = dimension, state = states[i], pi = pi[i])
    }
    value_rows[[length(value_rows) + 1L]] <-
      ly_record(dimension = dimension, entropy_rate = entropy_rate,
                initial_entropy = initial_entropy,
                ratio = if (initial_entropy > 0) entropy_rate / initial_entropy else 0,
                transitions = total)
  }

  if (!length(freq_rows)) {
    LY$status <- "empty"
    LY$error <- "状态数不足 2，没有可对照的转移概率"
  }
  LY$tables[["markov_N"]] <- freq_rows
  LY$tables[["markov_P"]] <- prob_rows
  LY$tables[["markov_pi"]] <- pi_rows
  LY$tables[["markov_values"]] <- value_rows
  LY$values[["self_transition"]] <- include_self
})
