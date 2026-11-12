## ---------------------------------------------------------------------------
## lsa_residual.R —— 滞后序列分析（IND01 同维 / GRP05「成员合并」口径）
##
## 复算 Sackett 调整残差的三件套：去对角观测 O、期望 E、残差 Z。
## 口径（与 Python 侧 analysis_stats_sequence.lsa_residuals 逐条对应）：
##   1) 自转移不进检验：先 O_ii = 0；
##   2) 行列合与总数 N 都在**去对角之后**的矩阵上取；
##   3) E_ij = row_i * col_j / N；
##   4) Z_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - row_i/N) * (1 - col_j/N))，分母为 0 记 0；
##   5) 转移计数只在**单条序列内部**滑窗，绝不跨序列、绝不跨组。
## 实现是 base R（另加 TraMineR 旁证），目的是给出「另一种语言、另一份代码」的对照。
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

  aggregate <- ly_txt("aggregate", "dimension")
  ## lag 上限跟 Python 一致：IND01 到 5，GRP05（组×维度）到 3
  cap <- if (identical(aggregate, "group+dimension")) 3L else 5L
  max_lag <- max(1L, min(ly_int("max_lag", if (cap == 3L) 1L else 3L), cap))
  LY$engine[["lsa"]] <- "base-R Sackett adjusted residual"

  ## 分桶：IND01 只按维度（同维口径跨组不合并序列，桶内仍按序列各自滑窗）；
  ## GRP05 按「组 × 维度」，把全组成员的转移计数相加。
  keys <- if (identical(aggregate, "group+dimension")) {
    paste(data$group_ids, data$dimensions, sep = "\r")
  } else {
    data$dimensions
  }
  buckets <- split(seq_along(keys), keys)

  counts_of <- function(streams, states, lag) {
    size <- length(states)
    matrix(numeric(size * size), nrow = size, ncol = size) -> empty
    for (stream in streams) {
      n <- length(stream)
      if (n <= lag) next
      for (i in seq_len(n - lag)) {
        a <- match(stream[i], states)
        b <- match(stream[i + lag], states)
        empty[a, b] <- empty[a, b] + 1
      }
    }
    empty
  }

  records <- list()
  for (key in names(buckets)) {
    index <- buckets[[key]]
    streams <- data$streams[index]
    states <- character(0)
    for (stream in streams) {
      for (code in stream) if (!(code %in% states)) states <- c(states, code)
    }
    if (length(states) < 2L) next
    for (lag in seq_len(max_lag)) {
      counts <- counts_of(streams, states, lag)
      off <- counts
      diag(off) <- 0
      total <- sum(off)
      k <- length(states)
      residuals <- matrix(0, k, k)
      expected <- matrix(0, k, k)
      if (total > 0) {
        rows <- rowSums(off)
        cols <- colSums(off)
        expected <- outer(rows, cols) / total
        factor <- outer(1 - rows / total, 1 - cols / total)
        denom <- sqrt(pmax(expected * factor, 0))
        residuals <- ifelse(denom > 0, (off - expected) / denom, 0)
        diag(residuals) <- 0
      }
      for (i in seq_len(k)) {
        for (j in seq_len(k)) {
          if (i == j) next
          record <- if (identical(aggregate, "group+dimension")) {
            list(group_id = data$group_ids[index[1]], dimension = data$dimensions[index[1]])
          } else {
            list(dimension = data$dimensions[index[1]])
          }
          record$lag <- lag
          record$from <- states[i]
          record$to <- states[j]
          record$O <- off[i, j]
          record$E <- expected[i, j]
          record$Z <- residuals[i, j]
          records[[length(records) + 1L]] <- record
        }
      }
    }
  }

  if (!length(records)) {
    LY$status <- "empty"
    LY$error <- "没有可检验的转移：状态数 < 2 或序列太短"
  }
  LY$tables[["lsa_cells"]] <- records
  LY$values[["n_cells"]] <- length(records)
  LY$values[["max_lag"]] <- max_lag
  LY$values[["aggregate"]] <- aggregate

  ## 立场：LSA adjusted residual 由上面的 base-R 实现逐格对照；TraMineR 只记版本。
  ## 它自己的滞后序列分析口径与本项目不同构（没有逐格可比性），本表的数字里没有它的成分。
  if (ly_has("TraMineR")) {
    LY$engine[["traminer"]] <- ly_loaded_only("TraMineR")
  }
})
