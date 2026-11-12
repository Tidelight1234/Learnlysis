## ---------------------------------------------------------------------------
## dtwclust_ref.R —— 动态时间规整距离（IND06）
##
## 这是 IND06 的**参考契约**：Python 侧 stage 6 实现 dtw_distance 时必须与这里同径，
## 否则交叉校验没有意义。约定：
##   * 每条序列 → **逐时点** one-hot（行 = 时点，列 = 全体出现的状态，按首次出现顺序；
##     存成行主序展平的长向量，第 i 个时点占第 (i-1)*k+1 … i*k 格）；
##   * 局部代价 = 两个 one-hot 向量的欧氏距离（同码 0，异码 √2）；
##   * 经典 DTW：全窗口（不约束），步集 {(1,0),(0,1),(1,1)}，
##     D[1,1]=c(1,1)，其余 D[i,j]=c(i,j)+min(D[i-1,j], D[i,j-1], D[i-1,j-1])（越界视为 Inf），
##     最终**不做长度归一化**（等价于第一行/列取前缀累加）；
##   * soft-DTW：把上式的 min 换成 soft-min −γ·log Σ exp(−v/γ)（参数 soft_gamma，
##     默认 1.0，恒 ≤ 硬 min，γ→0 时收回经典 DTW），另出 soft_distance_matrix；
##   * 标签 = uid / dimension（与 build_distance_matrix 的单通道标签一致）。
## 装了 dtwclust 只用于记录版本（engine），距离一律本脚本自算：dtwclust 的默认
## `preprocess` / `norm` 与本契约不同，硬套会引入与 Python 无关的差。
## ---------------------------------------------------------------------------

.ly_argv <- commandArgs(trailingOnly = TRUE)
source(file.path(if (length(.ly_argv) >= 2L) .ly_argv[2] else ".", "_common.R"),
       encoding = "UTF-8")

ly_init()

ly_guard(function() {
  data <- ly_sequences("stream")
  if (is.null(data)) {
    ly_input_missing("sequences")
    return(invisible(NULL))
  }
  gamma <- ly_num("soft_gamma", 1)
  want_soft <- ly_flag("soft_dtw", FALSE)
  ## method = SDTW 时主表 `distance_matrix` 本身就是 soft 距离（IND07 选 SDTW 走这里）
  is_soft <- identical(toupper(ly_txt("method", "DTW")), "SDTW")

  states <- character(0)
  for (stream in data$streams) {
    for (code in stream) if (!(code %in% states)) states <- c(states, code)
  }
  k <- length(states)
  count <- length(data$labels)
  if (k < 2L || count < 2L) {
    LY$status <- "empty"
    LY$error <- "状态或序列不足，DTW 无从对照"
    return(invisible(NULL))
  }

  onehot <- lapply(data$streams, function(stream) {
    if (!length(stream)) return(numeric(0))
    hits <- match(stream, states)
    binary <- matrix(0, nrow = length(stream), ncol = k)
    for (i in seq_along(hits)) if (!is.na(hits[i])) binary[i, hits[i]] <- 1
    as.vector(t(binary))          # 行主序：每个时点占 k 格，DTW 随时点滑动
  })

  dtw_pair <- function(x, y, soft = FALSE) {
    n <- length(x) / k
    m <- length(y) / k
    stopifnot(n * k == length(x), m * k == length(y))
    if (n < 1L || m < 1L) return(NaN)
    cells <- matrix(NA_real_, n, m)
    cost <- function(i, j) {
      left <- x[seq((i - 1L) * k + 1L, i * k)]
      right <- y[seq((j - 1L) * k + 1L, j * k)]
      sqrt(sum((left - right)^2))
    }
    pool <- function(values) {
      values <- values[is.finite(values)]
      if (!length(values)) return(Inf)
      if (!soft) return(min(values))
      scaled <- -values / gamma
      highest <- max(scaled)
      -gamma * (highest + log(sum(exp(scaled - highest))))
    }
    for (i in seq_len(n)) {
      for (j in seq_len(m)) {
        if (i == 1L && j == 1L) {
          cells[i, j] <- cost(i, j)
          next
        }
        up <- if (i > 1L) cells[i - 1L, j] else Inf
        left <- if (j > 1L) cells[i, j - 1L] else Inf
        diagonal <- if (i > 1L && j > 1L) cells[i - 1L, j - 1L] else Inf
        cells[i, j] <- cost(i, j) + pool(c(diagonal, up, left))
      }
    }
    cells[n, m]
  }

  plain <- list()
  softened <- list()
  for (i in seq_len(count - 1L)) {
    for (j in seq_len(count)[-seq_len(i)]) {
      plain[[length(plain) + 1L]] <- ly_record(
        a = data$labels[i], b = data$labels[j],
        d = as.numeric(dtw_pair(onehot[[i]], onehot[[j]], soft = is_soft)))
      if (want_soft && !is_soft) {
        softened[[length(softened) + 1L]] <- ly_record(
          a = data$labels[i], b = data$labels[j],
          d = as.numeric(dtw_pair(onehot[[i]], onehot[[j]], soft = TRUE)))
      }
    }
  }

  LY$tables[["distance_matrix"]] <- plain
  if (want_soft && !is_soft && length(softened)) {
    LY$tables[["soft_distance_matrix"]] <- softened
  }
  LY$engine[["dtw"]] <- if (is_soft) {
    "base-R soft-DTW (one-hot per time point, euclidean, no normalisation)"
  } else {
    "base-R DTW (one-hot per time point, euclidean, no normalisation)"
  }
  LY$values[["k_states"]] <- k
  LY$values[["soft_gamma"]] <- gamma
  LY$values[["soft_dtw"]] <- want_soft || is_soft
  ## dtwclust 只证「能加载」：本表按上面的 R 参照契约算（one-hot 逐时点 + 不做长度归一化），
  ## 包自己的默认归一化与局部代价与本契约不一致，直接调它只会造出假差异。
  if (ly_has("dtwclust")) {
    LY$engine[["dtwclust"]] <- ly_loaded_only("dtwclust")
  }
})
