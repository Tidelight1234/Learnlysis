## ---------------------------------------------------------------------------
## seqhmm_fit.R —— HMM 推断参照（IND08）
##
## 校验对象是**推断引擎**（前向后向 / Viterbi / 对数似然），不是 EM 训练：EM 的初值
## 与收敛路径两侧不可能完全一致，比对训练结果只会得到假差异。因此约定：
##   输入  model_pi.csv(dimension, state, pi) / model_A.csv(dimension, from, to, p)
##         model_B.csv(dimension, channel, state, symbol, p)
##         sequences.csv 的 stream 列 = 观测序列（已按参数口径取好），dimension 列
##         决定**该序列用哪一套模型**（IND08 单通道是逐维度各拟一套）
##   算法  带缩放因子的前向后向：
##           alpha_1(s) = pi_s * b_s(o_1)，c_t = Σ_s alpha_t(s)，alpha 除以 c_t；
##           alpha_t(s) = Σ_s' alpha_{t-1}(s') A[s',s] * b_s(o_t) 再除 c_t；
##           beta_T(s) = 1；beta_t(s) = Σ_s' A[s,s'] b_s'(o_{t+1}) beta_{t+1}(s') / c_{t+1}；
##           logLik = Σ_t log c_t；gamma_t(s) = alpha_t(s) * beta_t(s)（已归一）；
##           Viterbi 用对数域 max-product 回溯。
##         三张参数表都是**行归一**语义：pi 总和 1、A 每行（起态）和 1、
##         B 每行（隐状态）在符号轴上和 1；channel 列只取 0（多通道不在参照范围）。
##   输出  hmm_loglik{label, loglik}、hmm_gamma{label, position, state, posterior}、
##         hmm_viterbi{label, position, state}
## Python 侧（stage 6）必须用同一套缩放与前向后向递推来出参照值。
## 装了 seqHMM 时另记 engine 版本（其 `HmmLL` 口径与上式等价，但不参与逐格对照）。
## ---------------------------------------------------------------------------

.ly_argv <- commandArgs(trailingOnly = TRUE)
source(file.path(if (length(.ly_argv) >= 2L) .ly_argv[2] else ".", "_common.R"),
       encoding = "UTF-8")

ly_init()

ly_guard(function() {
  data <- ly_sequences("stream")
  pi_table <- ly_file("model_pi")
  a_table <- ly_file("model_A")
  b_table <- ly_file("model_B")
  if (is.null(data) || is.null(pi_table) || is.null(a_table) || is.null(b_table)) {
    LY$status <- "no-input"
    LY$error <- "需要 sequences.csv 与 model_pi/model_A/model_B 三张参数表（由 IND08 拟合后导出）"
    return(invisible(NULL))
  }

  states <- character(0)
  symbols <- character(0)
  k <- 0L
  has_dim <- "dimension" %in% names(pi_table)
  pick_table <- function(table, dimension) {
    if (!has_dim || !("dimension" %in% names(table))) return(table)
    hit <- table[as.character(table$dimension) == dimension, , drop = FALSE]
    if (!nrow(hit)) return(NULL)
    hit
  }
  model_for <- function(dimension) {
    pt <- pick_table(pi_table, dimension)
    at <- pick_table(a_table, dimension)
    bt <- pick_table(b_table, dimension)
    if (is.null(pt) || is.null(at) || is.null(bt)) return(NULL)
    if ("channel" %in% names(bt)) {
      bt <- bt[is.na(as.numeric(bt$channel)) | as.numeric(bt$channel) == 0,
               , drop = FALSE]
    }
    sts <- unique(as.character(pt$state))
    syms <- unique(as.character(bt$symbol))
    if (length(sts) < 2L || !length(syms)) return(NULL)
    init <- numeric(length(sts))
    names(init) <- sts
    for (row in seq_len(nrow(pt))) {
      s <- as.character(pt$state[row])
      if (s %in% sts) init[s] <- as.numeric(pt$pi[row])
    }
    if (sum(init) <= 0) init <- rep(1 / length(sts), length(sts)) else init <- init / sum(init)
    a <- matrix(0, length(sts), length(sts), dimnames = list(sts, sts))
    for (row in seq_len(nrow(at))) {
      from <- as.character(at$from[row]); to <- as.character(at$to[row])
      if (from %in% sts && to %in% sts) a[from, to] <- as.numeric(at$p[row])
    }
    for (i in seq_along(sts)) if (sum(a[i, ]) > 0) a[i, ] <- a[i, ] / sum(a[i, ])
    b <- matrix(0, length(sts), length(syms), dimnames = list(sts, syms))
    for (row in seq_len(nrow(bt))) {
      state <- as.character(bt$state[row]); symbol <- as.character(bt$symbol[row])
      if (state %in% sts && symbol %in% syms) b[state, symbol] <- as.numeric(bt$p[row])
    }
    for (i in seq_along(sts)) if (sum(b[i, ]) > 0) b[i, ] <- b[i, ] / sum(b[i, ])
    list(states = sts, symbols = syms, pi = init, a = a, b = b)
  }
  first <- model_for(if (has_dim) as.character(pi_table$dimension[1]) else "")
  if (is.null(first)) {
    LY$status <- "empty"
    LY$error <- "隐状态数 < 2 或发射表为空"
    return(invisible(NULL))
  }
  states <- first$states
  symbols <- first$symbols
  k <- length(states)

  loglik_rows <- list()
  gamma_rows <- list()
  viterbi_rows <- list()

  for (position in seq_along(data$streams)) {
    dimension <- ""
    if (length(data$dimensions) >= position) dimension <- as.character(data$dimensions[position])
    model <- model_for(dimension)
    if (is.null(model)) next
    states <- model$states
    symbols <- model$symbols
    k <- length(states)
    pi <- model$pi
    a <- model$a
    b <- model$b
    stream <- data$streams[[position]]
    stream <- stream[stream %in% symbols]
    n <- length(stream)
    if (n == 0L) next
    observed <- match(stream, symbols)
    alpha <- matrix(0, n, k, dimnames = list(NULL, states))
    beta <- matrix(0, n, k, dimnames = list(NULL, states))
    scales <- numeric(n)

    alpha[1, ] <- pi * b[, observed[1]]
    scales[1] <- sum(alpha[1, ])
    if (scales[1] <= 0) {
      loglik_rows[[length(loglik_rows) + 1L]] <- ly_record(
        label = data$labels[position], loglik = -Inf)
      next
    }
    alpha[1, ] <- alpha[1, ] / scales[1]
    if (n > 1L) {
      for (t in 2:n) {
        alpha[t, ] <- as.vector(alpha[t - 1, ] %*% a) * b[, observed[t]]
        scales[t] <- sum(alpha[t, ])
        if (scales[t] > 0) alpha[t, ] <- alpha[t, ] / scales[t] else alpha[t, ] <- 0
      }
    }
    beta[n, ] <- 1
    if (n > 1L) {
      for (t in (n - 1):1) {
        beta[t, ] <- as.vector(a %*% (b[, observed[t + 1]] * beta[t + 1, ])) / max(scales[t + 1], 1e-300)
      }
    }
    loglik <- sum(log(pmax(scales, 1e-300)))
    loglik_rows[[length(loglik_rows) + 1L]] <- ly_record(label = data$labels[position],
                                                         loglik = loglik)

    ## Viterbi（对数域）
    delta <- matrix(-Inf, n, k)
    psi <- matrix(0L, n, k)
    delta[1, ] <- log(pmax(pi * b[, observed[1]], 1e-300))
    if (n > 1L) {
      for (t in 2:n) {
        for (j in seq_len(k)) {
          values <- delta[t - 1, ] + log(pmax(a[, j], 1e-300))
          psi[t, j] <- which.max(values)[1]
          delta[t, j] <- max(values)[1] + log(max(b[j, observed[t]], 1e-300))
        }
      }
    }
    path <- integer(n)
    path[n] <- which.max(delta[n, ])[1]
    if (n > 1L) {
      for (t in (n - 1):1) path[t] <- psi[t + 1, path[t + 1]]
    }

    for (t in seq_len(n)) {
      weights <- alpha[t, ] * beta[t, ]
      total <- sum(weights)
      for (s in seq_len(k)) {
        gamma_rows[[length(gamma_rows) + 1L]] <- ly_record(
          label = data$labels[position], position = t - 1L, state = states[s],
          posterior = if (total > 0) weights[s] / total else 0)
      }
      viterbi_rows[[length(viterbi_rows) + 1L]] <- ly_record(
        label = data$labels[position], position = t - 1L, state = states[path[t]])
    }
  }

  if (!length(loglik_rows)) {
    LY$status <- "empty"
    LY$error <- "观测序列与发射矩阵的取值域没有交集"
  }
  LY$tables[["hmm_loglik"]] <- loglik_rows
  LY$tables[["hmm_gamma"]] <- gamma_rows
  LY$tables[["hmm_viterbi"]] <- viterbi_rows
  LY$engine[["hmm"]] <- "base-R scaled forward-backward"
  LY$values[["n_states"]] <- k
  if (has_dim) {
    LY$values[["n_models"]] <- length(unique(as.character(pi_table$dimension)))
  } else {
    LY$values[["n_models"]] <- 1L
  }
  ## seqHMM 只证「能加载」：本表推断路走上面的 base-R scaled forward-backward，
  ## EM 的初值与收敛路径两侧本来就不可能一致（见文件头），真调包也对照不出东西。
  if (ly_has("seqHMM")) {
    LY$engine[["seqHMM"]] <- ly_loaded_only("seqHMM")
  }
})
