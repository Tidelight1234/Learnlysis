## ---------------------------------------------------------------------------
## traminer_om.R —— 单通道序列距离（IND05 / IND07 的 OM / HAM / DHD）
##
## 距离在**给定的码流**（sequences.csv 的 stream 列，已按参数决定是否压缩）上算，
## 但状态全集与替换代价一律从**原始码流**（codes 列）推出 —— 与 Python 侧
## build_distance_matrix 完全一致：
##   * states      = 全部入选序列原始码的并集（首次出现顺序，仅影响行列不影响数值）
##   * subst=constant → 非对角 cost_value、对角 0
##   * subst=trate    → 2 - P - Pᵀ（P 由原始码 lag=1 的行向概率构成），对角 0，截到 [0,2]
##   * subst=custom   → 解析 "A,B=1.5; A,C=2"（未给出的非对角格默认 1、对角 0）
##   * indel     = indel_ratio × max(cost)（max(cost)=0 时按 1 处理），再取 max(·, 1e-9)
##   * DHD       = 单位替换阵 + indel 1.0（与 TraMineR 的 DHD 同径：不受 indel_ratio 影响）
##   * HAM       = 逐时点相加，不等长只比公共前缀
## 装了 TraMineR 且 method=OM 时改走 `seqdist(method="OM")`（代价阵由本脚本自行
## 构造后喂进去，保证两侧只差在实现、不差在参数），否则用 base R 的 Needleman–Wunsch。
## ---------------------------------------------------------------------------

## ---- prolog：由 bridge 传入 <workdir> <refsdir>，共用件在 refsdir ----------
.ly_argv <- commandArgs(trailingOnly = TRUE)
source(file.path(if (length(.ly_argv) >= 2L) .ly_argv[2] else ".", "_common.R"),
       encoding = "UTF-8")

ly_init()

ly_guard(function() {
  raw <- ly_sequences("codes")
  if (is.null(raw)) {
    ly_input_missing("sequences")
    return(invisible(NULL))
  }
  aligned <- ly_sequences("stream")
  if (is.null(aligned)) aligned <- raw

  method <- toupper(trimws(ly_txt("method", "OM")))
  if (method %in% c("MCO", "MCM")) {
    LY$status <- "unsupported"
    LY$error <- paste0(method, " 属多通道距离，请改跑 traminer_multichannel.R")
    return(invisible(NULL))
  }
  subst <- trimws(ly_txt("subst", "constant"))
  cost_value <- ly_num("cost_value", 1)
  ratio <- ly_num("indel_ratio", 0.5)
  custom_text <- ly_txt("custom_cost", "")

  states <- character(0)
  for (stream in raw$streams) {
    for (code in stream) if (!(code %in% states)) states <- c(states, code)
  }
  k <- length(states)
  if (k < 2L) {
    LY$status <- "empty"
    LY$error <- "状态数不足 2，距离无意义"
  }

  index_of <- function(stream) match(stream, states)

  cost <- matrix(cost_value, k, k)
  if (subst == "trate") {
    counts <- matrix(0, k, k)
    for (stream in raw$streams) {
      n <- length(stream)
      if (n < 2L) next
      positions <- index_of(stream)
      for (i in seq_len(n - 1L)) counts[positions[i], positions[i + 1L]] <-
        counts[positions[i], positions[i + 1L]] + 1
    }
    rows <- rowSums(counts)
    probabilities <- matrix(0, k, k)
    for (i in seq_len(k)) if (rows[i] > 0) probabilities[i, ] <- counts[i, ] / rows[i]
    cost <- 2 - probabilities - t(probabilities)
    diag(cost) <- 0
    cost <- pmin(pmax(cost, 0), 2)
  } else if (subst == "custom") {
    cost <- matrix(1, k, k)
    diag(cost) <- 0
    text <- gsub("；", ";", gsub("，", ",", custom_text))
    for (chunk in strsplit(text, ";", fixed = TRUE)[[1]]) {
      if (!grepl("=", chunk, fixed = TRUE)) next
      pair <- strsplit(chunk, "=", fixed = TRUE)[[1]]
      if (length(pair) < 2L) next
      sides <- strsplit(pair[1], ",", fixed = TRUE)[[1]]
      sides <- trimws(sides)
      if (length(sides) != 2L) {
        ## 不是“两元逗号式”时才试 AB=1.5：仍只认**恰好两个**非空白字符（与 Python 同径）
        chars <- strsplit(gsub("[[:space:]]", "", pair[1], perl = TRUE), "", fixed = TRUE)[[1]]
        if (length(chars) != 2L) next
        sides <- chars
      }
      left <- sides[1]; right <- sides[length(sides)]
      value <- suppressWarnings(as.numeric(trimws(pair[length(pair)])))
      a <- match(left, states); b <- match(right, states)
      if (!is.na(a) && !is.na(b) && !is.na(value)) cost[a, b] <- cost[b, a] <- value
    }
  } else {
    diag(cost) <- 0
  }

  max_cost <- if (all(is.finite(cost))) max(cost) else 0
  indel <- ratio * (if (max_cost == 0) 1 else max_cost)
  indel <- max(indel, 1e-9)

  om_dp <- function(a, b, matrix_cost, step) {
    n <- length(a); m <- length(b)
    previous <- (0:m) * step
    for (i in seq_len(n)) {
      current <- numeric(m + 1L)
      current[1] <- i * step
      row <- matrix_cost[a[i], ]
      for (j in seq_len(m)) {
        current[j + 1L] <- min(previous[j] + row[b[j]],
                               previous[j + 1L] + step,
                               current[j] + step)
      }
      previous <- current
    }
    previous[m + 1L]
  }

  ham <- function(a, b, matrix_cost) {
    shared <- min(length(a), length(b))
    total <- 0
    if (shared > 0) {
      for (i in seq_len(shared)) total <- total + matrix_cost[a[i], b[i]]
    }
    total
  }

  count <- length(aligned$labels)
  labels <- aligned$labels
  records <- list()
  ## 引擎名要跟上实际算的那段代码：HAM/DHD 走的不是 Needleman-Wunsch
  engine <- paste0("base-R ", switch(method,
                                    HAM = "Hamming",
                                    DHD = "unit-cost OM (indel=1)",
                                    "Needleman-Wunsch"))

  use_traminer <- identical(method, "OM") && ly_has("TraMineR")
  matrix_om <- NULL
  if (use_traminer) {
    ## TraMineR 2.x 的两处坑（1.x 写法在这儿直接报错，以前被 tryCatch 咽掉了）：
    ##   * seqdef 不吃 list，必须是「一行一条序列」的矩阵，而矩阵要求等长；
    ##   * 替换代价矩阵的参数名从 smatrix 改成了 sm（两个名字都试，兼容旧版）。
    seq_lengths <- vapply(aligned$streams, length, integer(1))
    if (length(unique(seq_lengths)) != 1L) {
      engine <- paste0("base-R Needleman-Wunsch（序列长 ", min(seq_lengths), ",",
                       max(seq_lengths), " 不等长，TraMineR::seqdef 需等长矩阵）")
      use_traminer <- FALSE
    } else {
      storage <- cost
      dimnames(storage) <- list(states, states)
      matrix_om <- tryCatch({
        seqd <- TraMineR::seqdef(do.call(rbind, aligned$streams), alphabet = states,
                                 states = states, with.legend = FALSE)
        diss <- tryCatch(TraMineR::seqdist(seqd, method = "OM", indel = indel, sm = storage),
                         error = function(e) TraMineR::seqdist(seqd, method = "OM",
                                                               indel = indel,
                                                               smatrix = storage))
        as.matrix(diss)
      }, error = function(e) {
        ## 失败必须写在 engine 里露出来：静默回退 base-R 会让人以为 TraMineR 已验
        LY$engine[["traminer_error"]] <- substring(conditionMessage(e), 1L, 160L)
        NULL
      })
      if (is.null(matrix_om)) {
        engine <- "base-R Needleman-Wunsch（TraMineR 调用失败，见 traminer_error）"
        use_traminer <- FALSE
      } else {
        engine <- paste0("TraMineR::seqdist(OM) ", utils::packageVersion("TraMineR"))
      }
    }
  }

  if (count < 2L) {
    LY$status <- "empty"
    LY$error <- "有效序列不足 2 条，没有可对照的距离"
  } else {
    for (i in seq_len(count - 1L)) {
      for (j in seq_len(count)[-seq_len(i)]) {
        value <- if (use_traminer) {
          matrix_om[i, j]
        } else if (identical(method, "HAM")) {
          ham(index_of(aligned$streams[[i]]), index_of(aligned$streams[[j]]), cost)
        } else if (identical(method, "DHD")) {
          unit <- matrix(1, k, k)
          diag(unit) <- 0
          om_dp(index_of(aligned$streams[[i]]), index_of(aligned$streams[[j]]), unit, 1)
        } else {
          om_dp(index_of(aligned$streams[[i]]), index_of(aligned$streams[[j]]), cost, indel)
        }
        records[[length(records) + 1L]] <- ly_record(a = labels[i], b = labels[j],
                                                     d = as.numeric(value))
      }
    }
  }

  LY$engine[["distance"]] <- engine
  LY$values[["method"]] <- method
  LY$values[["subst"]] <- subst
  LY$values[["indel"]] <- if (identical(method, "DHD")) 1 else indel
  LY$values[["k_states"]] <- k
  LY$values[["cost_matrix"]] <- as.data.frame(cost, stringsAsFactors = FALSE)
  LY$tables[["distance_matrix"]] <- records
})
