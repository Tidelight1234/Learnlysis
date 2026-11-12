## ---------------------------------------------------------------------------
## traminer_multichannel.R —— 多通道序列距离（IND05 的 MCO / MCM）
##
## 输入是 grid.csv（group_id / uid / dimension / position / code，空码写空串）：
## 它由 Python 侧的 aligned_channels 产出，属于**数据整形**（把各维度码轨铺到同一
## 时间栅格上），被校验的算法是下面的距离本身。通道顺序由参数 channel_order 给定。
##
##   MCO（多通道 CAT / 联合状态 OM）：
##     每通道去掉空码（可选再合并连续同码）；
##     pair_cost(i,j) = 各通道代价均值，通道缺项（一方已到末尾）不计入分子也不计入分母？
##       —— 与 Python 同径：只在**两边都有码**时才贡献 0/1，分母恒为通道数；
##     DP = Needleman–Wunsch，indel = max(indel_ratio, 1e-9)。
##   MCM（序列索引 MD）：
##     先把每通道编码成 (通道序号, 码, 起始时点, run 长) 的事件流并按 (时刻, 通道) 排序；
##     代价值：同通道同状态 0；同通道异状态 2/k；异通道 1/k + 1/k'（k = 该码出现的通道数，
##     由 codebook.csv 推出，网格里出现但 codebook 没有的码按 k=1 计入）；indel = 1。
##     即 TraMineR `seqMD(..., with.durations=FALSE)` 的口径（不加时权）。
## 标签：多通道距离的对象是**学生**，所以 a/b 写 uid（与 Python build_distance_matrix
## 在 MCO/MCM 分支返回的标签一致），不是 "uid / dimension"。
## ---------------------------------------------------------------------------

.ly_argv <- commandArgs(trailingOnly = TRUE)
source(file.path(if (length(.ly_argv) >= 2L) .ly_argv[2] else ".", "_common.R"),
       encoding = "UTF-8")

ly_init()

ly_guard(function() {
  grid <- ly_file("grid")
  if (is.null(grid) || !all(c("uid", "dimension", "position", "code") %in% names(grid))) {
    LY$status <- "no-input"
    LY$error <- "多通道距离需要 grid.csv（同一时间栅格上的各维度码轨）"
    return(invisible(NULL))
  }
  method <- toupper(trimws(ly_txt("method", "MCO")))
  compress <- ly_flag("compress", TRUE)
  ratio <- ly_num("indel_ratio", 0.5)
  channel_order <- ly_list("channel_order")
  grid$uid <- as.character(grid$uid)
  grid$dimension <- as.character(grid$dimension)
  grid$code <- trimws(as.character(grid$code))
  grid$code[is.na(grid$code)] <- ""
  grid$position <- as.integer(grid$position)

  uids <- sort(unique(grid$uid))
  if (!length(channel_order)) {
    channel_order <- sort(unique(grid$dimension))
  }
  channel_order <- channel_order[channel_order %in% unique(grid$dimension)]
  if (length(channel_order) < 2L) {
    LY$status <- "empty"
    LY$error <- "多通道距离需要 ≥ 2 个通道（维度）"
    return(invisible(NULL))
  }

  ## ---- MCO：每 uid 每通道一条码流 -----------------------------------------
  channel_stream <- function(uid, dimension) {
    part <- grid[grid$uid == uid & grid$dimension == dimension, , drop = FALSE]
    part <- part[order(part$position, method = "radix"), , drop = FALSE]
    values <- part$code[nzchar(part$code)]
    if (compress && length(values)) {
      keep <- c(TRUE, values[-length(values)] != values[-1])
      values <- values[keep]
    }
    values
  }

  mco_dp <- function(a_channels, b_channels, step) {
    channels <- min(length(a_channels), length(b_channels))
    if (channels == 0) return(0)
    left <- a_channels[seq_len(channels)]
    right <- b_channels[seq_len(channels)]
    n <- max(lengths(left))
    m <- max(lengths(right))
    ## i / j 是**0 基**下标（与 Python 的 pair_cost 同形）：R 取元素时才 +1
    pair_cost <- function(i, j) {
      acc <- 0
      for (channel in seq_len(channels)) {
        if (i < length(left[[channel]]) && j < length(right[[channel]])) {
          acc <- acc + if (identical(left[[channel]][i + 1L], right[[channel]][j + 1L])) 0 else 1
        }
      }
      acc / channels
    }
    previous <- (0:m) * step
    for (i in seq_len(n)) {
      current <- numeric(m + 1L)
      current[1] <- i * step
      for (j in seq_len(m)) {
        current[j + 1L] <- min(previous[j] + pair_cost(i - 1L, j - 1L),
                               previous[j + 1L] + step,
                               current[j] + step)
      }
      previous <- current
    }
    previous[m + 1L]
  }

  ## ---- MCM：序列索引 + 索引距离 -------------------------------------------
  index_of_uid <- function(uid) {
    elements <- list()
    for (position in seq_along(channel_order)) {
      dimension <- channel_order[position]
      part <- grid[grid$uid == uid & grid$dimension == dimension, , drop = FALSE]
      part <- part[order(part$position, method = "radix"), , drop = FALSE]
      codes <- part$code
      times <- part$position
      run_code <- NA_character_
      run_start <- 0
      flush <- function(step) {
        if (!is.na(run_code) && nzchar(run_code)) {
          elements[[length(elements) + 1L]] <<- list(
            moment = if (run_start <= length(times)) times[run_start] else run_start,
            channel = position, code = run_code, length = step - run_start)
        }
      }
      if (!length(codes)) next
      for (step in seq_len(length(codes) + 1L)) {
        code <- if (step <= length(codes)) codes[step] else NA_character_
        if (!is.na(code) && identical(code, run_code)) next
        flush(step)
        run_code <- code
        run_start <- step
      }
    }
    if (!length(elements)) return(NULL)
    order_now <- order(vapply(elements, function(e) as.numeric(e$moment), numeric(1)),
                       vapply(elements, function(e) e$channel, numeric(1)), method = "radix")
    lapply(elements[order_now], function(e) list(e$channel, e$code))
  }

  ## k = 该码在 codebook 里跨几个通道出现（与 Python 的 channels_of 同径）
  channels_of <- list()
  codebook <- ly_file("codebook")
  if (!is.null(codebook)) {
    pairs <- data.frame(dimension = as.character(codebook$dimension),
                        code = trimws(as.character(codebook$code)),
                        stringsAsFactors = FALSE)
    pairs <- pairs[pairs$dimension %in% channel_order & nzchar(pairs$code), , drop = FALSE]
    for (row in seq_len(nrow(pairs))) {
      code <- pairs$code[row]
      if (is.null(channels_of[[code]])) channels_of[[code]] <- 0L
      channels_of[[code]] <- channels_of[[code]] + 1L
    }
  }
  register_code <- function(code) {
    if (is.null(channels_of[[code]])) channels_of[[code]] <<- 1L
    invisible(TRUE)
  }
  channel_count <- function(code) {
    value <- channels_of[[code]]
    if (is.null(value) || value < 1L) 1L else as.integer(value)
  }

  md_distance <- function(a, b) {
    if (is.null(a) || is.null(b)) return(0)
    n <- length(a); m <- length(b)
    cost <- function(i, j) {
      ca <- a[[i]]
      cb <- b[[j]]
      if (identical(ca[[1]], cb[[1]])) {
        if (identical(ca[[2]], cb[[2]])) return(0)
        return(2 / max(channel_count(ca[[2]]), 1))
      }
      1 / max(channel_count(ca[[2]]), 1) + 1 / max(channel_count(cb[[2]]), 1)
    }
    previous <- as.numeric(0:m)
    for (i in seq_len(n)) {
      current <- numeric(m + 1L)
      current[1] <- as.numeric(i)
      for (j in seq_len(m)) {
        current[j + 1L] <- min(previous[j] + cost(i, j), previous[j + 1L] + 1, current[j] + 1)
      }
      previous <- current
    }
    previous[m + 1L]
  }

  records <- list()
  engine <- if (identical(method, "MCM")) {
    "base-R MD DP (sequence index)"
  } else {
    "base-R multichannel NW DP"
  }
  if (length(uids) < 2L) {
    LY$status <- "empty"
    LY$error <- "多通道距离需要 ≥ 2 名学生的对齐码轨"
  } else if (identical(method, "MCM")) {
    indexes <- lapply(uids, index_of_uid)
    names(indexes) <- uids
    for (item in indexes) {
      if (is.null(item)) next
      for (element in item) register_code(element[[2]])
    }
    for (i in seq_along(uids)[-length(uids)]) {
      for (j in seq_along(uids)[-seq_len(i)]) {
        records[[length(records) + 1L]] <- ly_record(
          a = uids[i], b = uids[j], d = as.numeric(md_distance(indexes[[i]], indexes[[j]])))
      }
    }
  } else {
    step <- max(ratio, 1e-9)
    streams <- lapply(uids, function(uid) {
      lapply(channel_order, function(dimension) channel_stream(uid, dimension))
    })
    names(streams) <- uids
    for (i in seq_along(uids)[-length(uids)]) {
      for (j in seq_along(uids)[-seq_len(i)]) {
        records[[length(records) + 1L]] <- ly_record(
          a = uids[i], b = uids[j],
          d = as.numeric(mco_dp(streams[[uids[i]]], streams[[uids[j]]], step)))
      }
    }
  }

  LY$engine[["multichannel"]] <- engine
  LY$values[["method"]] <- method
  LY$values[["channels"]] <- paste(channel_order, collapse = ",")
  LY$tables[["distance_matrix"]] <- records
})
