## ---------------------------------------------------------------------------
## rena_network.R —— 组级网络（GRP09 / IND10 的 ENA 共现网 + ONA 有向时序网）
##
## 两张表：
##   edges        —— ONA 口径：逐 (组 × 维度) 把全组成员的**滞后转移**计数相加，
##                   自环不计（i==j 剔除）、权重为 0 的边不输出。列：
##                   group_id / dimension / source / target / weight
##   cooccurrence —— ENA 口径：grid.csv 上同一 (成员, 时点) 同时出现的**跨维度**编码对，
##                   按组合并计数。节点名写成 "维度显示名:码"（与 Python
##                   pooled_cooccurrence 一致，显示名来自 dimension_labels.csv 的
##                   key→label 映射）。列：group_id / node_a / node_b / weight
##   cooccurrence_member —— 同上口径但**不合并到组**，逐成员一张边表（IND10 的
##                   ENA 画像就是逐学生建的）。列：group_id / member / node_a /
##                   node_b / weight，member = grid.csv 的 uid（成员短名）
## 口径说明（为什么不用 rENA 的函数直接跑）：rENA 的 ena.build.from(transitions=...)
## 与本脚本是同一套计数规则，但它会自带节点排序/边合并细节，且不是每台机器都装了包；
## 交叉校验要的是「另一种实现算出同一张边表」，故 base R 手写，装了 rENA 时另发
## engine 标记并比对边数（不参与逐格对照）。
## ---------------------------------------------------------------------------

.ly_argv <- commandArgs(trailingOnly = TRUE)
source(file.path(if (length(.ly_argv) >= 2L) .ly_argv[2] else ".", "_common.R"),
       encoding = "UTF-8")

ly_init()

ly_guard(function() {
  lag <- max(1L, ly_int("lag", 1L))
  min_edge <- ly_int("min_edge", 0L)
  edge_rows <- list()

  data <- ly_sequences("codes")
  if (!is.null(data)) {
    keys <- paste(data$group_ids, data$dimensions, sep = "\t")
    buckets <- split(seq_along(keys), keys)
    for (key in names(buckets)) {
      index <- buckets[[key]]
      group_id <- data$group_ids[index[1]]
      dimension <- data$dimensions[index[1]]
      weights <- list()
      sides <- list()
      for (stream in data$streams[index]) {
        n <- length(stream)
        if (n <= lag) next
        for (i in seq_len(n - lag)) {
          left <- stream[i]
          right <- stream[i + lag]
          if (identical(left, right)) next
          pair <- paste(left, right, sep = "\t")
          weights[[pair]] <- (if (is.null(weights[[pair]])) 0 else weights[[pair]]) + 1
          sides[[pair]] <- c(left, right)
        }
      }
      for (pair in names(weights)) {
        value <- weights[[pair]]
        if (value < min_edge) next
        edge_rows[[length(edge_rows) + 1L]] <- ly_record(
          group_id = group_id, dimension = dimension,
          source = sides[[pair]][1], target = sides[[pair]][2], weight = value)
      }
    }
    LY$engine[["ona"]] <- "base-R lagged transition count"
  }

  ## ---- ENA：跨维度同时刻共现（grid 缺席就跳过这两张表）--------------------
  co_rows <- list()
  member_rows <- list()
  grid <- ly_file("grid")
  labels <- ly_file("dimension_labels")
  if (!is.null(grid) && all(c("group_id", "uid", "dimension", "position", "code")
                            %in% names(grid))) {
    lookup <- function(dimension) {
      if (is.null(labels)) return(dimension)
      hit <- which(as.character(labels$key) == dimension)
      if (!length(hit)) return(dimension)
      as.character(labels$label[hit[1]])
    }
    grid$group_id <- as.character(grid$group_id)
    grid$uid <- as.character(grid$uid)
    grid$dimension <- as.character(grid$dimension)
    grid$code <- trimws(as.character(grid$code))
    grid$code[is.na(grid$code)] <- ""
    grid$position <- as.integer(grid$position)
    dims <- ly_list("dimension_order")
    if (!length(dims)) dims <- sort(unique(grid$dimension))
    dims <- dims[dims %in% unique(grid$dimension)]
    grid <- grid[nzchar(grid$code) & grid$dimension %in% dims, , drop = FALSE]
    weights <- list()
    member_weights <- list()
    meta <- list()
    member_meta <- list()
    if (nrow(grid)) {
      cell_key <- paste(grid$group_id, grid$uid, grid$position, sep = "\t")
      cells <- split(seq_len(nrow(grid)), cell_key)
      for (name in names(cells)) {
        rows <- cells[[name]]
        if (length(rows) < 2L) next
        for (i in seq_along(rows)[-length(rows)]) {
          for (j in seq_along(rows)[-seq_len(i)]) {
            dim_a <- grid$dimension[rows[i]]
            dim_b <- grid$dimension[rows[j]]
            if (identical(dim_a, dim_b)) next   # 同维度同时刻不构成共现对
            left <- paste(dim_a, grid$code[rows[i]], sep = "\t")
            right <- paste(dim_b, grid$code[rows[j]], sep = "\t")
            rank_now <- order(c(left, right), method = "radix")
            if (rank_now[1] > rank_now[2]) {
              swap <- left
              left <- right
              right <- swap
            }
            key <- paste(grid$group_id[rows[i]], left, right, sep = "\t")
            weights[[key]] <- (if (is.null(weights[[key]])) 0 else weights[[key]]) + 1
            meta[[key]] <- list(group_id = grid$group_id[rows[i]], left = left,
                                right = right)
            mkey <- paste(grid$group_id[rows[i]], grid$uid[rows[i]], left, right, sep = "\t")
            member_weights[[mkey]] <- (if (is.null(member_weights[[mkey]])) 0
                                       else member_weights[[mkey]]) + 1
            member_meta[[mkey]] <- list(group_id = grid$group_id[rows[i]],
                                        member = grid$uid[rows[i]], left = left,
                                        right = right)
          }
        }
      }
    }
    pair_name <- function(item) {
      sides_a <- strsplit(item$left, "\t", fixed = TRUE)[[1]]
      sides_b <- strsplit(item$right, "\t", fixed = TRUE)[[1]]
      c(paste0(lookup(sides_a[1]), ":", sides_a[2]),
        paste0(lookup(sides_b[1]), ":", sides_b[2]))
    }
    for (key in names(weights)) {
      value <- weights[[key]]
      if (value < min_edge) next
      item <- meta[[key]]
      sides <- pair_name(item)
      co_rows[[length(co_rows) + 1L]] <- ly_record(
        group_id = item$group_id, node_a = sides[1], node_b = sides[2], weight = value)
    }
    for (key in names(member_weights)) {
      value <- member_weights[[key]]
      if (value < min_edge) next
      item <- member_meta[[key]]
      sides <- pair_name(item)
      member_rows[[length(member_rows) + 1L]] <- ly_record(
        group_id = item$group_id, member = item$member,
        node_a = sides[1], node_b = sides[2], weight = value)
    }
    LY$engine[["ena"]] <- "base-R cross-dimension co-occurrence"
  }

  if (!length(edge_rows) && !length(co_rows)) {
    LY$status <- "empty"
    LY$error <- "既没有时序转移也没有跨维共现，网络无从建起"
  }
  LY$tables[["edges"]] <- edge_rows
  if (length(co_rows)) LY$tables[["cooccurrence"]] <- co_rows
  if (length(member_rows)) LY$tables[["cooccurrence_member"]] <- member_rows
  LY$values[["lag"]] <- lag
  LY$values[["n_edges"]] <- length(edge_rows)
  LY$values[["n_cooccurrence"]] <- length(co_rows)
  LY$values[["n_cooccurrence_member"]] <- length(member_rows)

  ## rENA 在本表里只证「能加载」：滞后转移计数与跨维共现一律 base-R 自实现，
  ## engine 串靠 ly_loaded_only 写明未参与，不让一串版本号被误读成包已验。
  if (ly_has("rENA")) {
    LY$engine[["rena"]] <- ly_loaded_only("rENA")
  }
})
