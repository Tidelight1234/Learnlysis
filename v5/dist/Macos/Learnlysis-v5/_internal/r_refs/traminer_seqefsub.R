## ---------------------------------------------------------------------------
## traminer_seqefsub.R —— 频繁模式（IND04）
##
## 主表 `patterns` = **连续 n-gram**（与 Python 的 ngram_table 同径，逐 n 对照）：
##   1) 码流取 sequences.csv 的 stream 列（raw / compressed 由参数决定）；
##   2) 长度 < n 的序列整条退出，且**总数 total 也只数留下的序列**（support 分母）；
##   3) occurrences = 窗口出现次数；docs = 含该 gram 的序列条数；support = docs / total；
##   4) 过滤阈值：support + 1e-12 < min_sup 才丢弃（两侧同一浮点容差）；
##   5) gram 的连接符是 U+2192「→」，与 Python 完全一致，否则两侧键对不上。
## 装了 TraMineR 再附一张 `seqefsub`（**可不连续**的频繁子序列）作旁证：它对应的是
## Python 侧 PrefixSpan 的口径，但 PrefixSpan 有投影截断（PROJECT_CAP），故不参与逐格
## 对照，只用于人工核对「有没有明显漏掉的高频模式」。
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
  max_n <- max(2L, min(ly_int("max_n", 3L), 6L))
  min_sup <- ly_num("min_sup", 0.1)
  arrow <- intToUtf8(8594)   # →

  records <- list()
  for (n in 2L:max_n) {
    streams <- data$streams[vapply(data$streams, length, integer(1)) >= n]
    total <- length(streams)
    if (total == 0L) next
    occurrences <- list()
    docs <- list()
    for (position in seq_len(total)) {
      stream <- streams[[position]]
      starts <- seq_len(length(stream) - n + 1L)
      for (start in starts) {
        gram <- stream[seq(start, length.out = n)]
        key <- paste(gram, collapse = arrow)
        occurrences[[key]] <- (if (is.null(occurrences[[key]])) 0 else occurrences[[key]]) + 1
        docs[[key]] <- unique(c(docs[[key]], position))
      }
    }
    for (key in names(occurrences)) {
      doc_count <- length(docs[[key]])
      support <- doc_count / total
      if (support + 1e-12 < min_sup) next
      records[[length(records) + 1L]] <- ly_record(
        n = n, gram = key, occurrences = occurrences[[key]],
        docs = doc_count, support = support)
    }
  }

  if (!length(records)) {
    LY$status <- "empty"
    LY$error <- paste0("min_sup=", min_sup, " 下没有频繁 n-gram")
  }
  LY$tables[["patterns"]] <- records
  LY$engine[["ngram"]] <- "base-R sliding window"
  LY$values[["max_n"]] <- max_n
  LY$values[["min_sup"]] <- min_sup
  LY$values[["n_sequences"]] <- length(data$streams)

  ## TraMineR 2.x 把「频繁子序列」挪到了事件序列 API：seqefsub 只吃 seqecreate()
  ## 产出的 seqelist，参数名也换成 pmin.support / max.k。老写法
  ## seqefsub(seqdef(list), minsup=, maxn=) 在 2.x 直接报 unused argument，
  ## 以前被 tryCatch 咽掉了。这一支只是旁证（不参与逐格对照），成败一律写进
  ## engine 输出来——静默缺席比报错难查。
  if (ly_has("TraMineR")) {
    extra <- tryCatch({
      tse <- do.call(rbind, lapply(seq_along(data$streams), function(i) {
        stream <- data$streams[[i]]
        data.frame(case = i, pos = seq_along(stream), code = as.character(stream),
                   stringsAsFactors = FALSE)
      }))
      eseq <- TraMineR::seqecreate(tse, id = "case", timestamp = "pos", event = "code")
      found <- TraMineR::seqefsub(eseq, pmin.support = 100 * max(min_sup, 1e-6),
                                  max.k = max_n)
      if (is.null(found) || !nrow(found)) NULL else {
        LY$engine[["seqefsub"]] <- paste0("TraMineR::seqefsub ",
                                          utils::packageVersion("TraMineR"))
        lapply(seq_len(nrow(found)), function(row) ly_record(
          gram = as.character(found[[1]][row]),
          support = as.numeric(found$supp[row])))
      }
    }, error = function(e) {
      LY$engine[["seqefsub"]] <- paste0("旁证未取到（TraMineR ",
                                        utils::packageVersion("TraMineR"),
                                        " 的 seqecreate→seqefsub 路线）：",
                                        conditionMessage(e))
      NULL
    })
    if (!is.null(extra)) LY$tables[["seqefsub"]] <- extra
  }
})
