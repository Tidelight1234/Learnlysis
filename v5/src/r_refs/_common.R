## ---------------------------------------------------------------------------
## Learnlysis R 参考层共用件（**只依赖 base R**，装不上任何包也能跑）
##
## 调用方式由 analysis_r_bridge.run_reference() 固定：
##     Rscript --vanilla <refsdir>/<脚本>.R <workdir> <refsdir>
##
## workdir 里可能有（缺即表示本次不需要）：
##   sequences.csv   label / group_id / uid / student / dimension / codes / stream / length
##                   codes = 去缺码后的原始码流（`|` 分隔）
##                   stream = 按参数口径（raw 或 compressed）的码流
##   grid.csv        group_id / uid / dimension / position / code —— 时间栅格上的码
##                   （含空码，用于多通道与跨维共现，是**数据整形**不是算法）
##   codebook.csv    dimension / code —— 维度编码框架
##   events.csv      交互事件表
##   params.tsv      key\tvalue
##   payload.json    同样的载荷与参数（装了 jsonlite 才用得上，当前脚本一律读 CSV）
##
## 脚本把结果写回 workdir/result.json：
##   {"status","error","package","engine","values","tables","r_version"}
##   tables 的每个值都是「记录数组」，列名与 Python 侧对照表**必须一致**。
##
## 为什么很多脚本用 base R 手写而不是直接调包：交叉校验要的是「另一种实现」，
## base R 版在**任何** R 安装上都能跑；装了 TraMineR 的机器上脚本会自动改走包内
## 实现并在 engine 里写清楚用了哪条路，两条路都失败才回 status=failed。
## ---------------------------------------------------------------------------

ly_args <- function() commandArgs(trailingOnly = TRUE)

LY <- new.env()
LY$workdir <- getwd()
LY$refsdir <- "."
LY$params <- list()
LY$tables <- list()
LY$values <- list()
LY$engine <- list()
LY$status <- "ok"
LY$error <- ""
LY$package <- ""

ly_read_params <- function(path) {
  out <- list()
  if (!file.exists(path)) return(out)
  lines <- tryCatch(readLines(path, encoding = "UTF-8", warn = FALSE), error = function(e) character(0))
  if (length(lines) <= 1L) return(out)
  for (line in lines[-1L]) {
    parts <- strsplit(line, "\t", fixed = TRUE)[[1]]
    if (length(parts) >= 1L) {
      out[[parts[1]]] <- if (length(parts) >= 2L) paste(parts[-1], collapse = "\t") else ""
    }
  }
  out
}

ly_init <- function() {
  ## 排序一律按字节序：否则 sort(unique(uid)) 会跟着系统 locale 走，与 Python 的
  ## sorted()（码点序）在不同机器上给出不同的行列顺序，对照表就会整片错位。
  suppressWarnings(Sys.setlocale("LC_COLLATE", "C"))
  args <- ly_args()
  if (length(args) >= 1L && nzchar(args[1])) LY$workdir <- args[1]
  if (length(args) >= 2L && nzchar(args[2])) LY$refsdir <- args[2]
  LY$params <- ly_read_params(file.path(LY$workdir, "params.tsv"))
  invisible(TRUE)
}

## ---- 参数读取 --------------------------------------------------------------

ly_txt <- function(key, default = "") {
  value <- LY$params[[key]]
  if (is.null(value) || is.na(value)) return(default)
  as.character(value)
}

ly_num <- function(key, default = 0) {
  text <- ly_txt(key, "")
  if (!nzchar(text)) return(default)
  value <- suppressWarnings(as.numeric(text))
  if (is.na(value)) default else value
}

ly_int <- function(key, default = 0L) as.integer(round(ly_num(key, default)))

ly_flag <- function(key, default = FALSE) {
  text <- tolower(trimws(ly_txt(key, if (default) "TRUE" else "FALSE")))
  text %in% c("true", "1", "yes", "on")
}

ly_list <- function(key) {
  text <- trimws(ly_txt(key, ""))
  if (!nzchar(text)) return(character(0))
  trimws(strsplit(text, ",", fixed = TRUE)[[1]])
}

## ---- 输入表 ---------------------------------------------------------------

ly_file <- function(name) {
  path <- file.path(LY$workdir, paste0(name, ".csv"))
  if (!file.exists(path)) return(NULL)
  frame <- tryCatch(utils::read.csv(path, stringsAsFactors = FALSE, fileEncoding = "UTF-8",
                                    check.names = FALSE),
                    error = function(e) NULL)
  if (is.null(frame) || !nrow(frame)) return(NULL)
  frame
}

ly_sequences <- function(column = "codes") {
  frame <- ly_file("sequences")
  if (is.null(frame)) return(NULL)
  if (!(column %in% names(frame))) column <- "codes"
  list(labels = as.character(frame[["label"]]),
       group_ids = as.character(frame[["group_id"]]),
       uids = as.character(frame[["uid"]]),
       dimensions = as.character(frame[["dimension"]]),
       lengths = as.integer(frame[["length"]]),
       streams = ly_split_column(frame[[column]]))
}

ly_split_column <- function(values) {
  lapply(as.character(values), function(text) {
    if (!nzchar(text)) return(character(0))
    strsplit(text, "|", fixed = TRUE)[[1]]
  })
}

## ---- 极简 JSON 序列化（不依赖 jsonlite）-----------------------------------

ly_json_str <- function(s) {
  if (is.na(s)) return("null")
  s <- enc2utf8(as.character(s))
  chars <- strsplit(s, "", fixed = TRUE)[[1]]
  out <- character(length(chars))
  for (i in seq_along(chars)) {
    ch <- chars[i]
    cp <- tryCatch(utf8ToInt(ch), error = function(e) NA_integer_)
    if (is.na(cp)) { out[i] <- "\\"; next }
    cp <- as.integer(cp)
    out[i] <- if (cp == 34L) "\\\""
      else if (cp == 92L) "\\\\"
      else if (cp < 32L) sprintf("\\u%04x", cp)
      else if (cp < 128L) ch
      else if (cp <= 65535L) sprintf("\\u%04x", cp)
      else {
        v <- cp - 65536L
        sprintf("\\u%04x\\u%04x", bitwOr(bitwShiftR(v, 10), 55296L),
                bitwOr(bitwAnd(v, 1023L), 56320L))
      }
  }
  paste0("\"", paste(out, collapse = ""), "\"")
}

ly_json <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    nms <- names(x)
    if (length(x) && !is.null(nms) && all(nzchar(nms))) {
      body <- paste(sprintf("%s:%s", vapply(nms, ly_json_str, character(1)),
                            vapply(x, ly_json, character(1))), collapse = ",")
      return(paste0("{", body, "}"))
    }
    if (!length(x)) return("[]")
    return(paste0("[", paste(vapply(x, ly_json, character(1)), collapse = ","), "]"))
  }
  if (is.data.frame(x)) {
    if (!nrow(x)) return("[]")
    cols <- names(x)
    records <- lapply(seq_len(nrow(x)), function(i) {
      paste0("{", paste(sprintf("%s:%s", vapply(cols, ly_json_str, character(1)),
                                vapply(cols, function(k) ly_json(x[[k]][i]), character(1))),
                        collapse = ","), "}")
    })
    return(paste0("[", paste(unlist(records), collapse = ","), "]"))
  }
  if (length(x) > 1L) {
    return(paste0("[", paste(vapply(as.list(x), ly_json, character(1)), collapse = ","), "]"))
  }
  if (!length(x)) return("null")
  if (is.na(x)) return("null")
  if (is.logical(x)) return(if (isTRUE(x)) "true" else "false")
  if (is.numeric(x)) {
    if (!is.finite(x)) return("null")
    return(sprintf("%.17g", as.numeric(x)))
  }
  ly_json_str(as.character(x))
}

ly_record <- function(...) {
  fields <- list(...)
  fields
}

ly_write <- function() {
  payload <- list(status = LY$status, error = LY$error, package = LY$package,
                  engine = LY$engine, values = LY$values,
                  r_version = paste0(R.version$major, ".", R.version$minor),
                  tables = LY$tables)
  text <- ly_json(payload)
  path <- file.path(LY$workdir, "result.json")
  con <- tryCatch(file(path, open = "wb"), error = function(e) NULL)
  if (is.null(con)) return(invisible(FALSE))
  on.exit(close(con))
  writeBin(charToRaw(text), con)
  invisible(TRUE)
}

## ---- 状态与降级 -----------------------------------------------------------

ly_has <- function(pkg) {
  isTRUE(tryCatch(requireNamespace(pkg, quietly = TRUE), error = function(e) FALSE))
}

ly_loaded_only <- function(pkg) {
  ## 只记录「包在场」，本表仍由 base-R 实现算：engine 必须把这层意思写出来。
  ## 否则质检卡上一串版本号很容易被读成「该包已验」——判断依据只有 engine 串。
  paste0(tryCatch(as.character(utils::packageVersion(pkg)),
                  error = function(e) "unknown"), "（已加载，未参与本表计算）")
}

ly_require <- function(pkg) {
  if (ly_has(pkg)) return(invisible(TRUE))
  LY$package <- pkg
  stop(paste0("LY-PACKAGE-MISSING:", pkg), call. = FALSE)
}

ly_guard <- function(body) {
  ## body 以**函数**传入：脚本里可以正常 return() 提前收工，收尾一律写 result.json
  ## 调用链用 withCallingHandlers （内层）先拿到，再由外层 tryCatch 接手：
  ## 否则 stack 已展开，只剩 handler 自己那一帧。结果只打 stderr，不影响回传结构。
  trace <- character(0)
  collect <- function(condition) {
    frames <- tryCatch(as.list(sys.calls()), error = function(ignore) list())
    if (length(frames) > 3L) {
      trace <<- utils::tail(vapply(frames, function(item)
        substr(paste(deparse(item, nlines = 1L), collapse = " "), 1L, 140L), character(1)), 10L)
    }
    NULL
  }
  catch <- function(condition) {
    message <- conditionMessage(condition)
    if (length(trace)) {
      message("LY-TRACEBACK: ", substr(paste(trace, collapse = " < "), 1L, 1500L))
    }
    if (grepl("^LY-PACKAGE-MISSING:", message)) {
      LY$status <- "package-missing"
      LY$error <- paste0("缺少 R 包 ", sub("^LY-PACKAGE-MISSING:", "", message))
      if (!nzchar(LY$package)) LY$package <- sub("^LY-PACKAGE-MISSING:", "", message)
    } else {
      LY$status <- "failed"
      LY$error <- message
    }
    NULL
  }
  tryCatch(withCallingHandlers(body(), error = collect), error = catch)
  ly_write()
  invisible(TRUE)
}

ly_input_missing <- function(name) {
  LY$status <- "no-input"
  LY$error <- paste0("缺少输入表 ", name, ".csv（本次参数不需要它，或数据太薄）")
  ly_write()
  invisible(TRUE)
}
