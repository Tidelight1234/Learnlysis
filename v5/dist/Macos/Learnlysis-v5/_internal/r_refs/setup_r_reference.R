## ---------------------------------------------------------------------------
## setup_r_reference.R —— 一键安装 R 参考层需要的包
##
## 用法（两种都一样）：
##   1) 应用内：工作台「安装 R 参考包 → 直接执行」，桥接层会带上库目录参数；
##   2) 手工：  "C:\...\Rscript.exe" --vanilla r_refs/setup_r_reference.R "<库目录>"
## 装到**用户库**（默认 data/Rlibrary），不需要管理员权限，也不污染站点库。
## 装不上（离线 / 没有二进制）只报告不报错：R 参考层是可选校验，Python 实现始终为准。
## ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
libs <- if (length(args) >= 1L && nzchar(args[1])) args[1] else Sys.getenv("R_LIBS_USER")
repo <- if (length(args) >= 2L && nzchar(args[2])) args[2] else "https://cloud.r-project.org"
## 清单必须与 analysis_r_bridge.py 的 REFERENCE_PACKAGES 一一对应：UI 的「缺包」角标按那八个探测，
## 这里少列一个，一键安装跑完后它就永远报缺。
packages <- c("jsonlite", "TraMineR", "seqHMM", "dtwclust", "rENA", "readxl",
              "cluster", "seqinr")

cat(sprintf("[setup] R %s.%s / 库目录 %s / 源 %s\n",
            R.version$major, R.version$minor, libs, repo))
created <- dir.create(libs, recursive = TRUE, showWarnings = FALSE)
if (!created && !dir.exists(libs)) {
  cat("[setup] 无法创建库目录，改用 R_LIBS_USER 默认位置\n")
}
.libPaths(c(libs, .libPaths()))

had_before <- vapply(packages, function(pkg) {
  isTRUE(tryCatch(requireNamespace(pkg, quietly = TRUE), error = function(e) FALSE))
}, logical(1))

tryCatch(utils::install.packages(packages[!had_before], lib = libs, repos = repo),
         error = function(e) cat(sprintf("[setup] install.packages 异常：%s\n",
                                         conditionMessage(e))))

cat("\n[setup] 结果：\n")
ok <- 0L
for (pkg in packages) {
  version <- tryCatch(as.character(utils::packageVersion(pkg)), error = function(e) NA_character_)
  if (is.na(version)) {
    cat(sprintf("  %-10s 未安装（该包的参考实现将降级为 base R 或提示不可用）\n", pkg))
  } else {
    ok <- ok + 1L
    cat(sprintf("  %-10s %s\n", pkg, version))
  }
}
cat(sprintf("[setup] %d/%d 个参考包就绪。\n", ok, length(packages)))
quit(save = "no", status = 0L)
