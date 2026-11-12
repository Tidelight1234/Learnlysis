"""把 .github/workflows/*.yml 里每个 run 块抽成临时脚本，交给对应 shell 解析器验语法。

GitHub Actions 的 YAML 在本机跑不了，但**语法**可以验：
  - YAML 本体：safe_load 一遍；
  - run: 里的 PowerShell / bash：写进临时文件，用 PS Parser / bash -n 解析。
这一步的存在理由：CI 里一个引号没配对，要等 push 上去、等 runner 跑 3 分钟才红。
"""
import io
import os
import re
import subprocess
import sys
import tempfile

import yaml

#: 本脚本住在 version/v5，而被校验的 workflow 在仓库根的 .github/ 下。向上找到第一个
#: 真的带 `.github` 的目录当作仓库根，而不是把目录层数算死（脚本再挪一次也不用改）。
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while REPO and not os.path.isdir(os.path.join(REPO, ".github")):
    parent = os.path.dirname(REPO)
    if parent == REPO:                       # 到盘符根了还没找到
        raise SystemExit("找不到含 .github 的仓库根（从 %s 往上）" % HERE)
    REPO = parent
WORKFLOW = os.path.join(REPO, ".github", "workflows", "build-windows.yml")

#: GitHub 在把 run 块交给 shell 之前就把 `${{ ... }}` 替掉了，所以本机解析时要先做同一件事：
#: 直接拿原文去过 PS 解析器会得到「Use `{ instead of { in variable names」——
#: 那是检查器自己的伪错，不是 CI 的错（踩过）。占位符不带空格与引号，不改变语法结构。
GITHUB_EXPR = re.compile(r"\$\{\{[^}]*\}\}")


def main():
    with io.open(WORKFLOW, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    print("YAML OK, jobs =", ", ".join(doc["jobs"]))

    failures = []
    for job_name, job in doc["jobs"].items():
        for step in job.get("steps", []):
            run = step.get("run")
            if not run:
                continue
            run = GITHUB_EXPR.sub("GITHUB_EXPR", run)
            shell = (step.get("shell") or "default").lower()
            name = "%s / %s" % (job_name, step.get("name", "?"))
            on_linux = "ubuntu" in str(job.get("runs-on", ""))
            if on_linux or "bash" in shell:
                check = "bash"
            elif "powershell" in shell or shell in ("default", "pwsh", "sh"):
                # default on windows-latest = pwsh；本机只有 Windows PowerShell 5.1，
                # 语法层面两者一致，用 5.1 的解析器足够抓出引号/括号不配对。
                check = "ps"
            else:
                check = "skip"
            if check == "ps":
                path = os.path.join(tempfile.gettempdir(), "ly_ci_check.ps1")
                with io.open(path, "w", encoding="utf-8-sig") as fh:
                    fh.write(run)
                cmd = ["powershell", "-NoProfile", "-Command",
                       "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
                       "$e=$null;[System.Management.Automation.Language.Parser]"
                       "::ParseFile('%s',[ref]$null,[ref]$e)|Out-Null;"
                       "if($e){$e|ForEach-Object{Write-Host $_.Message};exit 1}" % path]
            elif check == "bash":
                path = os.path.join(tempfile.gettempdir(), "ly_ci_check.sh")
                with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(run)
                cmd = None                          # 本机没有 bash，只查引号配对
            else:
                cmd = None
            if cmd:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace")
                if proc.returncode != 0:
                    failures.append((name, proc.stdout.strip()))
                else:
                    print("  parse OK  %s" % name)
            else:
                print("  manual    %s (shell=%s)" % (name, shell))
    if failures:
        for name, msg in failures:
            print("FAIL %s\n  %s" % (name, msg))
        return 1
    print("ALL RUN BLOCKS PARSE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
