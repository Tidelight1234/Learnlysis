"""开发期验收脚本：拿 _testdata 真跑 Rscript，逐方法逐参数组合看两侧差多少。

与 `_smoke_modes.py` 的分工：那边是「每个方法跑一遍默认参数」的常规冒烟，这边是
「把带分支的方法（OM/HAM/DHD/MCO/MCM、raw/compressed/span、scope、lag…）都推一遍」
的全量对照 —— 只在改统计层或改 `r_refs/*.R` 之后手动跑一次。

跑法（PowerShell）：
    cd version\\v5; $env:QT_QPA_PLATFORM="offscreen"
    ..\\.conda\\python.exe -X utf8 _crosscheck_r.py            # 全部
    ..\\.conda\\python.exe -X utf8 _crosscheck_r.py IND05      # 只跑某几个方法号

R 缺席时本脚本只打印降级说明并以 0 退出（Python 实现始终为准，不是错误）。
"""

from __future__ import annotations

import glob
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")

import analysis_core as ac                           # noqa: E402
import analysis_methods as am                        # noqa: E402
import analysis_r_bridge as rb                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(HERE, "_testdata", "*.xlsx")

#: 默认参数之外还要推的分支（覆盖各条计算路径，不只是「能跑」）
VARIANTS = {
    "IND01": [{}, {"scope": "both"}, {"max_lag": 5}],
    "IND02": [{}, {"self_transition": False}],
    "IND04": [{}, {"stream": "compressed"}, {"algorithm": "span"}, {"min_sup": 0.05}],
    "IND05": [{"method": "OM"}, {"method": "OM", "subst": "trate"},
              {"method": "OM", "subst": "custom", "custom_cost": "A,C=2.5; B,D=1.75"},
              {"method": "OM", "compress": False},
              {"method": "HAM"}, {"method": "DHD"}, {"method": "MCO"}, {"method": "MCM"}],
    "IND07": [{}, {"subst": "trate"}, {"cluster_method": "pam"},
              {"method": "DTW"}, {"method": "SDTW"}, {"method": "MCO"}],
    "IND06": [{}, {"soft_dtw": True}, {"compress": False},
              {"soft_dtw": True, "soft_gamma": 0.5}],
    "IND08": [{}, {"select_states": True}, {"criterion": "AIC"}, {"compress": False},
              {"engine": "hmmlearn"}, {"model": "mhmm"}],
    "IND10": [{}, {"min_edge": 2}, {"contrast_by": "group"}],
    "GRP05": [{}, {"max_lag": 3}, {"scope": "consensus"}],
    "GRP09": [{}, {"lag": 2}],
}


def params_of(spec, ctx):
    out = {item.key: item.default for item in spec.params}
    out.setdefault("students", list(ctx.students))
    out.setdefault("dimensions", list(ctx.dimensions))
    return out


def tag_of(spec_id, override):
    if not override:
        return spec_id
    return "%s/%s" % (spec_id, ",".join("%s=%s" % kv for kv in sorted(override.items())))


def main(argv):
    wanted = {str(item).upper() for item in argv}
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print("缺少 _testdata，请先跑 _gen_testdata.py")
        return 2
    status = rb.probe(force=True)
    print("R:", status.get("available"), status.get("version"), status.get("rscript"))
    print("packages:", {name: hit for name, hit in status.get("packages", {}).items() if hit})
    if not status.get("available"):
        print("未探测到 Rscript：跳过全量对照（Python 实现照常为准）")
        return 0

    failures = []
    checked = 0
    for mode in (am.MODE_INDIVIDUAL, am.MODE_GROUP):
        ctx = am.AnalysisContext(ac.load_bundles(paths), mode=mode)
        for spec in am.methods_for(mode):
            if spec.id not in rb.CHECK_PLANS or (wanted and spec.id not in wanted):
                continue
            base = params_of(spec, ctx)
            for override in VARIANTS.get(spec.id, [{}]):
                params = dict(base)
                params.update(override)
                checked += 1
                try:
                    result = spec.compute(ctx, params)
                except Exception:                        # noqa: BLE001
                    failures.append("%s 计算抛错\n%s" % (tag_of(spec.id, override),
                                                         traceback.format_exc()))
                    continue
                if getattr(result, "blocked", False):
                    failures.append("%s 被阻断：%s" % (spec.id, result.blocked_reason))
                    continue
                rows, note = rb.reference_check(spec, result, ctx, params)
                full = (result.meta or {}).get("r_crosscheck") or rows
                agreed, failed, missing, worst = rb.summarize_rows(full)
                print("\n== %-30s 行 %-4d 一致 %-4d 超差 %-3d 缺失 %-3d 最大差 %s" % (
                    tag_of(spec.id, override), len(full), agreed, failed, missing, worst))
                print("   ", note[:300])
                for row in (rows[:3] if failed or missing else []):
                    print("    !", row)
                if failed or missing or (worst or 0.0) > 1e-6:
                    failures.append(tag_of(spec.id, override))
    print("\n对照组合 %d 个" % checked)
    if failures:
        print("\n".join(failures))
        print("R_CROSSCHECK_FAILED %d 项" % len(failures))
        return 1
    print("R_CROSSCHECK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
