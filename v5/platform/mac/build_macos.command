#!/usr/bin/env bash
# 住在 platform/mac/，而 build_macos.sh 在仓库根：向上找而不是算死层数，
# 目录再动一层也不用回来改这里。
d="$(cd "$(dirname "$0")" && pwd)"
while [[ "$d" != "/" && ! -f "$d/build_macos.sh" ]]; do d="$(dirname "$d")"; done
[[ -f "$d/build_macos.sh" ]] || { echo "找不到仓库根的 build_macos.sh（从 $(dirname "$0") 往上）" >&2; exit 1; }
exec "$d/build_macos.sh" v5 "$@"
