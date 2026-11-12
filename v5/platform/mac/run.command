#!/usr/bin/env bash
set -euo pipefail
# 方案 B：这份启动器住在 version/<v>/platform/mac/。
# 产品码在 ../../src，构建 venv 与运行期 data/ 留在版本树根（../..）—— 树根才是「现场」，
# src/ 只是码。少剥这一层就会把 data/ 写进 src/ 里，等于又造出一份没人要的存档。
PLATFORM_DIR="$(cd "$(dirname "$0")" && pwd)"
TREE_ROOT="$(cd "$PLATFORM_DIR/../.." && pwd)"
SRC_DIR="$TREE_ROOT/src"
for py in "$TREE_ROOT/.mac-build-venv/bin/python" "$SRC_DIR/.mac-build-venv/bin/python" \
          "${LEARNLYSIS_PYTHON:-}" python3.13 python3.12 python3.11; do
    [[ -n "$py" ]] && command -v "$py" >/dev/null 2>&1 || continue
    "$py" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))' || continue
    cd "$SRC_DIR"; exec "$py" -X utf8 main.py
done
echo "Python 3.11–3.13 is required. Install Python 3.13 or set LEARNLYSIS_PYTHON." >&2
exit 1
