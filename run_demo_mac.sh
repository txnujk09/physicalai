#!/usr/bin/env bash
# One-command Murphy demo for macOS (also works on Linux).
#   ./run_demo_mac.sh            setup + check + open results + live viewer
#   ./run_demo_mac.sh setup      create .venv and install deps (once, ~1-2 min)
#   ./run_demo_mac.sh check      20-seed A vs B sanity eval on this machine (~20 s)
#   ./run_demo_mac.sh show       print results table, open chart + videos
#   ./run_demo_mac.sh live       interactive viewer, policy B (drag the cube)
#   ./run_demo_mac.sh live A     same, with the clean-only policy for contrast
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
OPEN=$(command -v open || command -v xdg-open || echo echo)

setup() {
  if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv (Python package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  [ -x "$PY" ] || uv venv --python 3.11 .venv
  uv pip install --python "$PY" -q -r requirements.txt
  "$PY" -c "import mujoco, torch; print('>> ready: mujoco', mujoco.__version__, 'torch', torch.__version__)"
}

check() {
  echo ">> sanity eval on this machine: 20 held-out seeds, clean + impulse + noise"
  "$PY" eval.py --policies A B --episodes 20 --conditions clean impulse noise --out /tmp/murphy_check
}

show() {
  "$PY" summarize.py
  "$OPEN" results/success_rates.png
  "$OPEN" results/demo_impulse_seed900000.mp4
}

live() {
  echo ">> viewer: double-click the cube, then Ctrl+right-drag to shove it. Close the window to quit."
  "$PY" demo.py viewer --policy "${1:-B}"
}

case "${1:-all}" in
  setup) setup ;;
  check) check ;;
  show) show ;;
  live) live "${2:-B}" ;;
  all) setup; check; show; live B ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
