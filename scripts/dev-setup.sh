#!/bin/zsh
# Developer setup (run from source): scripts/dev-setup.sh   (re-run any time to update dependencies)
# Friends don't need this: they use install.sh / the release download.
set -euo pipefail
ROOT="${0:A:h:h}"
cd "$ROOT"

[[ "$(uname)" == "Darwin" ]] || { echo "gesture-control only runs on macOS."; exit 1; }

# Inside iCloud Drive, a ".nosync" name keeps iCloud from syncing thousands of venv files.
VENV=".venv"
[[ "$ROOT" == *"Mobile Documents"* ]] && VENV=".venv.nosync"

echo "→ Creating Python environment ($VENV)"
if command -v uv >/dev/null; then
  uv venv --quiet --allow-existing --python 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install --quiet -r requirements-dev.txt
else
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v $c >/dev/null && $c -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY=$c; break; fi
  done
  [[ -n "$PY" ]] || { echo "Python 3.10+ is required: brew install python  (or install uv)"; exit 1; }
  $PY -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r requirements-dev.txt
fi

echo "→ Adding the 'gesture-control' command"
mkdir -p "$HOME/.local/bin"
ln -sf "$ROOT/bin/gesture-control" "$HOME/.local/bin/gesture-control"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo "   add this to your ~/.zshrc:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

echo "\n✓ Ready. Run: gesture-control   (tests: $VENV/bin/python -m pytest -q src/tests)"
echo "  When run from source, macOS asks for Camera and Accessibility for your Terminal app."
