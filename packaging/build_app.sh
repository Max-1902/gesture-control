#!/bin/zsh
# Builds Gesture Control.app and zips it for a GitHub release.
#
#   packaging/build_app.sh            # this Mac's architecture
#   packaging/build_app.sh x86_64     # Intel build (runs under Rosetta)
#
# Output: dist.nosync/Gesture-Control-mac-<arm64|intel>.zip
# Builds happen in ~/Library/Caches (never inside iCloud Drive).
set -euo pipefail
ROOT="${0:A:h:h}"
ARCH="${1:-$(uname -m)}"
case "$ARCH" in
  arm64)  LABEL=arm64; PYARCH=aarch64; RUN=() ;;
  x86_64) LABEL=intel; PYARCH=x86_64;  RUN=(arch -x86_64) ;;
  *) echo "unknown arch $ARCH"; exit 1 ;;
esac
VERSION=$(sed -E 's/__version__ = "(.*)"/\1/' "$ROOT/src/version.py")
WORK="$HOME/Library/Caches/gesture-control-build/$LABEL"
OUT="$ROOT/dist.nosync"
mkdir -p "$WORK" "$OUT"
export UV_PYTHON_INSTALL_DIR="$HOME/Library/Caches/gesture-control-build/python"

echo "→ Python environment ($LABEL)"
command -v uv >/dev/null || { echo "needs uv: https://docs.astral.sh/uv/"; exit 1; }
uv python install "cpython-3.12-macos-$PYARCH-none" >/dev/null
PY=$(uv python find "cpython-3.12-macos-$PYARCH-none")
[[ -x "$WORK/venv/bin/python" ]] || uv venv --quiet --python "$PY" "$WORK/venv"
VIRTUAL_ENV="$WORK/venv" uv pip install --quiet -r "$ROOT/requirements.txt" pyinstaller
# numpy's default macOS 14 wheel would raise the minimum macOS; use the macOS 11 build
rm -rf "$WORK/wheels"
"${RUN[@]}" "$WORK/venv/bin/python" -m pip --version >/dev/null 2>&1 || VIRTUAL_ENV="$WORK/venv" uv pip install --quiet pip
"${RUN[@]}" "$WORK/venv/bin/python" -m pip download --quiet --only-binary=:all: --no-deps \
    --platform "macosx_11_0_$ARCH" --platform "macosx_10_9_$ARCH" --platform "macosx_10_13_$ARCH" \
    --python-version 3.12 --dest "$WORK/wheels" "numpy>=2.0"
VIRTUAL_ENV="$WORK/venv" uv pip install --quiet --reinstall "$WORK"/wheels/numpy-*.whl

echo "→ Tests"
VIRTUAL_ENV="$WORK/venv" uv pip install --quiet pytest
(cd "$ROOT" && "${RUN[@]}" "$WORK/venv/bin/python" -m pytest -q -p no:cacheprovider src/tests)

echo "→ Building Gesture Control.app $VERSION ($LABEL)"
(cd "$WORK" && GC_VERSION="$VERSION" "${RUN[@]}" "$WORK/venv/bin/python" -m PyInstaller --noconfirm \
    --log-level WARN --workpath "$WORK/build" --distpath "$WORK/dist" "$ROOT/packaging/GestureControl.spec")
APP="$WORK/dist/Gesture Control.app"

echo "→ Privacy check (no personal paths or names in the bundle)"
# pip records where a local wheel came from (= this Mac's home folder): drop those records
find "$APP" -name direct_url.json -path "*.dist-info/*" -delete
LEAKS=$(grep -r -l -a -F -e "$HOME" -e "$(id -un)" -e "$(id -F 2>/dev/null || echo __none__)" \
        -e "$(git config --global user.email 2>/dev/null || echo __none__)" "$APP" || true)
[[ -z "$LEAKS" ]] || { echo "!! personal data found in:"; echo "$LEAKS"; exit 1; }

echo "→ Signing (ad hoc)"
xattr -cr "$APP"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

echo "→ Smoke test (the app imports its whole camera/UI stack)"
"${RUN[@]}" "$APP/Contents/MacOS/Gesture Control" --self-test

ZIP="$OUT/Gesture-Control-mac-$LABEL.zip"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
echo "✓ $ZIP ($(du -h "$ZIP" | cut -f1))"
