#!/bin/bash
# Installs Gesture Control into Applications and opens it:
#   curl -fsSL https://raw.githubusercontent.com/Max-1902/gesture-control/main/install.sh | bash
set -euo pipefail

REPO="Max-1902/gesture-control"
APP_NAME="Gesture Control"
BUNDLE_ID="io.github.max-1902.gesture-control"

[[ "$(uname)" == "Darwin" ]] || { echo "Gesture Control only runs on macOS."; exit 1; }
major=$(sw_vers -productVersion | cut -d. -f1)
(( major >= 12 )) || { echo "Gesture Control needs macOS 12 or newer (this Mac has $(sw_vers -productVersion))."; exit 1; }

if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then FLAVOR=arm64; else FLAVOR=intel; fi
URL="https://github.com/$REPO/releases/latest/download/Gesture-Control-mac-$FLAVOR.zip"

DEST="/Applications"
[[ -w "$DEST" ]] || { DEST="$HOME/Applications"; mkdir -p "$DEST"; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading Gesture Control…"
curl -fL --progress-bar "$URL" -o "$TMP/app.zip"
ditto -x -k "$TMP/app.zip" "$TMP"

if pgrep -f "$APP_NAME.app/Contents/MacOS" >/dev/null; then
  echo "Closing the running copy…"
  pkill -f "$APP_NAME.app/Contents/MacOS" || true
  sleep 1
fi
if [[ -d "$DEST/$APP_NAME.app" ]]; then
  rm -rf "$DEST/$APP_NAME.app"   # older version; your settings live elsewhere and are kept
  # macOS ties permissions to the exact build: clear the old entry so the new one can be allowed
  tccutil reset Accessibility "$BUNDLE_ID" >/dev/null 2>&1 || true
fi
mv "$TMP/$APP_NAME.app" "$DEST/"
xattr -dr com.apple.quarantine "$DEST/$APP_NAME.app" 2>/dev/null || true

echo "Installed to $DEST/$APP_NAME.app. Opening it…"
open "$DEST/$APP_NAME.app"
cat <<'MSG'

Next: allow Camera and Accessibility when asked (the settings window shows what's missing).
Then hold up an open palm toward the camera for a moment to start.
Look for the ✋ in the menu bar for settings and Quit.
MSG
