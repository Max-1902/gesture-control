#!/bin/zsh
# Records every Phase 1 session in sequence. Each one counts down 3 s, records, then closes.
cd "${0:A:h}"
PY=../.venv.nosync/bin/python
typeset -a sessions=(
  "idle-typing:150:Type and use the trackpad normally"
  "talking:120:Talk and gesture naturally, touch your face, drink"
  "pinch:60:Pinch and release slowly 10x at different distances"
  "fist-open:60:Alternate fist / open palm 10x, hold each ~1s"
  "point:60:Point with index finger and move it around"
  "two-hand:60:Pinch with both hands, spread and close"
  "low-light:60:Dim the lights, then pinch/release 10x"
)
for s in $sessions; do
  label=${s%%:*}; rest=${s#*:}; secs=${rest%%:*}; what=${rest#*:}
  echo "\n=== $label (${secs}s): $what"
  read "?Press Enter when ready (s = skip)... " ans
  [[ $ans == s ]] && continue
  $PY run.py --label $label --record $secs || exit 1
done
echo "\nDone:"; ls -1 ../recordings
