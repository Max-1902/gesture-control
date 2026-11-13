# Gesture Window Control for macOS: Architecture & Build Plan

A menu-bar app that uses the webcam to track your hands and control windows (move, resize, minimize/close, zoom) with gestures.

---

## 1. Goals and Non-Goals

**Goals**
- Move, resize, minimize, and close windows with hand gestures
- Zoom (two-hand pinch-spread)
- Low latency (feels live), low CPU/battery cost, near-zero accidental triggers
- Clean, testable architecture suitable as a portfolio piece

**Non-goals (v1)**
- Replacing the trackpad for daily use (mid-air gestures are slower and tiring)
- Multi-user / multi-camera support
- Mac App Store distribution (see Section 9, it's incompatible with sandboxing)
- Custom gesture training with ML (hand-written geometric rules first)

**Honest framing:** the CV is the easy part. The hard parts are (a) false positives, (b) jitter, and (c) arm fatigue. The design below spends most of its complexity budget on those three.

---

## 2. Decisions to Make Up Front

| # | Decision | Recommendation | Why |
|---|----------|----------------|-----|
| 1 | Prototype language | Python (MediaPipe + pyobjc) first | Iterate on gesture logic fast; throwaway shell |
| 2 | Production language | Swift (Vision + AppKit) | Native Neural Engine use, proper menu-bar app, permissions UX |
| 3 | Hand tracker | MediaPipe Hands (proto) → `VNDetectHumanHandPoseRequest` (prod) | Both give 21 landmarks per hand |
| 4 | Window targeting | Virtual cursor from index fingertip; pinch = grab | Avoids "which window?" ambiguity |
| 5 | Motion mapping | Relative/clutch (like lifting a mouse), not absolute | Absolute mapping makes you hold your arm at screen-edge positions |
| 6 | What "zoom" means | **You must pick one** (see below) | Three different mechanisms, different reliability |

**Zoom ambiguity: pick one for v1**
- **(a) Resize the window** via AX size attribute. Most reliable, works everywhere. *Recommended for v1.*
- **(b) App content zoom** via synthesized Cmd +/-. App-dependent; some apps ignore it.
- **(c) Screen magnifier** via macOS Accessibility Zoom (ctrl+scroll). Needs the user to enable it in System Settings.

---

## 3. System Architecture

```
┌──────────────┐   frames    ┌──────────────┐  landmarks  ┌──────────────┐
│ CameraSource │ ──────────▶ │ HandTracker  │ ──────────▶ │ LandmarkFilter│
│ AVCapture    │  640x480    │ Vision / MP  │  21 pts/hand│ One Euro     │
│ 30 fps       │             │              │             │              │
└──────────────┘             └──────────────┘             └──────┬───────┘
                                                                 │ smoothed
                                                                 ▼
┌──────────────┐  intents    ┌──────────────┐  features   ┌──────────────┐
│ ActionRouter │ ◀────────── │ GestureFSM   │ ◀────────── │ FeatureExtract│
│              │             │ (state       │             │ pinch dist,  │
│              │             │  machine)    │             │ palm open,   │
└──────┬───────┘             └──────┬───────┘             │ hand scale   │
       │                            │ state               └──────────────┘
       ▼                            ▼
┌──────────────┐             ┌──────────────┐
│ WindowCtrl   │             │ HUD Overlay  │
│ AX API +     │             │ cursor, ring,│
│ CGEvent      │             │ engage state │
└──────────────┘             └──────────────┘

        Cross-cutting: Config (YAML/JSON) · Logger/Recorder (JSONL) · Metrics
```

### Threading model
- **Capture queue** (serial, high priority): receives frames, drops frames if tracker is behind (never queue, always use the latest).
- **Vision queue**: runs hand pose request, outputs landmarks.
- **Logic queue**: filter → features → FSM → intents (pure, fast, deterministic).
- **AX queue** (separate, serial): window operations. AX calls can block for tens of ms on slow apps (Electron especially), so they must never run on the logic queue.
- **Main thread**: HUD overlay and menu bar UI only.

### Latency budget (target p95 < 100 ms end to end)

| Stage | Budget |
|-------|--------|
| Camera capture + delivery | ~33 ms (one frame at 30 fps) |
| Hand pose inference | 10–20 ms |
| Filter + features + FSM | < 2 ms |
| AX call | 2–15 ms (app-dependent) |
| HUD render | < 8 ms |

---

## 4. Component Specs

### 4.1 CameraSource
- `AVCaptureSession`, 640×480 (or 1280×720 if tracking is poor), 30 fps.
- Mirror horizontally so movement feels natural.
- Drop late frames (`alwaysDiscardsLateVideoFrames = true`).

### 4.2 HandTracker
- Output a `Hand` struct: `handedness`, `landmarks[21]` (normalized x,y + confidence), `timestamp`.
- Max 2 hands. Reject hands with low confidence.
- Note: Vision returns normalized coords with a **lower-left origin**; MediaPipe uses **upper-left**. Normalize this once at the boundary.

### 4.3 LandmarkFilter
- **One Euro filter** per landmark coordinate (adaptive low-pass: smooth when slow, responsive when fast).
- Tunables: `min_cutoff` (jitter at rest), `beta` (lag during fast motion). Expose both in config.

### 4.4 FeatureExtractor
All features are **scale-invariant** (divide by hand size = wrist to middle-finger MCP distance) so they work at any distance from the camera.
- `pinch = dist(thumb_tip, index_tip) / hand_size`
- `palm_open` = all five fingertips far from palm center
- `fist` = all fingertips close to palm center
- `pointer_xy` = index fingertip position
- `hand_center` = palm centroid (used for drag deltas)
- `two_hand_span` = distance between hand centers (zoom)

### 4.5 GestureFSM (the core of the project)

```
      ┌────────┐  open palm held 0.5s   ┌─────────┐
      │ IDLE   │ ─────────────────────▶ │ ARMED   │ ◀── timeout 3s w/o hand → IDLE
      └────────┘                        └────┬────┘
           ▲                                 │ index pointing → cursor moves
           │ lose hand > 1s                  ▼
           │                           ┌─────────┐   pinch (hysteresis)   ┌──────────┐
           └────────────────────────── │ HOVER   │ ─────────────────────▶ │ GRABBED  │
                                       └────┬────┘ ◀──── release ───────  └────┬─────┘
                                            │ both hands pinch                  │ delta → move window
                                            ▼                                   │
                                       ┌─────────┐                              │
                                       │ ZOOM    │ span ratio → resize          │
                                       └─────────┘                              │
                                       fist held 1s (HUD ring) → MINIMIZE ◀─────┘
```

Rules that make it usable:
- **Hysteresis:** pinch ON at < 0.25, OFF at > 0.35. Prevents flicker at the threshold.
- **Debounce:** every transition needs N consecutive frames (typically 3–5).
- **Dwell for destructive actions:** minimize/close only after a held gesture with a visible progress ring. Close should require a second confirm gesture, or just don't ship close and use minimize.
- **Timeouts:** ARMED returns to IDLE if no hand for 3s. Safe by default.
- **Kill switch:** global hotkey toggles the whole system off. Also a menu-bar toggle.

### 4.6 ActionRouter
Maps FSM intents to window operations. Intents are a small enum:
`MoveWindow(dx, dy)`, `ResizeWindow(scale)`, `Minimize`, `Close`, `Snap(left|right)`, `ZoomApp(±)`.

### 4.7 WindowController (macOS Accessibility API)
- **Target selection:** `AXUIElementCopyElementAtPosition(systemWide, x, y)` → walk up to the `AXWindow` ancestor.
- **Move:** `AXUIElementSetAttributeValue(window, kAXPositionAttribute, AXValue(CGPoint))`
- **Resize:** same with `kAXSizeAttribute` and `CGSize`
- **Minimize:** set `kAXMinimizedAttribute` to true
- **Close:** `AXUIElementPerformAction` with `kAXPressAction` on the element from `kAXCloseButtonAttribute`
- **Coordinate gotcha:** AX uses a **top-left global origin**; Cocoa/AppKit uses **bottom-left**. Convert once, in one place. Multi-display setups make this worse, so test with two monitors early.
- Rate-limit writes (~60 Hz max) and coalesce deltas.

### 4.8 HUD Overlay
- Borderless, click-through `NSPanel` at screen-saver window level.
- Shows: engaged/idle indicator, virtual cursor, highlighted target window, dwell progress ring.
- **This is not optional.** Without visual feedback gesture UIs feel broken.

### 4.9 Recorder / Replay
- Record raw landmarks + timestamps to JSONL.
- Replay through filter → FSM headlessly for deterministic tests and parameter tuning.

---

## 5. Gesture Vocabulary (v1)

| Gesture | Action | Notes |
|---------|--------|-------|
| Open palm, hold 0.5 s | Arm system | Engage gesture; nothing works without it |
| Index pointing | Move virtual cursor / pick window | Hover highlight in HUD |
| Pinch + move | Drag window | Relative motion with gain ≈ 2–3× |
| Two-hand pinch, spread/close | Resize window | Ratio of hand span |
| Fist, hold 1 s | Minimize | Progress ring; cancelable |
| Quick flick left/right (while grabbed) | Snap window to half | Stretch goal |
| Hand drops out of frame > 3 s | Disarm | Automatic |

---

## 6. Tech Stack

**Prototype (Python)**
- `opencv-python`, `mediapipe`, `pyobjc-framework-ApplicationServices`, `pyobjc-framework-Quartz`
- `numpy`, `pytest`

**Production (Swift)**
- AVFoundation, Vision, AppKit, ApplicationServices (AX), CoreGraphics (CGEvent)
- Swift Package for the core (filter, features, FSM) with XCTest so the logic is UI-free and testable
- `LSUIElement = true` (menu-bar only, no Dock icon)

---

## 7. Repository Layout

```
gesture-control/
├── README.md
├── src/                        # Python
│   ├── capture.py
│   ├── tracker.py
│   ├── one_euro.py
│   ├── features.py
│   ├── fsm.py
│   ├── window_ctrl.py
│   └── run.py
├── recordings/                 # JSONL landmark sessions (gitignored if large)
├── GestureCore/                # Swift Package (pure logic)
│   ├── Sources/GestureCore/
│   │   ├── OneEuroFilter.swift
│   │   ├── Features.swift
│   │   ├── GestureFSM.swift
│   │   └── Intents.swift
│   └── Tests/GestureCoreTests/
├── App/                        # Xcode project (menu-bar app)
│   ├── CameraSource.swift
│   ├── HandTracker.swift
│   ├── WindowController.swift
│   ├── HUDOverlay.swift
│   ├── MenuBar.swift
│   └── Permissions.swift
├── config/default.yaml
└── docs/                       # demo GIFs, design notes, metrics
```

---

## 8. Build Plan

Estimated effort: **~3–4 focused weekends** to a solid v1. Each phase has an exit criterion. Don't advance until it's met.

### Phase 0: Spike (1 evening)
- Webcam → MediaPipe → draw 21 landmarks live.
- **Exit:** stable landmarks at ≥ 25 fps. Note CPU %. If tracking is bad in your lighting, fix the environment now.

### Phase 1: Recording + replay harness (1 evening)
- Record 5–10 sessions: idle typing, talking with gestures, deliberate gestures, bad lighting.
- Build the replay runner.
- **Exit:** you can re-run a recorded session through the pipeline offline and get identical output twice.
- *Why this early:* it turns "tune by waving at the camera" into real engineering with regression tests.

### Phase 2: Filter + features
- Implement One Euro filter and scale-invariant features.
- **Exit:** feature plots on recorded data show clean separation (pinch vs. no pinch, fist vs. open). Pick thresholds from the data, not by guessing.

### Phase 3: Gesture FSM + engage gesture
- Implement states, hysteresis, debounce, dwell, timeouts.
- Unit tests driven by recorded sessions.
- **Exit:** on the "typing/talking" recordings, **zero** false arms. On deliberate recordings, engage succeeds > 95%.

### Phase 4: Window control (Python)
- AX move first, then resize, then minimize. Add close last (or skip it).
- Grant Accessibility permission to your terminal/Python.
- **Exit:** pinch-drag moves a real window smoothly, including on a second display, without fighting the coordinate flip.

### Phase 5: Two-hand resize
- Span-ratio resize with clamping to min/max window size.
- **Exit:** resizing doesn't jitter or run away when one hand drops out mid-gesture (must cleanly abort).

### Phase 6: Decision gate
Ask honestly: *Does this feel good enough to be worth a native rewrite?* If latency or fatigue makes it unpleasant, polish the Python demo, record the video, and stop. Otherwise continue.

### Phase 7: Swift port
- Port `GestureCore` first (filter, features, FSM) and make the same recorded sessions pass the same assertions in XCTest. **This is your correctness bridge.**
- Then AVFoundation + Vision tracker, then the AX controller.
- **Exit:** feature parity with the prototype; CPU < 15% on your machine.

### Phase 8: Menu-bar app, HUD, permissions
- HUD overlay, menu-bar toggle, global kill-switch hotkey, first-run permission flow (Camera, Accessibility) with clear explanations.
- **Exit:** a fresh user can install, grant permissions, and use it without reading code.

### Phase 9: Hardening and polish
- Config file, calibration screen (pick gain, thresholds), battery/CPU profiling, crash-safe teardown (never leave a window stuck mid-drag).
- **Exit:** run it all day with no stuck states.

### Phase 10: Ship the portfolio artifact
- README with architecture diagram, metrics table, and a 30-second demo GIF/video.
- Short write-up: latency budget, false-positive methodology, what you'd do differently.

---

## 9. Permissions, Signing, Distribution

- **Camera:** `NSCameraUsageDescription` in Info.plist. The green camera indicator will be on whenever armed or tracking. Consider only running the camera when enabled, or downclocking to ~10 fps when idle and waiting for the engage gesture.
- **Accessibility:** required for AX control and `CGEvent` posting. User grants it in System Settings → Privacy & Security → Accessibility. Check `AXIsProcessTrustedWithOptions` and guide the user.
- **Sandbox:** the App Sandbox must be **off** for AX control of other apps, which means **no Mac App Store**. Distribute via GitHub release with Developer ID signing + notarization (requires a paid Apple Developer account, ~$99/yr). For a portfolio piece, unsigned + build-from-source instructions are fine.

---

## 10. Testing and Metrics

Report these numbers in the README; they're what makes the project credible.

| Metric | Target | How measured |
|--------|--------|--------------|
| False arms per hour (idle typing/talking) | < 1 | Replay of recorded idle sessions |
| Engage success rate | > 95% | Replay of deliberate sessions |
| End-to-end latency p95 | < 100 ms | Timestamp at capture vs. after AX call |
| CPU usage while armed | < 15% | Activity Monitor / `powermetrics` |
| CPU usage while idle | < 5% | Same |
| Window jitter at rest (px RMS) | < 2 px | Log window position while holding a pinch still |

Test layers:
1. **Unit:** filter, feature math, FSM transitions (pure functions, no hardware).
2. **Replay/regression:** recorded landmark sessions with expected state traces.
3. **Manual checklist:** multi-display, fullscreen apps, Electron apps, Stage Manager, low light, glasses/sleeves/backlighting.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| False activations | Critical (windows move/close unexpectedly) | Engage gesture, dwell, hysteresis, default-minimize over close, kill-switch |
| Jitter | High (drag feels bad) | One Euro filter, relative motion, deadzone |
| Arm fatigue | High (limits real use) | Small-motion gain, clutching, keep it opt-in |
| AX quirks (Electron, fullscreen, Stage Manager, some apps reject writes) | Medium | Graceful failure, per-app fallback, test matrix |
| Coordinate-system bugs across displays | Medium | Single conversion layer + multi-monitor tests early |
| Poor lighting / backlight | Medium | Confidence gating, HUD warning when tracking quality drops |
| Privacy perception (always-on camera) | Medium | Process frames in memory only, never store/transmit; recording is explicit and off by default |
| macOS updates changing permission behavior | Low–Med | Pin test OS version in README |

---

## 12. Stretch Goals (only after v1 is solid)

- Snap gestures (flick to left/right half, quarter tiling)
- Virtual desktop / Mission Control switching via swipe
- Per-app gesture profiles
- Small learned classifier (landmarks → gesture) to compare against rule-based; report precision/recall
- Depth cue from hand scale for a "push" click gesture
- Continuity Camera / external webcam selection
- Optional audio/visual feedback (subtle tick on grab)

---

## 13. Code Sketches

**One Euro filter (core idea, per scalar)**
```python
class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * 3.141592653589793 * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat
```

**Pinch with hysteresis**
```python
def update_pinch(pinch_ratio, was_pinched, on=0.25, off=0.35):
    if was_pinched:
        return pinch_ratio < off
    return pinch_ratio < on
```

**Move a window (Swift, AX)**
```swift
func move(window: AXUIElement, to point: CGPoint) {
    var p = point  // AX coords: top-left global origin
    guard let value = AXValueCreate(.cgPoint, &p) else { return }
    AXUIElementSetAttributeValue(window, kAXPositionAttribute as CFString, value)
}

func window(at point: CGPoint) -> AXUIElement? {
    let system = AXUIElementCreateSystemWide()
    var element: AXUIElement?
    guard AXUIElementCopyElementAtPosition(system, Float(point.x), Float(point.y), &element) == .success,
          var current = element else { return nil }
    // Walk up until role == AXWindow
    while true {
        var role: CFTypeRef?
        AXUIElementCopyAttributeValue(current, kAXRoleAttribute as CFString, &role)
        if (role as? String) == kAXWindowRole as String { return current }
        var parent: CFTypeRef?
        guard AXUIElementCopyAttributeValue(current, kAXParentAttribute as CFString, &parent) == .success,
              let next = parent else { return nil }
        current = next as! AXUIElement
    }
}
```

**Config sketch (`default.yaml`)**
```yaml
camera: { width: 640, height: 480, fps: 30, mirror: true }
filter: { min_cutoff: 1.0, beta: 0.02 }
gestures:
  engage_hold_s: 0.5
  pinch_on: 0.25
  pinch_off: 0.35
  fist_dwell_s: 1.0
  disarm_timeout_s: 3.0
  drag_gain: 2.5
safety:
  allow_close: false        # minimize only until you trust it
  kill_switch_hotkey: "ctrl+opt+cmd+g"
```

---

## 14. Definition of Done (v1)

- [ ] Engage gesture arms/disarms reliably; zero false arms in recorded idle sessions
- [ ] Pinch-drag moves any normal window smoothly on single and dual displays
- [ ] Two-hand resize works and aborts cleanly
- [ ] Minimize via dwell gesture with visible progress ring
- [ ] HUD overlay shows state, cursor, target window
- [ ] Global kill switch and menu-bar toggle
- [ ] p95 latency < 100 ms, CPU < 15% armed
- [ ] Recorded-session regression tests pass in both Python and Swift
- [ ] README with architecture diagram, metrics table, and demo video
