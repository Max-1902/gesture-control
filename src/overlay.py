"""UI process: HUD overlay, menu bar icon, settings window, global hotkey.

Runs in its own process so drawing never competes with tracking for the GIL. Talks to
the tracking process (run.py) over a multiprocessing Pipe.

Parent → UI:
    dict                    HUD state (~60/s while something changes)
    ("settings", dict)      current settings + status, shown in the settings window
    ("show_settings",)      open the settings window
    ("fatal", str)          show an error dialog, then quit
    None                    quit
UI → parent:
    "toggle" | "quit"       hotkey / menu / settings switch
    ("set", {key: value})   a setting changed in the settings window (dotted keys, see config.USER_KEYS)

HUD coordinates are AX/CG global (top-left origin of the main display); each window
converts to its own Cocoa view coordinates (bottom-left origin).
"""
from __future__ import annotations

import json
import multiprocessing as mp

STATE_STYLE = {   # state → (ring rgba, fill rgba or None, radius)
    "hover":   ((1.0, 1.0, 1.0, 0.95), None, 16.0),
    "press":   ((0.2, 0.9, 1.0, 1.0), (0.2, 0.9, 1.0, 0.55), 11.0),
    "grabbed": ((1.0, 0.55, 0.1, 1.0), (1.0, 0.55, 0.1, 0.35), 14.0),
    "zoom":    ((0.95, 0.3, 0.85, 1.0), (0.95, 0.3, 0.85, 0.3), 18.0),
    "armed":   ((1.0, 1.0, 1.0, 0.35), None, 16.0),
}

HINTS = {
    "low":  ((1.0, 0.6, 0.1, 1.0), "Hand too low  ·  raise it a bit so the camera sees all of it"),
    "edge": ((1.0, 0.6, 0.1, 1.0), "Hand at the edge of the camera view  ·  move it toward the center"),
}

STATUS = {   # state → (dot rgba, text) for the status pill at the top of the main screen
    "off":     ((0.6, 0.6, 0.6, 1.0), "Gesture control OFF  ·  ⌃⌥⌘G to turn on"),
    "idle":    ((0.3, 0.95, 0.45, 1.0), "Gesture control ON  ·  hold up an open palm to start"),
    "armed":   ((1.0, 0.8, 0.2, 1.0), "Hand lost  ·  show your hand again"),
    "hover":   ((0.35, 0.7, 1.0, 1.0), "Active  ·  pinch = click / hold to grab  ·  fist = minimize"),
    "press":   ((0.2, 0.9, 1.0, 1.0), "Pinch…"),
    "grabbed": ((1.0, 0.55, 0.1, 1.0), "Holding window  ·  open hand to drop"),
    "zoom":    ((0.95, 0.3, 0.85, 1.0), "Resizing  ·  open hands to finish"),
}
ACTIVE_STATES = ("armed", "hover", "press", "grabbed", "zoom")

PRIVACY_URLS = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
}


def overlay_main(conn, settings_html: str = "") -> None:   # runs in the child process
    import ApplicationServices as AS
    from AppKit import (NSAlert, NSApplication, NSApplicationActivationPolicyAccessory,
                        NSBackingStoreBuffered, NSBezierPath, NSColor, NSEvent, NSFont,
                        NSFontAttributeName, NSForegroundColorAttributeName, NSImage, NSMenu,
                        NSMenuItem, NSScreen, NSScreenSaverWindowLevel, NSStatusBar, NSString,
                        NSTimer, NSURL, NSView, NSVariableStatusItemLength, NSWindow,
                        NSWindowCollectionBehaviorCanJoinAllSpaces,
                        NSWindowCollectionBehaviorFullScreenAuxiliary,
                        NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
                        NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable,
                        NSWindowStyleMaskTitled, NSWorkspace)
    from Foundation import NSNotificationCenter, NSObject

    from paths import FROZEN, LOG_DIR
    from version import __version__

    state: dict = {}
    settings: dict = {}
    ui = {"main_h": 0.0, "monitor": None, "ax": None, "web": None, "win": None, "ready": False,
          "status_key": None}
    views: list = []
    windows: list = []

    def color(c):
        return NSColor.colorWithCalibratedRed_green_blue_alpha_(*c)

    font = NSFont.systemFontOfSize_weight_(13.0, 0.3)

    def status_key() -> str:
        return "off" if state.get("off") else state.get("s", "idle")

    def draw_status(view):
        key = status_key()
        hint = state.get("hint") if key != "off" else None
        if not settings.get("show_status", True) and key not in ACTIVE_STATES and hint not in HINTS:
            return
        dot, text = STATUS.get(key, STATUS["idle"])
        if hint in HINTS:
            dot, text = HINTS[hint]
        attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: color((1, 1, 1, 0.95))}
        ns = NSString.stringWithString_(text)
        tw, th = ns.sizeWithAttributes_(attrs)
        b = view.bounds().size
        w, h = tw + 44, 30
        x, y = (b.width - w) / 2, b.height - h - 34   # just below the menu bar
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((x, y), (w, h)), 15, 15)
        color((0.08, 0.08, 0.1, 0.78)).set()
        pill.fill()
        color(dot).set()
        NSBezierPath.bezierPathWithOvalInRect_(((x + 13, y + 10), (10, 10))).fill()
        ns.drawAtPoint_withAttributes_((x + 31, y + (h - th) / 2), attrs)

    class HUDView(NSView):
        def isFlipped(self):
            return False

        def drawRect_(self, rect):
            NSColor.clearColor().set()
            NSBezierPath.fillRect_(rect)
            if views and self is views[0]:
                draw_status(self)
            if not state:
                return
            org = self.window().frame().origin
            main_h = ui["main_h"]

            def local(x, y):   # AX global → this view
                return x - org.x, (main_h - y) - org.y

            win = state.get("win")
            if win:
                x, y, w, h = win
                lx, ly = local(x, y + h)
                p = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((lx, ly), (w, h)), 10, 10)
                p.setLineWidth_(3.0 if state.get("s") == "grabbed" else 2.0)
                color((1.0, 0.55, 0.1, 0.9) if state.get("s") == "grabbed" else (0.35, 0.7, 1.0, 0.75)).set()
                p.stroke()

            ptr = state.get("p")
            if not ptr:
                return
            cx, cy = local(*ptr)
            ring, fill, r = STATE_STYLE.get(state.get("s"), STATE_STYLE["hover"])
            circ = NSBezierPath.bezierPathWithOvalInRect_(((cx - r, cy - r), (2 * r, 2 * r)))
            if fill:
                color(fill).set()
                circ.fill()
            color((0, 0, 0, 0.35)).set()          # soft dark halo: readable on light and dark
            circ.setLineWidth_(5.0)
            circ.stroke()
            color(ring).set()
            circ.setLineWidth_(2.5)
            circ.stroke()

            for key, col, rad in (("dwell", (1.0, 0.25, 0.25, 0.95), r + 10),
                                  ("arm", (0.3, 0.95, 0.45, 0.95), r + 10)):
                prog = state.get(key, 0.0)
                if prog > 0:
                    arc = NSBezierPath.bezierPath()
                    arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                        (cx, cy), rad, 90.0, 90.0 - 360.0 * prog, True)
                    arc.setLineWidth_(4.0)
                    color(col).set()
                    arc.stroke()

    def build_hud() -> None:
        """One transparent, click-through window per display. Rebuilt when displays change."""
        for w in windows:
            w.orderOut_(None)
        windows.clear()
        views.clear()
        screens = NSScreen.screens() or []
        ui["main_h"] = screens[0].frame().size.height if screens else 0.0   # AX y=0 = top of main screen
        for screen in screens:
            w = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                screen.frame(), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
            w.setReleasedWhenClosed_(False)
            w.setOpaque_(False)
            w.setBackgroundColor_(NSColor.clearColor())
            w.setIgnoresMouseEvents_(True)
            w.setHasShadow_(False)
            w.setLevel_(NSScreenSaverWindowLevel)
            w.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                     | NSWindowCollectionBehaviorStationary
                                     | NSWindowCollectionBehaviorFullScreenAuxiliary)
            v = HUDView.alloc().initWithFrame_(((0, 0), screen.frame().size))
            w.setContentView_(v)
            w.orderFrontRegardless()
            views.append(v)
            windows.append(w)

    def redraw() -> None:
        for v in views:
            v.setNeedsDisplay_(True)

    def send(msg) -> None:
        try:
            conn.send(msg)
        except (BrokenPipeError, OSError):
            NSApplication.sharedApplication().terminate_(None)

    # ----- global kill switch ⌃⌥⌘G: key monitors only work with Accessibility, so (re)install
    # the monitor whenever that permission appears ------------------------------------------
    CTRL, OPT, CMD = 1 << 18, 1 << 19, 1 << 20

    def on_key(event):
        flags = event.modifierFlags()
        if event.keyCode() == 5 and flags & CTRL and flags & OPT and flags & CMD:   # 5 = 'g'
            send("toggle")

    def install_hotkey() -> None:
        if ui["monitor"] is not None:
            NSEvent.removeMonitor_(ui["monitor"])
        ui["monitor"] = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(1 << 10, on_key)

    # ----- launch at login (needs the packaged .app) -----------------------------------------
    def login_service():
        if not FROZEN:
            return None
        try:
            from ServiceManagement import SMAppService
            return SMAppService.mainAppService()
        except Exception:
            return None

    def login_enabled() -> bool:
        svc = login_service()
        return bool(svc is not None and svc.status() == 1)   # 1 = SMAppServiceStatusEnabled

    def set_login(on: bool) -> None:
        svc = login_service()
        if svc is None:
            return
        try:
            if on:
                svc.registerAndReturnError_(None)
            else:
                svc.unregisterAndReturnError_(None)
        except Exception:
            pass

    # ----- settings window (WKWebView) -------------------------------------------------------
    def push_settings() -> None:
        web = ui["web"]
        if web is None or not settings or not ui["ready"] or not ui["win"].isVisible():
            return
        payload = dict(settings, ax=bool(AS.AXIsProcessTrusted()), login=login_enabled(),
                       login_supported=login_service() is not None, version=__version__,
                       state=status_key(), status_text=STATUS.get(status_key(), STATUS["idle"])[1])
        web.evaluateJavaScript_completionHandler_(f"window.gcUpdate({json.dumps(payload)})", None)

    def on_web_message(body: str) -> None:
        try:
            msg = json.loads(str(body))
        except ValueError:
            return
        kind = msg.get("type")
        if kind == "ready":
            ui["ready"] = True
            push_settings()
        elif kind == "set":
            send(("set", {str(msg["key"]): msg["value"]}))
        elif kind in ("toggle", "quit"):
            send(kind)
        elif kind == "open" and msg.get("what") in PRIVACY_URLS:
            if msg["what"] == "accessibility":
                AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
            NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(PRIVACY_URLS[msg["what"]]))
        elif kind == "login":
            set_login(bool(msg.get("value")))
            push_settings()
        elif kind == "logs":
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(LOG_DIR)))
        elif kind == "close" and ui["win"] is not None:
            ui["win"].performClose_(None)

    class WebBridge(NSObject):
        def userContentController_didReceiveScriptMessage_(self, controller, message):
            on_web_message(message.body())

    def show_settings() -> None:
        app = NSApplication.sharedApplication()
        if ui["win"] is None:
            from WebKit import WKUserContentController, WKWebView, WKWebViewConfiguration
            cfg = WKWebViewConfiguration.alloc().init()
            ucc = WKUserContentController.alloc().init()
            ui["bridge"] = WebBridge.alloc().init()
            ucc.addScriptMessageHandler_name_(ui["bridge"], "gc")
            cfg.setUserContentController_(ucc)
            style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                ((0, 0), (440, 640)), style, NSBackingStoreBuffered, False)
            win.setReleasedWhenClosed_(False)
            win.setTitle_("Gesture Control")
            win.setTitlebarAppearsTransparent_(True)
            win.setBackgroundColor_(NSColor.windowBackgroundColor())
            web = WKWebView.alloc().initWithFrame_configuration_(((0, 0), (440, 640)), cfg)
            web.setValue_forKey_(False, "drawsBackground")   # the window color shows through
            web.loadHTMLString_baseURL_(settings_html or "<p>settings.html missing</p>", None)
            win.setContentView_(web)
            win.center()
            ui["win"], ui["web"] = win, web
        app.activateIgnoringOtherApps_(True)
        ui["win"].makeKeyAndOrderFront_(None)
        push_settings()

    def fatal(text: str) -> None:
        app = NSApplication.sharedApplication()
        app.activateIgnoringOtherApps_(True)
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Gesture Control stopped")
        alert.setInformativeText_(text)
        alert.addButtonWithTitle_("Quit")
        alert.addButtonWithTitle_("Open Log Folder")
        if alert.runModal() == 1001:   # second button
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(LOG_DIR)))
        app.terminate_(None)

    class Driver(NSObject):
        def tick_(self, timer):
            changed = False
            try:
                while conn.poll():
                    msg = conn.recv()
                    if msg is None:
                        NSApplication.sharedApplication().terminate_(None)
                        return
                    if isinstance(msg, dict):
                        state.clear()
                        state.update(msg)
                        changed = True
                    elif isinstance(msg, tuple) and msg:
                        if msg[0] == "settings":
                            settings.clear()
                            settings.update(msg[1])
                            changed = True
                            push_settings()
                        elif msg[0] == "show_settings":
                            show_settings()
                        elif msg[0] == "fatal":
                            fatal(str(msg[1]))
                            return
            except (EOFError, OSError):
                NSApplication.sharedApplication().terminate_(None)
                return
            if changed:
                redraw()
                update_menu()
                if status_key() != ui["status_key"]:
                    ui["status_key"] = status_key()
                    push_settings()

        def slow_(self, timer):   # 1 Hz: permission changes
            ax = bool(AS.AXIsProcessTrusted())
            if ax != ui["ax"]:
                ui["ax"] = ax
                if ax:
                    install_hotkey()
                push_settings()

        def screensChanged_(self, note):
            build_hud()
            redraw()

        def applicationShouldHandleReopen_hasVisibleWindows_(self, app, flag):
            show_settings()   # opening the app again while it runs = open Settings
            return True

        def toggle_(self, sender):
            send("toggle")

        def settings_(self, sender):
            show_settings()

        def quit_(self, sender):
            send("quit")

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)   # no Dock icon
    build_hud()
    driver = Driver.alloc().init()
    app.setDelegate_(driver)
    NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
        driver, "screensChanged:", "NSApplicationDidChangeScreenParametersNotification", None)
    install_hotkey()

    # ----- menu bar icon -----------------------------------------------------------------------
    item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
    menu = NSMenu.alloc().init()
    status_line = menu.addItemWithTitle_action_keyEquivalent_("Gesture control: on", None, "")
    status_line.setEnabled_(False)
    menu.addItem_(NSMenuItem.separatorItem())
    toggle_item = menu.addItemWithTitle_action_keyEquivalent_("Turn off   (⌃⌥⌘G)", "toggle:", "")
    toggle_item.setTarget_(driver)
    settings_item = menu.addItemWithTitle_action_keyEquivalent_("Settings…", "settings:", ",")
    settings_item.setTarget_(driver)
    menu.addItem_(NSMenuItem.separatorItem())
    quit_item = menu.addItemWithTitle_action_keyEquivalent_("Quit Gesture Control", "quit:", "q")
    quit_item.setTarget_(driver)
    item.setMenu_(menu)
    icons: dict = {}

    def icon(name):
        if name not in icons:
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "Gesture control")
            if img is not None:
                img.setTemplate_(True)   # follows light/dark menu bar
            icons[name] = img
        return icons[name]

    last_menu = [None]

    def update_menu():
        key = status_key()
        if key == last_menu[0]:
            return
        last_menu[0] = key
        off = key == "off"
        img = icon("hand.raised.slash" if off else ("hand.pinch" if key in ("press", "grabbed") else "hand.raised"))
        if img is not None:
            item.button().setImage_(img)
            item.button().setTitle_("")
        else:
            item.button().setTitle_("✋")
        status_line.setTitle_(STATUS.get(key, STATUS["idle"])[1])
        toggle_item.setTitle_("Turn on   (⌃⌥⌘G)" if off else "Turn off   (⌃⌥⌘G)")

    update_menu()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1 / 60, driver, "tick:", None, True)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1.0, driver, "slow:", None, True)
    redraw()   # show the status pill right away
    app.run()


class Overlay:
    """Parent-side handle. send() never blocks the caller for long; drops if the UI is gone."""

    def __init__(self, settings_html: str = ""):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=overlay_main, args=(child, settings_html), daemon=True, name="ui")
        self._proc.start()
        self.alive = True

    def send(self, msg) -> None:
        if not self.alive:
            return
        try:
            self._conn.send(msg)
        except (BrokenPipeError, OSError):
            self.alive = False

    def poll_events(self) -> list:
        """Events from the UI process, in order: "toggle", "quit", ("set", {...})."""
        events: list = []
        try:
            while self.alive and self._conn.poll():
                events.append(self._conn.recv())
        except (EOFError, OSError):
            self.alive = False
        if not self._proc.is_alive():
            self.alive = False
            events.append("quit")   # the UI process died: nothing left to control the app with
        return events

    def wait(self, timeout: float) -> None:
        self._proc.join(timeout)

    def close(self) -> None:
        self.send(None)
        self._proc.join(timeout=1.0)
        if self._proc.is_alive():
            self._proc.terminate()
