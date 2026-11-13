"""Config: config/default.yaml, overlaid by the user's settings (written by the settings window),
optionally overlaid by a file passed with --config."""
from __future__ import annotations

from pathlib import Path

import yaml

from paths import DEFAULT_CONFIG, USER_SETTINGS

DEFAULT_PATH = DEFAULT_CONFIG

# Settings the settings window may change: dotted key → (type, allowed range or choices)
USER_KEYS = {
    "control.pointer_speed": (float, (0.4, 2.5)),
    "control.sounds": (bool, None),
    "ui.show_status": (bool, None),
    "camera.device": (str, None),
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def load(path: Path | str | None = None, user: bool = True) -> dict:
    cfg = yaml.safe_load(DEFAULT_PATH.read_text())
    if user:
        cfg = _merge(cfg, sanitize(_flatten(_read(USER_SETTINGS))))
    if path:
        cfg = _merge(cfg, _read(Path(path)))
    return cfg


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _nest(flat: dict) -> dict:
    out: dict = {}
    for key, v in flat.items():
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    return out


def sanitize(flat: dict) -> dict:
    """Keep only known user keys with valid values; returns them nested."""
    clean = {}
    for key, v in flat.items():
        if key not in USER_KEYS:
            continue
        typ, rng = USER_KEYS[key]
        try:
            if typ is bool:
                if not isinstance(v, bool):
                    continue
            else:
                v = typ(v)
        except (TypeError, ValueError):
            continue
        if rng and typ is float:
            v = min(max(v, rng[0]), rng[1])
        clean[key] = v
    return _nest(clean)


def apply_user(cfg: dict, changes: dict) -> dict:
    """Validate {dotted key: value}, apply in place to cfg, persist. Returns what was applied."""
    applied = _flatten(sanitize(changes))
    for key, v in applied.items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    if applied:
        saved = _flatten(_read(USER_SETTINGS))
        saved.update(applied)
        USER_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        tmp = USER_SETTINGS.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(_nest(_flatten(sanitize(saved))), sort_keys=True))
        tmp.replace(USER_SETTINGS)
    return applied


def user_settings_exist() -> bool:
    return USER_SETTINGS.exists()


def mark_onboarded() -> None:
    """Create the settings file (so the welcome window only opens automatically once)."""
    if not USER_SETTINGS.exists():
        USER_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        USER_SETTINGS.write_text("{}\n")
