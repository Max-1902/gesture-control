"""User settings: only known keys with valid values are applied and saved."""
import pytest

import config


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.yaml"
    monkeypatch.setattr(config, "USER_SETTINGS", path)
    return path


def test_apply_and_reload(settings_file):
    cfg = config.load()
    applied = config.apply_user(cfg, {"control.pointer_speed": 1.5, "control.sounds": False})
    assert applied == {"control.pointer_speed": 1.5, "control.sounds": False}
    assert cfg["control"]["pointer_speed"] == 1.5
    again = config.load()
    assert again["control"]["pointer_speed"] == 1.5 and again["control"]["sounds"] is False
    assert again["control"]["pointer_gain_fast"] == cfg["control"]["pointer_gain_fast"]   # defaults kept


def test_rejects_unknown_and_invalid(settings_file):
    cfg = config.load()
    applied = config.apply_user(cfg, {"gestures.pinch_on": 0.9, "control.sounds": "yes",
                                      "control.pointer_speed": 99})
    assert applied == {"control.pointer_speed": 2.5}   # clamped; others dropped
    assert cfg["gestures"]["pinch_on"] != 0.9


def test_corrupt_file_is_ignored(settings_file):
    settings_file.write_text("control: [this is: not valid")
    assert config.load()["control"]["pointer_speed"] == 1.0
    settings_file.write_text("- a list\n")
    assert config.load()["control"]["pointer_speed"] == 1.0
