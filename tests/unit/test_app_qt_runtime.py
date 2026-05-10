from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from atomstudio.app.main import _configure_qt_runtime


def test_qt_runtime_prefers_x11_when_display_is_available(monkeypatch) -> None:
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("DISPLAY", "localhost:0.0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    _configure_qt_runtime()

    assert os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_qt_runtime_uses_wayland_without_display(monkeypatch) -> None:
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    _configure_qt_runtime()

    assert os.environ["QT_QPA_PLATFORM"] == "wayland"


def test_qt_runtime_preserves_explicit_platform(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("DISPLAY", "localhost:0.0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    _configure_qt_runtime()

    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
