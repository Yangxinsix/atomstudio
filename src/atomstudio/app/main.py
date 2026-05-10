from __future__ import annotations

import sys
from typing import Any

from atomstudio.render.config_resolver import load_batch_config
from atomstudio.app.runtime import configure_qt_runtime


configure_qt_runtime()
_configure_qt_runtime = configure_qt_runtime

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtWidgets = None


def main(argv: list[str] | None = None, *, state: Any | None = None) -> int:
    if QtWidgets is None:
        raise RuntimeError("PySide6 is required to run the AtomStudio desktop application")

    from .window import AtomStudioWindow

    args = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(args)
    window = AtomStudioWindow(state=state)
    window.show()
    return int(app.exec())


def run_app(
    *,
    input_path: str | None = None,
    config_path: str | None = None,
    frame: str = "last",
    state: Any | None = None,
) -> int:
    if QtWidgets is None:
        raise RuntimeError("PySide6 is required to run the AtomStudio desktop application")

    from .window import AtomStudioWindow

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(list(sys.argv))

    window = AtomStudioWindow(state=state)
    if config_path:
        batch = load_batch_config(config_path)
        if len(batch.jobs) != 1:
            raise ValueError("Desktop app currently accepts config files with exactly one job")
        cfg = batch.jobs[0]
        window.set_render_config(cfg)
        input_path = input_path or cfg.input.path
        if frame == "last" and cfg.input.frames:
            frame = cfg.input.frames
    window.show()
    if input_path:
        window.load_input(input_path, frame)
    return int(app.exec()) if owns_app else 0


if __name__ == "__main__":
    raise SystemExit(main())
