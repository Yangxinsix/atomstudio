from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None


if QtCore is not None:  # pragma: no cover - exercised by Qt tests

    class PreviewInputController(QtCore.QObject):
        """Owns keyboard semantics for preview surfaces.

        Preview widgets/windows can be embedded, native-child, or detached
        top-level surfaces. Qt may deliver Esc as ShortcutOverride, KeyPress, or
        KeyRelease depending on which object owns focus. This controller
        centralizes that policy: inside preview, Esc only cancels/clears preview
        state.
        """

        def __init__(self, preview: Any, *, parent: Any | None = None) -> None:
            super().__init__(parent if parent is not None else preview)
            self.preview = preview

        def install_on(self, *targets: Any) -> None:
            for target in targets:
                if target is None or not hasattr(target, "installEventFilter"):
                    continue
                try:
                    target.installEventFilter(self)
                except Exception:
                    continue

        def eventFilter(self, _target: Any, event: Any) -> bool:
            if self.consume_event(event):
                return True
            return False

        def consume_event(self, event: Any) -> bool:
            if QtCore is None or not hasattr(event, "type"):
                return False
            event_type = event.type()
            if not hasattr(event, "key") or event.key() != QtCore.Qt.Key.Key_Escape:
                return False
            if event_type == QtCore.QEvent.Type.ShortcutOverride:
                event.accept()
                return True
            if event_type in {QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease}:
                if event_type == QtCore.QEvent.Type.KeyPress:
                    self._cancel_preview_interaction()
                event.accept()
                return True
            return False

        def consume_vispy_key_event(self, event: Any) -> bool:
            if not self._is_vispy_escape_event(event):
                return False
            self._cancel_preview_interaction()
            try:
                event.handled = True
            except Exception:
                pass
            return True

        def _cancel_preview_interaction(self) -> None:
            cancel = getattr(self.preview, "cancel_interaction", None)
            if callable(cancel):
                cancel()
                return
            clear = getattr(self.preview, "clear_selection", None)
            if callable(clear):
                clear()

        @staticmethod
        def _is_vispy_escape_event(event: Any) -> bool:
            key = getattr(event, "key", None)
            name = getattr(key, "name", None)
            candidates = (name, key)
            return any(str(item).strip().lower() in {"escape", "esc"} for item in candidates if item is not None)


else:  # pragma: no cover

    class PreviewInputController:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.preview = args[0] if args else None

        def install_on(self, *targets: Any) -> None:
            return None

        def consume_event(self, event: Any) -> bool:
            return False

        def consume_vispy_key_event(self, event: Any) -> bool:
            return False


__all__ = ["PreviewInputController"]
