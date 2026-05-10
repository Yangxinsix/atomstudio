from __future__ import annotations

import json
from typing import Any

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None
    QtGui = None
    QtWidgets = None


def _format_mapping(data: dict[str, Any] | None) -> str:
    if not data:
        return "No selection"
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False)


def _atom_display_payload(atom: dict[str, Any]) -> dict[str, Any]:
    position = atom.get("position") or (0.0, 0.0, 0.0)
    color = atom.get("color")
    if color is None:
        color = atom.get("face_color")
    return {
        "symbol": atom.get("symbol"),
        "position": ", ".join(f"{float(value):.5f}" for value in list(position)[:3]),
        "color": color,
        "radius": atom.get("radius"),
    }


def _payload_atoms(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    objects = payload.get("objects")
    if isinstance(objects, list):
        return [dict(item) for item in objects if isinstance(item, dict) and item.get("symbol") is not None]
    obj = payload.get("object")
    if isinstance(obj, dict) and obj.get("symbol") is not None:
        return [dict(obj)]
    return []


def _color_css(color: Any) -> str:
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return "#b8b8b8"
    values = [max(0, min(255, int(round(float(value) * 255.0)))) for value in color[:3]]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _color_text(color: Any) -> str:
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return ""
    values = [float(value) for value in color[:4]]
    if len(values) == 3:
        values.append(1.0)
    return ", ".join(f"{value:.4f}" for value in values)


def _parse_float_tuple(text: str, *, length: int) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]
    if len(values) != int(length):
        raise ValueError(f"Expected {length} comma-separated values")
    return tuple(values)


if QtWidgets is not None:  # pragma: no cover - exercised only when Qt is installed

    class SelectionInspector(QtWidgets.QTabWidget):
        def __init__(self, parent: Any | None = None) -> None:
            super().__init__(parent)
            self._payload: dict[str, Any] | None = None
            self._object_view = QtWidgets.QPlainTextEdit()
            self._summary_view = QtWidgets.QPlainTextEdit()
            self._metadata_view = QtWidgets.QPlainTextEdit()
            self._object_view.setObjectName("inspectorObjectView")
            self._summary_view.setObjectName("inspectorSummaryView")
            self._metadata_view.setObjectName("inspectorMetadataView")
            for view in (self._object_view, self._summary_view, self._metadata_view):
                view.setReadOnly(True)
                view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

            self.addTab(self._object_view, "Output")
            self.addTab(self._summary_view, "Summary")
            self.addTab(self._metadata_view, "Metadata")
            self.clear()

        def set_payload(self, payload: dict[str, Any] | None) -> None:
            self._payload = payload
            self._object_view.setPlainText(self._format_object_text(payload))
            self._metadata_view.setPlainText(_format_mapping(payload.get("metadata") if payload else None))

        def set_output_text(self, text: str) -> None:
            self._object_view.setPlainText(str(text or ""))
            self.setCurrentWidget(self._object_view)

        def _format_object_text(self, payload: dict[str, Any] | None) -> str:
            atoms = _payload_atoms(payload)
            if atoms:
                return "\n\n".join(json.dumps(_atom_display_payload(atom), indent=2, ensure_ascii=False) for atom in atoms)
            return _format_mapping(payload.get("object") if payload else None)

        def set_summary(self, summary: str) -> None:
            self._summary_view.setPlainText(str(summary or "No structure loaded"))

        def clear(self) -> None:
            self.set_payload(None)

else:  # pragma: no cover - importable fallback for tests without Qt

    class SelectionInspector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required to instantiate SelectionInspector")
