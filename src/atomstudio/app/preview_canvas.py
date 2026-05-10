"""Legacy VisPy preview path.

Rendering feature work in this module is frozen during the migration to the
QOpenGLWidget backend described in ``docs/opengl-preview-backend.md``. Keep
changes here limited to compatibility fixes required to keep the current app
usable while the OpenGL backend is brought up.
"""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from atomstudio.app.runtime import configure_preview_runtime
from atomstudio.app.mouse_tools import PreviewMouseToolController
from atomstudio.config import RenderJobConfig
from atomstudio.preview.lighting import DEFAULT_PREVIEW_LIGHTING, configure_shading_filter
from atomstudio.preview.mesh_builder import (
    build_atom_instance_payload,
    build_bond_instance_payload,
    build_cell_visual_payload,
    build_poly_visual_payload,
)
from atomstudio.preview.renderer import (
    CallbackSignal,
    PreviewCameraState,
    PreviewCanvasModel,
    PreviewRenderScene,
    PreviewSettings,
    build_preview_scene,
)
from atomstudio.preview.selection_visuals import build_selection_shell_payload, empty_selection_shell_payload
from atomstudio.preview.picking import rotation_basis, segment_distance_2d
from atomstudio.preview.interaction import AtomHit, HitTestCache
from atomstudio.preview.types import PreviewScene as BufferPreviewScene
from atomstudio.preview.types import PreviewSelection
from atomstudio.structure.structure import Structure


configure_preview_runtime()

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover - fallback for test environments
    QtCore = None
    QtGui = None

    class _FallbackWidget:
        def __init__(self, *args, **kwargs) -> None:
            self._layout = None
            self._visible = True

        def setLayout(self, layout) -> None:
            self._layout = layout

        def layout(self):
            return self._layout

        def setMinimumSize(self, *_args, **_kwargs) -> None:
            return None

        def setFixedSize(self, *_args, **_kwargs) -> None:
            return None

        def setAttribute(self, *_args, **_kwargs) -> None:
            return None

        def setVisible(self, visible: bool) -> None:
            self._visible = bool(visible)

        def isVisible(self) -> bool:
            return bool(self._visible)

        def size(self):
            return SimpleNamespace(width=lambda: 800, height=lambda: 600)

        def setFocusPolicy(self, *_args, **_kwargs) -> None:
            return None

        def setFocus(self) -> None:
            return None

    class _FallbackLayout:
        def __init__(self, *args, **kwargs) -> None:
            self._items: list[Any] = []

        def addWidget(self, widget, *args, **kwargs) -> None:
            self._items.append(widget)

        def setContentsMargins(self, *_args) -> None:
            return None

        def setSpacing(self, *_args) -> None:
            return None

    QtWidgets = SimpleNamespace(QWidget=_FallbackWidget, QGridLayout=_FallbackLayout, QVBoxLayout=_FallbackLayout)

try:  # pragma: no cover - optional GUI dependency
    from vispy import scene as vispy_scene  # type: ignore
    from vispy.geometry import create_cylinder, create_sphere  # type: ignore
except Exception:  # pragma: no cover - fallback for test environments
    create_cylinder = None
    create_sphere = None

    class _FallbackVisual:
        def __init__(self, *args, **kwargs) -> None:
            self.data: dict[str, Any] = {}
            self.parent = kwargs.get("parent")
            self.visible = True

        def set_data(self, **kwargs) -> None:
            self.data = dict(kwargs)

        def set_gl_state(self, *args, **kwargs) -> None:
            return None

        def update(self) -> None:
            return None

    class _FallbackAxis(_FallbackVisual):
        pass

    class _FallbackCamera:
        def __init__(self, *args, **kwargs) -> None:
            self.center = tuple(kwargs.get("center", (0.0, 0.0, 0.0)))
            self.azimuth = float(kwargs.get("azimuth", 45.0))
            self.elevation = float(kwargs.get("elevation", 30.0))
            self.roll = float(kwargs.get("roll", 0.0))
            self.distance = float(kwargs.get("distance", 10.0))
            self.scale_factor = float(kwargs.get("scale_factor", 1.0))
            self.fov = float(kwargs.get("fov", 0.0))
            self.events = SimpleNamespace(transform_change=SimpleNamespace(connect=lambda *_args, **_kwargs: None))

    class _FallbackView:
        def __init__(self) -> None:
            self.scene = SimpleNamespace()
            self.camera = _FallbackCamera()
            self._children: list[Any] = []

        def add(self, visual):
            self._children.append(visual)
            return visual

    class _FallbackCentralWidget:
        def add_view(self) -> _FallbackView:
            return _FallbackView()

    class _FallbackCanvas:
        def __init__(self, *args, **kwargs) -> None:
            self.native = QtWidgets.QWidget()
            self.central_widget = _FallbackCentralWidget()
            self.events = SimpleNamespace(
                mouse_press=SimpleNamespace(connect=lambda *_args, **_kwargs: None),
                draw=SimpleNamespace(connect=lambda *_args, **_kwargs: None),
            )

        def update(self) -> None:
            return None

    vispy_scene = SimpleNamespace(
        SceneCanvas=_FallbackCanvas,
        visuals=SimpleNamespace(Markers=_FallbackVisual, Line=_FallbackVisual, Mesh=_FallbackVisual, XYZAxis=_FallbackAxis),
        cameras=SimpleNamespace(TurntableCamera=_FallbackCamera),
    )


if QtWidgets is not None:  # pragma: no cover - exercised only when GUI deps are installed

    class AxisOverlayWidget(QtWidgets.QWidget):
        ORIGIN = (60.0, 84.0)
        AXIS_LENGTH = 48.0
        MIN_PROJECTED_LENGTH = 0.18

        def __init__(self, parent: Any | None = None, *, background: tuple[float, float, float, float]) -> None:
            super().__init__(parent)
            self.setFixedSize(128, 128)
            if QtCore is not None:
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self._background = tuple(float(v) for v in background)
            self._camera = PreviewCameraState()

        def sync_camera(self, camera: PreviewCameraState) -> None:
            # VESTA-style orientation marker: fixed UI overlay whose axes track view rotation only.
            self._camera = replace(camera)
            self.update()

        def paintEvent(self, _event: Any) -> None:
            if QtGui is None or QtCore is None:
                return
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self._draw_panel(painter)
            origin = QtCore.QPointF(*self.ORIGIN)
            axes = [
                ("X", (1.0, 0.0, 0.0), QtGui.QColor(210, 54, 45)),
                ("Y", (0.0, 1.0, 0.0), QtGui.QColor(40, 150, 72)),
                ("Z", (0.0, 0.0, 1.0), QtGui.QColor(42, 102, 210)),
            ]
            for label, _axis, color, endpoint, _projected_len, _depth in sorted(
                (
                    (label, axis, color, *self._axis_projection(origin, axis))
                    for label, axis, color in axes
                ),
                key=lambda item: item[5],
            ):
                self._draw_axis(painter, origin, endpoint, label, color)

        def _draw_panel(self, painter: Any) -> None:
            alpha = 36 if self._background[3] >= 0.9 else 20
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255, alpha))
            painter.drawRoundedRect(QtCore.QRectF(8.0, 8.0, 112.0, 112.0), 12.0, 12.0)

        def _draw_axis(self, painter: Any, start: Any, end: Any, label: str, color: Any) -> None:
            vector = end - start
            length = max(1.0, (vector.x() ** 2 + vector.y() ** 2) ** 0.5)
            direction = QtCore.QPointF(vector.x() / length, vector.y() / length)
            normal = QtCore.QPointF(-direction.y(), direction.x())
            arrow_a = end - direction * 9.0 + normal * 4.5
            arrow_b = end - direction * 9.0 - normal * 4.5

            pen = QtGui.QPen(color, 3.0)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(start, end)
            painter.setBrush(color)
            painter.drawPolygon(QtGui.QPolygonF([end, arrow_a, arrow_b]))

            font = painter.font()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(end + direction * 8.0 + normal * 2.0, label)

        def _project_axis(self, origin: Any, axis: tuple[float, float, float]) -> Any:
            return self._axis_projection(origin, axis)[0]

        def _axis_projection(self, origin: Any, axis: tuple[float, float, float]) -> tuple[Any, float, float]:
            right, up, forward = self._camera.basis()
            x = self._dot(axis, right)
            y = self._dot(axis, up)
            depth = self._dot(axis, forward)
            projected_len = (x * x + y * y) ** 0.5
            if projected_len < 0.12:
                y = -self.MIN_PROJECTED_LENGTH if depth < 0.0 else self.MIN_PROJECTED_LENGTH
                x = 0.0
                projected_len = self.MIN_PROJECTED_LENGTH
            visible_len = max(self.MIN_PROJECTED_LENGTH, projected_len)
            scale = self.AXIS_LENGTH * visible_len
            endpoint = origin + QtCore.QPointF((x / visible_len) * scale, -(y / visible_len) * scale)
            return endpoint, projected_len, depth

        def _axis_depth(self, axis: tuple[float, float, float]) -> float:
            return self._dot(axis, self._camera.forward)

        @staticmethod
        def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
            return float(left[0] * right[0] + left[1] * right[1] + left[2] * right[2])


    class PreviewCanvas(QtWidgets.QWidget):
        def __init__(self, parent: Any | None = None, *, settings: PreviewSettings | None = None) -> None:
            super().__init__(parent)
            self.settings = settings or PreviewSettings()
            self.model = PreviewCanvasModel(self.settings)
            self.model.selection_changed.connect(self._on_selection_changed)
            self.selection_changed = self.model.selection_changed
            self.interaction_message_changed = CallbackSignal()
            self.renderer_mode = "instanced"
            self._mesh_enabled = bool(create_sphere is not None and create_cylinder is not None)
            self._instancing_status = self._detect_instancing_status()
            if not self._mesh_enabled:
                raise RuntimeError("AtomStudio preview requires VisPy sphere/cylinder geometry support")
            if not self._instancing_status["supported"]:
                raise RuntimeError(
                    "AtomStudio preview requires instanced OpenGL rendering; "
                    f"{self._instancing_status['backend']} is not supported: {self._instancing_status['reason']}"
                )
            if not hasattr(getattr(vispy_scene, "visuals", None), "InstancedMesh"):
                raise RuntimeError("AtomStudio preview requires vispy.scene.visuals.InstancedMesh")

            self._canvas = vispy_scene.SceneCanvas(keys="interactive", bgcolor=self.settings.background, show=False, parent=self)
            self._view = self._canvas.central_widget.add_view()
            self._view.camera = vispy_scene.cameras.TurntableCamera(
                azimuth=self.model.camera.azimuth,
                elevation=self.model.camera.elevation,
                center=self.model.camera.center,
                scale_factor=self.model.camera.scale_factor,
                fov=0.0,
            )
            self._sphere_mesh = create_sphere(rows=14, cols=16, radius=1.0) if create_sphere is not None else None
            self._cylinder_mesh = (
                create_cylinder(2, 16, radius=[1.0, 1.0], length=1.0, offset=False) if create_cylinder is not None else None
            )
            visual_parent = getattr(self._view, "scene", None)
            self._atom_instance_visual = self._create_instanced_visual(self._sphere_mesh, visual_parent)
            self._bond_instance_visual = self._create_instanced_visual(self._cylinder_mesh, visual_parent)
            self._selection_shell_visual = vispy_scene.visuals.Mesh(parent=visual_parent, shading="smooth")
            self._cell_visual = vispy_scene.visuals.Line(parent=visual_parent)
            self._poly_visual = vispy_scene.visuals.Mesh(parent=visual_parent, shading="smooth")
            self._poly_edge_visual = vispy_scene.visuals.Line(parent=visual_parent)
            self._configure_selection_visuals()
            self._axis_overlay = AxisOverlayWidget(self, background=self.settings.background)
            self._axis_overlay_visible = True
            self._camera_sync_pending = False
            self._last_camera_signature: tuple[Any, ...] | None = None
            self._lighting = DEFAULT_PREVIEW_LIGHTING
            self._graphics_info: dict[str, Any] = {
                "opengl_version": "unavailable",
                "renderer": "VisPy/OpenGL preview",
                "max_viewport_dims": "unavailable",
                "depth_bits": None,
                "qt_platform": "unavailable",
                "display": "",
                "wayland_display": "",
                "vispy_backend": "unavailable",
                "gl_backend": self._instancing_status["backend"],
                "instancing_supported": self._instancing_status["supported"],
                "instancing_reason": self._instancing_status["reason"],
                "preview_renderer": "instanced",
                "preview_instances": {},
            }
            self._graphics_info_captured = False
            self._install_layout()
            self._mouse_tools = PreviewMouseToolController(self, getattr(self._canvas, "native", None), parent=self)
            self._mouse_tools.install()
            self._bind_events()
            self._sync_camera()

        def _create_instanced_visual(self, mesh_data: Any, parent: Any) -> Any | None:
            if mesh_data is None:
                raise RuntimeError("AtomStudio preview requires non-empty instanced mesh base geometry")
            dummy_positions = np.zeros((1, 3), dtype=np.float32)
            dummy_transforms = np.zeros((1, 3, 3), dtype=np.float32)
            dummy_colors = np.ones((1, 4), dtype=np.float32)
            visual = vispy_scene.visuals.InstancedMesh(
                meshdata=mesh_data,
                instance_positions=dummy_positions,
                instance_transforms=dummy_transforms,
                instance_colors=dummy_colors,
                parent=parent,
                shading="smooth",
                color=(1.0, 1.0, 1.0, 1.0),
            )
            try:
                visual.set_gl_state("opaque", depth_test=True, cull_face=False)
            except Exception:
                pass
            visual.visible = False
            return visual

        @staticmethod
        def _detect_instancing_status() -> dict[str, Any]:
            try:
                from vispy import gloo  # type: ignore

                backend = getattr(getattr(gloo, "gl", None), "current_backend", None)
                backend_name = getattr(backend, "__name__", str(backend or "unavailable"))
                required = ("glVertexAttribDivisor", "glDrawElementsInstanced", "glDrawArraysInstanced")
                missing = tuple(name for name in required if not PreviewCanvas._has_gl_function(gloo.gl, name))
                if not (
                    PreviewCanvas._has_gl_function(gloo.gl, "glCreateFramebuffer")
                    and PreviewCanvas._has_gl_function(gloo.gl, "glBindFramebuffer")
                    and PreviewCanvas._has_gl_function(gloo.gl, "glCheckFramebufferStatus")
                ):
                    missing = (*missing, "framebuffer helpers")
                if missing:
                    return {
                        "supported": False,
                        "backend": str(backend_name),
                        "reason": "missing " + ", ".join(missing),
                    }
                return {"supported": True, "backend": str(backend_name), "reason": "available"}
            except Exception as exc:
                return {"supported": False, "backend": "unavailable", "reason": str(exc)}

        @staticmethod
        def _has_gl_function(backend: Any, name: str) -> bool:
            func = getattr(backend, name, None)
            try:
                return bool(func)
            except Exception:
                return func is not None

        def _install_layout(self) -> None:
            layout = QtWidgets.QGridLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self._canvas.native, 0, 0)
            align = 0
            if QtCore is not None and hasattr(QtCore, "Qt"):
                align = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom
            layout.addWidget(self._axis_overlay, 0, 0, alignment=align)
            self.setLayout(layout)

        def _bind_events(self) -> None:
            events = getattr(self._canvas, "events", None)
            if events is not None and getattr(events, "mouse_press", None) is not None:
                events.mouse_press.connect(self._mouse_tools.handle_press)
            if events is not None and getattr(events, "mouse_double_click", None) is not None:
                events.mouse_double_click.connect(self._mouse_tools.handle_double_click)
            if events is not None and getattr(events, "mouse_move", None) is not None:
                events.mouse_move.connect(self._mouse_tools.handle_move)
            if events is not None and getattr(events, "mouse_release", None) is not None:
                events.mouse_release.connect(self._mouse_tools.handle_release)
            if events is not None and getattr(events, "draw", None) is not None:
                events.draw.connect(self._capture_graphics_info_once)
            for event_name in ("mouse_move", "mouse_release", "mouse_wheel"):
                event = getattr(events, event_name, None) if events is not None else None
                if event is not None:
                    event.connect(self._schedule_camera_sync)
            camera_events = getattr(getattr(self._view, "camera", None), "events", None)
            if camera_events is not None and getattr(camera_events, "transform_change", None) is not None:
                camera_events.transform_change.connect(self._on_camera_transform)

        def _schedule_camera_sync(self, _event: Any = None) -> None:
            if self._camera_sync_pending:
                return
            self._camera_sync_pending = True
            if QtCore is not None:
                QtCore.QTimer.singleShot(0, self._on_camera_transform)
                return
            self._on_camera_transform()

        def _on_camera_transform(self, _event: Any = None) -> None:
            self._camera_sync_pending = False
            self._pull_camera_state_from_view()
            signature = self._camera_signature()
            if signature == self._last_camera_signature:
                return
            self._last_camera_signature = signature
            self._apply_preview_lighting()
            self._sync_axis_overlay()

        def _camera_signature(self) -> tuple[Any, ...]:
            camera = self.model.camera
            return (
                tuple(round(float(v), 6) for v in camera.center),
                round(float(camera.scale_factor), 6),
                round(float(camera.azimuth), 6),
                round(float(camera.elevation), 6),
                round(float(camera.roll), 6),
                tuple(round(float(v), 6) for v in camera.right),
                tuple(round(float(v), 6) for v in camera.up),
                tuple(round(float(v), 6) for v in camera.forward),
            )

        def _pull_camera_state_from_view(self) -> None:
            camera = getattr(self._view, "camera", None)
            if camera is None:
                return
            self.model.camera.center = tuple(float(v) for v in getattr(camera, "center", self.model.camera.center))
            self.model.camera.scale_factor = float(getattr(camera, "scale_factor", self.model.camera.scale_factor))
            self.model.camera.azimuth = float(getattr(camera, "azimuth", self.model.camera.azimuth))
            self.model.camera.elevation = float(getattr(camera, "elevation", self.model.camera.elevation))
            self.model.camera.roll = float(getattr(camera, "roll", self.model.camera.roll))
            self.model.camera.right, self.model.camera.up, self.model.camera.forward = rotation_basis(
                self.model.camera.azimuth,
                self.model.camera.elevation,
                self.model.camera.roll,
            )

        def _sync_axis_overlay(self) -> None:
            if self._axis_overlay_visible:
                self._axis_overlay.sync_camera(self.model.camera)

        def set_axis_overlay_visible(self, visible: bool) -> None:
            self._axis_overlay_visible = bool(visible)
            self._axis_overlay.setVisible(bool(visible))
            if visible:
                self._sync_axis_overlay()

        def _sync_camera(self) -> None:
            camera = self.model.camera
            self._set_view_camera(camera.center, camera.scale_factor, camera.azimuth, camera.elevation, camera.roll)
            self._pull_camera_state_from_view()
            self._apply_preview_lighting()
            self._sync_axis_overlay()

        def _set_view_camera(
            self,
            center: tuple[float, float, float],
            scale_factor: float,
            azimuth: float,
            elevation: float,
            roll: float,
        ) -> None:
            camera = self._view.camera
            camera.center = center
            camera.scale_factor = scale_factor
            if hasattr(camera, "fov"):
                camera.fov = self._preview_camera_fov()
            # TurntableCamera property setters clamp elevation and wrap azimuth/roll.
            # Toolbar rotations need to continue indefinitely, so write through the
            # backing fields when available and then notify VisPy.
            if hasattr(camera, "_azimuth"):
                camera._azimuth = float(azimuth)
            else:
                camera.azimuth = float(azimuth)
            if hasattr(camera, "_elevation"):
                camera._elevation = float(elevation)
            else:
                camera.elevation = float(elevation)
            if hasattr(camera, "_roll"):
                camera._roll = float(roll)
            elif hasattr(camera, "roll"):
                camera.roll = float(roll)
            view_changed = getattr(camera, "view_changed", None)
            if callable(view_changed):
                view_changed()

        def _preview_camera_fov(self) -> float:
            cfg = getattr(self.model, "cfg", None)
            camera_cfg = getattr(cfg, "camera", None)
            projection = str(getattr(camera_cfg, "projection", "ORTHOGRAPHIC") or "ORTHOGRAPHIC").upper()
            return 45.0 if projection.startswith("PERSP") else 0.0

        def _apply_preview_lighting(self) -> None:
            configure_shading_filter(self._atom_instance_visual, self.model.camera, self._lighting)
            configure_shading_filter(self._bond_instance_visual, self.model.camera, self._lighting)
            configure_shading_filter(self._poly_visual, self.model.camera, self._lighting)
            configure_shading_filter(self._selection_shell_visual, self.model.camera, self._lighting)

        def _on_selection_changed(self, _selection: PreviewSelection | None) -> None:
            self._refresh_visuals()

        def set_mouse_mode(self, mode: str) -> str:
            return self._mouse_tools.set_mode(mode)

        @property
        def mouse_mode(self) -> str:
            return self._mouse_tools.mode

        def set_camera_mouse_enabled(self, enabled: bool) -> None:
            camera = getattr(self._view, "camera", None)
            if camera is not None and hasattr(camera, "interactive"):
                camera.interactive = bool(enabled)

        def emit_interaction_message(self, message: str) -> None:
            self.interaction_message_changed.emit(str(message))

        def atom_selection_at_screen(self, point: tuple[float, float]) -> PreviewSelection | None:
            return self._atom_hit_test_cache().pick_atom(point)

        def bond_selection_at_screen(self, point: tuple[float, float]) -> PreviewSelection | None:
            if self.model.scene is None:
                return None
            transform = self._scene_to_canvas_transform()
            if transform is None:
                return None
            candidates: list[tuple[float, float, int]] = []
            for bond in self.model.scene.bonds:
                for segment in bond.segments:
                    start = self._map_scene_point(transform, segment.start)
                    end = self._map_scene_point(transform, segment.end)
                    if start is None or end is None:
                        continue
                    distance, weight = segment_distance_2d(point, (start[0], start[1]), (end[0], end[1]))
                    threshold = max(6.0, 0.5 * float(segment.width_px))
                    if distance > threshold:
                        continue
                    depth = (1.0 - weight) * self._view_depth(segment.start) + weight * self._view_depth(segment.end)
                    candidates.append((float(depth), float(distance), int(bond.index)))
            if not candidates:
                return None
            _depth, _distance, index = min(candidates)
            return PreviewSelection(kind="bond", index=int(index))

        def object_selection_at_screen(self, point: tuple[float, float]) -> PreviewSelection | None:
            atom_selection = self.atom_selection_at_screen(point)
            if atom_selection is not None:
                return atom_selection
            return self.bond_selection_at_screen(point)

        def atom_screen_positions(self) -> dict[int, tuple[float, float, float]]:
            return {
                int(hit.index): (float(hit.x), float(hit.y), float(hit.depth))
                for hit in self._atom_hit_test_cache().atoms
            }

        def select_atom_at_screen(
            self,
            point: tuple[float, float],
            *,
            toggle: bool = False,
            clear_on_miss: bool = False,
        ) -> PreviewSelection | None:
            selection = self.atom_selection_at_screen(point)
            if selection is None:
                if clear_on_miss:
                    self.model.clear_selection()
                    self._refresh_visuals()
                    self._canvas.update()
                return None
            if toggle:
                selection = self.model.toggle_atom_selection(int(selection.index))
            else:
                selection = self.model.select_atom(int(selection.index))
            self._refresh_visuals()
            self._canvas.update()
            if selection is not None:
                self.interaction_message_changed.emit(f"Selected atom {selection.index}")
            return selection

        def select_object_at_screen(
            self,
            point: tuple[float, float],
            *,
            toggle: bool = False,
            clear_on_miss: bool = False,
        ) -> PreviewSelection | None:
            selection = self.object_selection_at_screen(point)
            if selection is None:
                if clear_on_miss:
                    self.model.clear_selection()
                    self._refresh_visuals()
                    self._canvas.update()
                return None
            if selection.kind == "atom":
                if toggle:
                    selection = self.model.toggle_atom_selection(int(selection.index))
                else:
                    selection = self.model.select_atom(int(selection.index))
                message = f"Selected atom {selection.index}" if selection is not None else "Selection cleared"
            else:
                if toggle:
                    selection = self.model.toggle_bond_selection(int(selection.index))
                else:
                    selection = self.model.select_bond(int(selection.index))
                message = f"Selected bond {selection.index}"
            self._refresh_visuals()
            self._canvas.update()
            self.interaction_message_changed.emit(message)
            return selection

        def select_atom_indices(self, atom_indices: set[int]) -> PreviewSelection | None:
            selection = self.model.select_atoms(set(atom_indices))
            self._refresh_visuals()
            self._canvas.update()
            return selection

        def select_atoms_in_screen_rect(
            self,
            start: tuple[float, float],
            end: tuple[float, float],
            *,
            append: bool = False,
        ) -> tuple[int, ...]:
            indices = self._atom_hit_test_cache().atoms_in_rect(start, end)
            self.model.select_atoms(set(indices), append=append)
            self._refresh_visuals()
            self._canvas.update()
            return indices

        def select_objects_in_screen_rect(
            self,
            start: tuple[float, float],
            end: tuple[float, float],
            *,
            append: bool = False,
        ) -> dict[str, tuple[int, ...]]:
            atom_indices = self._atom_hit_test_cache().atoms_in_rect(start, end)
            bond_indices = self._bond_indices_in_screen_rect(start, end)
            self.model.select_objects(set(atom_indices), set(bond_indices), append=append)
            self._refresh_visuals()
            self._canvas.update()
            return {"atoms": tuple(atom_indices), "bonds": tuple(bond_indices)}

        def _bond_indices_in_screen_rect(
            self,
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> tuple[int, ...]:
            if self.model.scene is None:
                return ()
            transform = self._scene_to_canvas_transform()
            if transform is None:
                return ()
            x0, x1 = sorted((float(start[0]), float(end[0])))
            y0, y1 = sorted((float(start[1]), float(end[1])))
            rect = (x0, y0, x1, y1)
            selected: set[int] = set()
            for bond in self.model.scene.bonds:
                for segment in bond.segments:
                    a = self._map_scene_point(transform, segment.start)
                    b = self._map_scene_point(transform, segment.end)
                    if a is None or b is None:
                        continue
                    half_width = max(6.0, float(segment.width_px) * 0.5)
                    if self._screen_segment_hits_rect((a[0], a[1]), (b[0], b[1]), rect, half_width):
                        selected.add(int(bond.index))
                        break
            return tuple(sorted(selected))

        def _atom_hit_test_cache(self) -> HitTestCache:
            if self.model.scene is None:
                return HitTestCache()
            transform = self._scene_to_canvas_transform()
            if transform is None:
                self._pull_camera_state_from_view()
                return self.model.hit_test_cache(self._viewport_size())
            hits: list[AtomHit] = []
            for atom in self.model.scene.atoms:
                projected = self._map_scene_point(transform, atom.position)
                if projected is None:
                    continue
                x, y, _mapped_depth = projected
                radius_px = self._projected_atom_radius(transform, atom.position, atom.radius)
                hits.append(
                    AtomHit(
                        index=int(atom.index),
                        x=float(x),
                        y=float(y),
                        depth=float(self._view_depth(atom.position)),
                        radius_px=max(6.0, float(radius_px)),
                    )
                )
            return HitTestCache(tuple(hits))

        def _view_depth(self, point: tuple[float, float, float]) -> float:
            camera = self.model.camera
            rel = np.asarray(point, dtype=float) - np.asarray(camera.center, dtype=float)
            return float(np.dot(rel, np.asarray(camera.forward, dtype=float)))

        def _scene_to_canvas_transform(self) -> Any | None:
            scene_node = getattr(self._view, "scene", None)
            canvas_scene = getattr(self._canvas, "scene", None)
            if scene_node is None or canvas_scene is None or not hasattr(scene_node, "node_transform"):
                return None
            try:
                return scene_node.node_transform(canvas_scene)
            except Exception:
                return None

        @staticmethod
        def _map_scene_point(transform: Any, point: tuple[float, float, float]) -> tuple[float, float, float] | None:
            try:
                mapped = transform.map((float(point[0]), float(point[1]), float(point[2]), 1.0))
            except Exception:
                return None
            if len(mapped) < 4:
                return None
            w = float(mapped[3])
            if abs(w) <= 1e-12:
                return None
            return float(mapped[0]) / w, float(mapped[1]) / w, float(mapped[2]) / w

        def _projected_atom_radius(
            self,
            transform: Any,
            position: tuple[float, float, float],
            radius: float,
        ) -> float:
            center = self._map_scene_point(transform, position)
            if center is None:
                return float(self.settings.picking_radius_px)
            px = []
            base = np.asarray(position, dtype=float)
            radius_value = max(1e-6, float(radius))
            for axis in np.eye(3, dtype=float):
                mapped = self._map_scene_point(transform, tuple(float(v) for v in base + axis * radius_value))
                if mapped is None:
                    continue
                dx = float(mapped[0]) - float(center[0])
                dy = float(mapped[1]) - float(center[1])
                px.append((dx * dx + dy * dy) ** 0.5)
            if not px:
                return float(self.settings.picking_radius_px)
            return max(px)

        @staticmethod
        def _screen_segment_intersects_rect(
            start: tuple[float, float],
            end: tuple[float, float],
            rect: tuple[float, float, float, float],
        ) -> bool:
            x0, y0, x1, y1 = rect
            ax, ay = start
            bx, by = end
            if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
                return False
            if x0 <= ax <= x1 and y0 <= ay <= y1:
                return True
            if x0 <= bx <= x1 and y0 <= by <= y1:
                return True
            edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
            return any(PreviewCanvas._segments_intersect(start, end, left, right) for left, right in edges)

        @staticmethod
        def _screen_segment_hits_rect(
            start: tuple[float, float],
            end: tuple[float, float],
            rect: tuple[float, float, float, float],
            half_width: float,
        ) -> bool:
            pad = max(0.0, float(half_width))
            x0, y0, x1, y1 = rect
            expanded = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            return PreviewCanvas._screen_segment_intersects_rect(start, end, expanded)

        @staticmethod
        def _segments_intersect(
            a: tuple[float, float],
            b: tuple[float, float],
            c: tuple[float, float],
            d: tuple[float, float],
        ) -> bool:
            def orient(p, q, r) -> float:
                return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

            def on_segment(p, q, r) -> bool:
                return (
                    min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                    and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
                )

            o1 = orient(a, b, c)
            o2 = orient(a, b, d)
            o3 = orient(c, d, a)
            o4 = orient(c, d, b)
            if o1 * o2 < 0.0 and o3 * o4 < 0.0:
                return True
            return (
                abs(o1) <= 1e-9 and on_segment(a, c, b)
                or abs(o2) <= 1e-9 and on_segment(a, d, b)
                or abs(o3) <= 1e-9 and on_segment(c, a, d)
                or abs(o4) <= 1e-9 and on_segment(c, b, d)
            )

        def select_atoms_in_rect(
            self,
            start: tuple[float, float],
            end: tuple[float, float],
            *,
            append: bool = False,
        ) -> tuple[int, ...]:
            return self.select_atoms_in_screen_rect(start, end, append=append)

        def update_selected_atom_properties(self, updates: dict[str, Any]) -> bool:
            selection = self.model.selection
            if selection is None or selection.kind != "atom" or selection.index is None:
                return False
            changed = self.model.update_atom_properties(int(selection.index), dict(updates))
            if changed is None:
                return False
            self._refresh_visuals()
            self._canvas.update()
            self.interaction_message_changed.emit(f"Updated atom {selection.index}")
            return True

        def measurement_message(self, atom_indices: list[int]) -> str:
            return self._mouse_tools.measurement_message(self.model.scene, atom_indices)

        def _viewport_size(self) -> tuple[int, int]:
            for widget in (getattr(self._canvas, "native", None), self):
                size = getattr(widget, "size", None)
                if callable(size):
                    widget_size = size()
                    if hasattr(widget_size, "width") and hasattr(widget_size, "height"):
                        width = int(widget_size.width())
                        height = int(widget_size.height())
                        if width > 1 and height > 1:
                            return width, height
            return (800, 600)

        def set_scene(self, structure: Structure, cfg: RenderJobConfig) -> PreviewRenderScene:
            scene = self.model.set_scene(structure, cfg)
            self._sync_camera()
            self._refresh_visuals()
            self._canvas.update()
            return scene

        def set_preview_scene(
            self,
            preview_scene: PreviewRenderScene | BufferPreviewScene,
            *,
            frame_index: int | None = None,
        ) -> PreviewRenderScene:
            scene = self.model.set_preview_scene(preview_scene)
            self._sync_camera()
            self._refresh_visuals()
            self._canvas.update()
            return scene

        def update_preview_scene(
            self,
            preview_scene: PreviewRenderScene | BufferPreviewScene,
            *,
            frame_index: int | None = None,
        ) -> PreviewRenderScene:
            return self.set_preview_scene(preview_scene, frame_index=frame_index)

        def fit_to_structure(self, padding: float | None = None) -> PreviewCameraState:
            camera = self.model.fit_to_structure(padding)
            self._sync_camera()
            self._canvas.update()
            return camera

        def set_view_preset(self, view: str) -> PreviewCameraState:
            camera = self.model.set_view_preset(view)
            self._sync_camera()
            self._canvas.update()
            return camera

        def set_axis_view(self, axis: str) -> PreviewCameraState:
            axis_key = str(axis or "").strip().lower()
            preset = {"a": "side", "x": "side", "b": "front", "y": "front", "c": "top", "z": "top"}.get(axis_key, "orbit")
            return self.set_view_preset(preset)

        def rotate_view(self, axis: str, direction: int, degrees: float = 15.0) -> PreviewCameraState:
            self._pull_camera_state_from_view()
            step = float(degrees) * (1.0 if int(direction) >= 0 else -1.0)
            axis_key = str(axis or "").strip().lower()
            if axis_key in {"c", "z"}:
                self.model.camera.azimuth += step
            elif axis_key in {"a", "x"}:
                self.model.camera.elevation += step
            elif axis_key in {"b", "y"}:
                self.model.camera.roll += step
            self.model.camera.right, self.model.camera.up, self.model.camera.forward = rotation_basis(
                self.model.camera.azimuth,
                self.model.camera.elevation,
                self.model.camera.roll,
            )
            self._sync_camera()
            self._canvas.update()
            return self.model.camera

        def pan_view(self, dx: float = 0.0, dy: float = 0.0) -> PreviewCameraState:
            self._pull_camera_state_from_view()
            camera = self.model.camera
            width, height = self._viewport_size()
            step_x = max(1e-6, float(camera.scale_factor)) / max(1.0, float(width) * 0.5)
            step_y = max(1e-6, float(camera.scale_factor)) / max(1.0, float(height) * 0.5)
            right = tuple(float(v) for v in camera.right)
            up = tuple(float(v) for v in camera.up)
            center = tuple(
                float(camera.center[i]) - float(dx) * step_x * right[i] - float(dy) * step_y * up[i]
                for i in range(3)
            )
            self.model.camera.center = center
            self._sync_camera()
            self._canvas.update()
            return self.model.camera

        def zoom_view(self, factor: float) -> PreviewCameraState:
            self._pull_camera_state_from_view()
            factor = max(1e-6, float(factor))
            self.model.camera.scale_factor = max(1e-6, float(self.model.camera.scale_factor) / factor)
            self._sync_camera()
            self._canvas.update()
            return self.model.camera

        def current_camera_state(self) -> PreviewCameraState:
            self._pull_camera_state_from_view()
            return self.model.camera

        def select_preview_object(self, selection: PreviewSelection | None) -> PreviewSelection | None:
            selection = self.model.select_preview_object(selection)
            self._refresh_visuals()
            self._canvas.update()
            return selection

        def select_selection(self, selection: PreviewSelection | None) -> PreviewSelection | None:
            return self.select_preview_object(selection)

        def select_atom(self, atom_index: int | None) -> PreviewSelection | None:
            selection = self.model.select_atom(atom_index)
            self._refresh_visuals()
            self._canvas.update()
            return selection

        def select_bond(self, bond_index: int | None) -> PreviewSelection | None:
            selection = self.model.select_bond(bond_index)
            self._refresh_visuals()
            self._canvas.update()
            return selection

        def clear_selection(self) -> None:
            self.model.clear_selection()
            self._refresh_visuals()
            self._canvas.update()

        def delete_selected_objects(self) -> dict[str, int]:
            deleted = self.model.delete_selected_objects()
            self._refresh_visuals()
            self._canvas.update()
            atoms = int(deleted.get("atoms", 0))
            bonds = int(deleted.get("bonds", 0))
            if atoms or bonds:
                self.interaction_message_changed.emit(f"Deleted {atoms} atom(s), {bonds} bond(s)")
            return deleted

        def select_next_atom(self) -> PreviewSelection | None:
            selection = self.model.select_next_atom()
            self._refresh_visuals()
            self._canvas.update()
            return selection

        def select_previous_atom(self) -> PreviewSelection | None:
            selection = self.model.select_previous_atom()
            self._refresh_visuals()
            self._canvas.update()
            return selection

        def select_next_bond(self) -> PreviewSelection | None:
            selection = self.model.select_next_bond()
            self._refresh_visuals()
            self._canvas.update()
            return selection

        def select_previous_bond(self) -> PreviewSelection | None:
            selection = self.model.select_previous_bond()
            self._refresh_visuals()
            self._canvas.update()
            return selection

        @property
        def selected_object(self) -> dict[str, Any] | None:
            return self.model.selected_object

        @property
        def selected_payload(self) -> dict[str, Any] | None:
            return self.model.selected_payload

        @property
        def render_mode(self) -> str:
            return self.renderer_mode

        def scene_report(self) -> dict[str, Any]:
            return self.model.scene_report()

        def graphics_info(self) -> dict[str, Any]:
            self._capture_graphics_info()
            return dict(self._graphics_info)

        def _capture_graphics_info_once(self, event: Any = None) -> None:
            if self._graphics_info_captured:
                return
            self._graphics_info_captured = True
            self._capture_graphics_info(event)
            draw_event = getattr(getattr(self._canvas, "events", None), "draw", None)
            disconnect = getattr(draw_event, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect(self._capture_graphics_info_once)
                except Exception:
                    pass

        def _capture_graphics_info(self, _event: Any = None) -> None:
            context = getattr(self._canvas, "context", None)
            config = getattr(context, "config", {}) if context is not None else {}
            depth = config.get("depth_size") if isinstance(config, dict) else None
            info = dict(self._graphics_info)
            info["depth_bits"] = depth
            info["display"] = os.environ.get("DISPLAY", "")
            info["wayland_display"] = os.environ.get("WAYLAND_DISPLAY", "")
            info["qt_platform"] = "unavailable"
            if QtGui is not None and hasattr(QtGui, "QGuiApplication"):
                try:
                    info["qt_platform"] = str(QtGui.QGuiApplication.platformName())
                except Exception:
                    pass
            try:
                from vispy import app as vispy_app  # type: ignore

                backend = getattr(vispy_app, "backend_name", None)
                info["vispy_backend"] = str(backend() if callable(backend) else backend or "unavailable")
            except Exception:
                pass
            info["preview_renderer"] = str(self.renderer_mode)
            info["gl_backend"] = self._instancing_status["backend"]
            info["instancing_supported"] = self._instancing_status["supported"]
            info["instancing_reason"] = self._instancing_status["reason"]
            info["preview_instances"] = self._preview_instance_report()
            if QtGui is not None and hasattr(QtGui, "QOpenGLContext"):
                qt_context = QtGui.QOpenGLContext.currentContext()
                if qt_context is not None:
                    try:
                        functions = qt_context.extraFunctions()
                        functions.initializeOpenGLFunctions()
                        info["opengl_version"] = self._decode_gl_string(functions.glGetString(0x1F02))
                        renderer = self._decode_gl_string(functions.glGetString(0x1F01))
                        vendor = self._decode_gl_string(functions.glGetString(0x1F00))
                        info["renderer"] = " ".join(part for part in (vendor, renderer) if part and part != "unavailable") or info["renderer"]
                    except Exception:
                        pass
            self._graphics_info = info

        def _preview_instance_report(self) -> dict[str, int | str]:
            scene = self.model.scene
            atom_count = len(scene.atoms) if scene is not None else 0
            bond_segments = sum(len(bond.segments) for bond in scene.bonds) if scene is not None else 0
            return {
                "mode": str(self.renderer_mode),
                "atoms": int(atom_count),
                "bond_segments": int(bond_segments),
            }

        @staticmethod
        def _decode_gl_string(value: Any) -> str:
            if value is None:
                return "unavailable"
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            data = getattr(value, "data", None)
            if callable(data):
                try:
                    raw = data()
                    if isinstance(raw, bytes):
                        return raw.decode("utf-8", errors="replace")
                except Exception:
                    pass
            return str(value)

        def _refresh_visuals(self) -> None:
            self._update_instanced_visuals()
            self.renderer_mode = "instanced"
            self.model.renderer_mode = self.renderer_mode
            if self.model.scene is not None and self.model.scene.report is not None:
                self.model.scene.report["preview_renderer"] = self.renderer_mode
            if self.model.shared_scene is not None and getattr(self.model.shared_scene, "report", None) is not None:
                self.model.shared_scene.report["preview_renderer"] = self.renderer_mode
            self._atom_instance_visual.visible = True
            self._bond_instance_visual.visible = True
            self._update_selection_visuals()
            self._cell_visual.set_data(**build_cell_visual_payload(self.model.scene))
            poly_payload = build_poly_visual_payload(self.model.scene)
            poly_mesh = poly_payload["mesh"]
            poly_visible = bool(poly_payload["visible"]) and bool(len(poly_mesh.get("vertices", ())))
            self._poly_visual.visible = poly_visible
            if poly_visible:
                self._poly_visual.set_data(**poly_mesh)
            self._poly_edge_visual.set_data(**poly_payload["edges"])

        def _update_instanced_visuals(self) -> bool:
            atom_payload = build_atom_instance_payload(self.model.atom_draw_data())
            bond_payload = build_bond_instance_payload(
                self.model.bond_draw_data(),
                bond_scale=self.settings.bond_scale,
            )
            self._set_instance_payload(self._atom_instance_visual, atom_payload)
            self._set_instance_payload(self._bond_instance_visual, bond_payload)
            self._apply_preview_lighting()
            return True

        @staticmethod
        def _set_instance_payload(visual: Any, payload: dict[str, np.ndarray]) -> None:
            positions = payload["instance_positions"]
            transforms = payload["instance_transforms"]
            colors = payload["instance_colors"]
            if len(positions) == 0:
                visual.visible = False
                positions = np.zeros((1, 3), dtype=np.float32)
                transforms = np.zeros((1, 3, 3), dtype=np.float32)
                colors = np.ones((1, 4), dtype=np.float32)
            visual.instance_positions = positions
            visual.instance_transforms = transforms
            visual.instance_colors = colors

        def _configure_selection_visuals(self) -> None:
            self._selection_shell_visual.visible = False
            try:
                self._selection_shell_visual.set_gl_state("translucent", depth_test=True, cull_face="back")
            except Exception:
                try:
                    self._selection_shell_visual.set_gl_state("translucent", depth_test=True)
                except Exception:
                    pass

        def _update_selection_visuals(self) -> None:
            payload = self._selection_shell_payload()
            visible = bool(len(payload["vertices"]))
            self._selection_shell_visual.visible = visible
            if not visible:
                return
            self._selection_shell_visual.set_data(**payload)
            configure_shading_filter(self._selection_shell_visual, self.model.camera, self._lighting)

        def _selection_shell_payload(self) -> dict[str, np.ndarray]:
            if not self._mesh_enabled or self._sphere_mesh is None:
                return empty_selection_shell_payload()
            selection = self.model.selection
            active_index = selection.index if selection is not None and selection.kind == "atom" else None
            return build_selection_shell_payload(
                self.model.scene,
                self.model.selected_atom_indices,
                self._sphere_mesh,
                active_atom_index=active_index,
            )


else:  # pragma: no cover - importable fallback for tests and documentation builds

    class AxisOverlayWidget:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required to instantiate AxisOverlayWidget")

    class PreviewCanvas:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required to instantiate PreviewCanvas")


__all__ = [
    "AxisOverlayWidget",
    "PreviewCanvas",
    "PreviewCanvasModel",
    "PreviewCameraState",
    "PreviewRenderScene",
    "PreviewSelection",
    "PreviewSettings",
    "build_preview_scene",
]
