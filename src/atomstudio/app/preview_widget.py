from __future__ import annotations

from dataclasses import replace
import os
from typing import Any

from atomstudio.config import RenderJobConfig
from atomstudio.app.axis_overlay import AxisOverlayWidget
from atomstudio.app.mouse_tools import wheel_steps_from_event
from atomstudio.app.preview_input import PreviewInputController
from atomstudio.preview.camera import (
    apply_view_preset,
    fit_bounds,
    model_rotation_euler_degrees,
    reset_model_rotation,
    rotate_model_about_axis,
    rotate_model_trackball,
    pan_camera,
    set_model_rotation_euler_degrees,
    zoom_camera,
)
from atomstudio.preview.gl.renderer import OpenGLRenderer
from atomstudio.preview.gl.state import graphics_environment
from atomstudio.preview.interaction import MeasurementController, normalize_mouse_mode
from atomstudio.preview.renderer import (
    PreviewCameraState,
    PreviewCanvasModel,
    PreviewRenderScene,
    PreviewSettings,
)
from atomstudio.preview.types import PreviewScene as BufferPreviewScene
from atomstudio.preview.types import PreviewSelection
from atomstudio.structure.structure import Structure

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = QtGui = QtOpenGLWidgets = QtWidgets = None


if QtOpenGLWidgets is not None:  # pragma: no cover - exercised only when GUI deps are installed

    class PreviewWidget(QtOpenGLWidgets.QOpenGLWidget):
        selection_changed = QtCore.Signal(object)
        interaction_message_changed = QtCore.Signal(str)
        camera_changed = QtCore.Signal(object)
        delete_requested = QtCore.Signal()
        undo_requested = QtCore.Signal()
        redo_requested = QtCore.Signal()

        def __init__(self, parent: Any | None = None, *, settings: PreviewSettings | None = None) -> None:
            super().__init__(parent)
            self.settings = settings or PreviewSettings()
            self.model = PreviewCanvasModel(self.settings)
            self.model.selection_changed.connect(self.selection_changed.emit)
            self.renderer = OpenGLRenderer(background=self.settings.background)
            self.renderer_mode = "opengl-atoms"
            self._measurement = MeasurementController()
            self._mouse_mode = "rotate"
            self._drag_start: tuple[float, float] | None = None
            self._drag_last: tuple[float, float] | None = None
            self._latest_mouse_point: tuple[float, float] | None = None
            self._drag_button: Any | None = None
            self._drag_moved = False
            self._selection_rubber_band: Any | None = None
            self._selection_rubber_origin: tuple[float, float] | None = None
            self._repaint_throttle_hz = 0.0
            self._interaction_driver_policy = self._resolve_interaction_driver_policy()
            self._interaction_repaint_policy = self._resolve_interaction_repaint_policy()
            self._interaction_timer_ms = self._int_env("ATOMSTUDIO_INTERACTION_TIMER_MS", 8, minimum=1)
            self._interaction_timer = self._build_interaction_timer()
            self._surface_depth_bits = self._int_env("ATOMSTUDIO_GL_DEPTH_BITS", 24, minimum=0)
            self._surface_samples = self._int_env("ATOMSTUDIO_GL_SAMPLES", 0, minimum=0)
            self._surface_swap_interval = self._int_env("ATOMSTUDIO_GL_SWAP_INTERVAL", 0, minimum=0)
            self._configure_driver_latency_hints()
            self._paint_context_info_refreshed = False
            self._axis_overlay_visible = True
            self._initialize_error: str | None = None
            self._graphics_info: dict[str, Any] = {
                "preview_renderer": self.renderer_mode,
                "qt_platform": "unavailable",
                "opengl_version": "unavailable",
                "renderer": "OpenGL preview",
                "depth_bits": None,
                "samples": None,
            }
            self._configure_surface()
            if QtCore is not None:
                self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.setMinimumSize(360, 360)
            self._axis_overlay = AxisOverlayWidget(self, background=self.settings.background)
            self._position_axis_overlay()
            self._sync_axis_overlay()
            self._input_controller = PreviewInputController(self, parent=self)
            self._input_controller.install_on(self)

        def _configure_surface(self) -> None:
            if QtGui is None:
                return
            surface_format = self.format()
            surface_format.setDepthBufferSize(int(self._surface_depth_bits))
            surface_format.setSamples(int(self._surface_samples))
            surface_format.setSwapInterval(int(self._surface_swap_interval))
            self.setFormat(surface_format)

        def initializeGL(self) -> None:
            try:
                self.renderer.initialize(enable_msaa=self._surface_samples > 0)
                self._initialize_error = None
            except Exception as exc:  # pragma: no cover - depends on host OpenGL
                self._initialize_error = str(exc)
            self._capture_graphics_info()

        def resizeGL(self, width: int, height: int) -> None:
            try:
                self.renderer.resize(width, height)
            except Exception as exc:  # pragma: no cover - depends on host OpenGL
                self._initialize_error = str(exc)

        def resizeEvent(self, event: Any) -> None:
            self._position_axis_overlay()
            super().resizeEvent(event)

        def paintGL(self) -> None:
            try:
                self.renderer.draw(self.model.camera, projection=self._projection_mode())
                if not self._paint_context_info_refreshed:
                    self.renderer.refresh_context_info()
                    self._paint_context_info_refreshed = True
            except Exception as exc:  # pragma: no cover - depends on host OpenGL
                self._initialize_error = str(exc)

        def closeEvent(self, event: Any) -> None:
            try:
                self.makeCurrent()
                self.renderer.release()
                self.doneCurrent()
            except Exception:
                pass
            super().closeEvent(event)

        def event(self, event: Any) -> bool:
            controller = getattr(self, "_input_controller", None)
            if controller is not None and controller.consume_event(event):
                return True
            return super().event(event)

        def set_scene(self, structure: Structure, cfg: RenderJobConfig) -> PreviewRenderScene:
            scene = self.model.set_scene(structure, cfg)
            self._publish_scene(scene)
            self._emit_camera_changed()
            return scene

        def set_preview_scene(
            self,
            preview_scene: PreviewRenderScene | BufferPreviewScene,
            *,
            frame_index: int | None = None,
            preserve_camera: bool | None = None,
        ) -> PreviewRenderScene:
            scene = self.model.set_preview_scene(preview_scene, preserve_camera=preserve_camera)
            self._publish_scene(scene)
            self._emit_camera_changed()
            return scene

        def update_preview_scene(
            self,
            preview_scene: PreviewRenderScene | BufferPreviewScene,
            *,
            frame_index: int | None = None,
            preserve_camera: bool | None = None,
        ) -> PreviewRenderScene:
            return self.set_preview_scene(preview_scene, frame_index=frame_index, preserve_camera=preserve_camera)

        def set_shader_style(self, shader_style: str | None) -> None:
            self.renderer.set_shader_style(shader_style)
            self.update()

        def set_preview_shader_style(self, shader_style: str | None) -> None:
            self.set_shader_style(shader_style)

        def _publish_scene(self, scene: PreviewRenderScene | None) -> None:
            self.renderer.update_scene(self.model.shared_scene or scene)
            self._publish_selection()
            for candidate in (scene, self.model.shared_scene):
                report = getattr(candidate, "report", None)
                if isinstance(report, dict):
                    report["preview_renderer"] = self.renderer_mode
            self.update()

        def _publish_selection(self) -> None:
            self.renderer.update_selection(
                self.model.shared_scene or self.model.scene,
                self.model.selection,
                selected_atom_indices=self.model.selected_atom_indices,
                selected_bond_indices=self.model.selected_bond_indices,
            )

        def fit_to_structure(self, padding: float | None = None) -> PreviewCameraState:
            camera = self._fit_camera_to_scene(padding=padding)
            self.update()
            self._emit_camera_changed()
            return camera

        def set_view_preset(self, view: str) -> PreviewCameraState:
            camera = apply_view_preset(self.model.camera, view)
            reset_model_rotation(camera)
            self.update()
            self._emit_camera_changed()
            return camera

        def set_axis_view(self, axis: str) -> PreviewCameraState:
            axis_key = str(axis or "").strip().lower()
            camera = apply_view_preset(self.model.camera, "top")
            if axis_key in {"a", "x"}:
                set_model_rotation_euler_degrees(camera, (0.0, -90.0, -90.0))
            elif axis_key in {"b", "y"}:
                set_model_rotation_euler_degrees(camera, (90.0, 0.0, 90.0))
            else:
                set_model_rotation_euler_degrees(camera, (0.0, 0.0, 0.0))
            self.update()
            self._emit_camera_changed()
            return camera

        def rotate_view(self, axis: str, direction: int, degrees: float = 15.0) -> PreviewCameraState:
            step = float(degrees) * (1.0 if int(direction) >= 0 else -1.0)
            axis_key = str(axis or "").strip().lower()
            camera = self.model.camera
            if axis_key in {"c", "z"}:
                rotate_model_about_axis(camera, (0.0, 0.0, 1.0), step)
            elif axis_key in {"a", "x"}:
                rotate_model_about_axis(camera, (1.0, 0.0, 0.0), step)
            elif axis_key in {"b", "y"}:
                rotate_model_about_axis(camera, (0.0, 1.0, 0.0), step)
            self.update()
            self._emit_camera_changed()
            return camera

        def pan_view(self, dx: float = 0.0, dy: float = 0.0) -> PreviewCameraState:
            camera = pan_camera(self.model.camera, dx, dy, self._viewport_size())
            self.update()
            self._emit_camera_changed()
            return camera

        def zoom_view(self, factor: float) -> PreviewCameraState:
            zoom_camera(self.model.camera, factor)
            self.update()
            self._emit_camera_changed()
            return self.model.camera

        def current_camera_state(self) -> PreviewCameraState:
            return replace(self.model.camera)

        def model_rotation_angles(self) -> tuple[float, float, float]:
            return model_rotation_euler_degrees(self.model.camera)

        def set_model_rotation_angles(self, x: float, y: float, z: float) -> PreviewCameraState:
            set_model_rotation_euler_degrees(self.model.camera, (float(x), float(y), float(z)))
            self.update()
            self._emit_camera_changed()
            return self.model.camera

        def _emit_camera_changed(self) -> None:
            self._sync_axis_overlay()
            self.camera_changed.emit(self.current_camera_state())

        def set_mouse_mode(self, mode: str) -> str:
            self._mouse_mode = normalize_mouse_mode(mode)
            self.emit_interaction_message(f"Mouse mode: {self._mouse_mode.replace('_', ' ')}")
            return self._mouse_mode

        @property
        def mouse_mode(self) -> str:
            return self._mouse_mode

        def set_camera_mouse_enabled(self, enabled: bool) -> None:
            return None

        def emit_interaction_message(self, message: str) -> None:
            self.interaction_message_changed.emit(str(message))

        def set_axis_overlay_visible(self, visible: bool) -> None:
            self._axis_overlay_visible = bool(visible)
            self._axis_overlay.setVisible(bool(visible))
            if visible:
                self._sync_axis_overlay()

        def atom_selection_at_screen(self, point: tuple[float, float]) -> PreviewSelection | None:
            return self.model.hit_test_cache(self._viewport_size()).pick_atom(point)

        def object_selection_at_screen(self, point: tuple[float, float]) -> PreviewSelection | None:
            return self.model.peek_selection_at(point, self._viewport_size())

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
                    self.clear_selection()
                return None
            if selection.kind == "atom":
                selection = self.model.toggle_atom_selection(selection.index) if toggle else self.model.select_atom(selection.index)
            else:
                selection = self.model.toggle_bond_selection(selection.index) if toggle else self.model.select_bond(selection.index)
            self._publish_selection()
            self.update()
            return selection

        def select_atom_indices(self, atom_indices: set[int]) -> PreviewSelection | None:
            selection = self.model.select_atoms(set(atom_indices))
            self._publish_selection()
            self.update()
            return selection

        def select_objects_in_screen_rect(
            self,
            start: tuple[float, float],
            end: tuple[float, float],
            *,
            append: bool = False,
        ) -> dict[str, tuple[int, ...]]:
            atom_indices = self.model.atom_indices_in_rect(start, end, self._viewport_size())
            bond_indices = self.model.bond_indices_in_rect(start, end, self._viewport_size())
            self.model.select_objects(set(atom_indices), set(bond_indices), append=append)
            self._publish_selection()
            self.update()
            return {"atoms": tuple(atom_indices), "bonds": tuple(bond_indices)}

        def measurement_message(self, atom_indices: list[int]) -> str:
            return self._measurement.message(self.model.scene, atom_indices)

        def select_preview_object(self, selection: PreviewSelection | None) -> PreviewSelection | None:
            selection = self.model.select_preview_object(selection)
            self._publish_selection()
            self.update()
            return selection

        def select_selection(self, selection: PreviewSelection | None) -> PreviewSelection | None:
            return self.select_preview_object(selection)

        def select_atom(self, atom_index: int | None) -> PreviewSelection | None:
            selection = self.model.select_atom(atom_index)
            self._publish_selection()
            self.update()
            return selection

        def select_bond(self, bond_index: int | None) -> PreviewSelection | None:
            selection = self.model.select_bond(bond_index)
            self._publish_selection()
            self.update()
            return selection

        def clear_selection(self) -> None:
            self.model.clear_selection()
            self._publish_selection()
            self.update()

        def cancel_interaction(self) -> None:
            self._stop_interaction_driver()
            self._reset_drag_state()
            self.clear_selection()

        def delete_selected_objects(self) -> dict[str, int]:
            deleted = self.model.delete_selected_objects()
            self._publish_scene(self.model.scene)
            return deleted

        def select_next_atom(self) -> PreviewSelection | None:
            selection = self.model.select_next_atom()
            self._publish_selection()
            self.update()
            return selection

        def select_previous_atom(self) -> PreviewSelection | None:
            selection = self.model.select_previous_atom()
            self._publish_selection()
            self.update()
            return selection

        def select_next_bond(self) -> PreviewSelection | None:
            selection = self.model.select_next_bond()
            self._publish_selection()
            self.update()
            return selection

        def select_previous_bond(self) -> PreviewSelection | None:
            selection = self.model.select_previous_bond()
            self._publish_selection()
            self.update()
            return selection

        def update_selected_atom_properties(self, updates: dict[str, Any]) -> bool:
            selection = self.model.selection
            if selection is None or selection.kind != "atom" or selection.index is None:
                return False
            changed = self.model.update_atom_properties(int(selection.index), dict(updates))
            if changed is None:
                return False
            self._publish_scene(self.model.scene)
            self.emit_interaction_message(f"Updated atom {selection.index}")
            return True

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
            report = self.model.scene_report()
            report["preview_renderer"] = self.renderer_mode
            return report

        def graphics_info(self) -> dict[str, Any]:
            self._capture_graphics_info()
            return dict(self._graphics_info)

        def _capture_graphics_info(self) -> None:
            info = self.renderer.graphics_info()
            info["preview_renderer"] = self.renderer_mode
            info["initialize_error"] = self._initialize_error
            info.update(graphics_environment())
            info["repaint_throttle_hz"] = float(self._repaint_throttle_hz)
            info["interaction_driver_policy"] = self._interaction_driver_policy
            info["interaction_repaint_policy"] = self._interaction_repaint_policy
            info["interaction_timer_ms"] = int(self._interaction_timer_ms)
            info["requested_depth_bits"] = int(self._surface_depth_bits)
            info["requested_samples"] = int(self._surface_samples)
            info["requested_swap_interval"] = int(self._surface_swap_interval)
            if QtGui is not None and hasattr(QtGui, "QGuiApplication"):
                try:
                    info["qt_platform"] = str(QtGui.QGuiApplication.platformName())
                except Exception:
                    info.setdefault("qt_platform", "unavailable")
            context = self.context() if hasattr(self, "context") else None
            if context is not None:
                try:
                    info["qt_surface_format"] = self._surface_format_label(context.format())
                except Exception:
                    info.setdefault("qt_surface_format", "unavailable")
            self._graphics_info = info

        def _viewport_size(self) -> tuple[int, int]:
            size = self.size()
            width = int(size.width()) if hasattr(size, "width") else 800
            height = int(size.height()) if hasattr(size, "height") else 600
            return max(1, width), max(1, height)

        def _sync_axis_overlay(self) -> None:
            if self._axis_overlay_visible:
                self._axis_overlay.sync_camera(self.model.camera)
                self._position_axis_overlay()
                self._axis_overlay.raise_()

        def _position_axis_overlay(self) -> None:
            if not hasattr(self, "_axis_overlay"):
                return
            margin = 8
            x = margin
            y = max(margin, int(self.height()) - int(self._axis_overlay.height()) - margin)
            self._axis_overlay.move(x, y)

        def _fit_camera_to_scene(self, *, padding: float | None = None) -> PreviewCameraState:
            scene = self.model.shared_scene or self.model.scene
            if scene is None:
                self.model.camera.center = (0.0, 0.0, 0.0)
                self.model.camera.scale_factor = 1.0
                return self.model.camera
            if getattr(scene, "bounds", None) is not None:
                minimum = scene.bounds.minimum
                maximum = scene.bounds.maximum
            else:
                minimum = getattr(scene, "bounds_min", (0.0, 0.0, 0.0))
                maximum = getattr(scene, "bounds_max", (0.0, 0.0, 0.0))
            center, scale, _radius = fit_bounds(
                minimum,
                maximum,
                padding=self.settings.fit_padding if padding is None else padding,
            )
            self.model.camera.center = center
            self.model.camera.scale_factor = scale
            return self.model.camera

        def _projection_mode(self) -> str:
            cfg = getattr(self.model, "cfg", None) or getattr(self.model.shared_scene, "config", None)
            camera_cfg = getattr(cfg, "camera", None)
            projection = str(getattr(camera_cfg, "projection", "ORTHOGRAPHIC") or "ORTHOGRAPHIC").lower()
            return "perspective" if projection.startswith("persp") else "orthographic"

        def mousePressEvent(self, event: Any) -> None:
            point = self._event_point(event)
            if point is None:
                return
            button = self._event_button(event)
            if self._mouse_mode in {"rotate", "pan", "select"} or self._is_middle_button(button) or self._is_right_button(button):
                self._drag_start = point
                self._drag_last = point
                self._latest_mouse_point = point
                self._drag_button = button
                self._drag_moved = False
                if self._is_select_drag_button(button):
                    self._show_selection_rubber_band(point, point)
                if self._is_camera_drag():
                    self._start_interaction_driver()
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event: Any) -> None:
            point = self._event_point(event)
            if point is None or self._drag_last is None:
                super().mouseMoveEvent(event)
                return
            self._latest_mouse_point = point
            if self._uses_timer_interaction() and self._is_camera_drag():
                start = self._drag_start or self._drag_last
                if abs(point[0] - start[0]) > 0.0 or abs(point[1] - start[1]) > 0.0:
                    self._drag_moved = True
                event.accept()
                return
            dx = float(point[0] - self._drag_last[0])
            dy = float(point[1] - self._drag_last[1])
            if abs(dx) > 0.0 or abs(dy) > 0.0:
                self._drag_moved = True
            if self._is_select_drag_button(self._drag_button):
                self._show_selection_rubber_band(self._drag_start or self._drag_last, point)
                self._drag_last = point
                event.accept()
                return
            self._apply_camera_drag_delta(self._drag_last, point)
            self._drag_last = point
            event.accept()

        def mouseReleaseEvent(self, event: Any) -> None:
            if self._uses_timer_interaction():
                self._drive_camera_interaction()
            self._stop_interaction_driver()
            point = self._event_point(event)
            if point is not None and self._mouse_mode == "select" and not self._drag_moved and self._is_left_button(self._drag_button):
                self.select_object_at_screen(point, toggle=self._event_has_ctrl(event), clear_on_miss=True)
            elif point is not None and self._mouse_mode == "select" and self._drag_moved and self._is_left_button(self._drag_button):
                self.select_objects_in_screen_rect(
                    self._drag_start or point,
                    point,
                    append=self._event_has_ctrl(event),
                )
            elif self._drag_moved:
                self.update()
            self._reset_drag_state()
            event.accept()

        def mouseDoubleClickEvent(self, event: Any) -> None:
            point = self._event_point(event)
            if point is not None and self._mouse_mode == "rotate" and self._is_left_button(self._event_button(event)):
                self.select_object_at_screen(point, toggle=self._event_has_ctrl(event), clear_on_miss=False)
                event.accept()
                return
            super().mouseDoubleClickEvent(event)

        def _reset_drag_state(self) -> None:
            self._hide_selection_rubber_band()
            self._drag_start = None
            self._drag_last = None
            self._latest_mouse_point = None
            self._drag_button = None
            self._drag_moved = False
            self._selection_rubber_origin = None

        def wheelEvent(self, event: Any) -> None:
            steps = wheel_steps_from_event(event)
            if steps is None or abs(steps) <= 0.0:
                super().wheelEvent(event)
                return
            zoom_camera(self.model.camera, 1.12**steps)
            self._request_interaction_update()
            self._emit_camera_changed()
            event.accept()

        def keyPressEvent(self, event: Any) -> None:
            if QtCore is None or not hasattr(event, "key"):
                super().keyPressEvent(event)
                return
            key = event.key()
            modifiers = event.modifiers() if hasattr(event, "modifiers") else QtCore.Qt.KeyboardModifier.NoModifier
            ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
            if key in {QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace}:
                self.delete_requested.emit()
                event.accept()
                return
            if ctrl and key == QtCore.Qt.Key.Key_Z and not shift:
                self.undo_requested.emit()
                event.accept()
                return
            if ctrl and (key == QtCore.Qt.Key.Key_Y or (key == QtCore.Qt.Key.Key_Z and shift)):
                self.redo_requested.emit()
                event.accept()
                return
            super().keyPressEvent(event)

        def _request_interaction_update(self) -> None:
            if self._interaction_repaint_policy == "immediate" and self.isVisible():
                self.repaint()
            else:
                self.update()

        def _apply_camera_drag_delta(self, start: Any, end: Any) -> None:
            start, end = self._coerce_drag_points(start, end)
            dx = float(end[0] - start[0])
            dy = float(end[1] - start[1])
            if abs(dx) <= 0.0 and abs(dy) <= 0.0:
                return
            self._drag_moved = True
            if self._is_right_button(self._drag_button):
                zoom_camera(self.model.camera, 1.01 ** (-dy))
            elif self._is_middle_button(self._drag_button) or self._mouse_mode == "pan":
                pan_camera(self.model.camera, dx, -dy, self._viewport_size())
            elif self._mouse_mode == "rotate":
                rotate_model_trackball(self.model.camera, start, end, self._viewport_size())
            else:
                return
            self._request_interaction_update()
            self._emit_camera_changed()

        def _coerce_drag_points(self, start: Any, end: Any) -> tuple[tuple[float, float], tuple[float, float]]:
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                width, height = self._viewport_size()
                origin = (float(width) * 0.5, float(height) * 0.5)
                return origin, (origin[0] + float(start), origin[1] + float(end))
            return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))

        def _build_interaction_timer(self) -> Any | None:
            if QtCore is None:
                return None
            timer = QtCore.QTimer(self)
            if hasattr(QtCore.Qt, "TimerType"):
                timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            timer.timeout.connect(self._drive_camera_interaction)
            return timer

        def _start_interaction_driver(self) -> None:
            if not self._uses_timer_interaction() or self._interaction_timer is None:
                return
            if not self._interaction_timer.isActive():
                self._interaction_timer.start(int(self._interaction_timer_ms))

        def _stop_interaction_driver(self) -> None:
            if self._interaction_timer is not None and self._interaction_timer.isActive():
                self._interaction_timer.stop()

        def _drive_camera_interaction(self) -> None:
            if not self._uses_timer_interaction() or self._drag_last is None or not self._is_camera_drag():
                return
            point = self._current_interaction_point()
            if point is None:
                return
            self._apply_camera_drag_delta(self._drag_last, point)
            self._drag_last = point

        def _current_interaction_point(self) -> tuple[float, float] | None:
            if self._latest_mouse_point is not None and self._drag_last is not None:
                if (
                    abs(float(self._latest_mouse_point[0] - self._drag_last[0])) > 0.0
                    or abs(float(self._latest_mouse_point[1] - self._drag_last[1])) > 0.0
                ):
                    return self._latest_mouse_point
            if QtGui is not None:
                try:
                    local = self.mapFromGlobal(QtGui.QCursor.pos())
                    if hasattr(local, "x") and hasattr(local, "y"):
                        x, y = float(local.x()), float(local.y())
                        if -64.0 <= x <= float(self.width() + 64) and -64.0 <= y <= float(self.height() + 64):
                            return x, y
                except Exception:
                    pass
            return self._latest_mouse_point

        def _is_camera_drag(self) -> bool:
            return bool(
                self._is_right_button(self._drag_button)
                or self._is_middle_button(self._drag_button)
                or self._mouse_mode in {"rotate", "pan"}
            )

        def _is_select_drag_button(self, button: Any) -> bool:
            return bool(self._mouse_mode == "select" and self._is_left_button(button))

        def _show_selection_rubber_band(self, start: tuple[float, float], end: tuple[float, float]) -> None:
            if QtWidgets is None or QtCore is None:
                return
            if self._selection_rubber_band is None:
                self._selection_rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self)
            self._selection_rubber_origin = (float(start[0]), float(start[1]))
            geometry = self._selection_rect(start, end)
            self._selection_rubber_band.setGeometry(geometry)
            if not self._selection_rubber_band.isVisible():
                self._selection_rubber_band.show()

        def _hide_selection_rubber_band(self) -> None:
            band = self._selection_rubber_band
            if band is not None:
                band.hide()

        @staticmethod
        def _selection_rect(start: tuple[float, float], end: tuple[float, float]) -> Any:
            if QtCore is None:
                return None
            x0, x1 = sorted((int(round(float(start[0]))), int(round(float(end[0])))))
            y0, y1 = sorted((int(round(float(start[1]))), int(round(float(end[1])))))
            return QtCore.QRect(QtCore.QPoint(x0, y0), QtCore.QPoint(x1, y1)).normalized()

        def _uses_timer_interaction(self) -> bool:
            return self._interaction_driver_policy == "timer"

        @staticmethod
        def _event_point(event: Any) -> tuple[float, float] | None:
            position = event.position() if hasattr(event, "position") else event.pos() if hasattr(event, "pos") else None
            if position is None:
                return None
            if hasattr(position, "x") and hasattr(position, "y"):
                return float(position.x()), float(position.y())
            return float(position[0]), float(position[1])

        @staticmethod
        def _event_button(event: Any) -> Any:
            return event.button() if hasattr(event, "button") else None

        @staticmethod
        def _is_middle_button(button: Any) -> bool:
            if QtCore is None:
                return False
            if button == QtCore.Qt.MouseButton.MiddleButton:
                return True
            value = getattr(button, "value", button)
            return bool(value in {3, 4, "middle", "MiddleButton"})

        @staticmethod
        def _is_left_button(button: Any) -> bool:
            if QtCore is not None and button == QtCore.Qt.MouseButton.LeftButton:
                return True
            value = getattr(button, "value", button)
            return bool(value in {1, "left", "LeftButton", None})

        @staticmethod
        def _is_right_button(button: Any) -> bool:
            if QtCore is not None and button == QtCore.Qt.MouseButton.RightButton:
                return True
            value = getattr(button, "value", button)
            return bool(value in {2, "right", "RightButton"})

        @staticmethod
        def _event_has_ctrl(event: Any) -> bool:
            if QtCore is None or not hasattr(event, "modifiers"):
                return False
            return bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)

        @staticmethod
        def _surface_format_label(surface_format: Any) -> str:
            profile = str(surface_format.profile()).rsplit(".", maxsplit=1)[-1]
            return (
                f"{surface_format.majorVersion()}.{surface_format.minorVersion()} "
                f"profile={profile} "
                f"depth={surface_format.depthBufferSize()} "
                f"samples={surface_format.samples()} "
                f"swap_interval={surface_format.swapInterval()}"
            )

        @staticmethod
        def _resolve_interaction_repaint_policy() -> str:
            value = str(os.environ.get("ATOMSTUDIO_INTERACTION_REPAINT", "update") or "update").strip().lower()
            return value if value in {"immediate", "update"} else "update"

        @staticmethod
        def _resolve_interaction_driver_policy() -> str:
            value = str(os.environ.get("ATOMSTUDIO_INTERACTION_DRIVER", "timer") or "timer").strip().lower()
            return value if value in {"timer", "events"} else "timer"

        @staticmethod
        def _int_env(name: str, default: int, *, minimum: int | None = None) -> int:
            raw = os.environ.get(name)
            try:
                value = int(str(raw).strip()) if raw is not None and str(raw).strip() else int(default)
            except ValueError:
                value = int(default)
            if minimum is not None:
                value = max(int(minimum), value)
            return value

        @staticmethod
        def _configure_driver_latency_hints() -> None:
            vsync = str(os.environ.get("ATOMSTUDIO_GL_VSYNC", "0") or "0").strip().lower()
            if vsync in {"1", "true", "yes", "on"}:
                return
            os.environ.setdefault("vblank_mode", "0")
            os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")

else:  # pragma: no cover - importable fallback for tests and documentation builds

    class PreviewWidget:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 QtOpenGLWidgets is required to instantiate PreviewWidget")


OpenGLPreviewWidget = PreviewWidget

__all__ = ["OpenGLPreviewWidget", "PreviewWidget"]
