from __future__ import annotations

from atomstudio.app.state import AppState, LoadedFrameBundle
from atomstudio.config import RenderJobConfig
from atomstudio.preview.types import PreviewSelection
from atomstudio.structure.structure import Structure


def _render_cfg() -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "app",
            "input": {"path": "example.xyz", "frames": "last"},
            "output": {"path": "/tmp/app.png"},
            "style": {"scene_style": "default"},
        }
    )


def test_app_state_round_trip_and_frame_selection():
    bundle = LoadedFrameBundle(
        source_path="/tmp/example.xyz",
        frame_selector="all",
        frames=[
            Structure(frame_index=2, source_path="/tmp/example.xyz"),
            Structure(frame_index=7, source_path="/tmp/example.xyz"),
        ],
        selected_index=0,
    )
    state = AppState()

    state.set_loaded_frames(bundle, render_config=_render_cfg(), status="Loaded")
    assert state.current_frame_index() == 2
    assert state.dirty is False

    state.mark_dirty("Output path changed")
    assert state.dirty is True
    assert state.logs[-1].level == "warning"
    state.set_selection(
        PreviewSelection(kind="atom", index=7),
        payload={"object": {"index": 7}, "material": {"pipeline": "principled-preview"}, "metadata": {}},
    )
    state.set_dock_visibility(inspector_visible=False, axis_overlay_visible=False)

    snapshot = state.to_dict()
    restored = AppState.from_dict(snapshot)
    assert restored.bundle is not None
    assert restored.bundle.source_path == "/tmp/example.xyz"
    assert restored.bundle.selected_index == 0
    assert restored.render_config is not None
    assert restored.render_config.id == "app"
    assert restored.dirty is True
    assert restored.selected_object is not None
    assert restored.selected_object.kind == "atom"
    assert restored.selected_object.index == 7
    assert restored.selected_payload is not None
    assert restored.selected_payload["material"]["pipeline"] == "principled-preview"
    assert restored.dock_visibility.inspector_visible is False
    assert restored.dock_visibility.axis_overlay_visible is False

    selected = restored.select_frame(1)
    assert selected is not None
    assert restored.current_frame_index() == 7
    assert restored.current_structure() is selected
