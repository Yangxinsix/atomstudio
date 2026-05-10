from __future__ import annotations

import sys
import types

from atomstudio import cli


def test_cli_app_command_forwards_to_desktop_entry(monkeypatch):
    seen: dict[str, object] = {}
    module = types.ModuleType("atomstudio.app.main")

    def fake_run_app(*, input_path=None, config_path=None, frame="last", state=None):
        seen["input_path"] = input_path
        seen["config_path"] = config_path
        seen["frame"] = frame
        seen["state"] = state
        return 7

    module.run_app = fake_run_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atomstudio.app.main", module)

    exit_code = cli.main(
        [
            "app",
            "--input",
            "tests/data/water.xyz",
            "--config",
            "configs/examples/water_single.yaml",
            "--frame",
            "0",
        ]
    )

    assert exit_code == 7
    assert seen == {
        "input_path": "tests/data/water.xyz",
        "config_path": "configs/examples/water_single.yaml",
        "frame": "0",
        "state": None,
    }
