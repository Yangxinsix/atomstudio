from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return subprocess.run(
        [sys.executable, "-m", "atomstudio.cli", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ATOMSTUDIO_RUN_INTEGRATION"), reason="set ATOMSTUDIO_RUN_INTEGRATION=1 to run")
def test_render_water(tmp_path: Path):
    out = tmp_path / "water.png"
    proc = _run_cli([
        "render",
        "--input",
        str(DATA / "water.xyz"),
        "--frame",
        "last",
        "--out",
        str(out),
        "--engine",
        "eevee",
        "--samples",
        "4",
        "--res-x",
        "320",
        "--res-y",
        "320",
    ])
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
    assert out.exists()


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ATOMSTUDIO_RUN_INTEGRATION"), reason="set ATOMSTUDIO_RUN_INTEGRATION=1 to run")
def test_render_silicon(tmp_path: Path):
    out = tmp_path / "silicon.png"
    proc = _run_cli([
        "render",
        "--input",
        str(DATA / "silicon.cif"),
        "--frame",
        "last",
        "--out",
        str(out),
        "--engine",
        "eevee",
        "--samples",
        "4",
        "--res-x",
        "320",
        "--res-y",
        "320",
    ])
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
    assert out.exists()


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ATOMSTUDIO_RUN_INTEGRATION"), reason="set ATOMSTUDIO_RUN_INTEGRATION=1 to run")
def test_render_md_sequence(tmp_path: Path):
    out_dir = tmp_path / "frames"
    proc = _run_cli([
        "render",
        "--input",
        str(DATA / "md.traj"),
        "--frames",
        "0:10:1",
        "--out-dir",
        str(out_dir),
        "--filename-template",
        "md_{frame:04d}.png",
        "--engine",
        "eevee",
        "--samples",
        "2",
        "--res-x",
        "256",
        "--res-y",
        "256",
    ])
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
    assert len(list(out_dir.glob("md_*.png"))) == 10


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ATOMSTUDIO_RUN_INTEGRATION"), reason="set ATOMSTUDIO_RUN_INTEGRATION=1 to run")
def test_batch_handdrawn_surface_v2():
    cfg = ROOT / "configs" / "examples" / "handdrawn_surface_v2.yaml"
    proc = _run_cli([
        "batch",
        "--config",
        str(cfg),
    ])
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ATOMSTUDIO_RUN_INTEGRATION"), reason="set ATOMSTUDIO_RUN_INTEGRATION=1 to run")
def test_render_handdrawn_default_space_filling(tmp_path: Path):
    out = tmp_path / "handdrawn.png"
    proc = _run_cli([
        "render",
        "--input",
        str(DATA / "water.xyz"),
        "--frame",
        "last",
        "--style",
        "handdrawn",
        "--engine",
        "eevee",
        "--samples",
        "4",
        "--res-x",
        "320",
        "--res-y",
        "320",
        "--out",
        str(out),
    ])
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
    assert out.exists()


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ATOMSTUDIO_RUN_INTEGRATION"), reason="set ATOMSTUDIO_RUN_INTEGRATION=1 to run")
def test_render_handdrawn_transparent_bg(tmp_path: Path):
    out = tmp_path / "handdrawn_transparent.png"
    proc = _run_cli([
        "render",
        "--input",
        str(DATA / "water.xyz"),
        "--frame",
        "last",
        "--style",
        "handdrawn",
        "--engine",
        "eevee",
        "--samples",
        "4",
        "--res-x",
        "320",
        "--res-y",
        "320",
        "--transparent-bg",
        "--out",
        str(out),
    ])
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
    assert out.exists()
