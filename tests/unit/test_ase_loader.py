from pathlib import Path

from atomstudio.io.ase_loader import load_structure, load_trajectory


DATA = Path(__file__).resolve().parents[1] / "data"


def test_load_structure_water():
    s = load_structure(str(DATA / "water.xyz"), frame="last")
    assert len(s.symbols) == 3
    assert s.frame_index == 0
    assert s.pbc == (False, False, False)


def test_load_trajectory_selector():
    frames = load_trajectory(str(DATA / "md.traj"), frame_selector="0:10:2")
    assert [f.frame_index for f in frames] == [0, 2, 4, 6, 8]


def test_load_structure_silicon_has_cell():
    s = load_structure(str(DATA / "silicon.cif"), frame="last")
    assert any(any(abs(v) > 1e-8 for v in row) for row in s.cell_vectors)
    assert any(s.pbc)
