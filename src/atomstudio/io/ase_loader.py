from __future__ import annotations

from pathlib import Path

from ase.io import iread, read
from ase.io.trajectory import TrajectoryReader

from atomstudio.paths import normalize_host_path
from atomstudio.structure.structure import Structure


def parse_frame_selector(frame_selector: str, length: int) -> list[int]:
    if length <= 0:
        return []
    selector = str(frame_selector).strip().lower()
    if selector == "all":
        return list(range(length))
    if selector == "last":
        return [length - 1]
    if ":" in selector:
        parts = selector.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid frame selector: {frame_selector}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else length
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step == 0:
            raise ValueError("Frame selector step cannot be 0")
        return [i for i in range(start, min(end, length), step)]
    index = int(selector)
    if index < 0:
        index = length + index
    if index < 0 or index >= length:
        raise IndexError(f"Frame index out of range: {index}")
    return [index]


def _atoms_to_structure(atoms, path: str, frame_index: int) -> Structure:
    return Structure.from_ase(atoms, source_path=path, frame_index=frame_index)


def _normalize_frame_arg(frame: int | str) -> str:
    return str(frame)


def load_structure(path: str, frame: int | str = "last") -> Structure:
    file_path = Path(normalize_host_path(path)).expanduser().resolve()
    frame_count = count_trajectory_frames(str(file_path))
    indices = parse_frame_selector(_normalize_frame_arg(frame), frame_count)
    if not indices:
        raise ValueError(f"No frame selected from {file_path}")
    return load_frame(str(file_path), indices[-1])


def load_trajectory(path: str, frame_selector: str = "all") -> list[Structure]:
    file_path = Path(normalize_host_path(path)).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    frame_count = count_trajectory_frames(str(file_path))
    indices = parse_frame_selector(frame_selector, frame_count)
    if not indices:
        return []
    selector = str(frame_selector).strip().lower()
    ase_index = ":" if selector == "all" else selector
    try:
        atoms = read(str(file_path), index=ase_index)
    except Exception:
        atoms = [read(str(file_path), index=index) for index in indices]
    if not isinstance(atoms, list):
        atoms = [atoms]
    if len(atoms) != len(indices):
        atoms = [read(str(file_path), index=index) for index in indices]
    return [_atoms_to_structure(item, str(file_path), index) for item, index in zip(atoms, indices, strict=True)]


def load_frame(path: str, frame_index: int) -> Structure:
    file_path = Path(normalize_host_path(path)).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    index = int(frame_index)
    atoms = _read_atoms_frame(file_path, index)
    return _atoms_to_structure(atoms, str(file_path), index)


def count_trajectory_frames(path: str) -> int:
    file_path = Path(normalize_host_path(path)).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() in {".xyz", ".extxyz"}:
        count = _count_xyz_frames(file_path)
        if count > 0:
            return count
    if file_path.suffix.lower() == ".traj":
        reader = TrajectoryReader(str(file_path))
        try:
            return len(reader)
        finally:
            reader.close()
    atoms = read(str(file_path), index=":")
    return len(atoms) if isinstance(atoms, list) else 1


def _read_atoms_frame(path: Path, frame_index: int):
    suffix = path.suffix.lower()
    if suffix == ".traj":
        reader = TrajectoryReader(str(path))
        try:
            return reader[int(frame_index)]
        finally:
            reader.close()
    if suffix in {".xyz", ".extxyz"}:
        try:
            return next(iread(str(path), index=int(frame_index)))
        except StopIteration as exc:
            raise IndexError(f"Frame index out of range: {frame_index}") from exc
    return read(str(path), index=int(frame_index))


def _count_xyz_frames(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    atom_count = int(stripped)
                except ValueError:
                    return 0
                if handle.readline() == "":
                    return 0
                for _ in range(atom_count):
                    if handle.readline() == "":
                        return 0
                count += 1
    except OSError:
        return 0
    return count
