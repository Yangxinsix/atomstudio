from __future__ import annotations

from pathlib import Path

from ase.io import read

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
    frames = load_trajectory(str(file_path), frame_selector=_normalize_frame_arg(frame))
    if not frames:
        raise ValueError(f"No frame selected from {file_path}")
    return frames[-1]


def load_trajectory(path: str, frame_selector: str = "all") -> list[Structure]:
    file_path = Path(normalize_host_path(path)).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    all_atoms = read(str(file_path), index=":")
    if not isinstance(all_atoms, list):
        all_atoms = [all_atoms]

    indices = parse_frame_selector(frame_selector, len(all_atoms))
    return [_atoms_to_structure(all_atoms[i], str(file_path), i) for i in indices]
