from __future__ import annotations


def normalize_host_path(path: str | None) -> str:
    """Normalize host-specific paths into paths usable by the current WSL process."""

    if path is None:
        return ""
    text = str(path).strip()
    if not text:
        return text

    slash_path = text.replace("\\", "/")
    lower = slash_path.lower()
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if not lower.startswith(prefix):
            continue
        rest = slash_path[len(prefix) :]
        parts = rest.split("/")
        if len(parts) < 2:
            return "/"
        return "/" + "/".join(part for part in parts[1:] if part)
    return text


__all__ = ["normalize_host_path"]
