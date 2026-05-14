"""Project paths for notebooks and scripts."""

from pathlib import Path


def project_root() -> Path:
    """Repository root (parent of `src/`)."""
    return Path(__file__).resolve().parent.parent


def figs_dir(create: bool = True) -> Path:
    """Directory for figures used in reports (`figs/` at repo root)."""
    d = project_root() / "figs"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
