"""Ultra-minimal smoke test for HedgeForge (uv-compatible, src layout)."""

import sys
from pathlib import Path

# Make sure src/ is importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_basic_import():
    """Confirm that the hedge_forge package is visible and importable."""
    import hedge_forge  # noqa: F401
