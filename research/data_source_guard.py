"""Data source safety guard.

Ensures the project never loads data from external APIs or providers.
Import this module early (e.g. in main.py or conftest.py) to enforce
CSV-only data ingestion at runtime.
"""

import importlib
import sys

_BLOCKED_MODULES = (
    "databento",
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
)

_original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _guarded_import(name, *args, **kwargs):
    """Block imports of external data / HTTP modules."""
    top_level = name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise RuntimeError(
            f"Blocked import of '{name}'. "
            f"This project uses local CSV data only (data/nq_15m_data.csv). "
            f"External data providers and HTTP libraries are not permitted."
        )
    return _original_import(name, *args, **kwargs)


def activate():
    """Install the import guard. Call once at application startup."""
    import builtins
    builtins.__import__ = _guarded_import


def scan_codebase(root: str = ".") -> list[str]:
    """Scan all .py files under *root* for blocked references.

    Returns a list of violation strings (empty means clean).
    """
    from pathlib import Path

    violations = []
    blocked_patterns = _BLOCKED_MODULES + ("DBNStore", "databento.com")

    for py_file in Path(root).rglob("*.py"):
        # Skip guard files themselves
        if py_file.name in ("data_source_guard.py", "external_data_guard.py"):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in blocked_patterns:
            if pattern in source:
                violations.append(f"{py_file}: contains '{pattern}'")

    return violations


if __name__ == "__main__":
    violations = scan_codebase()
    if violations:
        print("VIOLATIONS FOUND:")
        for v in violations:
            print(f"  - {v}")
        raise SystemExit(1)
    else:
        print("CLEAN: No external data source references found.")
        print("Project uses CSV-only data loading (data/nq_15m_data.csv).")
