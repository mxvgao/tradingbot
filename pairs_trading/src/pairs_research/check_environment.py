"""Verify the active interpreter and actual imports, not just package discovery."""

from importlib import import_module
from importlib.metadata import version
import sys

PACKAGES = (
    "numpy",
    "pandas",
    "statsmodels",
    "yfinance",
    "hmmlearn",
    "matplotlib",
    "requests",
    "scipy",
)


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    failures = []
    if sys.version_info[:2] != (3, 12):
        failures.append("Python 3.12 is required; run uv sync --frozen")
    for package in PACKAGES:
        try:
            import_module(package)
            print(f"  {package}: {version(package)}")
        except Exception as exc:
            failures.append(f"{package}: {exc}")
    if failures:
        raise SystemExit("Environment check failed:\n" + "\n".join(failures))
    print("Environment check passed.")


if __name__ == "__main__":
    main()
