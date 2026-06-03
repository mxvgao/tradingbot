"""Print the Python environment and package availability for this project."""

from __future__ import annotations

import importlib.util
import site
import sys


PACKAGES = ("pandas", "yfinance", "statsmodels", "matplotlib")


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("\nSite packages:")
    for path in site.getsitepackages():
        print(f"  {path}")

    user_site = site.getusersitepackages()
    print(f"\nUser site packages:\n  {user_site}")

    print("\nPackage availability:")
    for package in PACKAGES:
        spec = importlib.util.find_spec(package)
        status = "found" if spec else "missing"
        location = spec.origin if spec else ""
        print(f"  {package}: {status} {location}")


if __name__ == "__main__":
    main()
