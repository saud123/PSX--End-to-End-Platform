#!/usr/bin/env python3
"""
list_packages.py — Print every installed Python package and its version.

Usage (in VS Code terminal):
    python list_packages.py                 # print to terminal
    python list_packages.py > packages.txt  # save to a file (avoids terminal truncation)
    python list_packages.py --filter num    # only show packages matching a keyword
"""

import sys
from importlib.metadata import distributions


def get_packages() -> list[tuple[str, str]]:
    """Return a sorted list of (name, version) for all installed distributions."""
    pkgs = []
    for dist in distributions():
        name = dist.metadata["Name"]
        ver = dist.version
        if name:  # skip malformed entries
            pkgs.append((name, ver))
    pkgs.sort(key=lambda x: x[0].lower())
    return pkgs


def main() -> None:
    keyword = None
    if "--filter" in sys.argv:
        idx = sys.argv.index("--filter")
        if idx + 1 < len(sys.argv):
            keyword = sys.argv[idx + 1].lower()

    pkgs = get_packages()
    if keyword:
        pkgs = [(n, v) for n, v in pkgs if keyword in n.lower()]

    width = max((len(n) for n, _ in pkgs), default=10) + 2

    print(f"Python {sys.version.split()[0]} — {len(pkgs)} package(s)\n")
    print(f"{'Package':<{width}}Version")
    print("-" * (width + 10))
    for name, ver in pkgs:
        print(f"{name:<{width}}{ver}")


if __name__ == "__main__":
    main()