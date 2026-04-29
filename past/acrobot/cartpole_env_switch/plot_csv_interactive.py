#!/usr/bin/env python3
"""
Interactive CSV plotter with toggleable series via matplotlib CheckButtons.
Usage:
  python cartpole/plot_csv_interactive.py cartpole/runs/123_critic_ach_decoupled_tdlp/123_critic_ach_decoupled_tdlp.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons


def read_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError("CSV file is empty.")

    cols: list[list[float]] = [[] for _ in headers]
    for row in rows:
        for i, val in enumerate(row):
            cols[i].append(float(val))
    return headers, cols


def main() -> int:
    if len(sys.argv) not in [2, 3]:
        print("Usage: python cartpole/plot_csv_interactive.py <path-to-csv> [output-png-path]")
        return 2

    csv_path = Path(sys.argv[1]).expanduser().resolve()
    output_path = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) == 3 else None

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 2

    if output_path:
        import matplotlib
        matplotlib.use("Agg") # Non-interactive backend

    headers, cols = read_csv(csv_path)
    x = cols[0]

    n_series = len(headers) - 1
    fig, axes = plt.subplots(n_series, 1, figsize=(12, max(6, 2 * n_series)), sharex=True)
    fig.suptitle(f"{csv_path.name} - {output_path.name if output_path else 'Interactive'}")
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, hspace=0.35)

    if n_series == 1:
        axes = [axes]

    for ax, label, series in zip(axes, headers[1:], cols[1:]):
        ax.plot(x, series, linewidth=1.2, alpha=0.85, color="tab:blue")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(headers[0])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
