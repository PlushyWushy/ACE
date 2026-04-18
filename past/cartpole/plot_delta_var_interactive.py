#!/usr/bin/env python3
"""
Interactive plot for delta-var only with hardcoded CSV path.
Scroll to zoom; click+drag to pan using matplotlib defaults.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = Path("cartpole/runs/123_critic_ach_decoupled_tdlp/123_critic_ach_decoupled_tdlp.csv")
DELTA_VAR_COLUMN = "mean_delta_var"


def read_series(path: Path, column: str) -> tuple[list[float], list[float]]:
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        try:
            idx = headers.index(column)
        except ValueError as exc:
            raise ValueError(f"Column '{column}' not found in {path}") from exc

        x_idx = 0
        xs: list[float] = []
        ys: list[float] = []
        for row in reader:
            if not row:
                continue
            xs.append(float(row[x_idx]))
            ys.append(float(row[idx]))
    return xs, ys


def main() -> int:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    xs, ys = read_series(CSV_PATH, DELTA_VAR_COLUMN)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, linewidth=1.4, color="tab:brown")
    ax.set_title(f"{CSV_PATH.name} - {DELTA_VAR_COLUMN}")
    ax.set_xlabel("episode")
    ax.set_ylabel(DELTA_VAR_COLUMN)
    ax.grid(True, alpha=0.3)

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
