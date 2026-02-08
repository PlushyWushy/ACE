#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def total_reward(csv_path: Path) -> float:
    total = 0.0
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += float(row.get("reward", 0.0))
    return total


def main():
    parser = argparse.ArgumentParser(description="Sum the reward column from a CSV file.")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    args = parser.parse_args()

    total = total_reward(args.csv_path)
    print(f"total_reward={total:.2f}")


if __name__ == "__main__":
    main()
