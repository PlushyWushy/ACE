#!/usr/bin/env python3
import csv
import sys

def sum_post_switch(csv_path):
    total = 0.0
    with open(csv_path, 'r') as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            episode = int(row[0])
            reward = float(row[1])
            if episode >= 2001:
                total += reward
    return total

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sum_post_switch.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    total = sum_post_switch(csv_path)
    print(f"total_reward_post_switch={total:.0f}")