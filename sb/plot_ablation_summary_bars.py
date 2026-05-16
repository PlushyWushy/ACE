#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("/Users/chairstands/Desktop/Icarus/sb")
TOTAL_CSV = BASE_DIR / "sb_normal/total_rewards_comparison.csv"
POST_CSV = BASE_DIR / "sb_normal/post_switch_rewards_comparison.csv"
OUT_TOTAL_PNG = BASE_DIR / "sb_normal/total_rewards_bar.png"
OUT_POST_PNG = BASE_DIR / "sb_normal/post_switch_rewards_bar.png"

def main():
    if not TOTAL_CSV.exists() or not POST_CSV.exists():
        print("Aggregation CSVs not found. Run aggregate_ablations.py first.")
        return

    # Total Rewards Plot
    df_total = pd.read_csv(TOTAL_CSV)
    summary_total = df_total.groupby('type')['total_reward'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(summary_total['type'], summary_total['mean'], yerr=summary_total['std'], 
            capsize=10, color=['gray', 'black', 'tab:red', 'tab:blue'], alpha=0.8)
    plt.ylabel('Total Optimal Choices')
    plt.title('Total Performance Comparison (Mean ± STD)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUT_TOTAL_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved {OUT_TOTAL_PNG}")

    # Post-Switch Rewards Plot
    df_post = pd.read_csv(POST_CSV)
    summary_post = df_post.groupby('type')['post_switch_reward'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(summary_post['type'], summary_post['mean'], yerr=summary_post['std'], 
            capsize=10, color=['gray', 'black', 'tab:red', 'tab:blue'], alpha=0.8)
    plt.ylabel('Post-Switch Optimal Choices')
    plt.title('Recovery Performance Comparison (Mean ± STD)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUT_POST_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved {OUT_POST_PNG}")

if __name__ == "__main__":
    main()
