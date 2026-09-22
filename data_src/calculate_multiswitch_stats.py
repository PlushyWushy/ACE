import pandas as pd

df = pd.read_csv('sb_multiswitch/total_rewards_comparison.csv')
summary = df.groupby('type')['total_reward'].agg(['mean', 'std', 'count']).reset_index()

print("Summary of Total Rewards (30,000 Episodes):")
print(summary.to_string(index=False))

df_late = pd.read_csv('sb_multiswitch/late_rewards_comparison.csv')
summary_late = df_late.groupby('type')['late_reward'].agg(['mean', 'std', 'count']).reset_index()

print("\nSummary of Late Rewards (Episodes 20,000 - 30,000):")
print(summary_late.to_string(index=False))
