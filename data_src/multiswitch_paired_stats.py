import pandas as pd
from scipy import stats

df = pd.read_csv('sb_multiswitch/total_rewards_comparison.csv')

# Pivot to get ACE and classic rewards for each seed
pivoted = df.pivot(index='seed', columns='type', values='total_reward').dropna()

if 'ACE' in pivoted.columns and 'classic' in pivoted.columns:
    ace = pivoted['ACE']
    classic = pivoted['classic']
    
    # Paired T-test
    t_stat, p_t = stats.ttest_rel(ace, classic)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, p_w = stats.wilcoxon(ace, classic)
    
    print("Paired Statistical Test Results (Total Reward):")
    print(f"Number of seeds: {len(pivoted)}")
    print(f"ACE Mean:     {ace.mean():.2f}")
    print(f"Classic Mean: {classic.mean():.2f}")
    print(f"Mean Difference: {ace.mean() - classic.mean():.2f}")
    print("-" * 40)
    print(f"Paired T-test: p-value = {p_t:.6f}")
    print(f"Wilcoxon test: p-value = {p_w:.6f}")
    
    if p_t < 0.05:
        print("\nConclusion: The difference is statistically significant (p < 0.05).")
    else:
        print("\nConclusion: The difference is NOT statistically significant (p >= 0.05).")
else:
    print("Error: Could not find both 'ACE' and 'classic' types in the data.")
