import math

def logistic_drive(max_val, k, center, signal, base):
    z = k * (signal - center)
    if z >= 700: return max_val + base
    if z <= -700: return base
    return max_val / (1.0 + math.exp(-z)) + base

# Only NE settings
NE_MAX = 2.0
NE_K = 1.0
NE_CENTER = 1.5
BASE_NOISE = 0.0

print(f"NE Only Modulation:")
for surprise in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0]:
    ne = logistic_drive(NE_MAX, NE_K, NE_CENTER, surprise, BASE_NOISE)
    print(f"Surprise: {surprise:.2f} -> NE: {ne:.4f}")

# Only ACh settings
ACH_MAX = 0.0
BASE_LR = 0.01
print(f"\nNE Only ACh/LR (should be constant):")
for surprise in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0]:
    ach = logistic_drive(ACH_MAX, 5, 1.5, surprise, BASE_LR)
    print(f"Surprise: {surprise:.2f} -> ACh: {ach:.4f}")
