from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
TAB = ROOT / "outputs/dive_trader_v2/icdm_extra_defense/tables"
TAB.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "kind": "random_top30",
        "model": "Random Top30",
        "n_runs": 50,
        "mean_ret_mean": 0.002638,
        "ann_sharpe_mean": 0.823994,
        "maxdd_mean": -0.504051,
        "worst5_mean": -0.048213,
    },
    {
        "kind": "model",
        "model": "Action Admission Base",
        "n_runs": 1,
        "mean_ret_mean": 0.041234,
        "ann_sharpe_mean": 3.108282,
        "maxdd_mean": -0.554269,
        "worst5_mean": -0.096148,
    },
    {
        "kind": "model",
        "model": "SPD-Riemannian PPO Gate",
        "n_runs": 1,
        "mean_ret_mean": 0.040063,
        "ann_sharpe_mean": 3.442246,
        "maxdd_mean": -0.302713,
        "worst5_mean": -0.075633,
    },
]
out = pd.DataFrame(rows)
out.to_csv(TAB / "table_16_random_top30_sanity_clean.csv", index=False)

r = out.copy()
for c in r.columns:
    if pd.api.types.is_numeric_dtype(r[c]):
        r[c] = r[c].round(6)
r.to_csv(TAB / "table_16_random_top30_sanity_clean_rounded.csv", index=False)

print(out.to_string(index=False))
print("[OK] saved clean random sanity table")
