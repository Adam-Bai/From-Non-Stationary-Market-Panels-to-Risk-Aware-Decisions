from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "outputs/dive_trader_v2/final_paper_riemannian_main"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

base_tplus_daily_path = ROOT / "outputs/dive_trader_v2/final_paper_tables_clean_v2/daily_delay1_Tplus1_V2X_B.csv"
riem_daily_path = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_riemannian_ppo.csv"

old_table_path = TAB / "table_2_china_constraints_riemannian_main.csv"
out_table_path = TAB / "table_2_china_constraints_riemannian_main.csv"
out_round_path = TAB / "table_2_china_constraints_riemannian_main_rounded.csv"
out_daily_path = TAB / "daily_delay1_Tplus1_Riemannian_PPO.csv"

ANN = 252 / 5

def max_drawdown(ret):
    r = np.asarray(ret, dtype=float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())

def metrics(ret):
    r = np.asarray(ret, dtype=float)
    mean = float(r.mean())
    std = float(r.std() + 1e-12)
    return {
        "n_days": int(len(r)),
        "mean_ret": mean,
        "std_ret": std,
        "ann_ret_like": mean * ANN,
        "ann_sharpe_like": mean / std * np.sqrt(ANN),
        "max_drawdown_like": max_drawdown(r),
        "positive_day_ratio": float((r > 0).mean()),
    }

def pick_ret_col(df):
    candidates = [
        "ret", "day_net_ret", "base_ret", "delay_ret", "tplus1_ret",
        "model_ret", "net_ret", "daily_ret", "day_ret"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = [c for c in df.columns if c != "Date" and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise RuntimeError(f"Cannot identify return column. Columns={df.columns.tolist()}")

def main():
    if not base_tplus_daily_path.exists():
        raise FileNotFoundError(base_tplus_daily_path)
    if not riem_daily_path.exists():
        raise FileNotFoundError(riem_daily_path)

    base = pd.read_csv(base_tplus_daily_path)
    base["Date"] = pd.to_datetime(base["Date"])
    ret_col = pick_ret_col(base)
    base = (
        base[["Date", ret_col]]
        .drop_duplicates("Date")
        .rename(columns={ret_col: "base_delay_ret"})
        .sort_values("Date")
        .reset_index(drop=True)
    )

    riem = pd.read_csv(riem_daily_path)
    riem["Date"] = pd.to_datetime(riem["Date"])
    if "gate" not in riem.columns:
        raise RuntimeError(f"No gate column in {riem_daily_path}. Columns={riem.columns.tolist()}")
    gate = riem[["Date", "gate"]].drop_duplicates("Date").sort_values("Date")

    m = base.merge(gate, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    if len(m) < 500:
        raise RuntimeError(f"Too few aligned days: {m.shape}")

    m["ret"] = m["base_delay_ret"].astype(float) * m["gate"].astype(float)
    m.to_csv(out_daily_path, index=False)

    met = metrics(m["ret"])
    new = {
        "model": "SPD-Riemannian PPO Gate",
        "constraint": "delay1_Tplus1",
        "renorm": False,
        "cost_bps": 5,
        **met,
        "turnover_mean": np.nan,
        "avg_n_holding": np.nan,
        "kept_ratio_rows": 1.0,
        "gate_mean": float(m["gate"].mean()),
        "paper_interpretation": "T+1 delayed base action-admission returns rescaled by the frozen SPD-Riemannian PPO exposure gate.",
    }

    if old_table_path.exists():
        tab = pd.read_csv(old_table_path)
    else:
        tab = pd.DataFrame()

    if len(tab):
        model_s = tab.get("model", pd.Series([""] * len(tab))).astype(str)
        constraint_s = tab.get("constraint", pd.Series([""] * len(tab))).astype(str)

        # Remove previous ambiguous/final exposure T+1 row if present.
        remove = (
            constraint_s.str.contains("delay1_Tplus1", regex=False)
            & model_s.str.contains("V2X-RL|five-level|Riemannian|SPD-Riemannian", regex=True)
        )
        tab = tab.loc[~remove].copy()

    new_row = pd.DataFrame([new])

    # Union columns while preserving old table order.
    for c in new_row.columns:
        if c not in tab.columns:
            tab[c] = np.nan
    for c in tab.columns:
        if c not in new_row.columns:
            new_row[c] = np.nan
    new_row = new_row[tab.columns]

    tab2 = pd.concat([tab, new_row], ignore_index=True)
    tab2.to_csv(out_table_path, index=False)

    rtab = tab2.copy()
    for c in rtab.columns:
        if pd.api.types.is_numeric_dtype(rtab[c]):
            rtab[c] = rtab[c].round(6)
    rtab.to_csv(out_round_path, index=False)

    print("===== SOURCE =====")
    print("base_tplus_daily:", base_tplus_daily_path)
    print("base return col:", ret_col)
    print("riemann_daily:", riem_daily_path)
    print("aligned days:", len(m), m["Date"].min().date(), m["Date"].max().date())
    print()
    print("===== Riemannian-PPO T+1 metrics =====")
    print(pd.DataFrame([new]).to_string(index=False))
    print()
    print("===== Updated table T+1 rows =====")
    show_cols = [c for c in ["model","constraint","n_days","mean_ret","ann_ret_like","ann_sharpe_like","max_drawdown_like","positive_day_ratio","gate_mean"] if c in tab2.columns]
    print(tab2[tab2["constraint"].astype(str).str.contains("delay1_Tplus1", regex=False)][show_cols].to_string(index=False))
    print()
    print("[OK] updated:", out_table_path)
    print("[OK] rounded:", out_round_path)
    print("[OK] daily:", out_daily_path)

if __name__ == "__main__":
    main()
