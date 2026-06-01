from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "outputs/dive_trader_v2/icdm_extra_defense"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ANN = 252 / 5

BASE_LEDGER = ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv"
RIEM_DAILY = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_riemannian_ppo.csv"
PPO_DAILY = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_ppo.csv"
PG_DAILY = ROOT / "outputs/dive_trader_v2/v2x_rl_exposure_formal/daily_test_five_020_040_060_080_100_seed2022.csv"

def max_drawdown(ret):
    r = np.asarray(ret, dtype=float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())

def sortino_like(ret):
    r = np.asarray(ret, dtype=float)
    neg = r[r < 0]
    ds = float(np.std(neg) + 1e-12) if len(neg) else 1e-12
    return float(np.mean(r) / ds * np.sqrt(ANN))

def calmar_like(ret):
    r = np.asarray(ret, dtype=float)
    return float((np.mean(r) * ANN) / max(abs(max_drawdown(r)), 1e-12))

def worst_tail_mean(ret, q=0.05):
    r = np.asarray(ret, dtype=float)
    k = max(1, int(np.ceil(len(r) * q)))
    return float(np.sort(r)[:k].mean())

def summarize(ret, name, gate_mean, protocol):
    r = np.asarray(ret, dtype=float)
    mean = float(r.mean())
    std = float(r.std() + 1e-12)
    return {
        "model": name,
        "n_days": int(len(r)),
        "mean_ret": mean,
        "std_ret": std,
        "ann_ret_like": mean * ANN,
        "ann_sharpe_like": mean / std * np.sqrt(ANN),
        "sortino_like": sortino_like(r),
        "calmar_like": calmar_like(r),
        "max_drawdown_like": max_drawdown(r),
        "worst_5pct_day_mean": worst_tail_mean(r),
        "positive_day_ratio": float((r > 0).mean()),
        "gate_mean": gate_mean,
        "protocol": protocol,
    }

def read_base_from_ledger():
    led = pd.read_csv(BASE_LEDGER)
    led["Date"] = pd.to_datetime(led["Date"])
    if "day_net_ret" in led.columns:
        base = (
            led.groupby("Date", as_index=False)
            .agg(base_ret=("day_net_ret", "first"))
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        if "future_return_5d" not in led.columns:
            raise RuntimeError("ledger has no day_net_ret or future_return_5d")
        x = led["future_return_5d"].astype(float)
        if x.abs().quantile(0.999) > 1.0:
            led["future_return_5d"] = x / 100.0
        base = (
            led.groupby("Date", as_index=False)
            .apply(lambda g: pd.Series({
                "base_ret": float((g["weight"].astype(float) * g["future_return_5d"].astype(float)).sum())
            }))
            .reset_index(drop=True)
            .sort_values("Date")
        )
    return base

def read_daily(path, name):
    d = pd.read_csv(path)
    d["Date"] = pd.to_datetime(d["Date"])
    ret_col = None
    for c in ["ret", "day_net_ret", "base_ret"]:
        if c in d.columns:
            ret_col = c
            break
    if ret_col is None:
        raise RuntimeError(f"No return column in {path}: {d.columns.tolist()}")
    cols = ["Date", ret_col]
    if "gate" in d.columns:
        cols.append("gate")
    out = d[cols].drop_duplicates("Date").sort_values("Date").rename(columns={ret_col: "ret"})
    out["model"] = name
    return out

def main():
    riem = read_daily(RIEM_DAILY, "SPD-Riemannian PPO Gate")
    test_dates = riem[["Date"]].drop_duplicates()

    base = read_base_from_ledger()
    base = test_dates.merge(base, on="Date", how="inner").sort_values("Date")
    if len(base) != len(riem):
        print("[WARN] base/riem length mismatch", len(base), len(riem))

    rows = []
    for g in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rows.append(summarize(base["base_ret"].to_numpy(float) * g, f"Fixed gate {g:.1f}", g, "fixed_exposure_aligned_test"))

    rows.append(summarize(riem["ret"], "SPD-Riemannian PPO Gate", float(riem["gate"].mean()), "dynamic_exposure_aligned_test"))

    if PPO_DAILY.exists():
        ppo = read_daily(PPO_DAILY, "Vanilla PPO Gate")
        ppo = test_dates.merge(ppo, on="Date", how="inner")
        rows.append(summarize(ppo["ret"], "Vanilla PPO Gate", float(ppo["gate"].mean()), "dynamic_exposure_aligned_test"))

    if PG_DAILY.exists():
        pg = read_daily(PG_DAILY, "Current PG Gate")
        pg = test_dates.merge(pg, on="Date", how="inner")
        rows.append(summarize(pg["ret"], "Current PG Gate", float(pg["gate"].mean()), "dynamic_exposure_aligned_test"))

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_15_fixed_exposure_vs_dynamic_aligned_test.csv", index=False)

    rout = out.copy()
    for c in rout.columns:
        if pd.api.types.is_numeric_dtype(rout[c]):
            rout[c] = rout[c].round(6)
    rout.to_csv(TAB / "table_15_fixed_exposure_vs_dynamic_aligned_test_rounded.csv", index=False)

    plt.figure(figsize=(7.2, 4.2))
    plt.scatter(out["max_drawdown_like"], out["ann_sharpe_like"])
    for _, r in out.iterrows():
        plt.annotate(r["model"], (r["max_drawdown_like"], r["ann_sharpe_like"]), fontsize=7)
    plt.xlabel("Max drawdown-like")
    plt.ylabel("Annualized Sharpe-like")
    plt.title("Fixed exposure vs dynamic exposure, aligned test dates")
    plt.tight_layout()
    plt.savefig(FIG / "fig_15_fixed_exposure_vs_dynamic_aligned_test.pdf")
    plt.savefig(FIG / "fig_15_fixed_exposure_vs_dynamic_aligned_test.png", dpi=220)
    plt.close()

    print("===== ALIGNED FIXED EXPOSURE =====")
    print(out.to_string(index=False))
    print("[OK] saved aligned fixed exposure table and figure")

if __name__ == "__main__":
    main()
