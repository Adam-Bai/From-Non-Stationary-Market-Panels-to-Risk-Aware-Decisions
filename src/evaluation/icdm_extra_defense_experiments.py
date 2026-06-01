from pathlib import Path
import json
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
N_BOOT_RANDOM = 50
RANDOM_SEED = 2026

# Core sources
BASE_LEDGER = ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv"
BASE_DAILY_CANDIDATES = [
    ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/daily_test_61b_cross_action_admission.csv",
    ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/daily_61b_cross_action_admission.csv",
]
RIEM_DAILY = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_riemannian_ppo.csv"
PPO_DAILY = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_ppo.csv"
PG_DAILY = ROOT / "outputs/dive_trader_v2/v2x_rl_exposure_formal/daily_test_five_020_040_060_080_100_seed2022.csv"
RIEM_STATE = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_spd_riemannian_market_state.csv"

DATA_DIR = ROOT / "data_external/msan_samples_full_qfq_norm"
META_PATHS = [
    DATA_DIR / "meta_L60_H5_full_with_period_label.parquet",
    DATA_DIR / "meta_L60_H5_full.parquet",
]


def max_drawdown(ret):
    r = np.asarray(ret, dtype=float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def sortino_like(ret):
    r = np.asarray(ret, dtype=float)
    downside = r[r < 0]
    ds = float(np.std(downside) + 1e-12) if len(downside) else 1e-12
    return float(np.mean(r) / ds * np.sqrt(ANN))


def calmar_like(ret):
    r = np.asarray(ret, dtype=float)
    ann = float(np.mean(r) * ANN)
    mdd = abs(max_drawdown(r))
    return float(ann / max(mdd, 1e-12))


def worst_tail_mean(ret, q=0.05):
    r = np.asarray(ret, dtype=float)
    k = max(1, int(np.ceil(len(r) * q)))
    return float(np.sort(r)[:k].mean())


def summarize(ret, name=None, extra=None):
    r = np.asarray(ret, dtype=float)
    mean = float(r.mean())
    std = float(r.std() + 1e-12)
    out = {
        "model": name if name else "",
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
    }
    if extra:
        out.update(extra)
    return out


def find_base_daily():
    for p in BASE_DAILY_CANDIDATES:
        if p.exists():
            d = pd.read_csv(p)
            d["Date"] = pd.to_datetime(d["Date"])
            ret_col = None
            for c in ["ret", "day_net_ret", "base_ret"]:
                if c in d.columns:
                    ret_col = c
                    break
            if ret_col:
                return d[["Date", ret_col]].drop_duplicates("Date").rename(columns={ret_col: "base_ret"}).sort_values("Date")
    # Fallback from ledger
    led = pd.read_csv(BASE_LEDGER)
    led["Date"] = pd.to_datetime(led["Date"])
    if "day_net_ret" in led.columns:
        return led.groupby("Date", as_index=False).agg(base_ret=("day_net_ret", "first")).sort_values("Date")
    return led.groupby("Date", as_index=False).apply(
        lambda g: pd.Series({"base_ret": float((g["weight"].astype(float) * g["future_return_5d"].astype(float)).sum())})
    ).reset_index(drop=True).sort_values("Date")


def load_daily_model(path, name):
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


def experiment_fixed_exposure(base_daily):
    rows = []
    d = base_daily.copy().sort_values("Date")
    for g in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ret = d["base_ret"].to_numpy(float) * g
        rows.append(summarize(ret, f"Fixed gate {g:.1f}", {"gate_mean": g, "protocol": "fixed_exposure"}))

    if RIEM_DAILY.exists():
        rd = load_daily_model(RIEM_DAILY, "SPD-Riemannian PPO Gate")
        rows.append(summarize(rd["ret"], "SPD-Riemannian PPO Gate", {
            "gate_mean": float(rd["gate"].mean()) if "gate" in rd else np.nan,
            "protocol": "dynamic_exposure",
        }))
    if PPO_DAILY.exists():
        pd_ = load_daily_model(PPO_DAILY, "Vanilla PPO Gate")
        rows.append(summarize(pd_["ret"], "Vanilla PPO Gate", {
            "gate_mean": float(pd_["gate"].mean()) if "gate" in pd_ else np.nan,
            "protocol": "dynamic_exposure",
        }))
    if PG_DAILY.exists():
        pg = load_daily_model(PG_DAILY, "Current PG Gate")
        rows.append(summarize(pg["ret"], "Current PG Gate", {
            "gate_mean": float(pg["gate"].mean()) if "gate" in pg else np.nan,
            "protocol": "dynamic_exposure",
        }))

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_15_fixed_exposure_vs_dynamic.csv", index=False)

    plt.figure(figsize=(7.2, 4.2))
    plot_df = out.copy()
    plt.scatter(plot_df["max_drawdown_like"], plot_df["ann_sharpe_like"])
    for _, r in plot_df.iterrows():
        plt.annotate(r["model"], (r["max_drawdown_like"], r["ann_sharpe_like"]), fontsize=7)
    plt.xlabel("Max drawdown-like")
    plt.ylabel("Annualized Sharpe-like")
    plt.title("Fixed exposure vs dynamic exposure")
    plt.tight_layout()
    plt.savefig(FIG / "fig_15_fixed_exposure_vs_dynamic.pdf")
    plt.savefig(FIG / "fig_15_fixed_exposure_vs_dynamic.png", dpi=220)
    plt.close()

    return out


def load_meta_test():
    mp = None
    for p in META_PATHS:
        if p.exists():
            mp = p
            break
    if mp is None:
        raise FileNotFoundError("meta parquet not found")

    cols = ["Date", "Ticker", "future_return_5d"]
    meta = pd.read_parquet(mp, columns=cols)
    meta["Date"] = pd.to_datetime(meta["Date"])
    meta["Ticker"] = meta["Ticker"].astype(str)
    meta = meta[(meta["Date"] >= "2024-01-02") & (meta["Date"] <= "2026-04-23")].copy()
    x = meta["future_return_5d"].astype(float)
    if x.abs().quantile(0.999) > 1.0:
        meta["future_return_5d"] = x / 100.0
    else:
        meta["future_return_5d"] = x
    return meta.drop_duplicates(["Date", "Ticker"]).sort_values(["Date", "Ticker"])


def random_portfolio_daily(meta, seed, topk=30, exposure=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for date, g in meta.groupby("Date", sort=True):
        if len(g) < topk:
            continue
        idx = rng.choice(len(g), size=topk, replace=False)
        sub = g.iloc[idx]
        w = np.ones(topk, dtype=float) / topk * exposure
        ret = float(np.sum(w * sub["future_return_5d"].to_numpy(float)))
        rows.append({"Date": date, "ret": ret})
    return pd.DataFrame(rows)


def shuffled_score_daily(base_ledger, seed, topk=30, exposure=0.5):
    # Uses same available universe as base ledger if full score panel is unavailable.
    # Shuffles selected ledger candidates by date as a conservative sanity baseline.
    rng = np.random.default_rng(seed)
    d = base_ledger.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    rows = []
    for date, g in d.groupby("Date", sort=True):
        if "future_return_5d" not in g.columns:
            continue
        n = min(topk, len(g))
        idx = rng.choice(len(g), size=n, replace=False)
        sub = g.iloc[idx]
        w = np.ones(n, dtype=float) / n * exposure
        ret = float(np.sum(w * sub["future_return_5d"].astype(float).to_numpy() ))
        rows.append({"Date": date, "ret": ret})
    return pd.DataFrame(rows)


def experiment_random_sanity(base_daily):
    meta = load_meta_test()
    led = pd.read_csv(BASE_LEDGER)
    led["Date"] = pd.to_datetime(led["Date"])
    if "future_return_5d" in led.columns:
        x = led["future_return_5d"].astype(float)
        if x.abs().quantile(0.999) > 1.0:
            led["future_return_5d"] = x / 100.0

    rows = []

    # Base and final rows
    rows.append(summarize(base_daily["base_ret"], "Action Admission Base", {"kind": "model"}))
    if RIEM_DAILY.exists():
        rd = load_daily_model(RIEM_DAILY, "SPD-Riemannian PPO Gate")
        rows.append(summarize(rd["ret"], "SPD-Riemannian PPO Gate", {"kind": "model"}))

    rand_metrics = []
    shuffle_metrics = []
    for k in range(N_BOOT_RANDOM):
        seed = RANDOM_SEED + k
        rp = random_portfolio_daily(meta, seed=seed)
        rand_metrics.append(summarize(rp["ret"], f"Random Top30 seed{seed}", {"kind": "random_top30", "seed": seed}))

        sp = shuffled_score_daily(led, seed=seed)
        if len(sp):
            shuffle_metrics.append(summarize(sp["ret"], f"Shuffled selected-ledger seed{seed}", {"kind": "shuffled_selected_ledger", "seed": seed}))

    all_rows = rows + rand_metrics + shuffle_metrics
    full = pd.DataFrame(all_rows)
    full.to_csv(TAB / "table_16_random_shuffled_sanity_full.csv", index=False)

    summary_rows = []
    for kind, g in full.groupby("kind"):
        if kind == "model":
            for _, r in g.iterrows():
                summary_rows.append({
                    "kind": kind,
                    "model": r["model"],
                    "n_runs": 1,
                    "mean_ret_mean": r["mean_ret"],
                    "ann_sharpe_mean": r["ann_sharpe_like"],
                    "maxdd_mean": r["max_drawdown_like"],
                    "worst5_mean": r["worst_5pct_day_mean"],
                    "mean_ret_std": 0.0,
                    "ann_sharpe_std": 0.0,
                })
        else:
            summary_rows.append({
                "kind": kind,
                "model": kind,
                "n_runs": int(len(g)),
                "mean_ret_mean": float(g["mean_ret"].mean()),
                "ann_sharpe_mean": float(g["ann_sharpe_like"].mean()),
                "maxdd_mean": float(g["max_drawdown_like"].mean()),
                "worst5_mean": float(g["worst_5pct_day_mean"].mean()),
                "mean_ret_std": float(g["mean_ret"].std()),
                "ann_sharpe_std": float(g["ann_sharpe_like"].std()),
            })
    compact = pd.DataFrame(summary_rows)
    compact.to_csv(TAB / "table_16_random_shuffled_sanity_compact.csv", index=False)

    return compact


def experiment_gate_vs_riemannian():
    if not RIEM_DAILY.exists() or not RIEM_STATE.exists():
        print("[SKIP] gate vs riemannian: missing files")
        return pd.DataFrame()

    rd = load_daily_model(RIEM_DAILY, "SPD-Riemannian PPO Gate")
    rs = pd.read_csv(RIEM_STATE)
    rs["Date"] = pd.to_datetime(rs["Date"])
    m = rd.merge(rs, on="Date", how="inner").sort_values("Date")
    if "gate" not in m.columns:
        print("[SKIP] no gate")
        return pd.DataFrame()

    # Use riemann_dist_z if available; fallback to raw dist.
    col = "riemann_dist_z" if "riemann_dist_z" in m.columns else "riemann_dist"
    m["riem_bucket"] = pd.qcut(m[col].rank(method="first"), q=3, labels=["low", "mid", "high"])
    rows = []
    for bucket, g in m.groupby("riem_bucket", observed=True):
        rows.append(summarize(g["ret"], f"Riemannian bucket {bucket}", {
            "bucket": str(bucket),
            "n_days": int(len(g)),
            "gate_mean": float(g["gate"].mean()),
            "riemann_metric_mean": float(g[col].mean()),
            "full_exposure_ratio": float((g["gate"] >= 0.999).mean()),
            "reduced_exposure_ratio": float((g["gate"] < 0.999).mean()),
        }))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_17_gate_by_riemannian_state.csv", index=False)

    # Daily scatter
    plt.figure(figsize=(7.2, 4.2))
    plt.scatter(m[col], m["gate"], s=8, alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("Exposure gate")
    plt.title("SPD-Riemannian market-state distance vs exposure gate")
    plt.tight_layout()
    plt.savefig(FIG / "fig_17_gate_vs_riemannian_distance.pdf")
    plt.savefig(FIG / "fig_17_gate_vs_riemannian_distance.png", dpi=220)
    plt.close()

    # Time series with two axes saved as separate simple plot
    plt.figure(figsize=(8.0, 4.0))
    plt.plot(m["Date"], m["gate"], label="Gate", linewidth=1.2)
    plt.plot(m["Date"], m[col], label=col, linewidth=0.9)
    plt.xlabel("Date")
    plt.title("Exposure gate and SPD-Riemannian state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_17_gate_and_riemannian_timeseries.pdf")
    plt.savefig(FIG / "fig_17_gate_and_riemannian_timeseries.png", dpi=220)
    plt.close()

    return out


def experiment_regime_split(base_daily):
    if not RIEM_DAILY.exists() or not RIEM_STATE.exists():
        return pd.DataFrame()

    base = base_daily.copy()
    rd = load_daily_model(RIEM_DAILY, "SPD-Riemannian PPO Gate")
    rs = pd.read_csv(RIEM_STATE)
    rs["Date"] = pd.to_datetime(rs["Date"])
    col = "riemann_dist_z" if "riemann_dist_z" in rs.columns else "riemann_dist"

    m = base.merge(rd[["Date", "ret", "gate"]], on="Date", how="inner").merge(rs[["Date", col]], on="Date", how="inner")
    m = m.sort_values("Date")
    m["riem_regime"] = pd.qcut(m[col].rank(method="first"), q=3, labels=["low_riem", "mid_riem", "high_riem"])

    # Additional realized base-return regimes.
    m["base_down"] = np.where(m["base_ret"] < 0, "base_down", "base_up")
    m["base_tail"] = np.where(m["base_ret"] <= m["base_ret"].quantile(0.10), "base_worst10pct", "normal")

    rows = []
    for regime_col in ["riem_regime", "base_down", "base_tail"]:
        for regime, g in m.groupby(regime_col, observed=True):
            b = summarize(g["base_ret"], "Base", {"regime_type": regime_col, "regime": str(regime)})
            f = summarize(g["ret"], "SPD-Riemannian PPO Gate", {"regime_type": regime_col, "regime": str(regime), "gate_mean": float(g["gate"].mean())})
            rows.extend([b, f])
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_18_regime_performance.csv", index=False)
    return out


def experiment_validation_audit():
    rows = [
        {
            "component": "Stock sequence normalization",
            "selection_or_estimation_source": "training split only",
            "search_space_or_rule": "z-score using train statistics; clip [-8,8]",
            "selected_value": "fixed preprocessing rule",
            "test_used_for_selection": "No",
        },
        {
            "component": "Market sequence normalization",
            "selection_or_estimation_source": "training split only",
            "search_space_or_rule": "z-score using train statistics; clip [-8,8]",
            "selected_value": "fixed preprocessing rule",
            "test_used_for_selection": "No",
        },
        {
            "component": "Temporal candidate size M",
            "selection_or_estimation_source": "validation split",
            "search_space_or_rule": "{80,120,160,200,300}",
            "selected_value": "validation-selected",
            "test_used_for_selection": "No",
        },
        {
            "component": "Action admission weight omega",
            "selection_or_estimation_source": "validation split",
            "search_space_or_rule": "{0.5,1.0,1.5}",
            "selected_value": "validation-selected",
            "test_used_for_selection": "No",
        },
        {
            "component": "TopK portfolio size",
            "selection_or_estimation_source": "protocol",
            "search_space_or_rule": "Top30 main protocol",
            "selected_value": "30",
            "test_used_for_selection": "No",
        },
        {
            "component": "Base exposure",
            "selection_or_estimation_source": "protocol",
            "search_space_or_rule": "softmax projection total exposure",
            "selected_value": "0.5",
            "test_used_for_selection": "No",
        },
        {
            "component": "Max single-stock weight",
            "selection_or_estimation_source": "protocol",
            "search_space_or_rule": "portfolio projection cap",
            "selected_value": "0.10",
            "test_used_for_selection": "No",
        },
        {
            "component": "Geometry overlay coefficients",
            "selection_or_estimation_source": "validation split",
            "search_space_or_rule": "crowding / limit-up / liquidity coefficient grid",
            "selected_value": "validation-selected diagnostic",
            "test_used_for_selection": "No",
        },
        {
            "component": "SPD-Riemannian reference matrix",
            "selection_or_estimation_source": "training split only",
            "search_space_or_rule": "log-Euclidean mean of training-date SPD covariances",
            "selected_value": "train-period reference",
            "test_used_for_selection": "No",
        },
        {
            "component": "PPO reward coefficients",
            "selection_or_estimation_source": "validation split",
            "search_space_or_rule": "dd_penalty {0.5,1,2,4}; loss_penalty {0,0.2,0.5}; turnover_penalty {0,0.01,0.03}",
            "selected_value": "validation-selected",
            "test_used_for_selection": "No",
        },
        {
            "component": "PPO random seeds",
            "selection_or_estimation_source": "validation split",
            "search_space_or_rule": "{2021,2022,2023,2024,2025}",
            "selected_value": "validation-selected controller",
            "test_used_for_selection": "No",
        },
        {
            "component": "T+1 robustness",
            "selection_or_estimation_source": "frozen test diagnostic",
            "search_space_or_rule": "apply frozen final gate to delayed base-return sequence",
            "selected_value": "no retraining",
            "test_used_for_selection": "No model selection",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_19_validation_selection_audit.csv", index=False)
    return out


def main():
    print("[LOAD] base daily")
    base_daily = find_base_daily().sort_values("Date").reset_index(drop=True)
    print(base_daily.head().to_string(index=False))
    print(base_daily.tail().to_string(index=False))
    print("[BASE]", summarize(base_daily["base_ret"], "Action Admission Base"))

    print("\n[EXP] fixed exposure")
    fixed = experiment_fixed_exposure(base_daily)
    print(fixed.to_string(index=False))

    print("\n[EXP] random / shuffled sanity")
    sanity = experiment_random_sanity(base_daily)
    print(sanity.to_string(index=False))

    print("\n[EXP] gate vs riemannian")
    gate_riem = experiment_gate_vs_riemannian()
    if len(gate_riem):
        print(gate_riem.to_string(index=False))

    print("\n[EXP] regime split")
    regime = experiment_regime_split(base_daily)
    if len(regime):
        print(regime.head(30).to_string(index=False))

    print("\n[EXP] validation audit")
    audit = experiment_validation_audit()
    print(audit.to_string(index=False))

    # Rounded copies
    for p in TAB.glob("*.csv"):
        if p.name.endswith("_rounded.csv"):
            continue
        try:
            d = pd.read_csv(p)
            for c in d.columns:
                if pd.api.types.is_numeric_dtype(d[c]):
                    d[c] = d[c].round(6)
            d.to_csv(TAB / (p.stem + "_rounded.csv"), index=False)
        except Exception as e:
            print("[WARN] rounding failed", p, e)

    status = []
    for p in sorted(list(TAB.glob("*")) + list(FIG.glob("*"))):
        status.append({"file": str(p), "size_kb": round(p.stat().st_size / 1024, 2)})
    pd.DataFrame(status).to_csv(OUT / "status_icdm_extra_defense.csv", index=False)
    print("\n===== STATUS =====")
    print(pd.DataFrame(status).to_string(index=False))
    print("[OK] saved to", OUT)

if __name__ == "__main__":
    main()
