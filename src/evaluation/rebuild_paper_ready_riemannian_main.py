from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "outputs/dive_trader_v2/final_paper_riemannian_main"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ANN = 252 / 5

# -----------------------------
# Paths
# -----------------------------

PATHS = {
    "V2X-B Base": "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv",
    "Current PG Gate": "outputs/dive_trader_v2/v2x_rl_exposure_formal/daily_test_five_020_040_060_080_100_seed2022.csv",
    "PPO Gate": "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_ppo.csv",
    "Riemannian-PPO Gate": "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_riemannian_ppo.csv",
    "V2X-G Geometry Overlay": "outputs/dive_trader_v2/v2x_g_cross_geometry_overlay/daily_test_v2x_g_cross_geometry_overlay.csv",
    "V2R role-aware TopK30": "outputs/dive_trader_v2/v2r_role_aware_router_topk30/daily_test_v2r_role_aware_router.csv",
    "V2L Frequency-aware Temporal": "outputs/dive_trader_v2/experts/v2l_frequency_expert/daily_test_v2l_frequency_expert.csv",
    "V2H Factor Tree": "outputs/dive_trader_v2/experts/factor_tree_backtest/daily_test_factor_tree.csv",
    "V2M MASTER-style": "outputs/dive_trader_v2/experts/v2m_master_style/daily_test_v2m_master_style.csv",
    "V2N StockMixer-style": "outputs/dive_trader_v2/experts/v2n_stockmixer_style/daily_test_v2n_stockmixer_style.csv",
    "V2J RankIC Linear": "outputs/dive_trader_v2/experts/v2j_rankic_linear/daily_test_v2j_rankic_linear.csv",
    "Kronos-small official frozen": "outputs/dive_trader_v2/baselines/kronos_small_official_smoke20k/daily_test_kronos_small_official_frozen.csv",
    "TimeMoE-50M official frozen": "outputs/dive_trader_v2/baselines/timemoe_50m_official/daily_test_timemoe_50m_official_frozen.csv",
    "Chronos-Bolt official frozen": "outputs/dive_trader_v2/baselines/chronos_bolt_official/daily_test_chronos_bolt_official_frozen.csv",
}

FALLBACKS = {
    "PPO Gate": [
        "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_ppo_exposure.csv",
        "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_ppo_test.csv",
        "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_ppo_gate.csv",
    ],
    "Riemannian-PPO Gate": [
        "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_riemannian_ppo_exposure.csv",
        "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_riemannian_ppo_test.csv",
        "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_riemannian_ppo_gate.csv",
    ],
}

ROLE = {
    "Riemannian-PPO Gate": ("Ours Final", "Risk-regularized Riemannian-PPO exposure controller"),
    "Current PG Gate": ("Ours Ablation", "Lightweight policy-gradient exposure gate"),
    "PPO Gate": ("Ours Ablation", "Vanilla PPO exposure gate"),
    "V2X-B Base": ("Ours Base", "Cross-sectional action-admission portfolio before exposure control"),
    "V2X-G Geometry Overlay": ("Ours Diagnostic", "Geometry/concentration diagnostic overlay"),
    "V2R role-aware TopK30": ("Intermediate controller", "Role-aware router reference"),
    "V2L Frequency-aware Temporal": ("Single expert", "Temporal candidate discovery"),
    "V2H Factor Tree": ("Single expert", "Cross-sectional action scorer"),
    "V2M MASTER-style": ("Single expert", "Market-body transformer-style expert"),
    "V2N StockMixer-style": ("Single expert", "Market-body mixer-style expert"),
    "V2J RankIC Linear": ("Traditional factor", "Traditional interpretable factor anchor"),
    "Kronos-small official frozen": ("Official foundation baseline", "Official frozen foundation baseline"),
    "TimeMoE-50M official frozen": ("Official foundation baseline", "Official frozen foundation baseline"),
    "Chronos-Bolt official frozen": ("Official foundation baseline", "Official frozen foundation baseline"),
}

ORDER = [
    "Riemannian-PPO Gate",
    "Current PG Gate",
    "PPO Gate",
    "V2X-B Base",
    "V2X-G Geometry Overlay",
    "V2R role-aware TopK30",
    "V2L Frequency-aware Temporal",
    "V2H Factor Tree",
    "V2M MASTER-style",
    "V2N StockMixer-style",
    "V2J RankIC Linear",
    "Kronos-small official frozen",
    "TimeMoE-50M official frozen",
    "Chronos-Bolt official frozen",
]

# -----------------------------
# Utilities
# -----------------------------

def find_path(name, p):
    pp = Path(p)
    if pp.exists():
        return pp
    for alt in FALLBACKS.get(name, []):
        if Path(alt).exists():
            return Path(alt)
    return pp

def max_drawdown(ret):
    r = np.asarray(ret, dtype=float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min()), nav, dd

def sharpe_like(ret):
    r = np.asarray(ret, dtype=float)
    return float((r.mean() / (r.std() + 1e-12)) * np.sqrt(ANN))

def sortino_like(ret):
    r = np.asarray(ret, dtype=float)
    down = r[r < 0]
    if len(down) == 0:
        return np.nan
    return float((r.mean() / (down.std() + 1e-12)) * np.sqrt(ANN))

def calmar_like(ret):
    r = np.asarray(ret, dtype=float)
    mdd, _, _ = max_drawdown(r)
    return float((r.mean() * ANN) / abs(mdd)) if abs(mdd) > 1e-12 else np.nan

def worst_tail_mean(ret, q=0.05):
    r = np.asarray(ret, dtype=float)
    k = max(1, int(np.ceil(q * len(r))))
    return float(np.sort(r)[:k].mean())

def summarize_ret(ret):
    r = np.asarray(ret, dtype=float)
    mdd, _, _ = max_drawdown(r)
    return {
        "n_days": int(len(r)),
        "mean_ret": float(r.mean()),
        "std_ret": float(r.std() + 1e-12),
        "ann_ret_like": float(r.mean() * ANN),
        "ann_sharpe_like": sharpe_like(r),
        "sortino_like": sortino_like(r),
        "calmar_like": calmar_like(r),
        "max_drawdown_like": mdd,
        "worst_5pct_day_mean": worst_tail_mean(r),
        "positive_day_ratio": float((r > 0).mean()),
    }

def read_model(name, path):
    p = find_path(name, path)
    if not p.exists():
        print("[MISS]", name, p)
        return None

    d = pd.read_csv(p)
    d["Date"] = pd.to_datetime(d["Date"])

    if name == "V2X-B Base":
        if "day_net_ret" in d.columns:
            out = d[["Date", "day_net_ret"]].drop_duplicates("Date").rename(columns={"day_net_ret": "ret"})
            out["gate"] = 1.0
            out["turnover"] = d[["Date", "day_turnover"]].drop_duplicates("Date")["day_turnover"].values if "day_turnover" in d.columns else np.nan
            return out.sort_values("Date").reset_index(drop=True)
        raise RuntimeError("V2X-B Base missing day_net_ret")

    if "ret" not in d.columns:
        if "model_ret" in d.columns:
            d = d.rename(columns={"model_ret": "ret"})
        elif "rl_ret" in d.columns:
            d = d.rename(columns={"rl_ret": "ret"})
        elif "day_net_ret" in d.columns:
            d = d.rename(columns={"day_net_ret": "ret"})
        else:
            raise RuntimeError(f"{name} missing return column: {d.columns.tolist()}")

    if "gate" not in d.columns:
        d["gate"] = np.nan

    if "turnover" not in d.columns:
        if "day_turnover" in d.columns:
            d["turnover"] = d["day_turnover"]
        elif "turnover_mean" in d.columns:
            d["turnover"] = d["turnover_mean"]
        else:
            d["turnover"] = np.nan

    keep = ["Date", "ret", "gate", "turnover"]
    return d[keep].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

def load_all():
    out = {}
    for name in ORDER:
        if name not in PATHS:
            continue
        d = read_model(name, PATHS[name])
        if d is not None:
            out[name] = d
            print("[LOAD]", name, d.shape, d["Date"].min().date(), d["Date"].max().date())
    return out

def align_two(a, b):
    x = a[["Date", "ret"]].rename(columns={"ret": "ret_a"})
    y = b[["Date", "ret"]].rename(columns={"ret": "ret_b"})
    m = x.merge(y, on="Date", how="inner").sort_values("Date")
    return m["ret_a"].to_numpy(float), m["ret_b"].to_numpy(float)

# -----------------------------
# Tables
# -----------------------------

def build_main_table(models):
    rows = []
    for name in ORDER:
        if name not in models:
            continue
        d = models[name]
        s = summarize_ret(d["ret"].to_numpy(float))
        group, paper_role = ROLE[name]
        rows.append({
            "group": group,
            "model": name,
            "paper_role": paper_role,
            **s,
            "gate_mean": float(d["gate"].mean()) if not d["gate"].isna().all() else np.nan,
            "turnover_mean": float(d["turnover"].mean()) if not d["turnover"].isna().all() else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table_1_main_comparison_riemannian_main_full.csv", index=False)

    compact_cols = [
        "group", "model", "paper_role", "n_days", "mean_ret", "ann_ret_like",
        "ann_sharpe_like", "sortino_like", "calmar_like",
        "max_drawdown_like", "worst_5pct_day_mean", "positive_day_ratio", "gate_mean"
    ]
    compact = df[compact_cols].copy()
    for c in compact.columns:
        if compact[c].dtype.kind in "fc":
            compact[c] = compact[c].round(4)
    compact.to_csv(TAB / "table_1_main_comparison_riemannian_main_compact.csv", index=False)
    return df

def build_policy_behavior(models):
    # Frequency
    rows = []
    for name in ["Riemannian-PPO Gate", "Current PG Gate", "PPO Gate"]:
        if name not in models:
            continue
        d = models[name].copy()
        if d["gate"].isna().all():
            continue
        for g, c in d["gate"].round(6).value_counts().sort_index().items():
            sub = d[d["gate"].round(6) == g]
            rows.append({
                "model": name,
                "gate": float(g),
                "n_days": int(c),
                "ratio": float(c / len(d)),
                "mean_ret": float(sub["ret"].mean()),
                "min_ret": float(sub["ret"].min()),
                "positive_day_ratio": float((sub["ret"] > 0).mean()),
            })
    freq = pd.DataFrame(rows)
    freq.to_csv(TAB / "table_4a_policy_action_frequency_riemannian_main.csv", index=False)

    # Worst/best by base days.
    wb_rows = []
    if "V2X-B Base" in models:
        base = models["V2X-B Base"][["Date", "ret"]].rename(columns={"ret": "base_ret"}).sort_values("base_ret")
        worst_dates = set(base.head(20)["Date"])
        best_dates = set(base.tail(20)["Date"])

        for name in ["Riemannian-PPO Gate", "Current PG Gate", "PPO Gate"]:
            if name not in models:
                continue
            d = models[name].merge(base, on="Date", how="inner")
            for section, dates in [("worst20_base_days", worst_dates), ("best20_base_days", best_dates)]:
                sub = d[d["Date"].isin(dates)]
                wb_rows.append({
                    "model": name,
                    "section": section,
                    "n_days": int(len(sub)),
                    "base_ret_mean": float(sub["base_ret"].mean()),
                    "model_ret_mean": float(sub["ret"].mean()),
                    "gate_mean": float(sub["gate"].mean()) if not sub["gate"].isna().all() else np.nan,
                    "reduced_exposure_ratio": float((sub["gate"] < 0.999).mean()) if not sub["gate"].isna().all() else np.nan,
                    "full_exposure_ratio": float((sub["gate"] >= 0.999).mean()) if not sub["gate"].isna().all() else np.nan,
                    "protection_mean": float((sub["base_ret"] - sub["ret"]).mean()),
                })
    wb = pd.DataFrame(wb_rows)
    wb.to_csv(TAB / "table_4b_policy_worst_best_behavior_riemannian_main.csv", index=False)
    return freq, wb

def build_bootstrap_table():
    src = Path("outputs/dive_trader_v2/kdd_riemannian_ppo_diagnostics/tables/table_12_riemannian_ppo_block_bootstrap.csv")
    if src.exists():
        df = pd.read_csv(src)
        df.to_csv(TAB / "table_6_block_bootstrap_riemannian_main.csv", index=False)
        compact = df[df["comparison"].isin([
            "Riemannian-PPO Gate - V2X-B Base",
            "Riemannian-PPO Gate - Current PG Gate",
            "Current PG Gate - V2X-B Base",
        ])].copy()
        compact.to_csv(TAB / "table_6_block_bootstrap_riemannian_main_compact.csv", index=False)
        return df
    print("[WARN] bootstrap source missing:", src)
    return pd.DataFrame()

def build_institutional_table():
    src = Path("outputs/dive_trader_v2/kdd_final_extra/tables/appendix_table_institutional_risk_metrics.csv")
    if not src.exists():
        print("[WARN] institutional metrics source missing:", src)
        return pd.DataFrame()
    df = pd.read_csv(src)
    order = [
        "Riemannian-PPO Gate", "Current PG Gate", "PPO Gate", "V2X-B Base",
        "V2X-G Geometry Overlay", "V2L Temporal Expert", "V2H Factor Tree",
        "Kronos-small Frozen", "TimeMoE-50M Frozen", "Chronos-Bolt Frozen"
    ]
    df["__order"] = df["model"].map({m: i for i, m in enumerate(order)}).fillna(999)
    df = df.sort_values("__order").drop(columns="__order")
    df.to_csv(TAB / "table_A_institutional_risk_metrics_riemannian_main.csv", index=False)
    compact_cols = [
        "model", "n_days", "mean_ret", "ann_ret_like", "ann_sharpe_like",
        "sortino_like", "calmar_like", "max_drawdown_like",
        "worst_5pct_day_mean", "positive_day_ratio", "gate_mean"
    ]
    compact = df[[c for c in compact_cols if c in df.columns]].copy()
    compact.to_csv(TAB / "table_A_institutional_risk_metrics_riemannian_main_compact.csv", index=False)
    return df

def build_concentration_table(models):
    # Reuse old concentration if present, then add Riemannian row reconstructed from Base normalized concentration.
    src = Path("outputs/dive_trader_v2/final_paper_tables_ready/table_3_concentration_paper_ready.csv")
    if src.exists():
        old = pd.read_csv(src)
    else:
        old = pd.DataFrame()

    # Use known concentration inherited from V2X-B if old table has it.
    rows = []
    if len(old):
        rows.extend(old.to_dict(orient="records"))

    base_row = None
    for r in rows:
        if str(r.get("model", "")).startswith("V2X-B"):
            base_row = r
            break

    if base_row is not None and "Riemannian-PPO Gate" in models:
        rr = dict(base_row)
        rr["group"] = "Ours Final"
        rr["model"] = "Riemannian-PPO Gate"
        rr["protocol"] = "Riemannian-PPO exposure overlay reconstructed on V2X-B holdings"
        rr["gate_mean"] = float(models["Riemannian-PPO Gate"]["gate"].mean())
        rr["capital_exposure_mean"] = 0.5 * rr["gate_mean"]
        rr["concentration_note"] = "Riemannian-PPO rescales exposure without reselecting stocks; normalized concentration is inherited from V2X-B."
        rows.insert(0, rr)

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_3_concentration_riemannian_main.csv", index=False)
    return out

def build_constraints_table(models):
    src = Path("outputs/dive_trader_v2/final_paper_tables_ready/table_2_china_constraints_paper_ready.csv")
    if src.exists():
        df = pd.read_csv(src)
    else:
        df = pd.DataFrame()

    # Add Riemannian T+1 if available from diagnostics, otherwise do not fabricate.
    # The base constraints remain valid as action-admission constraints.
    df.to_csv(TAB / "table_2_china_constraints_riemannian_main.csv", index=False)
    return df

def build_yearly_table(models):
    rows = []
    for name in ["Riemannian-PPO Gate", "Current PG Gate", "PPO Gate", "V2X-B Base", "V2X-G Geometry Overlay"]:
        if name not in models:
            continue
        d = models[name].copy()
        d["year"] = d["Date"].dt.year
        for y, g in d.groupby("year"):
            s = summarize_ret(g["ret"].to_numpy(float))
            rows.append({
                "model": name,
                "year": int(y),
                **s,
                "gate_mean": float(g["gate"].mean()) if not g["gate"].isna().all() else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_8_yearly_exposure_variants_riemannian_main.csv", index=False)
    return out

# -----------------------------
# Figures
# -----------------------------

def plot_nav_dd(models):
    use = ["Riemannian-PPO Gate", "Current PG Gate", "PPO Gate", "V2X-B Base", "Kronos-small official frozen"]
    use = [x for x in use if x in models]

    plt.figure(figsize=(8, 4.8))
    for name in use:
        d = models[name].sort_values("Date")
        _, nav, _ = max_drawdown(d["ret"].to_numpy(float))
        lw = 2.4 if name == "Riemannian-PPO Gate" else 1.4
        plt.plot(d["Date"], nav, label=name, linewidth=lw)
    plt.xlabel("Date")
    plt.ylabel("Net Asset Value")
    plt.title("Main NAV Comparison with Riemannian-PPO Final Controller")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_2a_nav_curve_riemannian_main.pdf", bbox_inches="tight")
    plt.savefig(FIG / "fig_2a_nav_curve_riemannian_main.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4.8))
    for name in use:
        d = models[name].sort_values("Date")
        _, _, dd = max_drawdown(d["ret"].to_numpy(float))
        lw = 2.4 if name == "Riemannian-PPO Gate" else 1.4
        plt.plot(d["Date"], dd, label=name, linewidth=lw)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Main Drawdown Comparison with Riemannian-PPO Final Controller")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_2b_drawdown_curve_riemannian_main.pdf", bbox_inches="tight")
    plt.savefig(FIG / "fig_2b_drawdown_curve_riemannian_main.png", dpi=220, bbox_inches="tight")
    plt.close()

def plot_gate(models):
    plt.figure(figsize=(8, 3.8))
    for name in ["Riemannian-PPO Gate", "Current PG Gate", "PPO Gate"]:
        if name not in models:
            continue
        d = models[name].sort_values("Date")
        plt.plot(d["Date"], d["gate"], label=name, linewidth=1.3)
    plt.xlabel("Date")
    plt.ylabel("Exposure Gate")
    plt.title("Exposure Gate Time Series")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_3_gate_timeseries_riemannian_main.pdf", bbox_inches="tight")
    plt.savefig(FIG / "fig_3_gate_timeseries_riemannian_main.png", dpi=220, bbox_inches="tight")
    plt.close()

def plot_worst_best(wb):
    if wb is None or wb.empty:
        return
    pivot = wb.pivot_table(index="model", columns="section", values="model_ret_mean", aggfunc="first")
    order = [x for x in ["Riemannian-PPO Gate", "Current PG Gate", "PPO Gate"] if x in pivot.index]
    pivot = pivot.loc[order]
    ax = pivot.plot(kind="bar", figsize=(7, 4.5))
    ax.set_ylabel("Mean return on selected days")
    ax.set_title("Worst/Best Base-Day Behavior")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "fig_4_worst_best_behavior_riemannian_main.pdf", bbox_inches="tight")
    plt.savefig(FIG / "fig_4_worst_best_behavior_riemannian_main.png", dpi=220, bbox_inches="tight")
    plt.close()

def plot_risk_return(main_df):
    d = main_df.copy()
    d = d.dropna(subset=["ann_sharpe_like", "max_drawdown_like"])
    plt.figure(figsize=(7, 5))
    for _, r in d.iterrows():
        size = 80 if r["model"] == "Riemannian-PPO Gate" else 35
        plt.scatter(r["max_drawdown_like"], r["ann_sharpe_like"], s=size)
        if r["model"] in ["Riemannian-PPO Gate", "Current PG Gate", "V2X-B Base", "Kronos-small official frozen", "TimeMoE-50M official frozen", "Chronos-Bolt official frozen"]:
            plt.text(r["max_drawdown_like"], r["ann_sharpe_like"], r["model"], fontsize=7)
    plt.xlabel("Max Drawdown-like")
    plt.ylabel("Annualized Sharpe-like")
    plt.title("Risk-Return Diagnostic")
    plt.tight_layout()
    plt.savefig(FIG / "fig_7_risk_return_scatter_riemannian_main.pdf", bbox_inches="tight")
    plt.savefig(FIG / "fig_7_risk_return_scatter_riemannian_main.png", dpi=220, bbox_inches="tight")
    plt.close()

def plot_yearly(yearly):
    if yearly is None or yearly.empty:
        return
    use = yearly[yearly["model"].isin(["Riemannian-PPO Gate", "Current PG Gate", "V2X-B Base"])].copy()
    pivot = use.pivot_table(index="year", columns="model", values="ann_sharpe_like", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(7.5, 4.6))
    ax.set_ylabel("Annualized Sharpe-like")
    ax.set_title("Yearly Stability of Exposure Variants")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "fig_8_yearly_vs_indices_riemannian_main.pdf", bbox_inches="tight")
    plt.savefig(FIG / "fig_8_yearly_vs_indices_riemannian_main.png", dpi=220, bbox_inches="tight")
    plt.close()

# -----------------------------
# Main
# -----------------------------

def main():
    models = load_all()
    main_df = build_main_table(models)
    freq, wb = build_policy_behavior(models)
    boot = build_bootstrap_table()
    inst = build_institutional_table()
    conc = build_concentration_table(models)
    cons = build_constraints_table(models)
    yearly = build_yearly_table(models)

    plot_nav_dd(models)
    plot_gate(models)
    plot_worst_best(wb)
    plot_risk_return(main_df)
    plot_yearly(yearly)

    # Rounded copies
    for p in TAB.glob("*.csv"):
        try:
            d = pd.read_csv(p)
            for c in d.columns:
                if d[c].dtype.kind in "fc":
                    d[c] = d[c].round(6)
            d.to_csv(TAB / (p.stem + "_rounded.csv"), index=False)
        except Exception as e:
            print("[ROUND FAIL]", p, e)

    index = pd.DataFrame([
        {"table": "table_1_main_comparison_riemannian_main_compact.csv", "paper_section": "Main Results", "main_message": "Riemannian-PPO is the final exposure controller; Current PG is a lightweight ablation."},
        {"table": "table_2_china_constraints_riemannian_main.csv", "paper_section": "Market Constraint Robustness", "main_message": "A-share trading constraints are evaluated on the action-admission portfolio and remain relevant before exposure control."},
        {"table": "table_3_concentration_riemannian_main.csv", "paper_section": "Concentration Diagnostics", "main_message": "Riemannian-PPO rescales exposure without reselecting stocks; normalized concentration follows V2X-B."},
        {"table": "table_4a_policy_action_frequency_riemannian_main.csv", "paper_section": "Policy Behavior", "main_message": "Riemannian-PPO reduces exposure more frequently than Current PG and is more downside-sensitive."},
        {"table": "table_4b_policy_worst_best_behavior_riemannian_main.csv", "paper_section": "Policy Behavior", "main_message": "Riemannian-PPO slightly improves worst-day protection but sacrifices some best-day upside."},
        {"table": "table_6_block_bootstrap_riemannian_main_compact.csv", "paper_section": "Statistical Analysis", "main_message": "Riemannian-PPO significantly improves over Base; the difference versus Current PG is not statistically decisive."},
        {"table": "table_A_institutional_risk_metrics_riemannian_main_compact.csv", "paper_section": "Appendix", "main_message": "Practitioner-oriented risk metrics support the downside-sensitive role of Riemannian-PPO."},
    ])
    index.to_csv(TAB / "paper_ready_table_index_riemannian_main.csv", index=False)

    status = []
    for p in sorted(list(TAB.glob("*.csv")) + list(FIG.glob("*"))):
        status.append({"file": str(p), "size_kb": round(p.stat().st_size / 1024, 2)})
    pd.DataFrame(status).to_csv(OUT / "status_riemannian_main_rebuild.csv", index=False)

    print("===== MAIN COMPACT =====")
    compact = pd.read_csv(TAB / "table_1_main_comparison_riemannian_main_compact_rounded.csv")
    print(compact.to_string(index=False))
    print("===== POLICY FREQUENCY =====")
    print(pd.read_csv(TAB / "table_4a_policy_action_frequency_riemannian_main_rounded.csv").to_string(index=False))
    print("===== WORST/BEST =====")
    print(pd.read_csv(TAB / "table_4b_policy_worst_best_behavior_riemannian_main_rounded.csv").to_string(index=False))
    print("===== INDEX =====")
    print(index.to_string(index=False))
    print("===== STATUS =====")
    print(pd.DataFrame(status).to_string(index=False))

if __name__ == "__main__":
    main()
