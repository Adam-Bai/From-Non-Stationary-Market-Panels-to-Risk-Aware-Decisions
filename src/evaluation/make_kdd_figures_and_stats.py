from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
READY = Path("outputs/dive_trader_v2/final_paper_tables_ready")
CLEANV2 = Path("outputs/dive_trader_v2/final_paper_tables_clean_v2")
OUT = Path("outputs/dive_trader_v2/kdd_defense_pack")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252.0

def read_csv(p):
    p = Path(p)
    if not p.exists():
        print("[MISS]", p)
        return None
    return pd.read_csv(p)

def normalize_date(df):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def infer_ret_col(df):
    for c in ["ret", "day_net_ret", "net_ret", "daily_ret"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot infer ret col from {df.columns.tolist()}")

def daily_from_file(path, name=None):
    df = read_csv(path)
    if df is None:
        return None
    df = normalize_date(df)
    ret_col = infer_ret_col(df)
    out = df[["Date", ret_col]].rename(columns={ret_col: "ret"}).copy()
    out = out.groupby("Date", as_index=False)["ret"].first().sort_values("Date")
    if name:
        out["model"] = name
    return out

def daily_from_ledger(path, name=None):
    df = read_csv(path)
    if df is None:
        return None
    df = normalize_date(df)
    out = df.groupby("Date", as_index=False).agg(
        ret=("day_net_ret", "first"),
        turnover=("day_turnover", "first"),
        gate=("gate", "first") if "gate" in df.columns else ("rank", "size"),
    ).sort_values("Date")
    if name:
        out["model"] = name
    return out

def calc_metrics(ret):
    ret = pd.Series(ret, dtype="float64").dropna()
    n = len(ret)
    if n == 0:
        return {}
    mean = float(ret.mean())
    std = float(ret.std(ddof=0))
    nav = (1.0 + ret).cumprod()
    dd = nav / nav.cummax() - 1.0
    return {
        "n_days": n,
        "mean_ret": mean,
        "std_ret": std,
        "ann_ret_like": mean * TRADING_DAYS / 5.0,
        "ann_sharpe_like": mean / (std + 1e-12) * np.sqrt(TRADING_DAYS / 5.0),
        "max_drawdown_like": float(dd.min()),
        "positive_day_ratio": float((ret > 0).mean()),
    }

def savefig(name):
    for ext in ["png", "pdf"]:
        p = FIG / f"{name}.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=240)
        print("[OK]", p)
    plt.close()

def drawdown(ret):
    nav = (1.0 + pd.Series(ret, dtype="float64").fillna(0)).cumprod()
    return nav / nav.cummax() - 1.0

# --------------------------
# Load daily series
# --------------------------
paths = {
    "V2X-RL": "outputs/dive_trader_v2/v2x_rl_exposure_formal/daily_test_five_020_040_060_080_100_seed2022.csv",
    "V2X-B": "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv",
    "V2X-G": "outputs/dive_trader_v2/v2x_g_cross_geometry_overlay/daily_test_v2x_g_cross_geometry_overlay.csv",
    "V2L": "outputs/dive_trader_v2/experts/v2l_frequency_expert/daily_test_v2l_frequency_expert.csv",
    "V2H": "outputs/dive_trader_v2/experts/factor_tree_backtest/daily_test_factor_tree.csv",
    "Kronos": "outputs/dive_trader_v2/baselines/kronos_small_official_smoke20k/daily_test_kronos_small_official_frozen.csv",
    "TimeMoE": "outputs/dive_trader_v2/baselines/timemoe_50m_official/daily_test_timemoe_50m_official_frozen.csv",
    "Chronos": "outputs/dive_trader_v2/baselines/chronos_bolt_official/daily_test_chronos_bolt_official_frozen.csv",
}

daily = {}
for name, path in paths.items():
    if name == "V2X-B":
        d = daily_from_ledger(path, name)
    else:
        d = daily_from_file(path, name)
    if d is not None:
        daily[name] = d
        print("[LOAD]", name, d.shape, d["Date"].min(), d["Date"].max())

# indices
idx_path = Path("outputs/dive_trader_v2/final_paper_tables_v2/final_table_7b_strategy_vs_indices_aligned.csv")
idx_table = read_csv(idx_path)

# --------------------------
# Fig 1: NAV
# --------------------------
plt.figure(figsize=(10, 5.6))
for name in ["V2X-RL", "V2X-B", "V2X-G", "V2L", "V2H", "Kronos"]:
    if name in daily:
        d = daily[name].sort_values("Date")
        nav = (1.0 + d["ret"].fillna(0)).cumprod()
        plt.plot(d["Date"], nav, label=name, linewidth=1.8)
plt.xlabel("Date")
plt.ylabel("Cumulative NAV")
plt.title("Cumulative Portfolio Value on Test Period")
plt.legend(ncol=2)
plt.grid(True, alpha=0.25)
savefig("fig_2a_nav_curve")

# --------------------------
# Fig 2: Drawdown
# --------------------------
plt.figure(figsize=(10, 5.6))
for name in ["V2X-RL", "V2X-B", "V2X-G", "V2L", "V2H", "Kronos"]:
    if name in daily:
        d = daily[name].sort_values("Date")
        dd = drawdown(d["ret"])
        plt.plot(d["Date"], dd, label=name, linewidth=1.8)
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.title("Drawdown Curve on Test Period")
plt.legend(ncol=2)
plt.grid(True, alpha=0.25)
savefig("fig_2b_drawdown_curve")

# --------------------------
# Fig 3: RL gate behavior
# --------------------------
rl = daily.get("V2X-RL")
b = daily.get("V2X-B")
if rl is not None and "gate" not in rl.columns:
    raw = normalize_date(pd.read_csv(paths["V2X-RL"]))
    if "gate" in raw.columns:
        rl = raw[["Date", "ret", "gate"]].copy()
        daily["V2X-RL"] = rl

if rl is not None:
    if "gate" in rl.columns:
        plt.figure(figsize=(10, 4.8))
        plt.step(rl["Date"], rl["gate"], where="post", linewidth=1.6)
        plt.xlabel("Date")
        plt.ylabel("Exposure gate")
        plt.title("RL Exposure Gate over Time")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.25)
        savefig("fig_3_rl_gate_timeseries")

# --------------------------
# Fig 4: worst / best behavior
# --------------------------
if rl is not None and b is not None:
    x = b[["Date", "ret"]].rename(columns={"ret": "base_ret"}).merge(
        rl[["Date", "ret"] + (["gate"] if "gate" in rl.columns else [])].rename(columns={"ret": "rl_ret"}),
        on="Date",
        how="inner"
    )
    worst = x.nsmallest(20, "base_ret")
    best = x.nlargest(20, "base_ret")
    wb = pd.DataFrame([
        {"section": "Worst 20 base days", "V2X-B": worst["base_ret"].mean(), "V2X-RL": worst["rl_ret"].mean(),
         "gate_mean": worst["gate"].mean() if "gate" in worst.columns else np.nan},
        {"section": "Best 20 base days", "V2X-B": best["base_ret"].mean(), "V2X-RL": best["rl_ret"].mean(),
         "gate_mean": best["gate"].mean() if "gate" in best.columns else np.nan},
    ])
    wb.to_csv(TAB / "table_7a_worst_best_behavior_recomputed.csv", index=False)

    plt.figure(figsize=(7.5, 5))
    xpos = np.arange(len(wb))
    width = 0.35
    plt.bar(xpos - width/2, wb["V2X-B"], width, label="V2X-B")
    plt.bar(xpos + width/2, wb["V2X-RL"], width, label="V2X-RL")
    plt.xticks(xpos, wb["section"], rotation=0)
    plt.ylabel("Mean daily return")
    plt.title("Worst-Day Protection and Best-Day Preservation")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    savefig("fig_4_worst_best_behavior")

# --------------------------
# Fig 5: concentration
# --------------------------
conc = read_csv(READY / "table_3_concentration_paper_ready.csv")
if conc is not None:
    keep = conc[conc["model"].isin([
        "V2X-B Cross Action Admission",
        "V2X-RL+ five-level exposure",
        "V2X-G Cross-Geometry Overlay"
    ])].copy()
    metric_map = {
        "effective_n_mean": "Effective N",
        "top1_weight_mean": "Top1 weight",
        "top3_weight_mean": "Top3 weight",
        "top5_weight_mean": "Top5 weight",
        "herfindahl_mean": "Herfindahl",
        "capital_exposure_mean": "Capital exposure",
    }
    rows = []
    for _, r in keep.iterrows():
        for col, label in metric_map.items():
            if col in keep.columns and pd.notna(r.get(col, np.nan)):
                rows.append({"model": r["model"], "metric": label, "value": r[col]})
    cd = pd.DataFrame(rows)
    cd.to_csv(TAB / "table_concentration_plot_source.csv", index=False)

    for metric in ["Effective N", "Top3 weight", "Top5 weight", "Capital exposure"]:
        sub = cd[cd["metric"].eq(metric)]
        if len(sub) > 0:
            plt.figure(figsize=(7.5, 4.8))
            plt.bar(sub["model"], sub["value"])
            plt.ylabel(metric)
            plt.title(f"Concentration Diagnostic: {metric}")
            plt.xticks(rotation=20, ha="right")
            plt.grid(True, axis="y", alpha=0.25)
            savefig("fig_5_concentration_" + metric.lower().replace(" ", "_"))

# --------------------------
# Fig 6: constraints sensitivity
# --------------------------
constraints = read_csv(READY / "table_2_china_constraints_paper_ready.csv")
if constraints is not None:
    c = constraints.copy()
    c["constraint"] = c["constraint"].astype(str)

    # cost sensitivity for V2X-B
    sub = c[(c["model"].str.contains("V2X-B", na=False)) & (c["constraint"].str.contains("base_cost", na=False))]
    if len(sub) > 0:
        plt.figure(figsize=(7, 4.8))
        plt.plot(sub["cost_bps"], sub["mean_ret"], marker="o", label="mean_ret")
        plt.xlabel("Transaction cost (bps)")
        plt.ylabel("Mean return")
        plt.title("Cost Sensitivity")
        plt.grid(True, alpha=0.25)
        savefig("fig_6a_cost_sensitivity")

    # constraint deltas
    sub = c[(c["model"].str.contains("V2X-B", na=False)) & (~c["constraint"].str.contains("base_cost", na=False))]
    if len(sub) > 0 and "delta_mean_ret_vs_base5" in sub.columns:
        sub = sub.sort_values("delta_mean_ret_vs_base5")
        plt.figure(figsize=(10, 5.2))
        plt.bar(sub["constraint"], sub["delta_mean_ret_vs_base5"])
        plt.ylabel("Δ mean return vs base 5bps")
        plt.title("A-share Constraint Sensitivity")
        plt.xticks(rotation=35, ha="right")
        plt.grid(True, axis="y", alpha=0.25)
        savefig("fig_6b_constraint_delta_mean_ret")

# --------------------------
# Fig 7: risk-return scatter
# --------------------------
main = read_csv(READY / "table_1_main_comparison_paper_ready.csv")
if main is not None:
    plt.figure(figsize=(8, 5.6))
    for _, r in main.iterrows():
        if pd.isna(r.get("mean_ret")) or pd.isna(r.get("max_drawdown_like")):
            continue
        plt.scatter(r["max_drawdown_like"], r["mean_ret"], s=80)
        plt.text(r["max_drawdown_like"], r["mean_ret"], str(r["model"])[:18], fontsize=8)
    plt.xlabel("Max drawdown-like")
    plt.ylabel("Mean return")
    plt.title("Risk-Return Trade-off")
    plt.grid(True, alpha=0.25)
    savefig("fig_7_risk_return_scatter")

# --------------------------
# Fig 8: yearly vs indices
# --------------------------
idx_aligned = read_csv("outputs/dive_trader_v2/final_paper_tables_v2/final_table_7b_strategy_vs_indices_aligned.csv")
if idx_aligned is not None:
    df = idx_aligned.copy()
    df = df[df["period"].astype(str).isin(["2024", "2025", "2026", "Full"])]
    selected = df[df["asset"].isin([
        "V2X-RL+ five-level exposure",
        "V2X-RL+ best formal (five_020_040_060_080_100, seed=2022)",
        "CSI300", "CSI500", "CSI1000", "ChiNext", "Shanghai Composite"
    ])].copy()
    selected["asset"] = selected["asset"].replace({
        "V2X-RL+ best formal (five_020_040_060_080_100, seed=2022)": "V2X-RL"
    })
    pivot = selected.pivot_table(index="period", columns="asset", values="ann_sharpe_like", aggfunc="first")
    pivot = pivot.reindex(["2024", "2025", "2026", "Full"])
    pivot.to_csv(TAB / "table_yearly_vs_indices_plot_source.csv")
    plt.figure(figsize=(10, 5.4))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.ylabel("Annualized Sharpe-like")
    plt.title("Yearly Strategy Performance vs A-share Indices")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    savefig("fig_8_yearly_vs_indices")

# --------------------------
# Table 6: block bootstrap significance
# --------------------------
def align_pair(a, b):
    aa = daily[a][["Date", "ret"]].rename(columns={"ret": "a"})
    bb = daily[b][["Date", "ret"]].rename(columns={"ret": "b"})
    x = aa.merge(bb, on="Date", how="inner").sort_values("Date")
    return x

def block_bootstrap_diff(x, metric_func, block=10, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(x)
    starts = np.arange(max(1, n - block + 1))
    vals = []
    arr = x[["a", "b"]].to_numpy(dtype=float)
    for _ in range(n_boot):
        idxs = []
        while len(idxs) < n:
            s = int(rng.choice(starts))
            idxs.extend(range(s, min(s + block, n)))
        idxs = idxs[:n]
        sample = arr[idxs]
        vals.append(metric_func(sample[:, 0]) - metric_func(sample[:, 1]))
    vals = np.asarray(vals)
    return {
        "boot_mean": float(vals.mean()),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "p_value_two_sided": float(2 * min((vals <= 0).mean(), (vals >= 0).mean())),
        "win_rate": float((vals > 0).mean()),
    }

def m_mean(x):
    return float(np.mean(x))

def m_sharpe(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(TRADING_DAYS / 5.0))

def m_maxdd_improvement(x):
    # Higher is better: less negative max drawdown.
    return maxdd_metric(x)

def maxdd_metric(x):
    nav = np.cumprod(1.0 + np.asarray(x, dtype=float))
    dd = nav / np.maximum.accumulate(nav) - 1.0
    return float(dd.min())

pairs = [
    ("V2X-RL", "V2X-B"),
    ("V2X-RL", "V2L"),
    ("V2X-RL", "V2H"),
    ("V2X-RL", "Kronos"),
    ("V2X-G", "V2X-B"),
]

rows = []
for a, bname in pairs:
    if a not in daily or bname not in daily:
        continue
    x = align_pair(a, bname)
    if len(x) < 30:
        continue
    for metric_name, func in [
        ("mean_ret", m_mean),
        ("ann_sharpe_like", m_sharpe),
        ("max_drawdown_like", maxdd_metric),
    ]:
        obs = func(x["a"].to_numpy()) - func(x["b"].to_numpy())
        bs = block_bootstrap_diff(x, func, block=10, n_boot=2000, seed=2026)
        rows.append({
            "comparison": f"{a} - {bname}",
            "metric": metric_name,
            "n_days": len(x),
            "observed_delta": obs,
            **bs,
            "block_size": 10,
            "n_boot": 2000,
        })
boot = pd.DataFrame(rows)
boot.to_csv(TAB / "table_6_block_bootstrap_significance.csv", index=False)
print("[OK]", TAB / "table_6_block_bootstrap_significance.csv", boot.shape)

# --------------------------
# Table 7: failure cases
# --------------------------
if rl is not None and b is not None:
    x = b[["Date", "ret"]].rename(columns={"ret": "base_ret"}).merge(
        rl[["Date", "ret"] + (["gate"] if "gate" in rl.columns else [])].rename(columns={"ret": "rl_ret"}),
        on="Date",
        how="inner"
    )
    x["protection"] = x["rl_ret"] - x["base_ret"]
    x["protected"] = x["protection"] > 0
    fail = x.nsmallest(20, "rl_ret").copy()
    fail["failure_type"] = np.where(
        fail.get("gate", pd.Series([1]*len(fail))).to_numpy() < 1.0,
        "reduced exposure but residual loss remains",
        "full exposure on adverse day"
    )
    fail.to_csv(TAB / "table_7_failure_cases.csv", index=False)
    print("[OK]", TAB / "table_7_failure_cases.csv", fail.shape)

# --------------------------
# Table 8: module role audit
# --------------------------
audit = pd.DataFrame([
    {
        "module": "Temporal opportunity discovery",
        "representative_model": "V2L Frequency-aware Temporal",
        "intended_role": "Recall short-horizon candidate stocks from the full A-share universe.",
        "standalone_weakness": "Good candidate discovery but large drawdown as a final policy.",
        "evidence": "Single temporal expert underperforms V2X controllers in mean return and drawdown.",
        "why_kept": "Defines the candidate boundary for downstream action admission."
    },
    {
        "module": "Cross-sectional action admission",
        "representative_model": "V2X-B / V2H Factor Tree",
        "intended_role": "Choose tradable actions among candidates on each day.",
        "standalone_weakness": "Strong alpha but hidden concentration and large drawdown.",
        "evidence": "V2X-B mean_ret is high but maxDD is much worse than V2X-RL.",
        "why_kept": "Primary alpha source for stock selection."
    },
    {
        "module": "Geometry concentration diagnosis",
        "representative_model": "V2X-G",
        "intended_role": "Diagnose and regularize hidden concentration/crowding.",
        "standalone_weakness": "Improves breadth but sacrifices some raw return.",
        "evidence": "Effective holding number increases relative to V2X-B.",
        "why_kept": "Reveals the structural risk source behind nominal TopK portfolios."
    },
    {
        "module": "RL exposure overlay",
        "representative_model": "V2X-RL",
        "intended_role": "Control systematic market-level drawdown without reselecting stocks.",
        "standalone_weakness": "It cannot fix wrong stock selection, only rescale exposure.",
        "evidence": "Worst-day losses are reduced while best-day upside is mostly preserved.",
        "why_kept": "Final risk-aware controller."
    },
    {
        "module": "Market-body state encoding",
        "representative_model": "V2M / V2N",
        "intended_role": "Encode market-wide structure, diffusion, style rotation, and crowding.",
        "standalone_weakness": "Weak as a pure alpha engine.",
        "evidence": "Single market-body experts have lower return than V2X-B.",
        "why_kept": "Provides state context for risk and exposure control."
    },
    {
        "module": "Traditional anchor",
        "representative_model": "V2J RankIC Linear",
        "intended_role": "Provide interpretable linear factor calibration.",
        "standalone_weakness": "Weak return.",
        "evidence": "RankIC linear baseline is much weaker than neural/action controllers.",
        "why_kept": "Prevents the system from being an unanchored black-box stack."
    },
])
audit.to_csv(TAB / "table_8_module_role_audit.csv", index=False)
print("[OK]", TAB / "table_8_module_role_audit.csv", audit.shape)

# --------------------------
# Status
# --------------------------
status = []
for p in sorted(list(FIG.glob("*")) + list(TAB.glob("*"))):
    status.append({"file": str(p), "size_kb": round(p.stat().st_size / 1024, 2)})
status = pd.DataFrame(status)
status.to_csv(OUT / "kdd_defense_pack_status.csv", index=False)

print("\n===== STATUS =====")
print(status.to_string(index=False))

print("\n===== BOOTSTRAP HEAD =====")
print(boot.head(20).to_string(index=False))
