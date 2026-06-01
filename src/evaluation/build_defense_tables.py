from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = Path("outputs/dive_trader_v2/final_paper_tables_defense")
OUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

def read_csv_safe(path):
    path = Path(path)
    if not path.exists():
        print(f"[MISS] {path}")
        return None
    try:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        print(f"[ERR] {path}: {e}")
        return None

def max_drawdown(ret):
    r = pd.Series(ret).fillna(0).astype(float)
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def calc_daily_metrics(daily, ret_col=None, cost_bps=0):
    if daily is None or len(daily) == 0:
        return {}

    df = daily.copy()

    if ret_col is None:
        for c in ["day_net_ret", "ret", "net_ret", "mean_ret", "portfolio_ret", "strategy_ret"]:
            if c in df.columns:
                ret_col = c
                break

    if ret_col is None:
        raise ValueError(f"Cannot find return column in {df.columns.tolist()}")

    if "Date" in df.columns:
        df = df.sort_values("Date")

    r = df[ret_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n = int(len(r))
    mean_ret = float(r.mean())
    std_ret = float(r.std(ddof=0))
    sharpe_like = float(mean_ret / (std_ret + 1e-12))
    ann_ret_like = float(mean_ret * TRADING_DAYS / 5.0)
    ann_sharpe_like = float(sharpe_like * math.sqrt(TRADING_DAYS / 5.0))
    mdd = max_drawdown(r)
    pos = float((r > 0).mean())

    out = {
        "n_days": n,
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "ann_ret_like": ann_ret_like,
        "ann_sharpe_like": ann_sharpe_like,
        "max_drawdown_like": mdd,
        "positive_day_ratio": pos,
    }

    for c in ["day_turnover", "turnover_mean", "turnover", "gate", "exposure"]:
        if c in df.columns:
            out[c + "_mean"] = float(df[c].astype(float).mean())

    return out

def aggregate_ledger_to_daily(ledger, ret_col="day_net_ret"):
    """
    Ledger: one row per Date-Ticker.
    If day_net_ret already repeated per holding, take first per Date.
    """
    if ledger is None or len(ledger) == 0:
        return None
    df = ledger.copy()
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])

    if ret_col in df.columns:
        daily = df.groupby("Date", as_index=False).agg({
            ret_col: "first",
            "day_turnover": "first" if "day_turnover" in df.columns else "size",
            "gate": "first" if "gate" in df.columns else "size",
        })
        return daily.rename(columns={ret_col: "day_net_ret"})

    if "gross_contribution" in df.columns:
        rows = []
        prev_w = None
        for d, g in df.groupby("Date", sort=True):
            gross = float(g["gross_contribution"].sum())
            turnover = float(g["day_turnover"].iloc[0]) if "day_turnover" in g.columns else np.nan
            cost = float(g["day_cost"].iloc[0]) if "day_cost" in g.columns else 0.0
            gate = float(g["gate"].iloc[0]) if "gate" in g.columns else np.nan
            rows.append({
                "Date": d,
                "day_net_ret": gross - cost,
                "day_turnover": turnover,
                "gate": gate,
            })
        return pd.DataFrame(rows)

    return None

def concentration_from_ledger(ledger):
    if ledger is None or len(ledger) == 0 or "Date" not in ledger.columns or "weight" not in ledger.columns:
        return {}

    df = ledger.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    rows = []

    for d, g in df.groupby("Date", sort=True):
        w = g["weight"].astype(float).abs().values
        wsum = float(w.sum())
        if wsum <= 0:
            continue
        wn = w / wsum
        herf = float(np.sum(wn ** 2))
        eff = float(1.0 / (herf + 1e-12))
        sw = np.sort(wn)[::-1]
        rows.append({
            "Date": d,
            "nominal_n": int(len(w)),
            "weight_sum": wsum,
            "effective_n": eff,
            "top1_weight": float(sw[:1].sum()),
            "top3_weight": float(sw[:3].sum()),
            "top5_weight": float(sw[:5].sum()),
            "herfindahl": herf,
            "max_weight": float(sw[0]),
            "n_nonzero_1bp": int((wn > 0.0001).sum()),
            "n_nonzero_10bp": int((wn > 0.001).sum()),
            "n_nonzero_1pct": int((wn > 0.01).sum()),
        })

    daily_c = pd.DataFrame(rows)
    if daily_c.empty:
        return {}

    return {
        "effective_n_mean": float(daily_c["effective_n"].mean()),
        "effective_n_median": float(daily_c["effective_n"].median()),
        "effective_n_p10": float(daily_c["effective_n"].quantile(0.10)),
        "effective_n_p90": float(daily_c["effective_n"].quantile(0.90)),
        "top1_weight_mean": float(daily_c["top1_weight"].mean()),
        "top3_weight_mean": float(daily_c["top3_weight"].mean()),
        "top5_weight_mean": float(daily_c["top5_weight"].mean()),
        "herfindahl_mean": float(daily_c["herfindahl"].mean()),
        "max_weight_mean": float(daily_c["max_weight"].mean()),
        "max_weight_p95": float(daily_c["max_weight"].quantile(0.95)),
        "max_weight_max": float(daily_c["max_weight"].max()),
        "n_nonzero_1bp_mean": float(daily_c["n_nonzero_1bp"].mean()),
        "n_nonzero_10bp_mean": float(daily_c["n_nonzero_10bp"].mean()),
        "n_nonzero_1pct_mean": float(daily_c["n_nonzero_1pct"].mean()),
    }, daily_c

def pick_ret_col(df):
    if df is None:
        return None
    for c in ["day_net_ret", "ret", "net_ret", "strategy_ret", "portfolio_ret"]:
        if c in df.columns:
            return c
    return None

def daily_from_path(path, kind="daily"):
    df = read_csv_safe(path)
    if df is None:
        return None, None
    if kind == "ledger":
        daily = aggregate_ledger_to_daily(df)
        return daily, df
    return df, None

# =========================
# Source registry
# =========================
SOURCES = {
    "V2X-RL+ five-level exposure": {
        "group": "Ours",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/v2x_rl_exposure_formal/daily_test_five_020_040_060_080_100_seed2022.csv",
        "protocol": "test, seed=2022; 10-seed mean reported separately",
    },
    "V2X-B Cross Action Admission": {
        "group": "Ours",
        "kind": "ledger",
        "path": "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv",
        "protocol": "daily aggregated from ledger",
    },
    "V2X-G Cross-Geometry Overlay": {
        "group": "Ours diagnostic",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/v2x_g_cross_geometry_overlay/daily_test_v2x_g_cross_geometry_overlay.csv",
        "protocol": "geometry overlay diagnostic",
    },
    "V2R role-aware TopK30": {
        "group": "Intermediate controller",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/v2r_role_aware_router_topk30/daily_test_v2r_role_aware_router.csv",
        "protocol": "intermediate frozen test",
    },
    "V2L Frequency-aware Temporal": {
        "group": "Single expert",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/experts/v2l_frequency_expert/daily_test_v2l_frequency_expert.csv",
        "protocol": "temporal-only TopK",
    },
    "V2H Factor Tree": {
        "group": "Single expert",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/experts/factor_tree_backtest/daily_test_factor_tree.csv",
        "protocol": "cross-section-only TopK",
    },
    "V2M MASTER-style": {
        "group": "Single expert",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/experts/v2m_master_style/daily_test_v2m_master_style.csv",
        "protocol": "market-body-only TopK",
    },
    "V2N StockMixer-style": {
        "group": "Single expert",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/experts/v2n_stockmixer_style/daily_test_v2n_stockmixer_style.csv",
        "protocol": "market-body-only TopK",
    },
    "V2J RankIC Linear": {
        "group": "Traditional factor baseline",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/experts/v2j_rankic_linear/daily_test_v2j_rankic_linear.csv",
        "protocol": "linear factor TopK",
    },
    "Kronos-small official frozen": {
        "group": "Official foundation baseline",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/baselines/kronos_small_official_smoke20k/daily_test_kronos_small_official_frozen.csv",
        "protocol": "official frozen",
    },
    "TimeMoE-50M official frozen": {
        "group": "Official foundation baseline",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/baselines/timemoe_50m_official/daily_test_timemoe_50m_official_frozen.csv",
        "protocol": "official frozen",
    },
    "Chronos-Bolt official frozen": {
        "group": "Official foundation baseline",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/baselines/chronos_bolt_official/daily_test_chronos_bolt_official_frozen.csv",
        "protocol": "official frozen",
    },
    "Flat uniform aligned": {
        "group": "Flat ensemble",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/v2r_flat_vs_role_aligned/daily_test_flat_uniform_aligned.csv",
        "protocol": "flat uniform ensemble",
    },
    "Flat sharpe aligned": {
        "group": "Flat ensemble",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/v2r_flat_vs_role_aligned/daily_test_flat_sharpe_aligned.csv",
        "protocol": "flat sharpe ensemble",
    },
    "Flat objective aligned": {
        "group": "Flat ensemble",
        "kind": "daily",
        "path": "outputs/dive_trader_v2/v2r_flat_vs_role_aligned/daily_test_flat_objective_aligned.csv",
        "protocol": "flat objective ensemble",
    },
}

daily_cache = {}
ledger_cache = {}

# =========================
# 1. daily-aligned main comparison
# =========================
main_models = [
    "V2X-RL+ five-level exposure",
    "V2X-B Cross Action Admission",
    "V2X-G Cross-Geometry Overlay",
    "V2R role-aware TopK30",
    "V2L Frequency-aware Temporal",
    "V2H Factor Tree",
    "Kronos-small official frozen",
    "TimeMoE-50M official frozen",
    "Chronos-Bolt official frozen",
]

main_rows = []
for name in main_models:
    meta = SOURCES[name]
    daily, ledger = daily_from_path(meta["path"], meta["kind"])
    daily_cache[name] = daily
    ledger_cache[name] = ledger

    if daily is None or len(daily) == 0:
        row = {"model": name, "status": "missing", **meta}
    else:
        ret_col = pick_ret_col(daily)
        m = calc_daily_metrics(daily, ret_col=ret_col)
        row = {
            "group": meta["group"],
            "model": name,
            "protocol": meta["protocol"],
            "source_path": meta["path"],
            "status": "ready",
            "ret_col": ret_col,
            **m,
        }
    main_rows.append(row)

main_df = pd.DataFrame(main_rows)
main_df.to_csv(OUT / "table_1_daily_aligned_main_comparison.csv", index=False)

# =========================
# 2. China constraints ledger-consistent
# Prefer existing rebuilt table if available
# =========================
constraint_paths = [
    "outputs/dive_trader_v2/final_paper_tables_v4/final_table_4c4d_real_constraints_ledger_consistent.csv",
    "outputs/dive_trader_v2/final_paper_tables_v3/final_table_4c4d_real_portfolio_constraints_rebuilt.csv",
    "outputs/dive_trader_v2/final_paper_tables_v2/final_table_4_china_constraints_real_testonly.csv",
]
constraint_df = None
for p in constraint_paths:
    df = read_csv_safe(p)
    if df is not None:
        constraint_df = df
        print("[USE constraints]", p)
        break

if constraint_df is None:
    constraint_df = pd.DataFrame([{
        "status": "missing",
        "note": "Run the ledger-consistent China constraints builder first."
    }])

constraint_df.to_csv(OUT / "table_2_china_constraints_ledger_consistent.csv", index=False)

# =========================
# 3. concentration table
# =========================
conc_rows = []
for name, meta in SOURCES.items():
    daily = daily_cache.get(name)
    ledger = ledger_cache.get(name)

    if daily is None and ledger is None:
        daily, ledger = daily_from_path(meta["path"], meta["kind"])

    row = {
        "group": meta["group"],
        "model": name,
        "protocol": meta["protocol"],
        "source_path": meta["path"],
    }

    if ledger is not None and "weight" in ledger.columns:
        conc, daily_c = concentration_from_ledger(ledger)
        row.update({"status": "ready_from_ledger", **conc})
        if isinstance(daily_c, pd.DataFrame):
            safe = name.replace("/", "_").replace(" ", "_").replace("+", "plus")
            daily_c.to_csv(OUT / f"daily_concentration_{safe}.csv", index=False)
    elif daily is not None and all(c in daily.columns for c in ["effective_n", "top1_weight"]):
        row.update({
            "status": "ready_from_daily",
            "effective_n_mean": float(daily["effective_n"].mean()),
            "top1_weight_mean": float(daily["top1_weight"].mean()),
        })
    else:
        row.update({"status": "missing_weight_or_ledger"})

    if daily is not None and len(daily) > 0:
        ret_col = pick_ret_col(daily)
        try:
            row.update(calc_daily_metrics(daily, ret_col))
        except Exception:
            pass

    conc_rows.append(row)

conc_df = pd.DataFrame(conc_rows)
conc_df.to_csv(OUT / "table_3_concentration_effective_holding.csv", index=False)

# =========================
# 4. RL policy behavior
# =========================
rl_name = "V2X-RL+ five-level exposure"
rl_daily = daily_cache.get(rl_name)
if rl_daily is None:
    rl_daily, _ = daily_from_path(SOURCES[rl_name]["path"], SOURCES[rl_name]["kind"])

base_name = "V2X-B Cross Action Admission"
base_daily = daily_cache.get(base_name)
if base_daily is None:
    base_daily, _ = daily_from_path(SOURCES[base_name]["path"], SOURCES[base_name]["kind"])

policy_rows = []
worst_best_rows = []

if rl_daily is not None and base_daily is not None and "Date" in rl_daily.columns and "Date" in base_daily.columns:
    rld = rl_daily.copy()
    bd = base_daily.copy()
    rld["Date"] = pd.to_datetime(rld["Date"])
    bd["Date"] = pd.to_datetime(bd["Date"])

    rl_ret = pick_ret_col(rld)
    base_ret = pick_ret_col(bd)

    # Detect gate/exposure column
    gate_col = None
    for c in ["gate", "exposure", "action", "rl_gate"]:
        if c in rld.columns:
            gate_col = c
            break

    merged = bd[["Date", base_ret]].rename(columns={base_ret: "base_ret"}).merge(
        rld[["Date", rl_ret] + ([gate_col] if gate_col else [])].rename(columns={rl_ret: "rl_ret", gate_col: "gate"} if gate_col else {rl_ret: "rl_ret"}),
        on="Date",
        how="inner"
    )

    if "gate" not in merged.columns:
        # infer gate as rl_ret / base_ret where stable
        merged["gate"] = np.where(merged["base_ret"].abs() > 1e-8, merged["rl_ret"] / merged["base_ret"], np.nan)
        merged["gate"] = merged["gate"].replace([np.inf, -np.inf], np.nan).clip(0, 2)

    gate_counts = merged["gate"].round(4).value_counts(dropna=False).sort_index()
    for g, cnt in gate_counts.items():
        policy_rows.append({
            "section": "action_frequency",
            "gate": g,
            "n_days": int(cnt),
            "ratio": float(cnt / len(merged)),
        })

    policy_rows.append({
        "section": "summary",
        "n_days": int(len(merged)),
        "gate_mean": float(merged["gate"].mean()),
        "gate_min": float(merged["gate"].min()),
        "gate_max": float(merged["gate"].max()),
        "base_mean_ret": float(merged["base_ret"].mean()),
        "rl_mean_ret": float(merged["rl_ret"].mean()),
        "base_maxdd": max_drawdown(merged["base_ret"]),
        "rl_maxdd": max_drawdown(merged["rl_ret"]),
    })

    worst = merged.sort_values("base_ret").head(20).copy()
    best = merged.sort_values("base_ret", ascending=False).head(20).copy()

    for tag, sub in [("worst20_base_days", worst), ("best20_base_days", best)]:
        worst_best_rows.append({
            "section": tag,
            "n_days": int(len(sub)),
            "base_ret_mean": float(sub["base_ret"].mean()),
            "rl_ret_mean": float(sub["rl_ret"].mean()),
            "gate_mean": float(sub["gate"].mean()),
            "reduced_exposure_ratio": float((sub["gate"] < 0.999).mean()),
            "full_exposure_ratio": float((sub["gate"] >= 0.999).mean()),
        })

    merged.to_csv(OUT / "daily_rl_policy_behavior_detail.csv", index=False)
else:
    policy_rows.append({"status": "missing_rl_or_base_daily"})

policy_df = pd.DataFrame(policy_rows)
policy_df.to_csv(OUT / "table_4a_rl_policy_action_frequency.csv", index=False)

wb_df = pd.DataFrame(worst_best_rows)
wb_df.to_csv(OUT / "table_4b_rl_worst_best_day_behavior.csv", index=False)

# =========================
# 5. role-separation ablation
# =========================
role_models = [
    "V2X-RL+ five-level exposure",
    "V2X-B Cross Action Admission",
    "V2X-G Cross-Geometry Overlay",
    "V2R role-aware TopK30",
    "V2L Frequency-aware Temporal",
    "V2H Factor Tree",
    "V2M MASTER-style",
    "V2N StockMixer-style",
    "V2J RankIC Linear",
    "Flat uniform aligned",
    "Flat sharpe aligned",
    "Flat objective aligned",
]

role_notes = {
    "V2X-RL+ five-level exposure": "RL exposure overlay: controls systematic drawdown without reselecting stocks.",
    "V2X-B Cross Action Admission": "Temporal candidate + cross-sectional action admission.",
    "V2X-G Cross-Geometry Overlay": "Geometry overlay: concentration/crowding regularization after cross-sectional scoring.",
    "V2R role-aware TopK30": "Intermediate role-aware router.",
    "V2L Frequency-aware Temporal": "Temporal-only opportunity discovery.",
    "V2H Factor Tree": "Cross-section-only daily action scorer.",
    "V2M MASTER-style": "Market-body structural encoder, MASTER-style.",
    "V2N StockMixer-style": "Market-body structural encoder, StockMixer-style.",
    "V2J RankIC Linear": "Traditional interpretable alpha anchor.",
    "Flat uniform aligned": "Naive flat ensemble baseline.",
    "Flat sharpe aligned": "Flat ensemble weighted by Sharpe.",
    "Flat objective aligned": "Flat ensemble weighted by validation objective.",
}

role_rows = []
for name in role_models:
    meta = SOURCES[name]
    daily = daily_cache.get(name)
    ledger = ledger_cache.get(name)
    if daily is None and ledger is None:
        daily, ledger = daily_from_path(meta["path"], meta["kind"])
    row = {
        "group": meta["group"],
        "model": name,
        "role_interpretation": role_notes.get(name, ""),
        "protocol": meta["protocol"],
        "source_path": meta["path"],
    }
    if daily is None or len(daily) == 0:
        row["status"] = "missing"
    else:
        ret_col = pick_ret_col(daily)
        row.update({"status": "ready", "ret_col": ret_col, **calc_daily_metrics(daily, ret_col)})
    role_rows.append(row)

role_df = pd.DataFrame(role_rows)
role_df.to_csv(OUT / "table_5_role_separation_ablation.csv", index=False)

# =========================
# Final status
# =========================
status = []
for p in sorted(OUT.glob("*.csv")):
    try:
        n = len(pd.read_csv(p))
    except Exception:
        n = -1
    status.append({"file": str(p), "rows": n, "size_kb": round(p.stat().st_size / 1024, 2)})

status_df = pd.DataFrame(status)
status_df.to_csv(OUT / "table_status.csv", index=False)

print("\n===== SAVED FINAL DEFENSE TABLES =====")
print(status_df.to_string(index=False))

print("\n===== MAIN COMPARISON PREVIEW =====")
cols = ["group", "model", "status", "n_days", "mean_ret", "ann_sharpe_like", "max_drawdown_like", "positive_day_ratio"]
print(main_df[[c for c in cols if c in main_df.columns]].to_string(index=False))

print("\n===== ROLE ABLATION PREVIEW =====")
print(role_df[[c for c in cols if c in role_df.columns]].to_string(index=False))
