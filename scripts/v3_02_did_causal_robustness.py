#!/usr/bin/env python3
"""
v3_02_did_causal_robustness.py
==============================
Step 2 of Ocean Engineering (OE-D-26-13535) Revision:
Quasi-experimental causal robustness and parallel-trends audit of the
2014 EU Common Fisheries Policy (Reg. 1380/2013) entry-exit reform.

Addresses Reviewer 2 Comment (3):
  1. Evaluates 8 nested specifications: Naive 2x2 DiD vs. Country+Year TWFE vs.
     Full Structural Controls (+ gear subtype, size, volumetric fullness) vs.
     Verified Non-Imputed Build Years (N=32,034) vs. 2006-2024 Window vs.
     Hull Material FE vs. Country-Specific Linear Trends vs. Volumetric Fullness Outcome.
  2. Estimates dynamic Event-Study coefficients (2000-2023, 2-yr bins, ref=2012-2013)
     pooled and stratified by gear (Trawl vs. Passive), with formal joint pre-trend F-tests.
  3. Computes HC1, GSA-clustered (G=30), Country-clustered (G=24), and Rademacher
     Wild-Cluster Bootstrap p-values.
  4. Executes in-time placebo falsification tests (2006, 2008, 2010) and Leave-One-Country-Out checks.
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "eii_constructed.parquet")
RESULTS_V3 = os.path.join(BASE_DIR, "Version_2_Optimization", "results_v3")
FIGURES_V3 = os.path.join(BASE_DIR, "Version_2_Optimization", "figures_v3")
os.makedirs(RESULTS_V3, exist_ok=True)
os.makedirs(FIGURES_V3, exist_ok=True)

EU_COUNTRIES = {
    "Italy", "Greece", "Spain", "Croatia", "France",
    "Cyprus", "Malta", "Slovenia", "Bulgaria", "Romania"
}


def wild_cluster_bootstrap_p(df_in: pd.DataFrame, formula: str, target_var: str, cluster_col: str, n_boot: int = 999, seed: int = 42) -> float:
    """
    Fast Rademacher wild-cluster bootstrap p-value for a single DiD coefficient
    using restricted residual resampling (Webb/Cameron-Gelbach-Miller style).
    """
    rng = np.random.default_rng(seed)
    mod_full = smf.ols(formula, data=df_in).fit(cov_type="cluster", cov_kwds={"groups": df_in[cluster_col]})
    t_obs = abs(float(mod_full.tvalues[target_var]))

    # Restricted model under H0: target_var = 0
    rhs_terms = [t.strip() for t in formula.split("~")[1].split("+") if t.strip() != target_var]
    f_rest = formula.split("~")[0].strip() + " ~ " + " + ".join(rhs_terms)
    mod_rest = smf.ols(f_rest, data=df_in).fit()
    y_hat = mod_rest.fittedvalues.values
    resid = mod_rest.resid.values

    # Build design matrix once for fast OLS projection
    import patsy
    y_orig, X = patsy.dmatrices(formula, data=df_in, return_type="dataframe")
    col_idx = list(X.columns).index(target_var)
    X_np = X.values
    XtX_inv = np.linalg.pinv(X_np.T @ X_np)
    proj_row = XtX_inv @ X_np.T

    clusters = df_in[cluster_col].values
    uniq_cl, inv_idx = np.unique(clusters, return_inverse=True)
    n_cl = len(uniq_cl)

    # Compute clustered SE denominator matrix per cluster once
    exceed_count = 0
    for _ in range(n_boot):
        weights = rng.choice([-1.0, 1.0], size=n_cl)[inv_idx]
        y_star = y_hat + resid * weights
        beta_star = proj_row @ y_star
        e_star = y_star - X_np @ beta_star
        # Cluster-robust meat matrix
        meat = np.zeros((X_np.shape[1], X_np.shape[1]))
        for c_id in range(n_cl):
            m = inv_idx == c_id
            sc = X_np[m].T @ e_star[m]
            meat += np.outer(sc, sc)
        dof_corr = (n_cl / max(1, n_cl - 1)) * ((len(y_star) - 1) / max(1, len(y_star) - X_np.shape[1]))
        cov_star = dof_corr * (XtX_inv @ meat @ XtX_inv)
        se_star = np.sqrt(max(1e-12, cov_star[col_idx, col_idx]))
        if abs(beta_star[col_idx] / se_star) >= t_obs:
            exceed_count += 1
    return float((exceed_count + 1) / (n_boot + 1))


def fit_did_spec(name: str, df_sub: pd.DataFrame, formula: str, target_var: str = "did_int", run_wcb: bool = True) -> dict:
    m_hc1 = smf.ols(formula, data=df_sub).fit(cov_type="HC1")
    m_gsa = smf.ols(formula, data=df_sub).fit(cov_type="cluster", cov_kwds={"groups": df_sub["gsa_1"]})
    m_cnt = smf.ols(formula, data=df_sub).fit(cov_type="cluster", cov_kwds={"groups": df_sub["country"]})
    wcb_p = wild_cluster_bootstrap_p(df_sub, formula, target_var, "country", n_boot=499) if run_wcb else np.nan
    coef = float(m_hc1.params[target_var])
    return {
        "Specification": name,
        "N": int(m_hc1.nobs),
        "Coef": round(coef, 4),
        "HC1_SE": round(float(m_hc1.bse[target_var]), 4),
        "HC1_p": round(float(m_hc1.pvalues[target_var]), 4),
        "GSA_Cluster_SE": round(float(m_gsa.bse[target_var]), 4),
        "GSA_Cluster_p": round(float(m_gsa.pvalues[target_var]), 4),
        "Country_Cluster_SE": round(float(m_cnt.bse[target_var]), 4),
        "Country_Cluster_p": round(float(m_cnt.pvalues[target_var]), 4),
        "Wild_Cluster_Boot_p": round(wcb_p, 4) if not np.isnan(wcb_p) else np.nan,
        "R2": round(float(m_hc1.rsquared), 4),
    }


def run_event_study(df_in: pd.DataFrame, label: str) -> tuple[pd.DataFrame, float, float]:
    es = df_in[(df_in["yr_int"] >= 2000) & (df_in["yr_int"] <= 2023)].copy()
    es["bin"] = (es["yr_int"] - 2014) // 2  # -1 is 2012-2013 reference
    bins = sorted(es["bin"].unique())
    terms = []
    for b in bins:
        if b == -1:
            continue
        col = f"eu_b_{b if b >= 0 else 'm' + str(abs(b))}"
        es[col] = ((es["bin"] == b) & (es["is_eu"] == 1)).astype(int)
        terms.append(col)

    extra_cov = " + C(gear_group)" if es["gear_group"].nunique() > 1 else ""
    f_es = "residual_eii ~ " + " + ".join(terms) + " + C(country) + C(yr_int) + C(size_class) + ln_fullness" + extra_cov
    mod = smf.ols(f_es, data=es).fit(cov_type="HC1")

    pre_all = [t for t in terms if "m" in t]
    pre_06 = [t for t in terms if t in ("eu_b_m4", "eu_b_m3", "eu_b_m2")]
    p_pre_all = float(mod.f_test(" = ".join(pre_all) + " = 0").pvalue) if len(pre_all) >= 2 else np.nan
    p_pre_06 = float(mod.f_test(" = ".join(pre_06) + " = 0").pvalue) if len(pre_06) >= 2 else np.nan

    rows = []
    for b in bins:
        yr_lo = 2014 + 2 * b
        if b == -1:
            rows.append({
                "Segment": label, "Bin": int(b), "Period": "2012-2013",
                "Mid_Year": 2012.5, "Coef": 0.0, "SE": 0.0, "CI_Low": 0.0, "CI_High": 0.0,
                "p_value": 1.0, "PreTrend_2000_2011_p": round(p_pre_all, 4), "PreTrend_2006_2011_p": round(p_pre_06, 4),
            })
        else:
            col = f"eu_b_{b if b >= 0 else 'm' + str(abs(b))}"
            c = float(mod.params[col])
            se = float(mod.bse[col])
            rows.append({
                "Segment": label, "Bin": int(b), "Period": f"{yr_lo}-{yr_lo+1}",
                "Mid_Year": yr_lo + 0.5, "Coef": round(c, 4), "SE": round(se, 4),
                "CI_Low": round(c - 1.96 * se, 4), "CI_High": round(c + 1.96 * se, 4),
                "p_value": round(float(mod.pvalues[col]), 4),
                "PreTrend_2000_2011_p": round(p_pre_all, 4), "PreTrend_2006_2011_p": round(p_pre_06, 4),
            })
    return pd.DataFrame(rows), p_pre_all, p_pre_06


def main():
    print("=" * 72)
    print(" v3_02_did_causal_robustness.py — DiD Parallel Trends & TWFE Audit")
    print("=" * 72)

    df = pd.read_parquet(DATA_PATH)
    act = df[(df["engine_power_kw"] > 0) & (df["tonnage_gt"] > 0) & (df["loa_m"] > 0)].copy()
    act["ln_eii"] = np.log(act["eii_kw_per_gt"])
    act["ln_fullness"] = np.log(act["tonnage_gt"] / (act["loa_m"] ** 3))

    did = act[(act["construction_year"].notna()) & (act["construction_year"] >= 1990) & (act["construction_year"] <= 2025)].copy()
    did["yr_int"] = did["construction_year"].astype(int)
    did["is_eu"] = did["country"].isin(EU_COUNTRIES).astype(int)
    did["post_2014"] = (did["yr_int"] >= 2014).astype(int)
    did["did_int"] = did["is_eu"] * did["post_2014"]
    did["yr_centered"] = (did["yr_int"] - 2014) / 10.0

    # Collapse rare gear1 for clean FE controls
    g1_cnt = did["gear1"].fillna("NK").value_counts()
    did["gear1_c"] = did["gear1"].fillna("NK").where(did["gear1"].fillna("NK").isin(set(g1_cnt[g1_cnt >= 30].index)), "OTHER")

    did_ni = did[did["construction_year_imputed"] == 0].copy()
    did_win = did_ni[(did_ni["yr_int"] >= 2006) & (did_ni["yr_int"] <= 2024)].copy()
    did_mat = did_ni[did_ni["hull_material_raw"].isin(["Wood", "Fiberglass/Plastic", "Metal"])].copy()

    print(f"DiD full sample (1990-2025): N={len(did):,} | Non-imputed build years: N={len(did_ni):,}")

    # ------------------------------------------------------------------
    # 1. 8-Model Specification Battery
    # ------------------------------------------------------------------
    spec_rows = [
        fit_did_spec("1. Naive 2x2 DiD (v2 Baseline)", did, "residual_eii ~ is_eu + post_2014 + did_int", run_wcb=False),
        fit_did_spec("2. Country + Year TWFE (All 1990-2025)", did, "residual_eii ~ did_int + C(country) + C(yr_int)"),
        fit_did_spec("3. TWFE + Gear1 + Size + Fullness (All)", did, "residual_eii ~ did_int + C(country) + C(yr_int) + C(gear1_c) + C(size_class) + ln_fullness"),
        fit_did_spec("4. TWFE + Controls (Non-Imputed N=32,034)", did_ni, "residual_eii ~ did_int + C(country) + C(yr_int) + C(gear_group) + C(size_class) + ln_fullness"),
        fit_did_spec("5. TWFE + Controls (2006-2024 Window)", did_win, "residual_eii ~ did_int + C(country) + C(yr_int) + C(gear_group) + C(size_class) + ln_fullness"),
        fit_did_spec("6. TWFE + Hull Material FE (Subsample)", did_mat, "residual_eii ~ did_int + C(country) + C(yr_int) + C(gear_group) + C(size_class) + ln_fullness + C(hull_material_raw)"),
        fit_did_spec("7. TWFE + Country Linear Trends", did_ni, "residual_eii ~ did_int + C(country) + C(yr_int) + C(country):yr_centered + C(gear_group) + C(size_class) + ln_fullness", run_wcb=False),
        fit_did_spec("8. Alt Outcome: Volumetric Fullness ln(GT/LOA^3)", did_ni, "ln_fullness ~ did_int + C(country) + C(yr_int) + C(gear_group) + C(size_class)"),
    ]

    # Add Gear-Stratified TWFE (Non-Imputed)
    for g in ["trawl", "purse_seine", "passive", "other"]:
        sub_g = did_ni[did_ni["gear_group"] == g].copy()
        spec_rows.append(
            fit_did_spec(f"9. Gear TWFE: {g.title()} (Non-Imputed)", sub_g, "residual_eii ~ did_int + C(country) + C(yr_int) + C(size_class) + ln_fullness")
        )

    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(os.path.join(RESULTS_V3, "did_twfe_specifications.csv"), index=False)
    print("\nDiD Specification Battery:")
    print(spec_df[["Specification", "N", "Coef", "HC1_p", "GSA_Cluster_p", "Country_Cluster_p", "Wild_Cluster_Boot_p"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Dynamic Event Study (Pooled, Trawl, Passive on Non-Imputed)
    # ------------------------------------------------------------------
    es_pooled, p1_all, p1_06 = run_event_study(did_ni, "All Fleets (Pooled)")
    es_trawl, p2_all, p2_06 = run_event_study(did_ni[did_ni["gear_group"] == "trawl"], "Demersal & Pelagic Trawl")
    es_passive, p3_all, p3_06 = run_event_study(did_ni[did_ni["gear_group"] == "passive"], "Passive Coastal Gears")

    es_df = pd.concat([es_pooled, es_trawl, es_passive], ignore_index=True)
    es_df.to_csv(os.path.join(RESULTS_V3, "did_event_study_coefficients.csv"), index=False)
    print(f"\nEvent-Study Pre-Trend Joint F-Test p-values:")
    print(f"  Pooled  | 2000-2011 p={p1_all:.4f} | 2006-2011 p={p1_06:.4f}")
    print(f"  Trawl   | 2000-2011 p={p2_all:.4f} | 2006-2011 p={p2_06:.4f}")
    print(f"  Passive | 2000-2011 p={p3_all:.4f} | 2006-2011 p={p3_06:.4f}")

    # ------------------------------------------------------------------
    # 3. In-Time Placebos (Pre-2014 data only) & Leave-One-Country-Out
    # ------------------------------------------------------------------
    rob_rows = []
    pre_only = did_ni[did_ni["yr_int"] < 2014].copy()
    for p_yr in [2006, 2008, 2010]:
        pre_only["placebo_int"] = (pre_only["is_eu"] * (pre_only["yr_int"] >= p_yr)).astype(int)
        m_p = smf.ols("residual_eii ~ placebo_int + C(country) + C(yr_int) + C(gear_group) + C(size_class) + ln_fullness", data=pre_only).fit(cov_type="HC1")
        m_pg = smf.ols("residual_eii ~ placebo_int + C(country) + C(yr_int) + C(gear_group) + C(size_class) + ln_fullness", data=pre_only).fit(cov_type="cluster", cov_kwds={"groups": pre_only["gsa_1"]})
        rob_rows.append({
            "Check_Type": "In-Time Placebo",
            "Label": f"Pseudo-Reform {p_yr} (Pre-2014)",
            "N": int(m_p.nobs),
            "Coef": round(float(m_p.params["placebo_int"]), 4),
            "HC1_SE": round(float(m_p.bse["placebo_int"]), 4),
            "HC1_p": round(float(m_p.pvalues["placebo_int"]), 4),
            "GSA_p": round(float(m_pg.pvalues["placebo_int"]), 4),
        })

    for drop_c in ["Türkiye", "Egypt", "Greece", "Italy", "Spain", "Croatia", "Morocco"]:
        sub_loco = did_ni[did_ni["country"] != drop_c]
        m_l = smf.ols("residual_eii ~ did_int + C(country) + C(yr_int) + C(gear_group) + C(size_class) + ln_fullness", data=sub_loco).fit(cov_type="HC1")
        m_lg = smf.ols("residual_eii ~ did_int + C(country) + C(yr_int) + C(gear_group) + C(size_class) + ln_fullness", data=sub_loco).fit(cov_type="cluster", cov_kwds={"groups": sub_loco["gsa_1"]})
        rob_rows.append({
            "Check_Type": "Leave-One-Country-Out",
            "Label": f"Excl. {drop_c}",
            "N": int(m_l.nobs),
            "Coef": round(float(m_l.params["did_int"]), 4),
            "HC1_SE": round(float(m_l.bse["did_int"]), 4),
            "HC1_p": round(float(m_l.pvalues["did_int"]), 4),
            "GSA_p": round(float(m_lg.pvalues["did_int"]), 4),
        })

    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(os.path.join(RESULTS_V3, "did_placebo_and_loco.csv"), index=False)
    print("\nPlacebo & Leave-One-Country-Out Robustness:")
    print(rob_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Publication-Grade Figure 3: Event-Study & Specification Robustness
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=300)

    # Panel A: Event-Study Dynamic Coefficients
    ax = axes[0]
    styles = [
        ("All Fleets (Pooled)", "#1f77b4", "o", 0.0),
        ("Demersal & Pelagic Trawl", "#d62728", "s", -0.25),
        ("Passive Coastal Gears", "#2ca02c", "^", 0.25),
    ]
    for seg_name, col, mk, offset in styles:
        s_df = es_df[es_df["Segment"] == seg_name].sort_values("Mid_Year")
        ax.errorbar(
            s_df["Mid_Year"] + offset, s_df["Coef"],
            yerr=1.96 * s_df["SE"], fmt=mk + "-", color=col,
            linewidth=1.6, markersize=5.2, capsize=3, alpha=0.9, label=seg_name
        )
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.axvline(2013.5, color="#8b0000", linestyle=":", linewidth=1.5, label="2014 EU CFP Reform")
    ax.axvspan(2005.5, 2013.5, color="#eef5fb", alpha=0.5, zorder=0)
    ax.set_xlabel("Vessel Construction Cohort (2-Year Bins; Ref = 2012–2013)")
    ax.set_ylabel(r"TWFE Event-Study Effect on $\Delta EII_i$ (log kW residual)")
    ax.set_title("A. Dynamic Event Study & Pre-Trends (Non-Imputed, N=29,886)")
    ax.text(0.02, 0.04,
            f"Pre-trend F-test (2006–2011):\n• Pooled p = {p1_06:.3f}\n• Trawl p = {p2_06:.3f}\n• Passive p = {p3_06:.3f}",
            transform=ax.transAxes, fontsize=7.8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.22), frameon=True, framealpha=0.9, fontsize=7.8)
    ax.grid(alpha=0.25)

    # Panel B: Specification Forest Plot
    ax = axes[1]
    keep_specs = [
        "1. Naive 2x2 DiD (v2 Baseline)",
        "2. Country + Year TWFE (All 1990-2025)",
        "3. TWFE + Gear1 + Size + Fullness (All)",
        "4. TWFE + Controls (Non-Imputed N=32,034)",
        "5. TWFE + Controls (2006-2024 Window)",
        "6. TWFE + Hull Material FE (Subsample)",
        "7. TWFE + Country Linear Trends",
        "8. Alt Outcome: Volumetric Fullness ln(GT/LOA^3)",
        "9. Gear TWFE: Trawl (Non-Imputed)",
        "9. Gear TWFE: Passive (Non-Imputed)",
    ]
    sub_plot = spec_df[spec_df["Specification"].isin(keep_specs)].iloc[::-1].reset_index(drop=True)
    y_i = np.arange(len(sub_plot))
    cols_b = ["#b22222" if "Naive" in s else ("#2ca02c" if "Passive" in s else ("#d62728" if "Trawl" in s else "#1f77b4")) for s in sub_plot["Specification"]]
    for idx_r, r in sub_plot.iterrows():
        ax.errorbar(r.Coef, idx_r, xerr=1.96 * r.HC1_SE, fmt="o", color=cols_b[idx_r], capsize=3, markersize=5.5)
        ax.text(r.Coef + 1.96 * r.HC1_SE + 0.012, idx_r, f"{r.Coef:+.3f} (GSA p={r.GSA_Cluster_p:.2f})", va="center", fontsize=7.3)
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.1)
    ax.set_yticks(y_i)
    ax.set_yticklabels([s.split(". ", 1)[1] for s in sub_plot["Specification"]], fontsize=8.0)
    ax.set_xlabel(r"Estimated $\mathrm{EU} \times \mathrm{Post2014}$ Coefficient ($\pm 95\%$ HC1 CI)")
    ax.set_xlim(-0.42, 0.42)
    ax.set_title("B. DiD Specification & Confounding Sensitivity")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_V3, "Fig3_DiD_Event_Study_and_Robustness.png")
    pdf_path = os.path.join(FIGURES_V3, "Fig3_DiD_Event_Study_and_Robustness.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 3: {png_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
