#!/usr/bin/env python3
"""
v3_06_ablation_study.py
=======================
Step 6 of Ocean Engineering (OE-D-26-13535) Revision:
Formal 5-Module Ablation Study quantifying the marginal contribution of:
  1. Hydrodynamic Power Scaling (vs. naive linear EII = kW/GT benchmark)
  2. Unsupervised GMM ML Typologies (vs. coarse GFCM administrative labels)
  3. Optimization Structural Constraints & Engineering Levers:
       - National Subsidiarity Cap (Omega_c <= 20%)
       - Allometric Engine Derating lever
       - Trawl-to-Passive Gear Conversion lever
       - Selective Decommissioning lever
  4. Dynamic Endogenous ABM & Rebound Controls (vs. static 10-yr extrapolation
     and vs. dynamic ABM without carbon pricing / effort restraint)

Addresses Reviewer 2 Comment (7).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_04_exact_momilp_and_benchmarks import solve_exact_milp
from v3_05_abm_calibrated_sobol import prepare_agent_arrays, simulate_abm

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_V3 = os.path.join(BASE_DIR, "Version_2_Optimization", "results_v3")
FIGURES_V3 = os.path.join(BASE_DIR, "Version_2_Optimization", "figures_v3")
os.makedirs(RESULTS_V3, exist_ok=True)
os.makedirs(FIGURES_V3, exist_ok=True)


def run_ablation_suite():
    print("=" * 72)
    print(" v3_06_ablation_study.py — 5-Module Ablation Experiments")
    print("=" * 72)

    df = pd.read_parquet(os.path.join(RESULTS_V3, "fleet_v3_master.parquet"))
    n = len(df)
    e0 = df["co2_tier2_t"].values
    gt0 = df["tonnage_gt"].values
    kw0 = df["engine_power_kw"].values
    age = df["age_clean"].values
    loa = df["loa_m"].values
    is_trawl = (df["gear_group"].values == "trawl")
    is_small = (loa < 12.0)

    # Full hydrodynamic parameters
    kw_safe_hydro = df["kw_hydro_v3"].values * 1.25
    ex_kw_hydro = np.maximum(0.0, kw0 - kw_safe_hydro)
    elig_der_hydro = ex_kw_hydro > 15.0

    # Linear EII benchmark parameters (Ablation 1: w/o hydrodynamics -> linear median EII = 10.1 kW/GT * 1.25)
    med_linear_eii = float(np.median(df["eii_kw_per_gt"].values))
    kw_safe_linear = gt0 * med_linear_eii * 1.25
    ex_kw_linear = np.maximum(0.0, kw0 - kw_safe_linear)
    elig_der_linear = ex_kw_linear > 15.0

    # Quantify artisanal (<12m) false-positive overpowering rate under linear EII vs. hydrodynamics
    fp_small_linear = float(np.mean(elig_der_linear[is_small]) * 100.0)
    fp_small_hydro = float(np.mean(elig_der_hydro[is_small]) * 100.0)

    def make_params(der_elig, ex_kw, allow_der=True, allow_dec=True, allow_con=True):
        der_pct = ex_kw / np.maximum(1.0, kw0)
        return {
            "e0": e0, "gt0": gt0,
            "elig_derate": der_elig if allow_der else np.zeros(n, dtype=bool),
            "abate_derate": np.where(der_elig & allow_der, e0 * 0.90 * der_pct, 0.0),
            "cost_derate": np.where(der_elig & allow_der, ex_kw * 60.0, 1e12),
            "abate_decom": e0.copy() if allow_dec else np.zeros(n),
            "cost_decom": (gt0 * np.maximum(500.0, 2200.0 * np.exp(-0.035 * age))) if allow_dec else np.full(n, 1e12),
            "elig_conv": is_trawl if allow_con else np.zeros(n, dtype=bool),
            "abate_conv": np.where(is_trawl & allow_con, e0 * 0.55, 0.0),
            "cost_conv": np.where(is_trawl & allow_con, 45000.0, 1e12),
        }

    # ------------------------------------------------------------------
    # Part A: Static €300M Optimization Module Ablations (Modules 1, 2, 3)
    # ------------------------------------------------------------------
    # Full baseline
    p_full = make_params(elig_der_hydro, ex_kw_hydro, True, True, True)
    r_full = solve_exact_milp(df, p_full, 300e6, subsidiarity=True)

    # Ablation 1: w/o Hydrodynamic Scaling (linear kW/GT derating target)
    p_no_hydro = make_params(elig_der_linear, ex_kw_linear, True, True, True)
    r_no_hydro = solve_exact_milp(df, p_no_hydro, 300e6, subsidiarity=True)
    # Recompute true physical CO2 abatement of linear-EII derating (cannot derate below physical safe minimum on artisanal hulls without loss, evaluate nominal vs misallocated spend on <12m)
    act_lin = r_no_hydro["actions"]
    misalloc_spend_lin_meur = float(np.sum(np.where((act_lin == 1) & is_small & (~elig_der_hydro), p_no_hydro["cost_derate"], 0.0)) / 1e6)
    misalloc_vessels_lin = int(np.sum((act_lin == 1) & is_small & (~elig_der_hydro)))

    # Ablation 2: w/o ML Typologies (remove GMM Typology-4 Artisanal protection, replace with naive gear=="passive" OR unrestricted within country)
    df_no_ml = df.copy()
    df_no_ml["typology_name"] = "Unclassified"  # disables GMM artisanal shield
    r_no_ml = solve_exact_milp(df_no_ml, p_full, 300e6, subsidiarity=True)
    art_decom_full = int(np.sum((r_full["actions"] == 2) & df["typology_name"].str.contains("Artisanal").values))
    art_decom_noml = int(np.sum((r_no_ml["actions"] == 2) & df["typology_name"].str.contains("Artisanal").values))

    # Ablation 3a: w/o Subsidiarity Cap
    r_no_sub = solve_exact_milp(df, p_full, 300e6, subsidiarity=False)

    # Ablation 3b/c/d: Individual Optimization Levers Removed
    r_no_der = solve_exact_milp(df, make_params(elig_der_hydro, ex_kw_hydro, False, True, True), 300e6, subsidiarity=True)
    r_no_con = solve_exact_milp(df, make_params(elig_der_hydro, ex_kw_hydro, True, True, False), 300e6, subsidiarity=True)
    r_no_dec = solve_exact_milp(df, make_params(elig_der_hydro, ex_kw_hydro, True, False, True), 300e6, subsidiarity=True)

    opt_ablation_rows = [
        {
            "Module": "0. Full Integrated Framework",
            "CO2_Abated_kt": r_full["co2_abated_kt"],
            "CO2_Abated_Pct": r_full["co2_abated_pct"],
            "Delta_vs_Full_kt": 0.0,
            "Retained_GT_Pct": r_full["retained_gt_pct"],
            "Max_Country_Loss_Pct": r_full["max_country_loss_pct"],
            "Country_Gini": r_full["country_loss_gini"],
            "Key_Diagnostic": f"Artisanal (<12m) flagged: {fp_small_hydro:.1f}%; Artisanal decommissioned: {art_decom_full:,}",
        },
        {
            "Module": "1. w/o Hydrodynamic Scaling (Linear EII)",
            "CO2_Abated_kt": r_no_hydro["co2_abated_kt"],
            "CO2_Abated_Pct": r_no_hydro["co2_abated_pct"],
            "Delta_vs_Full_kt": round(r_no_hydro["co2_abated_kt"] - r_full["co2_abated_kt"], 2),
            "Retained_GT_Pct": r_no_hydro["retained_gt_pct"],
            "Max_Country_Loss_Pct": r_no_hydro["max_country_loss_pct"],
            "Country_Gini": r_no_hydro["country_loss_gini"],
            "Key_Diagnostic": f"<12m flagged jumps {fp_small_hydro:.1f}%->{fp_small_linear:.1f}%; {misalloc_vessels_lin:,} artisanal vessels mis-derated (€{misalloc_spend_lin_meur:.1f}M)",
        },
        {
            "Module": "2. w/o ML Typologies (No GMM Shield)",
            "CO2_Abated_kt": r_no_ml["co2_abated_kt"],
            "CO2_Abated_Pct": r_no_ml["co2_abated_pct"],
            "Delta_vs_Full_kt": round(r_no_ml["co2_abated_kt"] - r_full["co2_abated_kt"], 2),
            "Retained_GT_Pct": r_no_ml["retained_gt_pct"],
            "Max_Country_Loss_Pct": r_no_ml["max_country_loss_pct"],
            "Country_Gini": r_no_ml["country_loss_gini"],
            "Key_Diagnostic": f"Artisanal decommissioning surges {art_decom_full:,}->{art_decom_noml:,} (+{art_decom_noml - art_decom_full:,} boats)",
        },
        {
            "Module": "3a. w/o Subsidiarity Cap (Unconstrained)",
            "CO2_Abated_kt": r_no_sub["co2_abated_kt"],
            "CO2_Abated_Pct": r_no_sub["co2_abated_pct"],
            "Delta_vs_Full_kt": round(r_no_sub["co2_abated_kt"] - r_full["co2_abated_kt"], 2),
            "Retained_GT_Pct": r_no_sub["retained_gt_pct"],
            "Max_Country_Loss_Pct": r_no_sub["max_country_loss_pct"],
            "Country_Gini": r_no_sub["country_loss_gini"],
            "Key_Diagnostic": f"Max national loss hits {r_no_sub['max_country_loss_pct']:.1f}% (Gini {r_no_sub['country_loss_gini']:.3f})",
        },
        {
            "Module": "3b. w/o Engine Derating Lever",
            "CO2_Abated_kt": r_no_der["co2_abated_kt"],
            "CO2_Abated_Pct": r_no_der["co2_abated_pct"],
            "Delta_vs_Full_kt": round(r_no_der["co2_abated_kt"] - r_full["co2_abated_kt"], 2),
            "Retained_GT_Pct": r_no_der["retained_gt_pct"],
            "Max_Country_Loss_Pct": r_no_der["max_country_loss_pct"],
            "Country_Gini": r_no_der["country_loss_gini"],
            "Key_Diagnostic": f"Abatement drops {abs(r_no_der['co2_abated_kt'] - r_full['co2_abated_kt']):.1f} kt without low-cost derating",
        },
        {
            "Module": "3c. w/o Gear Conversion Lever",
            "CO2_Abated_kt": r_no_con["co2_abated_kt"],
            "CO2_Abated_Pct": r_no_con["co2_abated_pct"],
            "Delta_vs_Full_kt": round(r_no_con["co2_abated_kt"] - r_full["co2_abated_kt"], 2),
            "Retained_GT_Pct": r_no_con["retained_gt_pct"],
            "Max_Country_Loss_Pct": r_no_con["max_country_loss_pct"],
            "Country_Gini": r_no_con["country_loss_gini"],
            "Key_Diagnostic": f"Retained GT drops {r_full['retained_gt_pct']:.1f}%->{r_no_con['retained_gt_pct']:.1f}% (forced more buybacks)",
        },
        {
            "Module": "3d. w/o Decommissioning Lever",
            "CO2_Abated_kt": r_no_dec["co2_abated_kt"],
            "CO2_Abated_Pct": r_no_dec["co2_abated_pct"],
            "Delta_vs_Full_kt": round(r_no_dec["co2_abated_kt"] - r_full["co2_abated_kt"], 2),
            "Retained_GT_Pct": r_no_dec["retained_gt_pct"],
            "Max_Country_Loss_Pct": r_no_dec["max_country_loss_pct"],
            "Country_Gini": r_no_dec["country_loss_gini"],
            "Key_Diagnostic": f"100% GT retained, but max abatement capped at {r_no_dec['co2_abated_kt']:.1f} kt (€{r_no_dec['spend_meur']:.1f}M spent)",
        },
    ]

    # ------------------------------------------------------------------
    # Part B: Dynamic Behavioral ABM Ablations (Module 4)
    # ------------------------------------------------------------------
    base_10k = prepare_agent_arrays(df, sample_n=10000, seed=42)
    def_p = {"psi_cong": 0.28, "psi_eff": 0.15, "psi_tax": 0.20, "delta_p_fuel": 0.55, "tau_max": 120.0, "beta_adopt": 1.0, "kappa_stock": 0.22, "alpha_deg": 0.0035}

    # 4a: Static naive expectation (rebound coefficients = 0, no aging erosion)
    p_static = {**def_p, "psi_cong": 0.0, "psi_eff": 0.0, "psi_tax": 0.0, "alpha_deg": 0.0}
    t_static = simulate_abm(base_10k, "Coordinated Algorithmic Transition", p_static, seed=100, return_traj=True)

    # 4b: Dynamic ABM without Carbon Levy or Effort Controls (tau_max=0, psi_tax=0 on Coordinated Transition, or Static Heuristic)
    p_notax = {**def_p, "tau_max": 0.0}
    t_notax = simulate_abm(base_10k, "Coordinated Algorithmic Transition", p_notax, seed=100, return_traj=True)

    # 4c: Full Coordinated ABM
    t_full = simulate_abm(base_10k, "Coordinated Algorithmic Transition", def_p, seed=100, return_traj=True)

    # 4d: Static Heuristic Policy (rebound active, no carbon tax / adaptive controls)
    t_heur = simulate_abm(base_10k, "Static Heuristic Policy", def_p, seed=100, return_traj=True)

    co2_2026 = float(t_full.iloc[0]["co2_mt"])
    abm_ablation_rows = []
    for lbl, tdf, diag in [
        ("4a. Static Projection (w/o Rebound & Aging)", t_static, "Assumes zero Jevons rebound (0.0%) and zero thermal aging"),
        ("4b. Full ABM + Coordinated Transition", t_full, "Endogenous rebound restrained by phased €120/t levy & effort caps"),
        ("4c. Dynamic ABM w/o Carbon Pricing (tau=0)", t_notax, "Slower retrofit uptake & unmitigated efficiency/congestion rebound"),
        ("4d. Dynamic ABM + Unmanaged Static Buyback", t_heur, "One-off 2026 buyback triggers strong congestion rebound"),
    ]:
        r35 = tdf[tdf["year"] == 2035].iloc[0]
        cut_pct = (1.0 - r35["co2_mt"] / co2_2026) * 100.0
        abm_ablation_rows.append({
            "ABM_Variant": lbl,
            "CO2_2035_Mt": round(float(r35["co2_mt"]), 3),
            "Net_CO2_Reduction_Pct": round(cut_pct, 2),
            "Effort_Rebound_2035_Pct": round(float(r35["effort_rebound_pct"]), 2),
            "Profit_Margin_2035_Pct": round(float(r35["mean_profit_margin_pct"]), 2),
            "Active_Fleet_Pct": round(float(r35["retained_pct"]), 2),
            "Diagnostic": diag,
        })

    opt_ab_df = pd.DataFrame(opt_ablation_rows)
    abm_ab_df = pd.DataFrame(abm_ablation_rows)
    opt_ab_df.to_csv(os.path.join(RESULTS_V3, "ablation_optimization_modules.csv"), index=False)
    abm_ab_df.to_csv(os.path.join(RESULTS_V3, "ablation_dynamic_abm_modules.csv"), index=False)

    print("\nStatic Optimization Module Ablations (€300M Budget):")
    print(opt_ab_df[["Module", "CO2_Abated_kt", "Delta_vs_Full_kt", "Retained_GT_Pct", "Max_Country_Loss_Pct"]].to_string(index=False))
    print("\nDynamic Behavioral ABM Ablations (2026-2035):")
    print(abm_ab_df[["ABM_Variant", "CO2_2035_Mt", "Net_CO2_Reduction_Pct", "Effort_Rebound_2035_Pct", "Profit_Margin_2035_Pct"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Publication-Grade Figure 8: 4-Panel Component Ablation Breakdown
    # ------------------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.2, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), dpi=300)

    # Panel A: Hydrodynamic vs Linear EII Artisanal Misclassification
    ax = axes[0, 0]
    cats_a = ["Small Vessels (<12m)\nFlagged Overpowered (%)", "Misallocated Derating\nSpend on <12m (€M)"]
    vals_hydro = [fp_small_hydro, 0.0]
    vals_lin = [fp_small_linear, misalloc_spend_lin_meur]
    x_a = np.arange(len(cats_a))
    w = 0.34
    ax.bar(x_a - w / 2, vals_hydro, width=w, color="#1f77b4", label="Full (Hydrodynamic $\\Delta EII_i$)")
    ax.bar(x_a + w / 2, vals_lin, width=w, color="#d62728", label="Ablation 1 (Linear $kW/GT$)")
    ax.set_xticks(x_a)
    ax.set_xticklabels(cats_a)
    ax.set_ylabel("Misclassification / Spend Value")
    ax.set_title("A. Hydrodynamic Module: Small-Vessel Bias Correction")
    for i in range(len(cats_a)):
        ax.text(i - w / 2, vals_hydro[i] + 1.0, f"{vals_hydro[i]:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(i + w / 2, vals_lin[i] + 1.0, f"{vals_lin[i]:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Marginal CO2 Abatement Loss when Removing Each Module (€300M)
    ax = axes[0, 1]
    sub_b = opt_ab_df.iloc[1:].copy()
    labels_b = [s.split(". ", 1)[1] for s in sub_b["Module"]]
    y_b = np.arange(len(sub_b))[::-1]
    cols_b = ["#2ca02c" if v > 0 else "#d62728" for v in sub_b["Delta_vs_Full_kt"]]
    ax.barh(y_b, sub_b["Delta_vs_Full_kt"], color=cols_b, alpha=0.85, height=0.58)
    ax.axvline(0.0, color="#111111", linewidth=1.2)
    ax.set_yticks(y_b)
    ax.set_yticklabels(labels_b, fontsize=8.2)
    ax.set_xlabel(r"$\Delta \mathrm{CO}_2$ Abatement vs. Full Framework ($\mathrm{kt\,y}^{-1}$)")
    ax.set_title("B. Optimization Lever & Constraint Ablations (€300M)")
    for y_p, v in zip(y_b, sub_b["Delta_vs_Full_kt"]):
        off = 20 if v >= 0 else -20
        ha = "left" if v >= 0 else "right"
        ax.text(v + off, y_p, f"{v:+.0f} kt", va="center", ha=ha, fontsize=7.8)
    ax.set_xlim(sub_b["Delta_vs_Full_kt"].min() * 1.25, max(350, sub_b["Delta_vs_Full_kt"].max() * 1.35))
    ax.grid(axis="x", alpha=0.25)

    # Panel C: ML Typology Shield & National Subsidiarity Equity Outcomes
    ax = axes[1, 0]
    bars_c = ["Full\nFramework", "w/o ML Shield\n(Artisanal Loss /100)", "w/o Subsidiarity\n(Max Nat. Loss %)"]
    vals_c = [r_full["max_country_loss_pct"], art_decom_noml / 100.0, r_no_sub["max_country_loss_pct"]]
    cols_c = ["#1f77b4", "#ff7f0e", "#d62728"]
    ax.bar(np.arange(3), vals_c, color=cols_c, width=0.52, alpha=0.88)
    ax.axhline(20.0, color="#222222", linestyle="--", linewidth=1.2, label="20% Subsidiarity Ceiling")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(bars_c, fontsize=8.4)
    ax.set_ylabel("Fleet Loss Metric (% or ×100 Artisanal Boats)")
    ax.set_title("C. Sovereign Equity & ML Livelihood Protection")
    ax.text(0, vals_c[0] + 2, f"{r_full['max_country_loss_pct']:.1f}% max\n({art_decom_full:,} art.)", ha="center", fontsize=8.0)
    ax.text(1, vals_c[1] + 2, f"{art_decom_noml:,}\nartisanal boats", ha="center", fontsize=8.0)
    ax.text(2, vals_c[2] + 2, f"{r_no_sub['max_country_loss_pct']:.1f}%\nmax country", ha="center", fontsize=8.0)
    ax.set_ylim(0, max(vals_c) * 1.25)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Dynamic Behavioral ABM Rebound Erosion Gap
    ax = axes[1, 1]
    y_d = np.arange(len(abm_ab_df))[::-1]
    cols_d = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]
    ax.barh(y_d, abm_ab_df["Net_CO2_Reduction_Pct"], color=cols_d, alpha=0.88, height=0.55)
    ax.set_yticks(y_d)
    ax.set_yticklabels([s.split(". ", 1)[1] for s in abm_ab_df["ABM_Variant"]], fontsize=8.2)
    ax.set_xlabel("2035 Net Fleet CO2 Reduction vs. 2026 (%)")
    ax.set_title("D. Behavioral ABM & Rebound Erosion Gap")
    for y_p, r in zip(y_d, abm_ab_df.itertuples()):
        ax.text(r.Net_CO2_Reduction_Pct + 1.0, y_p, f"{r.Net_CO2_Reduction_Pct:.1f}% (reb {r.Effort_Rebound_2035_Pct:+.1f}%)", va="center", fontsize=7.8)
    ax.set_xlim(0, abm_ab_df["Net_CO2_Reduction_Pct"].max() * 1.38)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_V3, "Fig8_Framework_Ablation_Study.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_V3, "Fig8_Framework_Ablation_Study.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 8 in {FIGURES_V3}")
    print("=" * 72)


if __name__ == "__main__":
    run_ablation_suite()
