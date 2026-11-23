#!/usr/bin/env python3
"""
v3_05_abm_calibrated_sobol.py
=============================
Step 5 of Ocean Engineering (OE-D-26-13535) Revision:
Empirically Calibrated Endogenous Agent-Based Model (2026-2035) and
Global Sobol Variance-Based Sensitivity Analysis (SALib).

Addresses Reviewer 2 Comment (6):
  1. Eliminates all hardcoded rebound ramps and fixes the typology string bug.
  2. Computes vessel-level annual effort rebound strictly endogenously via Eq. (25)
     inside each GSA fishing ground across 10 annual time steps.
  3. Calibrates gear-specific fuel cost shares and Year-0 net profit margins to
     empirical EU STECF AER / FAO Mediterranean benchmarks.
  4. Replaces fixed +28% revenue boosts with an endogenous Schaefer density-dependent
     CPUE recovery curve linked to cumulative regional effort reduction.
  5. Replicates all 3 policy regimes across R = 30 independent random seeds (95% CI ribbons)
     and executes formal SALib Sobol sensitivity decomposition (S1, ST) across 8 behavioral
     and macroeconomic parameters.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from SALib.sample import sobol as saltelli_sample
from SALib.analyze import sobol as sobol_analyze
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_V3 = os.path.join(BASE_DIR, "Version_2_Optimization", "results_v3")
FIGURES_V3 = os.path.join(BASE_DIR, "Version_2_Optimization", "figures_v3")
os.makedirs(RESULTS_V3, exist_ok=True)
os.makedirs(FIGURES_V3, exist_ok=True)

# STECF AER / FAO Mediterranean Empirical Gear Calibration Targets
FUEL_SHARE_BY_GEAR = {"trawl": 0.32, "purse_seine": 0.19, "passive": 0.12, "other": 0.18, "unknown": 0.18}
TARGET_MARGIN_BY_GEAR = {"trawl": 0.08, "purse_seine": 0.11, "passive": 0.14, "other": 0.10, "unknown": 0.10}
FIXED_MAINT_SHARE = 0.08
CO2_PER_LITRE_KG = 2.64


def prepare_agent_arrays(df: pd.DataFrame, sample_n: int = 10000, seed: int = 42) -> dict:
    """Create stratified vessel agent arrays for fast vectorized 10-year simulation."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby("typology_name", observed=True):
        k = max(1, int(round(sample_n * len(grp) / len(df))))
        parts.append(grp.sample(n=min(k, len(grp)), random_state=seed))
    ag = pd.concat(parts, ignore_index=True)
    if len(ag) < sample_n:
        extra = df.sample(n=sample_n - len(ag), random_state=seed + 1)
        ag = pd.concat([ag, extra], ignore_index=True)
    elif len(ag) > sample_n:
        ag = ag.sample(n=sample_n, random_state=seed + 2).reset_index(drop=True)

    n = len(ag)
    scale_factor = len(df) / float(n)
    gear = ag["gear_group"].values
    fuel_l0 = ag["fuel_tier2_l"].values.astype(float)
    p_fuel0 = 0.90

    # Empirical Year-0 STECF cost & revenue calibration with +/-12% vessel heterogeneity
    het = rng.uniform(0.88, 1.12, size=n)
    f_sh = np.array([FUEL_SHARE_BY_GEAR.get(g, 0.18) for g in gear])
    m_tg = np.array([TARGET_MARGIN_BY_GEAR.get(g, 0.10) for g in gear])
    fuel_cost0 = fuel_l0 * p_fuel0
    rev0 = fuel_cost0 / (f_sh * het)
    maint0 = FIXED_MAINT_SHARE * rev0
    var_cost_ratio = np.clip(1.0 - f_sh - m_tg - FIXED_MAINT_SHARE, 0.25, 0.75)

    # GSA integer mapping for ground-level congestion & Schaefer CPUE
    gsa_codes, gsa_inv = np.unique(ag["gsa_1"].fillna("GSA_UNK").values, return_inverse=True)
    country_codes, c_inv = np.unique(ag["country"].values, return_inverse=True)
    c_counts0 = np.bincount(c_inv).astype(float)
    gsa_counts0 = np.bincount(gsa_inv).astype(float)

    return {
        "n": n,
        "scale_factor": scale_factor,
        "gt": ag["tonnage_gt"].values.astype(float),
        "kw0": ag["engine_power_kw"].values.astype(float),
        "kw_hydro": ag["kw_hydro_v3"].values.astype(float),
        "age0": ag["age_clean"].values.astype(float),
        "lf": ag["lf_tier2"].values.astype(float),
        "hr0": ag["hr_tier2"].values.astype(float),
        "fuel_l0": fuel_l0,
        "rev0": rev0,
        "maint0": maint0,
        "var_cost_ratio": var_cost_ratio,
        "is_trawl": (gear == "trawl"),
        "is_artisanal": ag["typology_name"].str.contains("Artisanal", na=False).values,
        "gear": gear,
        "gsa_inv": gsa_inv,
        "n_gsa": len(gsa_codes),
        "gsa_counts0": gsa_counts0,
        "c_inv": c_inv,
        "n_countries": len(country_codes),
        "c_counts0": c_counts0,
        "f_sh": f_sh,
        "m_tg": m_tg,
    }


def simulate_abm(base: dict, regime: str, params: dict, seed: int = 42, return_traj: bool = True):
    """
    Fast vectorized 10-year (2026-2035) endogenous ABM simulation.
    Zero hardcoded rebound ramps; effort h_{i,t} computed strictly via Eq. (25).
    """
    rng = np.random.default_rng(seed)
    n = base["n"]
    sf = base["scale_factor"]

    active = np.ones(n, dtype=bool)
    derated = np.zeros(n, dtype=bool)
    retrofitted = np.zeros(n, dtype=bool)
    kw = base["kw0"].copy()
    age = base["age0"].copy()
    # Baseline fuel per unit fishing hour (L/h)
    sfc0 = np.clip(0.22 * (1.0 + params["alpha_deg"] * age), 0.20, 0.28)
    l_per_h0 = kw * base["lf"] * sfc0
    l_per_h = l_per_h0.copy()
    h0 = base["hr0"].copy()

    # GSA cumulative effort saving tracker for Schaefer density-dependent CPUE recovery
    gsa_base_h = np.bincount(base["gsa_inv"], weights=h0, minlength=base["n_gsa"])
    cum_effort_red_gsa = np.zeros(base["n_gsa"], dtype=float)
    cum_spend_eur = 0.0
    min_active_total = int(round(0.80 * n))
    c_max_decom = np.floor(0.20 * base["c_counts0"]).astype(int)
    c_decom_used = np.zeros(base["n_countries"], dtype=int)

    p_fuel0 = 0.90
    years = range(2026, 2036)
    traj = []

    for t_idx, yr in enumerate(years):
        p_fuel = p_fuel0 + params["delta_p_fuel"] * (t_idx / 9.0)
        if regime == "Laissez-Faire Status Quo":
            tau = 0.0
        elif regime == "Static Heuristic Policy":
            tau = 0.0  # Static buyback + engine cap without adaptive carbon pricing
        else:
            tau = params["tau_max"] * (t_idx / 9.0) if yr >= 2027 else 0.0

        # 1. Vessel aging & thermal wear for non-retrofitted vessels
        age[active] += 1.0
        unretro = active & (~retrofitted)
        l_per_h[unretro] *= (1.0 + params["alpha_deg"])

        # 2. Policy interventions
        if regime == "Static Heuristic Policy" and yr == 2026:
            # Forced 20% buyback of oldest vessels (no national subsidiarity cap) + P50 trawl cap
            n_cut = n - min_active_total
            old_idx = np.argsort(-age)[:n_cut]
            active[old_idx] = False
            cum_spend_eur += float(np.sum(base["gt"][old_idx] * 2200.0) * sf)

            p50_trawl = float(np.median(kw[active & base["is_trawl"]]))
            cap_m = active & base["is_trawl"] & (kw > p50_trawl)
            kw_cut = np.maximum(0.0, kw[cap_m] - p50_trawl)
            l_per_h[cap_m] *= (p50_trawl / kw[cap_m])
            kw[cap_m] = p50_trawl
            cum_spend_eur += float(np.sum(kw_cut * 350.0) * sf)

        elif regime == "Coordinated Algorithmic Transition":
            # A. Endogenous allometric derating adoption (payback logistic diffusion)
            kw_safe = base["kw_hydro"] * 1.25
            ex_kw = np.maximum(0.0, kw - kw_safe)
            can_der = active & (~derated) & (ex_kw > 15.0)
            if np.any(can_der):
                ann_sav = ex_kw[can_der] * base["lf"][can_der] * h0[can_der] * 0.22 * 0.90 * p_fuel
                payback = (ex_kw[can_der] * 60.0) / np.maximum(1.0, ann_sav)
                prob_der = np.clip(1.0 / (1.0 + np.exp(params["beta_adopt"] * (payback - 2.5))), 0.15, 0.72)
                adopt_d = rng.random(np.sum(can_der)) < prob_der
                d_idx = np.where(can_der)[0][adopt_d]
                if len(d_idx) > 0:
                    ratio_d = 1.0 - 0.90 * (ex_kw[d_idx] / np.maximum(1.0, kw[d_idx]))
                    kw[d_idx] -= ex_kw[d_idx]
                    l_per_h[d_idx] *= ratio_d
                    derated[d_idx] = True
                    retrofitted[d_idx] = True
                    cum_spend_eur += float(np.sum(ex_kw[d_idx] * 60.0) * sf)

            # B. Hybrid-electric & low-drag gear retrofit diffusion (financed partly by carbon tax recycling)
            can_ret = active & (~retrofitted)
            if np.any(can_ret):
                # Higher carbon levy + fuel price accelerates efficiency retrofit adoption
                tax_push = 0.04 * (tau / 100.0)
                prob_r = np.clip(0.08 + 0.02 * t_idx + tax_push, 0.05, 0.30)
                adopt_r = rng.random(np.sum(can_ret)) < prob_r
                r_idx = np.where(can_ret)[0][adopt_r]
                if len(r_idx) > 0:
                    # Trawlers get 35% drag/hybrid reduction; others get 22% hydrodynamic/hybrid reduction
                    factor_r = np.where(base["is_trawl"][r_idx], 0.65, 0.78)
                    l_per_h[r_idx] *= factor_r
                    retrofitted[r_idx] = True
                    cum_spend_eur += float(len(r_idx) * 18000.0 * 0.50 * sf)  # 50% public co-funding

            # C. Gradual Subsidiarity Reverse-Auction Buyback (respects <=20% country cap & artisanal floor)
            rem_total_quota = max(0, int(np.sum(active)) - min_active_total)
            yr_quota = min(rem_total_quota, int(round((n - min_active_total) / 10.0)))
            if yr == 2035:
                yr_quota = rem_total_quota
            if yr_quota > 0:
                elig = active & (age > 24.0) & (~base["is_artisanal"])
                cand = np.where(elig)[0]
                if len(cand) > 0:
                    unit_val = np.maximum(500.0, 2200.0 * np.exp(-0.035 * age[cand]))
                    cost_c = base["gt"][cand] * unit_val
                    co2_c = l_per_h[cand] * h0[cand] * CO2_PER_LITRE_KG / 1000.0
                    # Score: CO2 abated per EUR
                    score = co2_c / np.maximum(1.0, cost_c)
                    ranked = cand[np.argsort(-score)]
                    picked = 0
                    for idx_v in ranked:
                        if picked >= yr_quota:
                            break
                        cid = base["c_inv"][idx_v]
                        if c_decom_used[cid] < c_max_decom[cid]:
                            active[idx_v] = False
                            c_decom_used[cid] += 1
                            cum_spend_eur += float(base["gt"][idx_v] * np.maximum(500.0, 2200.0 * np.exp(-0.035 * age[idx_v])) * sf)
                            picked += 1

        # 3. Endogenous Effort Rebound strictly via Eq. (25)
        gsa_active_n = np.bincount(base["gsa_inv"][active], minlength=base["n_gsa"]).astype(float)
        attr_gsa = (base["gsa_counts0"] - gsa_active_n) / np.maximum(1.0, base["gsa_counts0"])
        attr_i = attr_gsa[base["gsa_inv"]]
        eff_gain_i = np.maximum(0.0, (l_per_h0 - l_per_h) / np.maximum(1e-6, l_per_h0))

        # Effective fuel + carbon price index relative to Year 0
        p_eff0 = p_fuel0
        p_eff_t = p_fuel + tau * CO2_PER_LITRE_KG / 1000.0
        dp_eff = (p_eff_t - p_eff0) / p_eff0

        # In Coordinated Transition, adaptive effort rules restrain congestion rebound by 50%
        cong_mult = 0.50 if regime == "Coordinated Algorithmic Transition" else 1.0
        effort_mult = np.clip(
            1.0 + cong_mult * params["psi_cong"] * attr_i + params["psi_eff"] * eff_gain_i - params["psi_tax"] * dp_eff,
            0.55, 1.40
        )
        h_t = h0 * effort_mult
        fuel_l_t = l_per_h * h_t
        co2_t = fuel_l_t * CO2_PER_LITRE_KG / 1000.0

        # 4. Endogenous Schaefer Density-Dependent CPUE Recovery & Financial Ledger (Eq. 24)
        gsa_cur_h = np.bincount(base["gsa_inv"][active], weights=h_t[active], minlength=base["n_gsa"])
        effort_red_gsa = np.maximum(0.0, (gsa_base_h - gsa_cur_h) / np.maximum(1.0, gsa_base_h))
        cum_effort_red_gsa = np.clip(cum_effort_red_gsa + 0.35 * effort_red_gsa, 0.0, 1.0)
        cpue_mult = 1.0 + params["kappa_stock"] * cum_effort_red_gsa[base["gsa_inv"]]

        # Catch responds with sub-linear effort elasticity (0.65)
        rev_t = base["rev0"] * ((h_t / h0) ** 0.65) * cpue_mult
        fuel_exp_t = fuel_l_t * p_fuel
        carbon_tax_t = co2_t * tau
        # In Coordinated Transition, 75% of carbon levy is recycled back into fleet efficiency/livelihood rebates
        net_tax_burden = 0.25 * carbon_tax_t if regime == "Coordinated Algorithmic Transition" else carbon_tax_t
        op_cost_t = base["var_cost_ratio"] * rev_t + base["maint0"] + fuel_exp_t + net_tax_burden
        profit_t = rev_t - op_cost_t
        margin_t = profit_t / np.maximum(1.0, rev_t)

        # In Laissez-Faire Status Quo, sustained deeply negative margins (<-15%) force exit of older vessels
        if regime == "Laissez-Faire Status Quo":
            bankrupt = active & (margin_t < -0.15) & (age > 28.0)
            if np.any(bankrupt):
                # Realistic gradual insolvency hazard (35% exit probability per year under severe loss)
                exit_m = rng.random(np.sum(bankrupt)) < 0.35
                b_idx = np.where(bankrupt)[0][exit_m]
                active[b_idx] = False

        if return_traj:
            tot_co2_mt = float(co2_t[active].sum() * sf / 1e6)
            mean_reb_pct = float((np.mean(effort_mult[active]) - 1.0) * 100.0)
            mean_marg_pct = float(np.mean(margin_t[active]) * 100.0)
            traj.append({
                "regime": regime,
                "seed": seed,
                "year": yr,
                "active_vessels": int(round(np.sum(active) * sf)),
                "retained_pct": round(float(np.mean(active) * 100.0), 2),
                "co2_mt": round(tot_co2_mt, 4),
                "effort_rebound_pct": round(mean_reb_pct, 2),
                "mean_profit_margin_pct": round(mean_marg_pct, 2),
                "cumulative_spend_meur": round(cum_spend_eur / 1e6, 2),
            })

    if return_traj:
        return pd.DataFrame(traj)
    # Fast scalar summary for Sobol GSA
    co2_0 = float(base["fuel_l0"].sum() * CO2_PER_LITRE_KG / 1000.0)
    co2_2035 = float(co2_t[active].sum())
    co2_cut_pct = (1.0 - co2_2035 / co2_0) * 100.0
    reb_2035 = float((np.mean(effort_mult[active]) - 1.0) * 100.0)
    marg_2035 = float(np.mean(margin_t[active]) * 100.0)
    return co2_cut_pct, reb_2035, marg_2035


def main():
    print("=" * 72)
    print(" v3_05_abm_calibrated_sobol.py — Calibrated Endogenous ABM & Sobol GSA")
    print("=" * 72)

    df = pd.read_parquet(os.path.join(RESULTS_V3, "fleet_v3_master.parquet"))
    base_10k = prepare_agent_arrays(df, sample_n=10000, seed=42)

    # Save calibration check against STECF AER gear benchmarks
    cal_rows = []
    for g in ["trawl", "purse_seine", "passive", "other"]:
        m = base_10k["gear"] == g
        f_sh_obs = float(np.mean((base_10k["fuel_l0"][m] * 0.90) / base_10k["rev0"][m]))
        cal_rows.append({
            "Gear_Group": g,
            "Sample_Agents": int(m.sum()),
            "Target_STECF_Fuel_Share_Pct": round(FUEL_SHARE_BY_GEAR[g] * 100.0, 1),
            "Calibrated_Fuel_Share_Pct": round(f_sh_obs * 100.0, 2),
            "Target_STECF_Net_Margin_Pct": round(TARGET_MARGIN_BY_GEAR[g] * 100.0, 1),
        })
    pd.DataFrame(cal_rows).to_csv(os.path.join(RESULTS_V3, "abm_calibration_benchmarks.csv"), index=False)

    # Baseline calibrated parameters
    default_params = {
        "psi_cong": 0.28,
        "psi_eff": 0.15,
        "psi_tax": 0.20,
        "delta_p_fuel": 0.55,   # €0.90 -> €1.45/L
        "tau_max": 120.0,       # €/t CO2 by 2035
        "beta_adopt": 1.0,
        "kappa_stock": 0.22,
        "alpha_deg": 0.0035,
    }

    # ------------------------------------------------------------------
    # 1. R = 30 Multi-Seed Stochastic Replication across 3 Regimes
    # ------------------------------------------------------------------
    regimes = ["Laissez-Faire Status Quo", "Static Heuristic Policy", "Coordinated Algorithmic Transition"]
    all_runs = []
    t0 = time.time()
    for reg in regimes:
        for s in range(30):
            tdf = simulate_abm(base_10k, reg, default_params, seed=100 + s, return_traj=True)
            all_runs.append(tdf)
    runs_df = pd.concat(all_runs, ignore_index=True)

    summary_traj = runs_df.groupby(["regime", "year"], as_index=False).agg(
        active_vessels_mean=("active_vessels", "mean"),
        active_vessels_lo=("active_vessels", lambda x: np.percentile(x, 2.5)),
        active_vessels_hi=("active_vessels", lambda x: np.percentile(x, 97.5)),
        retained_pct_mean=("retained_pct", "mean"),
        co2_mt_mean=("co2_mt", "mean"),
        co2_mt_lo=("co2_mt", lambda x: np.percentile(x, 2.5)),
        co2_mt_hi=("co2_mt", lambda x: np.percentile(x, 97.5)),
        effort_rebound_mean=("effort_rebound_pct", "mean"),
        effort_rebound_lo=("effort_rebound_pct", lambda x: np.percentile(x, 2.5)),
        effort_rebound_hi=("effort_rebound_pct", lambda x: np.percentile(x, 97.5)),
        profit_margin_mean=("mean_profit_margin_pct", "mean"),
        profit_margin_lo=("mean_profit_margin_pct", lambda x: np.percentile(x, 2.5)),
        profit_margin_hi=("mean_profit_margin_pct", lambda x: np.percentile(x, 97.5)),
        spend_meur_mean=("cumulative_spend_meur", "mean"),
    )
    summary_traj.to_csv(os.path.join(RESULTS_V3, "abm_calibrated_trajectories_30seeds.csv"), index=False)
    print(f"Completed 30-seed ABM replication across 3 regimes in {time.time() - t0:.2f}s.")
    end_2035 = summary_traj[summary_traj["year"] == 2035]
    base_2026_co2 = float(summary_traj[summary_traj["year"] == 2026]["co2_mt_mean"].mean())
    print("\n2035 Endpoint Summary (30 Seeds Mean [95% CI]):")
    for _, r in end_2035.iterrows():
        cut = (1.0 - r["co2_mt_mean"] / base_2026_co2) * 100.0
        print(f"  {r['regime']:34s} | CO2={r['co2_mt_mean']:.3f} Mt ({cut:+.1f}% vs 2026) | Vessels={r['active_vessels_mean']:,.0f} ({r['retained_pct_mean']:.1f}%) | Rebound={r['effort_rebound_mean']:+.2f}% | Margin={r['profit_margin_mean']:+.2f}% | Spend=€{r['spend_meur_mean']:.1f}M")

    # ------------------------------------------------------------------
    # 2. Global Sobol Sensitivity Analysis (SALib, N=128 -> 2,304 runs)
    # ------------------------------------------------------------------
    print("\nRunning SALib Sobol Global Sensitivity Analysis (8 parameters, 2,304 ABM evaluations)...")
    problem = {
        "num_vars": 8,
        "names": ["psi_cong", "psi_eff", "psi_tax", "delta_p_fuel", "tau_max", "beta_adopt", "kappa_stock", "alpha_deg"],
        "bounds": [
            [0.10, 0.45],   # psi_cong
            [0.05, 0.25],   # psi_eff
            [0.10, 0.35],   # psi_tax
            [0.20, 0.80],   # delta_p_fuel
            [60.0, 180.0],  # tau_max
            [0.50, 2.00],   # beta_adopt
            [0.05, 0.30],   # kappa_stock
            [0.002, 0.006], # alpha_deg
        ],
    }
    param_values = saltelli_sample.sample(problem, 128, calc_second_order=True, seed=42)
    base_2500 = prepare_agent_arrays(df, sample_n=2500, seed=42)

    Y_co2_cut = np.zeros(len(param_values))
    Y_rebound = np.zeros(len(param_values))
    Y_margin = np.zeros(len(param_values))

    t_s0 = time.time()
    for i, row in enumerate(param_values):
        p_dict = dict(zip(problem["names"], row))
        c_cut, reb, mrg = simulate_abm(base_2500, "Coordinated Algorithmic Transition", p_dict, seed=42, return_traj=False)
        Y_co2_cut[i], Y_rebound[i], Y_margin[i] = c_cut, reb, mrg
    print(f"Evaluated {len(param_values):,} Sobol trajectories in {time.time() - t_s0:.2f}s.")

    sobol_rows = []
    for out_name, Y_arr in [("2035 CO2 Reduction (%)", Y_co2_cut), ("2035 Effort Rebound (%)", Y_rebound), ("2035 Profit Margin (%)", Y_margin)]:
        Si = sobol_analyze.analyze(problem, Y_arr, calc_second_order=True, num_resamples=500, seed=42)
        for j, pname in enumerate(problem["names"]):
            sobol_rows.append({
                "Output_Metric": out_name,
                "Parameter": pname,
                "S1": round(max(0.0, float(Si["S1"][j])), 4),
                "S1_conf": round(float(Si["S1_conf"][j]), 4),
                "ST": round(max(0.0, float(Si["ST"][j])), 4),
                "ST_conf": round(float(Si["ST_conf"][j]), 4),
            })

    sobol_df = pd.DataFrame(sobol_rows)
    sobol_df.to_csv(os.path.join(RESULTS_V3, "abm_sobol_sensitivity_indices.csv"), index=False)
    print("\nSobol Sensitivity Indices (2035 CO2 Reduction %):")
    print(sobol_df[sobol_df["Output_Metric"] == "2035 CO2 Reduction (%)"].sort_values("ST", ascending=False).to_string(index=False))

    # Save Sobol outcome distribution summary
    pd.DataFrame({
        "Y_co2_cut_pct": Y_co2_cut,
        "Y_rebound_pct": Y_rebound,
        "Y_margin_pct": Y_margin,
    }).to_csv(os.path.join(RESULTS_V3, "abm_sobol_outcome_distributions.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. Publication-Grade Figure 6: 4-Panel ABM Trajectories (30 Seeds)
    # ------------------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.2, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=300)
    reg_styles = {
        "Laissez-Faire Status Quo": ("#7f7f7f", "--", "Laissez-Faire Status Quo"),
        "Static Heuristic Policy": ("#d62728", "-.", "Static Heuristic (2026 Cut, No Levy)"),
        "Coordinated Algorithmic Transition": ("#1f77b4", "-", "Coordinated Algorithmic Transition"),
    }

    panels = [
        (axes[0, 0], "co2_mt", "Annual Operational Tier-2 CO2 (Mt)", "A. Fleet Decarbonization Trajectory (95% CI)"),
        (axes[0, 1], "active_vessels", "Active Fishing Vessels (n)", "B. Fleet Retention vs. 80% Subsidiarity Floor"),
        (axes[1, 0], "profit_margin", "Mean Vessel Operating Margin (%)", "C. Fleet Economic Viability (STECF-Calibrated)"),
        (axes[1, 1], "effort_rebound", "Endogenous Effort Rebound vs. 2026 (%)", "D. Emergent Behavioral Effort Rebound (Eq. 25)"),
    ]
    for ax, col_prefix, ylabel, title in panels:
        for reg, (col, ls, lbl) in reg_styles.items():
            sub = summary_traj[summary_traj["regime"] == reg].sort_values("year")
            ax.plot(sub["year"], sub[f"{col_prefix}_mean"], color=col, linestyle=ls, linewidth=2.2, label=lbl)
            ax.fill_between(sub["year"], sub[f"{col_prefix}_lo"], sub[f"{col_prefix}_hi"], color=col, alpha=0.16)
        if col_prefix == "active_vessels":
            ax.axhline(0.80 * len(df), color="#222222", linestyle=":", linewidth=1.3, label="80.0% Subsidiarity Floor (54,133)")
        if col_prefix in ("profit_margin", "effort_rebound"):
            ax.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Simulation Year")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7.6, frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_V3, "Fig6_ABM_Calibrated_Trajectories.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_V3, "Fig6_ABM_Calibrated_Trajectories.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Publication-Grade Figure 7: Sobol Global Sensitivity Analysis
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=300)
    ax = axes[0]
    s_co2 = sobol_df[sobol_df["Output_Metric"] == "2035 CO2 Reduction (%)"].sort_values("ST", ascending=True)
    y_p = np.arange(len(s_co2))
    h = 0.36
    p_labels = {
        "psi_cong": r"Congestion Rebound ($\psi_{cong}$)",
        "psi_eff": r"Efficiency Rebound ($\psi_{eff}$)",
        "psi_tax": r"Price/Tax Elasticity ($\psi_{tax}$)",
        "delta_p_fuel": r"Fuel Price Escalation ($\Delta P_{fuel}$)",
        "tau_max": r"2035 Carbon Levy ($\tau_{max}$)",
        "beta_adopt": r"Payback Adoption Slope ($\beta_{adopt}$)",
        "kappa_stock": r"Stock CPUE Elasticity ($\kappa_{stock}$)",
        "alpha_deg": r"Thermal Aging Rate ($\alpha_{deg}$)",
    }
    ax.barh(y_p - h / 2, s_co2["S1"], height=h, xerr=s_co2["S1_conf"], color="#aec7e8", edgecolor="#1f77b4", label="First-Order ($S_1$)", capsize=2.5)
    ax.barh(y_p + h / 2, s_co2["ST"], height=h, xerr=s_co2["ST_conf"], color="#1f77b4", label="Total-Order ($S_T$)", capsize=2.5)
    ax.set_yticks(y_p)
    ax.set_yticklabels([p_labels.get(x, x) for x in s_co2["Parameter"]], fontsize=8.3)
    ax.set_xlabel("Sobol Variance Sensitivity Index")
    ax.set_title("A. Sobol Variance Decomposition (2035 CO2 Cut)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    ax.hist(Y_co2_cut, bins=32, color="#1f77b4", alpha=0.78, edgecolor="white")
    med_c = float(np.median(Y_co2_cut))
    p5_c, p95_c = float(np.percentile(Y_co2_cut, 5)), float(np.percentile(Y_co2_cut, 95))
    ax.axvline(med_c, color="#111111", linestyle="--", linewidth=1.8, label=f"Median: {med_c:.1f}%")
    ax.axvline(p5_c, color="#d62728", linestyle=":", linewidth=1.5, label=f"5th–95th Pct: [{p5_c:.1f}%, {p95_c:.1f}%]")
    ax.axvline(p95_c, color="#d62728", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Realized 2035 CO2 Abatement across 2,304 Sobol Draws (%)")
    ax.set_ylabel("Simulation Count")
    ax.set_title("B. Multi-Parameter Uncertainty Envelope (2035)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_V3, "Fig7_ABM_Sobol_Global_Sensitivity.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_V3, "Fig7_ABM_Sobol_Global_Sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figures 6 & 7 in {FIGURES_V3}")
    print("=" * 72)


if __name__ == "__main__":
    main()
