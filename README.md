# Mediterranean and Black Sea Fishing Fleet Decarbonization: Replication Codebase

Replication Python codebase (**6 core analytical modules; zero proprietary or raw datasets included**) accompanying the manuscript:

> **Basin-Wide Decarbonization of Mediterranean and Black Sea Fishing Fleets: Physical Allometry, Exact Multi-Objective Optimization, and Calibrated Agent-Based Transition Governance**  
> *Submitted to Maritime Studies (Springer Nature)*

## Data Policy (No Datasets Included)

In accordance with institutional data-stewardship and repository hygiene practices, **no raw or derived vessel-level datasets (`.csv`, `.parquet`, `.xlsx`) are distributed in this repository**. Users wishing to reproduce the full empirical pipeline should download the publicly available primary registries directly from their authoritative custodians:

1. **GFCM Authorized Vessel List (AVL):** General Fisheries Commission for the Mediterranean (FAO), available at <https://www.fao.org/gfcm/data/fleet/en>.
2. **FAO Marine Capture Landings (FishStatJ / Area 37):** Food and Agriculture Organization of the United Nations, available at <https://www.fao.org/fishery/en/statistics/software/fishstatj>.
3. **STECF Annual Economic Report (AER) on the EU Fishing Fleet:** Joint Research Centre / European Commission, available at <https://stecf.jrc.ec.europa.eu/reports/economic>.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
└── scripts/
    ├── v3_01_hydrodynamic_validation.py       # Nested Admiralty/Froude allometry, material effects, Holtrop-Mennen drag, sea-trial & LOBO validation
    ├── v3_02_did_causal_robustness.py         # 2014 EU CFP TWFE & event-study DiD, GSA/country clustering, wild bootstrap, placebos & LOCO
    ├── v3_03_emissions_validation_and_ml.py   # Unified Tier-2 fuel/CO2 baseline, 14-benchmark cross-scale validation, GMM typology, non-circular ML & TreeSHAP
    ├── v3_04_exact_momilp_and_benchmarks.py   # Exact HiGHS branch-and-cut subsidiarity MOMILP, Cost of Subsidiarity, 6-strategy Pareto benchmark
    ├── v3_05_abm_calibrated_sobol.py          # Calibrated 67,666-vessel decadal ABM (30 seeds, 2026-2035) & 2,304-run Saltelli-Sobol GSA
    └── v3_06_ablation_study.py                # 5-module static optimization and dynamic ABM governance ablation experiments
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution Order

Once the harmonized fleet input file is placed in the expected local working directory (`results/eii_constructed.parquet`), execute the six core modules sequentially:

```bash
python3 scripts/v3_01_hydrodynamic_validation.py
python3 scripts/v3_02_did_causal_robustness.py
python3 scripts/v3_03_emissions_validation_and_ml.py
python3 scripts/v3_04_exact_momilp_and_benchmarks.py
python3 scripts/v3_05_abm_calibrated_sobol.py
python3 scripts/v3_06_ablation_study.py
```

## License

This codebase is released under the **MIT License** (see `LICENSE`).
