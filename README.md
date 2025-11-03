# 📈 HedgeForge: Portfolio Risk Modeling Engine (In Progress)

<p align="left">
  <!-- Project Status -->
  <img src="https://img.shields.io/badge/status-in_progress-orange?style=flat-square" alt="Project Status">
  
  <!-- Python Version -->
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python Version">
  
  <!-- Build Status -->
  <img src="https://img.shields.io/badge/build-passing-brightgreen?style=flat-square" alt="Build Status">
  
  <!-- Documentation -->
  <img src="https://img.shields.io/badge/docs-Quarto-success?style=flat-square" alt="Documentation">
  
  <!-- License -->
  <img src="https://img.shields.io/badge/license-review_only-lightgrey?style=flat-square" alt="License">
  
  <!-- Last Updated -->
  <img src="https://img.shields.io/badge/last%20updated-November%202025-informational?style=flat-square" alt="Last Updated">
</p>

---

A Python-based quantitative finance project for simulating and optimizing portfolio strategies using **Monte Carlo simulations**, **stochastic processes**, and **risk analytics**.  
Ideal for roles in **Asset-Liability Management (ALM)**, **ESG modeling**, and **institutional finance**.

**Author:** Brice A. Nelson  
**Tags:** quantitative finance, portfolio optimization, Monte Carlo, asset-liability management, ESG modeling, Python, risk analytics, financial modeling  
**Repository:** [github.com/Brice-Financial-Projects/HedgeForge](https://github.com/Brice-Financial-Projects/HedgeForge)


---

HedgeForge is a Python-based quantitative finance project focused on simulating and optimizing long-term portfolio performance under uncertainty. Using stochastic processes, Monte Carlo simulations, and modern portfolio theory, the engine is being developed to model and evaluate real-world portfolio strategies under risk — with applications in **asset-liability management (ALM)**, **economic scenario generation (ESG)**, and **institutional investing**.

This project is designed with modularity, transparency, and future extensibility in mind. It also serves as a portfolio centerpiece in my career transition from engineering and infrastructure planning into fintech and quantitative modeling.

---

## 🔍 Key Objectives

- Develop a flexible, research-grade simulation engine for multi-asset portfolio strategies  
- Apply stochastic modeling (e.g., GBM, OU process) to simulate market dynamics  
- Implement portfolio optimization under different risk/utility frameworks  
- Perform rolling backtests on historical and synthetic market data  
- Produce reproducible documentation using Quarto to explain methodology and results  
- Design with scalability in mind (e.g., Streamlit front-end, `cvxpy` constraint-based optimization planned)

---

## 🛠️ Tech Stack

**Core**: Python, NumPy, Pandas, SciPy, Matplotlib, Quarto  
**Planned**: Streamlit (interactive dashboard), cvxpy (constrained optimization), DuckDB/SQL (data layer)  
**Tooling**: Git, Conda, pytest, VSCode

---

## 📁 Project Structure

```plaintext
hedge_forge/
│
├── .github/                      # GitHub Actions workflows
│   └── workflows/                
│       ├── ci-cd.yaml
│       └── smoke.yaml             
│
├── docs/                              # Project documentation (Quarto site)
│   │
│   ├── 00_overview/
│   │   ├── project_overview.md
│   │   ├── project_overview_README.md
│   │   └── summary_outline.md
│   │
│   ├── 01_planning/
│   │   ├── planning_roadmap_checklist.md
│   │   ├── roadmap.md
│   │   └── phase_plan_logical.md
│   │
│   ├── 02_architecture/
│   │   ├── structure.md
│   │   ├── docs_structure.md
│   │   ├── system_design.md
│   │   └── api_endpoints.md
│   │
│   ├── 03_modeling/
│   │   ├── modeling_notes.md
│   │   ├── outline.md
│   │   └── formulas_appendix.md
│   │
│   ├── 04_engineering/
│   │   ├── ci_cd_pipeline.md
│   │   ├── deployment_notes.md
│   │   └── environment_setup.md
│   │
│   ├── 05_dev_notes/
│   │   ├── diary_logs/
│   │   │   ├── 2025_10_devlog.md
│   │   │   ├── ci_cd_progress.md
│   │   │   └── monte_carlo_experiments.md
│   │   └── dev_summary.md
│   │
│   ├── 06_references/
│   │   ├── references.md
│   │   ├── papers.md
│   │   └── citations.bib
│   │
│   └── archive/
│       └── 20250821/
│           ├── modeling_notes.md
│           ├── outline.md
│           ├── phase_1_2_logical_plan.md
│           ├── project_overview_scope.md
│           └── roadmap.md
│
├── data/                          # Raw and processed data (CSV, Parquet, etc.)
│   ├── raw/                       # Unmodified source 
│   │   └── synthetic/
│   └── processed/                 # Cleaned datasets ready for modeling
│
├── notebooks/                     # Exploratory work (EDA, prototype modeling)
│   ├── 01_eda.ipynb
│   ├── 02_simulation_tests.ipynb
│   └── 03_optimization_tests.ipynb
│
├── app/                           # Optional frontend (Streamlit, Dash)
│   └── main.py
│
├── src/                           # Core package logic (modular and importable)
│   └── hedgeforge/                # Core modules
│       ├── __init__.py
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── markowitz.py        # Mean-variance optimization
│       │   ├── black_litterman.py  # Black-Litterman model
│       │   └── cvar_min.py         # Conditional VaR minimization
│       ├── risk/                   # Risk Modeling and stochastic simulation
│       │   ├── __init__.py
│       │   ├── metrics.py          # Volatility, VaR, CVaR, Sharpe, etc.
│       │   └── monte_carlo.py      # Monte Carlo path simulation engine
│       ├── backtests/              # Logic for testing models
│       │   ├── __init__.py
│       │   └── rolling_window.py   
│       └── utils/                  # Data loading, transformation helpers
│
├── tests/                          # Unit tests for src modules
│   ├── test_optimizer.py
│   ├── test_risk.py
│   └── ...
│
├── docs/                          # Project documentation via Quarto
│   ├── index.qmd                  # Executive summary / overview
│   ├── methodology.qmd            # Stochastic modeling, math background
│   ├── results.qmd                # Output, charts, interpretations
│   └── appendix.qmd               # Extra formulas, derivations, notes
│
├── config/                        # Configuration files for pipeline
│   ├── settings.yaml              # Model parameters, toggles, etc.
│   └── logging.yaml               # Logging settings
│
├── logs/                          # Logging output for debugging / pipeline monitoring
│   └── hedgeforge.log
│
├── scripts/                       # Optional CLI / orchestration scripts
│   └── run_pipeline.py            # Entrypoint to run full modeling pipeline
│
├── pyproject.toml                # Python project metadata (PEP 621)
├── environment.yml               # Conda environment definition (recommended)
├── requirements.txt              # pip fallback for non-conda users
├── README.md                     # Project overview and usage instructions
└── .gitignore                    # Exclude data, logs, virtualenvs, etc.
```

---

## 📊 Features (Planned & In Progress)

- 📈 **Monte Carlo Simulation**  
  Asset path simulation using Geometric Brownian Motion, Ornstein-Uhlenbeck, and other stochastic processes.

- 💡 **Portfolio Optimization**  
  Mean-variance optimization, risk-adjusted return, and utility-based frameworks.

- 📉 **Risk & Performance Analytics**  
  VaR, CVaR, Sharpe Ratio, drawdown, volatility, and custom metrics.

- 🔁 **Backtesting & Statistical Validation**  
  Rolling window backtests to evaluate historical and synthetic strategies.  
  Includes hypothesis testing to assess the statistical significance of strategy performance differences — ensuring improvements aren't due to randomness.

- 🧪 **Stress Testing & Scenario Modeling**  
  Simulate edge cases: interest rate shifts, volatility spikes, correlation breakdowns.  
  Track outcome shifts across economic regimes and use A/B hypothesis testing to evaluate the impact of specific shocks on portfolio utility and drawdown.

- 📓 **Reproducible Docs with Quarto**  
  Full methodology, code explanations, charts, and results.

---

## 🧮 Stochastic Market Simulation

HedgeForge incorporates **stochastic modeling** techniques to realistically simulate market dynamics and evaluate portfolio risk under uncertainty. Using mathematical processes such as **Geometric Brownian Motion (GBM)** and the **Ornstein–Uhlenbeck (OU)** process, the system models both **trend-following** and **mean-reverting** behaviors observed in financial markets.

- **Geometric Brownian Motion (GBM)** captures multiplicative price evolution typical of equities and ETFs, where drift and volatility drive log-normal price distributions.  
- **Ornstein–Uhlenbeck (OU)** dynamics are applied to simulate mean-reverting variables such as interest rates, volatility levels, or asset spreads—useful for stress-testing strategies that depend on market equilibrium.

Through **Monte Carlo simulation**, HedgeForge generates thousands of potential future paths for asset prices and portfolio values. These simulated trajectories feed into the platform’s risk engine, enabling robust estimation of **Value-at-Risk (VaR)**, **Expected Shortfall**, and scenario-based performance metrics. This approach bridges quantitative finance and statistical modeling, providing a foundation for data-driven portfolio optimization and stress testing.

---

## 🎯 Use Case

HedgeForge is designed to mirror the complexity and requirements of institutional portfolio modeling environments. It’s especially aligned with roles in:

- Quantitative Research  
- Asset-Liability Management (ALM)  
- Economic Scenario Generation (ESG modeling)  
- Long-Horizon Portfolio Construction  
- Risk-Aware Financial Planning

---

## 🔧 Development Status

- This project is currently in active development, with the simulation engine and portfolio optimizer under construction.  
- Statistical testing modules (e.g., Sharpe ratio comparisons, t-tests for performance attribution) are also being prototyped to add interpretability and rigor to backtesting workflows.  
- All core modules are being built with production-quality structure and extensibility in mind.

---

## ✍️ Author

**Brice A. Nelson, P.E., MBA**  
Senior Civil Engineer & Data Strategist | Infrastructure Planning | Python, ML, SQL, Capital Forecasting<br> 
[LinkedIn](https://www.linkedin.com/in/brice-a-nelson-p-e-mba-36b28b15/) · [Portfolio](https://www.devbybrice.com) · [Medium](https://medium.com/@quantshift)

---

## 📄 License

This project is open for review but not currently licensed for commercial use.

---

## 📌 Notes

If you're a hiring manager, recruiter, or technical lead — I'm happy to walk through the architecture, modeling logic, or long-term project roadmap during an interview or follow-up. Thank you for your interest!

