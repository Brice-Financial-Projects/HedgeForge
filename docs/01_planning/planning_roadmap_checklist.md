# 🧭 HedgeForge Development Roadmap — dev_no_ai Branch

A structured, high-level plan for manually rebuilding HedgeForge with clarity, discipline, and strong backend fundamentals.

---

## ✅ Phase 1 — Foundation Setup (Environment + CI/CD + Core Skeleton)
**Goal:** Establish a clean, reproducible foundation for all future phases.

- [X] Verify and finalize project folder structure (`src/`, `docs/`, `quarto/`, etc.)
- [X] Create and activate Conda/uv environment
  - [X] Confirm imports (`hedge_forge`)
- [X] Implement basic `src/hedge_forge/__init__.py`
- [X] Add centralized logging setup in `src/hedge_forge/utils/logger.py`
- [X] Configure **pytest** and add one dummy test in `/tests`
- [X] Create `.pre-commit-config.yaml`
  - [X] Include Ruff, Black, isort, and YAML/JSON linting hooks
- [x] Configure **GitHub Actions CI pipeline**
  - [x] Lint → Test → Build stages
  - [ ] Cache dependencies for faster runs
- [x] Confirm logs write correctly to `/logs/hedgeforge.log`
- [ ] Align developer docs (`README.md`, `SETUP.md`) with the current `uv`-based workflow and the actual package layout
- [ ] Add a first-time contributor bootstrap checklist (clone → `uv sync` → `pytest` → `ruff`)

---

## 📊 Phase 2 — Data & Utility Layer
**Goal:** Build robust data ingestion and preprocessing utilities.

- [x] Implement `load_data()`, `validate_data()`, and `compute_log_returns()` in `src/hedge_forge/utils/utils.py`
- [x] Add sample CSV files to `/data/raw/`
- [ ] Test pipeline read → clean → save to `/data/processed/`
- [ ] Define a data contract/schema for raw and processed datasets (required columns, dtypes, timestamp format, missing-value rules)
- [ ] Implement a reusable data-cleaning stage for malformed portfolio CSVs (currency/number parsing, column alignment) and write cleaned outputs to `/data/processed/`
- [ ] Add a single pipeline orchestration flow for `load → validate → clean → transform → save` with structured logging/provenance
- [ ] Route data input/output paths and log destinations through the config loader so the pipeline is environment-aware
- [ ] Write one validation notebook (`notebooks/01_eda.ipynb`)
- [ ] Log data-pipeline runs to `/logs/`

---

## 🧮 Phase 3 — Core Modeling Foundations
**Goal:** Establish the base quantitative calculations.

- [ ] Implement the risk analytics package under `src/hedge_forge/risk/` with volatility, covariance, correlation, VaR, and CVaR functions
- [ ] Add public exports from `src/hedge_forge/risk/__init__.py`
- [ ] Add tests for numerical consistency using synthetic data
- [ ] Document formulas and assumptions in `docs/03_modeling/math_foundations.md`
- [ ] Define a stable input/output schema for risk metrics so downstream modules can reuse them consistently

---

## 📈 Phase 4 — Optimization Engine
**Goal:** Create the mathematical heart of HedgeForge.

- [ ] Implement Markowitz mean-variance optimization in `src/hedge_forge/optimization/`
- [ ] Add max-Sharpe and CVaR objectives
- [ ] Handle constraints in `src/hedge_forge/optimization/` (including long-only and weight-budget constraints)
- [ ] Add tests with small synthetic portfolios and constraint scenarios
- [ ] Visualize results in `notebooks/03_optimization_tests.ipynb`
- [ ] Define a reusable optimizer API that accepts market inputs and returns weights plus diagnostics

---

## 🔁 Phase 5 — Backtesting & Validation
**Goal:** Enable evaluation of model performance over time.

- [ ] Implement walk-forward testing in `src/hedge_forge/backtests/`
- [ ] Calculate portfolio metrics (returns, Sharpe, drawdown)
- [ ] Compare against benchmarks (SPY, 60/40)
- [ ] Store backtest results in `/data/processed/`
- [ ] Document approach in `docs/03_modeling/backtesting_framework.md`
- [ ] Add transaction-cost and rebalance assumptions to the backtest configuration
- [ ] Define a result schema for backtest outputs (metrics, trades, and benchmark comparison)

---

## 🧠 Phase 6 — Forecasting Module (Optional)
**Goal:** Introduce predictive capability.

- [ ] Create a forecasting module under `src/hedge_forge/` or `src/hedge_forge/strategies/`
- [ ] Implement regression and momentum-based forecasting models
- [ ] Integrate forecasts into the optimizer and backtest workflow
- [ ] Validate predictive accuracy and log results
- [ ] Add data validation for forecasting inputs and targets

---

## 🧩 Phase 7 — Integration & Orchestration
**Goal:** Connect all modules into a cohesive execution pipeline.

- [ ] Create a pipeline entry point in `src/hedge_forge/scripts/` or `scripts/` to run the full workflow end-to-end
- [ ] Add command-line arguments via `argparse` or `typer`
- [ ] Implement structured logging across all modules
- [ ] Add a package-level runner such as `python -m hedge_forge`
- [ ] Test full run with sample data

---

## 🧱 Phase 8 — Documentation & Visualization
**Goal:** Produce professional-grade internal and external documentation.

- [ ] Populate all `docs/` subfolders (overview, modeling, engineering)
- [ ] Finalize Quarto documents under `/quarto/`
- [ ] Create a simple Streamlit or Dash demo in `/app/main.py`
- [ ] Add generated charts and summary tables to `results.qmd`
- [ ] Add engineering docs for config, deployment, and data contracts under `docs/04_engineering/` and `docs/06_references/`

---

## ☁️ Phase 9 — Deployment & Packaging
**Goal:** Make HedgeForge reproducible and deployable.

- [ ] Write a Dockerfile for the full pipeline
- [ ] Extend GitHub Actions to build + push a Docker image
- [ ] Configure deployment targets and secrets handling (for example AWS App Runner / RDS / S3)
- [ ] Verify environment reproducibility via `environment.yml` or the `uv` lockfile plus setup documentation
- [ ] Add `.gitignore` rules for logs, data, and environments

---

## 🎯 Phase 10 — Polish & Showcase
**Goal:** Final refinement and presentation.

- [ ] Update top-level `README.md` with badges, diagrams, and summary
- [ ] Write executive Quarto report (`quarto/index.qmd`)
- [ ] Create a summary notebook for demos and Medium article
- [ ] Reflect on lessons in `docs/05_dev_notes/diary_logs/`
- [ ] Add a lightweight showcase command or notebook that runs the end-to-end pipeline on sample data
