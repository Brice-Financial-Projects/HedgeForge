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

---

## 📊 Phase 2 — Data & Utility Layer
**Goal:** Build robust data ingestion and preprocessing utilities.

- [x] Implement `load_data()`, `validate_data()`, and `compute_log_returns()` in `src/utils.py`
- [x] Add sample CSV files to `/data/raw/`
- [ ] Test pipeline read → clean → save to `/data/processed/`
- [ ] Write one validation notebook (`notebooks/01_eda.ipynb`)
- [ ] Log data-pipeline runs to `/logs/`

---

## 🧮 Phase 3 — Core Modeling Foundations
**Goal:** Establish the base quantitative calculations.

- [ ] Implement functions in `src/risk.py`  
  - [ ] Volatility  
  - [ ] Covariance matrix  
  - [ ] Correlation  
  - [ ] VaR / CVaR
- [ ] Add tests for numerical consistency using synthetic data
- [ ] Document formulas in `docs/03_modeling/math_foundations.md`

---

## 📈 Phase 4 — Optimization Engine
**Goal:** Create the mathematical heart of HedgeForge.

- [ ] Implement Markowitz mean-variance optimization in `src/optimizer.py`
- [ ] Add max-Sharpe and CVaR objectives
- [ ] Handle constraints in `src/constraints.py`
- [ ] Test with small synthetic portfolios
- [ ] Visualize results in `notebooks/03_optimization_tests.ipynb`

---

## 🔁 Phase 5 — Backtesting & Validation
**Goal:** Enable evaluation of model performance over time.

- [ ] Implement walk-forward testing in `src/backtest.py`
- [ ] Calculate portfolio metrics (returns, Sharpe, drawdown)
- [ ] Compare against benchmarks (SPY, 60/40)
- [ ] Store backtest results in `/data/processed/`
- [ ] Document approach in `docs/03_modeling/backtesting_framework.md`

---

## 🧠 Phase 6 — Forecasting Module (Optional)
**Goal:** Introduce predictive capability.

- [ ] Implement regression/momentum models in `src/forecasting.py`
- [ ] Integrate forecasts into optimizer workflow
- [ ] Validate predictive accuracy and log results

---

## 🧩 Phase 7 — Integration & Orchestration
**Goal:** Connect all modules into a cohesive execution pipeline.

- [ ] Expand `scripts/run_pipeline.py` to run end-to-end
- [ ] Add command-line arguments via `argparse` or `typer`
- [ ] Implement structured logging across all modules
- [ ] Test full run with sample data

---

## 🧱 Phase 8 — Documentation & Visualization
**Goal:** Produce professional-grade internal and external documentation.

- [ ] Populate all `docs/` subfolders (overview, modeling, engineering)
- [ ] Finalize Quarto documents under `/quarto/`
- [ ] Create simple Streamlit or Dash demo in `/app/main.py`
- [ ] Add generated charts and summary tables to `results.qmd`

---

## ☁️ Phase 9 — Deployment & Packaging
**Goal:** Make HedgeForge reproducible and deployable.

- [ ] Write Dockerfile for full pipeline
- [ ] Extend GitHub Actions to build + push Docker image
- [ ] Configure AWS App Runner / RDS / S3 connections
- [ ] Verify environment reproducibility via `environment.yml`
- [ ] Add `.gitignore` rules for logs, data, and environments

---

## 🎯 Phase 10 — Polish & Showcase
**Goal:** Final refinement and presentation.

- [ ] Update top-level `README.md` with badges, diagrams, and summary
- [ ] Write executive Quarto report (`quarto/index.qmd`)
- [ ] Create summary notebook for demos and Medium article
- [ ] Reflect on lessons in `docs/05_dev_notes/diary_logs/`

---

**Branch:** `dev_no_ai`  
**Focus:** Manual rebuild emphasizing correctness, reproducibility, and learning through full-stack quantitative design.
