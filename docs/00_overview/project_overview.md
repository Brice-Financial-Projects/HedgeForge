# HedgeForge — Project Documentation

---

## 1. Executive Overview

HedgeForge is a Python-based platform for **portfolio optimization and risk modeling** under uncertainty.
It blends **quantitative finance techniques** (Monte Carlo simulations, stochastic models, optimization) with **modern backend engineering** (FastAPI, PostgreSQL, AWS).

**Purpose:**
- Provide analysts, fintech platforms, and investment teams with a professional-grade risk analytics system.
- Serve as a **portfolio centerpiece project** demonstrating advanced finance, data science, and backend engineering skills.

**Core Value:**
- Makes sophisticated risk analytics accessible through APIs.
- Bridges the gap between **quantitative research** and **real-world fintech systems**.
- Flexible and extensible for evolving markets, asset classes, and client needs.

---

## 2. What the App Will Do

- Run simulations of market and macroeconomic scenarios.
- Conduct portfolio stress testing under shocks (rates, inflation, equity drawdowns).
- Calculate key metrics: VaR, CVaR, Sharpe, drawdowns, tracking error.
- Support portfolio construction with optimization engines.
- Enable rolling backtests with transaction costs and benchmarks.
- Provide outputs via API or optional visualization layers.

---

## 3. Roadmap & Phases

HedgeForge is built methodically across ten phases:

1. **Project Charter & Math Foundation**
   - Define objectives (min variance, max Sharpe, ESG).
   - Set constraints (weights, sector caps, no shorting).
   - Document risk metrics (volatility, VaR, CVaR).

2. **Data Engineering Pipeline**
   - Load synthetic/real data.
   - Validate and clean.
   - Compute returns, covariance matrices, rolling metrics.

3. **Exploratory Data Analysis (EDA)**
   - Visualize distributions, correlations, volatility clusters.

4. **Optimization Engine**
   - Implement mean-variance, Sharpe max, risk parity.
   - Enforce constraints (sector caps, weights).

5. **Risk Metrics Module**
   - VaR, CVaR, rolling volatility, tracking error.

6. **Forecasting Module (Optional)**
   - Rolling averages, regressions, tree-based models.
   - Momentum and macro features.

7. **Backtesting Framework**
   - Walk-forward tests.
   - Include slippage, transaction costs.
   - Compare vs benchmarks.

8. **UI / API Layer**
   - **Streamlit**: interactive demos and charts.
   - **FastAPI**: scalable backend endpoints.

9. **Documentation & Presentation**
   - Quarto docs: executive summary, methodology, results, appendix.

10. **Showcase Polish**
    - Code formatting (Black, flake8, mypy).
    - Structured logging.
    - Dockerization and deployment.

---

## 4. System Outline

### 4.1 Return Forecasting
- Historical and rolling returns.
- ML models: regression, Ridge, Lasso, LightGBM.
- Scenario-based forecasts, shrinkage estimators.

### 4.2 Risk Modeling
- Volatility, correlation, beta estimation.
- VaR, CVaR, Sharpe, drawdowns.
- Factor models and benchmark analysis.

### 4.3 Optimization Engines
- **Mean-Variance Optimization** (Markowitz).
- **Black-Litterman model**.
- **CVaR minimization**.
- Support for regulatory and operational constraints:
  - Sector caps
  - Turnover & leverage limits
  - Long-only or long-short

### 4.4 Backtesting
- Rolling windows with rebalance.
- Slippage and transaction cost modeling.
- Relative benchmarks (SPY, 60/40).
- Metrics: Sharpe, Sortino, max drawdown, turnover.
- Equity curves and rolling stats.

---

## 5. Modeling Foundations

### 5.1 Returns
- Log vs simple returns.
- Handling missing values and outliers.
- Frequency choices (daily, monthly).

### 5.2 Risk Metrics
- Volatility, VaR, CVaR, tracking error.
- Rationale for chosen measures.

### 5.3 Optimization Objectives
- Min variance, max Sharpe, risk parity.
- Extendable to ESG or custom constraints.

### 5.4 Constraints
- Weight caps, no shorting, leverage restrictions.
- Sector and regulatory limits.

### 5.5 Computational Considerations
- Libraries: `cvxpy`, `scipy.optimize`.
- Covariance stability and numerical conditioning.

### 5.6 Assumptions & Limitations
- Historical data as proxy for future.
- Simplifications of market frictions.
- Risk metrics may underperform in crises.

---

## 6. Deployment & Infrastructure

- **Containerization:** Docker for reproducibility.
- **Hosting:** AWS App Runner (backend service).
- **Database:** PostgreSQL on AWS RDS.
- **Storage:** S3 for large outputs.
- **Monitoring:** CloudWatch for logs and alerts.
- **Secrets:** AWS Secrets Manager.
- **Frontends:** Streamlit (demo), FastAPI (API).

---

## 7. Documentation & Deliverables

- **Quarto Documentation**
  - `index.qmd`: executive summary
  - `methodology.qmd`: math and models
  - `results.qmd`: charts, backtests
  - `appendix.qmd`: derivations, system design

- **GitHub Repository**
  - Clean, modular code
  - Utilities in `src/`
  - Unit tests for risk, optimization, backtests

- **Optional Outputs**
  - Streamlit/Dash demo app
  - Medium article or whitepaper

---

## 8. Status & Next Steps

- **Planning & structure:** ✅ Complete
- **Data ingestion:** 🚧 In progress
- **Optimization models:** 🔜 Next milestone
- **Backtesting:** ⏳ Upcoming
- **Documentation:** Ongoing

**Next immediate priorities:**
- Finalize Phase 1–2 deliverables (math notes + data pipeline).
- Implement optimizer core.
- Begin Quarto documentation.
- Decide on frontend (Streamlit vs FastAPI vs both).

---

## Appendix: Phase 1–2 Detailed Logic

**Phase 1:**
- Define objectives, constraints, and assumptions.
- Write formulas for optimization (mean-variance, Sharpe).
- Document deliverables in `roadmap.md`, `modeling_notes.md`.

**Phase 2:**
- Build utilities:
  - `load_data()`
  - `validate_data()`
  - `compute_log_returns()`
  - `compute_covariance_matrix()`
  - `compute_rolling_metrics()`
- Orchestrate pipeline: Load → Clean → Transform → Save.

Deliverables:
- Clean validated data, returns, covariance matrices.
- Ready inputs for optimization and EDA.
