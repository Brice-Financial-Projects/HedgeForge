# 📈 Constrained Portfolio Optimization System

**A Python-based framework for real-world portfolio construction under regulatory and operational constraints, featuring rolling backtests and multi-model optimization engines.**

---

## 🚀 Project Overview

This project aims to deliver a professional-grade, extensible system for portfolio optimization tailored to realistic financial environments. It goes far beyond academic examples, incorporating constraints, market volatility, and compliance considerations seen in institutional asset management and fintech products.

Key goals include:
- Portfolio construction using modern optimization techniques
- Integration of real-world constraints (e.g., sector caps, turnover limits, leverage restrictions)
- Risk-aware modeling and forecast analysis
- Rolling backtests with transaction costs and stress scenarios
- Full technical documentation via Quarto

---

## 🧠 Core Features

### 1. Return Forecasting
- Historical mean returns (static and rolling)
- Optional machine learning models (linear regression, Ridge, Lasso, LightGBM)
- Scenario-based forecasts or shrinkage estimators
- Rolling window return estimation for dynamic strategy updates

### 2. Risk Modeling
- Volatility and correlation matrix construction
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Rolling Sharpe Ratio and drawdown metrics
- Beta estimation and benchmark correlation
- Factor model support (optional extension)

### 3. Optimization Engines
- **Mean-Variance Optimization** (Markowitz)
- **Black-Litterman model** for subjective views blending
- **CVaR Minimization** via `cvxpy`
- Constraint support:
  - Long-only or long-short
  - Sector exposure limits
  - Minimum/maximum position sizing
  - Leverage caps and turnover constraints
  - Regulatory frameworks (UCITS, 40 Act)

---

## 📉 Backtesting System

A robust backtesting module is included to evaluate strategies over time.

Features:
- Rolling window backtests (e.g., 3-year lookback, monthly rebalance)
- Realistic slippage and transaction cost modeling
- Benchmark-relative performance comparisons (e.g., SPY, 60/40)
- Key metrics:
  - Sharpe, Sortino, Max Drawdown
  - Rolling volatility, rolling correlation
  - Turnover and risk-adjusted returns
- Visualization of equity curves and drawdowns

---

## 📘 Documentation with Quarto

All development, methodology, and results are documented using [**Quarto**](https://quarto.org/), a professional publishing framework for reproducible analysis.

### 📄 Quarto Structure:
- `index.qmd`: Executive summary and project intro
- `methodology.qmd`: Optimization models, risk metrics, math/assumptions
- `results.qmd`: Backtest visuals, metrics, constraint outcomes
- `appendix.qmd`: System architecture, CLI usage, dataset details

**Output formats**: PDF, HTML, GitHub Pages

---

## 🧪 Testing

Includes unit tests for:
- Optimization output validity
- Constraint enforcement logic
- Rolling window integrity
- Deterministic results using seeded runs

---

## 🧠 Stretch Goals

- Integrate Bayesian return estimators or macroeconomic covariates
- Add regime-switching logic using HMM or GARCH models
- Deploy as a service (API or Streamlit app) for dynamic user interaction
- Introduce ESG scoring or tax-aware optimization

---

## 📦 Deliverables

- 📂 Full GitHub repository with clean, modular code
- 📘 Quarto-generated documentation in PDF and HTML
- 🌐 (Optional) Streamlit or Dash application for public demonstration
- ✍️ Medium post or technical whitepaper documenting the development process

---

## 🧭 Status

| Milestone | Status |
|-----------|--------|
| Project planning and structure | ✅ Complete |
| Data ingestion and cleaning | 🚧 In progress |
| Optimization models implementation | 🔜 Next milestone |
| Backtesting module | 🔜 Upcoming |
| Full Quarto documentation | ⏳ In progress |
