# HedgeForge

HedgeForge is a Python-based platform for **portfolio optimization and risk modeling**.
It combines **quantitative finance techniques** (Monte Carlo, VaR/CVaR, Sharpe, drawdowns) with **backend engineering** (FastAPI, PostgreSQL, AWS) to deliver professional-grade analytics.

---

## 🚀 What It Does
- Run scenario simulations and stress tests
- Construct portfolios under real-world constraints (sector caps, turnover, leverage)
- Calculate advanced risk metrics (VaR, CVaR, tracking error)
- Perform rolling backtests with transaction costs
- Provide results via REST API or optional Streamlit dashboard

---

## 🔧 System Components
- **Return Forecasting**: rolling statistics, regression, ML models
- **Risk Modeling**: covariance matrices, volatility, VaR/CVaR
- **Optimization Engines**: Markowitz, Black-Litterman, CVaR minimization
- **Backtesting**: walk-forward tests, benchmark comparisons
- **Deployment**: Dockerized, hosted on AWS (App Runner + RDS + S3)

---

## 📑 Documentation
- `docs/index.qmd` — Executive summary
- `docs/methodology.qmd` — Math & assumptions
- `docs/results.qmd` — Backtests, charts
- `docs/appendix.qmd` — Derivations, system design

---

## 📈 Status
- ✅ Planning complete
- 🚧 Data ingestion in progress
- 🔜 Optimization models next
- ⏳ Backtesting & documentation upcoming

---

## 🎯 Outcome
HedgeForge is designed as a **showcase fintech project**:
- Demonstrates advanced quantitative finance
- Professional backend engineering practices
- Portfolio-ready for recruiters and collaborators
