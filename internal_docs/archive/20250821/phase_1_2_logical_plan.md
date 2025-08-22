# Hedge-Forge — Logical Breakdown for Phase 1 & 2

*No code yet — just methodical logic and planning!*

---

## 🚀 Phase 1 — Project Charter & Math Foundation

### 🎯 Big Picture

Phase 1 is all about defining:

- **What problem you’re solving** (e.g. minimize risk, maximize Sharpe ratio)
- **What math underpins your problem**
- **Your constraints and assumptions**

This phase is purely about clarity and design — nothing should surprise you later.

---

### 1. Define Your Business Problem

✅ *Questions to answer:*

- What do you want your portfolio optimizer to achieve?
  - Lower risk?
  - Maximize returns for a specific risk?
  - Target ESG scores?
  - Reduce transaction costs?

- Are there specific constraints you want?
  - No shorting
  - Sector limits
  - Max weight limits for single positions

✅ *Deliverables:*

- Write these decisions into:
  - `internal_docs/roadmap.md`
  - `internal_docs/modeling_notes.md`

These are the “project spec.”

---

### 2. Mathematical Formulations

This part is critical for future implementation. You’re not coding yet — you’re writing formulas, derivations, and how you’ll solve them.

- **Mean-variance optimization**:
  - Objective: minimize portfolio variance, or maximize Sharpe ratio

- **Constraints**:
  - sum(weights) = 1
  - weights ≥ 0 (if no shorting)
  - sector caps (e.g. Tech ≤ 20%)

- **Risk metrics**:
  - portfolio volatility
  - Value at Risk (VaR)
  - Conditional VaR (CVaR)

✅ *Deliverables:*

- Document formulas and assumptions in:
  - `docs/methodology.qmd`

---

### 3. Planned Inputs

Decide:

- What data will be needed?
  - Prices?
  - Returns?
  - Covariance matrix?
- How many assets will be tested on initially?
  - Start small (e.g. 5–10 tickers) to avoid overwhelm.

✅ *Deliverables:*

- Notes about data structure and expectations.

---

### 🔧 Utilities to Plan (Phase 1)

Not implemented yet, but start identifying what will be needed:

| Utility                    | Purpose                                           |
|----------------------------|---------------------------------------------------|
| Data Loader                | Read price data from CSV, API, or synthetic data. |
| Return Calculator          | Transform prices → returns.                       |
| Covariance Calculator      | Compute asset covariances for optimization.       |
| Risk Metrics Calculator    | Compute volatility, VaR, CVaR.                    |
| Constraints Handler        | Check weight allocations obey constraints.        |

Start a list of these for Phase 2!

---

## 🧪 Phase 2 — Data Engineering Pipeline

Phase 2 moves from “math on paper” to **building the data pipeline.** Still no modeling or optimization — just data wrangling and prepping inputs.

Goal:

- Load data
- Clean and validate it
- Transform it into returns, covariance matrices, etc.

Think of this phase as the prepping kitchen before cooking.

---

### 1. Data Loading

Decide the data source:

- CSV files
- Yahoo Finance API
- Synthetic generator

✅ *Logical tasks:*

- File reading:
  - Check columns: ticker, date, price
  - Ensure dates are sorted chronologically
- Validate:
  - No duplicate rows
  - Reasonable price values (e.g. no -9999s)

✅ *Planned utility function:*

- **load_data()**
  - Input → file path or API request
  - Output → clean DataFrame

---

### 2. Data Validation

Key step so garbage doesn’t sneak into the calculations.

✅ *Logical checks:*

- Missing values
- Data types (dates as datetime, prices as numeric)
- Outlier detection:
  - Extreme returns
  - Sudden price spikes/drops

✅ *Planned utility function:*

- **validate_data()**
  - Input → DataFrame
  - Output → issues found or confirmation of clean data

---

### 3. Return Calculations

Prices mean nothing in optimization. You need returns:

- Log returns → preferred for modeling stability
- Simple returns → sometimes used for readability

✅ *Logical steps:*

- Calculate daily log returns
- Drop rows with NaNs from differencing

✅ *Planned utility function:*

- **compute_log_returns()**
  - Input → price DataFrame
  - Output → returns DataFrame

---

### 4. Covariance Matrix Calculation

Optimization relies on the covariance matrix:

$$
\Sigma = \text{Cov}(r)
$$

✅ *Logical steps:*

- Calculate covariance over:
  - full sample
  - rolling windows (e.g. 60-day)

✅ *Planned utility function:*

- **compute_covariance_matrix()**
  - Input → returns DataFrame
  - Output → covariance matrix DataFrame

---

### 5. Rolling Metrics

Nice for EDA and future analysis:

- Rolling volatility
- Rolling correlations

✅ *Logical steps:*

- Decide window size (e.g. 30, 60 days)
- Compute rolling metrics for each asset

✅ *Planned utility function:*

- **compute_rolling_metrics()**
  - Input → returns DataFrame
  - Output → DataFrame of rolling statistics

---

### 6. Data Pipeline Orchestration

Tie all those utilities together:

- load_data
- validate_data
- compute_log_returns
- compute_covariance_matrix
- compute_rolling_metrics

✅ *Logical deliverable:*

- A single pipeline function that calls each piece in sequence.

Think:

> Load → Clean → Transform → Save/Return

---

## ✅ Phase 1–2 Deliverables

### Docs:

- `internal_docs/roadmap.md`
- `internal_docs/modeling_notes.md`
- `docs/methodology.qmd`

---

### Planned Utilities (Phase 2):

- load_data()
- validate_data()
- compute_log_returns()
- compute_covariance_matrix()
- compute_rolling_metrics()

These will eventually live in:

```
src/utils.py
```

---

## 🤹‍♂️ Logical Flow

Here’s how everything connects:

```
Math docs
     ↓
Decide data structure
     ↓
load_data()
     ↓
validate_data()
     ↓
compute_log_returns()
     ↓
compute_covariance_matrix()
     ↓
compute_rolling_metrics()
     ↓
READY FOR PHASE 3 (EDA)
```

---

At the end of Phase 2, items produced:

- Clean, validated data
- Returns data
- Covariance matrices
- Confidence in your inputs for optimization

…and most importantly, **a professional habit of writing things down and thinking methodically.**

---

## 🎯 Next Logical Steps

✅ Flesh out `modeling_notes.md`:

- Math formulas
- Constraints
- Data shapes you expect

✅ List out each utility to build, with:

- Inputs
- Outputs
- Purpose
