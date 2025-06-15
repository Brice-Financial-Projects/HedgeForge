# Project Structure

constrained-portfolio-optimization-system/
│
├── data/                    # Raw data files, cleaned CSVs, etc.
├── notebooks/               # Jupyter prototypes (initial models, EDA, visualizations)
├── app/                     # Streamlit or Dash frontend
│   └── main.py
├── src/                     # Core modules and reusable code
│   ├── optimizer.py         # Optimization logic (MVO, CVaR, Black-Litterman)
│   ├── constraints.py       # Constraint manager
│   ├── risk.py              # Risk metrics and calculators
│   ├── forecasting.py       # Return prediction models (basic + optional ML)
│   ├── backtest.py          # Rolling backtester and performance evaluation
│   └── utils.py             # Helper functions (data cleaning, loaders, etc.)
├── docs/                    # Quarto source for full documentation
│   ├── index.qmd
│   ├── methodology.qmd
│   ├── results.qmd
│   └── appendix.qmd
├── tests/                   # Unit tests
├── requirements.txt
├── README.md
└── .gitignore
