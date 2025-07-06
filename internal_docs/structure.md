# Project Structure

hedge_forge/
│
├── internal_docs/                 # Informal or working documentation (Markdown)
│   ├── structure.md               # Project structure overview (like this!)
│   ├── roadmap.md                 # Features, timeline, stretch goals
│   ├── modeling_notes.md          # Math, formula derivations, drafts
│   └── references.md              # Links, citations, papers, articles
│
├── data/                          # Raw and processed data (CSV, Parquet, etc.)
│   ├── raw/                       # Unmodified source data
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
│   ├── __init__.py
│   ├── optimizer.py               # Portfolio optimization algorithms
│   ├── constraints.py             # Constraint handling and validation
│   ├── risk.py                    # Risk metrics (VaR, CVaR, volatility, etc.)
│   ├── forecasting.py             # Return forecasting (ML or statistical models)
│   ├── backtest.py                # Rolling backtests, evaluation metrics
│   └── utils.py                   # Data loading, transformation helpers
│
├── tests/                         # Unit tests for src modules
│   ├── test_optimizer.py
│   ├── test_risk.py
│   └── ...
│
├── docs/                          # Project documentation via Quarto
│   ├── index.qmd                  # Executive summary / overview
│   ├── methodology.qmd            # Stochastic modeling, math background
│   ├── data_pipeline.qmd
│   ├── eda.qmd
│   ├── optimization_results.qmd   # Output, charts, interpretations
│   ├── risk_metrics.qmd
│   ├── appendix.qmd               # Extra formulas, derivations, notes
│   └── _quarto.yml
│
├── config/                        # Configuration files for pipeline
│   └── settings.yaml              # Model parameters, toggles, etc.
│
├── logs/                          # Logging output for debugging / pipeline monitoring
│   └── hedgeforge.log
│
├── scripts/                       # Optional CLI / orchestration scripts
│   └── run_pipeline.py            # Entrypoint to run full modeling pipeline
│
├── environment.yml               # Conda environment definition (recommended)
├── requirements.txt              # pip fallback for non-conda users
├── README.md                     # Project overview and usage instructions
└── .gitignore                    # Exclude data, logs, virtualenvs, etc.
