# Project Structure

```plaintext
hedge_forge/
│
├── quarto/                            # Public-facing Quarto documentation for reports and presentation
│   ├── index.qmd                      # Executive summary of the project
│   ├── methodology.qmd                # Mathematical methods and model assumptions
│   ├── data_pipeline.qmd              # Data ingestion and transformation pipeline
│   ├── results.qmd                    # Backtest results and performance charts
│   ├── appendix.qmd                   # Additional derivations and technical notes
│   └── _quarto.yml                    # Quarto site configuration file
│
├── docs/                              # Internal developer documentation and technical notes
│   ├── 00_overview/                   # High-level summaries and architecture context
│   │   ├── project_overview.md        # Overall description and goals of HedgeForge
│   │   ├── architecture_summary.md    # Summary of the core system and src structure
│   │   ├── design_philosophy.md       # Explanation of manual rebuild and design mindset
│   │   ├── structure_map_legacy.md    # Legacy structure reference for comparison
│   │   ├── project_overview_README_ref.md # Condensed executive overview used in README
│   │   └── _index.md                  # Optional local index for overview documents
│   │
│   ├── 01_planning/                   # Roadmaps, goals, and planning documentation
│   │   ├── roadmap.md                 # Development phases and overall roadmap
│   │   ├── milestones.md              # Key deliverables and progress checkpoints
│   │   ├── feature_tracking.md        # Table tracking features and implementation status
│   │   └── glossary.md                # Definitions of core financial and technical terms
│   │
│   ├── 02_architecture/               # Technical system architecture and data flow
│   │   ├── system_diagram.md          # Visual or textual system architecture diagram
│   │   ├── data_flow.md               # Description of end-to-end data movement
│   │   ├── config_strategy.md         # Overview of configuration management approach
│   │   ├── deployment_topology.md     # Deployment and hosting architecture on AWS
│   │   └── module_dependency_map.md   # Relationships between modules in src/
│   │
│   ├── 03_modeling/                   # Quantitative and mathematical model documentation
│   │   ├── math_foundations.md        # Core equations and mathematical derivations
│   │   ├── risk_metrics.md            # Explanation of VaR, CVaR, Sharpe, and related measures
│   │   ├── portfolio_construction.md  # Portfolio construction principles and constraints
│   │   ├── optimization_algorithms.md # Portfolio optimization techniques and objectives
│   │   ├── trading_strategies.md         # Overview of implemented trading strategies
│   │   ├── forecasting_models.md      # Forecasting methods and predictive model notes
│   │   └── backtesting_framework.md   # Rolling tests, evaluation, and benchmark comparisons
│   │
│   ├── 04_engineering/                # Software engineering and DevOps documentation
│   │   ├── coding_standards.md        # Naming, typing, and formatting conventions
│   │   ├── testing_strategy.md        # Unit test structure and coverage approach
│   │   ├── logging_and_monitoring.md  # Logging design and runtime monitoring details
│   │   ├── performance_and_scalability.md # Performance tuning and scalability notes
│   │   └── ci_cd_pipeline.md          # CI/CD pipeline structure and GitHub Actions overview
│   │
│   ├── 05_dev_notes/                  # Active development notes and ongoing logs
│   │   ├── diary_logs/                # Day-by-day or topic-specific developer logs
│   │   │   ├── 01_setup.md            # Environment setup notes
│   │   │   ├── 02_data_pipeline_dev.md# Development log for data pipeline
│   │   │   ├── 03_optimizer_dev.md    # Development log for optimization module
│   │   │   └── 04_backtesting_dev.md  # Development log for backtesting features
│   │   ├── scratchpad.md              # General notepad for quick ideas or code snippets
│   │   └── experiments.md             # Notes on modeling experiments or parameter tuning
│   │
│   └── 06_references/                 # Research papers, citations, and formula references
│       ├── papers_and_links.md        # List of related papers, articles, and links
│       ├── bibliography.bib           # BibTeX reference file for Quarto or citations
│       ├── formula_reference.md       # Quick reference for all mathematical formulas
│       └── resources.md               # Additional reference materials and external tools
│
├── src/                               # Core Python package source code
│   └── hedge_forge/                   # Main package namespace
│       ├── __init__.py                # Makes hedge_forge a package
│       │
│       ├── backtests/                 # Backtesting engines and result analyzers
│       │   ├── __init__.py
│       │   ├── rolling_backtest.py    # Walk-forward and rolling window backtests
│       │   ├── performance_metrics.py # Sharpe, Sortino, drawdowns, turnover
│       │   └── benchmark_compare.py   # Benchmark-relative analytics and reporting
│       │
│       ├── optimization/              # Portfolio construction and optimization logic
│       │   ├── __init__.py
│       │   ├── mean_variance.py       # Classic Markowitz and efficient frontier solver
│       │   ├── black_litterman.py     # Black-Litterman Bayesian blending model
│       │   ├── cvar_minimization.py   # Conditional VaR (CVaR) optimization routines
│       │   └── constraints.py         # Constraint definitions (sector caps, turnover, etc.)
│       │
│       ├── risk/                      # Risk modeling and statistical analytics
│       │   ├── __init__.py
│       │   ├── var_cvar.py            # Value-at-Risk and Conditional VaR calculations
│       │   ├── volatility.py          # Rolling and annualized volatility computations
│       │   ├── correlation.py         # Covariance and correlation matrix generation
│       │   └── stress_tests.py        # Scenario and stress test simulation utilities
│       │
│       ├── strategies/                # Trading and investment strategies
│       │   ├── __init__.py
│       │   ├── base.py                 # Base class for trading and investment strategies
│       │   ├── momentum.py             # Momentum-based trading strategies
│       │   ├── mean_reversion.py       # Mean reversion trading strategies
│       │   ├── trend_following.py      # Trend-following trading strategies
│       │   ├── pairs_trading.py        # Pairs trading strategy implementation
│       │   ├── signal_generation.py    # Signal generation and entry/exit logic for strategies
│       │   └── position_sizing.py      # Position sizing and risk management utilities
│       │
│       │
│       ├── utils/                     # Shared utility modules and helpers
│       │   ├── __init__.py
│       │   ├── config_loader.py       # YAML + environment variable configuration loader
│       │   ├── logger.py              # Structured logging setup and configuration
│       │   ├── file_io.py             # File load/save helpers for CSV, Parquet, etc.
│       │   ├── math_tools.py          # Common math operations (vectorized helpers)
│       │   └── decorators.py          # Timing, caching, and exception-handling decorators
│       │
│       └── forecasting/               # (Optional) predictive and ML-based forecasting models
│           ├── __init__.py
│           ├── regression_models.py   # Linear, Ridge, Lasso regression forecasters
│           ├── ml_models.py           # Tree-based or ensemble forecasting models
│           └── feature_engineering.py # Factor construction and lag feature utilities
│
├── data/                              # Local storage for input and output data
│   ├── raw/                           # Unmodified source data files
│   └── processed/                     # Cleaned and transformed datasets
│
├── notebooks/                         # Jupyter notebooks for exploratory and prototype work
│   ├── 01_eda.ipynb                   # Exploratory data analysis notebook
│   ├── 02_simulation_tests.ipynb      # Monte Carlo and stochastic simulation experiments
│   └── 03_optimization_tests.ipynb    # Portfolio optimization and validation tests
│
├── app/                               # Optional visualization or demo layer
│   └── main.py                        # Streamlit or Dash application entry point
│
├── scripts/                           # Utility and orchestration scripts
│   ├── run_pipeline.py                # Script to execute the full modeling pipeline
│   └── seed_data.py                   # Populate synthetic or sample datasets
│
├── tests/                             # Automated test suite for all modules
│   ├── unit_tests/
│   │   ├── test_utils.py
│   │   ├── test_risk_metrics.py
│   │   ├── test_optimizer_math.py
│   │   └── test_logger_setup.py
│   │
│   ├── functional_tests/
│   │   ├── test_constraints_validation.py
│   │   ├── test_pipeline_steps.py
│   │   └── test_data_io_roundtrip.py
│   │
│   ├── integration_tests/
│   │   ├── test_data_pipeline.py
│   │   ├── test_backtest_pipeline.py
│   │   ├── test_api_endpoints.py
│   │   └── test_forecasting_integration.py
│   │
│   └── stochastic_tests/              # Statistical and Monte Carlo validation tests
│       ├── test_monte_carlo_var.py
│       ├── test_risk_convergence.py
│       ├── test_backtest_consistency.py
│       └── test_forecasting_stability.py
│
├── config/                            # Configuration files and runtime settings
│   ├── settings.yaml                  # Base config (universal defaults)
│   ├── settings.windows.yaml          # Local dev on Windows
│   ├── settings.linux.yaml            # WSL2, Docker, or Linux dev
│   └── settings.aws.yaml              # Production / AWS deployment
│
├── logs/                              # Runtime logs and debug outputs
│   └── hedgeforge.log                 # Application log file
│
├── .github
│   └── workflows
│       ├── ci_cd.yaml                 # pytest pipeline tests
│       └── pre_commit.yaml            # precommit github actions
│
├── main.py                            # Entry point for CLI or API execution
├── pyproject.toml                     # Python project metadata and build configuration
├── uv.lock                            # UV lockfile for reproducible environments
├── .python-version                    # Python version for local dev and CI/CD
├── .dockerignore                      # Docker ignore file for local dev and CI/CD
├── .docker-compose.yml                # Docker Compose file for local dev and CI/CD
├── .pre-commit-config.yaml            # Pre-commit configuration file for local dev and CI/CD
├── .env                               # Shared across all environments
├── environment.yml                    # Conda environment definition
├── README.md                          # Top-level project README
└── .gitignore                         # Git ignore file for virtualenvs, data, and logs
```
