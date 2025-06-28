# 🚀 HedgeForge Setup Guide

This guide will walk you through setting up and running the HedgeForge portfolio risk modeling engine.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** (for cloning the repository)
- **pip** or **conda** (package manager)

## 🛠️ Installation

### Option 1: Using pip (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/hedge_forge.git
   cd hedge_forge
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv hedgeforge_env
   
   # On Windows:
   hedgeforge_env\Scripts\activate
   
   # On macOS/Linux:
   source hedgeforge_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using conda

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/hedge_forge.git
   cd hedge_forge
   ```

2. **Create conda environment:**
   ```bash
   conda create -n hedgeforge python=3.9
   conda activate hedgeforge
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

### 1. Environment Setup

The system uses a YAML configuration file located at `config/settings.yaml`. Key settings include:

- **Simulation parameters** (number of simulations, time steps)
- **Optimization settings** (risk-free rate, position limits)
- **Backtesting parameters** (lookback period, rebalance frequency)
- **Risk metrics** (confidence levels, VaR methods)
- **Logging configuration**

### 2. Customizing Settings

Edit `config/settings.yaml` to customize:

```yaml
# Example customizations
optimization:
  risk_free_rate: 0.03  # Change risk-free rate
  max_position_size: 0.15  # Increase max position size

backtest:
  lookback_period: 504  # Use 2-year lookback
  rebalance_frequency: 42  # Rebalance every 2 months
```

## 🏃‍♂️ Running HedgeForge

### 1. Command Line Interface

#### Run the Complete Pipeline
```bash
python scripts/run_pipeline.py
```

**Optional flags:**
- `--config path/to/config.yaml` - Use custom config file
- `--output-dir path/to/output` - Specify output directory
- `--skip-risk` - Skip risk analysis
- `--skip-optimization` - Skip optimization analysis
- `--skip-backtest` - Skip backtesting analysis
- `--skip-forecasting` - Skip forecasting analysis

#### Example with Custom Settings
```bash
python scripts/run_pipeline.py --config config/custom_settings.yaml --output-dir results/
```

### 2. Interactive Streamlit App

#### Launch the Web Interface
```bash
streamlit run app/main.py
```

The app will open in your default browser at `http://localhost:8501`

#### Streamlit Features
- **Interactive data generation** - Adjust number of assets and time periods
- **Real-time optimization** - Change parameters and see immediate results
- **Visual backtesting** - Run backtests with interactive charts
- **Strategy comparison** - Compare different optimization approaches

### 3. Jupyter Notebooks

#### Run Example Notebooks
```bash
jupyter notebook notebooks/
```

Open `01_basic_usage.ipynb` for a comprehensive walkthrough.

## 📊 Understanding the Output

### 1. Pipeline Results

The pipeline generates several output files in `data/processed/`:

- `hedgeforge_results.json` - Complete analysis results
- `summary_report.md` - Human-readable summary
- Various CSV files with detailed metrics

### 2. Key Metrics Explained

#### Risk Metrics
- **Volatility** - Annualized standard deviation of returns
- **VaR (95%)** - Value at Risk at 95% confidence level
- **CVaR** - Conditional Value at Risk (Expected Shortfall)
- **Sharpe Ratio** - Risk-adjusted return measure
- **Maximum Drawdown** - Largest peak-to-trough decline

#### Performance Metrics
- **Annualized Return** - Geometric mean return annualized
- **Information Ratio** - Excess return per unit of tracking error
- **Turnover** - Portfolio rebalancing frequency
- **Win Rate** - Percentage of positive return periods

## 🔧 Advanced Usage

### 1. Using Real Market Data

Replace sample data with real market data:

```python
from src.utils import load_market_data

# Load your CSV file with price data
prices = load_market_data('data/raw/your_market_data.csv')
returns = prices.pct_change().dropna()
```

### 2. Custom Constraints

Create custom portfolio constraints:

```python
from src.constraints import PortfolioConstraints

constraints = PortfolioConstraints(n_assets=10)
constraints.set_position_limits(min_position=0.02, max_position=0.15)
constraints.set_sector_limits(sector_assignments, sector_limits)
```

### 3. Custom Optimization

Implement custom optimization strategies:

```python
from src.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(returns)
result = optimizer.mean_variance_optimization(
    target_return=0.08,
    constraints=custom_constraints
)
```

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Module
```bash
pytest tests/test_risk.py -v
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem:** `ModuleNotFoundError: No module named 'src'`
**Solution:** Ensure you're in the project root directory and have installed dependencies:
```bash
cd hedge_forge
pip install -r requirements.txt
```

#### 2. Optimization Failures
**Problem:** Optimization returns `success: False`
**Solution:** 
- Check that you have sufficient data (at least 252 periods)
- Verify constraint parameters are reasonable
- Try different initial weights or optimization methods

#### 3. Streamlit Issues
**Problem:** Streamlit app won't start
**Solution:**
```bash
pip install streamlit plotly
streamlit run app/main.py
```

#### 4. Memory Issues
**Problem:** Out of memory during large backtests
**Solution:**
- Reduce lookback period in config
- Use fewer assets
- Increase rebalance frequency

### Getting Help

1. **Check the logs** - Look in `logs/hedgeforge.log`
2. **Review configuration** - Verify `config/settings.yaml`
3. **Run tests** - Ensure all tests pass: `pytest tests/`
4. **Check dependencies** - Verify all packages installed: `pip list`

## 📈 Next Steps

### 1. Explore the Codebase
- Review `src/` modules for implementation details
- Check `notebooks/` for examples
- Examine `tests/` for usage patterns

### 2. Customize for Your Needs
- Modify `config/settings.yaml` for your parameters
- Add custom risk metrics in `src/risk.py`
- Implement new optimization strategies in `src/optimizer.py`

### 3. Extend Functionality
- Add Monte Carlo simulation capabilities
- Implement factor models
- Create custom backtesting scenarios
- Add real-time data feeds

### 4. Production Deployment
- Set up proper logging and monitoring
- Implement error handling and recovery
- Add authentication and access controls
- Configure automated testing and deployment

## 📚 Additional Resources

- **Documentation:** Check `internal_docs/` for detailed technical documentation
- **Examples:** Review `notebooks/` for usage examples
- **Configuration:** See `config/settings.yaml` for all available options
- **Tests:** Examine `tests/` for implementation patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

---

**Happy optimizing! 🎯**

For questions or issues, please check the troubleshooting section above or create an issue in the repository. 