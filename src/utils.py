"""
Utility functions for data loading, transformation, and common operations.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hedgeforge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path] = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration parameters."""
    return {
        "simulation": {
            "n_simulations": 1000,
            "time_steps": 252,
            "dt": 1/252,
            "seed": 42
        },
        "optimization": {
            "risk_free_rate": 0.02,
            "target_return": None,
            "max_position_size": 0.1,
            "min_position_size": 0.0
        },
        "backtest": {
            "lookback_period": 756,  # 3 years
            "rebalance_frequency": 21,  # Monthly
            "transaction_cost": 0.001
        },
        "risk": {
            "confidence_level": 0.95,
            "var_method": "historical"
        }
    }


def load_market_data(file_path: Union[str, Path], 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load market data from CSV file.
    
    Args:
        file_path: Path to CSV file containing price data
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        
    Returns:
        DataFrame with datetime index and asset prices
    """
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        logger.info(f"Loaded market data: {df.shape[0]} rows, {df.shape[1]} assets")
        return df
        
    except FileNotFoundError:
        logger.error(f"Market data file not found: {file_path}")
        raise


def calculate_returns(prices: pd.DataFrame, 
                     method: str = 'log') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with price data
        method: 'log' for log returns, 'simple' for simple returns
        
    Returns:
        DataFrame with returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = (prices - prices.shift(1)) / prices.shift(1)
    else:
        raise ValueError("Method must be 'log' or 'simple'")
    
    return returns.dropna()


def calculate_rolling_statistics(returns: pd.DataFrame, 
                               window: int = 252,
                               min_periods: int = None) -> Dict[str, pd.DataFrame]:
    """
    Calculate rolling statistics for returns.
    
    Args:
        returns: DataFrame with return data
        window: Rolling window size
        min_periods: Minimum periods for calculation
        
    Returns:
        Dictionary containing rolling mean, volatility, and correlation
    """
    if min_periods is None:
        min_periods = window // 2
    
    rolling_mean = returns.rolling(window=window, min_periods=min_periods).mean()
    rolling_vol = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)
    
    # Rolling correlation matrix (simplified - returns correlation of first two assets)
    if returns.shape[1] >= 2:
        rolling_corr = returns.iloc[:, 0].rolling(window=window, min_periods=min_periods).corr(returns.iloc[:, 1])
    else:
        rolling_corr = pd.Series(index=returns.index)
    
    return {
        'mean': rolling_mean,
        'volatility': rolling_vol,
        'correlation': rolling_corr
    }


def generate_sample_data(n_assets: int = 10, 
                        n_days: int = 1000,
                        start_date: str = '2020-01-01',
                        seed: int = 42) -> pd.DataFrame:
    """
    Generate sample market data for testing and development.
    
    Args:
        n_assets: Number of assets to generate
        n_days: Number of trading days
        start_date: Start date for the data
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with simulated price data
    """
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate random walk prices
    returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))  # Daily returns
    prices = np.exp(np.cumsum(returns, axis=0)) * 100  # Starting at $100
    
    # Create DataFrame
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    df = pd.DataFrame(prices, index=dates, columns=asset_names)
    
    logger.info(f"Generated sample data: {df.shape[0]} days, {df.shape[1]} assets")
    return df


def save_results(results: Dict[str, Any], 
                filename: str,
                output_dir: Union[str, Path] = "data/processed") -> None:
    """
    Save results to file.
    
    Args:
        results: Dictionary containing results to save
        filename: Name of the output file
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    
    if filename.endswith('.csv'):
        if isinstance(results, pd.DataFrame):
            results.to_csv(file_path)
        else:
            pd.DataFrame(results).to_csv(file_path)
    elif filename.endswith('.json'):
        import json
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif filename.endswith('.pkl'):
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
    
    logger.info(f"Results saved to {file_path}")


def validate_portfolio_weights(weights: np.ndarray, 
                             min_weight: float = 0.0,
                             max_weight: float = 1.0,
                             sum_to_one: bool = True) -> bool:
    """
    Validate portfolio weights.
    
    Args:
        weights: Array of portfolio weights
        min_weight: Minimum allowed weight
        max_weight: Maximum allowed weight
        sum_to_one: Whether weights should sum to 1
        
    Returns:
        True if weights are valid, False otherwise
    """
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    
    # Check bounds
    if np.any(weights < min_weight) or np.any(weights > max_weight):
        return False
    
    # Check sum to one
    if sum_to_one and not np.isclose(np.sum(weights), 1.0, atol=1e-6):
        return False
    
    return True


def annualize_metrics(returns: pd.Series, 
                     periods_per_year: int = 252) -> Dict[str, float]:
    """
    Annualize return and volatility metrics.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Dictionary with annualized metrics
    """
    mean_return = returns.mean()
    volatility = returns.std()
    
    annualized_return = mean_return * periods_per_year
    annualized_volatility = volatility * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio
    } 