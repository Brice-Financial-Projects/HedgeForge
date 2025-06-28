"""
Risk metrics and calculations for portfolio analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_volatility(returns: Union[pd.DataFrame, pd.Series, np.ndarray],
                        annualize: bool = True,
                        periods_per_year: int = 252) -> Union[float, pd.Series]:
    """
    Calculate volatility (standard deviation) of returns.
    
    Args:
        returns: Return data
        annualize: Whether to annualize the volatility
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Volatility value(s)
    """
    if isinstance(returns, pd.DataFrame):
        vol = returns.std()
    else:
        vol = np.std(returns)
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


def calculate_var(returns: Union[pd.DataFrame, pd.Series, np.ndarray],
                 confidence_level: float = 0.95,
                 method: str = 'historical',
                 annualize: bool = True,
                 periods_per_year: int = 252) -> Union[float, pd.Series]:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Return data
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        method: 'historical' or 'parametric'
        annualize: Whether to annualize the VaR
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        VaR value(s)
    """
    if method == 'historical':
        if isinstance(returns, pd.DataFrame):
            var_values = returns.quantile(1 - confidence_level)
        else:
            var_values = np.percentile(returns, (1 - confidence_level) * 100)
    
    elif method == 'parametric':
        if isinstance(returns, pd.DataFrame):
            mean_returns = returns.mean()
            std_returns = returns.std()
            var_values = mean_returns - stats.norm.ppf(confidence_level) * std_returns
        else:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            var_values = mean_return - stats.norm.ppf(confidence_level) * std_return
    
    else:
        raise ValueError("Method must be 'historical' or 'parametric'")
    
    if annualize:
        var_values = var_values * np.sqrt(periods_per_year)
    
    return var_values


def calculate_cvar(returns: Union[pd.DataFrame, pd.Series, np.ndarray],
                  confidence_level: float = 0.95,
                  method: str = 'historical',
                  annualize: bool = True,
                  periods_per_year: int = 252) -> Union[float, pd.Series]:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns: Return data
        confidence_level: Confidence level for CVaR
        method: 'historical' or 'parametric'
        annualize: Whether to annualize the CVaR
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        CVaR value(s)
    """
    if method == 'historical':
        if isinstance(returns, pd.DataFrame):
            threshold = returns.quantile(1 - confidence_level)
            cvar_values = pd.Series(index=returns.columns, dtype=float)
            
            for col in returns.columns:
                tail_returns = returns[col][returns[col] <= threshold[col]]
                cvar_values[col] = tail_returns.mean()
        else:
            threshold = np.percentile(returns, (1 - confidence_level) * 100)
            tail_returns = returns[returns <= threshold]
            cvar_values = np.mean(tail_returns)
    
    elif method == 'parametric':
        if isinstance(returns, pd.DataFrame):
            mean_returns = returns.mean()
            std_returns = returns.std()
            # For normal distribution, CVaR has closed form
            z_score = stats.norm.ppf(1 - confidence_level)
            cvar_values = mean_returns - std_returns * stats.norm.pdf(z_score) / (1 - confidence_level)
        else:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            cvar_values = mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)
    
    else:
        raise ValueError("Method must be 'historical' or 'parametric'")
    
    if annualize:
        cvar_values = cvar_values * np.sqrt(periods_per_year)
    
    return cvar_values


def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray],
                          risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return data
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray],
                           risk_free_rate: float = 0.02,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (downside deviation).
    
    Args:
        returns: Return data
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(periods_per_year)
    
    return sortino


def calculate_max_drawdown(returns: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.
    
    Args:
        returns: Return data
        
    Returns:
        Dictionary with max drawdown, start date, end date, and recovery date
    """
    if isinstance(returns, pd.Series):
        cumulative = (1 + returns).cumprod()
        dates = returns.index
    else:
        cumulative = np.cumprod(1 + returns)
        dates = np.arange(len(returns))
    
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    
    # Find peak before max drawdown
    peak_idx = np.argmax(cumulative[:max_dd_idx + 1])
    
    # Find recovery point (where cumulative returns exceed the peak)
    recovery_idx = None
    if max_dd_idx < len(cumulative) - 1:
        recovery_mask = cumulative[max_dd_idx:] >= cumulative[peak_idx]
        if np.any(recovery_mask):
            recovery_idx = max_dd_idx + np.argmax(recovery_mask)
    
    return {
        'max_drawdown': max_dd,
        'peak_date': dates[peak_idx] if hasattr(dates, '__getitem__') else peak_idx,
        'trough_date': dates[max_dd_idx] if hasattr(dates, '__getitem__') else max_dd_idx,
        'recovery_date': dates[recovery_idx] if recovery_idx is not None and hasattr(dates, '__getitem__') else recovery_idx,
        'drawdown_duration': max_dd_idx - peak_idx if recovery_idx is None else recovery_idx - peak_idx
    }


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for asset returns.
    
    Args:
        returns: DataFrame with asset returns
        
    Returns:
        Correlation matrix
    """
    return returns.corr()


def calculate_covariance_matrix(returns: pd.DataFrame,
                              method: str = 'sample',
                              shrinkage: float = 0.0) -> pd.DataFrame:
    """
    Calculate covariance matrix for asset returns.
    
    Args:
        returns: DataFrame with asset returns
        method: 'sample' for sample covariance, 'ledoit_wolf' for shrinkage
        shrinkage: Shrinkage parameter (0-1)
        
    Returns:
        Covariance matrix
    """
    if method == 'sample':
        cov_matrix = returns.cov()
    elif method == 'ledoit_wolf':
        # Simple shrinkage estimator
        sample_cov = returns.cov()
        target = np.eye(len(returns.columns)) * np.trace(sample_cov) / len(returns.columns)
        cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * target
    else:
        raise ValueError("Method must be 'sample' or 'ledoit_wolf'")
    
    return cov_matrix


def calculate_beta(asset_returns: Union[pd.Series, np.ndarray],
                  market_returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate beta of an asset relative to the market.
    
    Args:
        asset_returns: Asset return data
        market_returns: Market return data
        
    Returns:
        Beta value
    """
    if isinstance(asset_returns, pd.Series):
        asset_returns = asset_returns.values
    if isinstance(market_returns, pd.Series):
        market_returns = market_returns.values
    
    # Remove any NaN values
    mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
    asset_clean = asset_returns[mask]
    market_clean = market_returns[mask]
    
    if len(asset_clean) < 2:
        return np.nan
    
    covariance = np.cov(asset_clean, market_clean)[0, 1]
    market_variance = np.var(market_clean)
    
    beta = covariance / market_variance if market_variance > 0 else np.nan
    
    return beta


def calculate_portfolio_risk(weights: np.ndarray,
                           returns: pd.DataFrame,
                           risk_metric: str = 'volatility',
                           **kwargs) -> float:
    """
    Calculate portfolio risk using specified metric.
    
    Args:
        weights: Portfolio weights
        returns: Asset returns DataFrame
        risk_metric: 'volatility', 'var', 'cvar'
        **kwargs: Additional arguments for risk calculation
        
    Returns:
        Portfolio risk value
    """
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    
    if risk_metric == 'volatility':
        cov_matrix = calculate_covariance_matrix(returns)
        portfolio_var = weights.T @ cov_matrix @ weights
        return np.sqrt(portfolio_var)
    
    elif risk_metric == 'var':
        portfolio_returns = (returns @ weights).dropna()
        return calculate_var(portfolio_returns, **kwargs)
    
    elif risk_metric == 'cvar':
        portfolio_returns = (returns @ weights).dropna()
        return calculate_cvar(portfolio_returns, **kwargs)
    
    else:
        raise ValueError("Risk metric must be 'volatility', 'var', or 'cvar'")


def calculate_rolling_risk_metrics(returns: pd.DataFrame,
                                 window: int = 252,
                                 metrics: list = None) -> Dict[str, pd.DataFrame]:
    """
    Calculate rolling risk metrics.
    
    Args:
        returns: Asset returns DataFrame
        window: Rolling window size
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary with rolling risk metrics
    """
    if metrics is None:
        metrics = ['volatility', 'var', 'sharpe']
    
    results = {}
    
    for metric in metrics:
        if metric == 'volatility':
            results[metric] = returns.rolling(window=window).std() * np.sqrt(252)
        elif metric == 'var':
            results[metric] = returns.rolling(window=window).quantile(0.05) * np.sqrt(252)
        elif metric == 'sharpe':
            rolling_mean = returns.rolling(window=window).mean() * 252
            rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
            results[metric] = rolling_mean / rolling_std
    
    return results 