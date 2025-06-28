"""
Backtesting engine for portfolio strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List, Tuple
import logging
from . import risk
from . import optimizer
from . import constraints
from . import forecasting
from . import utils

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for portfolio strategies.
    """
    
    def __init__(self, 
                 returns: pd.DataFrame,
                 prices: Optional[pd.DataFrame] = None,
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtesting engine.
        
        Args:
            returns: Asset returns DataFrame
            prices: Asset prices DataFrame (optional, for transaction costs)
            benchmark_returns: Benchmark returns Series
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.prices = prices
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        self.portfolio_weights = pd.DataFrame()
        self.portfolio_returns = pd.Series()
        self.performance_metrics = {}
        
        logger.info(f"Initialized backtest engine with {len(returns.columns)} assets")
    
    def run_backtest(self,
                    strategy: str = 'mean_variance',
                    lookback_period: int = 252,
                    rebalance_frequency: int = 21,
                    transaction_cost: float = 0.001,
                    constraints: Optional[Dict[str, Any]] = None,
                    forecast_method: str = 'historical_mean',
                    **kwargs) -> Dict[str, Any]:
        """
        Run backtest for a given strategy.
        
        Args:
            strategy: Optimization strategy ('mean_variance', 'cvar', 'black_litterman')
            lookback_period: Lookback period for optimization
            rebalance_frequency: Rebalancing frequency in periods
            transaction_cost: Transaction cost as fraction
            constraints: Portfolio constraints
            forecast_method: Return forecasting method
            **kwargs: Additional strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest: {strategy} strategy, {lookback_period} lookback, {rebalance_frequency} rebalance")
        
        # Initialize results storage
        weights_history = []
        returns_history = []
        dates = []
        
        # Run rolling optimization
        for i in range(lookback_period, len(self.returns), rebalance_frequency):
            # Get lookback data
            lookback_data = self.returns.iloc[i-lookback_period:i]
            
            # Generate return forecasts
            forecaster = forecasting.ReturnForecaster(lookback_data)
            if forecast_method == 'historical_mean':
                forecasts = forecaster.historical_mean_forecast()
            elif forecast_method == 'shrinkage':
                forecasts = forecaster.shrinkage_forecast()
            elif forecast_method == 'ml':
                forecasts = forecaster.ml_forecast()
            else:
                raise ValueError(f"Unknown forecast method: {forecast_method}")
            
            # Run optimization
            opt_engine = optimizer.PortfolioOptimizer(lookback_data, self.risk_free_rate)
            
            if strategy == 'mean_variance':
                result = opt_engine.mean_variance_optimization(constraints=constraints, **kwargs)
            elif strategy == 'cvar':
                result = opt_engine.cvar_optimization(constraints=constraints, **kwargs)
            elif strategy == 'black_litterman':
                # For Black-Litterman, we need market caps - using equal weight for now
                market_caps = pd.Series(1.0 / len(lookback_data.columns), index=lookback_data.columns)
                result = opt_engine.black_litterman_optimization(market_caps, constraints=constraints, **kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if not result['success']:
                logger.warning(f"Optimization failed at period {i}: {result['message']}")
                # Use equal weight as fallback
                weights = np.ones(len(lookback_data.columns)) / len(lookback_data.columns)
            else:
                weights = result['weights']
            
            # Store weights and date
            weights_history.append(weights)
            dates.append(self.returns.index[i])
        
        # Calculate portfolio returns
        self.portfolio_weights = pd.DataFrame(weights_history, index=dates, columns=self.returns.columns)
        self.portfolio_returns = self._calculate_portfolio_returns(transaction_cost)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()
        
        logger.info("Backtest completed successfully")
        
        return {
            'portfolio_weights': self.portfolio_weights,
            'portfolio_returns': self.portfolio_returns,
            'performance_metrics': self.performance_metrics
        }
    
    def _calculate_portfolio_returns(self, transaction_cost: float) -> pd.Series:
        """
        Calculate portfolio returns with transaction costs.
        
        Args:
            transaction_cost: Transaction cost as fraction
            
        Returns:
            Series with portfolio returns
        """
        portfolio_returns = pd.Series(index=self.returns.index, dtype=float)
        
        # Initialize with first weights
        current_weights = self.portfolio_weights.iloc[0]
        
        for i, date in enumerate(self.returns.index):
            # Calculate return for current weights
            if i > 0:  # Skip first period (no return yet)
                asset_returns = self.returns.iloc[i]
                portfolio_return = (current_weights * asset_returns).sum()
                portfolio_returns.iloc[i] = portfolio_return
            
            # Check if rebalancing is needed
            if date in self.portfolio_weights.index:
                new_weights = self.portfolio_weights.loc[date]
                
                # Calculate transaction costs
                if i > 0:  # Skip first period
                    weight_change = np.abs(new_weights - current_weights)
                    transaction_cost_total = weight_change.sum() * transaction_cost
                    portfolio_returns.iloc[i] -= transaction_cost_total
                
                current_weights = new_weights
        
        return portfolio_returns.dropna()
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        returns = self.portfolio_returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Risk metrics
        var_95 = risk.calculate_var(returns, confidence_level=0.95)
        cvar_95 = risk.calculate_cvar(returns, confidence_level=0.95)
        max_dd = risk.calculate_max_drawdown(returns)
        
        # Additional metrics
        sortino_ratio = risk.calculate_sortino_ratio(returns, self.risk_free_rate)
        calmar_ratio = annualized_return / abs(max_dd['max_drawdown']) if max_dd['max_drawdown'] != 0 else 0
        
        # Turnover
        turnover = self._calculate_turnover()
        
        # Benchmark comparison
        benchmark_metrics = {}
        if self.benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd['max_drawdown'],
            'turnover': turnover,
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf
        }
        
        metrics.update(benchmark_metrics)
        
        return metrics
    
    def _calculate_turnover(self) -> float:
        """
        Calculate average portfolio turnover.
        
        Returns:
            Average turnover
        """
        if len(self.portfolio_weights) < 2:
            return 0.0
        
        turnovers = []
        for i in range(1, len(self.portfolio_weights)):
            prev_weights = self.portfolio_weights.iloc[i-1]
            curr_weights = self.portfolio_weights.iloc[i]
            turnover = np.abs(curr_weights - prev_weights).sum() / 2
            turnovers.append(turnover)
        
        return np.mean(turnovers)
    
    def _calculate_benchmark_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate benchmark comparison metrics.
        
        Args:
            portfolio_returns: Portfolio returns
            
        Returns:
            Dictionary with benchmark metrics
        """
        # Align data
        common_dates = portfolio_returns.index.intersection(self.benchmark_returns.index)
        if len(common_dates) == 0:
            return {}
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = self.benchmark_returns.loc[common_dates]
        
        # Calculate metrics
        excess_returns = portfolio_aligned - benchmark_aligned
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Beta and alpha
        beta = risk.calculate_beta(portfolio_aligned, benchmark_aligned)
        alpha = portfolio_aligned.mean() - beta * benchmark_aligned.mean()
        
        # Correlation
        correlation = portfolio_aligned.corr(benchmark_aligned)
        
        return {
            'excess_return': excess_returns.mean() * 252,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha * 252,
            'correlation': correlation
        }
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Cumulative returns
            cumulative_returns = (1 + self.portfolio_returns).cumprod()
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, label='Portfolio')
            if self.benchmark_returns is not None:
                benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
                axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, label='Benchmark')
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Rolling Sharpe ratio
            rolling_sharpe = self.portfolio_returns.rolling(252).apply(
                lambda x: risk.calculate_sharpe_ratio(x, self.risk_free_rate)
            )
            axes[0, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
            axes[0, 1].set_title('Rolling Sharpe Ratio (252-day)')
            axes[0, 1].grid(True)
            
            # Drawdown
            cumulative = (1 + self.portfolio_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[1, 0].set_title('Drawdown')
            axes[1, 0].grid(True)
            
            # Weight evolution
            if len(self.portfolio_weights) > 0:
                for col in self.portfolio_weights.columns:
                    axes[1, 1].plot(self.portfolio_weights.index, self.portfolio_weights[col], label=col)
                axes[1, 1].set_title('Portfolio Weights')
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def run_strategy_comparison(returns: pd.DataFrame,
                          strategies: List[str],
                          benchmark_returns: Optional[pd.Series] = None,
                          **kwargs) -> Dict[str, Any]:
    """
    Compare multiple strategies in a backtest.
    
    Args:
        returns: Asset returns DataFrame
        strategies: List of strategy names to compare
        benchmark_returns: Benchmark returns
        **kwargs: Additional backtest parameters
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for strategy in strategies:
        logger.info(f"Running backtest for {strategy}")
        
        backtest = BacktestEngine(returns, benchmark_returns=benchmark_returns)
        result = backtest.run_backtest(strategy=strategy, **kwargs)
        
        results[strategy] = {
            'portfolio_returns': result['portfolio_returns'],
            'performance_metrics': result['performance_metrics']
        }
    
    # Create comparison summary
    comparison = pd.DataFrame()
    for strategy, result in results.items():
        metrics = result['performance_metrics']
        comparison[strategy] = pd.Series(metrics)
    
    return {
        'individual_results': results,
        'comparison': comparison
    } 