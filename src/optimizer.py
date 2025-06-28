"""
Portfolio optimization algorithms and methods.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple, List
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimization engine supporting multiple optimization methods.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the optimizer.
        
        Args:
            returns: Asset returns DataFrame
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252  # Annualized
        self.n_assets = len(returns.columns)
        
        logger.info(f"Initialized optimizer with {self.n_assets} assets")
    
    def mean_variance_optimization(self,
                                 target_return: Optional[float] = None,
                                 risk_aversion: float = 1.0,
                                 constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Mean-variance optimization (Markowitz).
        
        Args:
            target_return: Target portfolio return (if None, maximizes Sharpe ratio)
            risk_aversion: Risk aversion parameter
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary with optimization results
        """
        if constraints is None:
            constraints = {}
        
        # Default constraints
        bounds = constraints.get('bounds', [(0.0, 1.0)] * self.n_assets)
        sum_constraint = constraints.get('sum_to_one', True)
        
        if target_return is not None:
            # Minimize variance subject to target return
            def objective(weights):
                portfolio_var = weights.T @ self.cov_matrix @ weights
                return portfolio_var
            
            constraints_list = []
            if sum_constraint:
                constraints_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            constraints_list.append({'type': 'eq', 'fun': lambda w: w.T @ self.mean_returns - target_return})
            
            result = minimize(objective, 
                            x0=np.ones(self.n_assets) / self.n_assets,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints_list)
        else:
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = weights.T @ self.mean_returns
                portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                return -sharpe  # Minimize negative Sharpe ratio
            
            constraints_list = []
            if sum_constraint:
                constraints_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            
            result = minimize(objective,
                            x0=np.ones(self.n_assets) / self.n_assets,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints_list)
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        weights = result.x
        portfolio_return = weights.T @ self.mean_returns
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success,
            'message': result.message
        }
    
    def cvar_optimization(self,
                         confidence_level: float = 0.95,
                         target_return: Optional[float] = None,
                         constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        CVaR (Conditional Value at Risk) optimization.
        
        Args:
            confidence_level: Confidence level for CVaR
            target_return: Target portfolio return
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary with optimization results
        """
        if constraints is None:
            constraints = {}
        
        bounds = constraints.get('bounds', [(0.0, 1.0)] * self.n_assets)
        sum_constraint = constraints.get('sum_to_one', True)
        
        # Calculate VaR threshold
        portfolio_returns = self.returns @ np.ones(self.n_assets) / self.n_assets
        var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        def cvar_objective(params):
            weights = params[:-1]
            var = params[-1]
            
            portfolio_returns = self.returns @ weights
            tail_returns = portfolio_returns[portfolio_returns <= var]
            
            if len(tail_returns) == 0:
                return 0
            
            cvar = var - (1 / (1 - confidence_level)) * np.mean(tail_returns)
            return cvar
        
        # Initial guess
        x0 = np.append(np.ones(self.n_assets) / self.n_assets, var_threshold)
        
        constraints_list = []
        if sum_constraint:
            constraints_list.append({'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1})
        
        if target_return is not None:
            constraints_list.append({'type': 'eq', 'fun': lambda x: x[:-1].T @ self.mean_returns - target_return})
        
        # Bounds for weights and VaR
        bounds_with_var = bounds + [(-1, 1)]  # VaR bounds
        
        result = minimize(cvar_objective,
                        x0=x0,
                        method='SLSQP',
                        bounds=bounds_with_var,
                        constraints=constraints_list)
        
        if not result.success:
            logger.warning(f"CVaR optimization failed: {result.message}")
        
        weights = result.x[:-1]
        var = result.x[-1]
        
        portfolio_return = weights.T @ self.mean_returns
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Calculate actual CVaR
        portfolio_returns = self.returns @ weights
        threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        tail_returns = portfolio_returns[portfolio_returns <= threshold]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else threshold
        
        return {
            'weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'var': var,
            'cvar': cvar,
            'success': result.success,
            'message': result.message
        }
    
    def black_litterman_optimization(self,
                                   market_caps: pd.Series,
                                   views: Dict[str, float] = None,
                                   view_confidences: Dict[str, float] = None,
                                   tau: float = 0.05,
                                   constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Black-Litterman optimization with subjective views.
        
        Args:
            market_caps: Market capitalization weights
            views: Dictionary of views (asset: expected_return)
            view_confidences: Dictionary of view confidences
            tau: Scaling parameter
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary with optimization results
        """
        if views is None:
            views = {}
        if view_confidences is None:
            view_confidences = {asset: 0.1 for asset in views.keys()}
        
        # Market equilibrium returns
        market_weights = market_caps / market_caps.sum()
        pi = self.risk_free_rate + self.cov_matrix @ market_weights
        
        # Prior distribution
        prior_mean = pi
        prior_cov = tau * self.cov_matrix
        
        if len(views) > 0:
            # Create view matrix
            view_assets = list(views.keys())
            P = np.zeros((len(views), self.n_assets))
            Q = np.zeros(len(views))
            Omega = np.zeros((len(views), len(views)))
            
            for i, asset in enumerate(view_assets):
                asset_idx = self.returns.columns.get_loc(asset)
                P[i, asset_idx] = 1
                Q[i] = views[asset]
                Omega[i, i] = view_confidences.get(asset, 0.1)
            
            # Posterior distribution
            M1 = np.linalg.inv(prior_cov)
            M2 = P.T @ np.linalg.inv(Omega) @ P
            M3 = P.T @ np.linalg.inv(Omega) @ Q
            
            posterior_cov = np.linalg.inv(M1 + M2)
            posterior_mean = posterior_cov @ (M1 @ prior_mean + M3)
        else:
            posterior_mean = prior_mean
            posterior_cov = prior_cov
        
        # Optimize with posterior beliefs
        def objective(weights):
            portfolio_return = weights.T @ posterior_mean
            portfolio_var = weights.T @ self.cov_matrix @ weights
            return -portfolio_return + 0.5 * portfolio_var
        
        if constraints is None:
            constraints = {}
        
        bounds = constraints.get('bounds', [(0.0, 1.0)] * self.n_assets)
        sum_constraint = constraints.get('sum_to_one', True)
        
        constraints_list = []
        if sum_constraint:
            constraints_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        result = minimize(objective,
                        x0=np.ones(self.n_assets) / self.n_assets,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints_list)
        
        if not result.success:
            logger.warning(f"Black-Litterman optimization failed: {result.message}")
        
        weights = result.x
        portfolio_return = weights.T @ posterior_mean
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'posterior_mean': posterior_mean,
            'posterior_cov': posterior_cov,
            'success': result.success,
            'message': result.message
        }
    
    def efficient_frontier(self,
                         n_points: int = 50,
                         constraints: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """
        Generate efficient frontier.
        
        Args:
            n_points: Number of points on the frontier
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary with returns, volatilities, and weights
        """
        # Find minimum variance portfolio
        min_var_result = self.mean_variance_optimization(constraints=constraints)
        min_var_return = min_var_result['portfolio_return']
        min_var_vol = min_var_result['portfolio_volatility']
        
        # Find maximum return portfolio
        max_return = self.mean_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_var_return, max_return, n_points)
        
        returns = []
        volatilities = []
        weights_list = []
        
        for target_return in target_returns:
            result = self.mean_variance_optimization(target_return=target_return, constraints=constraints)
            if result['success']:
                returns.append(result['portfolio_return'])
                volatilities.append(result['portfolio_volatility'])
                weights_list.append(result['weights'])
        
        return {
            'returns': np.array(returns),
            'volatilities': np.array(volatilities),
            'weights': np.array(weights_list)
        }


def optimize_portfolio(returns: pd.DataFrame,
                      method: str = 'mean_variance',
                      **kwargs) -> Dict[str, Any]:
    """
    Convenience function for portfolio optimization.
    
    Args:
        returns: Asset returns DataFrame
        method: Optimization method ('mean_variance', 'cvar', 'black_litterman')
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = PortfolioOptimizer(returns, **kwargs)
    
    if method == 'mean_variance':
        return optimizer.mean_variance_optimization(**kwargs)
    elif method == 'cvar':
        return optimizer.cvar_optimization(**kwargs)
    elif method == 'black_litterman':
        return optimizer.black_litterman_optimization(**kwargs)
    else:
        raise ValueError("Method must be 'mean_variance', 'cvar', or 'black_litterman'") 