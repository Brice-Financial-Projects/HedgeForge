"""
Unit tests for risk module.
"""

import pytest
import numpy as np
import pandas as pd
from src import risk


class TestRiskMetrics:
    """Test risk metric calculations."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (1000, 5)),
            columns=['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4', 'Asset_5']
        )
        self.single_returns = self.returns['Asset_1']
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        vol = risk.calculate_volatility(self.single_returns)
        assert isinstance(vol, float)
        assert vol > 0
        
        vol_df = risk.calculate_volatility(self.returns)
        assert isinstance(vol_df, pd.Series)
        assert len(vol_df) == 5
        assert all(vol_df > 0)
    
    def test_calculate_var(self):
        """Test VaR calculation."""
        var = risk.calculate_var(self.single_returns, confidence_level=0.95)
        assert isinstance(var, float)
        
        var_df = risk.calculate_var(self.returns, confidence_level=0.95)
        assert isinstance(var_df, pd.Series)
        assert len(var_df) == 5
    
    def test_calculate_cvar(self):
        """Test CVaR calculation."""
        cvar = risk.calculate_cvar(self.single_returns, confidence_level=0.95)
        assert isinstance(cvar, float)
        
        cvar_df = risk.calculate_cvar(self.returns, confidence_level=0.95)
        assert isinstance(cvar_df, pd.Series)
        assert len(cvar_df) == 5
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = risk.calculate_sharpe_ratio(self.single_returns)
        assert isinstance(sharpe, float)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        sortino = risk.calculate_sortino_ratio(self.single_returns)
        assert isinstance(sortino, float)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        drawdown_info = risk.calculate_max_drawdown(self.single_returns)
        assert isinstance(drawdown_info, dict)
        assert 'max_drawdown' in drawdown_info
        assert drawdown_info['max_drawdown'] <= 0
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        corr_matrix = risk.calculate_correlation_matrix(self.returns)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (5, 5)
        assert all(corr_matrix.diagonal() == 1.0)
    
    def test_calculate_covariance_matrix(self):
        """Test covariance matrix calculation."""
        cov_matrix = risk.calculate_covariance_matrix(self.returns)
        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape == (5, 5)
        assert all(cov_matrix.diagonal() > 0)
    
    def test_calculate_beta(self):
        """Test beta calculation."""
        market_returns = pd.Series(np.random.normal(0.001, 0.015, 1000))
        beta = risk.calculate_beta(self.single_returns, market_returns)
        assert isinstance(beta, float)
    
    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        portfolio_risk = risk.calculate_portfolio_risk(weights, self.returns)
        assert isinstance(portfolio_risk, float)
        assert portfolio_risk > 0
    
    def test_calculate_rolling_risk_metrics(self):
        """Test rolling risk metrics calculation."""
        rolling_metrics = risk.calculate_rolling_risk_metrics(self.returns, window=100)
        assert isinstance(rolling_metrics, dict)
        assert 'volatility' in rolling_metrics
        assert 'var' in rolling_metrics
        assert 'sharpe' in rolling_metrics


class TestRiskEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_returns(self):
        """Test with empty returns."""
        empty_returns = pd.Series(dtype=float)
        with pytest.raises(ValueError):
            risk.calculate_volatility(empty_returns)
    
    def test_single_return(self):
        """Test with single return value."""
        single_return = pd.Series([0.01])
        vol = risk.calculate_volatility(single_return)
        assert vol == 0.0
    
    def test_all_positive_returns(self):
        """Test Sortino ratio with all positive returns."""
        positive_returns = pd.Series([0.01, 0.02, 0.03])
        sortino = risk.calculate_sortino_ratio(positive_returns)
        assert sortino == np.inf
    
    def test_invalid_confidence_level(self):
        """Test with invalid confidence level."""
        returns = pd.Series([0.01, -0.02, 0.03])
        with pytest.raises(ValueError):
            risk.calculate_var(returns, confidence_level=1.5)
    
    def test_invalid_method(self):
        """Test with invalid method."""
        returns = pd.Series([0.01, -0.02, 0.03])
        with pytest.raises(ValueError):
            risk.calculate_var(returns, method='invalid')
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        returns_with_nan = pd.Series([0.01, np.nan, 0.03, -0.02])
        vol = risk.calculate_volatility(returns_with_nan)
        assert not np.isnan(vol) 