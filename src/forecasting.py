"""
Return forecasting models and methods.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple, List
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
import warnings

logger = logging.getLogger(__name__)


class ReturnForecaster:
    """
    Return forecasting engine with multiple forecasting methods.
    """
    
    def __init__(self, returns: pd.DataFrame, features: Optional[pd.DataFrame] = None):
        """
        Initialize the forecaster.
        
        Args:
            returns: Asset returns DataFrame
            features: Optional features DataFrame for ML models
        """
        self.returns = returns
        self.features = features
        self.models = {}
        self.scalers = {}
        self.forecast_history = {}
        
        logger.info(f"Initialized forecaster with {len(returns.columns)} assets")
    
    def historical_mean_forecast(self, 
                               window: Optional[int] = None,
                               method: str = 'simple') -> pd.Series:
        """
        Historical mean return forecast.
        
        Args:
            window: Rolling window size (None for full history)
            method: 'simple' for arithmetic mean, 'geometric' for geometric mean
            
        Returns:
            Series with forecasted returns
        """
        if window is None:
            # Use full history
            if method == 'simple':
                forecasts = self.returns.mean() * 252  # Annualize
            elif method == 'geometric':
                forecasts = ((1 + self.returns).prod() ** (252 / len(self.returns))) - 1
            else:
                raise ValueError("Method must be 'simple' or 'geometric'")
        else:
            # Use rolling window
            if method == 'simple':
                forecasts = self.returns.rolling(window=window).mean().iloc[-1] * 252
            elif method == 'geometric':
                # Rolling geometric mean
                forecasts = pd.Series(index=self.returns.columns, dtype=float)
                for col in self.returns.columns:
                    rolling_returns = self.returns[col].rolling(window=window)
                    forecasts[col] = ((1 + rolling_returns).prod() ** (252 / window) - 1).iloc[-1]
            else:
                raise ValueError("Method must be 'simple' or 'geometric'")
        
        self.forecast_history['historical_mean'] = forecasts
        logger.info(f"Generated historical mean forecasts using {method} method")
        
        return forecasts
    
    def shrinkage_forecast(self, 
                          target: Union[str, pd.Series] = 'equal_weight',
                          shrinkage: float = 0.5) -> pd.Series:
        """
        Shrinkage estimator for return forecasts.
        
        Args:
            target: Target for shrinkage ('equal_weight', 'market_cap', or custom Series)
            shrinkage: Shrinkage parameter (0-1)
            
        Returns:
            Series with shrunk return forecasts
        """
        # Get sample mean
        sample_mean = self.returns.mean() * 252
        
        # Get target
        if target == 'equal_weight':
            target_mean = pd.Series(1.0 / len(self.returns.columns), index=self.returns.columns)
        elif target == 'market_cap':
            # Assume equal market cap for now (could be enhanced with actual market caps)
            target_mean = pd.Series(1.0 / len(self.returns.columns), index=self.returns.columns)
        elif isinstance(target, pd.Series):
            target_mean = target
        else:
            raise ValueError("Target must be 'equal_weight', 'market_cap', or a Series")
        
        # Apply shrinkage
        shrunk_forecasts = (1 - shrinkage) * sample_mean + shrinkage * target_mean
        
        self.forecast_history['shrinkage'] = shrunk_forecasts
        logger.info(f"Generated shrinkage forecasts with parameter {shrinkage}")
        
        return shrunk_forecasts
    
    def ml_forecast(self, 
                   model_type: str = 'ridge',
                   lookback_periods: int = 252,
                   target_horizon: int = 21,
                   **kwargs) -> pd.Series:
        """
        Machine learning-based return forecast.
        
        Args:
            model_type: Type of ML model ('linear', 'ridge', 'lasso', 'random_forest')
            lookback_periods: Number of periods to use for features
            target_horizon: Forecast horizon in periods
            **kwargs: Additional model parameters
            
        Returns:
            Series with ML forecasts
        """
        if self.features is None:
            # Create features from returns
            features = self._create_return_features(lookback_periods)
        else:
            features = self.features
        
        forecasts = pd.Series(index=self.returns.columns, dtype=float)
        
        for asset in self.returns.columns:
            # Prepare data
            y = self.returns[asset].shift(-target_horizon).dropna()
            X = features.loc[y.index].dropna()
            
            # Align data
            common_idx = y.index.intersection(X.index)
            y = y.loc[common_idx]
            X = X.loc[common_idx]
            
            if len(y) < 50:  # Need sufficient data
                logger.warning(f"Insufficient data for {asset}, using historical mean")
                forecasts[asset] = self.returns[asset].mean() * 252
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            if model_type == 'linear':
                model = LinearRegression(**kwargs)
            elif model_type == 'ridge':
                model = Ridge(**kwargs)
            elif model_type == 'lasso':
                model = Lasso(**kwargs)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(**kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[asset] = model
            self.scalers[asset] = scaler
            
            # Make forecast using most recent data
            latest_features = features.iloc[-1:].dropna()
            if len(latest_features) > 0:
                latest_scaled = scaler.transform(latest_features)
                forecast = model.predict(latest_scaled)[0] * 252  # Annualize
            else:
                forecast = self.returns[asset].mean() * 252
            
            forecasts[asset] = forecast
        
        self.forecast_history['ml'] = forecasts
        logger.info(f"Generated ML forecasts using {model_type} model")
        
        return forecasts
    
    def _create_return_features(self, lookback_periods: int) -> pd.DataFrame:
        """
        Create features from return data.
        
        Args:
            lookback_periods: Number of periods to look back
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=self.returns.index)
        
        for asset in self.returns.columns:
            returns = self.returns[asset]
            
            # Lagged returns
            for lag in [1, 5, 21]:
                features[f'{asset}_lag_{lag}'] = returns.shift(lag)
            
            # Rolling statistics
            features[f'{asset}_rolling_mean_21'] = returns.rolling(21).mean()
            features[f'{asset}_rolling_std_21'] = returns.rolling(21).std()
            features[f'{asset}_rolling_skew_21'] = returns.rolling(21).skew()
            
            # Momentum indicators
            features[f'{asset}_momentum_5'] = returns.rolling(5).sum()
            features[f'{asset}_momentum_21'] = returns.rolling(21).sum()
            
            # Volatility indicators
            features[f'{asset}_volatility_21'] = returns.rolling(21).std() * np.sqrt(252)
        
        return features
    
    def scenario_forecast(self, 
                         scenarios: Dict[str, Dict[str, float]],
                         base_forecast: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Scenario-based return forecasts.
        
        Args:
            scenarios: Dictionary of scenarios with asset return adjustments
            base_forecast: Base forecast to adjust (if None, uses historical mean)
            
        Returns:
            Dictionary with scenario forecasts
        """
        if base_forecast is None:
            base_forecast = self.historical_mean_forecast()
        
        scenario_forecasts = {}
        
        for scenario_name, adjustments in scenarios.items():
            forecast = base_forecast.copy()
            
            for asset, adjustment in adjustments.items():
                if asset in forecast.index:
                    forecast[asset] += adjustment
            
            scenario_forecasts[scenario_name] = forecast
        
        self.forecast_history['scenarios'] = scenario_forecasts
        logger.info(f"Generated scenario forecasts for {len(scenarios)} scenarios")
        
        return scenario_forecasts
    
    def ensemble_forecast(self, 
                         methods: List[str] = None,
                         weights: Optional[List[float]] = None) -> pd.Series:
        """
        Ensemble forecast combining multiple methods.
        
        Args:
            methods: List of forecasting methods to combine
            weights: Weights for each method (if None, equal weights)
            
        Returns:
            Series with ensemble forecast
        """
        if methods is None:
            methods = ['historical_mean', 'shrinkage']
        
        forecasts = []
        
        for method in methods:
            if method == 'historical_mean':
                forecast = self.historical_mean_forecast()
            elif method == 'shrinkage':
                forecast = self.shrinkage_forecast()
            elif method == 'ml':
                forecast = self.ml_forecast()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            forecasts.append(forecast)
        
        # Combine forecasts
        if weights is None:
            weights = [1.0 / len(forecasts)] * len(forecasts)
        
        ensemble_forecast = pd.Series(0.0, index=forecasts[0].index)
        for forecast, weight in zip(forecasts, weights):
            ensemble_forecast += weight * forecast
        
        self.forecast_history['ensemble'] = ensemble_forecast
        logger.info(f"Generated ensemble forecast using {methods}")
        
        return ensemble_forecast
    
    def evaluate_forecasts(self, 
                          actual_returns: pd.Series,
                          forecast_methods: List[str] = None) -> Dict[str, float]:
        """
        Evaluate forecast accuracy.
        
        Args:
            actual_returns: Actual returns for evaluation period
            forecast_methods: Methods to evaluate (if None, evaluates all)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if forecast_methods is None:
            forecast_methods = list(self.forecast_history.keys())
        
        results = {}
        
        for method in forecast_methods:
            if method not in self.forecast_history:
                continue
            
            forecast = self.forecast_history[method]
            
            # Align data
            common_assets = forecast.index.intersection(actual_returns.index)
            if len(common_assets) == 0:
                continue
            
            forecast_aligned = forecast.loc[common_assets]
            actual_aligned = actual_returns.loc[common_assets]
            
            # Calculate metrics
            mse = np.mean((forecast_aligned - actual_aligned) ** 2)
            mae = np.mean(np.abs(forecast_aligned - actual_aligned))
            correlation = np.corrcoef(forecast_aligned, actual_aligned)[0, 1]
            
            results[method] = {
                'mse': mse,
                'mae': mae,
                'correlation': correlation
            }
        
        return results


def rolling_forecast(returns: pd.DataFrame,
                    forecast_method: str = 'historical_mean',
                    window: int = 252,
                    step: int = 21,
                    **kwargs) -> pd.DataFrame:
    """
    Generate rolling forecasts.
    
    Args:
        returns: Asset returns DataFrame
        forecast_method: Forecasting method to use
        window: Rolling window size
        step: Step size for rolling window
        **kwargs: Additional arguments for forecaster
        
    Returns:
        DataFrame with rolling forecasts
    """
    forecaster = ReturnForecaster(returns)
    forecasts = []
    dates = []
    
    for i in range(window, len(returns), step):
        # Get window data
        window_data = returns.iloc[i-window:i]
        
        # Update forecaster with window data
        forecaster.returns = window_data
        
        # Generate forecast
        if forecast_method == 'historical_mean':
            forecast = forecaster.historical_mean_forecast(**kwargs)
        elif forecast_method == 'shrinkage':
            forecast = forecaster.shrinkage_forecast(**kwargs)
        elif forecast_method == 'ml':
            forecast = forecaster.ml_forecast(**kwargs)
        else:
            raise ValueError(f"Unknown forecast method: {forecast_method}")
        
        forecasts.append(forecast)
        dates.append(returns.index[i])
    
    return pd.DataFrame(forecasts, index=dates) 