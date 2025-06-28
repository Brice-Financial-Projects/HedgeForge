#!/usr/bin/env python3
"""
HedgeForge Pipeline Runner

Main script to run the complete portfolio optimization and backtesting pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, generate_sample_data, save_results
from risk import calculate_volatility, calculate_sharpe_ratio
from optimizer import PortfolioOptimizer, optimize_portfolio
from constraints import PortfolioConstraints, get_preset_constraints
from forecasting import ReturnForecaster
from backtest import BacktestEngine, run_strategy_comparison


def setup_logging(config):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_or_generate_data(config, logger):
    """Load market data or generate sample data."""
    data_path = Path("data/raw/market_data.csv")
    
    if data_path.exists():
        logger.info("Loading existing market data")
        prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
        returns = prices.pct_change().dropna()
    else:
        logger.info("Generating sample market data")
        prices = generate_sample_data(
            n_assets=10,
            n_days=1000,
            start_date=config['data']['start_date'],
            seed=config['simulation']['seed']
        )
        returns = prices.pct_change().dropna()
        
        # Save sample data
        data_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(data_path)
        logger.info(f"Sample data saved to {data_path}")
    
    return prices, returns


def run_risk_analysis(returns, config, logger):
    """Run comprehensive risk analysis."""
    logger.info("Running risk analysis...")
    
    risk_results = {}
    
    # Calculate basic risk metrics
    risk_results['volatility'] = calculate_volatility(returns)
    risk_results['sharpe_ratio'] = returns.apply(calculate_sharpe_ratio)
    
    # Calculate rolling metrics
    from risk import calculate_rolling_risk_metrics
    rolling_metrics = calculate_rolling_risk_metrics(
        returns, 
        window=config['risk']['rolling_window']
    )
    risk_results['rolling_metrics'] = rolling_metrics
    
    # Calculate correlation matrix
    from risk import calculate_correlation_matrix
    risk_results['correlation_matrix'] = calculate_correlation_matrix(returns)
    
    logger.info("Risk analysis completed")
    return risk_results


def run_optimization_analysis(returns, config, logger):
    """Run portfolio optimization analysis."""
    logger.info("Running optimization analysis...")
    
    opt_results = {}
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns, config['optimization']['risk_free_rate'])
    
    # Mean-variance optimization
    logger.info("Running mean-variance optimization...")
    mv_result = optimizer.mean_variance_optimization()
    opt_results['mean_variance'] = mv_result
    
    # CVaR optimization
    logger.info("Running CVaR optimization...")
    cvar_result = optimizer.cvar_optimization(
        confidence_level=config['optimization']['confidence_level']
    )
    opt_results['cvar'] = cvar_result
    
    # Efficient frontier
    logger.info("Generating efficient frontier...")
    frontier = optimizer.efficient_frontier(n_points=50)
    opt_results['efficient_frontier'] = frontier
    
    logger.info("Optimization analysis completed")
    return opt_results


def run_backtest_analysis(returns, config, logger):
    """Run backtesting analysis."""
    logger.info("Running backtesting analysis...")
    
    backtest_results = {}
    
    # Initialize backtest engine
    backtest = BacktestEngine(returns, risk_free_rate=config['optimization']['risk_free_rate'])
    
    # Test different strategies
    strategies = ['mean_variance', 'cvar']
    
    for strategy in strategies:
        logger.info(f"Running backtest for {strategy} strategy...")
        
        result = backtest.run_backtest(
            strategy=strategy,
            lookback_period=config['backtest']['lookback_period'],
            rebalance_frequency=config['backtest']['rebalance_frequency'],
            transaction_cost=config['backtest']['transaction_cost']
        )
        
        backtest_results[strategy] = result
    
    # Strategy comparison
    logger.info("Running strategy comparison...")
    comparison = run_strategy_comparison(
        returns,
        strategies,
        **config['backtest']
    )
    backtest_results['comparison'] = comparison
    
    logger.info("Backtesting analysis completed")
    return backtest_results


def run_forecasting_analysis(returns, config, logger):
    """Run forecasting analysis."""
    logger.info("Running forecasting analysis...")
    
    forecast_results = {}
    
    # Initialize forecaster
    forecaster = ReturnForecaster(returns)
    
    # Historical mean forecast
    logger.info("Generating historical mean forecasts...")
    hist_forecast = forecaster.historical_mean_forecast()
    forecast_results['historical_mean'] = hist_forecast
    
    # Shrinkage forecast
    logger.info("Generating shrinkage forecasts...")
    shrink_forecast = forecaster.shrinkage_forecast(
        shrinkage=config['forecasting']['shrinkage_parameter']
    )
    forecast_results['shrinkage'] = shrink_forecast
    
    # ML forecast (if enough data)
    if len(returns) > 500:
        logger.info("Generating ML forecasts...")
        try:
            ml_forecast = forecaster.ml_forecast(
                model_type='ridge',
                lookback_periods=config['forecasting']['lookback_periods']
            )
            forecast_results['ml'] = ml_forecast
        except Exception as e:
            logger.warning(f"ML forecasting failed: {e}")
    
    # Ensemble forecast
    logger.info("Generating ensemble forecasts...")
    ensemble_forecast = forecaster.ensemble_forecast()
    forecast_results['ensemble'] = ensemble_forecast
    
    logger.info("Forecasting analysis completed")
    return forecast_results


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Run HedgeForge pipeline')
    parser.add_argument('--config', default='config/settings.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--output-dir', default='data/processed',
                       help='Output directory for results')
    parser.add_argument('--skip-risk', action='store_true',
                       help='Skip risk analysis')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='Skip optimization analysis')
    parser.add_argument('--skip-backtest', action='store_true',
                       help='Skip backtesting analysis')
    parser.add_argument('--skip-forecasting', action='store_true',
                       help='Skip forecasting analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("Starting HedgeForge pipeline...")
    
    # Load or generate data
    prices, returns = load_or_generate_data(config, logger)
    
    # Initialize results storage
    all_results = {
        'config': config,
        'data_info': {
            'n_assets': len(returns.columns),
            'n_periods': len(returns),
            'date_range': [returns.index[0], returns.index[-1]]
        }
    }
    
    # Run analyses
    if not args.skip_risk:
        all_results['risk_analysis'] = run_risk_analysis(returns, config, logger)
    
    if not args.skip_optimization:
        all_results['optimization_analysis'] = run_optimization_analysis(returns, config, logger)
    
    if not args.skip_backtest:
        all_results['backtest_analysis'] = run_backtest_analysis(returns, config, logger)
    
    if not args.skip_forecasting:
        all_results['forecasting_analysis'] = run_forecasting_analysis(returns, config, logger)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_results(all_results, 'hedgeforge_results.json', output_dir)
    
    # Generate summary report
    generate_summary_report(all_results, output_dir, logger)
    
    logger.info("HedgeForge pipeline completed successfully!")


def generate_summary_report(results, output_dir, logger):
    """Generate a summary report of the analysis."""
    logger.info("Generating summary report...")
    
    report = []
    report.append("# HedgeForge Analysis Summary")
    report.append("")
    
    # Data summary
    data_info = results['data_info']
    report.append(f"## Data Summary")
    report.append(f"- Number of assets: {data_info['n_assets']}")
    report.append(f"- Number of periods: {data_info['n_periods']}")
    report.append(f"- Date range: {data_info['date_range'][0]} to {data_info['date_range'][1]}")
    report.append("")
    
    # Risk analysis summary
    if 'risk_analysis' in results:
        report.append("## Risk Analysis")
        risk = results['risk_analysis']
        avg_vol = risk['volatility'].mean()
        avg_sharpe = risk['sharpe_ratio'].mean()
        report.append(f"- Average volatility: {avg_vol:.4f}")
        report.append(f"- Average Sharpe ratio: {avg_sharpe:.4f}")
        report.append("")
    
    # Optimization summary
    if 'optimization_analysis' in results:
        report.append("## Optimization Analysis")
        opt = results['optimization_analysis']
        
        if 'mean_variance' in opt:
            mv = opt['mean_variance']
            report.append(f"- Mean-variance portfolio return: {mv['portfolio_return']:.4f}")
            report.append(f"- Mean-variance portfolio volatility: {mv['portfolio_volatility']:.4f}")
            report.append(f"- Mean-variance Sharpe ratio: {mv['sharpe_ratio']:.4f}")
        
        if 'cvar' in opt:
            cvar = opt['cvar']
            report.append(f"- CVaR portfolio return: {cvar['portfolio_return']:.4f}")
            report.append(f"- CVaR portfolio volatility: {cvar['portfolio_volatility']:.4f}")
            report.append(f"- CVaR portfolio CVaR: {cvar['cvar']:.4f}")
        report.append("")
    
    # Backtest summary
    if 'backtest_analysis' in results:
        report.append("## Backtest Analysis")
        backtest = results['backtest_analysis']
        
        for strategy, result in backtest.items():
            if strategy != 'comparison' and 'performance_metrics' in result:
                metrics = result['performance_metrics']
                report.append(f"### {strategy.replace('_', ' ').title()}")
                report.append(f"- Annualized return: {metrics.get('annualized_return', 0):.4f}")
                report.append(f"- Volatility: {metrics.get('volatility', 0):.4f}")
                report.append(f"- Sharpe ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                report.append(f"- Max drawdown: {metrics.get('max_drawdown', 0):.4f}")
                report.append("")
    
    # Save report
    report_path = output_dir / 'summary_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Summary report saved to {report_path}")


if __name__ == "__main__":
    main() 