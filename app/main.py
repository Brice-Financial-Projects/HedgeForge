"""
HedgeForge Streamlit App

Interactive web application for portfolio analysis and optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import specific functions to avoid relative import issues
from utils import generate_sample_data, load_config
from risk import calculate_volatility, calculate_sharpe_ratio, calculate_var
from optimizer import PortfolioOptimizer
from backtest import BacktestEngine
from constraints import PortfolioConstraints


def main():
    st.set_page_config(
        page_title="HedgeForge Portfolio Engine",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 HedgeForge Portfolio Risk Modeling Engine")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Data generation
    st.sidebar.subheader("Data Settings")
    n_assets = st.sidebar.slider("Number of Assets", 5, 20, 10)
    n_days = st.sidebar.slider("Number of Days", 500, 2000, 1000)
    
    # Optimization settings
    st.sidebar.subheader("Optimization Settings")
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.1, 0.02, 0.001)
    strategy = st.sidebar.selectbox("Strategy", ["mean_variance", "cvar"])
    
    # Backtest settings
    st.sidebar.subheader("Backtest Settings")
    lookback_period = st.sidebar.slider("Lookback Period", 126, 756, 252)
    rebalance_freq = st.sidebar.slider("Rebalance Frequency", 5, 63, 21)
    transaction_cost = st.sidebar.number_input("Transaction Cost", 0.0, 0.01, 0.001, 0.0001)
    
    # Generate data
    if st.sidebar.button("Generate New Data"):
        st.session_state.data = None
    
    if 'data' not in st.session_state or st.session_state.data is None:
        prices = generate_sample_data(n_assets=n_assets, n_days=n_days)
        returns = prices.pct_change().dropna()
        st.session_state.data = {'prices': prices, 'returns': returns}
    
    prices = st.session_state.data['prices']
    returns = st.session_state.data['returns']
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Data Overview")
        
        # Price chart
        fig = go.Figure()
        for col in prices.columns[:5]:  # Show first 5 assets
            fig.add_trace(go.Scatter(
                x=prices.index,
                y=prices[col],
                mode='lines',
                name=col,
                line=dict(width=1)
            ))
        
        fig.update_layout(
            title="Asset Prices (First 5 Assets)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Data Summary")
        st.write(f"**Assets:** {len(returns.columns)}")
        st.write(f"**Periods:** {len(returns)}")
        st.write(f"**Date Range:** {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
        
        # Basic statistics
        avg_return = returns.mean() * 252
        avg_vol = returns.std() * np.sqrt(252)
        
        st.write(f"**Avg Annual Return:** {avg_return.mean():.4f}")
        st.write(f"**Avg Annual Volatility:** {avg_vol.mean():.4f}")
    
    # Risk Analysis
    st.markdown("---")
    st.subheader("Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        volatility = calculate_volatility(returns)
        fig = px.bar(
            x=volatility.index,
            y=volatility.values,
            title="Asset Volatilities",
            labels={'x': 'Asset', 'y': 'Volatility'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sharpe_ratios = returns.apply(calculate_sharpe_ratio, risk_free_rate=risk_free_rate)
        fig = px.bar(
            x=sharpe_ratios.index,
            y=sharpe_ratios.values,
            title="Sharpe Ratios",
            labels={'x': 'Asset', 'y': 'Sharpe Ratio'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        var_95 = calculate_var(returns, confidence_level=0.95)
        fig = px.bar(
            x=var_95.index,
            y=var_95.values,
            title="VaR (95%)",
            labels={'x': 'Asset', 'y': 'VaR'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio Optimization
    st.markdown("---")
    st.subheader("Portfolio Optimization")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Run optimization
        optimizer_instance = PortfolioOptimizer(returns, risk_free_rate=risk_free_rate)
        
        if strategy == "mean_variance":
            result = optimizer_instance.mean_variance_optimization()
        else:  # cvar
            result = optimizer_instance.cvar_optimization()
        
        # Display results
        st.write("**Optimization Results:**")
        st.write(f"Portfolio Return: {result['portfolio_return']:.4f}")
        st.write(f"Portfolio Volatility: {result['portfolio_volatility']:.4f}")
        st.write(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        st.write(f"Success: {result['success']}")
        
        if not result['success']:
            st.warning(f"Optimization failed: {result['message']}")
    
    with col2:
        # Plot weights
        weights_df = pd.DataFrame({
            'Asset': returns.columns,
            'Weight': result['weights']
        })
        
        fig = px.pie(
            weights_df,
            values='Weight',
            names='Asset',
            title="Optimal Portfolio Weights"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Efficient Frontier
    st.markdown("---")
    st.subheader("Efficient Frontier")
    
    frontier = optimizer_instance.efficient_frontier(n_points=50)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier['volatilities'],
        y=frontier['returns'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=2)
    ))
    
    # Add optimal point
    fig.add_trace(go.Scatter(
        x=[result['portfolio_volatility']],
        y=[result['portfolio_return']],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Portfolio Volatility",
        yaxis_title="Portfolio Return",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Backtesting
    st.markdown("---")
    st.subheader("Backtesting Results")
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            backtest_instance = BacktestEngine(returns, risk_free_rate=risk_free_rate)
            backtest_result = backtest_instance.run_backtest(
                strategy=strategy,
                lookback_period=lookback_period,
                rebalance_frequency=rebalance_freq,
                transaction_cost=transaction_cost
            )
            
            # Display performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = backtest_result['performance_metrics']
            
            with col1:
                st.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.4f}")
            
            with col2:
                st.metric("Volatility", f"{metrics.get('volatility', 0):.4f}")
            
            with col3:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.4f}")
            
            with col4:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.4f}")
            
            # Plot cumulative returns
            portfolio_returns = backtest_result['portfolio_returns']
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative Portfolio Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot weight evolution
            weights_history = backtest_result['portfolio_weights']
            
            fig = go.Figure()
            for col in weights_history.columns:
                fig.add_trace(go.Scatter(
                    x=weights_history.index,
                    y=weights_history[col],
                    mode='lines',
                    name=col,
                    line=dict(width=1)
                ))
            
            fig.update_layout(
                title="Portfolio Weights Over Time",
                xaxis_title="Date",
                yaxis_title="Weight",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **HedgeForge** - A Python-based quantitative finance project for portfolio risk modeling and optimization.
        
        Built with ❤️ using Streamlit, NumPy, Pandas, and SciPy.
        """
    )


if __name__ == "__main__":
    main() 