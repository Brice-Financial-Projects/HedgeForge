"""
Utility functions for HedgeForge
--------------------------------
Provides data loading, validation, and transformation utilities
used throughout the HedgeForge project.
"""

import os

import numpy as np
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads time-series or tabular data from a CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded file is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"❌ The file {file_path} is empty.")

    return df


def validate_data(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """
    Validates that the DataFrame contains required columns
    and no critical missing data.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_cols (list[str]): List of columns that must exist.

    Returns:
        bool: True if valid, False otherwise.

    Raises:
        ValueError: If required columns are missing or data contains NaNs.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")

    if df[required_cols].isnull().any().any():
        raise ValueError("❌ Data contains NaN values in required columns.")

    return True


def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Computes log returns for a given price column.

    Args:
        df (pd.DataFrame): Input DataFrame with a price column.
        price_col (str): Column containing asset prices (default: 'close').

    Returns:
        pd.Series: Log returns.

    Raises:
        KeyError: If the specified price column is missing.
        ValueError: If the price column contains non-positive values.
    """
    if price_col not in df.columns:
        raise KeyError(f"❌ Column '{price_col}' not found in DataFrame.")

    if (df[price_col] <= 0).any():
        raise ValueError("❌ Price column contains non-positive values.")

    log_returns = np.log(df[price_col] / df[price_col].shift(1))
    log_returns.name = f"log_return_{price_col}"

    return log_returns.dropna()
