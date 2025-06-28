"""
Portfolio constraint handling and validation.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioConstraints:
    """
    Portfolio constraint manager for handling various types of constraints.
    """
    
    def __init__(self, n_assets: int, asset_names: Optional[List[str]] = None):
        """
        Initialize constraint manager.
        
        Args:
            n_assets: Number of assets in the portfolio
            asset_names: List of asset names
        """
        self.n_assets = n_assets
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(n_assets)]
        
        # Initialize constraint storage
        self.bounds = [(0.0, 1.0)] * n_assets  # Default: long-only, max 100%
        self.sector_limits = {}
        self.position_limits = {}
        self.turnover_limits = {}
        self.leverage_limits = {}
        self.regulatory_constraints = {}
        
        logger.info(f"Initialized constraint manager for {n_assets} assets")
    
    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """
        Set bounds for individual assets.
        
        Args:
            bounds: List of (min_weight, max_weight) tuples
        """
        if len(bounds) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} bounds, got {len(bounds)}")
        
        self.bounds = bounds
        logger.info("Updated asset bounds")
    
    def set_long_only(self) -> None:
        """Set long-only constraints (no short selling)."""
        self.bounds = [(0.0, 1.0)] * self.n_assets
        logger.info("Set long-only constraints")
    
    def set_long_short(self, max_short: float = 0.5) -> None:
        """
        Set long-short constraints.
        
        Args:
            max_short: Maximum short position (as fraction)
        """
        self.bounds = [(-max_short, 1.0)] * self.n_assets
        logger.info(f"Set long-short constraints with max short {max_short}")
    
    def set_sector_limits(self, sector_assignments: Dict[str, str], 
                         sector_limits: Dict[str, float]) -> None:
        """
        Set sector exposure limits.
        
        Args:
            sector_assignments: Dictionary mapping assets to sectors
            sector_limits: Dictionary mapping sectors to maximum weights
        """
        self.sector_assignments = sector_assignments
        self.sector_limits = sector_limits
        logger.info(f"Set sector limits: {sector_limits}")
    
    def set_position_limits(self, min_position: float = 0.0, 
                           max_position: float = 0.1) -> None:
        """
        Set individual position limits.
        
        Args:
            min_position: Minimum position size
            max_position: Maximum position size
        """
        self.position_limits = {
            'min': min_position,
            'max': max_position
        }
        self.bounds = [(min_position, max_position)] * self.n_assets
        logger.info(f"Set position limits: {min_position} to {max_position}")
    
    def set_turnover_limit(self, max_turnover: float = 0.2, 
                          current_weights: Optional[np.ndarray] = None) -> None:
        """
        Set turnover constraints.
        
        Args:
            max_turnover: Maximum allowed turnover
            current_weights: Current portfolio weights
        """
        self.turnover_limits = {
            'max_turnover': max_turnover,
            'current_weights': current_weights
        }
        logger.info(f"Set turnover limit: {max_turnover}")
    
    def set_leverage_limit(self, max_leverage: float = 1.0) -> None:
        """
        Set leverage constraints.
        
        Args:
            max_leverage: Maximum leverage (1.0 = no leverage)
        """
        self.leverage_limits = {'max_leverage': max_leverage}
        logger.info(f"Set leverage limit: {max_leverage}")
    
    def set_ucits_constraints(self) -> None:
        """Set UCITS-compliant constraints."""
        self.regulatory_constraints['ucits'] = {
            'max_single_position': 0.05,  # 5% max per position
            'max_aggregate_large': 0.4,   # 40% max for positions > 5%
            'max_aggregate_small': 0.6    # 60% max for positions < 5%
        }
        logger.info("Set UCITS regulatory constraints")
    
    def set_40_act_constraints(self) -> None:
        """Set 40 Act fund constraints."""
        self.regulatory_constraints['40_act'] = {
            'max_single_position': 0.25,  # 25% max per position
            'max_sector': 0.25,           # 25% max per sector
            'diversification_requirement': 0.75  # 75% in diversified holdings
        }
        logger.info("Set 40 Act regulatory constraints")
    
    def get_constraint_dict(self) -> Dict[str, Any]:
        """
        Get all constraints as a dictionary for optimization.
        
        Returns:
            Dictionary with constraint parameters
        """
        constraints = {
            'bounds': self.bounds,
            'sum_to_one': True
        }
        
        # Add sector constraints if defined
        if hasattr(self, 'sector_assignments') and self.sector_limits:
            constraints['sector_assignments'] = self.sector_assignments
            constraints['sector_limits'] = self.sector_limits
        
        # Add position limits
        if self.position_limits:
            constraints['position_limits'] = self.position_limits
        
        # Add turnover limits
        if self.turnover_limits:
            constraints['turnover_limits'] = self.turnover_limits
        
        # Add leverage limits
        if self.leverage_limits:
            constraints['leverage_limits'] = self.leverage_limits
        
        # Add regulatory constraints
        if self.regulatory_constraints:
            constraints['regulatory_constraints'] = self.regulatory_constraints
        
        return constraints
    
    def validate_weights(self, weights: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate portfolio weights against all constraints.
        
        Args:
            weights: Portfolio weights to validate
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check bounds
        for i, (weight, (min_w, max_w)) in enumerate(zip(weights, self.bounds)):
            if weight < min_w or weight > max_w:
                violations.append(f"Asset {i} weight {weight:.4f} outside bounds [{min_w}, {max_w}]")
        
        # Check sum to one
        if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            violations.append(f"Weights sum to {np.sum(weights):.4f}, should be 1.0")
        
        # Check sector limits
        if hasattr(self, 'sector_assignments') and self.sector_limits:
            sector_weights = {}
            for asset, weight in zip(self.asset_names, weights):
                if asset in self.sector_assignments:
                    sector = self.sector_assignments[asset]
                    sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            for sector, weight in sector_weights.items():
                if sector in self.sector_limits and weight > self.sector_limits[sector]:
                    violations.append(f"Sector {sector} weight {weight:.4f} exceeds limit {self.sector_limits[sector]}")
        
        # Check regulatory constraints
        if 'ucits' in self.regulatory_constraints:
            ucits = self.regulatory_constraints['ucits']
            
            # Check single position limit
            if np.any(weights > ucits['max_single_position']):
                violations.append(f"UCITS: Position exceeds {ucits['max_single_position']} limit")
            
            # Check aggregate limits
            large_positions = weights[weights > 0.05]
            small_positions = weights[weights <= 0.05]
            
            if np.sum(large_positions) > ucits['max_aggregate_large']:
                violations.append(f"UCITS: Large positions aggregate {np.sum(large_positions):.4f} exceeds {ucits['max_aggregate_large']}")
            
            if np.sum(small_positions) > ucits['max_aggregate_small']:
                violations.append(f"UCITS: Small positions aggregate {np.sum(small_positions):.4f} exceeds {ucits['max_aggregate_small']}")
        
        if '40_act' in self.regulatory_constraints:
            act40 = self.regulatory_constraints['40_act']
            
            # Check single position limit
            if np.any(weights > act40['max_single_position']):
                violations.append(f"40 Act: Position exceeds {act40['max_single_position']} limit")
            
            # Check sector limits
            if hasattr(self, 'sector_assignments'):
                sector_weights = {}
                for asset, weight in zip(self.asset_names, weights):
                    if asset in self.sector_assignments:
                        sector = self.sector_assignments[asset]
                        sector_weights[sector] = sector_weights.get(sector, 0) + weight
                
                for sector, weight in sector_weights.items():
                    if weight > act40['max_sector']:
                        violations.append(f"40 Act: Sector {sector} weight {weight:.4f} exceeds {act40['max_sector']} limit")
        
        return len(violations) == 0, violations
    
    def calculate_turnover(self, new_weights: np.ndarray, 
                          current_weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            new_weights: New portfolio weights
            current_weights: Current portfolio weights (if None, uses stored weights)
            
        Returns:
            Turnover as a fraction
        """
        if current_weights is None:
            current_weights = self.turnover_limits.get('current_weights')
            if current_weights is None:
                return 0.0
        
        turnover = np.sum(np.abs(new_weights - current_weights)) / 2
        return turnover
    
    def check_turnover_limit(self, new_weights: np.ndarray, 
                           current_weights: Optional[np.ndarray] = None) -> bool:
        """
        Check if turnover is within limits.
        
        Args:
            new_weights: New portfolio weights
            current_weights: Current portfolio weights
            
        Returns:
            True if turnover is within limits
        """
        if not self.turnover_limits:
            return True
        
        turnover = self.calculate_turnover(new_weights, current_weights)
        max_turnover = self.turnover_limits['max_turnover']
        
        return turnover <= max_turnover


def create_constraint_functions(constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create constraint functions for scipy.optimize.
    
    Args:
        constraints: Dictionary of constraints
        
    Returns:
        List of constraint dictionaries for scipy.optimize
    """
    constraint_functions = []
    
    # Sum to one constraint
    if constraints.get('sum_to_one', True):
        constraint_functions.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
    
    # Sector constraints
    if 'sector_assignments' in constraints and 'sector_limits' in constraints:
        sector_assignments = constraints['sector_assignments']
        sector_limits = constraints['sector_limits']
        
        for sector, limit in sector_limits.items():
            sector_assets = [asset for asset, s in sector_assignments.items() if s == sector]
            if sector_assets:
                constraint_functions.append({
                    'type': 'ineq',
                    'fun': lambda x, assets=sector_assets: limit - sum(x[i] for i in assets)
                })
    
    # Turnover constraints
    if 'turnover_limits' in constraints:
        current_weights = constraints['turnover_limits'].get('current_weights')
        max_turnover = constraints['turnover_limits'].get('max_turnover')
        
        if current_weights is not None and max_turnover is not None:
            constraint_functions.append({
                'type': 'ineq',
                'fun': lambda x: max_turnover - np.sum(np.abs(x - current_weights)) / 2
            })
    
    # Leverage constraints
    if 'leverage_limits' in constraints:
        max_leverage = constraints['leverage_limits'].get('max_leverage')
        if max_leverage is not None:
            constraint_functions.append({
                'type': 'ineq',
                'fun': lambda x: max_leverage - np.sum(np.abs(x))
            })
    
    return constraint_functions


def get_preset_constraints(preset: str, n_assets: int) -> 'PortfolioConstraints':
    """
    Get preset constraint configurations.
    
    Args:
        preset: Preset name ('conservative', 'moderate', 'aggressive', 'ucits', '40_act')
        n_assets: Number of assets
        
    Returns:
        PortfolioConstraints object with preset configuration
    """
    constraints = PortfolioConstraints(n_assets)
    
    if preset == 'conservative':
        constraints.set_long_only()
        constraints.set_position_limits(min_position=0.02, max_position=0.08)
        constraints.set_turnover_limit(max_turnover=0.1)
        
    elif preset == 'moderate':
        constraints.set_long_only()
        constraints.set_position_limits(min_position=0.01, max_position=0.15)
        constraints.set_turnover_limit(max_turnover=0.2)
        
    elif preset == 'aggressive':
        constraints.set_long_short(max_short=0.3)
        constraints.set_position_limits(min_position=-0.3, max_position=0.2)
        constraints.set_turnover_limit(max_turnover=0.5)
        
    elif preset == 'ucits':
        constraints.set_ucits_constraints()
        constraints.set_long_only()
        
    elif preset == '40_act':
        constraints.set_40_act_constraints()
        constraints.set_long_only()
        
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    return constraints 