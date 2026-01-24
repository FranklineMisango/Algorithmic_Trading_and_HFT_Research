"""Risk management module for position sizing and exposure control."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class RiskManager:
    """
    Manages risk limits and position sizing.
    
    Key features:
    - Position size limits
    - Sector exposure limits
    - Volatility-based position sizing
    - Real-time risk monitoring
    """
    
    def __init__(
        self,
        max_position_size: float = 0.04,
        max_sector_exposure: float = 0.25,
        max_gross_leverage: float = 2.0,
        max_net_exposure: float = 0.10
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum size per position (fraction of capital)
            max_sector_exposure: Maximum exposure to any sector
            max_gross_leverage: Maximum gross leverage
            max_net_exposure: Maximum net market exposure
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_gross_leverage = max_gross_leverage
        self.max_net_exposure = max_net_exposure
        
        logger.info(
            f"RiskManager initialized: max position={max_position_size*100:.1f}%, "
            f"max sector={max_sector_exposure*100:.1f}%"
        )
    
    def check_position_limits(self, portfolio: pd.DataFrame, total_capital: float) -> List[str]:
        """
        Check if positions exceed size limits.
        
        Args:
            portfolio: Portfolio DataFrame
            total_capital: Total capital
            
        Returns:
            List of warnings
        """
        warnings = []
        
        for _, position in portfolio.iterrows():
            position_size = position['capital'] / total_capital
            
            if position_size > self.max_position_size:
                warnings.append(
                    f"Position {position['ticker']} exceeds limit: "
                    f"{position_size*100:.2f}% > {self.max_position_size*100:.2f}%"
                )
        
        return warnings
    
    def check_exposure_limits(self, portfolio: pd.DataFrame, total_capital: float) -> List[str]:
        """
        Check portfolio exposure limits.
        
        Args:
            portfolio: Portfolio DataFrame
            total_capital: Total capital
            
        Returns:
            List of warnings
        """
        warnings = []
        
        # Calculate exposures
        long_exposure = portfolio[portfolio['side'] == 'long']['capital'].sum()
        short_exposure = portfolio[portfolio['side'] == 'short']['capital'].sum()
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = abs(long_exposure - short_exposure)
        
        gross_leverage = gross_exposure / total_capital
        net_leverage = net_exposure / total_capital
        
        # Check limits
        if gross_leverage > self.max_gross_leverage:
            warnings.append(
                f"Gross leverage exceeds limit: {gross_leverage:.2f} > {self.max_gross_leverage:.2f}"
            )
        
        if net_leverage > self.max_net_exposure:
            warnings.append(
                f"Net exposure exceeds limit: {net_leverage*100:.2f}% > {self.max_net_exposure*100:.2f}%"
            )
        
        return warnings
    
    def validate_portfolio(
        self,
        portfolio: pd.DataFrame,
        total_capital: float
    ) -> tuple[bool, List[str]]:
        """
        Validate portfolio against all risk limits.
        
        Args:
            portfolio: Portfolio DataFrame
            total_capital: Total capital
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check position limits
        warnings.extend(self.check_position_limits(portfolio, total_capital))
        
        # Check exposure limits
        warnings.extend(self.check_exposure_limits(portfolio, total_capital))
        
        is_valid = len(warnings) == 0
        
        if not is_valid:
            logger.warning(f"Portfolio validation failed: {len(warnings)} warnings")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("Portfolio validation passed")
        
        return is_valid, warnings


if __name__ == "__main__":
    # Example usage
    risk_manager = RiskManager()
    
    # Sample portfolio
    portfolio = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT'],
        'side': ['long', 'long', 'short'],
        'capital': [50000, 45000, 40000]
    })
    
    total_capital = 1000000
    
    is_valid, warnings = risk_manager.validate_portfolio(portfolio, total_capital)
    print(f"Portfolio valid: {is_valid}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
