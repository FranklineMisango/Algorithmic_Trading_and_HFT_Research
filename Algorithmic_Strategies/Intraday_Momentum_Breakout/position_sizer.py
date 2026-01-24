"""
Position Sizing Module

Implements volatility-targeted position sizing for constant risk exposure.

Core Logic:
- Target: 3% daily portfolio volatility
- Method: Size positions inversely to realized volatility
- Constraints: 1x - 8x leverage bounds

Position Size = (Target Volatility / Instrument Volatility) * Portfolio Value / Price

Key Features:
- EWMA volatility estimation (20-day span)
- Dynamic leverage adjustment
- Per-instrument position limits
- Portfolio allocation weights
- Margin requirements validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class PositionSizer:
    """
    Calculates position sizes based on volatility targeting.
    """
    
    def __init__(self, config: dict):
        """
        Initialize position sizer.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        
        # Position sizing parameters
        self.method = config['strategy']['position_sizing']['method']
        self.target_vol = config['strategy']['position_sizing']['target_daily_volatility'] / 100
        self.max_leverage = config['strategy']['position_sizing']['max_leverage']
        self.min_leverage = config['strategy']['position_sizing']['min_leverage']
        self.max_contracts = config['strategy']['position_sizing']['max_contracts_per_instrument']
        self.vol_estimation = config['strategy']['position_sizing']['volatility_estimation']
        self.ewma_span = config['strategy']['position_sizing']['ewma_span']
        
        # Portfolio allocation
        self.allocation = config['strategy']['portfolio']['allocation']
        self.initial_capital = config['strategy']['portfolio']['initial_capital']
        
        # Margin requirements
        self.margin = config['strategy']['risk_management']['margin_requirements']
        
        # Instrument specifications
        self.instruments = config['data']['instruments']
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate log returns.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        pd.Series
            Log returns
        """
        return np.log(data['Close'] / data['Close'].shift(1))
    
    def calculate_ewma_volatility(self, returns: pd.Series, span: int = None) -> pd.Series:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        span : int, optional
            EWMA span (default from config)
            
        Returns
        -------
        pd.Series
            Annualized volatility
        """
        if span is None:
            span = self.ewma_span
        
        # Calculate EWMA variance
        ewma_var = returns.ewm(span=span, adjust=False).var()
        
        # Annualize (assuming 252 trading days, 78 5-min bars per day)
        # For intraday data: multiply by sqrt(bars per day)
        bars_per_day = 78  # Approximate 5-min bars in RTH session
        annualized_vol = np.sqrt(ewma_var * bars_per_day * 252)
        
        return annualized_vol
    
    def calculate_rolling_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling window volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        window : int
            Rolling window in days
            
        Returns
        -------
        pd.Series
            Annualized volatility
        """
        # Calculate rolling std
        rolling_std = returns.rolling(window=window).std()
        
        # Annualize
        bars_per_day = 78
        annualized_vol = rolling_std * np.sqrt(bars_per_day * 252)
        
        return annualized_vol
    
    def calculate_realized_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate realized volatility using configured method.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        pd.Series
            Annualized volatility
        """
        # Calculate returns
        returns = self.calculate_returns(data)
        
        # Calculate volatility based on method
        if self.vol_estimation == 'ewma':
            volatility = self.calculate_ewma_volatility(returns, self.ewma_span)
        elif self.vol_estimation == 'rolling':
            volatility = self.calculate_rolling_volatility(returns, window=self.ewma_span)
        else:
            raise ValueError(f"Unknown volatility estimation method: {self.vol_estimation}")
        
        return volatility
    
    def calculate_contract_value(self, price: float, symbol: str) -> float:
        """
        Calculate notional value of one contract.
        
        Parameters
        ----------
        price : float
            Current price
        symbol : str
            Instrument symbol
            
        Returns
        -------
        float
            Notional value per contract
        """
        # Get multiplier for instrument
        instrument_config = next(
            (inst for inst in self.instruments if inst['symbol'] == symbol), 
            None
        )
        
        if instrument_config is None:
            raise ValueError(f"Unknown instrument: {symbol}")
        
        multiplier = instrument_config.get('multiplier', 1)
        return price * multiplier
    
    def calculate_position_size_volatility_target(
        self, 
        portfolio_value: float,
        price: float,
        instrument_vol: float,
        symbol: str,
        allocation_weight: float
    ) -> int:
        """
        Calculate position size using volatility targeting.
        
        Formula:
        contracts = (target_vol * allocation_weight * portfolio_value) / (instrument_vol * contract_value)
        
        Parameters
        ----------
        portfolio_value : float
            Current portfolio value
        price : float
            Current instrument price
        instrument_vol : float
            Instrument volatility (annualized)
        symbol : str
            Instrument symbol
        allocation_weight : float
            Portfolio allocation weight (0-1)
            
        Returns
        -------
        int
            Number of contracts to hold
        """
        # Calculate contract value
        contract_value = self.calculate_contract_value(price, symbol)
        
        # Avoid division by zero
        if instrument_vol == 0 or np.isnan(instrument_vol):
            return 0
        
        # Calculate target position size
        target_dollar_vol = self.target_vol * allocation_weight * portfolio_value
        position_dollar_value = target_dollar_vol / instrument_vol
        
        # Convert to contracts
        num_contracts = position_dollar_value / contract_value
        
        # Round to integer
        num_contracts = int(round(num_contracts))
        
        # Apply limits
        num_contracts = min(num_contracts, self.max_contracts)
        num_contracts = max(num_contracts, 0)
        
        return num_contracts
    
    def calculate_leverage(self, position_value: float, portfolio_value: float) -> float:
        """
        Calculate current leverage.
        
        Parameters
        ----------
        position_value : float
            Total notional position value
        portfolio_value : float
            Portfolio value
            
        Returns
        -------
        float
            Leverage ratio
        """
        if portfolio_value == 0:
            return 0.0
        return position_value / portfolio_value
    
    def check_margin_requirement(self, num_contracts: int, symbol: str) -> bool:
        """
        Check if position meets margin requirements.
        
        Parameters
        ----------
        num_contracts : int
            Number of contracts
        symbol : str
            Instrument symbol
            
        Returns
        -------
        bool
            True if margin requirements are met
        """
        # Get margin requirement
        if symbol == 'ES':
            margin_per_contract = self.margin['ES']
        elif symbol == 'NQ':
            margin_per_contract = self.margin['NQ']
        else:
            margin_per_contract = 10000  # Default
        
        total_margin = num_contracts * margin_per_contract
        
        # Check if we have enough capital (simplified check)
        return total_margin <= self.initial_capital
    
    def calculate_portfolio_positions(
        self,
        es_data: pd.DataFrame,
        nq_data: pd.DataFrame,
        portfolio_value: float = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate position sizes for entire portfolio.
        
        Portfolio allocation:
        - 50% NQ momentum
        - 25% ES momentum  
        - 25% NQ long-only
        
        Parameters
        ----------
        es_data : pd.DataFrame
            ES data with signals
        nq_data : pd.DataFrame
            NQ data with signals
        portfolio_value : float, optional
            Current portfolio value (default: initial capital)
            
        Returns
        -------
        dict
            Position sizes for each strategy
        """
        print("="*60)
        print("POSITION SIZING")
        print("="*60)
        
        if portfolio_value is None:
            portfolio_value = self.initial_capital
        
        print(f"Portfolio value: ${portfolio_value:,.0f}")
        print(f"Target volatility: {self.target_vol*100:.1f}%")
        print(f"Leverage bounds: {self.min_leverage}x - {self.max_leverage}x")
        
        # Calculate realized volatility for each instrument
        print("\nCalculating volatility...")
        es_vol = self.calculate_realized_volatility(es_data)
        nq_vol = self.calculate_realized_volatility(nq_data)
        
        es_data['realized_vol'] = es_vol
        nq_data['realized_vol'] = nq_vol
        
        print(f"  ES avg vol: {es_vol.mean()*100:.1f}%")
        print(f"  NQ avg vol: {nq_vol.mean()*100:.1f}%")
        
        # Calculate position sizes
        print("\nCalculating position sizes...")
        
        # ES momentum (25% allocation)
        es_data['position_size'] = 0
        for idx in range(len(es_data)):
            row = es_data.iloc[idx]
            if row['signal'] != 0 and not np.isnan(row['realized_vol']):
                size = self.calculate_position_size_volatility_target(
                    portfolio_value=portfolio_value,
                    price=row['Close'],
                    instrument_vol=row['realized_vol'],
                    symbol='ES',
                    allocation_weight=self.allocation['ES_momentum'] / 100
                )
                es_data.at[row.name, 'position_size'] = size * row['signal']
        
        # NQ momentum (50% allocation)
        nq_momentum = nq_data.copy()
        nq_momentum['position_size'] = 0
        for idx in range(len(nq_momentum)):
            row = nq_momentum.iloc[idx]
            if row['signal'] != 0 and not np.isnan(row['realized_vol']):
                size = self.calculate_position_size_volatility_target(
                    portfolio_value=portfolio_value,
                    price=row['Close'],
                    instrument_vol=row['realized_vol'],
                    symbol='NQ',
                    allocation_weight=self.allocation['NQ_momentum'] / 100
                )
                nq_momentum.at[row.name, 'position_size'] = size * row['signal']
        
        # NQ long-only (25% allocation)
        nq_long = nq_data.copy()
        nq_long['position_size'] = 0
        for idx in range(len(nq_long)):
            row = nq_long.iloc[idx]
            if not np.isnan(row['realized_vol']) and row['realized_vol'] > 0:
                size = self.calculate_position_size_volatility_target(
                    portfolio_value=portfolio_value,
                    price=row['Close'],
                    instrument_vol=row['realized_vol'],
                    symbol='NQ',
                    allocation_weight=self.allocation['NQ_long_only'] / 100
                )
                nq_long.at[row.name, 'position_size'] = size  # Always long
        
        # Calculate leverage
        es_data['notional_value'] = abs(es_data['position_size']) * es_data['Close'] * 50
        nq_momentum['notional_value'] = abs(nq_momentum['position_size']) * nq_momentum['Close'] * 20
        nq_long['notional_value'] = abs(nq_long['position_size']) * nq_long['Close'] * 20
        
        es_data['leverage'] = es_data['notional_value'] / portfolio_value
        nq_momentum['leverage'] = nq_momentum['notional_value'] / portfolio_value
        nq_long['leverage'] = nq_long['notional_value'] / portfolio_value
        
        # Apply leverage limits
        for data in [es_data, nq_momentum, nq_long]:
            # Scale down if exceeds max leverage
            over_leveraged = data['leverage'] > self.max_leverage
            if over_leveraged.any():
                scale_factor = self.max_leverage / data.loc[over_leveraged, 'leverage']
                data.loc[over_leveraged, 'position_size'] = (
                    data.loc[over_leveraged, 'position_size'] * scale_factor
                ).round().astype(int)
                data.loc[over_leveraged, 'leverage'] = self.max_leverage
        
        # Statistics
        print("\nPosition Statistics:")
        print(f"  ES momentum:")
        print(f"    Avg position: {es_data['position_size'].abs().mean():.1f} contracts")
        print(f"    Max position: {es_data['position_size'].abs().max():.0f} contracts")
        print(f"    Avg leverage: {es_data['leverage'].mean():.2f}x")
        
        print(f"  NQ momentum:")
        print(f"    Avg position: {nq_momentum['position_size'].abs().mean():.1f} contracts")
        print(f"    Max position: {nq_momentum['position_size'].abs().max():.0f} contracts")
        print(f"    Avg leverage: {nq_momentum['leverage'].mean():.2f}x")
        
        print(f"  NQ long-only:")
        print(f"    Avg position: {nq_long['position_size'].abs().mean():.1f} contracts")
        print(f"    Max position: {nq_long['position_size'].abs().max():.0f} contracts")
        print(f"    Avg leverage: {nq_long['leverage'].mean():.2f}x")
        
        print("="*60)
        
        return {
            'ES_momentum': es_data,
            'NQ_momentum': nq_momentum,
            'NQ_long_only': nq_long
        }


def main():
    """
    Test position sizer.
    """
    import yaml
    from noise_area import NoiseAreaCalculator
    from signal_generator import SignalGenerator
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate sample data
    print("Generating sample data...")
    dates = pd.date_range('2023-01-01 09:30', '2023-12-31 16:00', freq='5min')
    n = len(dates)
    
    np.random.seed(42)
    
    # ES data
    es_price = 4500 + np.cumsum(np.random.randn(n) * 2) + np.random.randn(n) * 10
    es_data = pd.DataFrame({
        'Open': es_price + np.random.randn(n) * 2,
        'High': es_price + abs(np.random.randn(n) * 5),
        'Low': es_price - abs(np.random.randn(n) * 5),
        'Close': es_price,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # NQ data
    nq_price = 15000 + np.cumsum(np.random.randn(n) * 5) + np.random.randn(n) * 20
    nq_data = pd.DataFrame({
        'Open': nq_price + np.random.randn(n) * 5,
        'High': nq_price + abs(np.random.randn(n) * 10),
        'Low': nq_price - abs(np.random.randn(n) * 10),
        'Close': nq_price,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Calculate noise area and signals
    calculator = NoiseAreaCalculator(config)
    signal_gen = SignalGenerator(config)
    
    es_data = calculator.calculate_noise_area(es_data)
    es_data = calculator.identify_breakouts(es_data)
    es_data = signal_gen.generate_signals(es_data)
    
    nq_data = calculator.calculate_noise_area(nq_data)
    nq_data = calculator.identify_breakouts(nq_data)
    nq_data = signal_gen.generate_signals(nq_data)
    
    # Calculate position sizes
    sizer = PositionSizer(config)
    portfolio = sizer.calculate_portfolio_positions(es_data, nq_data)
    
    # Save
    portfolio['ES_momentum'].to_csv('results/es_positions_sample.csv')
    portfolio['NQ_momentum'].to_csv('results/nq_momentum_positions_sample.csv')
    portfolio['NQ_long_only'].to_csv('results/nq_long_positions_sample.csv')
    
    print("\nPosition sizing complete. Results saved to results/")


if __name__ == "__main__":
    main()
