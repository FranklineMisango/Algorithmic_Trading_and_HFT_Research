"""
Canadian Bond Data Acquisition Module

Fetches Canadian Government bond data from Bloomberg, Bank of Canada,
and alternative sources for the day count arbitrage strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from pathlib import Path

# Optional imports (handle gracefully if not available)
try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    logging.warning("Databento API not available.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not available.")


class CanadianBondDataAcquisition:
    """
    Acquire Canadian Government bond data for day count arbitrage strategy.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data acquisition with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Data source settings
        self.data_source = self.config['data_sources']['primary']['provider']
        self.fields = self.config['data_sources']['primary']['fields']
        
        # Universe filters
        self.min_maturity = self.config['universe']['min_maturity_years']
        self.max_maturity = self.config['universe']['max_maturity_years']
        
        # Bloomberg session (if available)
        self.bb_session = None
        if BLOOMBERG_AVAILABLE:
            self._init_bloomberg()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_level = self.config['logging']['level']
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'data_acquisition.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _init_bloomberg(self):
        """Initialize Bloomberg API session."""
        try:
            session_options = SessionOptions()
            session_options.setServerHost('localhost')
            session_options.setServerPort(8194)
            
            self.bb_session = Session(session_options)
            if not self.bb_session.start():
                raise Exception("Failed to start Bloomberg session")
            
            if not self.bb_session.openService("//blp/refdata"):
                raise Exception("Failed to open Bloomberg reference data service")
            
            self.logger.info("Bloomberg session initialized successfully")
        except Exception as e:
            self.logger.error(f"Bloomberg initialization failed: {e}")
            self.bb_session = None
    
    def _init_databento(self):
        """Initialize Databento API client."""
        try:
            import os
            api_key = os.getenv('DATABENTO_API_KEY')
            if not api_key:
                raise ValueError("DATABENTO_API_KEY environment variable not set")
            
            self.db_client = db.Historical(api_key)
            self.logger.info("Databento client initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Databento initialization failed: {e}")
            return False
    
    def get_canadian_government_bonds(self, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve universe of Canadian Government bonds.
        
        Args:
            as_of_date: Date for point-in-time universe (default: today)
        
        Returns:
            DataFrame with bond identifiers and key characteristics
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        self.logger.info(f"Fetching Canadian Government bonds as of {as_of_date}")
        
        if self.data_source == "databento" and DATABENTO_AVAILABLE:
            return self._get_bonds_from_databento(as_of_date)
        else:
            return self._get_bonds_from_bank_of_canada(as_of_date)
    
    def _get_bonds_from_bloomberg(self, as_of_date: datetime) -> pd.DataFrame:
        """
        Fetch bond universe from Bloomberg.
        
        Uses SRCH functionality to find Canadian Government bonds.
        """
        self.logger.info("Fetching bonds from Bloomberg")
        
        # Bloomberg search criteria for Canadian Government bonds
        search_criteria = {
            'Country': 'CA',
            'Issuer Type': 'Government',
            'Coupon Type': 'FIXED',
            'Currency': 'CAD',
            'Market Sector': 'Govt'
        }
        
        try:
            # Example Bloomberg request (simplified)
            # In practice, use proper Bloomberg API calls
            bonds = self._execute_bloomberg_search(search_criteria, as_of_date)
            
            # Filter by maturity
            bonds = self._filter_by_maturity(bonds, as_of_date)
            
            # Exclude callables and strips
            if self.config['universe']['exclude_callable']:
                bonds = bonds[bonds['callable'] == False]
            if self.config['universe']['exclude_strips']:
                bonds = bonds[bonds['security_type'] != 'STRIP']
            
            self.logger.info(f"Retrieved {len(bonds)} Canadian Government bonds")
            return bonds
            
        except Exception as e:
            self.logger.error(f"Bloomberg bond fetch failed: {e}")
            return pd.DataFrame()
    
    def _get_bonds_from_databento(self, as_of_date: datetime) -> pd.DataFrame:
        """
        Fetch Canadian Government bond futures data from Databento.
        
        Retrieves CGB (Canadian Government Bond) futures contracts.
        """
        if not DATABENTO_AVAILABLE:
            self.logger.error("Databento not available")
            return pd.DataFrame()
        
        self.logger.info("Fetching bonds from Databento")
        
        try:
            if not self._init_databento():
                return pd.DataFrame()
            
            # Fetch CGB futures data
            symbols = self.config['data_sources']['primary'].get('symbols', 'CGB*')
            
            # Query historical data for the date
            data = self.db_client.timeseries(
                dataset='XCAN.ITCH',
                symbols=symbols,
                start=as_of_date.strftime('%Y-%m-%d'),
                end=as_of_date.strftime('%Y-%m-%d'),
                schema='ohlcv'
            )
            
            # Convert to DataFrame
            df = data.to_pandas()
            
            # Extract unique contracts
            if len(df) > 0:
                bonds = df.groupby('symbol').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()
                
                self.logger.info(f"Retrieved {len(bonds)} CGB contracts from Databento")
                return bonds
            else:
                self.logger.warning(f"No data found for {as_of_date}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Databento fetch failed: {e}")
            return pd.DataFrame()
    
    def _get_bonds_from_bank_of_canada(self, as_of_date: datetime) -> pd.DataFrame:
        """
        Fetch bond data from Bank of Canada (alternative source).
        
        Uses Bank of Canada's public data API.
        """
        if not REQUESTS_AVAILABLE:
            self.logger.error("Requests library required for Bank of Canada API")
            return pd.DataFrame()
        
        self.logger.info("Fetching bonds from Bank of Canada")
        
        # Bank of Canada Valet API endpoint
        base_url = self.config['data_sources']['alternative']['endpoint']
        
        try:
            # Fetch government bond data
            # Note: Actual endpoint structure may vary
            url = f"{base_url}observations/group/bond_yields/json"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse and structure the data
            bonds = self._parse_bank_of_canada_data(data, as_of_date)
            
            self.logger.info(f"Retrieved {len(bonds)} bonds from Bank of Canada")
            return bonds
            
        except Exception as e:
            self.logger.error(f"Bank of Canada fetch failed: {e}")
            return pd.DataFrame()
    
    def get_bond_details(self, bond_identifiers: List[str], 
                        as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch detailed bond data including prices, yields, and coupon schedules.
        
        Args:
            bond_identifiers: List of bond ISINs or CUSIPs
            as_of_date: Date for historical data (default: latest)
        
        Returns:
            DataFrame with comprehensive bond details
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        self.logger.info(f"Fetching details for {len(bond_identifiers)} bonds")
        
        if self.data_source == "databento" and DATABENTO_AVAILABLE:
            return self._get_bond_details_databento(bond_identifiers, as_of_date)
        else:
            return self._get_bond_details_alternative(bond_identifiers, as_of_date)
    
    def _get_bond_details_bloomberg(self, identifiers: List[str], 
                                   as_of_date: datetime) -> pd.DataFrame:
        """Fetch detailed bond data from Bloomberg."""
        service = self.bb_session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")
        
        # Add securities
        for identifier in identifiers:
            request.append("securities", identifier)
        
        # Add fields from config
        for field in self.fields:
            request.append("fields", field)
        
        # Override for historical date
        if as_of_date.date() != datetime.now().date():
            overrides = request.getElement("overrides")
            override = overrides.appendElement()
            override.setElement("fieldId", "REFERENCE_DATE")
            override.setElement("value", as_of_date.strftime("%Y%m%d"))
        
        try:
            self.bb_session.sendRequest(request)
            
            # Collect responses
            bond_data = []
            while True:
                event = self.bb_session.nextEvent(500)
                
                if event.eventType() == Event.RESPONSE or event.eventType() == Event.PARTIAL_RESPONSE:
                    for msg in event:
                        security_data = msg.getElement("securityData")
                        
                        for i in range(security_data.numValues()):
                            field_data = security_data.getValue(i).getElement("fieldData")
                            
                            bond_info = {
                                'identifier': security_data.getValue(i).getElement("security").getValue()
                            }
                            
                            # Extract all fields
                            for field in self.fields:
                                if field_data.hasElement(field):
                                    bond_info[field] = field_data.getElement(field).getValue()
                            
                            bond_data.append(bond_info)
                
                if event.eventType() == Event.RESPONSE:
                    break
            
            df = pd.DataFrame(bond_data)
            self.logger.info(f"Retrieved details for {len(df)} bonds from Bloomberg")
            return df
            
        except Exception as e:
            self.logger.error(f"Bloomberg bond details fetch failed: {e}")
            return pd.DataFrame()
    
    def _get_bond_details_alternative(self, identifiers: List[str], 
                                     as_of_date: datetime) -> pd.DataFrame:
        """Fetch bond details from alternative sources."""
        # Placeholder for alternative data sources
        # Could integrate with Refinitiv, IEX Cloud, or other providers
        
        self.logger.warning("Alternative bond detail source not fully implemented")
        
        # Return mock data structure for development
        return pd.DataFrame({
            'identifier': identifiers,
            'PX_DIRTY': np.nan,
            'PX_CLEAN': np.nan,
            'YLD_YTM_MID': np.nan,
            'DUR_ADJ_MID': np.nan,
            'INT_ACC': np.nan,
            'NXT_CPN_DT': pd.NaT,
            'PREV_CPN_DT': pd.NaT,
            'CPN': np.nan
        })
    def _get_bond_details_databento(self, identifiers: List[str],
                                   as_of_date: datetime) -> pd.DataFrame:
        """Fetch detailed bond data from Databento."""
        if not DATABENTO_AVAILABLE:
            self.logger.error("Databento not available")
            return pd.DataFrame()

        try:
            if not self._init_databento():
                return pd.DataFrame()

            # Fetch tick data for each symbol
            all_data = []

            for symbol in identifiers:
                try:
                    data = self.db_client.timeseries(
                        dataset='XCAN.ITCH',
                        symbols=symbol,
                        start=as_of_date.strftime('%Y-%m-%d'),
                        end=as_of_date.strftime('%Y-%m-%d'),
                        schema='trades'
                    )

                    df = data.to_pandas()
                    if len(df) > 0:
                        # Aggregate to get OHLCV
                        agg_data = {
                            'symbol': symbol,
                            'open': df['price'].iloc[0],
                            'high': df['price'].max(),
                            'low': df['price'].min(),
                            'close': df['price'].iloc[-1],
                            'volume': df['size'].sum(),
                            'bid': df.get('bid', np.nan).iloc[-1] if 'bid' in df else np.nan,
                            'ask': df.get('ask', np.nan).iloc[-1] if 'ask' in df else np.nan,
                        }
                        all_data.append(agg_data)

                except Exception as e:
                    self.logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue

            if all_data:
                df = pd.DataFrame(all_data)
                self.logger.info(f"Retrieved details for {len(df)} bonds from Databento")
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Databento bond details fetch failed: {e}")
            return pd.DataFrame()


    
    def get_coupon_schedule(self, bond_identifier: str) -> pd.DataFrame:
        """
        Retrieve complete coupon payment schedule for a bond.
        
        Args:
            bond_identifier: Bond ISIN or CUSIP
        
        Returns:
            DataFrame with coupon dates and amounts
        """
        self.logger.info(f"Fetching coupon schedule for {bond_identifier}")
        
        # This would typically call Bloomberg DES or similar
        # Placeholder implementation
        
        return pd.DataFrame(columns=['payment_date', 'coupon_amount', 'period_days'])
    
    def _filter_by_maturity(self, bonds: pd.DataFrame, as_of_date: datetime) -> pd.DataFrame:
        """Filter bonds by maturity constraints."""
        if 'maturity_date' not in bonds.columns:
            return bonds
        
        bonds['years_to_maturity'] = (
            (bonds['maturity_date'] - as_of_date).dt.days / 365.25
        )
        
        return bonds[
            (bonds['years_to_maturity'] >= self.min_maturity) &
            (bonds['years_to_maturity'] <= self.max_maturity)
        ]
    
    def _execute_bloomberg_search(self, criteria: Dict, as_of_date: datetime) -> pd.DataFrame:
        """Execute Bloomberg bond search (placeholder)."""
        # In production, implement proper Bloomberg SRCH functionality
        self.logger.warning("Bloomberg search not fully implemented")
        return pd.DataFrame()
    
    def _parse_bank_of_canada_data(self, data: Dict, as_of_date: datetime) -> pd.DataFrame:
        """Parse Bank of Canada API response."""
        # Implementation depends on actual API structure
        self.logger.warning("Bank of Canada parser not fully implemented")
        return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save bond data to disk."""
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        filepath = data_dir / filename
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved data to {filepath}")
    
    def load_cached_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load previously cached bond data."""
        filepath = Path('data') / filename
        
        if filepath.exists():
            self.logger.info(f"Loading cached data from {filepath}")
            return pd.read_csv(filepath, parse_dates=['NXT_CPN_DT', 'PREV_CPN_DT'])
        else:
            self.logger.warning(f"Cached file {filepath} not found")
            return None


if __name__ == "__main__":
    # Example usage
    acquirer = CanadianBondDataAcquisition()
    
    # Get universe of Canadian Government bonds
    bonds = acquirer.get_canadian_government_bonds()
    print(f"Found {len(bonds)} Canadian Government bonds")
    
    # Save to cache
    acquirer.save_data(bonds, 'canadian_govt_bonds.csv')
