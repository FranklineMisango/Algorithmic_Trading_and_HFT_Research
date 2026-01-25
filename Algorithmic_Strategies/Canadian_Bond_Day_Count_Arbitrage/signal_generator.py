"""
Real-time Signal Generator for Canadian Bond Day Count Arbitrage

Scans for live arbitrage opportunities and generates actionable signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
import logging
from pathlib import Path

from data_acquisition import CanadianBondDataAcquisition
from feature_engineering import CanadianBondFeatureEngineering


class SignalGenerator:
    """
    Real-time signal generator for monitoring arbitrage opportunities.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize signal generator."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        self.data_acquirer = CanadianBondDataAcquisition(config_path)
        self.feature_engineer = CanadianBondFeatureEngineering(config_path)
        
        # Alert thresholds
        self.min_profit_bps = self.config['signals']['min_profit_bps']
        self.spread_threshold = self.config['advanced'].get('spread_threshold_bps', 10.0)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def scan_for_opportunities(self, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Scan current bond universe for arbitrage opportunities.
        
        Args:
            as_of_date: Evaluation date (default: now)
        
        Returns:
            DataFrame with ranked opportunities
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        self.logger.info(f"Scanning for opportunities as of {as_of_date}")
        
        # Get bond universe
        bonds = self.data_acquirer.get_canadian_government_bonds(as_of_date)
        
        if bonds is None or len(bonds) == 0:
            self.logger.warning("No bonds retrieved")
            return pd.DataFrame()
        
        # Get detailed data
        bond_details = self.data_acquirer.get_bond_details(
            bonds['identifier'].tolist(),
            as_of_date
        )
        
        # Engineer features
        features = self.feature_engineer.engineer_features(bond_details, as_of_date)
        
        # Generate signals
        signals = self.feature_engineer.generate_signals(features)
        
        # Filter for active signals only
        opportunities = signals[signals['signal'] == 1].copy()
        
        # Rank by signal strength
        opportunities = opportunities.sort_values('signal_strength', ascending=False)
        
        # Add risk ratings
        opportunities['risk_rating'] = opportunities.apply(self._assess_risk, axis=1)
        
        self.logger.info(f"Found {len(opportunities)} opportunities")
        
        return opportunities
    
    def _assess_risk(self, bond: pd.Series) -> str:
        """Assess risk level for a given opportunity."""
        risk_score = 0
        
        # 182-day periods are riskier
        if bond['coupon_period_length'] == 182:
            risk_score += 2
        
        # Very close to coupon date is riskier
        if bond['days_to_next_coupon'] <= 1:
            risk_score += 2
        
        # Low expected profit is riskier
        if bond['arbitrage_profit_bps'] < 1.0:
            risk_score += 1
        
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_trade_recommendations(self, 
                                      opportunities: pd.DataFrame,
                                      max_recommendations: int = 5) -> List[Dict]:
        """
        Generate actionable trade recommendations.
        
        Args:
            opportunities: DataFrame with opportunities
            max_recommendations: Maximum number to recommend
        
        Returns:
            List of trade recommendation dictionaries
        """
        if len(opportunities) == 0:
            return []
        
        recommendations = []
        
        for i, (_, bond) in enumerate(opportunities.head(max_recommendations).iterrows()):
            rec = {
                'rank': i + 1,
                'bond_identifier': bond['identifier'],
                'coupon_rate': bond['CPN'],
                'next_coupon_date': bond['NXT_CPN_DT'],
                'days_to_coupon': bond['days_to_next_coupon'],
                'coupon_period_length': bond['coupon_period_length'],
                'expected_profit_bps': bond['arbitrage_profit_bps'],
                'current_price_clean': bond['PX_CLEAN'],
                'current_price_dirty': bond['PX_DIRTY'],
                'modified_duration': bond['DUR_ADJ_MID'],
                'risk_rating': bond['risk_rating'],
                'signal_strength': bond['signal_strength'],
                'recommendation': self._generate_recommendation_text(bond)
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_recommendation_text(self, bond: pd.Series) -> str:
        """Generate human-readable recommendation text."""
        text = (
            f"BUY {bond['identifier']} "
            f"(Coupon: {bond['CPN']:.2f}%, Period: {bond['coupon_period_length']} days). "
            f"Expected profit: {bond['arbitrage_profit_bps']:.2f} bps. "
            f"Entry: TODAY, Exit: {bond['NXT_CPN_DT'].date()} "
            f"({bond['days_to_next_coupon']} days). "
            f"Risk: {bond['risk_rating']}. "
        )
        
        if bond['coupon_period_length'] == 182:
            text += "⚠️ 182-day period - timing critical!"
        
        return text
    
    def monitor_spreads(self, opportunities: pd.DataFrame) -> Dict[str, bool]:
        """
        Monitor bid-ask spreads for potential crowding.
        
        Args:
            opportunities: DataFrame with current opportunities
        
        Returns:
            Dictionary mapping bond IDs to spread alert status
        """
        # Placeholder - would need actual bid-ask data
        spread_alerts = {}
        
        for _, bond in opportunities.iterrows():
            # In production, fetch actual bid-ask spread
            # spread = get_bid_ask_spread(bond['identifier'])
            # alert = spread > self.spread_threshold
            
            spread_alerts[bond['identifier']] = False  # Placeholder
        
        return spread_alerts
    
    def export_signals_to_csv(self, opportunities: pd.DataFrame, 
                             filename: Optional[str] = None):
        """Export signals to CSV for external use."""
        if filename is None:
            filename = f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_dir = Path('signals')
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        # Select key columns
        export_cols = [
            'identifier', 'CPN', 'NXT_CPN_DT', 'days_to_next_coupon',
            'coupon_period_length', 'arbitrage_profit_bps', 'PX_CLEAN',
            'PX_DIRTY', 'DUR_ADJ_MID', 'signal_strength'
        ]
        
        opportunities[export_cols].to_csv(filepath, index=False)
        self.logger.info(f"Exported signals to {filepath}")
    
    def create_alert_summary(self, recommendations: List[Dict]) -> str:
        """Create formatted alert summary for notifications."""
        if len(recommendations) == 0:
            return "No arbitrage opportunities detected."
        
        summary = "🔔 CANADIAN BOND ARBITRAGE ALERTS\n"
        summary += "=" * 60 + "\n\n"
        
        for rec in recommendations:
            summary += f"#{rec['rank']} - {rec['bond_identifier']}\n"
            summary += f"  Expected Profit: {rec['expected_profit_bps']:.2f} bps\n"
            summary += f"  Days to Coupon: {rec['days_to_coupon']}\n"
            summary += f"  Risk: {rec['risk_rating']}\n"
            summary += f"  {rec['recommendation']}\n\n"
        
        summary += "=" * 60 + "\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return summary


def main():
    """Main execution for signal generation."""
    generator = SignalGenerator()
    
    # Scan for opportunities
    print("\n🔍 Scanning for arbitrage opportunities...\n")
    opportunities = generator.scan_for_opportunities()
    
    if len(opportunities) == 0:
        print("❌ No opportunities found at this time.")
        return
    
    # Generate recommendations
    recommendations = generator.generate_trade_recommendations(opportunities)
    
    # Create and display alert summary
    summary = generator.create_alert_summary(recommendations)
    print(summary)
    
    # Export signals
    generator.export_signals_to_csv(opportunities)
    
    # Display detailed table
    print("\n📊 DETAILED OPPORTUNITY TABLE")
    print("=" * 120)
    
    display_cols = [
        'identifier', 'coupon_period_length', 'days_to_next_coupon',
        'arbitrage_profit_bps', 'PX_CLEAN', 'risk_rating'
    ]
    
    if len(opportunities) > 0:
        print(opportunities[display_cols].to_string(index=False))
    
    print("\n✅ Signal generation complete.")


if __name__ == "__main__":
    main()
