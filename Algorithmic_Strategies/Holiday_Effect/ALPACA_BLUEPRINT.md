# Alpaca Paper Trading Blueprint
## Holiday Effect Strategy Implementation

---

## Executive Summary

This blueprint outlines the translation of the Holiday Effect trading strategy from a backtest-only implementation to a live paper trading system using Alpaca's API. The strategy trades AMZN around Black Friday and Prime Day events.

**Deployment Model**: Scheduled daemon with daily execution checks  
**Hosting**: Can run on any server/VPS with Python 3.8+  
**Execution**: Market-on-open orders triggered by calendar events

---

## 1. Architecture Overview

### Current (Backtest) vs Target (Paper Trading)

| Component | Current State | Target State |
|-----------|--------------|--------------|
| **Data Source** | yfinance historical | Alpaca Market Data API |
| **Signal Generation** | Batch processing (entire history) | Real-time daily checks |
| **Order Execution** | Simulated in backtest | Real paper trading via Alpaca API |
| **Position Management** | Event-driven backtest logic | Live position tracking + monitoring |
| **Scheduling** | Manual notebook/script runs | Cron job or systemd timer |
| **State Management** | In-memory per run | Persistent state (JSON/SQLite) |
| **Monitoring** | Post-hoc analysis | Real-time logging + alerts |

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Alpaca Paper Trading                     │
│                                                              │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │ Scheduler      │────────▶│ Strategy Runner  │           │
│  │ (Cron/Systemd) │         │ (Main Daemon)    │           │
│  └────────────────┘         └─────────┬────────┘           │
│                                        │                     │
│                     ┌──────────────────┼──────────────┐    │
│                     ▼                  ▼              ▼     │
│          ┌───────────────┐  ┌──────────────┐  ┌─────────┐ │
│          │ Signal Engine │  │ Alpaca API   │  │ State   │ │
│          │ - Event Check │  │ - Orders     │  │ Manager │ │
│          │ - Filters     │  │ - Positions  │  │ (JSON)  │ │
│          └───────────────┘  │ - Market Data│  └─────────┘ │
│                              └──────────────┘               │
│                                                              │
│          ┌──────────────┐         ┌──────────────┐         │
│          │ Risk Monitor │         │ Logger/Alerts│         │
│          │ - Drawdown   │────────▶│ - Files      │         │
│          │ - Position   │         │ - Email/Slack│         │
│          └──────────────┘         └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Modules to Create

### 2.1 `alpaca_trader.py` - Alpaca API Wrapper

**Purpose**: Interface with Alpaca paper trading account

**Key Functions**:
```python
class AlpacaTrader:
    def __init__(self, api_key, secret_key, paper=True)
    def get_account_info() -> Dict  # Cash, buying power, equity
    def get_position(symbol: str) -> Optional[Position]
    def get_all_positions() -> List[Position]
    def submit_order(symbol, qty, side, type='market', time_in_force='day') -> Order
    def cancel_order(order_id: str)
    def get_order_status(order_id: str) -> Order
    def get_latest_quote(symbol: str) -> Quote
    def get_bars(symbol, start, end, timeframe='1Day') -> pd.DataFrame
    def is_market_open() -> bool
    def get_next_market_open() -> datetime
    def get_clock() -> Clock
```

**Dependencies**:
- `alpaca-trade-api` or `alpaca-py` (newer SDK)
- Credentials from environment variables or `.env` file

**Error Handling**:
- Rate limit handling (429 errors)
- Network retry logic
- Invalid order rejection handling

---

### 2.2 `strategy_engine.py` - Live Signal Generation

**Purpose**: Adapt existing `SignalGenerator` for real-time operation

**Key Changes**:
```python
class LiveStrategyEngine:
    def __init__(self, config_path, alpaca_trader)
    
    def should_enter_position(current_date: datetime) -> Tuple[bool, str]:
        """
        Check if today is 10 days before BF or Prime Day.
        Returns: (should_enter, event_type)
        """
        
    def should_exit_position(current_date: datetime, entry_date: datetime) -> bool:
        """
        Check if we've reached exit date (day before event).
        """
        
    def passes_market_filters(current_date: datetime) -> bool:
        """
        Apply filters:
        - SPY > 200-day MA
        - VIX < 25
        - No significant gap down
        """
        
    def calculate_position_size(account_equity: float) -> int:
        """
        Returns shares to buy (100% allocation for this strategy).
        """
```

**New Requirements**:
- Load recent SPY/VIX data for filters
- Use Alpaca data instead of yfinance
- Handle partial fills
- Cache calculations to avoid repeated API calls

---

### 2.3 `state_manager.py` - Persistent State Tracking

**Purpose**: Track open positions, orders, and strategy state between runs

**State Schema** (JSON file):
```json
{
  "last_run": "2026-02-03T09:30:00Z",
  "current_position": {
    "symbol": "AMZN",
    "qty": 500,
    "entry_date": "2025-11-14",
    "entry_price": 185.50,
    "event_type": "black_friday",
    "event_date": "2025-11-28",
    "exit_date": "2025-11-27",
    "order_id": "abc-123-def"
  },
  "pending_orders": [],
  "trade_history": [
    {
      "entry_date": "2025-07-03",
      "exit_date": "2025-07-14",
      "event_type": "prime_day",
      "shares": 450,
      "entry_price": 180.25,
      "exit_price": 187.90,
      "pnl": 3442.50,
      "return_pct": 4.24
    }
  ],
  "performance": {
    "total_trades": 1,
    "winning_trades": 1,
    "total_pnl": 3442.50,
    "sharpe_ratio": null
  }
}
```

**Key Functions**:
```python
class StateManager:
    def __init__(self, state_file_path='state.json')
    def load_state() -> Dict
    def save_state(state: Dict)
    def update_position(position_data: Dict)
    def close_position(exit_price: float, exit_date: datetime)
    def add_trade_to_history(trade: Dict)
    def get_current_position() -> Optional[Dict]
    def is_position_open() -> bool
```

---

### 2.4 `strategy_daemon.py` - Main Execution Loop

**Purpose**: Orchestrate daily strategy execution

**Execution Flow**:
```python
def run_daily_check():
    """
    Called daily at market open (or shortly before).
    """
    # 1. Initialize
    trader = AlpacaTrader()
    engine = LiveStrategyEngine(trader)
    state = StateManager()
    
    # 2. Check if market is open
    if not trader.is_market_open():
        log.info("Market closed, skipping")
        return
    
    # 3. Reconcile state with actual positions
    reconcile_state(trader, state)
    
    # 4. Check for exit signals
    if state.is_position_open():
        if engine.should_exit_position(datetime.now(), state.entry_date):
            execute_exit(trader, state)
            return
    
    # 5. Check for entry signals
    if not state.is_position_open():
        should_enter, event_type = engine.should_enter_position(datetime.now())
        
        if should_enter and engine.passes_market_filters(datetime.now()):
            execute_entry(trader, state, event_type)
    
    # 6. Monitor existing positions
    if state.is_position_open():
        monitor_position(trader, state)
    
    # 7. Log status
    log_daily_status(trader, state)
```

**Error Recovery**:
- Handle partial fills
- Retry failed orders
- Alert on unexpected states
- Emergency shutdown on critical errors

---

## 3. Configuration Changes

### 3.1 New `alpaca_config.yaml`

```yaml
alpaca:
  # API Credentials (use environment variables in production)
  api_key: ${ALPACA_API_KEY}
  secret_key: ${ALPACA_SECRET_KEY}
  base_url: "https://paper-api.alpaca.markets"  # Paper trading
  
  # Data API
  data_api_url: "https://data.alpaca.markets"
  
  # Trading parameters
  paper_trading: true
  fractional_shares: false
  
execution:
  # When to run daily checks
  check_time: "09:25:00"  # 5 min before market open (EST)
  
  # Order settings
  order_type: "market"
  time_in_force: "day"
  
  # Retry logic
  max_retries: 3
  retry_delay_seconds: 60
  
  # Position sizing
  allocation_pct: 100  # Full capital
  min_shares: 1
  max_position_value: 500000  # Safety limit

risk_management:
  # Override backtest params with live guardrails
  max_drawdown_threshold: 20  # Pause trading if hit
  stop_loss_pct: 8  # Per trade stop loss
  
  # Market filters (from original strategy)
  require_spy_above_200ma: true
  max_vix: 25
  
monitoring:
  log_level: "INFO"
  log_file: "logs/strategy.log"
  
  # Alerts
  email_alerts: true
  email_recipient: "your-email@example.com"
  
  slack_webhook: ${SLACK_WEBHOOK_URL}  # Optional
```

---

## 4. Scheduling & Deployment

### 4.1 Cron Job (Linux/Mac)

Create cron entry to run daily at 9:25 AM EST:

```bash
# Edit crontab
crontab -e

# Add this line (adjust timezone as needed)
25 9 * * 1-5 cd /path/to/Holiday_Effect && /path/to/venv/bin/python strategy_daemon.py >> logs/cron.log 2>&1
```

### 4.2 Systemd Timer (Production Linux)

**Service file** (`holiday-strategy.service`):
```ini
[Unit]
Description=Holiday Effect Trading Strategy
After=network.target

[Service]
Type=oneshot
User=trader
WorkingDirectory=/opt/holiday_strategy
ExecStart=/opt/holiday_strategy/venv/bin/python strategy_daemon.py
Environment="ALPACA_API_KEY=your_key"
Environment="ALPACA_SECRET_KEY=your_secret"

[Install]
WantedBy=multi-user.target
```

**Timer file** (`holiday-strategy.timer`):
```ini
[Unit]
Description=Run Holiday Strategy Daily

[Timer]
OnCalendar=Mon-Fri 14:25:00  # UTC time (9:25 AM EST)
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable holiday-strategy.timer
sudo systemctl start holiday-strategy.timer
```

### 4.3 Python Scheduler (Alternative)

Use `schedule` library within long-running process:

```python
import schedule
import time

def job():
    run_daily_check()

schedule.every().monday.at("09:25").do(job)
schedule.every().tuesday.at("09:25").do(job)
# ... repeat for weekdays

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 5. Data Source Migration

### 5.1 Replace yfinance with Alpaca Data API

**Old (Backtest)**:
```python
import yfinance as yf
amzn = yf.download("AMZN", start="2020-01-01", end="2025-01-01")
```

**New (Live)**:
```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

data_client = StockHistoricalDataClient(api_key, secret_key)
request = StockBarsRequest(
    symbol_or_symbols="AMZN",
    timeframe=TimeFrame.Day,
    start=datetime(2020, 1, 1),
    end=datetime(2025, 1, 1)
)
bars = data_client.get_stock_bars(request)
```

**Market Data Needed Daily**:
- AMZN latest price (for position monitoring)
- SPY 200-day MA (for filter)
- VIX latest value (for filter)

**Optimization**: Cache SPY/VIX historical data, only fetch latest bar each day

---

## 6. Order Execution Logic

### 6.1 Entry Order Flow

```python
def execute_entry(trader, state, event_type):
    """Submit market-on-open order for next trading day."""
    
    # 1. Get account equity
    account = trader.get_account_info()
    buying_power = float(account.buying_power)
    
    # 2. Get current AMZN price
    quote = trader.get_latest_quote("AMZN")
    current_price = float(quote.ask_price)
    
    # 3. Calculate shares (100% allocation)
    shares = int(buying_power / current_price)
    
    if shares < 1:
        log.error("Insufficient buying power")
        return
    
    # 4. Submit order
    order = trader.submit_order(
        symbol="AMZN",
        qty=shares,
        side="buy",
        type="market",
        time_in_force="day"
    )
    
    # 5. Update state
    state.update_position({
        'symbol': 'AMZN',
        'qty': shares,
        'entry_date': datetime.now().date(),
        'event_type': event_type,
        'order_id': order.id,
        'status': 'pending'
    })
    
    log.info(f"Entry order submitted: {shares} shares AMZN @ market")
    
    # 6. Wait for fill (check after 5 min)
    time.sleep(300)
    filled_order = trader.get_order_status(order.id)
    
    if filled_order.status == 'filled':
        state.update_position({
            'entry_price': float(filled_order.filled_avg_price),
            'status': 'open'
        })
        log.info(f"Order filled at ${filled_order.filled_avg_price}")
    else:
        log.warning(f"Order not filled: {filled_order.status}")
```

### 6.2 Exit Order Flow

```python
def execute_exit(trader, state):
    """Close position at market."""
    
    position = state.get_current_position()
    
    if not position:
        log.error("No position to exit")
        return
    
    # 1. Submit sell order
    order = trader.submit_order(
        symbol="AMZN",
        qty=position['qty'],
        side="sell",
        type="market",
        time_in_force="day"
    )
    
    log.info(f"Exit order submitted: {position['qty']} shares AMZN")
    
    # 2. Wait for fill
    time.sleep(300)
    filled_order = trader.get_order_status(order.id)
    
    if filled_order.status == 'filled':
        exit_price = float(filled_order.filled_avg_price)
        pnl = (exit_price - position['entry_price']) * position['qty']
        return_pct = (exit_price / position['entry_price'] - 1) * 100
        
        # 3. Record trade
        state.close_position(exit_price, datetime.now().date())
        
        log.info(f"Trade closed: PnL=${pnl:.2f} ({return_pct:.2f}%)")
        
        # 4. Send alert
        send_alert(f"Trade closed: {return_pct:.2f}% return")
    else:
        log.error(f"Exit order failed: {filled_order.status}")
```

---

## 7. Risk Management & Monitoring

### 7.1 Position Monitoring (Intraday)

```python
def monitor_position(trader, state):
    """Check position during holding period."""
    
    position = state.get_current_position()
    current_position = trader.get_position("AMZN")
    
    if not current_position:
        log.error("Position mismatch - no Alpaca position found!")
        return
    
    # 1. Calculate unrealized P&L
    unrealized_pnl = float(current_position.unrealized_pl)
    unrealized_pct = float(current_position.unrealized_plpc) * 100
    
    # 2. Check stop loss
    if unrealized_pct < -8.0:  # 8% stop loss
        log.warning(f"Stop loss triggered: {unrealized_pct:.2f}%")
        execute_exit(trader, state)
        send_alert(f"STOP LOSS: Position closed at {unrealized_pct:.2f}%")
        return
    
    # 3. Log status
    log.info(f"Position: {position['qty']} AMZN | Unrealized: ${unrealized_pnl:.2f} ({unrealized_pct:.2f}%)")
```

### 7.2 State Reconciliation

```python
def reconcile_state(trader, state):
    """Ensure internal state matches Alpaca account."""
    
    saved_position = state.get_current_position()
    actual_position = trader.get_position("AMZN")
    
    # Case 1: We think we have a position, but Alpaca shows none
    if saved_position and not actual_position:
        log.error("State mismatch: position closed externally?")
        # Force close position in state
        state.close_position(0, datetime.now().date())
        send_alert("Position mismatch detected - state corrected")
    
    # Case 2: We think we're flat, but Alpaca shows a position
    elif not saved_position and actual_position:
        log.error("Unknown position in Alpaca account!")
        send_alert("Manual position detected - liquidating")
        trader.submit_order("AMZN", actual_position.qty, "sell")
    
    # Case 3: Quantity mismatch
    elif saved_position and actual_position:
        if saved_position['qty'] != int(actual_position.qty):
            log.warning(f"Qty mismatch: expected {saved_position['qty']}, actual {actual_position.qty}")
            # Update state to match reality
            saved_position['qty'] = int(actual_position.qty)
            state.save_state()
```

### 7.3 Alerting System

```python
def send_alert(message: str, level="INFO"):
    """Send alerts via email and Slack."""
    
    # Email
    if config['monitoring']['email_alerts']:
        send_email(
            to=config['monitoring']['email_recipient'],
            subject=f"Holiday Strategy Alert - {level}",
            body=message
        )
    
    # Slack
    if config['monitoring'].get('slack_webhook'):
        requests.post(
            config['monitoring']['slack_webhook'],
            json={"text": f"🤖 *Holiday Strategy*\n{message}"}
        )
    
    # Always log
    log.info(f"ALERT: {message}")
```

---

## 8. Testing Strategy

### 8.1 Pre-Production Checklist

**Phase 1: Unit Tests**
- [ ] Test Alpaca API connection
- [ ] Test signal generation for known dates
- [ ] Test position size calculation
- [ ] Test state save/load
- [ ] Test order submission (paper account)

**Phase 2: Integration Tests**
- [ ] Run full daily check with no positions
- [ ] Simulate entry signal
- [ ] Simulate exit signal
- [ ] Test state reconciliation with manual position
- [ ] Test stop loss trigger

**Phase 3: Paper Trading Validation**
- [ ] Monitor for 1 full event cycle without intervention
- [ ] Verify orders execute correctly
- [ ] Confirm P&L tracking matches Alpaca
- [ ] Test alert system

### 8.2 Dry Run Mode

Add `--dry-run` flag to test without submitting orders:

```python
if args.dry_run:
    log.info("DRY RUN: Would submit order for 500 shares AMZN")
else:
    trader.submit_order("AMZN", 500, "buy")
```

---

## 9. Monitoring & Observability

### 9.1 Logging Structure

```
logs/
├── strategy.log           # Main strategy decisions
├── trades.log             # All trade executions
├── errors.log             # Errors only
└── daily_status.log       # Daily health checks
```

**Log Format**:
```
2026-02-03 09:25:00 | INFO | No position open, checking entry signals...
2026-02-03 09:25:05 | INFO | 5 days until Prime Day (2026-02-08)
2026-02-03 09:25:10 | INFO | Market filters passed: SPY > 200MA, VIX = 18.5
2026-02-03 09:25:15 | INFO | Entry signal triggered for prime_day
2026-02-03 09:25:20 | INFO | Submitting buy order: 450 shares AMZN
2026-02-03 09:30:22 | INFO | Order filled at $192.50
```

### 9.2 Daily Status Report

Send daily summary email at market close:

```python
def generate_daily_report():
    """
    Daily summary:
    - Current position (if any)
    - Unrealized P&L
    - Days until exit
    - Upcoming events
    - Account equity
    """
    
    report = f"""
    === Holiday Effect Strategy - Daily Report ===
    Date: {datetime.now().date()}
    
    Position Status:
    - Symbol: AMZN
    - Shares: 450
    - Entry: $192.50 (2026-02-03)
    - Current: $195.20
    - Unrealized P&L: +$1,215 (+1.40%)
    - Event: Prime Day (2026-02-08)
    - Days to Exit: 4
    
    Account:
    - Equity: $1,001,215
    - Cash: $0
    - Buying Power: $0
    
    Upcoming Events:
    - Black Friday: 2026-11-27 (298 days)
    
    YTD Performance:
    - Total Return: +0.12%
    - Trades: 2
    - Win Rate: 100%
    """
    
    return report
```

### 9.3 Performance Dashboard (Optional)

Use Streamlit or Dash to create live dashboard:

- Real-time P&L chart
- Position history
- Upcoming event calendar
- Trade log
- Alert history

---

## 10. Deployment Environments

### 10.1 Development Environment

- **Purpose**: Local testing
- **Alpaca**: Paper trading account
- **Scheduler**: Manual runs
- **State**: Local JSON file
- **Alerts**: Console logs only

### 10.2 Staging Environment

- **Purpose**: Final validation before live
- **Alpaca**: Dedicated paper account
- **Scheduler**: Cron job on local machine
- **State**: Local JSON + backup
- **Alerts**: Email to test address

### 10.3 Production Environment

- **Purpose**: Live paper trading
- **Alpaca**: Production paper account
- **Scheduler**: Systemd timer on VPS/cloud
- **State**: JSON + cloud backup (S3/GCS)
- **Alerts**: Email + Slack
- **Monitoring**: Uptime checks, log aggregation

### 10.4 Recommended Infrastructure

**Option 1: AWS EC2 t3.micro**
- Cost: ~$7/month
- 1 vCPU, 1GB RAM (sufficient)
- Systemd for scheduling
- CloudWatch for logs

**Option 2: DigitalOcean Droplet**
- Cost: $6/month (basic)
- Simple setup
- Built-in monitoring

**Option 3: Local Machine**
- Cost: $0
- Requires always-on computer
- Less reliable

---

## 11. Migration Checklist

### Pre-Migration
- [ ] Open Alpaca paper trading account
- [ ] Generate API keys
- [ ] Install dependencies (`alpaca-py`, `alpaca-trade-api`)
- [ ] Test API connection
- [ ] Review event calendar (Black Friday, Prime Day dates)

### Code Development
- [ ] Implement `alpaca_trader.py`
- [ ] Adapt `signal_generator.py` → `strategy_engine.py`
- [ ] Create `state_manager.py`
- [ ] Build `strategy_daemon.py`
- [ ] Add `alpaca_config.yaml`
- [ ] Write unit tests
- [ ] Add error handling & logging

### Testing
- [ ] Run dry-run mode for 2 weeks
- [ ] Manually trigger entry/exit signals
- [ ] Test state reconciliation
- [ ] Verify stop loss logic
- [ ] Test alert system

### Deployment
- [ ] Set up server/VPS
- [ ] Configure scheduler (cron/systemd)
- [ ] Set environment variables
- [ ] Test scheduled execution
- [ ] Monitor first live cycle

### Post-Deployment
- [ ] Review logs daily for 1 month
- [ ] Compare paper results to backtest
- [ ] Adjust parameters if needed
- [ ] Document any edge cases discovered

---

## 12. Key Differences from Backtest

| Aspect | Backtest | Paper Trading |
|--------|----------|---------------|
| **Data** | Historical, complete | Real-time, may have gaps |
| **Orders** | Instant fill at known price | Asynchronous, slippage |
| **Position Tracking** | Perfect information | Must reconcile with broker |
| **Timing** | Post-hoc analysis | Real-time decisions |
| **Errors** | Retry indefinitely | Must handle failures |
| **Market Conditions** | Can replay | Live market surprises |
| **Costs** | Simulated | Real commissions (paper) |

**Critical Considerations**:
1. **Market-on-open orders**: Submit night before, fill next morning
2. **Partial fills**: Handle case where not all shares fill
3. **Network issues**: Retry logic, timeout handling
4. **State persistence**: Never lose track of open positions
5. **Holiday calendar**: NYSE closures affect entry/exit dates

---

## 13. Risk Mitigation

### Operational Risks
1. **Server downtime** → Missed entry/exit signals
   - *Mitigation*: Use reliable hosting, set up monitoring
   
2. **API failures** → Orders not submitted
   - *Mitigation*: Retry logic, manual override capability
   
3. **State corruption** → Lost track of positions
   - *Mitigation*: Daily state backups, reconciliation checks
   
4. **Wrong event date** → Trade at wrong time
   - *Mitigation*: Manual review of event calendar each year

### Market Risks
1. **Flash crash during holding** → Large loss
   - *Mitigation*: 8% stop loss, monitor intraday
   
2. **Gap down at exit** → Worse fill than expected
   - *Mitigation*: Accept as part of strategy, track slippage
   
3. **Low liquidity** → Order doesn't fill
   - *Mitigation*: AMZN highly liquid, use market orders

---

## 14. Future Enhancements

1. **Multi-event support**: Add other holidays (Cyber Monday, Prime Day 2)
2. **Options overlay**: Implement put-selling strategy from research
3. **Portfolio allocation**: Combine with SPY base position
4. **Dynamic position sizing**: Risk parity based on volatility
5. **Machine learning**: Predict event returns, adjust sizing
6. **Multi-asset**: Extend to other retail stocks (WMT, TGT)
7. **Webhooks**: Real-time alerts via Alpaca webhooks
8. **Database**: Replace JSON state with PostgreSQL
9. **Web dashboard**: Live monitoring interface
10. **Backtesting integration**: Compare live results to backtest

---

## 15. Cost Analysis

### Development Time
- Initial implementation: 20-30 hours
- Testing & debugging: 10-15 hours
- Deployment & monitoring: 5-10 hours
- **Total**: 35-55 hours

### Operating Costs (Monthly)
- VPS hosting: $6-10
- Domain (optional): $1
- Monitoring (optional): $0-20
- **Total**: $7-31/month

### Alpaca Costs
- Paper trading: **FREE**
- Real trading: Commission-free for stocks
- Data: Basic real-time data included

---

## 16. Success Metrics

### Technical Metrics
- **Uptime**: > 99% during market hours
- **Order success rate**: > 95%
- **State reconciliation**: 100% accuracy
- **Alert response time**: < 5 minutes

### Performance Metrics (vs Backtest)
- **Sharpe ratio**: Target 0.50+ (backtest: 0.54)
- **Win rate**: Target 70%+ (backtest: 75%)
- **Max drawdown**: < 20% (backtest: -14.26%)
- **Slippage**: < 10 bps average

---

## 17. Resources & References

### Documentation
- [Alpaca API Docs](https://docs.alpaca.markets/)
- [alpaca-py SDK](https://github.com/alpacahq/alpaca-py)
- [Alpaca Market Data API](https://docs.alpaca.markets/docs/market-data)

### Libraries
```txt
alpaca-py>=0.30.1
python-dotenv>=1.0.0
pyyaml>=6.0
pandas>=2.0.0
schedule>=1.2.0
requests>=2.31.0
```

### Example Code Repositories
- Alpaca examples: https://github.com/alpacahq/alpaca-py
- Trading bots: Search GitHub for "alpaca trading bot"

---

## Appendix A: Sample File Structure

```
Holiday_Effect/
├── alpaca/
│   ├── __init__.py
│   ├── alpaca_trader.py          # Alpaca API wrapper
│   ├── strategy_engine.py         # Live signal generation
│   ├── state_manager.py           # Persistent state
│   └── utils.py                   # Helper functions
├── strategy_daemon.py             # Main entry point
├── alpaca_config.yaml             # Alpaca configuration
├── state.json                     # Current state (generated)
├── logs/                          # Log files
│   ├── strategy.log
│   ├── trades.log
│   └── errors.log
├── tests/
│   ├── test_alpaca_trader.py
│   ├── test_strategy_engine.py
│   └── test_state_manager.py
├── .env                           # API keys (gitignored)
├── requirements_alpaca.txt        # Additional dependencies
└── README_ALPACA.md              # Deployment instructions
```

---

## Appendix B: Sample .env File

```bash
# Alpaca API Credentials
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Alpaca Endpoints
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets

# Alert Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_RECIPIENT=alerts@example.com

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Environment
ENVIRONMENT=production  # or 'staging', 'development'
LOG_LEVEL=INFO
```

---

## Appendix C: Quick Start Commands

```bash
# 1. Clone and setup
cd Holiday_Effect
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install -r requirements_alpaca.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Alpaca API keys

# 3. Test connection
python -c "from alpaca.alpaca_trader import AlpacaTrader; t = AlpacaTrader(); print(t.get_account_info())"

# 4. Dry run
python strategy_daemon.py --dry-run

# 5. Single execution (manual test)
python strategy_daemon.py

# 6. Set up cron (Linux/Mac)
crontab -e
# Add: 25 9 * * 1-5 cd /path/to/Holiday_Effect && ./venv/bin/python strategy_daemon.py

# 7. Monitor logs
tail -f logs/strategy.log
```

---

**End of Blueprint**

*This blueprint provides a complete roadmap for translating the Holiday Effect strategy from backtest to live paper trading on Alpaca. Follow the checklist systematically, test thoroughly, and monitor closely during initial deployment.*
