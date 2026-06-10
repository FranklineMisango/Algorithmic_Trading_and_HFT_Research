#!/usr/bin/env python3
"""
Arithmax Research Data Chest - Interactive Mode
A user-friendly terminal app for downloading financial data
with smart ticker classification and guided workflows.
"""

import sys
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Rich UI imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn
)
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.text import Text
from rich.rule import Rule

try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.completion import FuzzyWordCompleter
    from prompt_toolkit.history import InMemoryHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

from config import (
    DEFAULT_EQUITY_SYMBOLS, DEFAULT_CRYPTO_SYMBOLS, DEFAULT_OPTION_SYMBOLS,
    DEFAULT_FUTURES_SYMBOLS, DEFAULT_DATABENTO_FUTURES_SYMBOLS,
    DEFAULT_START_DATE, DEFAULT_END_DATE, SUPPORTED_RESOLUTIONS,
    DEFAULT_YAHOO_ETFS, DEFAULT_YAHOO_INDICES, DEFAULT_YAHOO_FOREX, DEFAULT_YAHOO_CRYPTO,
    DEFAULT_NSE_STOCKS, DEFAULT_NSE_INDICES, DEFAULT_BSE_STOCKS,
    DEFAULT_FRED_SERIES, DEFAULT_INVESTING_STOCKS, DEFAULT_INVESTING_FOREX,
    DEFAULT_INVESTING_COMMODITIES, DEFAULT_INVESTING_CRYPTO, DEFAULT_INVESTING_INDICES,
    DEFAULT_STOOQ_STOCKS, DEFAULT_STOOQ_FOREX, DEFAULT_STOOQ_INDICES, DEFAULT_STOOQ_COMMODITIES,
    DATA_ROOT,
    DEFAULT_OUTPUT_FORMAT,
)
from alpaca_downloader import AlpacaDataDownloader
from binance_downloader import BinanceDataDownloader
from polygon_futures_downloader import PolygonFuturesDownloader
from databento_downloader import DatabentoFuturesDownloader
from alpha_vantage_downloader import AlphaVantageDownloader
from yahoo_finance_downloader import YahooFinanceDownloader
from nse_india_downloader import NSEIndiaDownloader
from bse_india_downloader import BSEIndiaDownloader
from tiingo_downloader import TiingoDownloader
from fred_downloader import FREDDownloader
from quandl_downloader import QuandlDownloader
from coindesk_downloader import CoinDeskDownloader
from investing_com_downloader import InvestingComDownloader
from stooq_downloader import StooqDownloader
from utils import setup_logging

logger = setup_logging()
console = Console()


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class TickerInfo:
    """Information about a ticker symbol"""
    symbol: str
    asset_type: str  # 'stock', 'crypto', 'forex', 'etf', 'index', 'futures', 'option', 'economic', 'commodity'
    source: str      # Best data source
    display_name: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.symbol


# ---------------------------------------------------------------------------
# Smart Ticker Classifier
# ---------------------------------------------------------------------------

POPULAR_TICKERS = {
    # US Stocks
    "AAPL": TickerInfo("AAPL", "stock", "yahoo", "Apple Inc.", "Technology - Consumer Electronics"),
    "MSFT": TickerInfo("MSFT", "stock", "yahoo", "Microsoft Corporation", "Technology - Software"),
    "GOOGL": TickerInfo("GOOGL", "stock", "yahoo", "Alphabet Inc. (Google)", "Technology - Internet"),
    "GOOG": TickerInfo("GOOG", "stock", "yahoo", "Alphabet Inc. (Google)", "Technology - Internet"),
    "AMZN": TickerInfo("AMZN", "stock", "yahoo", "Amazon.com Inc.", "Technology - E-Commerce"),
    "TSLA": TickerInfo("TSLA", "stock", "yahoo", "Tesla Inc.", "Automotive - Electric Vehicles"),
    "NVDA": TickerInfo("NVDA", "stock", "yahoo", "NVIDIA Corporation", "Technology - Semiconductors"),
    "META": TickerInfo("META", "stock", "yahoo", "Meta Platforms Inc.", "Technology - Social Media"),
    "NFLX": TickerInfo("NFLX", "stock", "yahoo", "Netflix Inc.", "Entertainment - Streaming"),
    "JPM": TickerInfo("JPM", "stock", "yahoo", "JPMorgan Chase & Co.", "Finance - Banking"),
    "V": TickerInfo("V", "stock", "yahoo", "Visa Inc.", "Finance - Payment Processing"),
    "WMT": TickerInfo("WMT", "stock", "yahoo", "Walmart Inc.", "Retail - Discount Stores"),
    "JNJ": TickerInfo("JNJ", "stock", "yahoo", "Johnson & Johnson", "Healthcare - Pharmaceuticals"),
    "PG": TickerInfo("PG", "stock", "yahoo", "Procter & Gamble Co.", "Consumer Goods - Household"),
    "MA": TickerInfo("MA", "stock", "yahoo", "Mastercard Inc.", "Finance - Payment Processing"),
    "UNH": TickerInfo("UNH", "stock", "yahoo", "UnitedHealth Group Inc.", "Healthcare - Insurance"),
    "HD": TickerInfo("HD", "stock", "yahoo", "The Home Depot Inc.", "Retail - Home Improvement"),
    "DIS": TickerInfo("DIS", "stock", "yahoo", "The Walt Disney Company", "Entertainment - Media"),
    "BAC": TickerInfo("BAC", "stock", "yahoo", "Bank of America Corp.", "Finance - Banking"),
    "INTC": TickerInfo("INTC", "stock", "yahoo", "Intel Corporation", "Technology - Semiconductors"),
    "AMD": TickerInfo("AMD", "stock", "yahoo", "Advanced Micro Devices Inc.", "Technology - Semiconductors"),
    "PLTR": TickerInfo("PLTR", "stock", "yahoo", "Palantir Technologies Inc.", "Technology - Data Analytics"),
    "COIN": TickerInfo("COIN", "stock", "yahoo", "Coinbase Global Inc.", "Finance - Cryptocurrency"),
    "GME": TickerInfo("GME", "stock", "yahoo", "GameStop Corp.", "Retail - Video Games"),
    "AMC": TickerInfo("AMC", "stock", "yahoo", "AMC Entertainment Holdings", "Entertainment - Theaters"),

    # ETFs
    "SPY": TickerInfo("SPY", "etf", "yahoo", "SPDR S&P 500 ETF Trust", "Tracks S&P 500 Index"),
    "QQQ": TickerInfo("QQQ", "etf", "yahoo", "Invesco QQQ Trust", "Tracks Nasdaq-100 Index"),
    "IWM": TickerInfo("IWM", "etf", "yahoo", "iShares Russell 2000 ETF", "Tracks Russell 2000 Small-Cap"),
    "VTI": TickerInfo("VTI", "etf", "yahoo", "Vanguard Total Stock Market ETF", "Tracks CRSP US Total Market"),
    "VOO": TickerInfo("VOO", "etf", "yahoo", "Vanguard S&P 500 ETF", "Tracks S&P 500 Index"),
    "VEA": TickerInfo("VEA", "etf", "yahoo", "Vanguard FTSE Developed Markets ETF", "International Developed Markets"),
    "VWO": TickerInfo("VWO", "etf", "yahoo", "Vanguard FTSE Emerging Markets ETF", "Emerging Markets"),
    "BND": TickerInfo("BND", "etf", "yahoo", "Vanguard Total Bond Market ETF", "US Bond Market"),
    "AGG": TickerInfo("AGG", "etf", "yahoo", "iShares Core US Aggregate Bond ETF", "US Investment-Grade Bonds"),
    "GLD": TickerInfo("GLD", "etf", "yahoo", "SPDR Gold Trust", "Gold Bullion"),
    "SLV": TickerInfo("SLV", "etf", "yahoo", "iShares Silver Trust", "Silver Bullion"),
    "XLF": TickerInfo("XLF", "etf", "yahoo", "Financial Select Sector SPDR Fund", "Financial Sector"),
    "XLK": TickerInfo("XLK", "etf", "yahoo", "Technology Select Sector SPDR Fund", "Technology Sector"),
    "XLE": TickerInfo("XLE", "etf", "yahoo", "Energy Select Sector SPDR Fund", "Energy Sector"),
    "XLV": TickerInfo("XLV", "etf", "yahoo", "Health Care Select Sector SPDR Fund", "Healthcare Sector"),
    "ARKK": TickerInfo("ARKK", "etf", "yahoo", "ARK Innovation ETF", "Disruptive Innovation"),
    "TQQQ": TickerInfo("TQQQ", "etf", "yahoo", "ProShares UltraPro QQQ", "3x Leveraged Nasdaq-100"),
    "UPRO": TickerInfo("UPRO", "etf", "yahoo", "ProShares UltraPro S&P500", "3x Leveraged S&P 500"),
    "UDOW": TickerInfo("UDOW", "etf", "yahoo", "ProShares UltraPro Dow30", "3x Leveraged Dow Jones"),
    "TMF": TickerInfo("TMF", "etf", "yahoo", "Direxion Daily 20+ Year Treasury Bull 3x", "3x Leveraged Treasuries"),
    "UGL": TickerInfo("UGL", "etf", "yahoo", "ProShares Ultra Gold", "2x Leveraged Gold"),
    "DIG": TickerInfo("DIG", "etf", "yahoo", "ProShares Ultra Oil & Gas", "2x Leveraged Oil & Gas"),
    "FAS": TickerInfo("FAS", "etf", "yahoo", "Direxion Daily Financial Bull 3x", "3x Leveraged Financials"),
    "FNGU": TickerInfo("FNGU", "etf", "yahoo", "MicroSectors FANG+ Index 3x Leveraged", "3x Leveraged FANG+"),
    "SOXL": TickerInfo("SOXL", "etf", "yahoo", "Direxion Daily Semiconductor Bull 3x", "3x Leveraged Semiconductors"),
    "LABU": TickerInfo("LABU", "etf", "yahoo", "Direxion Daily S&P Biotech Bull 3x", "3x Leveraged Biotech"),
    "JNUG": TickerInfo("JNUG", "etf", "yahoo", "Direxion Daily Junior Gold Miners Bull 2x", "2x Leveraged Gold Miners"),
    "NUGT": TickerInfo("NUGT", "etf", "yahoo", "Direxion Daily Gold Miners Bull 2x", "2x Leveraged Gold Miners"),
    "RETL": TickerInfo("RETL", "etf", "yahoo", "Direxion Daily Retail Bull 3x", "3x Leveraged Retail"),
    "MIDU": TickerInfo("MIDU", "etf", "yahoo", "Direxion Daily Mid Cap Bull 3x", "3x Leveraged Mid Caps"),
    "DPST": TickerInfo("DPST", "etf", "yahoo", "Direxion Daily Regional Banks Bull 3x", "3x Leveraged Regional Banks"),
    "PILL": TickerInfo("PILL", "etf", "yahoo", "Direxion Daily Pharmaceutical Bull 3x", "3x Leveraged Pharma"),
    "CURE": TickerInfo("CURE", "etf", "yahoo", "Direxion Daily Healthcare Bull 3x", "3x Leveraged Healthcare"),
    "WANT": TickerInfo("WANT", "etf", "yahoo", "Direxion Daily Consumer Discretionary Bull 3x", "3x Leveraged Consumer Disc."),
    "INDL": TickerInfo("INDL", "etf", "yahoo", "Direxion Daily India Bull 3x", "3x Leveraged India"),
    "CHAU": TickerInfo("CHAU", "etf", "yahoo", "Direxion Daily China Bull 3x", "3x Leveraged China"),
    "EURL": TickerInfo("EURL", "etf", "yahoo", "Direxion Daily FTSE Europe Bull 3x", "3x Leveraged Europe"),

    # Indices
    "^GSPC": TickerInfo("^GSPC", "index", "yahoo", "S&P 500 Index", "US Large-Cap Stock Index"),
    "^DJI": TickerInfo("^DJI", "index", "yahoo", "Dow Jones Industrial Average", "US Blue-Chip Stock Index"),
    "^IXIC": TickerInfo("^IXIC", "index", "yahoo", "Nasdaq Composite Index", "US Technology-Heavy Index"),
    "^RUT": TickerInfo("^RUT", "index", "yahoo", "Russell 2000 Index", "US Small-Cap Stock Index"),
    "^VIX": TickerInfo("^VIX", "index", "yahoo", "CBOE Volatility Index", "Market Volatility / Fear Gauge"),
    "^TNX": TickerInfo("^TNX", "index", "yahoo", "10-Year Treasury Note Yield", "US Bond Yield Benchmark"),

    # Cryptocurrencies (Yahoo format)
    "BTC-USD": TickerInfo("BTC-USD", "crypto", "yahoo", "Bitcoin", "The original cryptocurrency"),
    "ETH-USD": TickerInfo("ETH-USD", "crypto", "yahoo", "Ethereum", "Smart contract platform"),
    "ADA-USD": TickerInfo("ADA-USD", "crypto", "yahoo", "Cardano", "Proof-of-stake blockchain"),
    "SOL-USD": TickerInfo("SOL-USD", "crypto", "yahoo", "Solana", "High-performance blockchain"),
    "DOT-USD": TickerInfo("DOT-USD", "crypto", "yahoo", "Polkadot", "Multi-chain blockchain"),
    "DOGE-USD": TickerInfo("DOGE-USD", "crypto", "yahoo", "Dogecoin", "Meme cryptocurrency"),
    "LINK-USD": TickerInfo("LINK-USD", "crypto", "yahoo", "Chainlink", "Oracle network"),
    "AVAX-USD": TickerInfo("AVAX-USD", "crypto", "yahoo", "Avalanche", "Smart contract platform"),
    "XRP-USD": TickerInfo("XRP-USD", "crypto", "yahoo", "XRP", "Payment settlement token"),

    # Binance Crypto (for Binance downloader)
    "BTCUSDT": TickerInfo("BTCUSDT", "crypto", "binance", "Bitcoin/USDT"),
    "ETHUSDT": TickerInfo("ETHUSDT", "crypto", "binance", "Ethereum/USDT"),
    "SOLUSDT": TickerInfo("SOLUSDT", "crypto", "binance", "Solana/USDT"),
    "ADAUSDT": TickerInfo("ADAUSDT", "crypto", "binance", "Cardano/USDT"),
    "BNBUSDT": TickerInfo("BNBUSDT", "crypto", "binance", "Binance Coin/USDT"),

    # Forex
    "EURUSD=X": TickerInfo("EURUSD=X", "forex", "yahoo", "Euro / US Dollar", "Major Forex Pair"),
    "GBPUSD=X": TickerInfo("GBPUSD=X", "forex", "yahoo", "British Pound / US Dollar", "Major Forex Pair"),
    "USDJPY=X": TickerInfo("USDJPY=X", "forex", "yahoo", "US Dollar / Japanese Yen", "Major Forex Pair"),
    "USDCHF=X": TickerInfo("USDCHF=X", "forex", "yahoo", "US Dollar / Swiss Franc", "Major Forex Pair"),
    "AUDUSD=X": TickerInfo("AUDUSD=X", "forex", "yahoo", "Australian Dollar / US Dollar", "Major Forex Pair"),

    # NSE India Stocks
    "RELIANCE": TickerInfo("RELIANCE", "stock", "nse-india", "Reliance Industries Ltd.", "Energy & Telecom - India"),
    "TCS": TickerInfo("TCS", "stock", "nse-india", "Tata Consultancy Services", "IT Services - India"),
    "HDFCBANK": TickerInfo("HDFCBANK", "stock", "nse-india", "HDFC Bank Ltd.", "Banking - India"),
    "INFY": TickerInfo("INFY", "stock", "nse-india", "Infosys Ltd.", "IT Services - India"),
    "SBIN": TickerInfo("SBIN", "stock", "nse-india", "State Bank of India", "Banking - India"),
    "ICICIBANK": TickerInfo("ICICIBANK", "stock", "nse-india", "ICICI Bank Ltd.", "Banking - India"),

    # Economic Indicators (FRED)
    "GDP": TickerInfo("GDP", "economic", "fred", "Gross Domestic Product", "US Economic Indicator"),
    "UNRATE": TickerInfo("UNRATE", "economic", "fred", "Unemployment Rate", "US Economic Indicator"),
    "CPIAUCSL": TickerInfo("CPIAUCSL", "economic", "fred", "Consumer Price Index", "US Inflation Measure"),
    "FEDFUNDS": TickerInfo("FEDFUNDS", "economic", "fred", "Federal Funds Rate", "US Interest Rate"),
    "DGS10": TickerInfo("DGS10", "economic", "fred", "10-Year Treasury Yield", "US Bond Yield"),

    # Commodities
    "GC=F": TickerInfo("GC=F", "commodity", "yahoo", "Gold Futures", "Precious Metal"),
    "CL=F": TickerInfo("CL=F", "commodity", "yahoo", "Crude Oil Futures", "Energy"),
    "SI=F": TickerInfo("SI=F", "commodity", "yahoo", "Silver Futures", "Precious Metal"),
    "NG=F": TickerInfo("NG=F", "commodity", "yahoo", "Natural Gas Futures", "Energy"),
}


def classify_ticker(symbol: str) -> TickerInfo:
    """
    Automatically classify a ticker symbol using pattern matching.
    No API calls needed.
    """
    clean = symbol.upper().strip()

    # 1. Check known tickers
    if clean in POPULAR_TICKERS:
        return POPULAR_TICKERS[clean]

    # 2. Check alternate forms
    for known_sym, info in POPULAR_TICKERS.items():
        alt = known_sym.replace("-", "").replace("=X", "").replace("=F", "")
        if clean == alt:
            return info

    # 3. Pattern-based classification
    if clean.endswith("USDT") or clean.endswith("USDC"):
        return TickerInfo(clean, "crypto", "binance", f"{clean[:-4]}/USDT", "Cryptocurrency")
    if clean.endswith("-USD") or clean.endswith("-USDT"):
        return TickerInfo(clean, "crypto", "yahoo", clean, "Cryptocurrency")
    if clean.endswith("=X"):
        return TickerInfo(clean, "forex", "yahoo", clean, "Forex Pair")
    if re.match(r'^[A-Z]{1,4}=F$', clean):
        return TickerInfo(clean, "commodity", "yahoo", clean, "Futures/Commodity")
    if clean.startswith("^"):
        return TickerInfo(clean, "index", "yahoo", clean, "Market Index")
    if re.match(r'^[A-Z]{1,4}\.FUT$', clean):
        return TickerInfo(clean, "futures", "databento", clean, "Futures Contract")
    if clean in ["ES", "NQ", "YM", "RTY", "CL", "GC", "SI", "ZB", "ZN", "NG", "ZS"]:
        return TickerInfo(clean, "futures", "polygon", clean, "Futures Contract")
    if re.match(r'^[A-Z]{2,8}$', clean) and len(clean) <= 8:
        return TickerInfo(clean, "stock", "yahoo", clean, "Stock (auto-classified)")

    return TickerInfo(clean, "stock", "yahoo", clean, "Unknown - will try Yahoo Finance")


def get_type_color(asset_type: str) -> str:
    colors = {
        "stock": "green", "crypto": "cyan", "forex": "magenta",
        "etf": "blue", "index": "yellow", "futures": "orange1",
        "economic": "purple", "commodity": "bright_red", "option": "bright_blue",
    }
    return colors.get(asset_type, "white")


def get_type_label(asset_type: str) -> str:
    labels = {
        "stock": "[green]STOCK[/green]",
        "crypto": "[cyan]CRYPTO[/cyan]",
        "forex": "[magenta]FOREX[/magenta]",
        "etf": "[blue]ETF[/blue]",
        "index": "[yellow]INDEX[/yellow]",
        "futures": "[orange1]FUTURES[/orange1]",
        "economic": "[purple]ECONOMIC[/purple]",
        "commodity": "[bright_red]COMMODITY[/bright_red]",
        "option": "[bright_blue]OPTION[/bright_blue]",
    }
    return labels.get(asset_type, asset_type.upper())


# ---------------------------------------------------------------------------
# UI FUNCTIONS
# ---------------------------------------------------------------------------

def show_banner():
    """Display main header"""
    console.print()
    console.print(Panel.fit(
        "[bold blue]ARITHMAX RESEARCH[/bold blue] - [bold white]Data Chest[/bold white]",
        border_style="bright_blue",
        padding=(1, 4),
    ))
    console.print()
    console.print(Panel.fit(
        "[bold]Financial Data Downloader - Interactive Mode[/bold]\n"
        "Auto-detect [green]stocks[/green], [cyan]crypto[/cyan], [magenta]forex[/magenta], "
        "[blue]ETFs[/blue], [yellow]indices[/yellow], and more.\n"
        "Just type tickers and I will figure out the right data source.",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()


def show_ticker_table(symbols: List[str]) -> Table:
    """Display classified tickers in a table."""
    table = Table(
        title="Ticker Classification Results",
        box=box.ROUNDED,
        border_style="bright_green",
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="bold white", width=14)
    table.add_column("Type", width=12)
    table.add_column("Source", style="blue", width=18)
    table.add_column("Description", style="white", width=50)

    for i, symbol in enumerate(symbols, 1):
        info = classify_ticker(symbol)
        table.add_row(
            str(i),
            info.display_name,
            get_type_label(info.asset_type),
            info.source,
            info.description,
        )
    return table


def parse_date_input(prompt_text: str, default: str = "") -> datetime:
    """Parse date input with validation."""
    while True:
        date_str = Prompt.ask(prompt_text, default=default)
        try:
            return datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            console.print("[red]Invalid date format! Please use YYYY-MM-DD[/red]")


def get_symbols_interactive(message: str = "Enter ticker symbols") -> List[str]:
    """
    Get ticker symbols interactively from user.
    Supports comma-separated or one-by-one entry.
    """
    console.print(f"\n[bold]{message}[/bold]")
    console.print("[dim]Enter symbols one at a time, or comma-separated.[/dim]")
    console.print("[dim]Type [bold yellow]done[/bold yellow] when finished, [bold red]cancel[/bold red] to go back.[/dim]")
    console.print()

    # Show suggestion strip
    suggestion_text = "Examples: AAPL, MSFT, NVDA, TSLA, AMZN, SPY, QQQ, BTC-USD, ETH-USD, EURUSD=X"
    console.print(f"[dim]{suggestion_text}[/dim]")
    console.print()

    all_symbols = []

    if HAS_PROMPT_TOOLKIT:
        console.print("[dim]Tab-completion available! Press Tab for suggestions.[/dim]")

    while True:
        if HAS_PROMPT_TOOLKIT:
            try:
                completer = FuzzyWordCompleter(list(POPULAR_TICKERS.keys()) + ["done", "cancel"])
                history = InMemoryHistory()
                raw = pt_prompt(
                    "  Symbol: ",
                    completer=completer,
                    history=history,
                    bottom_toolbar="Type 'done' to finish. Press Tab for suggestions."
                )
            except (KeyboardInterrupt, EOFError):
                raw = "cancel"
        else:
            raw = Prompt.ask("  Symbol", default="")

        raw = raw.strip()

        if raw.lower() == "cancel":
            console.print("[yellow]Cancelled.[/yellow]")
            return []
        if raw.lower() == "done":
            break
        if not raw:
            continue

        if "," in raw:
            symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        else:
            symbols = [s.upper() for s in raw.split() if s.strip()]

        for sym in symbols:
            if sym not in all_symbols:
                info = classify_ticker(sym)
                all_symbols.append(sym)
                console.print(
                    f"  [{get_type_color(info.asset_type)}]{sym}[/{get_type_color(info.asset_type)}] "
                    f"-> {get_type_label(info.asset_type)} via {info.source}"
                )
            else:
                console.print(f"  [dim]{sym} already added[/dim]")

        console.print(f"  [dim]({len(all_symbols)} so far)[/dim]")

    if not all_symbols:
        console.print("[yellow]No symbols entered. Using defaults.[/yellow]")
        return []

    console.print(f"\n[bold green]Selected {len(all_symbols)} symbol(s)![/bold green]")
    return all_symbols


def select_resolution() -> str:
    """Let user select data resolution."""
    resolutions = ["tick", "second", "minute", "hour", "daily"]
    descriptions = {
        "tick": "Every trade - largest files",
        "second": "Every second - high frequency",
        "minute": "Every minute - standard intraday [default]",
        "hour": "Every hour - good balance",
        "daily": "Once per day - smallest files",
    }

    console.print("\n[bold]Select Resolution:[/bold]")

    table = Table(box=box.MINIMAL, show_header=False)
    table.add_column("Option", style="bold", width=6)
    table.add_column("Resolution", style="bold white", width=12)
    table.add_column("Description", width=50)

    for i, res in enumerate(resolutions, 1):
        desc = descriptions.get(res, "")
        default_tag = " [yellow](default)[/yellow]" if res == "minute" else ""
        table.add_row(f"[{i}]", res, desc + default_tag)

    console.print(table)

    choice = Prompt.ask(
        "Choose",
        choices=[str(i) for i in range(1, len(resolutions) + 1)] + resolutions,
        default="3"
    )

    if choice.isdigit():
        return resolutions[int(choice) - 1]
    return choice


def select_source(symbols: List[str]) -> str:
    """
    Determine which sources to use based on classified tickers.
    If all same source, use that. Otherwise ask.
    """
    sources = set()
    for sym in symbols:
        info = classify_ticker(sym)
        sources.add(info.source)

    if len(sources) == 1:
        return sources.pop()
    return "auto"


def ask_additional_options() -> dict:
    """Ask about additional data options."""
    options = {}
    options["fundamentals"] = Confirm.ask("Download fundamentals?", default=False)
    options["news"] = Confirm.ask("Download news?", default=False)
    options["earnings"] = Confirm.ask("Download earnings?", default=False)
    return options


def select_output_format() -> str:
    """Let user select desired output format (lean/raw/parquet)."""
    choices = ["lean", "raw", "csv", "parquet"]
    default = DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in choices else "raw"

    console.print("\n[bold]Select output format:[/bold]")
    console.print("- lean: QuantConnect LEAN-compatible zip files")
    console.print("- raw/csv: flat CSV files under data_chest/")
    console.print("- parquet: Parquet files (pyarrow or fastparquet required)")

    choice = Prompt.ask("Format", choices=choices, default=default)
    return choice
    return options


def show_summary(symbols: List[str], source: str, start_date: datetime,
                 end_date: datetime, resolution: str, options: dict,
                 use_yahoo: bool = True):
    """Show a summary table before download."""
    console.print()
    console.print(Rule(style="bright_blue"))
    console.print("[bold]Download Summary[/bold]")
    console.print(Rule(style="bright_blue"))

    # Show tickers
    console.print(show_ticker_table(symbols))

    # Show config
    config_table = Table(box=box.MINIMAL, show_header=False)
    config_table.add_column("Setting", style="bold", width=18)
    config_table.add_column("Value", width=50)

    config_table.add_row("Source(s)", source if source != "auto" else "Auto-detected per ticker")
    config_table.add_row("Period", f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    config_table.add_row("Resolution", resolution)
    config_table.add_row("Fundamentals", "Yes" if options.get("fundamentals") else "No")
    config_table.add_row("News", "Yes" if options.get("news") else "No")
    config_table.add_row("Earnings", "Yes" if options.get("earnings") else "No")

    console.print()
    console.print(config_table)
    console.print()

def download_for_symbol(symbol: str, source: str, start_date: datetime,
                        end_date: datetime, resolution: str, options: dict,
                        progress, task_id, output_format: Optional[str] = None):
    """Download data for a single symbol using the right source."""
    info = classify_ticker(symbol)

    # Use the classified source if source is "auto"
    actual_source = source if source != "auto" else info.source

    try:
        if actual_source == "yahoo":
            downloader = YahooFinanceDownloader()

            if info.asset_type in ("crypto",):
                downloader.download_crypto_symbols(
                    [symbol], start_date, end_date, resolution, output_format=output_format
                )
            elif info.asset_type in ("forex",):
                downloader.download_forex_pairs(
                    [symbol.replace("=X", "")], resolution, start_date, end_date, output_format=output_format
                )
            elif info.asset_type in ("etf", "index"):
                downloader.download_stock_symbols(
                    [symbol], resolution, start_date, end_date, info.asset_type, output_format=output_format
                )
            else:
                downloader.download_stock_symbols(
                    [symbol], resolution, start_date, end_date, "equity", output_format=output_format
                )

            if options.get("fundamentals"):
                downloader.download_fundamentals([symbol], output_format=output_format)
            if options.get("earnings"):
                downloader.download_earnings([symbol], output_format=output_format)
            if options.get("news"):
                downloader.download_news([symbol], 10, output_format=output_format)

        elif actual_source == "binance":
            downloader = BinanceDataDownloader()
            downloader.download_multiple_symbols(
                [symbol], resolution, start_date, end_date, output_format=output_format
            )

        elif actual_source == "alpaca":
            downloader = AlpacaDataDownloader()
            downloader.download_multiple_symbols(
                [symbol], resolution, start_date, end_date, output_format=output_format
            )

        elif actual_source == "nse-india":
            downloader = NSEIndiaDownloader()
            downloader.download_equity_symbols([symbol], start_date, end_date, output_format=output_format)

        elif actual_source == "bse-india":
            downloader = BSEIndiaDownloader()
            downloader.download_equity_symbols([symbol], start_date, end_date, output_format=output_format)

        elif actual_source == "fred":
            downloader = FREDDownloader()
            downloader.download_economic_series([symbol], start_date, end_date, output_format=output_format)

        elif actual_source == "polygon":
            downloader = PolygonFuturesDownloader()
            downloader.download_symbols([symbol], start_date, end_date, resolution, output_format=output_format)

        elif actual_source == "databento":
            downloader = DatabentoFuturesDownloader()
            downloader.download_symbols([symbol], start_date, end_date, resolution, output_format=output_format)

        elif actual_source == "alpha-vantage":
            downloader = AlphaVantageDownloader()
            if info.asset_type == "crypto":
                downloader.download_crypto_symbols([symbol], resolution, start_date, end_date, output_format=output_format)
            else:
                downloader.download_stock_symbols([symbol], resolution, start_date, end_date, output_format=output_format)

        elif actual_source == "tiingo":
            downloader = TiingoDownloader()
            downloader.download_stock_symbols([symbol], start_date, end_date, resolution, output_format=output_format)

        elif actual_source == "stooq":
            downloader = StooqDownloader()
            downloader.download_stock_symbols([symbol], resolution, start_date, end_date, output_format=output_format)

        elif actual_source == "investing-com":
            downloader = InvestingComDownloader()
            downloader.download_stock_symbols([symbol], resolution, start_date, end_date, output_format=output_format)

        else:
            # Fallback to Yahoo
            downloader = YahooFinanceDownloader()
            downloader.download_stock_symbols(
                [symbol], resolution, start_date, end_date, "equity", output_format=output_format
            )

        progress.update(task_id, advance=1)

    except Exception as e:
        logger.error(f"Error downloading {symbol} via {actual_source}: {str(e)}")
        console.print(f"  [red]Failed: {symbol} - {str(e)}[/red]")
        progress.update(task_id, advance=1)
    



def run_downloads(symbols: List[str], start_date: datetime, end_date: datetime,
                  resolution: str, options: dict, use_yahoo: bool = True,
                  source: str = "auto", output_format: Optional[str] = None):
    """Run the download pipeline for all symbols.

    Args:
        output_format: Optional output format to pass to downloaders ('lean','raw','csv','parquet').
    """
    console.print()
    console.print("[bold]Starting downloads...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(symbols))

        for symbol in symbols:
            progress.update(task, description=f"Processing {symbol}")
            download_for_symbol(
                symbol, source, start_date, end_date,
                resolution, options, progress, task, output_format=output_format
            )

    console.print()
    console.print(Panel.fit(
        "[bold green]All downloads completed![/bold green]\n"
        f"Downloaded {len(symbols)} symbol(s) from "
        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Main Flow
# ---------------------------------------------------------------------------

def main():
    try:
        show_banner()

        # Step 1: Get symbols
        symbols = get_symbols_interactive()
        if not symbols:
            # Use defaults as fallback
            console.print("\n[yellow]No custom symbols entered. You can:[/yellow]")
            console.print("  1. Enter your own stock/crypto/forex symbols")
            console.print("  2. Use the command-line directly (python main.py --source yahoo --equity-symbols YOUR_SYMBOLS)")
            console.print()
            try:
                choice = Prompt.ask("Try again?", choices=["y", "n"], default="n")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Exiting.[/yellow]")
                return
            if choice.lower() == "y":
                symbols = get_symbols_interactive()
            if not symbols:
                console.print("[yellow]Exiting. Please run with your custom symbols next time.[/yellow]")
                return

        # Show how they were classified
        console.print()
        console.print(show_ticker_table(symbols))

        # Step 2: Date range
        console.print()
        end_default = DEFAULT_END_DATE.strftime("%Y-%m-%d")
        start_default = DEFAULT_START_DATE.strftime("%Y-%m-%d")
        start_date = parse_date_input("Start date (YYYY-MM-DD)", default=start_default)
        end_date = parse_date_input("End date (YYYY-MM-DD)", default=end_default)

        # Validate dates
        if start_date >= end_date:
            console.print("[red]Start date must be before end date![/red]")
            return

        # Step 3: Resolution
        resolution = select_resolution()

        # Step 4: Auto-detect source
        source = "auto"

        # Step 5: Additional options
        options = ask_additional_options()

        # Step 5b: Output format selection
        output_format = select_output_format()

        # Step 6: Show summary and confirm
        show_summary(symbols, source, start_date, end_date, resolution, options)

        confirm = Confirm.ask("[bold]Proceed with download?[/bold]", default=True)
        if not confirm:
            console.print("[yellow]Download cancelled.[/yellow]")
            return

        # Step 7: Run the downloads
        run_downloads(symbols, start_date, end_date, resolution, options, source=source, output_format=output_format)

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        logger.exception("Fatal error in interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
