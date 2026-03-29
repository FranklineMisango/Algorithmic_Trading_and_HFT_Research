"""
Scrape Quiver Congress Trading data and build a dataset CSV.

This script uses Selenium to render the page, then BeautifulSoup to parse
embedded trade data from page scripts. The output schema is compatible with
data_acquisition.py's CSV loader.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


QUVER_URL = "https://www.quiverquant.com/congresstrading/"


@dataclass
class ScrapeConfig:
    output_csv: str
    start_date: Optional[str]
    end_date: Optional[str]
    headless: bool
    timeout_seconds: int
    mode: str
    max_politicians: Optional[int]
    sleep_seconds: float


def load_project_defaults(config_path: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Read defaults from config.yaml when available."""
    if not os.path.exists(config_path):
        return None, None, None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_csv = cfg.get("data_sources", {}).get("congressional_trades", {}).get("csv_path")
    start_date = cfg.get("data", {}).get("start_date")
    end_date = cfg.get("data", {}).get("end_date")
    return output_csv, start_date, end_date


def build_driver(headless: bool) -> webdriver.Chrome:
    """Create a Chrome WebDriver instance."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    return webdriver.Chrome(options=options)


def fetch_rendered_html(
    url: str,
    timeout_seconds: int,
    headless: bool,
    expected_marker: Optional[str] = None,
) -> str:
    """Open target page with Selenium and return fully rendered HTML."""
    if not SELENIUM_AVAILABLE:
        print("Selenium not installed in current environment. Falling back to requests fetch.")
        safe_url = requests.utils.requote_uri(url)
        response = requests.get(
            safe_url,
            timeout=timeout_seconds,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Referer": "https://www.quiverquant.com/",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        response.raise_for_status()
        return response.text

    driver = build_driver(headless=headless)
    try:
        driver.get(url)
        wait = WebDriverWait(driver, timeout_seconds)

        # Wait for the page to load scripts that include trade data.
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "script")))

        if expected_marker:
            try:
                wait.until(lambda d: expected_marker in d.page_source)
            except TimeoutException:
                pass

        return driver.page_source
    finally:
        driver.quit()


def normalize_party(value: str) -> str:
    mapping = {
        "D": "Democrat",
        "R": "Republican",
        "I": "Independent",
    }
    txt = (value or "").strip()
    return mapping.get(txt, txt if txt else "Unknown")


def extract_ticker(primary: str, fallback: str) -> Optional[str]:
    """Pick the first ticker-looking token from candidate fields."""
    candidates = [primary, fallback]
    pattern = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")

    for candidate in candidates:
        if candidate is None:
            continue
        txt = str(candidate).strip().upper()
        if not txt or txt in {"-", "STOCK", "ETF", "OPTION"}:
            continue
        if pattern.match(txt):
            return txt
    return None


def normalize_transaction_type(value: str) -> str:
    """Normalize transaction labels to strategy-friendly values."""
    txt = str(value or "").strip().lower()
    replacements = {
        "purchase": "buy",
        "sale": "sell",
        "sale (full)": "sell",
        "sale (partial)": "sell",
        "exchange": "exchange",
    }
    return replacements.get(txt, txt)


def parse_js_array_literal(array_text: str) -> List[list]:
    """Parse JavaScript array literals that may include NaN/null/true/false."""
    try:
        return json.loads(array_text)
    except Exception:
        pass

    safe_text = re.sub(r"\bNaN\b", "null", array_text)
    safe_text = re.sub(r"\bInfinity\b", "null", safe_text)
    safe_text = re.sub(r"\b-Infinity\b", "null", safe_text)
    safe_text = re.sub(r"\btrue\b", "True", safe_text)
    safe_text = re.sub(r"\bfalse\b", "False", safe_text)
    safe_text = re.sub(r"\bnull\b", "None", safe_text)
    return ast.literal_eval(safe_text)


def parse_recent_trades_from_html(html: str) -> pd.DataFrame:
    """Parse recentTradesData JS array from rendered HTML."""
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")

    target_text = None
    for script in scripts:
        text = script.string or script.get_text() or ""
        if "recentTradesData" in text:
            target_text = text
            break

    if not target_text:
        raise RuntimeError("Could not find recentTradesData in page source.")

    match = re.search(r"let\s+recentTradesData\s*=\s*(\[.*?\]);", target_text, flags=re.S)
    if not match:
        raise RuntimeError("Could not parse recentTradesData array.")

    rows: List[list] = parse_js_array_literal(match.group(1))

    records = []
    for row in rows:
        if len(row) < 10:
            continue

        ticker = extract_ticker(row[0], row[2])
        if not ticker:
            continue

        records.append(
            {
                "filing_date": row[8],
                "transaction_date": row[9],
                "politician": row[5],
                "party": normalize_party(row[7]),
                "committee": "Unknown",
                "ticker": ticker,
                "transaction_type": row[3],
                "amount": row[4],
                "asset_name": row[1],
                "chamber": row[6],
                "filing_id": row[11] if len(row) > 11 else None,
                "politician_display": row[13] if len(row) > 13 else None,
                "bioguide_id": row[15] if len(row) > 15 else None,
            }
        )

    if not records:
        raise RuntimeError("No valid ticker rows extracted from recentTradesData.")

    df = pd.DataFrame(records)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["transaction_type"] = df["transaction_type"].apply(normalize_transaction_type)
    df = df.dropna(subset=["filing_date", "transaction_date", "ticker"]).copy()
    df = df.sort_values("filing_date").reset_index(drop=True)
    return df


def extract_politician_urls(main_html: str) -> List[str]:
    """Extract politician profile URLs from the main congress trading page."""
    soup = BeautifulSoup(main_html, "html.parser")
    urls = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if "/congresstrading/politician/" not in href:
            continue
        if "/net-worth" in href:
            continue
        full_url = urljoin(QUVER_URL, href)
        urls.add(full_url.split("?")[0])

    return sorted(urls)


def parse_politician_trade_data(page_html: str, profile_url: str) -> pd.DataFrame:
    """Parse tradeData array from a politician profile page."""
    match = re.search(r"let\s+tradeData\s*=\s*(\[.*?\]);", page_html, flags=re.S)
    if not match:
        return pd.DataFrame()

    rows: List[list] = parse_js_array_literal(match.group(1))
    records = []
    for row in rows:
        if len(row) < 13:
            continue

        ticker = extract_ticker(row[0], row[9] if len(row) > 9 else None)
        if not ticker:
            continue

        records.append(
            {
                "filing_date": row[2],
                "transaction_date": row[3],
                "politician": row[6],
                "party": normalize_party(row[12]),
                "committee": "Unknown",
                "ticker": ticker,
                "transaction_type": normalize_transaction_type(row[1]),
                "amount": row[10] if len(row) > 10 else None,
                "asset_name": row[8] if len(row) > 8 else None,
                "asset_type": row[9] if len(row) > 9 else None,
                "chamber": row[11] if len(row) > 11 else None,
                "filing_id": row[7] if len(row) > 7 else None,
                "sector": row[13] if len(row) > 13 else None,
                "amount_estimate": row[14] if len(row) > 14 else None,
                "source_profile": profile_url,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["filing_date", "transaction_date", "ticker"]).copy()
    return df


def scrape_full_history_dataset(main_html: str, cfg: ScrapeConfig) -> pd.DataFrame:
    """Build a larger dataset by crawling politician pages and extracting tradeData."""
    politician_urls = extract_politician_urls(main_html)
    if cfg.max_politicians and cfg.max_politicians > 0:
        politician_urls = politician_urls[: cfg.max_politicians]

    total_profiles = len(politician_urls)
    print(f"Found {total_profiles} politician profile URLs")

    dfs = []
    recent_df = parse_recent_trades_from_html(main_html)
    dfs.append(recent_df)

    success_count = 0
    fail_count = 0
    row_total = len(recent_df)

    for idx, profile_url in enumerate(politician_urls, start=1):
        try:
            print(f"Profile {idx}/{total_profiles}: {profile_url}", flush=True)
            page_html = fetch_rendered_html(
                profile_url,
                timeout_seconds=cfg.timeout_seconds,
                headless=cfg.headless,
                expected_marker="tradeData",
            )
            pol_df = parse_politician_trade_data(page_html, profile_url)
            if len(pol_df) > 0:
                dfs.append(pol_df)
                row_total += len(pol_df)

            success_count += 1

            if idx % 5 == 0 or idx == total_profiles:
                progress_pct = 100.0 * idx / total_profiles if total_profiles else 100.0
                print(
                    f"Progress: {idx}/{total_profiles} ({progress_pct:.1f}%) | "
                    f"success={success_count} fail={fail_count} | rows_so_far={row_total}",
                    flush=True,
                )

            if cfg.sleep_seconds > 0:
                time.sleep(cfg.sleep_seconds)
        except Exception as exc:
            fail_count += 1
            print(f"WARN: Failed profile {profile_url}: {exc}")

            if idx % 5 == 0 or idx == total_profiles:
                progress_pct = 100.0 * idx / total_profiles if total_profiles else 100.0
                print(
                    f"Progress: {idx}/{total_profiles} ({progress_pct:.1f}%) | "
                    f"success={success_count} fail={fail_count} | rows_so_far={row_total}",
                    flush=True,
                )

    print(
        f"Profile crawl complete: success={success_count}, fail={fail_count}, "
        f"profiles={total_profiles}",
        flush=True,
    )

    merged = pd.concat(dfs, ignore_index=True)
    if "filing_id" in merged.columns and merged["filing_id"].notna().any():
        merged = merged.sort_values("filing_date").drop_duplicates(subset=["filing_id"], keep="last")
    else:
        merged = merged.sort_values("filing_date").drop_duplicates(
            subset=["filing_date", "transaction_date", "politician", "ticker", "amount"],
            keep="last",
        )

    merged = merged.reset_index(drop=True)
    return merged


def apply_date_filter(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["filing_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["filing_date"] <= pd.to_datetime(end_date)]
    return filtered.reset_index(drop=True)


def run_scrape(cfg: ScrapeConfig) -> None:
    html = fetch_rendered_html(
        QUVER_URL,
        timeout_seconds=cfg.timeout_seconds,
        headless=cfg.headless,
        expected_marker="recentTradesData",
    )
    mode = cfg.mode.lower()
    if mode == "full-history":
        df = scrape_full_history_dataset(html, cfg)
    else:
        df = parse_recent_trades_from_html(html)

    df = apply_date_filter(df, cfg.start_date, cfg.end_date)

    output_dir = os.path.dirname(cfg.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(cfg.output_csv, index=False)
    print("Saved Quiver congress dataset")
    print(f"Output: {cfg.output_csv}")
    print(f"Rows: {len(df)}")
    if len(df) > 0:
        print(f"Date range: {df['filing_date'].min()} -> {df['filing_date'].max()}")
        print(f"Unique tickers: {df['ticker'].nunique()}")


def parse_args() -> argparse.Namespace:
    default_csv, _, _ = load_project_defaults("config.yaml")

    parser = argparse.ArgumentParser(description="Scrape Quiver Congress Trading dataset")
    parser.add_argument("--output", default=default_csv or "data/quiver_congress_trades.csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--mode", choices=["recent", "full-history"], default="full-history")
    parser.add_argument("--max-politicians", type=int, default=None, help="Optional cap for profile crawl count")
    parser.add_argument("--sleep-seconds", type=float, default=0.15, help="Delay between profile requests")
    parser.add_argument("--headed", action="store_true", help="Run Chrome in non-headless mode")
    parser.add_argument("--timeout", type=int, default=45)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scrape_cfg = ScrapeConfig(
        output_csv=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        headless=not args.headed,
        timeout_seconds=args.timeout,
        mode=args.mode,
        max_politicians=args.max_politicians,
        sleep_seconds=args.sleep_seconds,
    )
    run_scrape(scrape_cfg)
