"""
Roster-seeded Quiver Congress Trading scraper.

This script builds a roster of U.S. House and Senate members with terms since
2010, probes Quiver politician profile URLs, parses tradeData from valid pages,
and saves a merged, deduplicated CSV dataset.

It is intentionally separate from quiver_scraper.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

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


MAIN_URL = "https://www.quiverquant.com/congresstrading/"
CURRENT_ROSTER_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/"
    "legislators-current.yaml"
)
HISTORICAL_ROSTER_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/"
    "legislators-historical.yaml"
)


@dataclass
class ScrapeConfig:
    output_csv: str
    start_date: Optional[str]
    end_date: Optional[str]
    start_year: int
    timeout_seconds: int
    headless: bool
    sleep_seconds: float
    max_members: Optional[int]
    include_recent_panel: bool


class QuiverRosterScraper:
    def __init__(self, cfg: ScrapeConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.quiverquant.com/",
            }
        )
        self._selenium_warning_printed = False

    def _build_driver(self) -> webdriver.Chrome:
        options = Options()
        if self.cfg.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
        return webdriver.Chrome(options=options)

    def fetch_html(self, url: str, expected_marker: Optional[str] = None) -> str:
        safe_url = requests.utils.requote_uri(url)

        if SELENIUM_AVAILABLE:
            driver = self._build_driver()
            try:
                driver.get(safe_url)
                wait = WebDriverWait(driver, self.cfg.timeout_seconds)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "script")))
                if expected_marker:
                    try:
                        wait.until(lambda d: expected_marker in d.page_source)
                    except TimeoutException:
                        pass
                return driver.page_source
            finally:
                driver.quit()

        if not self._selenium_warning_printed:
            print("Selenium not installed. Using requests fallback (may get 403 on some profiles).")
            self._selenium_warning_printed = True

        response = self.session.get(safe_url, timeout=self.cfg.timeout_seconds)
        response.raise_for_status()
        return response.text

    @staticmethod
    def parse_js_array_literal(text: str) -> List[list]:
        try:
            return json.loads(text)
        except Exception:
            pass

        safe_text = re.sub(r"\bNaN\b", "null", text)
        safe_text = re.sub(r"\bInfinity\b", "null", safe_text)
        safe_text = re.sub(r"\b-Infinity\b", "null", safe_text)
        safe_text = re.sub(r"\btrue\b", "True", safe_text)
        safe_text = re.sub(r"\bfalse\b", "False", safe_text)
        safe_text = re.sub(r"\bnull\b", "None", safe_text)
        return ast.literal_eval(safe_text)

    @staticmethod
    def normalize_party(value: str) -> str:
        mapping = {"D": "Democrat", "R": "Republican", "I": "Independent"}
        txt = (value or "").strip()
        return mapping.get(txt, txt if txt else "Unknown")

    @staticmethod
    def normalize_transaction_type(value: str) -> str:
        txt = str(value or "").strip().lower()
        replacements = {
            "purchase": "buy",
            "sale": "sell",
            "sale (full)": "sell",
            "sale (partial)": "sell",
            "exchange": "exchange",
        }
        return replacements.get(txt, txt)

    @staticmethod
    def extract_ticker(primary: str, fallback: str) -> Optional[str]:
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

    def load_roster(self) -> pd.DataFrame:
        start_dt = pd.Timestamp(f"{self.cfg.start_year}-01-01")

        def _download_yaml(url: str) -> List[Dict]:
            resp = self.session.get(url, timeout=self.cfg.timeout_seconds)
            resp.raise_for_status()
            return yaml.safe_load(resp.text)

        current = _download_yaml(CURRENT_ROSTER_URL)
        historical = _download_yaml(HISTORICAL_ROSTER_URL)
        members = current + historical

        rows = []
        for member in members:
            bioguide = member.get("id", {}).get("bioguide")
            if not bioguide:
                continue

            name = member.get("name", {})
            first = name.get("first", "")
            middle = name.get("middle", "")
            last = name.get("last", "")
            official_full = name.get("official_full", "")
            nickname = name.get("nickname", "")

            terms = member.get("terms", [])
            has_target_term = False
            chamber = None
            party = None
            for t in terms:
                term_type = t.get("type")
                if term_type not in {"rep", "sen"}:
                    continue
                end = pd.to_datetime(t.get("end"), errors="coerce")
                if pd.isna(end):
                    continue
                if end >= start_dt:
                    has_target_term = True
                    chamber = "House" if term_type == "rep" else "Senate"
                    party = t.get("party")
                    break

            if not has_target_term:
                continue

            rows.append(
                {
                    "bioguide_id": bioguide,
                    "first": first,
                    "middle": middle,
                    "last": last,
                    "official_full": official_full,
                    "nickname": nickname,
                    "chamber_roster": chamber,
                    "party_roster": party,
                }
            )

        roster_df = pd.DataFrame(rows).drop_duplicates(subset=["bioguide_id"]).reset_index(drop=True)
        if self.cfg.max_members and self.cfg.max_members > 0:
            roster_df = roster_df.head(self.cfg.max_members).copy()

        return roster_df

    @staticmethod
    def build_name_variants(row: pd.Series) -> List[str]:
        variants = []

        def add(v: str) -> None:
            cleaned = " ".join(str(v or "").split()).strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        first = row.get("first", "")
        middle = row.get("middle", "")
        last = row.get("last", "")
        nickname = row.get("nickname", "")
        official = row.get("official_full", "")

        add(official)
        add(f"{first} {middle} {last}")
        add(f"{first} {last}")
        if nickname:
            add(f"{nickname} {last}")
            add(f"{first} \"{nickname}\" {last}")

        return variants

    def build_profile_url_candidates(self, row: pd.Series) -> List[str]:
        bioguide = row["bioguide_id"]
        variants = self.build_name_variants(row)
        urls = []
        for name in variants:
            encoded = quote(name, safe="")
            urls.append(f"https://www.quiverquant.com/congresstrading/politician/{encoded}-{bioguide}")
        return urls

    def parse_trade_data(self, html: str, source_url: str) -> pd.DataFrame:
        match = re.search(r"let\s+tradeData\s*=\s*(\[.*?\]);", html, flags=re.S)
        if not match:
            return pd.DataFrame()

        rows = self.parse_js_array_literal(match.group(1))
        records = []
        for r in rows:
            if len(r) < 13:
                continue

            ticker = self.extract_ticker(r[0], r[9] if len(r) > 9 else None)
            if not ticker:
                continue

            records.append(
                {
                    "filing_date": r[2],
                    "transaction_date": r[3],
                    "politician": r[6],
                    "party": self.normalize_party(r[12]),
                    "committee": "Unknown",
                    "ticker": ticker,
                    "transaction_type": self.normalize_transaction_type(r[1]),
                    "amount": r[10] if len(r) > 10 else None,
                    "asset_name": r[8] if len(r) > 8 else None,
                    "asset_type": r[9] if len(r) > 9 else None,
                    "chamber": r[11] if len(r) > 11 else None,
                    "filing_id": r[7] if len(r) > 7 else None,
                    "sector": r[13] if len(r) > 13 else None,
                    "amount_estimate": r[14] if len(r) > 14 else None,
                    "source_profile": source_url,
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df = df.dropna(subset=["filing_date", "transaction_date", "ticker"]).copy()
        return df

    def scrape_recent_panel(self) -> pd.DataFrame:
        html = self.fetch_html(MAIN_URL, expected_marker="recentTradesData")
        match = re.search(r"let\s+recentTradesData\s*=\s*(\[.*?\]);", html, flags=re.S)
        if not match:
            return pd.DataFrame()

        rows = self.parse_js_array_literal(match.group(1))
        records = []
        for r in rows:
            if len(r) < 10:
                continue
            ticker = self.extract_ticker(r[0], r[2])
            if not ticker:
                continue

            records.append(
                {
                    "filing_date": r[8],
                    "transaction_date": r[9],
                    "politician": r[5],
                    "party": self.normalize_party(r[7]),
                    "committee": "Unknown",
                    "ticker": ticker,
                    "transaction_type": self.normalize_transaction_type(r[3]),
                    "amount": r[4],
                    "asset_name": r[1],
                    "asset_type": r[2],
                    "chamber": r[6],
                    "filing_id": r[11] if len(r) > 11 else None,
                    "sector": None,
                    "amount_estimate": None,
                    "source_profile": None,
                    "bioguide_id": r[15] if len(r) > 15 else None,
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df = df.dropna(subset=["filing_date", "transaction_date", "ticker"]).copy()
        return df

    def apply_date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.cfg.start_date:
            out = out[out["filing_date"] >= pd.to_datetime(self.cfg.start_date)]
        if self.cfg.end_date:
            out = out[out["filing_date"] <= pd.to_datetime(self.cfg.end_date)]
        return out.reset_index(drop=True)

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        if "filing_id" in df.columns and df["filing_id"].notna().any():
            out = df.sort_values("filing_date").drop_duplicates(subset=["filing_id"], keep="last")
        else:
            out = df.sort_values("filing_date").drop_duplicates(
                subset=["filing_date", "transaction_date", "politician", "ticker", "amount"],
                keep="last",
            )
        return out.reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        roster = self.load_roster()
        total = len(roster)
        print(f"Roster members in scope: {total}")

        collected = []
        success = 0
        fail = 0
        skipped = 0

        for idx, (_, row) in enumerate(roster.iterrows(), start=1):
            candidate_urls = self.build_profile_url_candidates(row)
            profile_df = pd.DataFrame()
            matched_url = None

            for url in candidate_urls:
                try:
                    html = self.fetch_html(url, expected_marker="tradeData")
                    parsed = self.parse_trade_data(html, source_url=url)
                    if len(parsed) > 0:
                        profile_df = parsed
                        matched_url = url
                        break
                except Exception:
                    continue

            if len(profile_df) > 0:
                profile_df["bioguide_id"] = row["bioguide_id"]
                profile_df["chamber_roster"] = row.get("chamber_roster")
                profile_df["party_roster"] = row.get("party_roster")
                collected.append(profile_df)
                success += 1
            else:
                # Some members may have no profile or no trade disclosures.
                skipped += 1

            if idx % 10 == 0 or idx == total:
                pct = 100.0 * idx / total if total else 100.0
                rows_so_far = sum(len(x) for x in collected)
                print(
                    f"Progress: {idx}/{total} ({pct:.1f}%) | "
                    f"success={success} skipped={skipped} fail={fail} | rows_so_far={rows_so_far}",
                    flush=True,
                )

            if self.cfg.sleep_seconds > 0:
                time.sleep(self.cfg.sleep_seconds)

        if not collected:
            raise RuntimeError("No roster-seeded trade rows were collected.")

        merged = pd.concat(collected, ignore_index=True)

        if self.cfg.include_recent_panel:
            try:
                recent = self.scrape_recent_panel()
                if len(recent) > 0:
                    merged = pd.concat([merged, recent], ignore_index=True)
                    print(f"Merged recent panel rows: {len(recent)}")
            except Exception as exc:
                print(f"WARN: Could not merge recent panel rows: {exc}")

        merged = self.apply_date_filter(merged)
        merged = self.deduplicate(merged)

        output_dir = os.path.dirname(self.cfg.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        merged.to_csv(self.cfg.output_csv, index=False)

        print("Saved roster-seeded Quiver dataset")
        print(f"Output: {self.cfg.output_csv}")
        print(f"Rows: {len(merged)}")
        print(f"Unique politicians: {merged['politician'].nunique() if len(merged) else 0}")
        print(f"Unique tickers: {merged['ticker'].nunique() if len(merged) else 0}")
        if len(merged) > 0:
            print(f"Date range: {merged['filing_date'].min()} -> {merged['filing_date'].max()}")

        return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roster-seeded Quiver Congress Trading scraper")
    parser.add_argument("--output", default="data/quiver_congress_trades_roster.csv")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--headed", action="store_true", help="Run Chrome in non-headless mode")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--max-members", type=int, default=None)
    parser.add_argument(
        "--exclude-recent-panel",
        action="store_true",
        help="Do not merge recent panel rows from /congresstrading/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ScrapeConfig(
        output_csv=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        start_year=args.start_year,
        timeout_seconds=args.timeout,
        headless=not args.headed,
        sleep_seconds=args.sleep_seconds,
        max_members=args.max_members,
        include_recent_panel=not args.exclude_recent_panel,
    )
    scraper = QuiverRosterScraper(cfg)
    scraper.run()


if __name__ == "__main__":
    main()
