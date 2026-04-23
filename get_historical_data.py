"""
get_historical_data.py
──────────────────────
Fetches ALL historical data from the BudgetBakers Wallet REST API
and saves it locally as Parquet files (one per dataset).

Datasets saved
──────────────
  historical_data/
  ├── records.parquet       ← every transaction ever recorded
  ├── accounts.parquet      ← your wallet accounts
  ├── categories.parquet    ← your category tree
  └── metadata.json         ← fetch timestamp, record counts, API sync status

Why Parquet?
────────────
  • Typed columns  – dates stay dates, numbers stay numbers (no CSV string conversion)
  • Compressed     – typically 5-10× smaller than equivalent CSV
  • Fast           – pandas reads it in milliseconds even for 100k+ rows
  • Wide support   – readable by pandas, Polars, DuckDB, Power BI, Spark, etc.
  • CSV export     – run with --csv flag to also write .csv alongside each .parquet

Usage
──────
  python get_historical_data.py           # fetch everything, save Parquet
  python get_historical_data.py --csv     # also write CSV copies
  python get_historical_data.py --update  # only fetch records newer than last save

Requirements
────────────
  pip install requests pandas pyarrow python-dotenv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    import boto3
    from botocore.config import Config as BotoConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

BASE_URL   = "https://rest.budgetbakers.com/wallet/v1/api"
TOKEN      = os.getenv("API_TOKEN", "")
OUTPUT_DIR = Path("historical_data")
META_FILE  = OUTPUT_DIR / "metadata.json"

# Max records per page (API ceiling)
PAGE_SIZE  = 200
# Pause between pages to stay well under the 500 req/hour rate limit
PAGE_DELAY = 0.05   # seconds

# R2 config (optional – only needed for --upload)
R2_ENDPOINT          = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET            = os.getenv("R2_BUCKET", "")


# ── Helpers ────────────────────────────────────────────────────────────────────
def make_headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}


def api_get(path: str, params: dict | None = None, retries: int = 3) -> dict | list | None:
    """GET with retries and rate-limit back-off."""
    url = f"{BASE_URL}{path}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=make_headers(), params=params or {}, timeout=20)
        except requests.exceptions.ConnectionError as e:
            print(f"  ✗ Connection error: {e}")
            return None

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 401:
            print("  ✗ Invalid or expired API token. Check your .env file.")
            sys.exit(1)

        if resp.status_code == 409:
            wait = int(resp.json().get("retry_after_minutes", 5)) * 60
            print(f"  ⏳ Initial sync in progress. Retrying in {wait // 60} min…")
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            # Respect rate limit: wait proportionally
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 0))
            wait = 60 if remaining == 0 else 10
            print(f"  ⚠ Rate limited. Waiting {wait}s (attempt {attempt}/{retries})…")
            time.sleep(wait)
            continue

        print(f"  ✗ HTTP {resp.status_code} on {path} — {resp.text[:200]}")
        return None

    print(f"  ✗ Failed after {retries} attempts: {path}")
    return None


def fetch_all_pages(path: str, extra_params: dict | None = None) -> list[dict]:
    """Paginate through every page of a User Data endpoint."""
    params = {"limit": PAGE_SIZE, "offset": 0}
    if extra_params:
        params.update(extra_params)

    results = []
    page = 1

    while True:
        print(f"    page {page:>4}  (offset {params['offset']:>6})…", end="", flush=True)
        data = api_get(path, params)

        if data is None:
            print(" ✗ error")
            break

        # Normalise: endpoint may return a list or a dict with a known key
        if isinstance(data, list):
            chunk = data
            next_offset = None
        else:
            chunk = (
                data.get("records")
                or data.get("items")
                or data.get("data")
                or []
            )
            next_offset = data.get("nextOffset")

        results.extend(chunk)
        print(f" {len(chunk):>4} records  (total so far: {len(results):>6})")

        if not chunk or next_offset is None:
            break

        params["offset"] = next_offset
        page += 1
        time.sleep(PAGE_DELAY)

    return results


def load_metadata() -> dict:
    if META_FILE.exists():
        with open(META_FILE) as f:
            return json.load(f)
    return {}


def save_metadata(meta: dict) -> None:
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def to_parquet(df: pd.DataFrame, name: str, also_csv: bool = False) -> None:
    path = OUTPUT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    size_kb = path.stat().st_size / 1024
    print(f"  💾 Saved {path}  ({len(df):,} rows, {size_kb:.1f} KB)")

    if also_csv:
        csv_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        csv_kb = csv_path.stat().st_size / 1024
        print(f"  📄 Also saved {csv_path}  ({csv_kb:.1f} KB)")


def upload_to_r2(file_path: Path) -> None:
    """Upload a local file to Cloudflare R2."""
    if not HAS_BOTO3:
        print("  ✗ boto3 not installed. Run: pip install boto3")
        return
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )
    key = file_path.name  # e.g. "records.parquet"
    s3.upload_file(str(file_path), R2_BUCKET, key)
    size_kb = file_path.stat().st_size / 1024
    print(f"  ☁️  Uploaded {key} to R2  ({size_kb:.1f} KB)")


# ── Dataset fetchers ───────────────────────────────────────────────────────────
def fetch_records(since: str | None = None) -> pd.DataFrame:
    """Fetch transactions. If `since` is given, only fetch newer ones."""
    params = {}
    if since:
        params["recordDate"] = f"gte.{since}"
        print(f"  → Incremental mode: fetching records from {since} onward")
    else:
        # No date filter — paginate through all records the API has
        print("  → Full mode: fetching all available records")

    rows = fetch_all_pages("/records", params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Type-cast known columns
    for col in ["recordDate", "createdAt", "updatedAt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ["amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_accounts() -> pd.DataFrame:
    print("  Fetching accounts…")
    data = api_get("/accounts")
    if data is None:
        return pd.DataFrame()
    rows = data if isinstance(data, list) else (
        data.get("accounts") or data.get("items") or data.get("data") or []
    )
    df = pd.DataFrame(rows)
    for col in ["balance", "initialBalance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_categories() -> pd.DataFrame:
    print("  Fetching categories…")
    data = api_get("/categories")
    if data is None:
        return pd.DataFrame()
    rows = data if isinstance(data, list) else (
        data.get("categories") or data.get("items") or data.get("data") or []
    )
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download all BudgetBakers Wallet data.")
    parser.add_argument("--csv",    action="store_true", help="Also export CSV copies.")
    parser.add_argument("--update", action="store_true",
                        help="Incremental: only fetch records newer than the last saved run.")
    parser.add_argument("--upload", action="store_true",
                        help="Upload parquet files to Cloudflare R2 after saving locally.")
    args = parser.parse_args()

    if not TOKEN:
        print("✗ API_TOKEN not found in .env. Aborting.")
        sys.exit(1)

    if args.upload and not all([R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET]):
        print("✗ R2 credentials not found in .env. Add R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.")
        sys.exit(1)

    if args.upload and not HAS_BOTO3:
        print("✗ boto3 is required for --upload. Run: pip install boto3")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    meta = load_metadata()

    print("\n══════════════════════════════════════════")
    print("  BudgetBakers — Historical Data Fetcher")
    print("══════════════════════════════════════════\n")

    # ── Records ───────────────────────────────────────────────────────────────
    since = None
    existing_records = pd.DataFrame()

    if args.update and "last_fetch_utc" in meta:
        # Use the last known most-recent record date as the lower bound
        since = meta.get("latest_record_date")
        records_path = OUTPUT_DIR / "records.parquet"
        if records_path.exists():
            print("📂 Loading existing records to merge with new ones…")
            existing_records = pd.read_parquet(records_path)

    print("📥 Fetching records (transactions)…")
    new_records = fetch_records(since=since)

    if not new_records.empty:
        if not existing_records.empty:
            # Merge and deduplicate by record id
            id_col = "id" if "id" in new_records.columns else None
            combined = pd.concat([existing_records, new_records], ignore_index=True)
            if id_col:
                combined = combined.drop_duplicates(subset=[id_col], keep="last")
            records_df = combined.sort_values("recordDate", ascending=False)
            print(f"  ↳ Merged {len(existing_records):,} existing + {len(new_records):,} new "
                  f"→ {len(records_df):,} total (deduplicated)")
        else:
            records_df = new_records.sort_values("recordDate", ascending=False)

        to_parquet(records_df, "records", also_csv=args.csv)

        # Track the latest record date for future incremental runs
        if "recordDate" in records_df.columns:
            latest = records_df["recordDate"].max()
            meta["latest_record_date"] = str(latest.date()) if pd.notna(latest) else None
    else:
        print("  ℹ No records returned.")
        records_df = existing_records

    # ── Accounts ──────────────────────────────────────────────────────────────
    print("\n📥 Fetching accounts…")
    accounts_df = fetch_accounts()
    if not accounts_df.empty:
        to_parquet(accounts_df, "accounts", also_csv=args.csv)

    # ── Categories ────────────────────────────────────────────────────────────
    print("\n📥 Fetching categories…")
    categories_df = fetch_categories()
    if not categories_df.empty:
        to_parquet(categories_df, "categories", also_csv=args.csv)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta.update({
        "last_fetch_utc":   datetime.now(timezone.utc).isoformat(),
        "record_count":     len(records_df),
        "account_count":    len(accounts_df),
        "category_count":   len(categories_df),
        "incremental_run":  args.update,
    })
    save_metadata(meta)
    # ── Upload to R2 ──────────────────────────────────────────────────────────
    if args.upload:
        print("\n☁️  Uploading to Cloudflare R2…")
        for name in ["records", "accounts", "categories"]:
            fpath = OUTPUT_DIR / f"{name}.parquet"
            if fpath.exists():
                upload_to_r2(fpath)
        # Also upload metadata.json
        upload_to_r2(META_FILE)
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  ✅ Done!")
    print(f"     Records   : {len(records_df):>8,}")
    print(f"     Accounts  : {len(accounts_df):>8,}")
    print(f"     Categories: {len(categories_df):>8,}")
    print(f"     Saved to  : {OUTPUT_DIR.resolve()}")
    print("══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
