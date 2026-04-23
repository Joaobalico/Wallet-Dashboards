"""
BudgetBakers Wallet – Streamlit Dashboard
Connects to the official REST API at https://rest.budgetbakers.com/wallet
Requires a Premium Wallet account and a Bearer API token.
"""

import hmac
import os
from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import numpy as np
from calendar import monthrange
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()  # reads .env for local dev; on Streamlit Cloud, st.secrets is used

HISTORICAL_DIR = Path("historical_data")
CURRENCY = "€"

# ── Portuguese → English category name mapping ─────────────
CATEGORY_TRANSLATION = {
    "Combustível": "Fuel",
    "Comidas e bebidas": "Food & Drinks",
    "Compras": "Shopping",
    "Farmácia": "Pharmacy",
    "Impostos": "Taxes",
    "Investimentos financeiros": "Financial Investments",
    "Passatempos": "Hobbies",
    "Restaurante, fast food": "Restaurant, Fast Food",
}

# ── Configuration ───────────────────────────────────────────
BASE_URL = "https://rest.budgetbakers.com/wallet/v1/api"
# Prefer st.secrets (Streamlit Cloud); fall back to env var (local dev)
TOKEN = st.secrets.get("API_TOKEN", "") or os.getenv("API_TOKEN", "")

st.set_page_config(
    page_title="BudgetBakers Dashboard",
    page_icon="💰",
    layout="wide",
)

# ── Password gate ──────────────────────────────────────────
def check_password() -> bool:
    """Show a login form and return True only if the correct password is entered."""
    def _password_entered():
        """Check whether the password is correct."""
        if hmac.compare_digest(
            st.session_state.get("password", ""),
            st.secrets.get("APP_PASSWORD", ""),
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't keep it around
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # No APP_PASSWORD configured → skip auth (local dev)
    if not st.secrets.get("APP_PASSWORD", ""):
        return True

    st.text_input(
        "Password", type="password", key="password",
        on_change=_password_entered,
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Incorrect password")
    return False


if not check_password():
    st.stop()

# ── Custom CSS for better styling ──────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #ddd;
    }
    [data-testid="stMetric"]:nth-of-type(1) {
        border-left-color: #2ecc71;
    }
    [data-testid="stMetric"]:nth-of-type(2) {
        border-left-color: #e74c3c;
    }
    [data-testid="stMetric"]:nth-of-type(3) {
        border-left-color: #3498db;
    }
    /* Scale down the whole app so it fits at 100% browser zoom */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-size: 13px;
    }
    [data-testid="stMetric"] {
        padding: 8px 12px;
    }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
    /* Force sidebar radio buttons to 2 per row */
    [data-testid="stSidebar"] [role="radiogroup"] {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 4px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── API helpers ────────────────────────────────────────────────────────────────
def make_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def api_get(token: str, path: str, params: dict | None = None) -> dict | list | None:
    """GET wrapper with error handling."""
    try:
        resp = requests.get(
            f"{BASE_URL}{path}",
            headers=make_headers(token),
            params=params or {},
            timeout=15,
        )
        if resp.status_code == 401:
            st.error("❌ Invalid or expired API token. Please check your token.")
            return None
        if resp.status_code == 409:
            st.warning(
                "⏳ Initial data sync is still in progress. "
                "Please wait a few minutes and refresh."
            )
            return None
        if resp.status_code == 429:
            st.warning("⚠️ Rate limit reached (500 req/hour). Please wait before retrying.")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("🌐 Could not reach the BudgetBakers API. Check your internet connection.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def fetch_all_pages(token: str, path: str, params: dict | None = None) -> list[dict]:
    """Fetch all pages for a paginated endpoint."""
    params = dict(params or {})
    params.setdefault("limit", 200)
    results = []
    offset = 0
    while True:
        params["offset"] = offset
        data = api_get(token, path, params)
        if data is None:
            break
        # Records endpoint returns {"records": [...], "nextOffset": ...}
        # Some endpoints return lists directly
        if isinstance(data, list):
            results.extend(data)
            break
        items = data.get("records") or data.get("items") or data.get("data") or []
        results.extend(items)
        next_offset = data.get("nextOffset")
        if next_offset is None:
            break
        offset = next_offset
    return results


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Data source ───────────────────────────────────────────────────────────
    historical_available = (HISTORICAL_DIR / "records.parquet").exists()

    st.subheader("Data source")
    if historical_available:
        import json as _json
        _meta = {}
        if (HISTORICAL_DIR / "metadata.json").exists():
            with open(HISTORICAL_DIR / "metadata.json") as _f:
                _meta = _json.load(_f)
        _fetched_at = _meta.get("last_fetch_utc", "unknown")[:19].replace("T", " ")
        st.caption(f"Local snapshot last updated: **{_fetched_at} UTC**")
        use_local = st.radio(
            "Load from",
            ["📂 Local historical data", "🌐 Live API"],
            index=0,
        ) == "📂 Local historical data"
    else:
        use_local = False
        st.caption("No local snapshot found. Run `get_historical_data.py` to create one.")

    st.divider()

    st.subheader("Filters")
    today = date.today()

    # Quick date range buttons
    def _on_default_toggle():
        """When Default is toggled off, restore previous radio selection or default to Custom."""
        if not st.session_state["default_toggle"]:
            prev = st.session_state.get("_prev_quick_range")
            st.session_state["quick_range"] = prev if prev else "Custom"

    def _on_quick_range_change():
        """Remember the last radio selection so toggling Default preserves it."""
        st.session_state["_prev_quick_range"] = st.session_state["quick_range"]

    is_default = st.toggle("Default", value=True, key="default_toggle",
                           on_change=_on_default_toggle)

    quick_range_radio = st.radio(
        "Quick range",
        ["Custom", "Today", "This Week", "This Month", "6 Months", "This Year"],
        index=None if is_default else 0,
        horizontal=True,
        key="quick_range",
        disabled=is_default,
        on_change=_on_quick_range_change,
    )

    quick_range = "Default" if is_default else (quick_range_radio or "Custom")

    if quick_range == "Default":
        # From the most recent 26th to today
        if today.day >= 26:
            _qfrom = today.replace(day=26)
        else:
            first_this_month = today.replace(day=1)
            prev_month = first_this_month - timedelta(days=1)
            _qfrom = prev_month.replace(day=26)
        _qto = today
    elif quick_range == "Today":
        _qfrom, _qto = today, today
    elif quick_range == "This Week":
        _qfrom = today - timedelta(days=(today.weekday() + 1) % 7)  # Sunday
        _qto = today
    elif quick_range == "This Month":
        _qfrom = today.replace(day=1)
        _qto = today
    elif quick_range == "6 Months":
        _qfrom = (today.replace(day=1) - timedelta(days=180)).replace(day=1)
        _qto = today
    elif quick_range == "This Year":
        _qfrom = date(today.year, 1, 1)
        _qto = today
    else:  # Custom
        _qfrom = today - timedelta(days=30)
        _qto = today

    date_from = st.date_input("From", value=_qfrom)
    date_to = st.date_input("To", value=_qto)

    if date_from > date_to:
        st.error("'From' must be before 'To'.")

    min_amount = st.number_input("Min amount", value=0.0, step=1.0)
    max_amount = st.number_input("Max amount", value=0.0, step=1.0,
                                  help="Set to 0 to ignore this filter.")

    payee_filter = st.text_input("Payee contains", placeholder="e.g. Amazon")
    note_filter  = st.text_input("Note contains",  placeholder="e.g. grocery")

    st.divider()

    # Label exclusion filter – populated after data loads
    exclude_labels_placeholder = st.empty()

    st.divider()
    refresh = st.button("🔄 Refresh data", use_container_width=True)

    st.caption(
        "Rate limit: **500 req / hour**. "
        "Data syncs from the Wallet app asynchronously."
    )


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("💰 BudgetBakers Wallet Dashboards")

token = TOKEN  # sourced from .env

if not token:
    st.error(
        "⚠️ No API token found. Add `API_TOKEN=your_token` to your `.env` file "
        "and restart the app.\n\n"
        "You can generate a token at "
        "[web.budgetbakers.com/settings/apiTokens](https://web.budgetbakers.com/settings/apiTokens) "
        "(Premium plan required)."
    )
    st.stop()

# ── Fetch data ─────────────────────────────────────────────────────────────────
if use_local:
    with st.spinner("Loading from local historical data…"):
        df = pd.read_parquet(HISTORICAL_DIR / "records.parquet")

        # Extract amount from nested baseAmount dict (API stores {value, currencyCode})
        if "baseAmount" in df.columns:
            df["amount"] = df["baseAmount"].apply(
                lambda x: x["value"] if isinstance(x, dict) and "value" in x else None
            )
        df["amount"]     = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

        # Extract category info from nested category dict
        if "category" in df.columns and df["category"].apply(lambda x: isinstance(x, dict)).any():
            df["categoryId"] = df["category"].apply(
                lambda x: x.get("id") if isinstance(x, dict) else None
            )
            df["categoryName"] = df["category"].apply(
                lambda x: x.get("name") if isinstance(x, dict) else None
            )

        df["recordDate"] = pd.to_datetime(df["recordDate"], errors="coerce", utc=True)
        df["recordDate"] = df["recordDate"].dt.tz_localize(None)  # make tz-naive for comparisons

        # Ensure all expected columns exist (parity with Live API path)
        for col in ["amount", "payee", "note", "recordDate", "categoryId", "accountId", "type"]:
            if col not in df.columns:
                df[col] = None

        # Keep a full copy before filtering (used for period comparison)
        df_all_raw = df.copy()

        # Apply date filter on the full local dataset
        mask = (
            (df["recordDate"].dt.date >= date_from) &
            (df["recordDate"].dt.date <= date_to)
        )
        if min_amount > 0:
            mask &= df["amount"].abs() >= min_amount
        if max_amount > 0:
            mask &= df["amount"].abs() <= max_amount
        if payee_filter and "payee" in df.columns:
            mask &= df["payee"].str.contains(payee_filter, case=False, na=False)
        if note_filter and "note" in df.columns:
            mask &= df["note"].str.contains(note_filter, case=False, na=False)
        df = df[mask].copy()

        # Load reference tables
        acc_df  = pd.read_parquet(HISTORICAL_DIR / "accounts.parquet")   if (HISTORICAL_DIR / "accounts.parquet").exists()   else pd.DataFrame()
        cat_df  = pd.read_parquet(HISTORICAL_DIR / "categories.parquet") if (HISTORICAL_DIR / "categories.parquet").exists() else pd.DataFrame()
        account_map  = dict(zip(acc_df["id"], acc_df["name"])) if not acc_df.empty and "id" in acc_df.columns else {}
        # Build category map: start from categories parquet, then overlay inline names from full data
        category_map = dict(zip(cat_df["id"], cat_df["name"])) if not cat_df.empty and "id" in cat_df.columns else {}
        if "categoryId" in df_all_raw.columns and "categoryName" in df_all_raw.columns:
            category_map.update(dict(zip(df_all_raw["categoryId"].dropna(), df_all_raw["categoryName"].dropna())))
        accounts_display = acc_df

    st.info("📂 Showing data from local snapshot. Switch to **Live API** in the sidebar to fetch fresh data.")

else:
    with st.spinner("Fetching your financial data from the API…"):
        # Fetch ALL records (no date filter) so comparison feature can access any date range
        records_raw    = fetch_all_pages(token, "/records", {"limit": 200})
        accounts_raw   = api_get(token, "/accounts") or []
        categories_raw = api_get(token, "/categories") or []

    if isinstance(accounts_raw, dict):
        accounts_raw   = accounts_raw.get("accounts") or accounts_raw.get("items") or []
    if isinstance(categories_raw, dict):
        categories_raw = categories_raw.get("categories") or categories_raw.get("items") or []

    account_map  = {a["id"]: a.get("name", a["id"]) for a in accounts_raw}
    category_map = {c["id"]: c.get("name", c["id"]) for c in categories_raw}
    accounts_display = pd.DataFrame(accounts_raw)

    if not records_raw:
        st.warning("No records found.")
        st.stop()

    df = pd.DataFrame(records_raw)
    for col in ["amount", "payee", "note", "recordDate", "categoryId", "accountId", "type"]:
        if col not in df.columns:
            df[col] = None

    # Extract amount from nested baseAmount dict
    if "baseAmount" in df.columns:
        df["amount"] = df["baseAmount"].apply(
            lambda x: x["value"] if isinstance(x, dict) and "value" in x else None
        )
    df["amount"]     = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Extract category info from nested category dict
    if "category" in df.columns and df["category"].apply(lambda x: isinstance(x, dict)).any():
        df["categoryId"] = df["category"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None
        )
        df["categoryName"] = df["category"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else None
        )

    df["recordDate"] = pd.to_datetime(df["recordDate"], errors="coerce", utc=True)
    df["recordDate"] = df["recordDate"].dt.tz_localize(None)  # make tz-naive (parity with local path)

    # Keep full copy before filtering (used for period comparison)
    df_all_raw = df.copy()

    # Apply same client-side filters as local path
    mask = (
        (df["recordDate"].dt.date >= date_from) &
        (df["recordDate"].dt.date <= date_to)
    )
    if min_amount > 0:
        mask &= df["amount"].abs() >= min_amount
    if max_amount > 0:
        mask &= df["amount"].abs() <= max_amount
    if payee_filter and "payee" in df.columns:
        mask &= df["payee"].str.contains(payee_filter, case=False, na=False)
    if note_filter and "note" in df.columns:
        mask &= df["note"].str.contains(note_filter, case=False, na=False)
    df = df[mask].copy()

# ── Shared post-processing ─────────────────────────────────────────────────────
for col in ["amount", "payee", "note", "recordDate", "categoryId", "accountId", "type"]:
    if col not in df.columns:
        df[col] = None

if "categoryName" in df.columns:
    df["category"] = df["categoryName"].fillna(
        df["categoryId"].map(category_map) if "categoryId" in df.columns else "Uncategorised"
    )
else:
    df["category"] = df["categoryId"].map(category_map).fillna("Uncategorised")
df["category"] = df["category"].replace(CATEGORY_TRANSLATION)
df["account"]  = df["accountId"].map(account_map).fillna("Unknown")

# Extract label names from nested labels column
if "labels" in df.columns:
    df["label"] = df["labels"].apply(
        lambda x: ", ".join(d["name"] for d in x if isinstance(d, dict) and "name" in d)
        if isinstance(x, (list,)) or (hasattr(x, '__iter__') and not isinstance(x, str))
        else None
    )
else:
    df["label"] = None

# Collect all unique label names for the sidebar filter
if "labels" in df.columns:
    all_labels = sorted(
        {name for labels in df["labels"].dropna()
         if hasattr(labels, '__iter__')
         for d in labels
         if isinstance(d, dict) and "name" in d
         for name in [d["name"]]}
    )
else:
    all_labels = []
with st.sidebar:
    exclude_labels = exclude_labels_placeholder.multiselect(
        "Exclude labels",
        options=all_labels,
        default=[],
        help="Records with any of these labels will be removed.",
    )

# Filter out records that carry any excluded label
if exclude_labels:
    df = df[~df["label"].apply(
        lambda x: any(lbl in x for lbl in exclude_labels) if isinstance(x, str) else False
    )].copy()

# Exclude transfers – they are not real income/expenses
# But keep transfers labelled "Savings" for the By Label view
TRANSFER_CATEGORIES = ["Transfer, withdraw"]
is_transfer = df["category"].isin(TRANSFER_CATEGORIES)
has_savings_label = df["label"].str.contains(
    "Savings", case=False, na=False
)
savings_transfers = df[is_transfer & has_savings_label].copy()
df = df[~is_transfer].copy()

# Build df_all: full dataset with same post-processing (for period comparison)
df_all = df_all_raw.copy()
if "categoryName" in df_all.columns:
    df_all["category"] = df_all["categoryName"].fillna(
        df_all["categoryId"].map(category_map) if "categoryId" in df_all.columns else "Uncategorised"
    )
else:
    df_all["category"] = df_all["categoryId"].map(category_map).fillna("Uncategorised")
df_all["category"] = df_all["category"].replace(CATEGORY_TRANSLATION)
df_all["account"] = df_all["accountId"].map(account_map).fillna("Unknown")
if "labels" in df_all.columns:
    df_all["label"] = df_all["labels"].apply(
        lambda x: ", ".join(d["name"] for d in x if isinstance(d, dict) and "name" in d)
        if isinstance(x, (list,)) or (hasattr(x, '__iter__') and not isinstance(x, str))
        else None
    )
else:
    df_all["label"] = None
if exclude_labels:
    df_all = df_all[~df_all["label"].apply(
        lambda x: any(lbl in x for lbl in exclude_labels) if isinstance(x, str) else False
    )].copy()
df_all = df_all[~df_all["category"].isin(TRANSFER_CATEGORIES)].copy()
df_all["day"] = df_all["recordDate"].dt.date
df_all["month"] = df_all["recordDate"].dt.to_period("M").astype(str)
df_all["weekday"] = df_all["recordDate"].dt.day_name()

if df.empty:
    st.warning("No records found for the selected filters.")
    st.stop()

# ── Derived columns ────────────────────────────────────────
df["day"] = df["recordDate"].dt.date
df["month"] = df["recordDate"].dt.to_period("M").astype(str)
df["weekday"] = df["recordDate"].dt.day_name()

expenses = df[df["amount"] < 0]["amount"].sum()
income = df[df["amount"] > 0]["amount"].sum()
net = income + expenses
num_days = max((date_to - date_from).days, 1)
avg_daily_expense = abs(expenses) / num_days
savings_rate = (net / income * 100) if income > 0 else 0

# ── Metrics row ────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "📥 Income",
    f"{CURRENCY}{income:,.2f}",
)
c2.metric(
    "📤 Expenses",
    f"{CURRENCY}{abs(expenses):,.2f}",
)
c3.metric(
    "💹 Net",
    f"{CURRENCY}{net:,.2f}",
    delta=f"{CURRENCY}{net:,.2f}",
)
c4.metric(
    "📊 Avg Daily Spend",
    f"{CURRENCY}{avg_daily_expense:,.2f}",
)
c5.metric(
    "🎯 Savings Rate",
    f"{savings_rate:.1f}%",
    help="Net / Income × 100",
)

# Second metrics row
c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("🔢 Transactions", len(df))
biggest_expense = df[df["amount"] < 0]["amount"].min()
c7.metric(
    "💸 Biggest Expense",
    f"{CURRENCY}{abs(biggest_expense):,.2f}"
    if pd.notna(biggest_expense) else "—",
)
top_cat = (
    df[df["amount"] < 0]
    .groupby("category")["amount"]
    .sum()
    .abs()
    .idxmax()
) if not df[df["amount"] < 0].empty else "—"
c8.metric("🏷️ Top Category", top_cat)
c9.metric("📅 Period", f"{num_days} days")
unique_payees = df["payee"].dropna().nunique()
c10.metric("👤 Unique Payees", unique_payees)

st.divider()

# ── Charts ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📅 Over Time",
    "🏷️ By Category",
    "🏦 By Account",
    "📊 Monthly Comparison",
    "🔍 Spending Patterns",
    "📋 Transactions",
    "🏷️ By Label",
])

# ── Tab 1: Over Time ──────────────────────────────────────
with tab1:
    period = st.radio(
        "Aggregate by",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="time_period",
    )
    if period == "Daily":
        agg = (
            df.groupby("day")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"day": "Date", "amount": "Net"})
        )
    elif period == "Weekly":
        df["week"] = (
            df["recordDate"]
            .dt.to_period("W")
            .apply(lambda p: p.start_time.date())
        )
        agg = (
            df.groupby("week")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"week": "Date", "amount": "Net"})
        )
    else:
        df["month_date"] = (
            df["recordDate"]
            .dt.to_period("M")
            .apply(lambda p: p.start_time.date())
        )
        agg = (
            df.groupby("month_date")["amount"]
            .sum()
            .reset_index()
            .rename(
                columns={"month_date": "Date", "amount": "Net"}
            )
        )

    agg["Cumulative"] = agg["Net"].cumsum()

    # Bar chart for net flow
    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        x=agg["Date"], y=agg["Net"],
        name="Net Flow",
        marker_color=[
            "#2ecc71" if v >= 0 else "#e74c3c"
            for v in agg["Net"]
        ],
    ))
    fig_time.add_trace(go.Scatter(
        x=agg["Date"], y=agg["Cumulative"],
        name="Cumulative",
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        yaxis="y2",
    ))
    fig_time.update_layout(
        title=f"{period} Net Cash Flow & Cumulative Balance",
        yaxis=dict(title="Net Amount"),
        yaxis2=dict(
            title="Cumulative",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Income vs Expenses breakdown
    if period == "Daily":
        grp_col = "day"
        grp_data = df.copy()
    elif period == "Weekly":
        grp_col = "week"
        grp_data = df.copy()
    else:
        grp_col = "month_date"
        grp_data = df.copy()

    inc_exp = grp_data.assign(
        Income=grp_data["amount"].clip(lower=0),
        Expenses=grp_data["amount"].clip(upper=0).abs(),
    ).groupby(grp_col)[["Income", "Expenses"]].sum().reset_index()
    inc_exp.rename(columns={grp_col: "Date"}, inplace=True)

    fig_ie = go.Figure()
    fig_ie.add_trace(go.Bar(
        x=inc_exp["Date"], y=inc_exp["Income"],
        name="Income", marker_color="#2ecc71",
    ))
    fig_ie.add_trace(go.Bar(
        x=inc_exp["Date"], y=inc_exp["Expenses"],
        name="Expenses", marker_color="#e74c3c",
    ))
    fig_ie.update_layout(
        title=f"{period} Income vs Expenses",
        barmode="group",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_ie, use_container_width=True)


# ── Tab 2: By Category ────────────────────────────────────
with tab2:
    cat_view = st.radio(
        "Show",
        ["Expenses", "Income", "Both"],
        horizontal=True,
        key="cat_view",
    )

    if cat_view == "Expenses":
        cat_data = df[df["amount"] < 0].copy()
        cat_data["abs_amount"] = cat_data["amount"].abs()
    elif cat_view == "Income":
        cat_data = df[df["amount"] > 0].copy()
        cat_data["abs_amount"] = cat_data["amount"]
    else:
        cat_data = df.copy()
        cat_data["abs_amount"] = cat_data["amount"].abs()

    if cat_data.empty:
        st.info("No records for this selection.")
    else:
        cat_agg = (
            cat_data.groupby("category")["abs_amount"]
            .sum()
            .reset_index()
            .rename(columns={"abs_amount": "Amount"})
            .sort_values("Amount", ascending=False)
        )

        # Add percentage labels for the legend
        total = cat_agg["Amount"].sum()
        cat_agg["legend_label"] = cat_agg.apply(
            lambda r: f"{r['category']} ({r['Amount'] / total * 100:.1f}%)" if total else r["category"],
            axis=1,
        )

        col_chart, col_pie = st.columns([1, 1])

        with col_pie:
            # Top 4 categories + "Others" grouping the rest
            top4 = cat_agg.head(4).copy()
            others_amount = cat_agg.iloc[4:]["Amount"].sum() if len(cat_agg) > 4 else 0

            if others_amount > 0:
                others_pct = others_amount / total * 100 if total else 0
                others_row = pd.DataFrame([{
                    "category": "Others",
                    "Amount": others_amount,
                    "legend_label": f"Others ({others_pct:.1f}%)",
                }])
                pie_data = pd.concat([top4, others_row], ignore_index=True)
            else:
                pie_data = top4.copy()

            fig_pie = px.pie(
                pie_data, names="legend_label", values="Amount",
                title=f"{cat_view} by Category",
                hole=0.4,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+text",
                text=pie_data["category"].tolist(),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart:
            top_n = min(10, len(cat_agg))
            fig_bar = px.bar(
                cat_agg.head(top_n),
                x="Amount",
                y="category",
                orientation="h",
                title=f"Top {top_n} Categories",
                color="Amount",
                color_continuous_scale="Reds"
                if cat_view == "Expenses"
                else "Greens",
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            cat_agg.style.format(
                {"Amount": f"{CURRENCY}{{:,.2f}}"}
            ),
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 3: By Account ─────────────────────────────────────
with tab3:
    acc_summary = (
        df.groupby("account")
        .agg(
            Income=("amount", lambda x: x[x > 0].sum()),
            Expenses=("amount", lambda x: x[x < 0].sum()),
            Net=("amount", "sum"),
            Transactions=("amount", "count"),
        )
        .reset_index()
        .sort_values("Net")
    )
    acc_summary["Expenses"] = acc_summary["Expenses"].abs()

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        y=acc_summary["account"],
        x=acc_summary["Income"],
        name="Income",
        orientation="h",
        marker_color="#2ecc71",
    ))
    fig_acc.add_trace(go.Bar(
        y=acc_summary["account"],
        x=-acc_summary["Expenses"],
        name="Expenses",
        orientation="h",
        marker_color="#e74c3c",
    ))
    fig_acc.update_layout(
        title="Income & Expenses by Account",
        barmode="relative",
        xaxis_title="Amount",
        hovermode="y unified",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.dataframe(
        acc_summary.style.format({
            "Income": f"{CURRENCY}{{:,.2f}}",
            "Expenses": f"{CURRENCY}{{:,.2f}}",
            "Net": f"{CURRENCY}{{:,.2f}}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Account drill-down ─────────────────────────────────
    st.divider()
    st.subheader("🔍 Account Drill-Down")
    account_names = sorted(df["account"].dropna().unique().tolist())
    selected_account = st.selectbox(
        "Select an account", options=account_names, key="acc_drilldown",
    )

    if selected_account:
        acc_df_drill = df[df["account"] == selected_account]
        acc_inc = acc_df_drill[acc_df_drill["amount"] > 0]["amount"].sum()
        acc_exp = acc_df_drill[acc_df_drill["amount"] < 0]["amount"].sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Income", f"{CURRENCY}{acc_inc:,.2f}")
        m2.metric("Expenses", f"{CURRENCY}{abs(acc_exp):,.2f}")
        m3.metric("Net", f"{CURRENCY}{(acc_inc + acc_exp):,.2f}")

        # Category breakdown for this account
        acc_cat = (
            acc_df_drill[acc_df_drill["amount"] < 0]
            .groupby("category")["amount"].sum().abs()
            .reset_index().rename(columns={"amount": "Spent"})
            .sort_values("Spent", ascending=False)
        )
        if not acc_cat.empty:
            ad_col1, ad_col2 = st.columns(2)
            with ad_col1:
                fig_acc_cat = px.pie(
                    acc_cat, names="category", values="Spent",
                    title=f"Expense Categories – {selected_account}",
                    hole=0.4,
                )
                fig_acc_cat.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_acc_cat, use_container_width=True)
            with ad_col2:
                st.dataframe(
                    acc_cat.style.format({"Spent": f"{CURRENCY}{{:,.2f}}"}),
                    use_container_width=True, hide_index=True,
                )

        # Recent transactions for this account
        st.markdown(f"**Recent transactions – {selected_account}**")
        acc_tx_cols = [c for c in ["recordDate", "payee", "category", "amount", "label", "note"] if c in acc_df_drill.columns]
        acc_tx = (
            acc_df_drill[acc_tx_cols]
            .sort_values("recordDate", ascending=False)
            .head(20)
            .rename(columns={
                "recordDate": "Date", "payee": "Payee", "category": "Category",
                "amount": "Amount", "label": "Labels", "note": "Note",
            })
        )
        st.dataframe(
            acc_tx.style.format({
                "Amount": f"{CURRENCY}{{:,.2f}}",
                "Date": lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "",
            }),
            use_container_width=True, hide_index=True,
        )


# ── Tab 4: Monthly Comparison ─────────────────────────────
with tab4:
    monthly = (
        df.assign(
            Income=df["amount"].clip(lower=0),
            Expenses=df["amount"].clip(upper=0).abs(),
        )
        .groupby("month")
        .agg(
            Income=("Income", "sum"),
            Expenses=("Expenses", "sum"),
            Net=("amount", "sum"),
            Transactions=("amount", "count"),
        )
        .reset_index()
        .rename(columns={"month": "Month"})
    )

    # ── Custom Period Comparison ───────────────────────────
    st.subheader("Compare with Another Period")
    st.caption(
        f"**Current period** is your sidebar filter: "
        f"**{date_from.strftime('%d %b %Y')}** – **{date_to.strftime('%d %b %Y')}**. "
        f"Pick a comparison period below."
    )

    all_min = df_all["day"].min()
    all_max = df_all["day"].max()

    # Smart comparison defaults based on quick_range selection
    if quick_range == "Default":
        # Previous 26th-to-25th cycle
        # date_from is the current cycle's 26th
        # Compare: one month before that 26th → day before date_from (25th)
        default_cmp_to = date_from - timedelta(days=1)  # the 25th before current start
        # Go back one month from date_from for the compare start
        if date_from.month == 1:
            default_cmp_from = date(date_from.year - 1, 12, 26)
        else:
            default_cmp_from = date(date_from.year, date_from.month - 1, 26)
    elif quick_range == "Today":
        # Compare with yesterday
        default_cmp_from = today - timedelta(days=1)
        default_cmp_to = today - timedelta(days=1)
    elif quick_range == "This Week":
        # Compare with last week (Sun-Sat)
        this_sunday = today - timedelta(days=(today.weekday() + 1) % 7)
        last_saturday = this_sunday - timedelta(days=1)
        last_sunday = last_saturday - timedelta(days=6)
        default_cmp_from = last_sunday
        default_cmp_to = last_saturday
    elif quick_range == "This Month":
        # Compare with last month (1st to last day)
        first_this_month = today.replace(day=1)
        last_day_prev = first_this_month - timedelta(days=1)
        default_cmp_from = last_day_prev.replace(day=1)
        default_cmp_to = last_day_prev
    elif quick_range == "6 Months":
        # Compare with the previous 6 months
        current_start = (today.replace(day=1) - timedelta(days=180)).replace(day=1)
        default_cmp_to = current_start - timedelta(days=1)
        default_cmp_from = (default_cmp_to - timedelta(days=180)).replace(day=1)
    elif quick_range == "This Year":
        # Compare with last year
        default_cmp_from = date(today.year - 1, 1, 1)
        default_cmp_to = date(today.year - 1, 12, 31)
    else:  # Custom – mirror same length before
        current_days = (date_to - date_from).days
        default_cmp_to = date_from - timedelta(days=1)
        default_cmp_from = default_cmp_to - timedelta(days=current_days)
    # Clamp to available data range
    default_cmp_from = max(default_cmp_from, all_min)
    default_cmp_to = max(default_cmp_to, all_min)

    # Force-update comparison dates when quick_range changes
    if st.session_state.get("_last_quick_range") != quick_range:
        st.session_state["_last_quick_range"] = quick_range
        st.session_state["cmp_from"] = default_cmp_from
        st.session_state["cmp_to"] = default_cmp_to

    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        cmp_from = st.date_input(
            "Compare From",
            min_value=all_min, max_value=all_max, key="cmp_from",
        )
    with cmp_col2:
        cmp_to = st.date_input(
            "Compare To",
            min_value=all_min, max_value=all_max, key="cmp_to",
        )

    if cmp_from > cmp_to:
        st.error("'Compare From' must be on or before 'Compare To'.")
    else:
        df_a = df  # current period (already filtered by sidebar)
        df_b = df_all[(df_all["day"] >= cmp_from) & (df_all["day"] <= cmp_to)]

        def _period_stats(period_df, start, end):
            inc = period_df[period_df["amount"] > 0]["amount"].sum()
            exp = period_df[period_df["amount"] < 0]["amount"].sum()
            days = max((end - start).days, 1)
            return {
                "Income": inc,
                "Expenses": abs(exp),
                "Net": inc + exp,
                "Avg Daily Spend": abs(exp) / days,
                "Transactions": len(period_df),
                "Days": days,
            }

        stats_a = _period_stats(df_a, date_from, date_to)
        stats_b = _period_stats(df_b, cmp_from, cmp_to)

        label_a = f"{date_from.strftime('%d %b')} – {date_to.strftime('%d %b %Y')}"
        label_b = f"{cmp_from.strftime('%d %b')} – {cmp_to.strftime('%d %b %Y')}"

        # Side-by-side metrics
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"##### Current: {label_a}")
            ma1, ma2, ma3 = st.columns(3)
            ma1.metric("Income", f"{CURRENCY}{stats_a['Income']:,.2f}")
            ma2.metric("Expenses", f"{CURRENCY}{stats_a['Expenses']:,.2f}")
            ma3.metric("Net", f"{CURRENCY}{stats_a['Net']:,.2f}")
            ma4, ma5, ma6 = st.columns(3)
            ma4.metric("Avg Daily Spend", f"{CURRENCY}{stats_a['Avg Daily Spend']:,.2f}")
            ma5.metric("Transactions", stats_a["Transactions"])
            ma6.metric("Days", stats_a["Days"])
        with m_col2:
            st.markdown(f"##### Compare: {label_b}")
            mb1, mb2, mb3 = st.columns(3)
            mb1.metric("Income", f"{CURRENCY}{stats_b['Income']:,.2f}")
            mb2.metric("Expenses", f"{CURRENCY}{stats_b['Expenses']:,.2f}")
            mb3.metric("Net", f"{CURRENCY}{stats_b['Net']:,.2f}")
            mb4, mb5, mb6 = st.columns(3)
            mb4.metric("Avg Daily Spend", f"{CURRENCY}{stats_b['Avg Daily Spend']:,.2f}")
            mb5.metric("Transactions", stats_b["Transactions"])
            mb6.metric("Days", stats_b["Days"])

        # Deltas row – color-coded cards
        st.markdown("---")
        st.markdown("##### Change (Current vs Compare)")

        inc_delta = stats_a["Income"] - stats_b["Income"]
        exp_delta = stats_a["Expenses"] - stats_b["Expenses"]
        net_delta = stats_a["Net"] - stats_b["Net"]
        avg_delta = stats_a["Avg Daily Spend"] - stats_b["Avg Daily Spend"]

        inc_pct = (inc_delta / stats_b["Income"] * 100) if stats_b["Income"] else 0
        exp_pct = (exp_delta / stats_b["Expenses"] * 100) if stats_b["Expenses"] else 0
        net_pct = (net_delta / abs(stats_b["Net"]) * 100) if stats_b["Net"] else 0
        avg_pct = (avg_delta / stats_b["Avg Daily Spend"] * 100) if stats_b["Avg Daily Spend"] else 0

        def _delta_card(label, delta, pct, positive_is_good=True):
            """Return styled HTML for a delta card."""
            is_good = (delta > 0) if positive_is_good else (delta < 0)
            is_neutral = delta == 0
            if is_neutral:
                bg, border, text_color = "#f8f9fa", "#dee2e6", "#6c757d"
                arrow = ""
            elif is_good:
                bg, border, text_color = "#d4edda", "#28a745", "#155724"
                arrow = "▲"
            else:
                bg, border, text_color = "#f8d7da", "#dc3545", "#721c24"
                arrow = "▼"
            sign = "+" if delta > 0 else ""
            return f"""
            <div style="background:{bg}; border:2px solid {border}; border-radius:10px;
                        padding:16px; text-align:center;">
                <div style="font-size:0.85rem; color:#555; margin-bottom:4px;">{label}</div>
                <div style="font-size:1.5rem; font-weight:700; color:{text_color};">
                    {arrow} {CURRENCY}{sign}{delta:,.2f}
                </div>
                <div style="font-size:0.95rem; color:{text_color}; margin-top:4px;">
                    {sign}{pct:.1f}%
                </div>
            </div>
            """

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(_delta_card("Income", inc_delta, inc_pct, positive_is_good=True), unsafe_allow_html=True)
        with d2:
            st.markdown(_delta_card("Expenses", exp_delta, exp_pct, positive_is_good=False), unsafe_allow_html=True)
        with d3:
            st.markdown(_delta_card("Net", net_delta, net_pct, positive_is_good=True), unsafe_allow_html=True)
        with d4:
            st.markdown(_delta_card("Avg Daily Spend", avg_delta, avg_pct, positive_is_good=False), unsafe_allow_html=True)

        # Category comparison bar chart
        st.markdown("#### Category Breakdown Comparison")
        cat_a = (
            df_a[df_a["amount"] < 0]
            .groupby("category")["amount"].sum().abs()
            .reset_index().rename(columns={"amount": "Current"})
        )
        cat_b = (
            df_b[df_b["amount"] < 0]
            .groupby("category")["amount"].sum().abs()
            .reset_index().rename(columns={"amount": "Compare"})
        )
        cat_cmp = pd.merge(cat_a, cat_b, on="category", how="outer").fillna(0)
        cat_cmp = cat_cmp.sort_values("Current", ascending=False)

        if not cat_cmp.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=cat_cmp["category"], y=cat_cmp["Current"],
                name=label_a, marker_color="#3498db",
            ))
            fig_cmp.add_trace(go.Bar(
                x=cat_cmp["category"], y=cat_cmp["Compare"],
                name=label_b, marker_color="#e67e22",
            ))
            fig_cmp.update_layout(
                barmode="group",
                title="Expenses by Category – Current vs Compare",
                xaxis_title="Category",
                yaxis_title=f"Amount ({CURRENCY})",
                legend=dict(orientation="h", y=1.12),
                hovermode="x unified",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Summary table
            cat_cmp["Δ"] = cat_cmp["Current"] - cat_cmp["Compare"]
            cat_cmp["Δ %"] = np.where(
                cat_cmp["Compare"] > 0,
                (cat_cmp["Δ"] / cat_cmp["Compare"] * 100).round(1),
                np.nan,
            )
            st.dataframe(
                cat_cmp.style.format({
                    "Current": f"{CURRENCY}{{:,.2f}}",
                    "Compare": f"{CURRENCY}{{:,.2f}}",
                    "Δ": f"{CURRENCY}{{:+,.2f}}",
                    "Δ %": "{:+.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No expenses in either period to compare.")

    # ── Monthly Income vs Expenses ─────────────────────────
    st.divider()
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly["Month"], y=monthly["Income"],
        name="Income", marker_color="#2ecc71",
    ))
    fig_monthly.add_trace(go.Bar(
        x=monthly["Month"], y=monthly["Expenses"],
        name="Expenses", marker_color="#e74c3c",
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["Net"],
        name="Net", mode="lines+markers",
        line=dict(color="#3498db", width=3),
    ))
    fig_monthly.update_layout(
        title="Monthly Income vs Expenses",
        barmode="group",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Category trend over months
    st.subheader("Category Trend Over Time")
    expense_cats = (
        df[df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .abs()
        .nlargest(8)
        .index.tolist()
    )
    if expense_cats:
        cat_monthly = (
            df[df["category"].isin(expense_cats)]
            .groupby(["month", "category"])["amount"]
            .sum()
            .abs()
            .reset_index()
        )
        fig_cat_trend = px.line(
            cat_monthly,
            x="month",
            y="amount",
            color="category",
            markers=True,
            title="Top Categories Over Time",
        )
        fig_cat_trend.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount",
            hovermode="x unified",
        )
        st.plotly_chart(fig_cat_trend, use_container_width=True)

    st.dataframe(
        monthly.style.format({
            "Income": f"{CURRENCY}{{:,.2f}}",
            "Expenses": f"{CURRENCY}{{:,.2f}}",
            "Net": f"{CURRENCY}{{:,.2f}}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 5: Spending Patterns ──────────────────────────────
with tab5:
    expense_df = df[df["amount"] < 0].copy()

    if expense_df.empty:
        st.info("No expenses in the selected period.")
    else:
        p_col1, p_col2 = st.columns(2)

        # Spending by day of week
        with p_col1:
            dow_order = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday",
            ]
            dow_spend = (
                expense_df.groupby("weekday")["amount"]
                .sum()
                .abs()
                .reindex(dow_order)
                .fillna(0)
                .reset_index()
                .rename(columns={
                    "weekday": "Day",
                    "amount": "Spent",
                })
            )
            fig_dow = px.bar(
                dow_spend, x="Day", y="Spent",
                title="Spending by Day of Week",
                color="Spent",
                color_continuous_scale="Reds",
            )
            fig_dow.update_layout(showlegend=False)
            st.plotly_chart(fig_dow, use_container_width=True)

        # Spending by hour of day
        with p_col2:
            expense_df["hour"] = (
                expense_df["recordDate"].dt.hour
            )
            hour_spend = (
                expense_df.groupby("hour")["amount"]
                .sum()
                .abs()
                .reindex(range(24), fill_value=0)
                .reset_index()
                .rename(columns={
                    "hour": "Hour",
                    "amount": "Spent",
                })
            )
            fig_hour = px.bar(
                hour_spend, x="Hour", y="Spent",
                title="Spending by Hour of Day",
                color="Spent",
                color_continuous_scale="Reds",
            )
            fig_hour.update_layout(showlegend=False)
            st.plotly_chart(fig_hour, use_container_width=True)

        # Top 10 biggest expenses
        st.subheader("💸 Top 10 Biggest Expenses")
        top_expenses = (
            expense_df.nsmallest(10, "amount")[
                ["recordDate", "payee", "category",
                 "account", "amount", "note"]
            ]
            .copy()
        )
        top_expenses["amount"] = top_expenses["amount"].abs()
        top_expenses = top_expenses.rename(columns={
            "recordDate": "Date",
            "payee": "Payee",
            "category": "Category",
            "account": "Account",
            "amount": "Amount",
            "note": "Note",
        })
        st.dataframe(
            top_expenses.style.format({
                "Amount": f"{CURRENCY}{{:,.2f}}",
                "Date": lambda d: (
                    d.strftime("%Y-%m-%d")
                    if pd.notna(d) else ""
                ),
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Top recurring payees
        st.subheader("🔄 Top Recurring Payees")
        payee_agg = (
            expense_df[expense_df["payee"].notna()]
            .groupby("payee")
            .agg(
                Total=("amount", lambda x: x.sum()),
                Count=("amount", "count"),
                Avg=("amount", "mean"),
            )
            .reset_index()
        )
        payee_agg["Total"] = payee_agg["Total"].abs()
        payee_agg["Avg"] = payee_agg["Avg"].abs()
        payee_agg = payee_agg.sort_values(
            "Total", ascending=False
        ).head(15)
        st.dataframe(
            payee_agg.rename(columns={
                "payee": "Payee",
            }).style.format({
                "Total": f"{CURRENCY}{{:,.2f}}",
                "Avg": f"{CURRENCY}{{:,.2f}}",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 6: Transactions ───────────────────────────────────
with tab6:
    display_cols = [
        "recordDate", "payee", "category",
        "account", "amount", "label", "note",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    show_df = (
        df[display_cols]
        .sort_values("recordDate", ascending=False)
        .rename(columns={
            "recordDate": "Date",
            "payee": "Payee",
            "category": "Category",
            "account": "Account",
            "amount": "Amount",
            "label": "Labels",
            "note": "Note",
        })
    )

    # Quick search within transactions
    tx_search = st.text_input(
        "🔍 Search transactions",
        placeholder="Type to filter by any column...",
        key="tx_search",
    )
    if tx_search:
        mask = show_df.astype(str).apply(
            lambda row: row.str.contains(
                tx_search, case=False, na=False
            ).any(),
            axis=1,
        )
        show_df = show_df[mask]

    st.caption(f"Showing {len(show_df)} transactions")
    st.dataframe(
        show_df.style.format({
            "Amount": f"{CURRENCY}{{:,.2f}}",
            "Date": lambda d: (
                d.strftime("%Y-%m-%d")
                if pd.notna(d) else ""
            ),
        }),
        use_container_width=True,
        hide_index=True,
    )

    csv = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export as CSV",
        data=csv,
        file_name=f"budgetbakers_{date_from}_{date_to}.csv",
        mime="text/csv",
    )


# ── Tab 7: By Label ───────────────────────────────────────
with tab7:
    # Include Savings-labelled transfers in the label view
    label_df = pd.concat(
        [df, savings_transfers], ignore_index=True
    )
    label_df = label_df[
        label_df["label"].notna() & (label_df["label"] != "")
    ].copy()

    if label_df.empty:
        st.info("No labelled records in this period.")
    else:
        label_df = label_df.assign(
            label=label_df["label"].str.split(", ")
        ).explode("label")

        label_spend = (
            label_df[label_df["amount"] < 0]
            .groupby("label")["amount"]
            .sum()
            .abs()
            .reset_index()
            .rename(columns={"amount": "Spent"})
            .sort_values("Spent", ascending=False)
        )

        if label_spend.empty:
            st.info("No expenses with labels in this period.")
        else:
            fig5 = px.pie(
                label_spend,
                names="label",
                values="Spent",
                title="Spending by Label",
                hole=0.4,
            )
            fig5.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )
            st.plotly_chart(fig5, use_container_width=True)

            st.dataframe(
                label_spend.style.format(
                    {"Spent": f"{CURRENCY}{{:,.2f}}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        # ── Drill-down by label ────────────────────────────
        st.divider()
        st.subheader("🔍 Label drill-down")
        unique_labels = (
            label_df["label"].dropna().unique().tolist()
        )
        selected_label = st.selectbox(
            "Select a label to drill down",
            options=sorted(unique_labels),
        )

        if selected_label:
            drill = label_df[
                label_df["label"] == selected_label
            ]
            drill_expenses = drill[
                drill["amount"] < 0
            ].copy()

            if drill_expenses.empty:
                st.info(
                    f"No expenses for label "
                    f"**{selected_label}**."
                )
            else:
                by_cat = (
                    drill_expenses
                    .groupby("category")["amount"]
                    .sum()
                    .abs()
                    .reset_index()
                    .rename(columns={"amount": "Spent"})
                    .sort_values("Spent", ascending=False)
                )

                d_col1, d_col2 = st.columns([1, 1])
                with d_col1:
                    fig6 = px.pie(
                        by_cat,
                        names="category",
                        values="Spent",
                        title=(
                            f"'{selected_label}' "
                            f"by Category"
                        ),
                        hole=0.4,
                    )
                    fig6.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                    )
                    st.plotly_chart(
                        fig6, use_container_width=True
                    )

                with d_col2:
                    st.dataframe(
                        by_cat.style.format(
                            {"Spent": f"{CURRENCY}{{:,.2f}}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown(
                    f"**Transactions for "
                    f"'{selected_label}'**"
                )
                tx_cols = [
                    "recordDate", "payee", "category",
                    "account", "amount", "note",
                ]
                tx_cols = [
                    c for c in tx_cols
                    if c in drill_expenses.columns
                ]
                tx_df = (
                    drill_expenses[tx_cols]
                    .sort_values(
                        "recordDate", ascending=False
                    )
                    .rename(columns={
                        "recordDate": "Date",
                        "payee": "Payee",
                        "category": "Category",
                        "account": "Account",
                        "amount": "Amount",
                        "note": "Note",
                    })
                )
                st.dataframe(
                    tx_df.style.format({
                        "Amount": f"{CURRENCY}{{:,.2f}}",
                        "Date": lambda d: (
                            d.strftime("%Y-%m-%d")
                            if pd.notna(d) else ""
                        ),
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

