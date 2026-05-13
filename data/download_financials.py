"""
download_financials.py
─────────────────────────────────────────────────────────────────────────────
Downloads all financial data required to replicate and extend:
    Jha, Qian, Weber, Yang (2024) — "ChatGPT and Corporate Policies"
    NBER Working Paper 32161

Data sources (all via WRDS):
    1. Compustat Quarterly  →  comp.fundq + comp.company
    2. CCM Link Table       →  crsp.ccmxpf_lnkhist
    3. CRSP Monthly         →  crsp.msf
    4. Fama-French Factors  →  ff.factors_monthly
    5. I/B/E/S Forecasts    →  ibes.statsum_epsus (EPS) + ibes.statsum_xepsus (CPX)

Outputs saved to data/financials/:
    compustat_quarterly.parquet
    ccm_link.parquet
    crsp_monthly.parquet
    ff5_factors_monthly.parquet
    ibes_consensus.parquet

Usage:
    cd data/
    python download_financials.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import wrds
import pandas as pd
from dotenv import load_dotenv
from config import (
    START_DATE, END_DATE,
    COMPUSTAT_VARS, COMPUSTAT_FILTERS,
    CRSP_VARS, FF5_VARS, IBES_VARS,
    CCM_LINK_TYPES, CCM_LINK_PRIMARY,
    FINANCIALS,
)

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
WRDS_USERNAME = os.getenv("WRDS_USERNAME")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ticker_list_from_transcripts() -> str:
    """
    Read unique tickers from the transcript parquet.
    Scopes all WRDS queries to our 685 S&P 500 companies —
    avoids pulling the full Compustat / CRSP universe and keeps RAM usage low.
    """
    path = os.path.join(os.path.dirname(__file__), "transcripts", "transcripts.parquet")
    tickers = (
        pd.read_parquet(path, columns=["symbol"])["symbol"]
        .dropna()
        .unique()
        .tolist()
    )
    print(f"  {len(tickers):,} unique tickers loaded from transcript data")
    return ", ".join(f"'{t}'" for t in tickers)


def _save(df: pd.DataFrame, name: str) -> None:
    """Write DataFrame to financials/ as parquet and print confirmation."""
    path = os.path.join(FINANCIALS, f"{name}.parquet")
    df.to_parquet(path, index=False)
    print(f"  Saved  {name}.parquet  —  {len(df):,} rows")


# ── Compustat ─────────────────────────────────────────────────────────────────

def pull_compustat(db: wrds.Connection, ticker_list: str) -> pd.DataFrame:
    """
    Pull quarterly fundamentals from comp.fundq.
    SIC code joined from comp.company (not a fundq column).
    Scoped to our transcript tickers and sample period.
    atq > 0 excludes firms with zero or negative book assets.
    """
    # Build column list with table alias
    cols = ",\n".join(f"f.{col}" for col in COMPUSTAT_VARS.keys())

    # Build filter string (indfmt, datafmt, popsrc, consol)
    where_filters = "\n AND ".join(
        f"f.{col} = '{val}'" for col, val in COMPUSTAT_FILTERS.items()
    )

    query = f"""
        SELECT
            {cols},
            c.sic
          FROM
            comp.fundq AS f
            LEFT JOIN comp.company AS c USING (gvkey)
         WHERE
            f.datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND {where_filters}
          AND f.atq > 0
          AND f.tic IN ({ticker_list})
    """

    print("  Querying comp.fundq...")
    df = db.raw_sql(query, date_cols=["datadate"])
    print(f"  {len(df):,} firm-quarter obs  |  {df['gvkey'].nunique():,} unique firms")
    return df


# ── CCM Link ──────────────────────────────────────────────────────────────────

def pull_ccm_link(db: wrds.Connection, gvkey_list: str) -> pd.DataFrame:
    """
    Pull the Compustat–CRSP Merged link table.
    Maps gvkey (Compustat) ↔ permno (CRSP) with valid date ranges.
    Caller must apply: linkdt <= datadate <= linkenddt (NaT = still active).
    Scoped to gvkeys from the Compustat pull.
    """
    link_types   = ", ".join(f"'{t}'" for t in CCM_LINK_TYPES)
    link_primary = ", ".join(f"'{p}'" for p in CCM_LINK_PRIMARY)

    query = f"""
        SELECT
            gvkey,
            lpermno AS permno,
            linktype,
            linkprim,
            linkdt,
            linkenddt
        FROM
            crsp.ccmxpf_lnkhist
        WHERE
            linktype  IN ({link_types})
          AND linkprim IN ({link_primary})
          AND gvkey   IN ({gvkey_list})
    """

    print("  Querying crsp.ccmxpf_lnkhist...")
    df = db.raw_sql(query, date_cols=["linkdt", "linkenddt"])
    print(f"  {len(df):,} link records  |  {df['gvkey'].nunique():,} unique gvkeys")
    return df


# ── CRSP ──────────────────────────────────────────────────────────────────────

def pull_crsp(db: wrds.Connection, permno_list: str) -> pd.DataFrame:
    """
    Pull monthly stock returns and prices from crsp.msf.
    Scoped to permnos from the CCM link — avoids loading full CRSP universe.
    """
    cols = ",\n            ".join(CRSP_VARS.keys())

    query = f"""
        SELECT
            {cols}
        FROM
            crsp.msf
        WHERE
            date BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND permno IN ({permno_list})
    """

    print("Querying crsp.msf...")
    df = db.raw_sql(query, date_cols=["date"])
    print(f"  {len(df):,} firm-month obs  |  {df['permno'].nunique():,} unique permnos")
    return df


# ── Fama-French Factors ───────────────────────────────────────────────────────

def pull_ff5_factors(db: wrds.Connection) -> pd.DataFrame:
    """
    Pull Fama-French 5-factor + momentum monthly returns from WRDS.
    Source: ff.fivefactors_monthly (Ken French data library mirror).
    Used to compute:
        - FF5-adjusted abnormal returns (Table 8, 9)
        - Carhart 4-factor event returns (Table 10, uses UMD)
    Full history pulled — small table, no firm-level scoping needed.
    """
    cols = ",\n            ".join(FF5_VARS.keys())

    query = f"""
        SELECT
            {cols}
        FROM
            ff.fivefactors_monthly
        WHERE
            date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """

    print("Querying ff.factors_monthly...")
    df = db.raw_sql(query, date_cols=["date"])
    print(f"{len(df):,} monthly factor observations")
    return df


# ── I/B/E/S Analyst Forecasts ─────────────────────────────────────────────────

def resolve_ibes_tickers(db: wrds.Connection, cusip_list: str) -> str:
    """
    Map our Compustat CUSIPs to I/B/E/S tickers via ibes.idsum security master.
    I/B/E/S ticker often differs from market ticker — CUSIP is stable identifier.
    Returns SQL-safe quoted ticker list ready for statsum queries.
    """
    query = f"""
        SELECT DISTINCT
            ticker
        FROM
            ibes.idsum
        WHERE
            cusip IN ({cusip_list})
    """

    print("  Resolving I/B/E/S tickers via CUSIP (ibes.idsum)...")
    df = db.raw_sql(query)
    tickers = df["ticker"].dropna().unique().tolist()
    print(f"  {len(tickers):,} I/B/E/S tickers resolved from CUSIP match")
    return ", ".join(f"'{t}'" for t in tickers)


def pull_ibes(db: wrds.Connection, ticker_list: str) -> pd.DataFrame:
    """
    Pull I/B/E/S consensus capex and EPS forecasts.

    EPS  — from ibes.statsum_epsus  (standard EPS summary table)
    CPX  — from ibes.statsum_xepsus (extended measures table; 'CPEX' does not exist,
            capital expenditure forecasts are stored under measure = 'CPX')

    Our substitute for the Duke CFO Survey (Table 11 of paper):
        LLM investment score high + analysts revised CPX upward in same quarter
        → independent evidence the model extracts real investment signal.

    Filters:
        fpi IN ('1', '6') — annual and next-quarter forecast horizons
    """
    cols = ",\n".join(IBES_VARS.keys())

    # EPS from the standard table
    eps_query = f"""
        SELECT
            {cols}
          FROM
            ibes.statsum_epsus
         WHERE
            statpers   BETWEEN '{START_DATE}' AND '{END_DATE}'
            AND measure = 'EPS'
            AND fpi    IN ('1', '6')
            AND ticker IN ({ticker_list})
    """

    # capital expenditure forecasts from the extended table
    cpx_query = f"""
        SELECT
            {cols}
          FROM
            ibes.statsum_xepsus
         WHERE
            statpers   BETWEEN '{START_DATE}' AND '{END_DATE}'
            AND measure = 'CPX'
            AND fpi    IN ('1', '6')
            AND ticker IN ({ticker_list})
    """

    print("Querying ibes.statsum_epsus (EPS)...")
    eps = db.raw_sql(eps_query, date_cols=["fpedats", "statpers"])
    print(f"  {len(eps):,} EPS rows")

    print("Querying ibes.statsum_xepsus (CPX)...")
    cpx = db.raw_sql(cpx_query, date_cols=["fpedats", "statpers"])
    print(f"  {len(cpx):,} CPX rows")

    df = pd.concat([eps, cpx], ignore_index=True)
    print(f"  {len(df):,} total rows  |  {df['ticker'].nunique():,} unique tickers")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FINANCIALS, exist_ok=True)
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    # ── Step 1: universe of tickers from transcript data ──────────────────────
    print("\n[1/5]  Loading transcript tickers...")
    ticker_list = _ticker_list_from_transcripts()

    # ── Step 2: Compustat quarterly fundamentals ──────────────────────────────
    print("\n[2/5]  Compustat quarterly...")
    compustat = pull_compustat(db, ticker_list)
    _save(compustat, "compustat_quarterly")

    # ── Step 3: CCM link table (gvkey → permno) ───────────────────────────────
    print("\n[3/5]  CCM link table...")
    gvkey_list = ", ".join(f"'{g}'" for g in compustat["gvkey"].unique())
    ccm = pull_ccm_link(db, gvkey_list)
    _save(ccm, "ccm_link")

    # ── Step 4: CRSP monthly returns ──────────────────────────────────────────
    # Scoped to permnos from CCM — never pulls full CRSP universe
    print("\n[4/5]  CRSP monthly returns...")
    permno_list = ", ".join(str(int(p)) for p in ccm["permno"].dropna().unique())
    crsp = pull_crsp(db, permno_list)
    _save(crsp, "crsp_monthly")

    # ── Step 5: Fama-French factors + I/B/E/S ────────────────────────────────
    print("\n[5/5]  Fama-French factors + I/B/E/S forecasts...")
    ff5  = pull_ff5_factors(db)
    _save(ff5, "ff5_factors_monthly")

    # I/B/E/S — recover full coverage via CUSIP match (ticker drift fix)
    # Compustat cusip is 9-digit (with check digit), I/B/E/S idsum is 8-digit. Truncate.
    cusip_list  = ", ".join(f"'{c[:8]}'" for c in compustat["cusip"].dropna().unique())
    ibes_tickers = resolve_ibes_tickers(db, cusip_list)
    ibes         = pull_ibes(db, ibes_tickers)
    _save(ibes, "ibes_consensus")

    db.close()
    print("\nDone. All files saved to data/financials/")


if __name__ == "__main__":
    main()