"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the LLM Investment Score project.

Replicates and extends:
    Jha, Qian, Weber, Yang (2024) — "ChatGPT and Corporate Policies"
    NBER Working Paper 32161

All parameters live here. No hardcoding anywhere else.
─────────────────────────────────────────────────────────────────────────────
"""

import os

# ── Transcript Dataset ────────────────────────────────────────────────────────
# Pinned commit hash guarantees identical data across machines and time.
# To update: change the hash, document why in git commit message.

TRANSCRIPT_DATASET          = "kurry/sp500_earnings_transcripts"
TRANSCRIPT_DATASET_REVISION = "f3ded372da8d18dc6ad98955c4558e34b5fe6d45"

# ── Sample Period ─────────────────────────────────────────────────────────────
# Paper covers 2006–2020. We extend to 2025 as a contribution.

START_DATE = "2005-01-01"
END_DATE   = "2025-12-31"

# ── Compustat Quarterly Variables ─────────────────────────────────────────────
# Source: comp.fundq joined with comp.company (sic only).
# Definitions follow Peters and Taylor (2017) and Jha et al. (2024) Appendix A.
#
# Note on capsq: this is the direct quarterly SCF capex figure (97.7% populated
# for S&P 500). It is NOT the same as capxy (year-to-date cumulative) — no
# differencing required.
#
# Note on xsgaq: stored raw. Downstream code applies the 0.3 multiplier when
# computing Intangible Investment = (xrdq + 0.3 × xsgaq) / Total Capital.
#
# Note on ibq + dpq: both required for Total Cash Flow (Peters & Taylor 2017):
#   Total Cash Flow = (ibq + dpq + (1 - tax_rate) × intangible_inv) / Total Capital

COMPUSTAT_VARS = {
    # ── Identifiers ──────────────────────────────────────────────────────────
    "gvkey":    "Compustat permanent firm identifier",
    "tic":      "ticker symbol",
    "conm":     "company name",
    "datadate": "fiscal quarter end date",
    "fyearq":   "fiscal year",
    "fqtr":     "fiscal quarter (1–4)",

    # ── Investment outcomes  [Table 1, 3, 4, 5, 6] ───────────────────────────
    "capsq":    "capital expenditures — quarterly SCF (direct, not year-to-date)",
    "atq":      "total book assets",
    "ppentq":   "net PP&E — Physical Capital stock",
    "xrdq":     "R&D expenses — Intangible Capital component",
    "xsgaq":    "SG&A expenses — Intangible Capital component (× 0.3 in computations)",

    # ── Market value  →  Total q = (mktcap + debt) / Total Capital ───────────
    "cshoq":    "common shares outstanding",
    "prccq":    "stock price at fiscal quarter end",

    # ── Leverage  [Table 3, 4, 5, 6, 8] ─────────────────────────────────────
    "dlttq":    "long-term debt",
    "dlcq":     "current portion of long-term debt",

    # ── Cash flow components  [Total Cash Flow control — Table 3, 4, 5, 6] ───
    # Total Cash Flow = (ibq + dpq + (1 − 0.35) × intangible_inv) / Total Capital
    "ibq":      "income before extraordinary items (quarterly)",
    "dpq":      "depreciation and amortization (quarterly)",

    # ── Additional controls  [Table 14 robustness] ───────────────────────────
    "saleq":    "net sales",
    "epspxq":   "EPS excluding extraordinary items (base for earnings surprise)",
    # sic: pulled from comp.company via LEFT JOIN — not a fundq column
}

# Standard Compustat sample filters (paper Section 2.1)
COMPUSTAT_FILTERS = {
    "indfmt":  "INDL",   # industrial format — excludes financial-statement format
    "datafmt": "STD",    # standardized data
    "popsrc":  "D",      # domestic population source
    "consol":  "C",      # consolidated accounts only
}

# ── CRSP Monthly Variables ────────────────────────────────────────────────────
# Source: crsp.msf
# Raw returns used directly; FF5/q5-adjusted returns computed separately.

CRSP_VARS = {
    "permno":  "CRSP permanent firm identifier",
    "date":    "month end date",
    "ret":     "monthly raw return",
    "prc":     "closing price (negative = bid-ask midpoint if no trade)",
    "shrout":  "shares outstanding (thousands)",
}

# ── Fama-French 5 Factor Variables ────────────────────────────────────────────
# Source: ff.fivefactors_monthly on WRDS (Ken French data library mirror).
# Note: ff.factors_monthly only has 3 factors + momentum; FF5 is separate table.
# Used for FF5-adjusted returns (Table 8, 9) and Carhart 4-factor
# short-term event returns (Table 10, UMD momentum factor included).

FF5_VARS = {
    "date":   "month end date",
    "mktrf":  "market excess return (Mkt-RF)",
    "smb":    "small-minus-big size factor",
    "hml":    "high-minus-low value factor",
    "rmw":    "robust-minus-weak profitability factor",
    "cma":    "conservative-minus-aggressive investment factor",
    "umd":    "momentum factor (Carhart 4-factor model, Table 10)",
    "rf":     "risk-free rate (1-month T-bill)",
}

# ── I/B/E/S Analyst Forecast Variables ───────────────────────────────────────
# Sources:
#   EPS — ibes.statsum_epsus  (standard EPS summary table)
#   CPX — ibes.statsum_xepsus (extended measures; capex stored as measure = 'CPX',
#          NOT 'CPEX' — that measure does not exist in WRDS I/B/E/S)
#
# Use case (replaces paper's Table 11):
#   If LLM investment score is high AND analysts revised CPX upward in same quarter
#   → independent evidence the model extracts real investment signal from the call.
#
#   fpi = '1' → next fiscal year; fpi = '6' → next fiscal quarter.

IBES_VARS = {
    "ticker":   "I/B/E/S ticker (may differ from Compustat tic — match via CUSIP or name)",
    "fpedats":  "forecast period end date",
    "statpers": "statistical period date (consensus snapshot date)",
    "measure":  "forecast measure: CPEX = capex, EPS = earnings per share",
    "fpi":      "forecast period indicator: 1 = annual, 6 = next quarter",
    "meanest":  "consensus mean estimate",
    "medest":   "consensus median estimate",
    "numest":   "number of contributing analysts",
    "stdev":    "cross-analyst standard deviation of estimates",
}

# ── CCM Link Table Settings ───────────────────────────────────────────────────
# Maps Compustat gvkey ↔ CRSP permno with valid date ranges.
# Must filter: linkdt <= datadate <= linkenddt (NaT linkenddt = still active).
# Use primary links only — LU (researched) and LC (links with conditions).

CCM_LINK_TYPES   = ("LU", "LC")
CCM_LINK_PRIMARY = ("P", "C")

# ── Output Paths ──────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(__file__)
FINANCIALS  = os.path.join(BASE_DIR, "financials")
TRANSCRIPTS = os.path.join(BASE_DIR, "transcripts")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
