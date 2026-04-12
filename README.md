# Earnings Call Investment Scorer

**NYU · Text as Data · Spring 2026**

Replicates and extends [Jha, Qian, Weber & Yang (2024) — "ChatGPT and Corporate Policies"](https://www.nber.org/papers/w32161) (NBER WP 32161) using a large-language model to score S&P 500 earnings call transcripts for expected capital investment, then validates those scores against realized Compustat capex and CRSP stock returns.

---

## What this does

1. **Score** — feeds earnings call transcripts into an LLM and returns a continuous investment score (−1 = strong disinvestment signal, +1 = strong investment signal), separately for the management presentation and Q&A sections.
2. **Validate** — merges scores with Compustat quarterly fundamentals, CRSP monthly returns, Fama-French 5 factors, and I/B/E/S analyst forecasts to run panel regressions replicating Tables 3–11 of the paper.
3. **Demo** — Streamlit web app where you paste any earnings call transcript and get a score back in real time.

## Improvements over the paper

| | Paper (Jha et al. 2024) | This project |
|---|---|---|
| Model | GPT-3.5 | Claude (200k context) |
| Chunking | Yes — transcripts split into pieces | No — full call in one prompt |
| Temperature | Unspecified | Explicitly 0 (reproducible) |
| Scoring granularity | Single score | Management vs. Q&A breakdown |
| Name masking | Robustness check only | Primary approach |
| Returns | Through 2020 | Extended to 2025 |
| CFO Survey validation | Duke CFO Survey (private) | I/B/E/S capex analyst forecasts |

## Data sources

| Dataset | Source | Notes |
|---|---|---|
| Earnings call transcripts | HuggingFace `kurry/sp500_earnings_transcripts` | 33,362 calls, 685 companies, 2005–2025 |
| Compustat Quarterly | WRDS `comp.fundq` | capex, assets, R&D, SG&A, cash flows |
| CRSP Monthly | WRDS `crsp.msf` | raw returns, prices, shares |
| CCM Link | WRDS `crsp.ccmxpf_lnkhist` | gvkey ↔ permno mapping |
| Fama-French 5 Factors | WRDS `ff.factors_monthly` | FF5 + momentum (Carhart) |
| I/B/E/S Consensus | WRDS `ibes.statsum_epsus` | analyst capex & EPS forecasts |

WRDS access required for financial data. Transcripts download automatically from HuggingFace.

## Project structure

```
.
├── data/
│   ├── config.py                  # all dataset vars and paths — no hardcoding elsewhere
│   ├── download_transcripts.py    # pulls HuggingFace dataset → transcripts.parquet
│   └── download_financials.py     # pulls all WRDS tables → financials/*.parquet
├── src/
│   ├── masker.py                  # strips company/exec names before LLM scoring
│   └── scorer.py                  # LLM scoring pipeline — returns [-1, 1] score
├── analysis/
│   └── validate.py                # merges scores with financials, runs regressions
├── app/
│   └── app.py                     # Streamlit demo UI
├── notebooks/
│   ├── eda.ipynb                  # exploratory data analysis
│   └── llama3_instruct_test.ipynb # LLM prompt development
├── results/                       # regression tables, figures (git-ignored)
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-username>/earnings-scorer.git
cd earnings-scorer

conda create -n earnings-scorer python=3.12 -y
conda activate earnings-scorer
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# edit .env and fill in your WRDS username and LLM API key
```

### 3. Download data

```bash
# Transcripts (no credentials needed)
python data/download_transcripts.py

# Financial data (requires WRDS access)
python data/download_financials.py
```

### 4. Score transcripts

```bash
python src/scorer.py   # (in progress)
```

### 5. Run validation

```bash
python analysis/validate.py   # (in progress)
```

### 6. Launch demo

```bash
streamlit run app/app.py      # (in progress)
```

## Status

| Component | Status |
|---|---|
| Transcript download | Done |
| Financial data download | In progress |
| LLM scoring pipeline | Planned |
| Regression validation | Planned |
| Streamlit demo | Planned |

## Reference

Jha, M., Qian, J., Weber, M., & Yang, B. (2024). *ChatGPT and Corporate Policies*. NBER Working Paper 32161.
