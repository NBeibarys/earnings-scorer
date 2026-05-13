# Open-Source LLM Replication of *ChatGPT and Corporate Policies*

**NYU Data Science · Text as Data · Spring 2026**

Replication and methodological extension of [Jha, Qian, Weber, Yang (2024), *ChatGPT and Corporate Policies* (NBER WP 32161)](https://www.nber.org/papers/w32161).
The original paper extracts a capital-expenditure direction signal from S&P 500 earnings call transcripts using GPT-3.5 and a single fixed prompt. This project replaces that pipeline with an open-weight GPT-4-class model (`Qwen/Qwen3-32B` via DeepInfra) and adds a turn-preserving 1,500-word chunker, a 4-class direction scheme with evidence-aware aggregation, a 3-variant prompt-sensitivity test, and an I/B/E/S analyst capex consensus validation that substitutes for the paper's proprietary Duke CFO Survey check.

The full paper is in [`paper/paper.md`](paper/paper.md). All figures referenced there are in [`paper/figures/`](paper/figures/).

---

## Headline numbers (from the realised analytic sample)

| Quantity | Value | Source |
|---|---|---|
| Transcripts in stratified subsample | 10,000 | Section 4 of `notebooks/corpus_scoring.ipynb` |
| Transcripts fully scored (analytic sample) | 7,602 | Section 11 |
| Total chunks scored on DeepInfra | ~57,000 | scoring checkpoint |
| Prompt-robustness chunk-level agreement, V1 vs V2 | 78.6 % | Section 9 |
| Prompt-robustness chunk-level agreement, V1 vs V3 | 74.6 % | Section 9 |
| Within-firm Spearman ρ, LLM score vs post-call I/B/E/S CPX consensus | +0.060, p = 3.75e-6, n = 5,853 | Section 10b |
| Headline panel coefficient β on LLM score (capex_{t+2}, firm + year FE) | +0.0040, t = 1.17, p = 0.24, n = 7,144 | Section 12 |

The I/B/E/S external-validity test is significant at p < 10⁻⁵; the panel regression on realised capex is directionally correct but not significant at α = 0.05 in the headline specification (full discussion of this asymmetry in `paper/paper.md`).

---

## Project layout

```
project/
  data/
    config.py                       all dataset variables and paths (no hardcoding)
    download_transcripts.py         HuggingFace dataset (pinned commit hash)
    download_financials.py          all WRDS pulls (Compustat, CRSP, FF5, IBES)
    transcripts/
      transcripts.parquet           33,234 cleaned earnings call transcripts
    financials/
      compustat_quarterly.parquet
      ccm_link.parquet
      crsp_monthly.parquet
      ff5_factors_monthly.parquet
      ibes_consensus.parquet
  notebooks/
    eda.ipynb                       exploratory data analysis (full corpus)
    corpus_scoring.ipynb            full LLM scoring pipeline + validation + regression
  results/
    chunks_w1500.parquet            turn-preserving 1,500-word chunks
    transcript_scores.parquet       firm-quarter LLM scores (headline file)
    transcript_scores_with_partials.parquet     sensitivity file
    prompt_robustness_transcript_scores.parquet
    external_validation_ibes.parquet
    scoring_checkpoints/            chunk-level scores, resumable
  paper/
    paper.md                        publishable write-up
    generate_figures.py             rebuilds all paper figures from results/
    figures/                        PNG figures referenced in paper.md
  references/
    jha_qian_weber_yang_2024_nber32161.pdf
  requirements.txt
  README.md
  .env.example                      template for DeepInfra and WRDS credentials
```

## Data sources

| Dataset | Source | Notes |
|---|---|---|
| Earnings call transcripts | HuggingFace `kurry/sp500_earnings_transcripts` | Pinned to a specific commit hash; 33,234 calls across 685 firms, 2005-2024 |
| Compustat Quarterly | WRDS `comp.fundq` (joined to `comp.company` for SIC) | capex, total assets, sales, R&D, SG&A, market cap, debt |
| CRSP Monthly | WRDS `crsp.msf` | Returns and prices, used downstream for return tests |
| CCM Link | WRDS `crsp.ccmxpf_lnkhist` | gvkey ↔ permno bridge |
| Fama-French 5 Factors | WRDS `ff.fivefactors_monthly` | FF5 plus the UMD momentum factor |
| I/B/E/S Consensus | WRDS `ibes.statsum_xepsus` (`measure='CPX'`) | Analyst consensus capex forecasts; CUSIP-bridged to Compustat ticker |

WRDS access is required to refresh the financials. Transcripts download from HuggingFace without credentials.

## Pinned reproducibility constants

| Knob | Value | Where |
|---|---|---|
| Transcript dataset commit | `f3ded372...d45` | `data/config.py` |
| Sample period | 2005-01-01 to 2024-12-31 | `data/config.py` |
| Scoring model | `Qwen/Qwen3-32B` on DeepInfra | `notebooks/corpus_scoring.ipynb` Section 2 |
| Generation params | `temperature=0`, `top_p=1.0`, `seed=42`, `response_format=json_object` | Section 7 |
| Chunk cap | 1,500 words, turn-preserving with `[continues in next chunk]` markers | Section 4 |
| Stratified sample seed | 42 | Section 4 |
| Calibration seed | 7 | Section 8 |
| Robustness seed | 11 | Section 9 |

## Quickstart

### 1. Environment

```bash
git clone <this-repo-url>
cd project

conda create -n earnings-scorer python=3.12 -y
conda activate earnings-scorer
pip install -r requirements.txt
```

### 2. Credentials

```bash
cp .env.example .env
# Fill in:
#   DEEPINFRA_API_KEY=...   (sign up at https://deepinfra.com)
#   WRDS_USERNAME=...        (only needed if you re-pull financials)
```

### 3. Data

```bash
cd data
python download_transcripts.py     # HuggingFace, ~5 min, no credentials
python download_financials.py      # WRDS, ~10-15 min, needs WRDS login
cd ..
```

### 4. Reproduce the scoring pipeline

Open `notebooks/corpus_scoring.ipynb` and run cells top-to-bottom. The notebook is sectioned:

| Section | What |
|---|---|
| 1-2 | Methodology choice (DeepInfra + Qwen 3 32B), env setup |
| 3-4 | Transcript loading, turn-preserving chunker, stratified subsample |
| 5-7 | Prompt design (V1 primary + V2 alt + V3 terse), async DeepInfra client |
| 8 | Free-credit calibration (50 chunks, ~$0.05) |
| 9 | Prompt robustness on 500-transcript subsample (~$3) |
| 10 | Main run on the 10k subsample (~$15-25), I/B/E/S validation |
| 11 | Aggregation to firm-quarter score, headline + sensitivity files |
| 12 | Headline capex regression |
| 13 | Limitations and reproducibility notes |

End-to-end wall-clock on a typical DeepInfra session is roughly 1 to 2 hours, dominated by the main scoring pass; everything else is seconds-to-minutes.

### 5. Rebuild paper figures

```bash
python paper/generate_figures.py
```

Reads from `results/` and overwrites `paper/figures/*.png`.

## What this project contributes vs the original paper

| Dimension | Jha et al. (2024) | This project |
|---|---|---|
| Scoring model | GPT-3.5-Turbo (proprietary, MMLU ~70) | Qwen 3 32B Instruct (open weights, MMLU ~82) |
| Chunking | Fixed 2,500-word blocks, splits speakers mid-sentence | Turn-preserving 1,500-word, with explicit continuation markers for oversize turns |
| Direction scale | 5-class with magnitude qualifiers | 4-class direction only (cleaner inter-rater agreement) |
| Aggregation | Mean over all chunks | Evidence-aware mean (drops `insufficient_information` chunks before averaging) |
| Prompt sensitivity | Single prompt, no test | Three prompt variants, agreement reported (chunk-level and transcript-level) |
| External validation | Duke CFO Survey (proprietary) | I/B/E/S analyst CPX consensus (public, high-frequency) |
| Sample period | 2006-2020 | 2005-2024 |
| Reproducibility | Closed model, no seed disclosure | Pinned model ID, seed 42, dataset commit hash, public posted price |

## Reference

Jha, M., Qian, J., Weber, M., & Yang, B. (2024). *ChatGPT and Corporate Policies.* NBER Working Paper 32161. [PDF](references/jha_qian_weber_yang_2024_nber32161.pdf).

Peters, R. H., & Taylor, L. A. (2017). Intangible capital and the investment-q relation. *Journal of Financial Economics*, 123(2), 251-272.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139-1168.
