# Graph Report - C:\Users\nyuss\OneDrive\Documentos\Bars\Portfollio\Books\NYU\Spring 2026\Text as Data\project  (2026-04-22)

## Corpus Check
- Corpus is ~7,602 words - fits in a single context window. You may not need a graph.

## Summary
- 46 nodes · 48 edges · 16 communities detected
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Project Reference Paper|Project Reference Paper]]
- [[_COMMUNITY_LLM Scoring Pipeline|LLM Scoring Pipeline]]
- [[_COMMUNITY_Financial Validation Data|Financial Validation Data]]
- [[_COMMUNITY_Financial Validation Data|Financial Validation Data]]
- [[_COMMUNITY_Financial Validation Data|Financial Validation Data]]
- [[_COMMUNITY_LLM Scoring Pipeline|LLM Scoring Pipeline]]
- [[_COMMUNITY_Project Scaffolding|Project Scaffolding]]
- [[_COMMUNITY_Project Scaffolding|Project Scaffolding]]
- [[_COMMUNITY_Financial Validation Data|Financial Validation Data]]
- [[_COMMUNITY_Financial Validation Data|Financial Validation Data]]
- [[_COMMUNITY_Financial Validation Data|Financial Validation Data]]
- [[_COMMUNITY_Financial Validation Data|Financial Validation Data]]
- [[_COMMUNITY_Project Scaffolding|Project Scaffolding]]
- [[_COMMUNITY_Project Scaffolding|Project Scaffolding]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_LLM Scoring Pipeline|LLM Scoring Pipeline]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 8 edges
2. `LLM investment score` - 5 edges
3. `Claude scoring pipeline` - 5 edges
4. `WRDS financial data` - 5 edges
5. `Validation regressions` - 4 edges
6. `_ticker_list_from_transcripts()` - 3 edges
7. `_save()` - 3 edges
8. `pull_compustat()` - 3 edges
9. `pull_ccm_link()` - 3 edges
10. `pull_crsp()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `I/B/E/S analyst forecasts` --part_of--> `WRDS financial data`  [EXTRACTED]
  data/download_financials.py → README.md
- `Claude scoring pipeline` --generates--> `LLM investment score`  [EXTRACTED]
  src/scorer.py → README.md
- `Earnings call transcripts` --input_to--> `Claude scoring pipeline`  [EXTRACTED]
  README.md → src/scorer.py
- `Reproducibility` --constrains--> `Claude scoring pipeline`  [EXTRACTED]
  README.md → src/scorer.py
- `WRDS financial data` --input_to--> `Validation regressions`  [EXTRACTED]
  README.md → analysis/validate.py

## Communities

### Community 0 - "Project Reference Paper"
Cohesion: 0.28
Nodes (9): Capital expenditures prediction, ChatGPT and Corporate Policies, Earnings Call Investment Scorer, I/B/E/S analyst forecasts, LLM investment score, Management presentation score, Negative abnormal returns, Q&A score (+1 more)

### Community 1 - "LLM Scoring Pipeline"
Cohesion: 0.33
Nodes (6): Claude scoring pipeline, Earnings call transcripts, kurry/sp500_earnings_transcripts, Name masking, Reproducibility, Streamlit demo

### Community 2 - "Financial Validation Data"
Cohesion: 0.4
Nodes (5): Compustat quarterly fundamentals, CRSP monthly returns, Extended 2025 sample, Fama-French factors, WRDS financial data

### Community 3 - "Financial Validation Data"
Cohesion: 0.5
Nodes (3): pull_ff5_factors(), download_financials.py ─────────────────────────────────────────────────────────, Pull Fama-French 5-factor + momentum monthly returns from WRDS.     Source: ff.f

### Community 4 - "Financial Validation Data"
Cohesion: 0.67
Nodes (3): main(), pull_ibes(), Pull I/B/E/S consensus capex and EPS forecasts.      EPS  — from ibes.statsum_ep

### Community 5 - "LLM Scoring Pipeline"
Cohesion: 0.67
Nodes (1): download_transcripts.py Downloads earnings call transcripts from HuggingFace and

### Community 6 - "Project Scaffolding"
Cohesion: 1.0
Nodes (1): config.py ──────────────────────────────────────────────────────────────────────

### Community 7 - "Project Scaffolding"
Cohesion: 1.0
Nodes (2): Write DataFrame to financials/ as parquet and print confirmation., _save()

### Community 8 - "Financial Validation Data"
Cohesion: 1.0
Nodes (2): pull_compustat(), Pull quarterly fundamentals from comp.fundq.     SIC code joined from comp.compa

### Community 9 - "Financial Validation Data"
Cohesion: 1.0
Nodes (2): pull_crsp(), Pull monthly stock returns and prices from crsp.msf.     Scoped to permnos from

### Community 10 - "Financial Validation Data"
Cohesion: 1.0
Nodes (2): pull_ccm_link(), Pull the Compustat–CRSP Merged link table.     Maps gvkey (Compustat) ↔ permno (

### Community 11 - "Financial Validation Data"
Cohesion: 1.0
Nodes (2): Read unique tickers from the transcript parquet.     Scopes all WRDS queries to, _ticker_list_from_transcripts()

### Community 12 - "Project Scaffolding"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Project Scaffolding"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "LLM Scoring Pipeline"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **19 isolated node(s):** `config.py ──────────────────────────────────────────────────────────────────────`, `download_financials.py ─────────────────────────────────────────────────────────`, `Read unique tickers from the transcript parquet.     Scopes all WRDS queries to`, `Write DataFrame to financials/ as parquet and print confirmation.`, `Pull quarterly fundamentals from comp.fundq.     SIC code joined from comp.compa` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Project Scaffolding`** (2 nodes): `config.py ──────────────────────────────────────────────────────────────────────`, `config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Project Scaffolding`** (2 nodes): `Write DataFrame to financials/ as parquet and print confirmation.`, `_save()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Financial Validation Data`** (2 nodes): `pull_compustat()`, `Pull quarterly fundamentals from comp.fundq.     SIC code joined from comp.compa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Financial Validation Data`** (2 nodes): `pull_crsp()`, `Pull monthly stock returns and prices from crsp.msf.     Scoped to permnos from`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Financial Validation Data`** (2 nodes): `pull_ccm_link()`, `Pull the Compustat–CRSP Merged link table.     Maps gvkey (Compustat) ↔ permno (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Financial Validation Data`** (2 nodes): `Read unique tickers from the transcript parquet.     Scopes all WRDS queries to`, `_ticker_list_from_transcripts()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Project Scaffolding`** (1 nodes): `validate.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Project Scaffolding`** (1 nodes): `app.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `masker.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LLM Scoring Pipeline`** (1 nodes): `scorer.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LLM investment score` connect `Project Reference Paper` to `LLM Scoring Pipeline`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `Validation regressions` connect `Project Reference Paper` to `Financial Validation Data`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `Claude scoring pipeline` connect `LLM Scoring Pipeline` to `Project Reference Paper`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **What connects `config.py ──────────────────────────────────────────────────────────────────────`, `download_financials.py ─────────────────────────────────────────────────────────`, `Read unique tickers from the transcript parquet.     Scopes all WRDS queries to` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._