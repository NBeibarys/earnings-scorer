"""
generate_figures.py
Reads saved parquets from `results/` and produces all figures used in paper.md.
Run once after the main scoring + IBES validation cells finish.

Outputs (PNG, 150 dpi):
    paper/figures/fig1_score_distribution.png
    paper/figures/fig2_annual_mean_score.png
    paper/figures/fig3_prompt_robustness.png
    paper/figures/fig4_ibes_within_firm.png
    paper/figures/fig5_chunks_per_transcript.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parents[1]
RESULTS  = ROOT / "results"
FIG_DIR  = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "figure.autolayout": True,
})

scores = pd.read_parquet(RESULTS / "transcript_scores.parquet")
chunks = pd.read_parquet(RESULTS / "chunks_w1500.parquet")


# ── Figure 1: transcript score distribution ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(scores["informative_mean_score"], bins=40, color="#3a76a8", edgecolor="white")
ax.axvline(0, color="grey", linestyle=":", linewidth=1)
ax.set_xlabel("Informative-mean transcript score")
ax.set_ylabel("Number of transcripts")
ax.set_title("Distribution of LLM transcript scores (n = {:,})".format(len(scores)))
fig.savefig(FIG_DIR / "fig1_score_distribution.png", bbox_inches="tight")
plt.close(fig)


# ── Figure 2: annual mean score with crisis shading ────────────────────────
annual = scores.groupby("year")["informative_mean_score"].agg(["mean", "sem"]).reset_index()
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(annual["year"], annual["mean"], yerr=annual["sem"],
            fmt="-o", color="#1f4e79", capsize=3, markersize=4)
ax.axhline(0, color="grey", linestyle=":", linewidth=1)
for crisis in (2008, 2009, 2020):
    if crisis in annual["year"].values:
        ax.axvspan(crisis - 0.4, crisis + 0.4, color="red", alpha=0.08)
ax.set_xlabel("Year")
ax.set_ylabel("Mean transcript score")
ax.set_title("Annual mean LLM transcript score (financial crises shaded)")
fig.savefig(FIG_DIR / "fig2_annual_mean_score.png", bbox_inches="tight")
plt.close(fig)


# ── Figure 3: prompt robustness scatter (V1 vs V2 alt, V1 vs V3 terse) ─────
robust = pd.read_parquet(RESULTS / "prompt_robustness_transcript_scores.parquet")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, col, title in zip(axes, ("v2", "v3"),
                           ("V1 (primary) vs V2 (alternative wording)",
                            "V1 (primary) vs V3 (terse, no scaffold)")):
    ax.scatter(robust["v1"], robust[col], s=10, alpha=0.45, color="#3a76a8")
    lim = (-1.05, 1.05)
    ax.plot(lim, lim, color="grey", linestyle=":", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("V1 transcript score")
    ax.set_title(title)
axes[0].set_ylabel("Alternative-prompt transcript score")
fig.suptitle("Cross-prompt agreement on the 500-transcript robustness subsample")
fig.savefig(FIG_DIR / "fig3_prompt_robustness.png", bbox_inches="tight")
plt.close(fig)


# ── Figure 4: IBES validation - direction-conditional means ────────────────
ibes_v = pd.read_parquet(RESULTS / "external_validation_ibes.parquet")
buckets = []
for label, sub in [
    ("score < -0.25",  ibes_v[ibes_v["informative_mean_score"] < -0.25]),
    ("|score| <= 0.25", ibes_v[ibes_v["informative_mean_score"].abs() <= 0.25]),
    ("score > +0.25",  ibes_v[ibes_v["informative_mean_score"] >  0.25]),
]:
    buckets.append({
        "label":   label,
        "n":       len(sub),
        "mean":    sub["cpx_post_demean"].mean() if len(sub) else np.nan,
        "sem":     sub["cpx_post_demean"].sem()  if len(sub) else np.nan,
    })
b = pd.DataFrame(buckets)
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#c0504d", "#7f7f7f", "#1f4e79"]
ax.bar(b["label"], b["mean"], yerr=b["sem"], color=colors, capsize=4, edgecolor="white")
ax.axhline(0, color="grey", linestyle=":", linewidth=1)
for i, row in b.iterrows():
    ax.text(i, row["mean"], f"  n = {row['n']:,}", ha="center",
            va="bottom" if row["mean"] >= 0 else "top", fontsize=9)
ax.set_ylabel("Within-firm-demeaned post-call CPX consensus  (USD millions)")
ax.set_title("Mean post-call analyst capex revision by LLM score sign")
fig.savefig(FIG_DIR / "fig4_ibes_within_firm.png", bbox_inches="tight")
plt.close(fig)


# ── Figure 5: chunks per transcript distribution ───────────────────────────
per_tr = chunks.groupby("transcript_id").size()
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(per_tr, bins=range(1, int(per_tr.max()) + 2), color="#3a76a8", edgecolor="white")
ax.set_xlabel("Chunks per transcript (1,500-word turn-preserving)")
ax.set_ylabel("Number of transcripts")
ax.set_title("Chunk distribution under turn-preserving 1,500-word segmentation")
ax.text(0.95, 0.92,
        f"median = {per_tr.median():.0f}\np95 = {per_tr.quantile(0.95):.0f}\nmax = {per_tr.max():.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"))
fig.savefig(FIG_DIR / "fig5_chunks_per_transcript.png", bbox_inches="tight")
plt.close(fig)

print("Figures saved to", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p.name)
