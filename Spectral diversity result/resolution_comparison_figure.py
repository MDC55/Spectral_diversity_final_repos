# -*- coding: utf-8 -*-
"""
Resolution Comparison Figure
------------------------------
Reads the per-resolution regression_summary CSVs and produces two
publication-ready figures:

  Figure A — Line plot: R² vs spatial resolution
             Two highlighted groups: AE-based vs Traditional metrics
             Mean group trends shown as bold lines

  Figure B — Heatmap: R² × (metric × resolution) with
             significance annotations and group dividers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── 0. CONFIG ──────────────────────────────────────────────────────────────────
# Point these to wherever your CSVs are saved
CSV_DIR = "./"   # change if CSVs are in a different folder

RESOLUTIONS   = ["2cm", "5cm", "10cm", "50cm", "1m"]
RES_LABELS    = ["2 cm", "5 cm", "10 cm", "50 cm", "1 m"]   # for axis ticks
RES_X         = [2, 5, 10, 50, 100]                          # numeric x-axis (cm)

# Metric groups — edit if your metric names differ
TRAD_METRICS = ["PCA-CHV", "SAM", "SID", "CHA", "CV"]
AE_METRICS   = ["AE-CHV", "AE Latent Variance Sum", "AE Mean Pairwise Distance"]

# Colours
TRAD_PALETTE = ["#1565C0", "#1976D2", "#42A5F5", "#0D47A1", "#29B6F6"]
AE_PALETTE   = ["#B71C1C", "#C62828", "#EF5350"]
TRAD_MEAN_C  = "#1A237E"
AE_MEAN_C    = "#7F0000"

# ── 1. LOAD ALL CSVs ──────────────────────────────────────────────────────────
dfs = {}
for res in RESOLUTIONS:
    path = f"{CSV_DIR}regression_summary_{res}.csv"
    dfs[res] = pd.read_csv(path).set_index("Metric")

all_metrics = TRAD_METRICS + AE_METRICS

# ── 2. BUILD PIVOT TABLES ─────────────────────────────────────────────────────
def best_r2(row):
    return row["R2_log"] if row["Best_model"] == "log-linear" else row["R2_lin"]

def best_p(row):
    return row["p_log"] if row["Best_model"] == "log-linear" else row["p_lin"]

r2_pivot  = pd.DataFrame(index=all_metrics, columns=RESOLUTIONS, dtype=float)
sig_pivot = pd.DataFrame(index=all_metrics, columns=RESOLUTIONS, dtype=bool)
bonf_pivot= pd.DataFrame(index=all_metrics, columns=RESOLUTIONS, dtype=bool)

for res, df in dfs.items():
    for m in all_metrics:
        r2_pivot.loc[m, res]   = best_r2(df.loc[m])
        sig_pivot.loc[m, res]  = df.loc[m, "sig_FDR"]
        bonf_pivot.loc[m, res] = df.loc[m, "sig_Bonferroni"]

r2_pivot = r2_pivot.astype(float)

# Group means
trad_mean = r2_pivot.loc[TRAD_METRICS].mean()
ae_mean   = r2_pivot.loc[AE_METRICS].mean()

# ── 3. FIGURE A — Line Plot ────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(11, 6))   # wider figure = more room for labels

log_x = np.log10(RES_X)   # log scale makes spacing readable

# Collect label info for adjustText
label_texts  = []
label_points = []   # (x, y) of line endpoint

# — Individual metric lines (thin, semi-transparent)
for i, m in enumerate(TRAD_METRICS):
    vals = r2_pivot.loc[m].values
    ax.plot(log_x, vals, color=TRAD_PALETTE[i], lw=1.2,
            alpha=0.55, marker="o", ms=4, zorder=2)
    label_points.append((log_x[-1], vals[-1]))
    label_texts.append((m, TRAD_PALETTE[i]))

for i, m in enumerate(AE_METRICS):
    vals = r2_pivot.loc[m].values
    ax.plot(log_x, vals, color=AE_PALETTE[i], lw=1.2,
            alpha=0.55, marker="s", ms=4, zorder=2,
            linestyle="--")
    label_points.append((log_x[-1], vals[-1]))
    label_texts.append((m, AE_PALETTE[i]))

# — Label placement with guaranteed minimum vertical spacing
# Sort labels by their true endpoint y-value
MIN_GAP = 0.062   # minimum R² units between label centres
LABEL_X = log_x[-1] + 0.10   # fixed x position for all labels

indexed = sorted(enumerate(zip(label_texts, label_points)),
                 key=lambda x: x[1][1][1])   # sort by true y ascending

# Pass 1: assign ideal y = true y
assigned_y = [pt[1] for _, (_, pt) in indexed]

# Pass 2: push labels apart iteratively until no overlap remains
for _ in range(500):
    moved = False
    for k in range(1, len(assigned_y)):
        if assigned_y[k] - assigned_y[k-1] < MIN_GAP:
            mid = (assigned_y[k] + assigned_y[k-1]) / 2
            assigned_y[k-1] = mid - MIN_GAP / 2
            assigned_y[k]   = mid + MIN_GAP / 2
            moved = True
    if not moved:
        break

# Clamp to axis range while preserving order
y_lo, y_hi = -0.01, 1.03
for k in range(len(assigned_y)):
    assigned_y[k] = max(y_lo, min(y_hi, assigned_y[k]))

# Pass 3: draw labels + thin connector lines
for rank, (orig_i, ((txt, col), (px, py))) in enumerate(indexed):
    y_label = assigned_y[rank]
    ax.annotate(
        txt,
        xy=(px, py),                          # arrow tip at line endpoint
        xytext=(LABEL_X, y_label),            # label position
        fontsize=10, color=col,
        va="center", ha="left",
        arrowprops=dict(arrowstyle="-",
                        color=col, lw=0.5,
                        relpos=(0, 0.5)),
    )

# — Group mean lines (bold)
ax.plot(log_x, trad_mean.values, color=TRAD_MEAN_C, lw=2.8,
        marker="o", ms=7, zorder=4, label="Traditional metrics (mean)")
ax.plot(log_x, ae_mean.values,   color=AE_MEAN_C,   lw=2.8,
        marker="s", ms=7, zorder=4, label="AE-based metrics (mean)",
        linestyle="--")

# — Shaded areas between mean lines (show where AE > Traditional)
ax.fill_between(log_x,
                ae_mean.values, trad_mean.values,
                where=(ae_mean.values >= trad_mean.values),
                alpha=0.10, color=AE_MEAN_C,
                label="AE advantage region")
ax.fill_between(log_x,
                trad_mean.values, ae_mean.values,
                where=(trad_mean.values > ae_mean.values),
                alpha=0.10, color=TRAD_MEAN_C,
                label="Traditional advantage region")

# — Significance threshold line
ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.7)
ax.text(log_x[0] - 0.05, 0.505, "R²=0.50", fontsize=12,
        color="gray", va="bottom")

# — Axes formatting
ax.set_xticks(log_x)
ax.set_xticklabels(RES_LABELS, fontsize=14)
ax.set_xlabel("Spatial Resolution (coarser →)", fontsize=14)
ax.set_ylabel("R² (best model: linear or log-linear)", fontsize=14)
ax.set_ylim(-0.03, 1.05)
ax.set_xlim(log_x[0] - 0.15, log_x[-1] + 0.90)  # wider right margin for labels
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.grid(True, which="major", linestyle="--", alpha=0.35)
ax.grid(True, which="minor", linestyle=":",  alpha=0.18)

# — Crossover annotation
cross_x = (log_x[2] + log_x[3]) / 2
ax.annotate("AE metrics\noutperform\nTraditional\nhere →",
            xy=(log_x[2], ae_mean.iloc[2]),
            xytext=(log_x[2] - 0.35, 0.35),
            fontsize=10, color=AE_MEAN_C,
            arrowprops=dict(arrowstyle="->", color=AE_MEAN_C, lw=0.9))

ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
          edgecolor="lightgray")
ax.set_title("Spectral Diversity Metrics vs Species Richness:\n"
             "R² Across Spatial Resolutions  (n = 12)",
             fontsize=14, pad=10)

plt.tight_layout()
plt.savefig("fig_resolution_lineplot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → fig_resolution_lineplot.png")

# ── 4. FIGURE B — Annotated Heatmap ───────────────────────────────────────────
# Custom colormap: white (0) → deep green (1)
cmap = LinearSegmentedColormap.from_list(
    "r2map", ["#ffffff", "#c8e6c9", "#43a047", "#1b5e20"], N=256)

fig2, ax2 = plt.subplots(figsize=(8, 5.5))

data = r2_pivot.values   # shape (8 metrics × 5 resolutions)
im   = ax2.imshow(data, cmap=cmap, vmin=0, vmax=1,
                  aspect="auto")

# — Annotate each cell
for i, metric in enumerate(all_metrics):
    for j, res in enumerate(RESOLUTIONS):
        val  = r2_pivot.loc[metric, res]
        fdr  = sig_pivot.loc[metric, res]
        bonf = bonf_pivot.loc[metric, res]
        stars = ("**" if bonf else ("*" if fdr else ""))
        txt_color = "white" if val > 0.65 else "black"
        ax2.text(j, i, f"{val:.2f}{stars}",
                 ha="center", va="center",
                 fontsize=9, color=txt_color, fontweight="bold")

# — Group divider between Traditional and AE metrics
divider_y = len(TRAD_METRICS) - 0.5
ax2.axhline(divider_y, color="white", lw=3)
ax2.axhline(divider_y, color="#37474F", lw=1.5, linestyle="--")

# — Group labels on left margin
ax2.annotate("Traditional\nMetrics",
             xy=(-0.62, (len(TRAD_METRICS) - 1) / 2),
             xycoords=("axes fraction", "data"),
             fontsize=8.5, color=TRAD_MEAN_C, fontweight="bold",
             ha="center", va="center", rotation=90)
ax2.annotate("AE-Based\nMetrics",
             xy=(-0.62, len(TRAD_METRICS) + (len(AE_METRICS) - 1) / 2),
             xycoords=("axes fraction", "data"),
             fontsize=8.5, color=AE_MEAN_C, fontweight="bold",
             ha="center", va="center", rotation=90)

# — Axes
ax2.set_xticks(range(len(RESOLUTIONS)))
ax2.set_xticklabels(RES_LABELS, fontsize=10)
ax2.set_yticks(range(len(all_metrics)))
ax2.set_yticklabels(all_metrics, fontsize=9)
ax2.set_xlabel("Spatial Resolution (coarser →)", fontsize=11)
ax2.xaxis.set_label_position("bottom")

# — Colorbar
cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
cbar.set_label("R²", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# — Legend for stars
star_patch = mpatches.Patch(color="none",
    label="\n** Bonferroni  and FDR sig.  * FDR sig. only  (no mark = ns)")
ax2.legend(handles=[star_patch], fontsize=7.5,
           loc="upper right", bbox_to_anchor=(1.0, -0.07),
           frameon=False)

ax2.set_title("R² Heatmap: Spectral Diversity vs Species Richness\n"
              "Across Metrics and Spatial Resolutions  (n = 12)",
              fontsize=11, pad=10)

plt.tight_layout()
plt.savefig("fig_resolution_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → fig_resolution_heatmap.png")

print("\nDone. Two figures produced:")
print("  fig_resolution_lineplot.png  — R² vs resolution, AE vs Traditional")
print("  fig_resolution_heatmap.png   — Full R² heatmap with significance stars")
