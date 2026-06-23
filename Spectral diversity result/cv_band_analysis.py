# -*- coding: utf-8 -*-
"""
CV Spectral Band Analysis — Species Richness vs CV across Wavelength Regions
=============================================================================
Based on:
  - Wang et al. (2018) Ecological Applications — CV spectral bands across scales
  - Gholizadeh et al. (2018) Remote Sensing of Environment — band selection &
    scale dependence of CV spectral regions

Analyses performed:
  1.  OLS regression: each CV band vs Species Richness, per resolution
      (linear + log-linear, AIC selection, LOOCV-RMSE, FDR correction)
  2.  Cross-resolution R² decay plot per spectral band (Wang Fig 4 style)
  3.  Spectral band comparison at each resolution: which wavelength region
      is most informative? (Wang Table 2 / Gholizadeh Fig 10 style)
  4.  ANCOVA slope comparison across resolutions for each band
      (Wang's test: do regression slopes differ across scales?)
  5.  Heatmap: R² across (band × resolution)
  6.  Line plot: mean CV per band vs Species Richness at each resolution
      (Wang Fig 5 style — shows how separation between richness levels collapses)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import linregress, shapiro, f as f_dist
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── 0. CONFIG ──────────────────────────────────────────────────────────────────
FILE = "D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/spectral_metrics_results_all_v3_CV_only.xlsx"
ALPHA = 0.05

SHEETS = ["2cm_CV", "5cm_CV", "10cm_CV", "50cm_CV", "1m_CV"]
RES_LABELS = ["2 cm", "5 cm", "10 cm", "50 cm", "1 m"]
RES_X = [2, 5, 10, 50, 100]       # numeric for plotting (cm)

# CV columns and their readable labels + wavelength centres (nm)
CV_COLS = [
    "CV_mean_400_2500",
    "CV_mean_430_925",
    "CV_mean_430_700",
    "CV_mean_700_900",
    "CV_mean_1500_1800",
    "CV_mean_2000_2500",
]
CV_LABELS = [
    "CV 400–2500 nm\n(Full range)",
    "CV 430–925 nm\n(VIS-NIR)",
    "CV 430–700 nm\n(Visible)",
    "CV 700–900 nm\n(NIR)",
    "CV 1500–1800 nm\n(SWIR-1)",
    "CV 2000–2500 nm\n(SWIR-2)",
]
CV_SHORT = ["Full", "VIS-NIR", "VIS", "NIR", "SWIR-1", "SWIR-2"]

# Colours per band (for multi-line plots)
BAND_COLORS = ["#212121", "#1565C0", "#2E7D32", "#B71C1C", "#6A1B9A", "#E65100"]
BAND_LINES  = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,1))]

# ── 1. LOAD DATA ───────────────────────────────────────────────────────────────
dfs = {}
for s in SHEETS:
    dfs[s] = pd.read_excel(FILE, sheet_name=s)

n = len(dfs[SHEETS[0]])
print(f"n = {n} plots, species range: "
      f"{dfs[SHEETS[0]]['Species Count'].min()}–{dfs[SHEETS[0]]['Species Count'].max()}")

# ── 2. HELPER FUNCTIONS ────────────────────────────────────────────────────────
def aic_ols(x, y):
    slope, intercept, *_ = linregress(x, y)
    resid = y - (slope * x + intercept)
    sse = np.sum(resid**2)
    return n * np.log(sse / n) + 2 * 3   # k=3: intercept, slope, sigma

def loocv_rmse(x, y):
    preds = []
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        s, b, *_ = linregress(x[mask], y[mask])
        preds.append(s * x[i] + b)
    return np.sqrt(mean_squared_error(y, preds))

def best_model_r2_p(x, y):
    """Return R², p, model_label, LOOCV-RMSE for best AIC model."""
    x_log = np.log(x + 1)
    s1, b1, r1, p1, _ = linregress(x, y)
    s2, b2, r2, p2, _ = linregress(x_log, y)
    aic1 = aic_ols(x, y)
    aic2 = aic_ols(x_log, y)
    if aic2 < aic1:
        return r2**2, p2, "Log-lin", loocv_rmse(x_log, y)
    else:
        return r1**2, p1, "Lin", loocv_rmse(x, y)

def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    bh = np.empty(m)
    for rank, idx in enumerate(order, 1):
        bh[idx] = min(pvals[idx] * m / rank, 1.0)
    for i in range(m - 2, -1, -1):
        bh[order[i]] = min(bh[order[i]], bh[order[i + 1]])
    return bh

def fmtP(p):
    if p < 0.001: return "< 0.001"
    if p < 0.01:  return p.toFixed(4) if hasattr(p,'toFixed') else f"{p:.4f}"
    return f"{p:.3f}"

# ── 3. REGRESSION ANALYSIS: ALL BANDS × ALL RESOLUTIONS ───────────────────────
print("\n" + "="*70)
print("REGRESSION ANALYSIS: CV bands vs Species Richness")
print("="*70)

results = []
for res_i, (sheet, rlab) in enumerate(zip(SHEETS, RES_LABELS)):
    df = dfs[sheet]
    x  = df["Species Count"].values.astype(float)

    # collect raw p-values for FDR correction across the 6 bands
    raw_p = []
    for col in CV_COLS:
        y = df[col].values.astype(float)
        _, p, _, _ = best_model_r2_p(x, y)
        raw_p.append(p)
    fdr_p = bh_correct(np.array(raw_p))

    for col_i, (col, label, short) in enumerate(zip(CV_COLS, CV_LABELS, CV_SHORT)):
        y = df[col].values.astype(float)
        r2, p_raw, model, loocv = best_model_r2_p(x, y)
        sw_stat, sw_p = shapiro(y - (linregress(
            np.log(x+1) if model == "Log-lin" else x, y)[0] *
            (np.log(x+1) if model == "Log-lin" else x) +
            linregress(np.log(x+1) if model == "Log-lin" else x, y)[1]))
        p_fdr = fdr_p[col_i]
        results.append({
            "Resolution": rlab,
            "Band": short,
            "Full_label": label,
            "R2": round(r2, 3),
            "p_raw": round(p_raw, 5),
            "p_FDR": round(p_fdr, 5),
            "sig_FDR": p_fdr < ALPHA,
            "Model": model,
            "LOOCV_RMSE": round(loocv, 3),
            "SW_p": round(sw_p, 3),
        })

res_df = pd.DataFrame(results)
print(res_df[["Resolution","Band","R2","p_raw","p_FDR","sig_FDR","Model","LOOCV_RMSE","SW_p"]].to_string(index=False))
res_df.to_csv("cv_band_regression_summary.csv", index=False)
print("\nSaved → cv_band_regression_summary.csv")

# ── 4. FIGURE A: R² DECAY ACROSS RESOLUTIONS PER BAND  ────────────────────────
# Wang et al. Fig 4 equivalent: each band is a line, x = resolution
fig1, ax = plt.subplots(figsize=(9, 5.5))
log_x = np.log10(RES_X)

for col_i, (col, short, color, ls) in enumerate(zip(CV_COLS, CV_SHORT, BAND_COLORS, BAND_LINES)):
    r2_vals = []
    sig_vals = []
    for sheet in SHEETS:
        df = dfs[sheet]
        x  = df["Species Count"].values.astype(float)
        y  = df[col].values.astype(float)
        r2, p, model, _ = best_model_r2_p(x, y)
        r2_vals.append(r2)
        sig_vals.append(p < ALPHA)   # raw; FDR done per-resolution above
    r2_arr = np.array(r2_vals)
    ax.plot(log_x, r2_arr, color=color, lw=1.8, linestyle=ls,
            marker="o", ms=6, label=short)
    # mark non-significant points with open circles
    for j, (sig, rv) in enumerate(zip(sig_vals, r2_arr)):
        if not sig:
            ax.scatter(log_x[j], rv, s=60, facecolors="white",
                       edgecolors=color, lw=1.5, zorder=5)

ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.7)
ax.text(log_x[0] - 0.07, 0.505, "R²=0.50", fontsize=14, color="gray")
ax.set_xticks(log_x)
ax.set_xticklabels(RES_LABELS, fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel("Spatial Resolution (coarser →)", fontsize=14)
ax.set_ylabel("R² (best AIC model)", fontsize=14)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, linestyle="--", alpha=0.35)
#ax.legend(fontsize=9, loc="upper right", framealpha=0.9, title="CV Spectral Band")
ax.legend(
    fontsize=9,
    title='CV Spectral Band',
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    framealpha=0.9,
    borderaxespad=0.0
)
ax.set_title("CV Spectral Band vs Species Richness: R² Across Resolutions\n"
             "Open symbols = not significant (raw p > 0.05)  |  n = 12",
             fontsize=14)
plt.tight_layout()
plt.savefig("fig_cv_band_r2_decay.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → fig_cv_band_r2_decay.png")
#%%
# ── 5. FIGURE B: R² HEATMAP (band × resolution)  ─────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "r2map", ["#ffffff","#c8e6c9","#43a047","#1b5e20"], N=256)

r2_mat = np.zeros((len(CV_COLS), len(SHEETS)))
p_mat  = np.zeros_like(r2_mat)
for ri, sheet in enumerate(SHEETS):
    df = dfs[sheet]
    x  = df["Species Count"].values.astype(float)
    raw_p = []
    r2s = []
    for col in CV_COLS:
        y = df[col].values.astype(float)
        r2, p, model, _ = best_model_r2_p(x, y)
        raw_p.append(p); r2s.append(r2)
    fdr_p = bh_correct(np.array(raw_p))
    for bi in range(len(CV_COLS)):
        r2_mat[bi, ri] = r2s[bi]
        p_mat[bi, ri]  = fdr_p[bi]

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
im = ax2.imshow(r2_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for bi in range(len(CV_COLS)):
    for ri in range(len(SHEETS)):
        v = r2_mat[bi, ri]
        p = p_mat[bi, ri]
        stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        tc = "white" if v > 0.65 else "black"
        ax2.text(ri, bi, f"{v:.2f}{stars}", ha="center", va="center",
                 fontsize=12, color=tc, fontweight="bold")

ax2.set_xticks(range(len(SHEETS)))
ax2.set_xticklabels(RES_LABELS, fontsize=14)
ax2.set_yticks(range(len(CV_COLS)))
ax2.set_yticklabels(CV_SHORT, fontsize=14)
ax2.set_xlabel("Spatial Resolution (coarser →)", fontsize=14)
ax2.set_title("R² Heatmap: CV Spectral Bands vs Species Richness\n"
              "** FDR p < 0.01  * FDR p < 0.05  |  n = 12", fontsize=14)
cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
cbar.set_label("R²", fontsize=14)
plt.tight_layout()
plt.savefig("fig_cv_band_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → fig_cv_band_heatmap.png")
#%%
# ── 6. FIGURE C: CV vs Species Richness scatter — one panel per band/resolution
# Wang Fig 4 style: all resolutions overlaid on one axis per CV band
fig3, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
RES_COLORS = ["#1A237E","#1565C0","#0288D1","#F57F17","#E65100"]

x_plot = np.linspace(3, 31, 100)

for bi, (col, short, ax) in enumerate(zip(CV_COLS, CV_SHORT, axes)):
    for ri, (sheet, rlab, rc) in enumerate(zip(SHEETS, RES_LABELS, RES_COLORS)):
        df = dfs[sheet]
        x  = df["Species Count"].values.astype(float)
        y  = df[col].values.astype(float)
        r2, p, model, _ = best_model_r2_p(x, y)
        x_fit = np.log(x_plot + 1) if model == "Log-lin" else x_plot
        sl, ic, *_ = linregress(np.log(x+1) if model=="Log-lin" else x, y)
        y_fit = sl * x_fit + ic
        stars = ("***" if p < 0.001 else "**" if p < 0.01 else
                 "*" if p < 0.05 else "ns")
        ls = "-" if p < 0.05 else "--"
        ax.scatter(x, y, color=rc, s=20, alpha=0.7, zorder=3)
        ax.plot(x_plot, y_fit, color=rc, lw=1.4, linestyle=ls,
                label=f"{rlab}: R²={r2:.2f} ({stars})")
    ax.set_title(f"CV — {short}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Species Count", fontsize=14)
    ax.set_ylabel("CV", fontsize=14)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig3.suptitle("CV Spectral Bands vs Species Richness — All Resolutions Overlaid\n"
              "Solid lines = FDR-significant  Dashed = not significant  |  n = 12",
              fontsize=14, y=0.98)
plt.savefig("fig_cv_band_scatter_overlay.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → fig_cv_band_scatter_overlay.png")

# ── 7. ANCOVA: do regression slopes differ across resolutions? ────────────────
# Wang et al. used ANCOVA to test whether scale changes the slope.
# Here: for each CV band, test interaction (resolution group × species richness)
# using a simple pooled F-test approach.
print("\n" + "="*70)
print("ANCOVA: Do slopes differ across resolutions? (Wang Table 2 equivalent)")
print("="*70)

from scipy.stats import f as f_dist

ancova_results = []
for col, short in zip(CV_COLS, CV_SHORT):
    # Pool all resolution data; fit common slope vs separate slopes
    all_x, all_y, all_g = [], [], []
    for gi, sheet in enumerate(SHEETS):
        df = dfs[sheet]
        all_x.extend(df["Species Count"].values)
        all_y.extend(df[col].values)
        all_g.extend([gi] * len(df))
    all_x = np.array(all_x, float)
    all_y = np.array(all_y, float)
    all_g = np.array(all_g)

    # Model 1 (restricted): common slope, separate intercepts
    X_r = np.column_stack([np.eye(len(SHEETS))[all_g], all_x])
    b_r, rss_r, *_ = np.linalg.lstsq(X_r, all_y, rcond=None)
    if len(rss_r) == 0:
        rss_r = np.sum((all_y - X_r @ b_r)**2)
    else:
        rss_r = rss_r[0]

    # Model 2 (full): separate slopes AND intercepts
    X_f_cols = [np.eye(len(SHEETS))[all_g]]   # group intercepts
    for gi in range(len(SHEETS)):              # group slopes
        slope_col = np.where(all_g == gi, all_x, 0)
        X_f_cols.append(slope_col.reshape(-1,1))
    X_f = np.column_stack([np.eye(len(SHEETS))[all_g]] +
                          [np.where(all_g==gi, all_x, 0) for gi in range(len(SHEETS))])
    b_f, *_ = np.linalg.lstsq(X_f, all_y, rcond=None)
    rss_f = np.sum((all_y - X_f @ b_f)**2)

    df_r = len(all_y) - (len(SHEETS) + 1)  # restricted df
    df_f = len(all_y) - (2 * len(SHEETS))  # full df
    df_diff = df_r - df_f
    if df_diff > 0 and rss_f > 0:
        F = ((rss_r - rss_f) / df_diff) / (rss_f / df_f)
        p_ancova = 1 - f_dist.cdf(F, df_diff, df_f)
    else:
        F, p_ancova = np.nan, np.nan

    sig = "***" if p_ancova < 0.001 else "**" if p_ancova < 0.01 else \
          "*" if p_ancova < 0.05 else "ns"
    print(f"  {short:8s}: F({df_diff},{df_f}) = {F:.2f}, p = {p_ancova:.4f} {sig}")
    ancova_results.append({"Band": short, "F_stat": round(F,3),
                           "p_ANCOVA": round(p_ancova,4), "sig": sig})

ancova_df = pd.DataFrame(ancova_results)
#%%
# ── 8. FIGURE D: Spectral region contribution per resolution ──────────────────
# Gholizadeh Fig 10 style: at each resolution, bar chart of R² per spectral band
fig4, axes4 = plt.subplots(1, 5, figsize=(14, 4), sharey=True)
BAR_COLORS = BAND_COLORS

for ri, (sheet, rlab, ax) in enumerate(zip(SHEETS, RES_LABELS, axes4)):
    df = dfs[sheet]
    x  = df["Species Count"].values.astype(float)
    raw_p = []
    r2s = []
    for col in CV_COLS:
        y = df[col].values.astype(float)
        r2, p, model, _ = best_model_r2_p(x, y)
        r2s.append(r2); raw_p.append(p)
    fdr_p = bh_correct(np.array(raw_p))

    bars = ax.bar(range(len(CV_COLS)), r2s, color=BAR_COLORS, edgecolor="white",
                  linewidth=0.5)
    # hatching for non-significant
    for bar, p in zip(bars, fdr_p):
        if p >= ALPHA:
            bar.set_hatch("///")
            bar.set_alpha(0.5)
    ax.set_xticks(range(len(CV_COLS)))
    ax.set_xticklabels(CV_SHORT, rotation=45, ha="right", fontsize=7.5)
    ax.set_title(rlab, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", lw=0.7, linestyle=":")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    if ri == 0:
        ax.set_ylabel("R² (best model)", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.90])
fig4.suptitle("Spectral Band Contribution to Species Richness Prediction per Resolution\n"
              "Hatched bars = not FDR-significant  Dotted line = R²=0.50",
              fontsize=11, y=0.97)
plt.savefig("fig_cv_band_contribution.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → fig_cv_band_contribution.png")

#%%
# ── 9. FIGURE E: Mean CV per band, grouped by species richness ────────────────
# Wang Fig 5 style — shows how spectral diversity separates richness levels
# Group plots into richness bins: Low (3–8), Mid (9–20), High (21–30)
def richness_group(sr):
    if sr <= 8:  return "Low (3–8)"
    if sr <= 20: return "Mid (9–20)"
    return "High (21–30)"

fig5, axes5 = plt.subplots(1, 5, figsize=(15, 4), sharey=False)
GROUP_COLORS = {"Low (3–8)": "#E53935", "Mid (9–20)": "#FB8C00", "High (21–30)": "#43A047"}

for ri, (sheet, rlab, ax) in enumerate(zip(SHEETS, RES_LABELS, axes5)):
    df = dfs[sheet].copy()
    df["Group"] = df["Species Count"].apply(richness_group)
    x_pos = np.arange(len(CV_COLS))
    for grp, gc in GROUP_COLORS.items():
        sub = df[df["Group"] == grp]
        if len(sub) == 0: continue
        means = [sub[col].mean() for col in CV_COLS]
        sems  = [sub[col].sem()  for col in CV_COLS]
        ax.errorbar(x_pos, means, yerr=sems, color=gc, lw=1.8,
                    marker="o", ms=5, capsize=3, label=grp)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CV_SHORT, rotation=45, ha="right", fontsize=7)
    ax.set_title(rlab, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=14)
    if ri == 4:
        ax.set_ylabel("Mean CV ± SEM", fontsize=14)
        ax.legend(fontsize=10, loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.85])
fig5.suptitle("Mean CV ± SEM by Spectral Band and Richness Group Across Resolutions\n"
              "Separation between groups indicates biodiversity signal\n",
              fontsize=16, y=1.07)
plt.savefig("fig_cv_band_richness_separation.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → fig_cv_band_richness_separation.png")

# ── 10. SUMMARY PRINT ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE: Best band per resolution")
print("="*70)
for res in RES_LABELS:
    sub = res_df[res_df["Resolution"]==res].sort_values("R2", ascending=False)
    best = sub.iloc[0]
    nsig = sub["sig_FDR"].sum()
    print(f"  {res:6s}: Best = {best['Band']:7s} R²={best['R2']:.3f} "
          f"({best['Model']}) | {nsig}/{len(CV_COLS)} bands FDR-significant")

print("\n" + "="*70)
print("ANCOVA SUMMARY: Slopes change across resolutions?")
print("="*70)
print(ancova_df.to_string(index=False))
print("\nDone. Figures and CSV saved.")
