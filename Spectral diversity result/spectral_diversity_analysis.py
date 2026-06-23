# -*- coding: utf-8 -*-
"""
Spectral Diversity vs Species Count — Publication-Ready Analysis
----------------------------------------------------------------
Improvements over original:
  1. Leave-One-Out Cross-Validation (LOOCV) — appropriate for n=12
  2. Bonferroni & FDR (Benjamini-Hochberg) correction for multiple comparisons
  3. Linear vs log-linear model comparison (AIC-based)
  4. Cook's distance — flags influential/leverage points
  5. 95% confidence bands on all regression plots
  6. Residual normality test (Shapiro-Wilk)
  7. Publication-quality figure layout
  8. Full summary table exported to CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import linregress, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── 0. CONFIG ──────────────────────────────────────────────────────────────────
file_path = (
    "D:/Drive F 5-11-2024/A_BiosCape_Project3/"
    "Spectral_diversity/spectral_metrics_results_all_v3.xlsx"
)
SHEET = "1m"          # change as needed
#SHEET = "1m_CV"          # change as needed
ALPHA = 0.05           # nominal significance level

# ── 1. LOAD DATA ───────────────────────────────────────────────────────────────
df = pd.read_excel(file_path, sheet_name=SHEET)
print(f"Loaded sheet '{SHEET}': {df.shape[0]} rows × {df.shape[1]} cols")
print(df.head())

x_raw   = df["Species Count"].values
metrics = [c for c in df.columns if c != "Species Count"]
n       = len(x_raw)
print(f"\nn = {n} data points, {len(metrics)} metrics: {metrics}")

# ── 2. HELPER: confidence band for OLS ────────────────────────────────────────
def ols_confidence_band(x, y, x_plot, ci=0.95):
    """Return lower and upper CI bands for the regression line."""
    n   = len(x)
    xm  = x.mean()
    slope, intercept, r, p, se = linregress(x, y)
    y_pred_all = slope * x + intercept
    s2  = np.sum((y - y_pred_all) ** 2) / (n - 2)      # MSE
    ssx = np.sum((x - xm) ** 2)
    se_line = np.sqrt(s2 * (1/n + (x_plot - xm)**2 / ssx))
    t_crit  = stats.t.ppf((1 + ci) / 2, df=n - 2)
    y_fit   = slope * x_plot + intercept
    return y_fit, y_fit - t_crit * se_line, y_fit + t_crit * se_line

# ── 3. HELPER: Cook's distance ─────────────────────────────────────────────────
def cooks_distance(x, y):
    """Cook's D for simple linear regression."""
    n  = len(x)
    X  = np.column_stack([np.ones(n), x])
    b  = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ b
    resid = y - y_hat
    mse   = np.sum(resid**2) / (n - 2)
    H     = X @ np.linalg.inv(X.T @ X) @ X.T   # hat matrix
    h     = np.diag(H)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        D = (resid**2 * h) / (2 * mse * (1 - h)**2)
    return D

# ── 4. HELPER: LOOCV RMSE ──────────────────────────────────────────────────────
def loocv_rmse(x, y, log_x=False):
    """Leave-One-Out CV root mean squared error for simple linear regression."""
    xx = np.log(x + 1) if log_x else x
    preds = []
    for i in range(len(xx)):
        mask   = np.arange(len(xx)) != i
        xi, yi = xx[mask], y[mask]
        slope, intercept, *_ = linregress(xi, yi)
        preds.append(slope * xx[i] + intercept)
    return np.sqrt(mean_squared_error(y, preds))

# ── 5. HELPER: AIC for simple regression ──────────────────────────────────────
def aic_linreg(x, y):
    n   = len(x)
    slope, intercept, *_ = linregress(x, y)
    resid = y - (slope * x + intercept)
    sse   = np.sum(resid**2)
    k     = 3          # intercept, slope, sigma
    aic   = n * np.log(sse / n) + 2 * k
    return aic

# ── 6. MAIN ANALYSIS LOOP ──────────────────────────────────────────────────────
results = []

for metric in metrics:
    y = df[metric].values

    # — Linear fit
    slope, intercept, r, p_lin, se = linregress(x_raw, y)
    r2_lin   = r ** 2
    loocv_lin = loocv_rmse(x_raw, y, log_x=False)
    aic_lin  = aic_linreg(x_raw, y)

    # — Log-linear fit  (x → log(x+1))
    x_log = np.log(x_raw + 1)
    slope_l, intercept_l, r_l, p_log, se_l = linregress(x_log, y)
    r2_log   = r_l ** 2
    loocv_log = loocv_rmse(x_raw, y, log_x=True)
    aic_log  = aic_linreg(x_log, y)

    # — Best model by AIC
    best = "log-linear" if aic_log < aic_lin else "linear"

    # — Shapiro-Wilk on residuals of best model
    if best == "linear":
        resid = y - (slope * x_raw + intercept)
    else:
        resid = y - (slope_l * x_log + intercept_l)
    sw_stat, sw_p = shapiro(resid)

    # — Cook's distance (linear model)
    D = cooks_distance(x_raw, y)
    influential = np.where(D > 4 / n)[0].tolist()   # common threshold

    results.append({
        "Metric"          : metric,
        # Linear
        "Slope_lin"       : round(slope, 4),
        "Intercept_lin"   : round(intercept, 4),
        "R2_lin"          : round(r2_lin, 4),
        "p_lin"           : round(p_lin, 6),
        "LOOCV_RMSE_lin"  : round(loocv_lin, 4),
        "AIC_lin"         : round(aic_lin, 2),
        # Log-linear
        "Slope_log"       : round(slope_l, 4),
        "R2_log"          : round(r2_log, 4),
        "p_log"           : round(p_log, 6),
        "LOOCV_RMSE_log"  : round(loocv_log, 4),
        "AIC_log"         : round(aic_log, 2),
        # Best model
        "Best_model"      : best,
        # Diagnostics
        "SW_p_residuals": round(sw_p, 4),
        "Influential_pts" : str(influential) if influential else "none",
        # Raw for correction later
        "_p_best"         : p_log if best == "log-linear" else p_lin,
    })

res_df = pd.DataFrame(results)

# ── 7. MULTIPLE COMPARISON CORRECTION ─────────────────────────────────────────
p_vals = res_df["_p_best"].values
m      = len(p_vals)

# Bonferroni
bonf   = np.minimum(p_vals * m, 1.0)

# Benjamini-Hochberg (FDR)
order  = np.argsort(p_vals)
bh     = np.empty(m)
for rank, idx in enumerate(order, 1):
    bh[idx] = min(p_vals[idx] * m / rank, 1.0)
# Make monotone
for i in range(m - 2, -1, -1):
    bh[order[i]] = min(bh[order[i]], bh[order[i + 1]])

res_df["p_Bonferroni"] = np.round(bonf, 6)
res_df["p_BH_FDR"]     = np.round(bh,   6)
res_df["sig_Bonferroni"] = bonf < ALPHA
res_df["sig_FDR"]        = bh   < ALPHA
res_df.drop(columns=["_p_best"], inplace=True)

# ── 8. PRINT SUMMARY ──────────────────────────────────────────────────────────
cols_show = [
    "Metric", "Best_model", "Slope_lin", "Slope_log",
    "R2_lin", "p_lin", "LOOCV_RMSE_lin",
    "R2_log", "p_log", "LOOCV_RMSE_log",
    "p_Bonferroni", "p_BH_FDR",
    "sig_Bonferroni", "sig_FDR",
    "SW_p_residuals", "Influential_pts"
]
print("\n── Full Regression Summary ──")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(res_df[cols_show].to_string(index=False))

# ── 9. PUBLICATION-QUALITY PLOTS ───────────────────────────────────────────────
ncols  = 4
nrows  = int(np.ceil(len(metrics) / ncols))
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 4.5, nrows * 4),
                         constrained_layout=False)
axes = axes.flatten()

x_plot = np.linspace(x_raw.min() * 0.95, x_raw.max() * 1.05, 200)

for ax, metric, row in zip(axes, metrics, res_df.itertuples()):
    y = df[metric].values
    use_log = (row.Best_model == "log-linear")

    # — Fit line + CI band
    if use_log:
        slope_b, intercept_b, r_b, p_b, _ = linregress(np.log(x_raw + 1), y)
        x_fit   = np.log(x_plot + 1)
        y_fit   = slope_b * x_fit + intercept_b
        # manual CI band for log model
        xm = np.log(x_raw + 1).mean()
        s2 = np.sum((y - (slope_b * np.log(x_raw + 1) + intercept_b))**2) / (n - 2)
        ssx = np.sum((np.log(x_raw + 1) - xm)**2)
        se_line = np.sqrt(s2 * (1/n + (x_fit - xm)**2 / ssx))
        t_crit  = stats.t.ppf(0.975, df=n - 2)
        ci_lo   = y_fit - t_crit * se_line
        ci_hi   = y_fit + t_crit * se_line
        r2_disp = row.R2_log
        p_disp  = row.p_log
        loocv_disp = row.LOOCV_RMSE_log
    else:
        y_fit, ci_lo, ci_hi = ols_confidence_band(x_raw, y, x_plot)
        slope_b, intercept_b, r_b, p_b, _ = linregress(x_raw, y)
        r2_disp = row.R2_lin
        p_disp  = row.p_lin
        loocv_disp = row.LOOCV_RMSE_lin

    # — Cook's distance for point size / colour
    D    = cooks_distance(x_raw, y)
    high = D > 4 / n

    ax.fill_between(x_plot, ci_lo, ci_hi, alpha=0.18, color="#2196F3", label="95% CI")
    ax.plot(x_plot, y_fit, color="#D32F2F", lw=1.8,
            linestyle="--", label=f"{'log-lin' if use_log else 'linear'} fit")

    # Normal points
    ax.scatter(x_raw[~high], y[~high], color="#1565C0", s=55, zorder=5,
               edgecolors="white", linewidths=0.5)
    # Influential points
    # if high.any():
    #     ax.scatter(x_raw[high], y[high], color="#FF6F00", s=90, zorder=6,
    #                edgecolors="black", linewidths=0.8,
    #                label="Influential (Cook's D > 4/n)")

    # Significance stars
    p_fdr = row.p_BH_FDR
    stars = ("***" if p_fdr < 0.001 else
             "**"  if p_fdr < 0.01  else
             "*"   if p_fdr < 0.05  else
             "ns")
    ax.set_title(f"{metric}\n"
                 f"R²={r2_disp:.2f}, p={p_disp:.3f} ({stars})\n"
                 f"LOOCV-RMSE={loocv_disp:.3f}",
                 fontsize=14, pad=4)
    ax.set_xlabel("Species Count", fontsize=14)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=12)
    # Negative-slope fits: legend top-right; positive: top-left
    legend_loc = "upper right" if slope_b < 0 else "upper left"
    ax.legend(fontsize=10, loc=legend_loc, framealpha=0.8,
              edgecolor="gray", borderpad=0.5)

# Hide unused subplots
for ax in axes[len(metrics):]:
    ax.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
# fig.suptitle(
#     f"Spectral Metrics vs Species Count  |  Sheet: {SHEET}  |  n={n}\n"
#     f"Significance after FDR (BH) correction  •  Orange = influential point (Cook's D > 4/n)",
#     fontsize=10, y=0.98
# )
fig.suptitle(
    f"Spectral Metrics vs Species Count  |  Resolution: {SHEET}  |  n={n}\n"
    f"Significance after FDR (BH) correction ",
    fontsize=14, y=0.98
)
plt.savefig(f"spectral_diversity_regression_{SHEET}.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nFigure saved → spectral_diversity_regression.png")

# ── 10. RESIDUAL DIAGNOSTIC GRID ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(nrows, ncols,
                           figsize=(ncols * 4, nrows * 3.2),
                           constrained_layout=False)
axes2 = axes2.flatten()

for ax, metric, row in zip(axes2, metrics, res_df.itertuples()):
    y = df[metric].values
    use_log = (row.Best_model == "log-linear")
    x_model = np.log(x_raw + 1) if use_log else x_raw
    slope_b, intercept_b, *_ = linregress(x_model, y)
    resid = y - (slope_b * x_model + intercept_b)

    ax.scatter(slope_b * x_model + intercept_b, resid,
               color="#5C6BC0", edgecolors="white", s=55)
    ax.axhline(0, color="red", lw=1.2, linestyle="--")
    sw_label = f"SW p={row._asdict()['SW_p_residuals']:.3f}"
    ax.set_title(f"{metric}\n{sw_label}", fontsize=14)
    ax.set_xlabel("Fitted", fontsize=14)
    ax.set_ylabel("Residual", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=12)

for ax in axes2[len(metrics):]:
    ax.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig2.suptitle("Residual Plots (red line = zero)  |  SW p < 0.05 → non-normal residuals",
              fontsize=14, y=0.98)

plt.savefig(f"spectral_diversity_residuals_{SHEET}.png", dpi=300, bbox_inches="tight")
plt.show()
print("Residual figure saved → spectral_diversity_residuals.png")

# ── 11. EXPORT SUMMARY TABLE ──────────────────────────────────────────────────
out_csv = f"regression_summary_{SHEET}.csv"
res_df[cols_show].to_csv(out_csv, index=False)
print(f"\nSummary table saved → {out_csv}")
print("\nDone.")
