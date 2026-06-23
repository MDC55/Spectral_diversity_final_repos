# Spectral Diversity Final Repository

This repository contains Python scripts, analysis workflows, and results for spectral diversity metric calculations across multiple spatial resolutions.

## Project overview

The project includes:
- `AE_global_*` for autoencoder-based spectral analysis.
- `spectral_workflow_*` scripts for processing spectral data by resolution.
- `run_metrics_*` scripts for computing diversity metrics across resolutions.
- `metrics_common.py`, `metrics_common_CV_only.py`, and `metrics_common_without_CV.py` for common metric functions and CV-specific calculations.
- `plot_saved_cv_spectra.py` for visualizing spectral results.
- `Metrics results/` and `Spectral diversity result/` directories with CSV outputs, plots, and analysis summary files.

## Folder structure

- `AE_global_*.py`: autoencoder experiments for different resolutions
- `spectral_workflow_*.py`: spectral processing pipelines
- `run_metrics_*.py`: metric generation scripts
- `metrics_*`: shared metric code and CV variants
- `Metrics results/`: computed metrics, plots, and generated spectrum outputs
- `Spectral diversity result/`: summary analysis and regression result files
- `Notes.txt`: project-specific notes

## Requirements

This project uses Python and the following packages:

- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- tensorflow

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate the environment:

```powershell
.\.venv\Scripts\Activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Example usage

Run a metric computation script:

```powershell
python run_metrics_5cm.py
```

Run a workflow or autoencoder experiment:

```powershell
python spectral_workflow_10cm.py
python AE_global_10cm.py
```

Analyze the spectral diversity results:

```powershell
python "Spectral diversity result\cv_band_analysis.py"
```

<img src="Metrics results/AE_CHV.jpg" alt="Workflow Diagram" width="600">

- Owner: [MDC55](https://github.com/MDC55)
