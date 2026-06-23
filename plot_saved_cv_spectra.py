# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:44:05 2026

@author: Manis
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_saved_cv_spectra(csv_file, savepath=None, show_gap=False, gap_threshold_nm=20):
    """
    Load saved CV spectra from CSV and plot spectra by species richness.

    Parameters
    ----------
    csv_file : str or Path
        Path to cv_spectra_by_richness.csv
    savepath : str or Path, optional
        Path to save the figure
    show_gap : bool, optional
        If True, insert visible gaps where wavelength jumps are large
    gap_threshold_nm : float, optional
        Threshold for detecting a masked-region jump
    """
    csv_file = Path(csv_file)
    df = pd.read_csv(csv_file)

    required_cols = {'species_richness', 'wavelength_nm', 'cv_spectrum_percent'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in CSV: {missing}')

    df = df.sort_values(['species_richness', 'wavelength_nm'])
    pivot_df = df.pivot(index='wavelength_nm', columns='species_richness', values='cv_spectrum_percent')
    print(pivot_df.head())

    plt.figure(figsize=(10, 6))

    for richness, grp in df.groupby('species_richness'):
        wl = grp['wavelength_nm'].to_numpy(dtype=float)
        cv = grp['cv_spectrum_percent'].to_numpy(dtype=float)

        if show_gap:
            wl_plot = [wl[0]]
            cv_plot = [cv[0]]

            for i in range(1, len(wl)):
                if (wl[i] - wl[i - 1]) > gap_threshold_nm:
                    wl_plot.append(np.nan)
                    cv_plot.append(np.nan)

                wl_plot.append(wl[i])
                cv_plot.append(cv[i])

            plt.plot(wl_plot, cv_plot, linewidth=1.5, label=f'{int(richness)} sp')
        else:
            plt.plot(wl, cv, linewidth=1.5, label=f'{int(richness)} sp')

    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('CV (%)', fontsize=14)
    plt.title('Saved CV Spectra by Species Richness', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Species richness', fontsize=9)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()
    return df, pivot_df 

if __name__ == '__main__':
    csv_file = r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/CV_2cm_results/csv/cv_spectra_by_richness.csv'

    df, pivot_df =plot_saved_cv_spectra(
        csv_file,
        savepath=r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/CV_2cm_results/cv_spectra_loaded_plot.png',
        show_gap=True
    )
    
# pivot_df.plot(figsize=(10, 6), linewidth=1.5)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('CV (%)')
# plt.title('Saved CV Spectra by Species Richness')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()    