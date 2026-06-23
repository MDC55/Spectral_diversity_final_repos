# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:18:35 2026

@author: Manis
"""

from metrics_common import run_workflow, save_all_outputs
from spectral_workflow_5cm import process_spectral_workflow

if __name__ == "__main__":
    data_dirs = [
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/3sp/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/5sp_with_abundance/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/7sp/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/10sp_v2/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/12sp/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/15sp_with_abundance/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/17sp/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/20sp/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/22sp/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/25sp_with_abundance/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/27sp/5cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/30sp_with_abundance/5cm',
    ]

    output_root = r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/CHV_CHA_SID_SAM_CV_5cm_results'

    results_df, global_pca, global_mean_spectrum, processed, wavelengths_nm = run_workflow(
        data_dirs=data_dirs,
        workflow_func=process_spectral_workflow,
        n_components=3,
        max_pixels_for_pca=200000,
        visualize_one=False,
        visualize_richness=30,
    )

    print('\nFinal summary results:')
    print(results_df)

    save_all_outputs(
        results_df=results_df,
        processed=processed,
        global_mean_spectrum=global_mean_spectrum,
        wavelengths_nm=wavelengths_nm,
        output_root=output_root,
    )