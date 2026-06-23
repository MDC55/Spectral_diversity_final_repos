from metrics_common_CV_only import run_workflow, save_all_outputs #for CV at 2cm- too big -memory issue
from spectral_workflow_2cm import process_spectral_workflow

if __name__ == "__main__":
    data_dirs = [
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/3sp/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/5sp_with_abundance/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/7sp/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/10sp_v2/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/12sp/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/15sp_with_abundance/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/17sp/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/20sp/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/22sp/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/25sp_with_abundance/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/27sp/2cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/30sp_with_abundance/2cm',
    ]

    output_root = r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/CV_2cm_results'

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