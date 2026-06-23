from metrics_common import run_global_chv_and_cha_workflow, save_all_outputs
from spectral_workflow_2cm import processspectralworkflow as workflow_2cm
from spectral_workflow_5cm import processspectralworkflow as workflow_5cm
from spectral_workflow_10cm import processspectralworkflow as workflow_10cm
from spectral_workflow_50cm import processspectralworkflow as workflow_50cm
from spectral_workflow_1m import processspectralworkflow as workflow_1m

WORKFLOW_MAP = {
    "2cm": workflow_2cm,
    "5cm": workflow_5cm,
    "10cm": workflow_10cm,
    "50cm": workflow_50cm,
    "1m": workflow_1m,
}

def build_data_dirs(resolution):
    base = r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species'
    return [
        fr'{base}/3sp/{resolution}',
        fr'{base}/5sp_with_abundance/{resolution}',
        fr'{base}/7sp/{resolution}',
        fr'{base}/10sp_v2/{resolution}',
        fr'{base}/12sp/{resolution}',
        fr'{base}/15sp_with_abundance/{resolution}',
        fr'{base}/17sp/{resolution}',
        fr'{base}/20sp/{resolution}',
        fr'{base}/22sp/{resolution}',
        fr'{base}/25sp_with_abundance/{resolution}',
        fr'{base}/27sp/{resolution}',
        fr'{base}/30sp_with_abundance/{resolution}',
    ]

if __name__ == "__main__":
    resolution = "5cm"

    workflow_func = WORKFLOW_MAP[resolution]
    data_dirs = build_data_dirs(resolution)
    output_root = fr'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/CHV_CHA_SID_SAM_{resolution}_results'

    results_df, global_pca, global_mean_spectrum, processed, wavelengths_nm = run_global_chv_and_cha_workflow(
        data_dirs=data_dirs,
        workflow_func=workflow_func,
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