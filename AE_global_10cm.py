# -*- coding: utf-8 -*-
"""
Global autoencoder workflow for spectral diversity analysis.

This script replaces per-image autoencoders with one shared autoencoder trained
on pooled pixels from all species-richness images, analogous to global PCA.
Each image is then projected into the same latent space and spectral diversity
metrics are computed in that shared space.

Adapted from the user's Auto_encoder.py, spectral_workflow_5cm.py,
and CHV_5cm-2.py structure.
"""

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spectral_workflow_10cm import process_spectral_workflow


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def extract_species_richness(data_dir):
    parts = data_dir.replace("\\", "/").split("/")
    for part in parts[::-1]:
        match = re.search(r"(\d+)\s*sp", part.lower())
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not extract species richness from path: {data_dir}")


def clean_roi_for_autoencoder(roi_ref, wl):
    if isinstance(roi_ref, list):
        roi_cube = np.array(roi_ref[0], dtype=np.float64)
    else:
        roi_cube = np.array(roi_ref, dtype=np.float64)

    if roi_cube.ndim == 4 and roi_cube.shape[0] == 1:
        roi_cube = np.squeeze(roi_cube, axis=0)

    wl_nm = 1000 * np.array(wl, dtype=np.float64)

    roi_cube[:, :, (1350 <= wl_nm) & (wl_nm <= 1500)] = np.nan
    roi_cube[:, :, (1790 <= wl_nm) & (wl_nm <= 1995)] = np.nan
    roi_cube[:, :, (wl_nm > 2475)] = np.nan

    valid_band_mask = ~np.all(np.isnan(roi_cube), axis=(0, 1))
    roi_cube_clean = roi_cube[:, :, valid_band_mask]
    wl_nm_clean = wl_nm[valid_band_mask]

    valid_pixel_mask = ~np.any(np.isnan(roi_cube_clean), axis=2)
    valid_pixels = roi_cube_clean[valid_pixel_mask, :]

    return roi_cube_clean, wl_nm_clean, valid_pixels, valid_pixel_mask


def subsample_pixels(pixel_matrix, max_pixels=None, random_state=42):
    if max_pixels is None or pixel_matrix.shape[0] <= max_pixels:
        return pixel_matrix
    rng = np.random.default_rng(random_state)
    idx = rng.choice(pixel_matrix.shape[0], size=max_pixels, replace=False)
    return pixel_matrix[idx]


def build_autoencoder(input_dim, latent_dim=3):
    input_layer = Input(shape=(input_dim,))

    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(latent_dim)(x)
    encoded = LeakyReLU(negative_slope=0.1, name='latent_layer')(x)

    x = Dense(64, activation='relu')(encoded)
    x = Dense(128, activation='relu')(x)
    decoded = Dense(input_dim, activation='sigmoid')(x)

    autoencoder = Model(input_layer, decoded, name='global_autoencoder')
    encoder = Model(input_layer, encoded, name='global_encoder')

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def fit_global_autoencoder(pixel_matrices,
                           latent_dim=3,
                           max_pixels_for_training=200000,
                           epochs=50,
                           batch_size=128,
                           validation_split=0.1,
                           random_state=42):
    all_pixels = np.vstack(pixel_matrices)
    fit_pixels = subsample_pixels(all_pixels, max_pixels=max_pixels_for_training, random_state=random_state)

    scaler = MinMaxScaler()
    fit_pixels_scaled = scaler.fit_transform(fit_pixels)

    autoencoder, encoder = build_autoencoder(fit_pixels_scaled.shape[1], latent_dim=latent_dim)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    history = autoencoder.fit(
        fit_pixels_scaled,
        fit_pixels_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stopping]
    )

    return autoencoder, encoder, scaler, history, fit_pixels_scaled


def compute_latent_metrics(pixel_matrix, encoder, scaler, max_pixels_for_distance=50000, random_state=42):
    scaled = scaler.transform(pixel_matrix)
    latent = encoder.predict(scaled, verbose=0)
    latent_unique = np.unique(latent, axis=0)

    reconstruction = None
    latent_variance = np.var(latent, axis=0)
    latent_variance_sum = np.sum(latent_variance)

    if latent_unique.shape[0] >= 4:
        try:
            hull = ConvexHull(latent_unique)
            chv = hull.volume
        except QhullError:
            chv = np.nan
    else:
        chv = np.nan

    if latent.shape[0] > 1:
        sampled_latent = subsample_pixels(latent, max_pixels=max_pixels_for_distance, random_state=random_state)
        mean_pairwise_distance = np.mean(pdist(sampled_latent)) if sampled_latent.shape[0] > 1 else np.nan
    else:
        mean_pairwise_distance = np.nan

    return {
        'latent': latent,
        'latent_unique': latent_unique,
        'CHV': chv,
        'latent_variance': latent_variance,
        'latent_variance_sum': latent_variance_sum,
        'mean_pairwise_distance': mean_pairwise_distance,
        'scaled_pixels': scaled,
        'reconstruction': reconstruction
    }


def compute_reconstruction_errors(pixel_matrix, autoencoder, scaler):
    scaled = scaler.transform(pixel_matrix)
    reconstructed = autoencoder.predict(scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(scaled - reconstructed), axis=1)
    return reconstruction_errors, reconstructed


def plot_training_history(history, output_dir=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Global Autoencoder Training Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'global_autoencoder_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.title('Global Autoencoder Training vs Validation Loss (Log Scale)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'global_autoencoder_loss_logscale.png'), dpi=300, bbox_inches='tight')
    plt.show()

    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    loss_gap = val_loss - train_loss

    plt.figure(figsize=(8, 5))
    plt.plot(loss_gap, label='Validation - Training Loss Gap')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('Global Autoencoder Training–Validation Loss Gap')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Gap (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'global_autoencoder_loss_gap.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_chv_vs_species_richness(results_df, output_dir=None):
    df = results_df.sort_values('species_richness')
    plt.figure(figsize=(8, 6))
    plt.plot(df['species_richness'], df['CHV'], marker='o', linewidth=2)
    plt.xlabel('Species richness', fontsize=14)
    plt.ylabel('CHV (global autoencoder latent space)', fontsize=14)
    plt.title('CHV vs Species Richness', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'global_autoencoder_chv_vs_richness.png'), dpi=300, bbox_inches='tight')
    plt.show()

def visualize_ae_chv_3d(latent_points,
                        title='Convex Hull in Global Autoencoder Space',
                        max_points_to_plot=8000,
                        random_state=42,
                        save_path=None):
    if latent_points is None or latent_points.ndim != 2 or latent_points.shape[1] != 3:
        print('Visualization skipped: latent_points must have shape (n_samples, 3).')
        return

    latent_unique = np.unique(latent_points, axis=0)

    if latent_unique.shape[0] < 4:
        print('Visualization skipped: need at least 4 unique 3D points.')
        return

    try:
        hull = ConvexHull(latent_unique)
    except QhullError as e:
        print(f'Convex hull failed: {e}')
        return

    plot_points = latent_unique
    if max_points_to_plot is not None and latent_unique.shape[0] > max_points_to_plot:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(latent_unique.shape[0], size=max_points_to_plot, replace=False)
        plot_points = latent_unique[idx]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        s=3,
        c='royalblue',
        alpha=0.18,
        label='Latent points'
    )

    for simplex in hull.simplices:
        simplex_closed = np.append(simplex, simplex[0])
        ax.plot(
            latent_unique[simplex_closed, 0],
            latent_unique[simplex_closed, 1],
            latent_unique[simplex_closed, 2],
            color='black',
            linewidth=0.5
        )

    hull_faces = [latent_unique[simplex] for simplex in hull.simplices]
    hull_poly = Poly3DCollection(hull_faces, alpha=0.15, facecolor='cyan', edgecolor='none')
    ax.add_collection3d(hull_poly)

    ax.set_xlabel('Latent Dimension 1', fontsize=13)
    ax.set_ylabel('Latent Dimension 2', fontsize=13)
    ax.set_zlabel('Latent Dimension 3', fontsize=13)
    ax.set_title(f'{title}\nCHV = {hull.volume:.4f}', fontsize=15)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_richness_hull(per_image_outputs,
                            richness,
                            output_dir=None,
                            max_points_to_plot=8000,
                            random_state=42):
    for item in per_image_outputs:
        if item['species_richness'] == richness:
            save_path = None
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f'ae_convex_hull_{richness}sp.png')

            visualize_ae_chv_3d(
                item['latent'],
                title=f'Global Autoencoder CHV | richness = {richness}',
                max_points_to_plot=max_points_to_plot,
                random_state=random_state,
                save_path=save_path
            )
            return

    print(f'No entry found for species richness = {richness}')
    
    
# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
def run_global_autoencoder_workflow(data_dirs,
                                    latent_dim=3,
                                    max_pixels_for_training=200000,
                                    max_pixels_for_distance=50000,
                                    epochs=50,
                                    batch_size=128,
                                    validation_split=0.1,
                                    random_state=42,
                                    output_dir=None,
                                    visualize_one=False,
                                    visualize_richness=None):
    set_seed(random_state)

    processed = []

    for data_dir in data_dirs:
        richness = extract_species_richness(data_dir)
        print(f'\nProcessing species richness = {richness}')
        print(data_dir)

        result = process_spectral_workflow(
            data_dir,
            plot_rgb=False,
            plot_pixel_spectrum=False,
            plot_gt=False,
            plot_roi=False,
            plot_roi_mean_spectra=False,
            compute_snr=False,
            return_gt=False
        )

        roi_ref = result['roi_Ref']
        wl = result['wl']

        roi_cube_clean, wl_nm_clean, pixel_matrix, valid_pixel_mask = clean_roi_for_autoencoder(roi_ref, wl)

        print(f'Clean ROI shape: {roi_cube_clean.shape}')
        print(f'Valid pixels: {pixel_matrix.shape[0]}')
        print(f'Valid bands: {pixel_matrix.shape[1]}')
        print(f'NaN count in pixel_matrix: {np.isnan(pixel_matrix).sum()}')
        print(f'Inf count in pixel_matrix: {np.isinf(pixel_matrix).sum()}')
        print(f'Max abs reflectance: {np.nanmax(np.abs(pixel_matrix)) if pixel_matrix.size > 0 else np.nan}')

        processed.append({
            'species_richness': richness,
            'data_dir': data_dir,
            'roi_cube_clean': roi_cube_clean,
            'wl_nm_clean': wl_nm_clean,
            'pixel_matrix': pixel_matrix,
            'valid_pixel_mask': valid_pixel_mask
        })

    reference_wl = processed[0]['wl_nm_clean']
    for item in processed[1:]:
        if len(item['wl_nm_clean']) != len(reference_wl) or not np.allclose(item['wl_nm_clean'], reference_wl):
            raise ValueError('Cleaned wavelengths are not identical across all images.')

    autoencoder, encoder, scaler, history, fit_pixels_scaled = fit_global_autoencoder(
        [item['pixel_matrix'] for item in processed],
        latent_dim=latent_dim,
        max_pixels_for_training=max_pixels_for_training,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        random_state=random_state
    )

    print('\nGlobal autoencoder fitted successfully.')
    print('Latent dimension:', latent_dim)
    print('Training pixels used:', fit_pixels_scaled.shape[0])
    print('Bands used:', fit_pixels_scaled.shape[1])

    results = []
    per_image_outputs = []

    for item in processed:
        metrics = compute_latent_metrics(
            item['pixel_matrix'],
            encoder,
            scaler,
            max_pixels_for_distance=max_pixels_for_distance,
            random_state=random_state
        )

        reconstruction_errors, reconstructed = compute_reconstruction_errors(
            item['pixel_matrix'],
            autoencoder,
            scaler
        )

        metrics['reconstruction'] = reconstructed
        mean_reconstruction_error = np.mean(reconstruction_errors)

        results.append({
            'species_richness': item['species_richness'],
            'data_dir': item['data_dir'],
            'n_valid_pixels': item['pixel_matrix'].shape[0],
            'n_valid_bands': item['pixel_matrix'].shape[1],
            'CHV': metrics['CHV'],
            'latent_variance_sum': metrics['latent_variance_sum'],
            'mean_pairwise_distance': metrics['mean_pairwise_distance'],
            'mean_reconstruction_error': mean_reconstruction_error
        })

        per_image_outputs.append({
            **item,
            **metrics,
            'reconstruction_errors': reconstruction_errors
        })

        print(
            f"Richness {item['species_richness']:>2} --> "
            f"CHV = {metrics['CHV']}, "
            f"LatVarSum = {metrics['latent_variance_sum']}, "
            f"MeanDist = {metrics['mean_pairwise_distance']}, "
            f"ReconErr = {mean_reconstruction_error}"
        )

    results_df = pd.DataFrame(results).sort_values('species_richness').reset_index(drop=True)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'global_autoencoder_metrics.csv'), index=False)

    plot_training_history(history, output_dir=output_dir)
    plot_chv_vs_species_richness(results_df, output_dir=output_dir)

    if visualize_one:
        if visualize_richness is None and len(per_image_outputs) > 0:
            visualize_richness = per_image_outputs[-1]['species_richness']
        visualize_richness_hull(
            per_image_outputs,
            richness=visualize_richness,
            output_dir=output_dir,
            max_points_to_plot=8000,
            random_state=random_state
        )

    return results_df, autoencoder, encoder, scaler, history, processed, per_image_outputs


# ---------------------------------------------------------
# Run example
# ---------------------------------------------------------

if __name__ == "__main__":

    data_dirs = [
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/3sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/5sp_with_abundance/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/7sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/10sp_v2/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/12sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/15sp_with_abundance/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/17sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/20sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/22sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/25sp_with_abundance/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/27sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/30sp_with_abundance/10cm',
    ]

    output_dir = r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/global_autoencoder_results_10cm'

    results_df, autoencoder, encoder, scaler, history, processed, per_image_outputs = run_global_autoencoder_workflow(
        data_dirs=data_dirs,
        latent_dim=3,
        max_pixels_for_training=200000,
        max_pixels_for_distance=50000,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        random_state=42,
        output_dir=output_dir,
        visualize_one=True,
        visualize_richness=30
    )

    print('\nFinal global autoencoder results:')
    print(results_df)
    


#%%
'''
def run_global_autoencoder_workflow(data_dirs,
                                    latent_dim=3,
                                    max_pixels_for_training=200000,
                                    max_pixels_for_distance=50000,
                                    epochs=50,
                                    batch_size=128,
                                    validation_split=0.1,
                                    random_state=42,
                                    output_dir=None):
    set_seed(random_state)

    processed = []

    for data_dir in data_dirs:
        richness = extract_species_richness(data_dir)
        print(f"\nProcessing species richness = {richness}")
        print(data_dir)

        result = process_spectral_workflow(
            data_dir,
            plot_rgb=False,
            plot_pixel_spectrum=False,
            plot_gt=False,
            plot_roi=False,
            plot_roi_mean_spectra=False,
            compute_snr=False,
            return_gt=False
        )

        roi_ref = result['roi_Ref']
        wl = result['wl']

        roi_cube_clean, wl_nm_clean, pixel_matrix, valid_pixel_mask = clean_roi_for_autoencoder(roi_ref, wl)

        print(f"Clean ROI shape: {roi_cube_clean.shape}")
        print(f"Valid pixels: {pixel_matrix.shape[0]}")
        print(f"Valid bands: {pixel_matrix.shape[1]}")

        processed.append({
            'species_richness': richness,
            'data_dir': data_dir,
            'roi_cube_clean': roi_cube_clean,
            'wl_nm_clean': wl_nm_clean,
            'pixel_matrix': pixel_matrix,
            'valid_pixel_mask': valid_pixel_mask
        })

    reference_wl = processed[0]['wl_nm_clean']
    for item in processed[1:]:
        if len(item['wl_nm_clean']) != len(reference_wl) or not np.allclose(item['wl_nm_clean'], reference_wl):
            raise ValueError('Cleaned wavelengths are not identical across all images.')

    autoencoder, encoder, scaler, history, fit_pixels_scaled = fit_global_autoencoder(
        [item['pixel_matrix'] for item in processed],
        latent_dim=latent_dim,
        max_pixels_for_training=max_pixels_for_training,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        random_state=random_state
    )

    print('\nGlobal autoencoder fitted successfully.')
    print('Latent dimension:', latent_dim)
    print('Training pixels used:', fit_pixels_scaled.shape[0])
    print('Bands used:', fit_pixels_scaled.shape[1])

    results = []
    per_image_outputs = []

    for item in processed:
        metrics = compute_latent_metrics(
            item['pixel_matrix'],
            encoder,
            scaler,
            max_pixels_for_distance=max_pixels_for_distance,
            random_state=random_state
        )

        reconstruction_errors, reconstructed = compute_reconstruction_errors(
            item['pixel_matrix'],
            autoencoder,
            scaler
        )

        metrics['reconstruction'] = reconstructed
        mean_reconstruction_error = np.mean(reconstruction_errors)

        results.append({
            'species_richness': item['species_richness'],
            'data_dir': item['data_dir'],
            'n_valid_pixels': item['pixel_matrix'].shape[0],
            'n_valid_bands': item['pixel_matrix'].shape[1],
            'CHV': metrics['CHV'],
            'latent_variance_sum': metrics['latent_variance_sum'],
            'mean_pairwise_distance': metrics['mean_pairwise_distance'],
            'mean_reconstruction_error': mean_reconstruction_error
        })

        per_image_outputs.append({
            **item,
            **metrics,
            'reconstruction_errors': reconstruction_errors
        })

        print(
            f"Richness {item['species_richness']:>2} --> "
            f"CHV = {metrics['CHV']}, "
            f"LatVarSum = {metrics['latent_variance_sum']}, "
            f"MeanDist = {metrics['mean_pairwise_distance']}, "
            f"ReconErr = {mean_reconstruction_error}"
        )

    results_df = pd.DataFrame(results).sort_values('species_richness').reset_index(drop=True)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'global_autoencoder_metrics.csv'), index=False)

    plot_training_history(history, output_dir=output_dir)
    plot_chv_vs_species_richness(results_df, output_dir=output_dir)

    return results_df, autoencoder, encoder, scaler, history, processed, per_image_outputs


if __name__ == "__main__":

    data_dirs = [
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/3sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/5sp_with_abundance/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/7sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/10sp_v2/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/12sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/15sp_with_abundance/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/17sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/20sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/22sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/25sp_with_abundance/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/27sp/10cm',
        r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/30sp_with_abundance/10cm',
    ]

    output_dir = r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/global_autoencoder_results_10cm'

    results_df, autoencoder, encoder, scaler, history, processed, per_image_outputs = run_global_autoencoder_workflow(
        data_dirs=data_dirs,
        latent_dim=3,
        max_pixels_for_training=200000,
        max_pixels_for_distance=50000,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        random_state=42,
        output_dir=output_dir
    )

    print('\nFinal global autoencoder results:')
    print(results_df)
'''