import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.spatial import ConvexHull, QhullError
from scipy.stats import entropy
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def extract_species_richness(data_dir):
    parts = data_dir.replace('\\', '/').split('/')
    for part in parts[::-1]:
        match = re.search(r'(\d+)\s*sp', part.lower())
        if match:
            return int(match.group(1))
    raise ValueError(f'Could not extract species richness from path: {data_dir}')


def clean_roi_for_metrics(roi_ref, wl):
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

    # remove bands that are completely NaN
    valid_band_mask = ~np.all(np.isnan(roi_cube), axis=(0, 1))
    roi_cube_clean = roi_cube[:, :, valid_band_mask]
    wl_nm_clean = wl_nm[valid_band_mask]

    # remove pixels containing NaN in any remaining band
    valid_pixel_mask = ~np.any(np.isnan(roi_cube_clean), axis=2)
    valid_pixels = roi_cube_clean[valid_pixel_mask, :]

    return roi_cube_clean, wl_nm_clean, valid_pixels, valid_pixel_mask


def fit_global_pca(pixel_matrices, n_components=3, max_pixels=200000, random_state=42):
    all_pixels = np.vstack(pixel_matrices)

    if max_pixels is not None and all_pixels.shape[0] > max_pixels:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(all_pixels.shape[0], size=max_pixels, replace=False)
        fit_pixels = all_pixels[idx]
    else:
        fit_pixels = all_pixels

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(fit_pixels)
    return pca


def compute_chv_in_global_pca(pixel_matrix, global_pca):
    if pixel_matrix.shape[0] < 4:
        return np.nan, None

    transformed = global_pca.transform(pixel_matrix)
    transformed_unique = np.unique(transformed, axis=0)

    if transformed_unique.shape[0] < 4:
        return np.nan, transformed_unique

    try:
        hull = ConvexHull(transformed_unique)
        return hull.volume, transformed_unique
    except QhullError:
        return np.nan, transformed_unique


def compute_cha_values(pixel_matrix, reference_spectrum):
    cha_values = []

    for pixel_spectrum in pixel_matrix:
        points_2d = np.column_stack((reference_spectrum, pixel_spectrum))
        try:
            hull = ConvexHull(points_2d)
            cha_values.append(hull.area)
        except QhullError:
            cha_values.append(np.nan)

    return np.array(cha_values, dtype=np.float64)


def visualize_chv(pixel_matrix, n_components=3, save_path=None):
    if pixel_matrix.shape[0] <= n_components:
        print('Not enough valid data points for CHV visualization.')
        return None

    pca = PCA(n_components=n_components)
    transformed_pixels = pca.fit_transform(pixel_matrix)

    try:
        hull = ConvexHull(transformed_pixels)
    except QhullError:
        print('Convex hull failed for CHV visualization.')
        return None

    volume = hull.volume

    if n_components == 3:
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            transformed_pixels[:, 0], transformed_pixels[:, 1], transformed_pixels[:, 2],
            s=2, c='b', alpha=0.3, label='Latent Points'
        )

        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(
                transformed_pixels[simplex, 0],
                transformed_pixels[simplex, 1],
                transformed_pixels[simplex, 2],
                'k-', lw=0.5
            )

        hull_faces = [transformed_pixels[simplex] for simplex in hull.simplices]
        hull_poly = Poly3DCollection(hull_faces, alpha=0.2, color='cyan')
        ax.add_collection3d(hull_poly)

        ax.set_xlabel('PC1', fontsize=16, labelpad=12)
        ax.set_ylabel('PC2', fontsize=16, labelpad=12)
        ax.set_zlabel('PC3', fontsize=16, labelpad=-35)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='z', which='major', labelsize=16)
        ax.set_title(f'3D Convex Hull in PCA Space \nCHV = {volume:.4f}', fontsize=18)
        ax.view_init(elev=30, azim=45)
        plt.subplots_adjust(left=0.25, right=0.99, bottom=0.05, top=0.85)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

    return volume


def visualize_cha(pixel_matrix, mean_spectrum, save_path=None, pixel_idx=1133):
    n_pixels, n_bands = pixel_matrix.shape
    if n_pixels == 0:
        print('No valid pixels available for CHA visualization.')
        return None

    idx = min(pixel_idx, n_pixels - 1)
    pixel_spectrum = pixel_matrix[idx, :]
    points_2d = np.column_stack((mean_spectrum, pixel_spectrum))

    if np.linalg.matrix_rank(points_2d) < 2:
        print(f'Pixel {idx} is degenerate and skipped.')
        return None

    try:
        hull = ConvexHull(points_2d)
    except QhullError:
        print(f'Convex hull failed for CHA visualization at pixel {idx}.')
        return None

    plt.figure(figsize=(6, 5))
    plt.scatter(mean_spectrum, pixel_spectrum, color='blue', label='Bands')
    plt.plot(points_2d[hull.vertices, 0], points_2d[hull.vertices, 1], 'r-', lw=1.5, label='Convex Hull')
    plt.fill(points_2d[hull.vertices, 0], points_2d[hull.vertices, 1], color='red', alpha=0.2)
    plt.title(f'Pixel {idx} - CHA = {hull.area:.4f}', fontsize=18)
    plt.xlabel('Mean Spectrum Reflectance', fontsize=18)
    plt.ylabel('Pixel Spectrum Reflectance', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    return hull.area



def compute_sam_values(pixel_matrix, reference_spectrum, epsilon=1e-12):
    """Compute Spectral Angle Mapper (SAM) for each pixel"""
    pixels = np.asarray(pixel_matrix, dtype=np.float64)
    ref = np.asarray(reference_spectrum, dtype=np.float64)

    pixel_norms = np.linalg.norm(pixels, axis=1)
    ref_norm = np.linalg.norm(ref)
    denom = np.maximum(pixel_norms * ref_norm, epsilon)
    dots = np.dot(pixels, ref)
    cos_ang = np.clip(dots / denom, -1.0, 1.0)

    return np.arccos(cos_ang)



def compute_sid_values(pixel_matrix, reference_spectrum, epsilon=1e-12):
    """Compute Spectral Information Divergence (SID) for each pixel"""
    pixels = np.asarray(pixel_matrix, dtype=np.float64)
    ref = np.asarray(reference_spectrum, dtype=np.float64)

    # Clip to avoid log(0)
    pixels = np.clip(pixels, epsilon, None)
    ref = np.clip(ref, epsilon, None)

    # Normalize to probability distributions
    p = pixels / np.sum(pixels, axis=1, keepdims=True)
    q = ref / np.sum(ref)

    # Expand q to 2D so shapes match exactly
    q_row = q[np.newaxis, :]              # shape: (1, bands)
    q_tiled = np.repeat(q_row, p.shape[0], axis=0)   # shape: (n_pixels, bands)

    # SID = KL(p || q) + KL(q || p), computed per pixel along bands axis
    sid_values = entropy(p, q_tiled, axis=1) + entropy(q_tiled, p, axis=1)

    return np.asarray(sid_values, dtype=np.float64)


def summarize_metric(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n': 0,
        }
    return {
        'mean': float(np.nanmean(values)),
        'median': float(np.nanmedian(values)),
        'std': float(np.nanstd(values)),
        'min': float(np.nanmin(values)),
        'max': float(np.nanmax(values)),
        'n': int(values.size),
    }


def ensure_output_dirs(base_dir):
    """Create output directory structure"""
    base_dir = Path(base_dir)
    plots_dir = base_dir / 'plots'
    csv_dir = base_dir / 'csv'
    spectra_dir = base_dir / 'spectra'

    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, plots_dir, csv_dir, spectra_dir


def save_metric_values(processed, csv_dir):
    """Save per-pixel metric values to CSV"""
    rows = []
    for item in sorted(processed, key=lambda x: x['species_richness']):
        n = max(
            len(item['cha_local_values']),
            len(item['cha_global_values']),
            len(item['sam_local_values']),
            len(item['sam_global_values']),
            len(item['sid_local_values']),
            len(item['sid_global_values'])
        )
        for i in range(n):
            rows.append({
                'species_richness': item['species_richness'],
                'pixel_index': i,
                'CHA_local': item['cha_local_values'][i] if i < len(item['cha_local_values']) else np.nan,
                'CHA_global': item['cha_global_values'][i] if i < len(item['cha_global_values']) else np.nan,
                'SAM_local_rad': item['sam_local_values'][i] if i < len(item['sam_local_values']) else np.nan,
                'SAM_global_rad': item['sam_global_values'][i] if i < len(item['sam_global_values']) else np.nan,
                'SAM_local_deg': np.degrees(item['sam_local_values'][i]) if i < len(item['sam_local_values']) else np.nan,
                'SAM_global_deg': np.degrees(item['sam_global_values'][i]) if i < len(item['sam_global_values']) else np.nan,
                'SID_local': item['sid_local_values'][i] if i < len(item['sid_local_values']) else np.nan,
                'SID_global': item['sid_global_values'][i] if i < len(item['sid_global_values']) else np.nan,
            })

    pd.DataFrame(rows).to_csv(Path(csv_dir) / 'metric_values_by_pixel.csv', index=False)


def save_global_spectrum(global_mean_spectrum, wavelengths_nm, csv_dir):
    """Save global mean spectrum to CSV"""
    pd.DataFrame({
        'wavelength_nm': wavelengths_nm,
        'global_mean_spectrum': global_mean_spectrum
    }).to_csv(Path(csv_dir) / 'global_mean_spectrum.csv', index=False)


def save_local_mean_spectra(processed, wavelengths_nm, csv_dir):
    """Save local mean spectra for each richness level to CSV"""
    rows = []
    for item in sorted(processed, key=lambda x: x['species_richness']):
        for wl, val in zip(wavelengths_nm, item['local_mean_spectrum']):
            rows.append({
                'species_richness': item['species_richness'],
                'wavelength_nm': wl,
                'local_mean_spectrum': val
            })

    pd.DataFrame(rows).to_csv(Path(csv_dir) / 'local_mean_spectra_by_richness.csv', index=False)


def plot_metric_comparison(results_df, y_local, y_global, ylabel, title, savepath=None, to_degrees=False):
    """Generic comparison plot for local vs global metrics"""
    df = results_df.sort_values('species_richness').copy()
    y1 = df[y_local].to_numpy()
    y2 = df[y_global].to_numpy()

    if to_degrees:
        y1 = np.degrees(y1)
        y2 = np.degrees(y2)

    plt.figure(figsize=(9, 6))
    plt.plot(df['species_richness'], y1, marker='o', linewidth=2, label=f'Local-mean {ylabel}')
    plt.plot(df['species_richness'], y2, marker='s', linewidth=2, label=f'Global-mean {ylabel}')
    plt.xlabel('Species richness', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_chv(results_df, savepath=None):
    """Plot CHV vs species richness"""
    df = results_df.sort_values('species_richness')

    plt.figure(figsize=(9, 6))
    plt.plot(df['species_richness'], df['CHV'], marker='o', linewidth=2, color='teal')
    plt.xlabel('Species richness', fontsize=14)
    plt.ylabel('CHV (global PCA space)', fontsize=14)
    plt.title('CHV vs Species Richness', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_distribution_grid(processed, metric_key, title_prefix, xlabel, savepath=None, transform=None):
    """Plot distribution histograms in a grid layout"""
    processed_sorted = sorted(processed, key=lambda x: x['species_richness'])
    n = len(processed_sorted)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, item in zip(axes, processed_sorted):
        vals = np.asarray(item[metric_key], dtype=np.float64)
        vals = vals[np.isfinite(vals)]

        if transform is not None:
            vals = transform(vals)

        ax.hist(vals, bins=40, color='steelblue', edgecolor='black', alpha=0.75)
        ax.set_title(f"{item['species_richness']}sp")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Pixel count')
        ax.grid(True, alpha=0.2)

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_mean_spectra(processed, wavelengths_nm, global_mean_spectrum, savepath=None):
    """Plot local mean spectra and global mean spectrum"""
    plt.figure(figsize=(10, 6))

    for item in sorted(processed, key=lambda x: x['species_richness']):
        plt.plot(
            wavelengths_nm, 
            item['local_mean_spectrum'], 
            linewidth=1.2, 
            alpha=0.8, 
            label=f"{item['species_richness']}sp local mean"
        )

    plt.plot(wavelengths_nm, global_mean_spectrum, color='black', linewidth=3, label='Global mean spectrum')
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Reflectance', fontsize=14)
    plt.title('Local Mean Spectra and Global Mean Spectrum', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def save_all_outputs(results_df, processed, global_mean_spectrum, wavelengths_nm, output_root):
    """Save all outputs to organized folder structure"""
    base_dir, plots_dir, csv_dir, spectra_dir = ensure_output_dirs(output_root)

    # Save CSVs
    results_df.to_csv(csv_dir / 'chv_cha_sam_sid_summary.csv', index=False)
    save_global_spectrum(global_mean_spectrum, wavelengths_nm, csv_dir)
    save_local_mean_spectra(processed, wavelengths_nm, csv_dir)
    save_metric_values(processed, csv_dir)

    # Save plots - CHV and metric comparisons
    plot_chv(results_df, savepath=plots_dir / 'chv_vs_species_richness.png')
    
    
    
    
    if len(processed) > 0:
        processed_sorted = sorted(processed, key=lambda x: x['species_richness'])
        chv_item = processed_sorted[-1] #second_chv_item = processed_sorted[1]
        visualize_chv(
            chv_item['pixel_matrix'],
            n_components=3,
            save_path=plots_dir / f"chv_visualization_{chv_item['species_richness']}sp.png"
        )

        cha_item = processed_sorted[-1]
        visualize_cha(
           cha_item['pixel_matrix'],
           cha_item['local_mean_spectrum'],
           save_path=plots_dir / f"cha_visualization_{cha_item['species_richness']}sp.png"
        )
    
    plot_metric_comparison(
        results_df, 'CHA_local_mean', 'CHA_global_mean', 'CHA', 
        'Local vs Global-Mean CHA', 
        savepath=plots_dir / 'cha_local_vs_global.png'
    )

    plot_metric_comparison(
        results_df, 'SAM_local_mean', 'SAM_global_mean', 'SAM (degrees)', 
        'Local vs Global-Mean SAM', 
        savepath=plots_dir / 'sam_local_vs_global_degrees.png',
        to_degrees=True
    )

    plot_metric_comparison(
        results_df, 'SID_local_mean', 'SID_global_mean', 'SID', 
        'Local vs Global-Mean SID', 
        savepath=plots_dir / 'sid_local_vs_global.png'
    )

    # Save distribution plots
    plot_distribution_grid(
        processed, 'cha_local_values', 'Local-Mean CHA Distribution', 'CHA', 
        savepath=plots_dir / 'cha_local_distribution.png'
    )

    plot_distribution_grid(
        processed, 'cha_global_values', 'Global-Mean CHA Distribution', 'CHA', 
        savepath=plots_dir / 'cha_global_distribution.png'
    )

    plot_distribution_grid(
        processed, 'sam_local_values', 'Local-Mean SAM Distribution', 'SAM (degrees)', 
        savepath=plots_dir / 'sam_local_distribution_degrees.png',
        transform=np.degrees
    )

    plot_distribution_grid(
        processed, 'sam_global_values', 'Global-Mean SAM Distribution', 'SAM (degrees)', 
        savepath=plots_dir / 'sam_global_distribution_degrees.png',
        transform=np.degrees
    )

    plot_distribution_grid(
        processed, 'sid_local_values', 'Local-Mean SID Distribution', 'SID', 
        savepath=plots_dir / 'sid_local_distribution.png'
    )

    plot_distribution_grid(
        processed, 'sid_global_values', 'Global-Mean SID Distribution', 'SID', 
        savepath=plots_dir / 'sid_global_distribution.png'
    )

    # Save spectra plot
    plot_mean_spectra(
        processed, wavelengths_nm, global_mean_spectrum, 
        savepath=spectra_dir / 'local_and_global_mean_spectra.png'
    )

    print(f'\nAll outputs saved in: {base_dir}')
    print(f'CSV folder: {csv_dir}')
    print(f'Plots folder: {plots_dir}')
    print(f'Spectra folder: {spectra_dir}')


def run_workflow(
    data_dirs,
    workflow_func,
    n_components=3,
    max_pixels_for_pca=200000,
    visualize_one=False,
    visualize_richness=None
):
    processed = []

    for data_dir in data_dirs:
        richness = extract_species_richness(data_dir)
        print(f'\nProcessing species richness = {richness}')
        print(data_dir)

        result = workflow_func(
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

        roi_cube_clean, wl_nm_clean, pixel_matrix, valid_pixel_mask = clean_roi_for_metrics(roi_ref, wl)

        print(f'Clean ROI shape: {roi_cube_clean.shape}')
        print(f'Valid pixels: {pixel_matrix.shape[0]}')
        print(f'Valid bands: {pixel_matrix.shape[1]}')
        print(f'NaN count in pixel_matrix: {np.isnan(pixel_matrix).sum()}')
        print(f'Inf count in pixel_matrix: {np.isinf(pixel_matrix).sum()}')

        processed.append({
            'species_richness': richness,
            'data_dir': data_dir,
            'roi_cube_clean': roi_cube_clean,
            'wl_nm_clean': wl_nm_clean,
            'pixel_matrix': pixel_matrix,
            'valid_pixel_mask': valid_pixel_mask,
        })

    reference_wl = processed[0]['wl_nm_clean']
    for item in processed[1:]:
        if len(item['wl_nm_clean']) != len(reference_wl) or not np.allclose(item['wl_nm_clean'], reference_wl):
            raise ValueError('Cleaned wavelengths are not identical across all images.')

    global_pca = fit_global_pca(
        [item['pixel_matrix'] for item in processed],
        n_components=n_components,
        max_pixels=max_pixels_for_pca,
        random_state=42
    )

    all_pixels = np.vstack([item['pixel_matrix'] for item in processed])
    global_mean_spectrum = np.nanmean(all_pixels, axis=0)

    print('\nGlobal PCA fitted successfully.')
    print('Explained variance ratio:', global_pca.explained_variance_ratio_)
    print('Cumulative explained variance:', np.sum(global_pca.explained_variance_ratio_))

    results = []
    for item in processed:
        pixel_matrix = item['pixel_matrix']
        local_mean_spectrum = np.nanmean(pixel_matrix, axis=0)

        # Compute all metrics
        chv_value, transformed_unique = compute_chv_in_global_pca(pixel_matrix, global_pca)
        cha_local_values = compute_cha_values(pixel_matrix, local_mean_spectrum)
        cha_global_values = compute_cha_values(pixel_matrix, global_mean_spectrum)
        sam_local_values = compute_sam_values(pixel_matrix, local_mean_spectrum)
        sam_global_values = compute_sam_values(pixel_matrix, global_mean_spectrum)
        sid_local_values = compute_sid_values(pixel_matrix, local_mean_spectrum)
        sid_global_values = compute_sid_values(pixel_matrix, global_mean_spectrum)

        # Store in processed item
        item['local_mean_spectrum'] = local_mean_spectrum
        item['cha_local_values'] = cha_local_values
        item['cha_global_values'] = cha_global_values
        item['sam_local_values'] = sam_local_values
        item['sam_global_values'] = sam_global_values
        item['sid_local_values'] = sid_local_values
        item['sid_global_values'] = sid_global_values

        # Summarize statistics
        cha_local_stats = summarize_metric(cha_local_values)
        cha_global_stats = summarize_metric(cha_global_values)
        sam_local_stats = summarize_metric(sam_local_values)
        sam_global_stats = summarize_metric(sam_global_values)
        sid_local_stats = summarize_metric(sid_local_values)
        sid_global_stats = summarize_metric(sid_global_values)

        results.append({
            'species_richness': item['species_richness'],
            'data_dir': item['data_dir'],
            'n_valid_pixels': pixel_matrix.shape[0],
            'n_valid_bands': pixel_matrix.shape[1],
            'CHV': chv_value,
            'CHA_local_mean': cha_local_stats['mean'],
            'CHA_local_median': cha_local_stats['median'],
            'CHA_local_std': cha_local_stats['std'],
            'CHA_global_mean': cha_global_stats['mean'],
            'CHA_global_median': cha_global_stats['median'],
            'CHA_global_std': cha_global_stats['std'],
            'SAM_local_mean': sam_local_stats['mean'],
            'SAM_local_median': sam_local_stats['median'],
            'SAM_local_std': sam_local_stats['std'],
            'SAM_global_mean': sam_global_stats['mean'],
            'SAM_global_median': sam_global_stats['median'],
            'SAM_global_std': sam_global_stats['std'],
            'SID_local_mean': sid_local_stats['mean'],
            'SID_local_median': sid_local_stats['median'],
            'SID_local_std': sid_local_stats['std'],
            'SID_global_mean': sid_global_stats['mean'],
            'SID_global_median': sid_global_stats['median'],
            'SID_global_std': sid_global_stats['std'],
        })

        print(
            f"Richness {item['species_richness']:>2} --> "
            f"CHV={chv_value:.4f}, "
            f"CHA(local/global)=({cha_local_stats['mean']:.4f}, {cha_global_stats['mean']:.4f}), "
            f"SAM(local/global)=({sam_local_stats['mean']:.4f}, {sam_global_stats['mean']:.4f}), "
            f"SID(local/global)=({sid_local_stats['mean']:.4f}, {sid_global_stats['mean']:.4f})"
        )

        if visualize_one:
            if visualize_richness is None or item['species_richness'] == visualize_richness:
                if transformed_unique is not None and transformed_unique.shape[1] == 3:
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    hull = ConvexHull(transformed_unique)
                    volume = hull.volume
                    ax.scatter(
                        transformed_unique[:, 0],
                        transformed_unique[:, 1],
                        transformed_unique[:, 2],
                        s=2, c='b', alpha=0.2
                    )
                    for simplex in hull.simplices:
                        simplex = np.append(simplex, simplex[0])
                        ax.plot(
                            transformed_unique[simplex, 0],
                            transformed_unique[simplex, 1],
                            transformed_unique[simplex, 2],
                            'k-', lw=0.4
                        )
                    hull_faces = [transformed_unique[simplex] for simplex in hull.simplices]
                    hull_poly = Poly3DCollection(hull_faces, alpha=0.15, color='cyan')
                    ax.add_collection3d(hull_poly)
                    ax.set_xlabel('PC1', fontsize=14)
                    ax.set_ylabel('PC2', fontsize=14)
                    ax.set_zlabel('PC3', fontsize=14)
                    ax.set_title(f"Global PCA CHV | richness={item['species_richness']}\\nCHV = {volume:.4f}", fontsize=16)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

    results_df = pd.DataFrame(results).sort_values('species_richness').reset_index(drop=True)
    return results_df, global_pca, global_mean_spectrum, processed, reference_wl


