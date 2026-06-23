# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:01:45 2026

@author: Manis
"""

from spectral import *
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import threshold_otsu


# ---------------------------------------------------------
# Basic IO
# ---------------------------------------------------------
def open_envi_image(data_dir, basename):
    hsi_hdr_file = data_dir + '/' + '{}.img.hdr'.format(basename)
    hsi_img_file = data_dir + '/' + '{}.img'.format(basename)
    img_envi = envi.open(hsi_hdr_file, hsi_img_file)
    img = img_envi.load()
    return img


def open_envi_gt_image(data_dir, basename, verbose=False):
    hsi_hdr_file = data_dir + '/' + '{}.img.hdr'.format(basename)
    hsi_img_file = data_dir + '/' + '{}.img'.format(basename)
    img_envi = envi.open(hsi_hdr_file, hsi_img_file)
    img = img_envi.load()
    header_file = envi.read_envi_header(hsi_hdr_file)
    if verbose:
        print(header_file)
    return img, header_file


def wavelength(data_dir, basename):
    hsi_hdr_file = data_dir + '/' + '{}.img.hdr'.format(basename)
    hsi_img_file = data_dir + '/' + '{}.img'.format(basename)
    img_envi = envi.open(hsi_hdr_file, hsi_img_file)
    return img_envi.bands.centers


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def view_msi_image_color(hsi_img, plot=True):
    hsi_img_b = hsi_img[:, :, 3]
    hsi_img_b_normalized = hsi_img_b / np.nanmax(hsi_img_b)

    hsi_img_g = hsi_img[:, :, 2]
    hsi_img_g_normalized = hsi_img_g / np.nanmax(hsi_img_g)

    hsi_img_r = hsi_img[:, :, 1]
    hsi_img_r_normalized = hsi_img_r / np.nanmax(hsi_img_r)

    image = np.asarray(
        255 * np.concatenate(
            [hsi_img_b_normalized, hsi_img_g_normalized, hsi_img_r_normalized],
            axis=2
        ),
        dtype=np.uint8
    )

    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("RGB view")
        plt.axis("off")
        plt.show()

    return image


def plot_spectral_footprint(img_ref, wl, pixel_x, pixel_y, plot=True):
    leaf_pixel = img_ref[pixel_y:pixel_y+1, pixel_x:pixel_x+1, :]
    leaf_pixel_squeezed = np.squeeze(leaf_pixel)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(wl, leaf_pixel_squeezed)
        plt.title(f'Spectral Footprint\n(Pixel {pixel_x},{pixel_y})')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.show()

    return leaf_pixel_squeezed


# ---------------------------------------------------------
# Reflectance conversion
# ---------------------------------------------------------
def get_reflectance_image(input_radiance_img,
                          cp80y1, cp80x1, cp80y2, cp80x2,
                          cp20y1, cp20x1, cp20y2, cp20x2):

    r1_calpanel_80, c1_calpanel_80 = cp80y1, cp80x1
    r2_calpanel_80, c2_calpanel_80 = cp80y2, cp80x2

    r1_calpanel_20, c1_calpanel_20 = cp20y1, cp20x1
    r2_calpanel_20, c2_calpanel_20 = cp20y2, cp20x2

    avg_calpanel_80_radiance_vec = (
        input_radiance_img[r1_calpanel_80, c1_calpanel_80, :] +
        input_radiance_img[r2_calpanel_80, c2_calpanel_80, :]
    ) / 2

    avg_calpanel_20_radiance_vec = (
        input_radiance_img[r1_calpanel_20, c1_calpanel_20, :] +
        input_radiance_img[r2_calpanel_20, c2_calpanel_20, :]
    ) / 2

    a_vec = (avg_calpanel_80_radiance_vec - avg_calpanel_20_radiance_vec) / (0.8 - 0.2)
    b_vec = (avg_calpanel_20_radiance_vec * 0.8 - avg_calpanel_80_radiance_vec * 0.2) / (0.8 - 0.2)

    reflectance_img = (input_radiance_img - b_vec) / a_vec
    return reflectance_img


def process_radiance_image(data_dir,
                           cp80y1, cp80x1, cp80y2, cp80x2,
                           cp20y1, cp20x1, cp20y2, cp20x2,
                           plot_rgb=False):
    basename = 'AVIRIS-NG'
    input_radiance_img = open_envi_image(data_dir, basename)

    if plot_rgb:
        view_msi_image_color(input_radiance_img, plot=True)

    img_ref = get_reflectance_image(
        input_radiance_img,
        cp80y1, cp80x1, cp80y2, cp80x2,
        cp20y1, cp20x1, cp20y2, cp20x2
    )

    wl = wavelength(data_dir, basename)
    return img_ref, wl


# ---------------------------------------------------------
# Ground truth + ROI
# ---------------------------------------------------------
def ground_truth_image(data_dir, plot=False, verbose=False):
    basename = 'AVIRIS-NG_truth'
    truth_img, header_file = open_envi_gt_image(data_dir, basename, verbose=verbose)

    if plot:
        labelled_data = truth_img.read_bands(0)
        plt.figure(figsize=(12, 6))
        plt.grid(False)
        plt.imshow(labelled_data, cmap='jet')
        plt.colorbar()
        plt.title('Ground Truth Labeled Image')
        plt.show()

    return truth_img, header_file


def extract_rois(arr, x, y, w, h, intensity, line):
    roi = arr[y:y+h, x:x+w, :]

    bounding_box = np.array(arr, copy=True)
    bounding_box[y-line:y, x-line:x+w+line, :] = intensity
    bounding_box[y:y+h, x-line:x, :] = intensity
    bounding_box[y+h:y+h+line, x-line:x+w+line, :] = intensity
    bounding_box[y:y+h, x+w:x+w+line, :] = intensity

    return roi, bounding_box


def GT_roi(data_dir, coordinates, width, height, intensity=2, line_width=2, plot=False, verbose=False):
    basename = 'AVIRIS-NG_truth'
    truth_img, header_file = open_envi_gt_image(data_dir, basename, verbose=verbose)

    rois_gt = []
    last_roi = None

    for coordinate in coordinates:
        x, y = coordinate
        roi_gt, image_bboxed = extract_rois(truth_img, x, y, width, height, intensity, line_width)
        rois_gt.append(roi_gt)
        last_roi = roi_gt

    if plot and last_roi is not None:
        labelled_data_roi = last_roi.read_bands(0)
        plt.figure(figsize=(12, 6))
        plt.grid(False)
        plt.imshow(labelled_data_roi, cmap='jet')
        plt.colorbar()
        plt.title('Extracted ROI from Ground Truth Image')
        plt.show()

    return rois_gt, header_file


def Reflectance_img_roi(data_dir,
                        cp80y1, cp80x1, cp80y2, cp80x2,
                        cp20y1, cp20x1, cp20y2, cp20x2,
                        coordinates, width, height,
                        intensity=2, line_width=2,
                        plot_roi=False, plot_mean_spectra=False):
    img_ref, wl = process_radiance_image(
        data_dir,
        cp80y1, cp80x1, cp80y2, cp80x2,
        cp20y1, cp20x1, cp20y2, cp20x2,
        plot_rgb=False
    )

    rois_reflectance = []
    image_bboxed = None

    for coordinate in coordinates:
        x, y = coordinate
        roi, image_bboxed = extract_rois(img_ref, x, y, width, height, intensity, line_width)
        rois_reflectance.append(roi)

    if plot_roi and image_bboxed is not None:
        plt.figure(figsize=(12, 6))
        plt.grid(False)
        q = 3
        plt.imshow(image_bboxed[:, :, q])
        plt.colorbar()
        plt.title(f'band - {q}')
        plt.show()

    if plot_mean_spectra:
        plt.figure(figsize=(12, 6))
        for i, roi in enumerate(rois_reflectance):
            intensity_vals = np.nanmean(roi, axis=(0, 1))
            plt.plot(wl, intensity_vals, label=f'ROI {i+1}')
        plt.legend(loc='upper left')
        plt.title('Spectral Footprint\nMean in ROI Area')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.grid(False)
        plt.show()

    return rois_reflectance


# ---------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------
def remove_outliers_iqr(data):
    q1 = np.percentile(data, 25, axis=(0, 1), keepdims=True)
    q3 = np.percentile(data, 75, axis=(0, 1), keepdims=True)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data_without_outliers = np.where((data < lower_bound) | (data > upper_bound), np.nan, data)
    return data_without_outliers


def mask_bad_wavelength_regions(roi_cube, wl):
    wl_nm = 1000 * np.array(wl)
    roi_cube = np.array(roi_cube, dtype=np.float64, copy=True)

    roi_cube[:, :, (1350 <= wl_nm) & (wl_nm <= 1500)] = np.nan
    roi_cube[:, :, (1790 <= wl_nm) & (wl_nm <= 1995)] = np.nan
    roi_cube[:, :, (wl_nm > 2475)] = np.nan
    return roi_cube


# ---------------------------------------------------------
# Spectral stats
# ---------------------------------------------------------
def calculate_and_plot_spectral_stats(Ref_ROI, wl, plot=True):
    roi = Ref_ROI[0]
    roi_ = remove_outliers_iqr(roi)
    roi_ = mask_bad_wavelength_regions(roi_, wl)

    wl_nm = 1000 * np.array(wl)

    mean_spectra = np.nanmean(roi_, axis=(0, 1))
    std_dev_spectra = np.nanstd(roi_, axis=(0, 1))
    min_spectra = np.nanmin(roi_, axis=(0, 1))
    max_spectra = np.nanmax(roi_, axis=(0, 1))
    cv_spectra = (std_dev_spectra / mean_spectra) * 100

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(wl_nm, mean_spectra, label='Mean', color='blue')
        plt.fill_between(
            wl_nm,
            mean_spectra - std_dev_spectra,
            mean_spectra + std_dev_spectra,
            color='blue', alpha=0.2, label='±1 Std Dev'
        )
        plt.title('Spectral Statistics', fontsize=20)
        plt.xlabel('Wavelength (nm)', fontsize=20)
        plt.ylabel('Reflectance', fontsize=20)
        plt.ylim([-0.01, 0.35])
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(wl_nm, cv_spectra, label='CV Spectra (%)', color='r')
        plt.title('Coefficient of Variation (CV) per Band', fontsize=20)
        plt.xlabel('Wavelength (nm)', fontsize=20)
        plt.ylabel('CV (%)', fontsize=20)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()

    return mean_spectra, std_dev_spectra, min_spectra, max_spectra, cv_spectra


def SNR(roi_calpanel, wl, plot=True):
    roi_calpanel = roi_calpanel[0]
    roi_calpanel = mask_bad_wavelength_regions(roi_calpanel, wl)
    wl_nm = 1000 * np.array(wl)

    snr_per_band = []
    for band in range(roi_calpanel.shape[2]):
        band_data = roi_calpanel[:, :, band]
        mean_signal = np.nanmean(band_data)
        std_noise = np.nanstd(band_data)
        snr = mean_signal / std_noise if std_noise > 0 else float('inf')
        snr_per_band.append(snr)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(wl_nm, snr_per_band, marker='s', color='red', label='SNR per Band')
        plt.title('Signal-to-Noise Ratio (SNR) per Band', fontsize=20)
        plt.xlabel('Wavelength (nm)', fontsize=20)
        plt.ylabel('SNR', fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.show()

    return np.array(snr_per_band)


# ---------------------------------------------------------
# NDVI helpers
# ---------------------------------------------------------
def compute_and_mask_ndvi(roi_Ref, wl, red_wl=0.670, nir_wl=0.800, ndvi_threshold=0.4, plot=True):
    wl = np.array(wl)
    roi_Ref = np.array(roi_Ref)

    red_band_idx = np.argmin(np.abs(wl - red_wl))
    nir_band_idx = np.argmin(np.abs(wl - nir_wl))

    red_band = roi_Ref[0, :, :, red_band_idx]
    nir_band = roi_Ref[0, :, :, nir_band_idx]

    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.clip(ndvi, -1, 1)

    veg_mask = ndvi >= ndvi_threshold
    masked_ref = np.copy(roi_Ref)
    masked_ref[0, ~veg_mask, :] = np.nan

    if plot:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(ndvi, cmap='RdYlGn')
        plt.colorbar(label='NDVI')
        plt.title('NDVI Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.hist(ndvi.flatten(), bins=50, color='green', alpha=0.7)
        plt.title('NDVI Histogram')
        plt.xlabel('NDVI')
        plt.ylabel('Pixel Count')
        plt.tight_layout()
        plt.show()

    masked_ref = np.squeeze(masked_ref)
    masked_ref = [masked_ref]

    return masked_ref, ndvi


def apply_otsu_ndvi_mask(roi_Ref, wl, red_wl=0.670, nir_wl=0.800, plot=True):
    wl = np.array(wl)
    roi_Ref = np.array(roi_Ref)

    if roi_Ref.ndim == 4 and roi_Ref.shape[0] == 1:
        roi_Ref = roi_Ref[0]

    red_idx = np.argmin(np.abs(wl - red_wl))
    nir_idx = np.argmin(np.abs(wl - nir_wl))

    red = roi_Ref[:, :, red_idx]
    nir = roi_Ref[:, :, nir_idx]

    ndvi = (nir - red) / (nir + red)
    ndvi = np.clip(ndvi, -1, 1)

    ndvi_flat = ndvi[~np.isnan(ndvi)]
    otsu_thresh = threshold_otsu(ndvi_flat)

    veg_mask = ndvi >= otsu_thresh
    masked_ref = np.where(veg_mask[:, :, np.newaxis], roi_Ref, np.nan)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(ndvi, cmap='RdYlGn')
        axes[0].set_title('NDVI')
        axes[0].axis('off')

        axes[1].imshow(veg_mask, cmap='gray')
        axes[1].set_title(f'Otsu Vegetation Mask\n(Threshold = {otsu_thresh:.3f})')
        axes[1].axis('off')

        axes[2].imshow(np.where(veg_mask, ndvi, np.nan), cmap='RdYlGn')
        axes[2].set_title('NDVI (Masked)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    return masked_ref, ndvi, otsu_thresh


# ---------------------------------------------------------
# Main reusable workflow
# ---------------------------------------------------------
def process_spectral_workflow(
    data_dir,
    roi_coordinates=[(345, 345)],
    roi_width=10,
    roi_height=10,
    calpanel_coordinates=[(340, 355)],
    calpanel_width=4,
    calpanel_height=4,
    sample_pixel=(348, 347),  
    plot_rgb=False,
    plot_pixel_spectrum=False,
    plot_gt=False,
    plot_roi=False,
    plot_roi_mean_spectra=False,
    compute_snr=False,
    plot_snr=False,
    return_gt=False
):
    """
    Reusable workflow for import-based processing.
    Default behavior is quiet for batch processing.
    """

    cp80y1, cp80x1 = 356, 342
    cp80y2, cp80x2 = 357, 343
    cp20y1, cp20x1 = 341, 355
    cp20y2, cp20x2 = 342, 357
    
    # cp80y1,cp80x1 = 365,332 #y,x#row,column
    # cp80y2,cp80x2 = 363,330
    # cp20y1,cp20x1 = 334,362
    # cp20y2,cp20x2 = 335,365

    ref, wl = process_radiance_image(
        data_dir,
        cp80y1, cp80x1, cp80y2, cp80x2,
        cp20y1, cp20x1, cp20y2, cp20x2,
        plot_rgb=plot_rgb
    )

    if plot_pixel_spectrum:
        leaf_pixel_x, leaf_pixel_y = sample_pixel
        plot_spectral_footprint(ref, wl, leaf_pixel_x, leaf_pixel_y, plot=True)

    roi_Ref = Reflectance_img_roi(
        data_dir,
        cp80y1, cp80x1, cp80y2, cp80x2,
        cp20y1, cp20x1, cp20y2, cp20x2,
        roi_coordinates, roi_width, roi_height,
        plot_roi=plot_roi,
        plot_mean_spectra=plot_roi_mean_spectra
    )

    result = {
        "roi_Ref": roi_Ref,
        "wl": wl,
        "ref": ref
    }

    if return_gt:
        roi_gt, header_file = GT_roi(
            data_dir,
            roi_coordinates,
            roi_width,
            roi_height,
            plot=plot_gt
        )
        result["roi_gt"] = roi_gt
        result["gt_header"] = header_file

    if compute_snr:
        roi_calpanel = Reflectance_img_roi(
            data_dir,
            cp80y1, cp80x1, cp80y2, cp80x2,
            cp20y1, cp20x1, cp20y2, cp20x2,
            calpanel_coordinates, calpanel_width, calpanel_height,
            plot_roi=False,
            plot_mean_spectra=False
        )
        result["snr"] = SNR(roi_calpanel, wl, plot=plot_snr)

    return result


# ---------------------------------------------------------
# Example usage only when run directly
# ---------------------------------------------------------
# if __name__ == "__main__":
#     data_dir = r'D:/Drive F 5-11-2024/A_BiosCape_Project3/Spectral_diversity/different_species/3sp/1m'

#     result = process_spectral_workflow(
#         data_dir,
#         plot_rgb=True,
#         plot_pixel_spectrum=True,
#         plot_gt=True,
#         plot_roi=True,
#         plot_roi_mean_spectra=True,
#         compute_snr=True,
#         plot_snr=True,
#         return_gt=True
#     )

#     roi_Ref = result["roi_Ref"]
#     wl= result["wl"]