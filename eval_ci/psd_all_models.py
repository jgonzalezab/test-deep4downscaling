import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# =========================
# Paths
# =========================
DATA_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/input'
PREDS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/preds'
FIGS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/eval_ci/figs'
os.makedirs(FIGS_PATH, exist_ok=True)

# =========================
# Config
# =========================
var_target = 'pr'

model_list = [
    'vit_MSE',
    'vit_ASYM',
    'vit_BerGamma',
    'vit_CRPS',
    'vit_CRPS_spectral'
]

selected_days = ['2011-03-04', '2015-04-04', '2018-03-10', '2020-09-20']

# =========================
# Fixed colors
# =========================
model_colors = {
    "Ground Truth": "black",
    "vit_MSE": "blue",
    "vit_ASYM": "red",
    "vit_BerGamma": "green",
    "vit_CRPS": "purple",
    "vit_CRPS_spectral": "gray"
}

# =========================
# PSD function
# =========================
def compute_spatial_psd_2d(data):
    from scipy import fft

    data = np.nan_to_num(data, nan=0.0)
    ny, nx = data.shape[-2:]

    fft_data = fft.fft2(data, axes=(-2, -1))
    power = np.abs(fft_data) ** 2

    if data.ndim > 2:
        power = power.mean(axis=0)

    kx = fft.fftfreq(nx)
    ky = fft.fftfreq(ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)

    k_bins = np.linspace(0, k_grid.max(), min(nx, ny)//2)
    psd_radial = np.zeros(len(k_bins) - 1)

    for i in range(len(k_bins) - 1):
        mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i+1])
        if mask.any():
            psd_radial[i] = power[mask].mean()

    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    valid = k_centers > 0

    return k_centers[valid], psd_radial[valid]

# =========================
# Load model helper
# =========================
def load_model_prediction(model_name):
    ds = xr.open_dataset(f'{PREDS_PATH}/{model_name}.nc')
    var = ds[var_target]

    if 'member' in var.dims:
        np.random.seed(42)
        n_members = len(var.member)
        random_member = np.random.randint(0, n_members)
        print(f"{model_name}: ensemble detected → using member {random_member}/{n_members}")
        var = var.isel(member=random_member)
    else:
        print(f"{model_name}: deterministic model")

    return var

# =========================
# Main
# =========================
def main():

    # ---- Load GT ----
    predictand = xr.open_dataset(f'{DATA_PATH}/pr_AEMET.nc').load()
    predictand = predictand.sel(lon=slice(-9.425, 3.375))

    lat = predictand.lat.values
    dlat = np.diff(lat).mean()
    n_extra = 256 - lat.size
    new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
    predictand = predictand.reindex(lat=new_lat)

    y_test = predictand.sel(time=slice('2011', '2020'))
    y_test_var = y_test[var_target]

    output_pdf = f'{FIGS_PATH}/multi_model_psd_comparison.pdf'
    print(f"\nGenerating PDF report: {output_pdf}\n")

    with PdfPages(output_pdf) as pdf:

        # ==========================================================
        # 1) Overall PSD
        # ==========================================================
        print("Computing overall PSD...")
        k_gt, psd_gt_all = compute_spatial_psd_2d(y_test_var.values)

        fig, ax = plt.subplots(figsize=(11, 8))

        # ---- Ground truth ----
        ax.loglog(
            k_gt, psd_gt_all,
            label="Ground Truth",
            linewidth=3,
            color=model_colors["Ground Truth"]
        )

        # ---- Models ----
        for model_name in model_list:
            try:
                predictions_var = load_model_prediction(model_name)
                k_pred, psd_pred_all = compute_spatial_psd_2d(predictions_var.values)

                ax.loglog(
                    k_pred, psd_pred_all,
                    '--',
                    linewidth=2,
                    label=model_name,
                    color=model_colors.get(model_name, None)
                )

            except Exception as e:
                print(f"Error with {model_name}: {e}")

        ax.set_title('Spatial Power Spectral Density\n(2011–2020 average)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Spatial Wavenumber', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nReport saved to: {output_pdf}\n")

# =========================
if __name__ == "__main__":
    main()