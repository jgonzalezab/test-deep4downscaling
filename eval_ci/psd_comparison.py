import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# Paths
DATA_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/input'
PREDS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/preds'
FIGS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/eval_ci/figs'

# Ensure output directory exists
os.makedirs(FIGS_PATH, exist_ok=True)

# Variable target
var_target = 'pr'

# Model name from environment or default
model_name = os.getenv('MODEL_NAME', 'vit_ASYM')

# Selectable days (format: 'YYYY-MM-DD')
selected_days = ['2011-03-04', '2015-04-04', '2018-03-10', '2020-09-20']

def compute_spatial_psd_2d(data):
    """
    Compute 2D power spectral density for spatial fields.
    Returns radially averaged power spectrum.
    """
    from scipy import fft
    
    # Fill NaNs with zeros as requested for precipitation
    data = np.nan_to_num(data, nan=0.0)
    
    # Get spatial dimensions
    ny, nx = data.shape[-2:]
    
    # Apply 2D FFT
    fft_data = fft.fft2(data, axes=(-2, -1))
    power = np.abs(fft_data) ** 2
    
    # Average over time if present
    if len(data.shape) > 2:
        power = power.mean(axis=0)
    
    # Compute wavenumbers
    kx = fft.fftfreq(nx)
    ky = fft.fftfreq(ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Radial averaging
    k_bins = np.linspace(0, k_grid.max(), min(nx, ny)//2)
    psd_radial = np.zeros(len(k_bins) - 1)
    
    for i in range(len(k_bins) - 1):
        mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i+1])
        if mask.any():
            psd_radial[i] = power[mask].mean()
    
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    
    return k_centers[k_centers > 0], psd_radial[k_centers > 0]

def main():
    # Load test data (same as training script)
    predictand_filename = f'{DATA_PATH}/pr_AEMET.nc'
    predictand = xr.open_dataset(predictand_filename).load()
    
    # Subset predictand (same as training)
    predictand = predictand.sel(lon=slice(-9.425, 3.375))
    
    # Extend the latitude to 256 grid points (same as training)
    lat = predictand.lat.values
    dlat = np.diff(lat).mean()
    n_extra = 256 - lat.size
    new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
    predictand = predictand.reindex(lat=new_lat)
    
    # Test period (same as training script)
    years_test = ('2011', '2020')
    y_test = predictand.sel(time=slice(*years_test))
    
    # Load predictions
    predictions = xr.open_dataset(f'{PREDS_PATH}/{model_name}.nc')
    
    # Check if predictions have ensemble dimension
    has_ensemble = 'member' in predictions[var_target].dims
    print(f"Model: {model_name}")
    print(f"Has ensemble dimension: {has_ensemble}")
    
    # If ensemble, randomly select one member for PSD comparison
    if has_ensemble:
        np.random.seed(42)  # For reproducibility
        n_members = len(predictions[var_target].member)
        random_member = np.random.randint(0, n_members)
        print(f"Randomly selected member {random_member} out of {n_members} for PSD comparison")
        predictions_var = predictions[var_target].isel(member=random_member)
    else:
        predictions_var = predictions[var_target]
    
    y_test_var = y_test[var_target]

    # Generate PDF report
    output_pdf = f'{FIGS_PATH}/{model_name}_psd_comparison.pdf'
    print(f"Generating PDF report: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        # 1. Overall PSD for the whole test period
        print("Computing overall PSD...")
        try:
            k_gt, psd_gt_all = compute_spatial_psd_2d(y_test_var.values)
            k_pred, psd_pred_all = compute_spatial_psd_2d(predictions_var.values)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.loglog(k_gt, psd_gt_all, label="Ground Truth", color='black', linewidth=2)
            ax.loglog(k_pred, psd_pred_all, label=f"Prediction ({model_name})", color='red', linestyle='--', linewidth=2)
            
            ensemble_info = f" (random member {random_member})" if has_ensemble else ""
            ax.set_title(f'Overall Spatial Power Spectral Density | {model_name}{ensemble_info}\n(Average over 2011-2020)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Spatial Wavenumber', fontsize=12)
            ax.set_ylabel('Power', fontsize=12)
            ax.grid(True, which="both", ls="-", alpha=0.3)
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print(f"Error computing overall PSD: {e}")

        # 2. Daily PSD for selected days
        for day_str in selected_days:
            try:
                # Select the day
                day_gt = y_test_var.sel(time=day_str, method='nearest')
                day_pred = predictions_var.sel(time=day_str, method='nearest')
                
                actual_date = pd.to_datetime(day_gt.time.values).strftime('%Y-%m-%d')
                
                print(f"Computing PSD for {actual_date}...")
                k_gt, psd_gt = compute_spatial_psd_2d(day_gt.values)
                k_pred, psd_pred = compute_spatial_psd_2d(day_pred.values)
                
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.loglog(k_gt, psd_gt, label="Ground Truth", color='black', linewidth=2)
                ax.loglog(k_pred, psd_pred, label=f"Prediction ({model_name})", color='red', linestyle='--', linewidth=2)
                
                ensemble_info = f" (random member {random_member})" if has_ensemble else ""
                ax.set_title(f'Spatial Power Spectral Density | {model_name}{ensemble_info} | {actual_date}', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Spatial Wavenumber', fontsize=12)
                ax.set_ylabel('Power', fontsize=12)
                ax.grid(True, which="both", ls="-", alpha=0.3)
                ax.legend(fontsize=12)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error processing day {day_str}: {e}")

    print(f"Report saved to: {output_pdf}")

if __name__ == "__main__":
    main()

