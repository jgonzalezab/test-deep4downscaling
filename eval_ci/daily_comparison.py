import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
# These should be within the test period (2011-2020)
selected_days = ['2011-03-04', '2015-04-04', '2018-03-10', '2020-09-20']

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
    
    # If ensemble, randomly select one member for daily comparison
    if has_ensemble:
        np.random.seed(42)  # For reproducibility
        n_members = len(predictions[var_target].member)
        random_member = np.random.randint(0, n_members)
        print(f"Randomly selected member {random_member} out of {n_members} for daily comparison")
        predictions_var = predictions[var_target].isel(member=random_member)
    else:
        predictions_var = predictions[var_target]
    
    y_test_var = y_test[var_target]

    # Generate PDF report
    output_pdf = f'{FIGS_PATH}/{model_name}_daily_comparison.pdf'
    print(f"Generating PDF report: {output_pdf}")

    # Fixed limits for consistent comparison
    VMIN_DATA, VMAX_DATA = 0, 60
    VMIN_BIAS, VMAX_BIAS = -30, 30

    with PdfPages(output_pdf) as pdf:
        for day_str in selected_days:
            try:
                # Select the day
                day_gt = y_test_var.sel(time=day_str, method='nearest')
                day_pred = predictions_var.sel(time=day_str, method='nearest')
                
                # Get the actual date string for the title
                actual_date = pd.to_datetime(day_gt.time.values).strftime('%Y-%m-%d')
                
                # Compute bias for the day
                day_bias = day_pred - day_gt
                
                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                
                # Colormaps
                cmap_data = 'turbo'
                cmap_bias = 'BrBG'
                
                # 1. Ground Truth
                day_gt.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap=cmap_data, 
                           vmin=VMIN_DATA, vmax=VMAX_DATA, add_colorbar=False, levels=21)
                axes[0].set_title(f'Ground Truth ({actual_date})', fontsize=12, fontweight='bold')
                
                # 2. Prediction
                im1 = day_pred.plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap=cmap_data, 
                                   vmin=VMIN_DATA, vmax=VMAX_DATA, add_colorbar=False, levels=21)
                axes[1].set_title(f'Prediction ({model_name})', fontsize=12, fontweight='bold')
                
                # Add common colorbar for GT and Prediction
                cbar_data = plt.colorbar(im1, ax=axes[:2], orientation='horizontal', pad=0.1, fraction=0.05)
                cbar_data.set_label(f'{var_target} (mm/day)')
                
                # 3. Bias
                im2 = day_bias.plot(ax=axes[2], transform=ccrs.PlateCarree(), cmap=cmap_bias,
                                   vmin=VMIN_BIAS, vmax=VMAX_BIAS, add_colorbar=False, levels=21)
                axes[2].set_title('Bias (Pred - GT)', fontsize=12, fontweight='bold')
                
                cbar_bias = plt.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.1, fraction=0.05)
                cbar_bias.set_label('Bias (mm/day)')
                
                # Formatting
                for ax in axes:
                    ax.coastlines(resolution='50m', linewidth=0.6)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
                
                ensemble_info = f" (random member {random_member})" if has_ensemble else ""
                plt.suptitle(f'Daily Comparison | {model_name}{ensemble_info} | {actual_date}', fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                pdf.savefig(fig)
                plt.close(fig)
                print(f"Processed day: {actual_date}")
                
            except Exception as e:
                print(f"Error processing day {day_str}: {e}")

    print(f"Report saved to: {output_pdf}")

if __name__ == "__main__":
    main()

