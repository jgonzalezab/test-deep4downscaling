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
model_name = os.getenv('MODEL_NAME', 'vit_CRPS')

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
    
    if not has_ensemble:
        print(f"Model {model_name} does not have ensemble dimension. Skipping ensemble comparison.")
        return
    
    print(f"Model: {model_name}")
    print(f"Has ensemble dimension: {has_ensemble}")
    print(f"Number of members: {len(predictions[var_target].member)}")
    
    predictions_var = predictions[var_target]
    y_test_var = y_test[var_target]

    # Generate PDF report
    output_pdf = f'{FIGS_PATH}/{model_name}_ensemble_comparison.pdf'
    print(f"Generating PDF report: {output_pdf}")

    # Fixed limits for consistent comparison
    VMIN_DATA, VMAX_DATA = 0, 60

    with PdfPages(output_pdf) as pdf:
        for day_str in selected_days:
            try:
                # Select the day
                day_gt = y_test_var.sel(time=day_str, method='nearest')
                day_ensemble = predictions_var.sel(time=day_str, method='nearest')
                
                # Get the actual date string for the title
                actual_date = pd.to_datetime(day_gt.time.values).strftime('%Y-%m-%d')
                
                # Get number of members
                num_members = len(day_ensemble.member)
                
                # Calculate grid layout (add 1 for ground truth)
                total_plots = min(num_members, 11) + 1  # GT + up to 11 members
                ncols = 4
                nrows = (total_plots + ncols - 1) // ncols
                
                # Create figure
                fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), 
                                        subplot_kw={'projection': ccrs.PlateCarree()})
                axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
                
                # Colormaps
                cmap_data = 'turbo'
                
                # 1. Ground Truth (first subplot)
                im = day_gt.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap=cmap_data, 
                                 vmin=VMIN_DATA, vmax=VMAX_DATA, add_colorbar=False, levels=21)
                axes[0].set_title(f'Ground Truth ({actual_date})', fontsize=12, fontweight='bold')
                axes[0].coastlines(resolution='50m', linewidth=0.6)
                axes[0].add_feature(cfeature.BORDERS, linewidth=0.4)
                
                # 2. Ensemble Members (subsequent subplots) - plot all available, up to 11
                for i in range(min(11, num_members)):
                    day_pred = day_ensemble.isel(member=i)
                    day_pred.plot(ax=axes[i+1], transform=ccrs.PlateCarree(), cmap=cmap_data, 
                                 vmin=VMIN_DATA, vmax=VMAX_DATA, add_colorbar=False, levels=21)
                    axes[i+1].set_title(f'Member {i}', fontsize=12, fontweight='bold')
                    axes[i+1].coastlines(resolution='50m', linewidth=0.6)
                    axes[i+1].add_feature(cfeature.BORDERS, linewidth=0.4)
                
                # Turn off unused axes if any
                for j in range(total_plots, len(axes)):
                    axes[j].axis('off')
                
                plt.suptitle(f'Ensemble Comparison | {model_name} | {actual_date} | {num_members} members', 
                            fontsize=16, fontweight='bold')
                
                # Adjust layout to leave space for colorbar at the bottom
                plt.tight_layout(rect=[0, 0.06, 1, 0.96])
                
                # Add common colorbar in a dedicated axis to avoid overlapping
                cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
                cbar_data = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                cbar_data.set_label(f'{var_target} (mm/day)', fontsize=12, fontweight='bold')
                
                pdf.savefig(fig)
                plt.close(fig)
                print(f"Processed day: {actual_date}")
                
            except Exception as e:
                print(f"Error processing day {day_str}: {e}")

    print(f"Report saved to: {output_pdf}")

if __name__ == "__main__":
    main()

