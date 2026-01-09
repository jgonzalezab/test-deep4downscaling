import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages

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

# If ensemble, randomly select one member for metrics
if has_ensemble:
    np.random.seed(42)  # For reproducibility
    n_members = len(predictions[var_target].member)
    random_member = np.random.randint(0, n_members)
    print(f"Randomly selected member {random_member} out of {n_members} for metrics computation")
    pred_for_metrics = predictions[var_target].isel(member=random_member)
else:
    pred_for_metrics = predictions[var_target]

# Ensure both have same time coordinates
y_test_var = y_test[var_target]

# Compute simple diagnostics
rmse = np.sqrt(((pred_for_metrics - y_test_var) ** 2).mean(dim='time'))
bias_mean = (pred_for_metrics - y_test_var).mean(dim='time')

# Additional metrics for precipitation
# RX1day: Mean of annual maximum daily precipitation
y_test_rx1day = y_test_var.groupby('time.year').max(dim='time').mean(dim='year')
pred_rx1day = pred_for_metrics.groupby('time.year').max(dim='time').mean(dim='year')
bias_rx1day = pred_rx1day - y_test_rx1day

diagnostics = {
    'rmse': rmse, 
    'bias_mean': bias_mean,
    'bias_rx1day': bias_rx1day
}

# Generate PDF report with spatial maps and boxplots
# Changed filename to match run_all_eval.py expectations
output_pdf = f'{FIGS_PATH}/{model_name}_standard_metrics.pdf'
print(f"Generating PDF report: {output_pdf}")

# Metric titles and colormap configuration
metric_titles = {
    'rmse': 'RMSE (mm/day)',
    'bias_mean': 'Mean Bias (mm/day)',
    'bias_rx1day': 'Bias RX1day (mm/day)'
}

# Fixed limits for consistent comparison across models
METRIC_LIMITS = {
    'rmse': {'vmin': 0, 'vmax': 12, 'cmap': 'Reds_r'},
    'bias_mean': {'vmin': -3, 'vmax': 3, 'cmap': 'BrBG'},
    'bias_rx1day': {'vmin': -40, 'vmax': 40, 'cmap': 'BrBG'}
}

with PdfPages(output_pdf) as pdf:
    for name, value in diagnostics.items():
        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        
        # 1. Spatial Map
        ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        
        # Get fixed configuration
        config = METRIC_LIMITS.get(name, {'vmin': None, 'vmax': None, 'cmap': 'turbo'})
        vmin = config['vmin']
        vmax = config['vmax']
        cmap = config['cmap']
        
        # Plot spatial data with discrete colorbar (21 levels)
        plot_kwargs = {
            'ax': ax_map,
            'transform': ccrs.PlateCarree(),
            'cmap': cmap,
            'add_colorbar': False,
            'levels': 21, # Increased discrete levels
            'vmin': vmin,
            'vmax': vmax
        }
        
        im = value.plot(**plot_kwargs)
        ax_map.coastlines(resolution='50m', linewidth=0.6)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax_map.set_title('Spatial Map', fontsize=12, fontweight='bold')
        
        # Add discrete colorbar
        cbar = plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.1, fraction=0.05)
        cbar.set_label(metric_titles.get(name, name))
        
        # 2. Boxplot
        ax_box = fig.add_subplot(1, 2, 2)
        flat_data = value.values.flatten()
        flat_data = flat_data[np.isfinite(flat_data)]
        
        bp = ax_box.boxplot(flat_data, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax_box.set_title('Distribution', fontsize=12, fontweight='bold')
        ax_box.set_ylabel(metric_titles.get(name, name))
        ax_box.grid(True, alpha=0.3)
        ax_box.set_axisbelow(True)
        
        # Add mean value
        mean_val = np.nanmean(flat_data)
        ax_box.text(0.95, 0.95, f'Mean: {mean_val:.2f}',
                    transform=ax_box.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, fontweight='bold')
        
        # Overall title
        ensemble_info = f" (random member {random_member})" if has_ensemble else ""
        plt.suptitle(f'{metric_titles.get(name, name)} | {model_name}{ensemble_info}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        pdf.savefig(fig)
        plt.close(fig)

print(f"Report saved to: {output_pdf}")