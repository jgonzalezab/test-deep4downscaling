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
    
    y_test_var = y_test[var_target]
    predictions_var = predictions[var_target]

    # Flatten data and remove NaNs for histogram
    gt_flat = y_test_var.values.flatten()
    gt_flat = gt_flat[~np.isnan(gt_flat)]
    
    # Generate PDF report
    output_pdf = f'{FIGS_PATH}/{model_name}_histogram_comparison.pdf'
    print(f"Generating PDF report: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Large number of bins for detail
        # Using a range that covers the data, with bins every 1 mm/day up to a reasonable max or auto
        max_val = max(np.nanmax(gt_flat), np.nanmax(predictions_var.values))
        bins = np.linspace(0, max_val, 150)
        
        # 1. Ground Truth Histogram
        ax.hist(gt_flat, bins=bins, histtype='step', color='black', label='Target (GT)', 
                linewidth=2.5, density=False)
        
        # 2. Prediction(s) Histogram
        if has_ensemble:
            num_members = len(predictions_var.member)
            print(f"Plotting histograms for {num_members} ensemble members...")
            for i in range(num_members):
                member_data = predictions_var.isel(member=i).values.flatten()
                member_data = member_data[~np.isnan(member_data)]
                # First member gets the label, others don't to avoid legend clutter
                label = f'Ensemble Members ({model_name})' if i == 0 else None
                ax.hist(member_data, bins=bins, histtype='step', alpha=0.6, 
                        linewidth=1.0, label=label)
        else:
            pred_flat = predictions_var.values.flatten()
            pred_flat = pred_flat[~np.isnan(pred_flat)]
            ax.hist(pred_flat, bins=bins, histtype='step', color='red', 
                    label=f'Prediction ({model_name})', linewidth=2.0)
            
        # Formatting
        ax.set_yscale('log')
        ax.set_title(f'Precipitation Intensity Histogram | {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Precipitation (mm/day)', fontsize=12)
        ax.set_ylabel('Counts (Log Scale)', fontsize=12)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report saved to: {output_pdf}")

if __name__ == "__main__":
    main()
