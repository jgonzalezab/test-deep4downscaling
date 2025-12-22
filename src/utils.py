"""
Utility functions for deep4downscaling evaluation
"""
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import deep4downscaling.metrics


def _boxplot_metric(ax, target, pred, metric_func, var_target='pr', title='Metric', n_members=4):
    """
    Helper function to create a boxplot of a metric across spatial locations.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    target : xr.Dataset
        Target/reference data
    pred : xr.Dataset
        Model predictions data
    metric_func : callable
        Function to compute metric (takes target and pred, returns metric)
    var_target : str
        Variable name
    title : str
        Title for the plot
    n_members : int
        Number of ensemble members
    """
    # Compute metric for each ensemble member
    metric_data = []
    for member in range(n_members):
        metric = metric_func(target=target, pred=pred.isel(member=member), 
                           var_target=var_target)
        # Flatten spatial dimensions
        metric_flat = metric[var_target].values.flatten()
        metric_flat = metric_flat[~np.isnan(metric_flat)]
        metric_data.append(metric_flat)
    
    # Create boxplot
    bp = ax.boxplot(metric_data, labels=[f'M{i+1}' for i in range(n_members)],
                    patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, n_members))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Value')
    ax.set_xlabel('Ensemble Members')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)


def _map_plot(ax, data, var_to_plot='pr', colorbar_cmap='coolwarm', vlimits=(None, None),
              num_levels=20, central_longitude=0):
    """
    Helper function to create a map plot on a given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on (must have cartopy projection)
    data : xr.Dataset or xr.DataArray
        Data to plot
    var_to_plot : str
        Variable name to plot (ignored if data is DataArray)
    colorbar_cmap : str
        Colormap name
    vlimits : tuple
        (vmin, vmax) for colorbar limits
    num_levels : int
        Number of colorbar levels
    central_longitude : int
        Central longitude for map projection
        
    Returns
    -------
    cs : mappable
        The plot object for adding colorbar
    """
    if isinstance(data, xr.Dataset):
        data = data[var_to_plot]
    
    # Create discrete colormap
    continuous_cmap = plt.get_cmap(colorbar_cmap)
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))
    
    # Create pcolormesh
    if None in vlimits:
        cs = ax.pcolormesh(data.lon, data.lat, data,
                          transform=ccrs.PlateCarree(),
                          cmap=discrete_cmap)
    else:
        cs = ax.pcolormesh(data.lon, data.lat, data,
                          transform=ccrs.PlateCarree(),
                          cmap=discrete_cmap,
                          vmin=vlimits[0], vmax=vlimits[1])
    
    ax.coastlines(resolution='10m')
    
    return cs


def evaluate_and_save_plots(target, pred, output_path, time_to_plot=800, n_members=4):
    """
    Evaluate model predictions and save all visualization plots in a single PDF.
    
    Parameters
    ----------
    target : xr.Dataset
        Target/reference data
    pred : xr.Dataset
        Model predictions data
    output_path : str
        Path to save the combined PDF (e.g., '/path/to/figures/evaluation.pdf')
    time_to_plot : int
        Time index to plot, default=800
    n_members : int
        Number of ensemble members to plot, default=4
    """

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a PDF with all plots
    with PdfPages(output_path) as pdf:
        # 1. Target plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cs = _map_plot(ax, target.isel(time=time_to_plot),
                      var_to_plot='pr', colorbar_cmap='hot_r',
                      vlimits=(0, 16))
        plt.colorbar(cs, ax=ax, orientation='horizontal')
        ax.set_title('Target', fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Prediction plots (ensemble members)
        for member in range(n_members):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
            cs = _map_plot(ax, pred.isel(time=time_to_plot, member=member),
                          var_to_plot='pr', colorbar_cmap='hot_r',
                          vlimits=(0, 16))
            plt.colorbar(cs, ax=ax, orientation='horizontal')
            ax.set_title(f'Prediction - Member {member + 1}', fontsize=14, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # 3. RMSE - Spatial Map
        fig = plt.figure(figsize=(12, 8))
        rmse = deep4downscaling.metrics.rmse(
            target=target,
            pred=pred.isel(member=0),
            var_target='pr'
        )
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cs = _map_plot(ax, rmse, var_to_plot='pr',
                      colorbar_cmap='YlOrRd', vlimits=(0, 5))
        plt.colorbar(cs, ax=ax, orientation='horizontal')
        ax.set_title('RMSE - Spatial Map', fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 3b. RMSE - Boxplot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        _boxplot_metric(ax, target, pred, deep4downscaling.metrics.rmse,
                       var_target='pr', title='RMSE Distribution (Spatial)', 
                       n_members=n_members)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 4. Bias Mean - Spatial Map
        fig = plt.figure(figsize=(12, 8))
        bias_mean = deep4downscaling.metrics.bias_rel_mean(
            target=target,
            pred=pred.isel(member=0),
            var_target='pr'
        )
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cs = _map_plot(ax, bias_mean, var_to_plot='pr',
                      colorbar_cmap='BrBG', vlimits=(-20, 20))
        plt.colorbar(cs, ax=ax, orientation='horizontal')
        ax.set_title('Bias Mean (%) - Spatial Map', fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 4b. Bias Mean - Boxplot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        _boxplot_metric(ax, target, pred, deep4downscaling.metrics.bias_rel_mean,
                       var_target='pr', title='Bias Mean (%) Distribution (Spatial)',
                       n_members=n_members)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 5. Bias R01 - Spatial Map
        fig = plt.figure(figsize=(12, 8))
        bias_rel_r01 = deep4downscaling.metrics.bias_rel_R01(
            target=target,
            pred=pred.isel(member=0),
            var_target='pr'
        )
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cs = _map_plot(ax, bias_rel_r01, var_to_plot='pr',
                      colorbar_cmap='BrBG', vlimits=(-40, 40))
        plt.colorbar(cs, ax=ax, orientation='horizontal')
        ax.set_title('Bias R01 (%) - Spatial Map', fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 5b. Bias R01 - Boxplot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        _boxplot_metric(ax, target, pred, deep4downscaling.metrics.bias_rel_R01,
                       var_target='pr', title='Bias R01 (%) Distribution (Spatial)',
                       n_members=n_members)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 6. Bias RX1day - Spatial Map
        fig = plt.figure(figsize=(12, 8))
        bias_rel_rx1day = deep4downscaling.metrics.bias_rel_rx1day(
            target=target,
            pred=pred.isel(member=0),
            var_target='pr'
        )
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cs = _map_plot(ax, bias_rel_rx1day, var_to_plot='pr',
                      colorbar_cmap='BrBG', vlimits=(-60, 60))
        plt.colorbar(cs, ax=ax, orientation='horizontal')
        ax.set_title('Bias RX1day (%) - Spatial Map', fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 6b. Bias RX1day - Boxplot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        _boxplot_metric(ax, target, pred, deep4downscaling.metrics.bias_rel_rx1day,
                       var_target='pr', title='Bias RX1day (%) Distribution (Spatial)',
                       n_members=n_members)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Evaluation plots saved to: {output_path}")