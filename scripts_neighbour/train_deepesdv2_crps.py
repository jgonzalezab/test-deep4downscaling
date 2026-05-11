# Set paths
DATA_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/input'
FIGURES_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/scripts_ci/figures'
MODELS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/models'
PREDS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/preds'
ASYM_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/asym_parameters'

# Import libraries
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from importlib import reload

import sys; sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/src')
import deep4downscaling.viz
import deep4downscaling.trans
import deep4downscaling.deep.loss
import deep4downscaling.deep.utils
import deep4downscaling.deep.models
import deep4downscaling.deep.train
import deep4downscaling.deep.pred
import deep4downscaling.metrics
import deep4downscaling.metrics_ccs
import utils as utils

# Uncertainty approach
uncertainty_approach = 'CRPS'

# Set device
device = 'cuda'

# Load predictors
predictor_filename = f'{DATA_PATH}/ERA5_NorthAtlanticRegion_1-5dg_full.nc'
predictor = xr.open_dataset(predictor_filename)
predictor = predictor.load()

# Subset predictors 
predictor = predictor.sel(lon=slice(-24, 22.5))

# Extent lattiude to 32 grid points
lat = predictor.lat.values
dlat = np.diff(lat).mean()
n_extra = 32 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
predictor = predictor.reindex(lat=new_lat, method='nearest')

# Load predictand
predictand_filename = f'{DATA_PATH}/pr_AEMET.nc'
predictand = xr.open_dataset(predictand_filename)
predictand = predictand.load()

# Subset predictand
predictand = predictand.sel(lon=slice(-9.425, 3.375))

# Extend the latitude to 264 grid points
lat = predictand.lat.values
dlat = np.diff(lat).mean()

n_extra = 256 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])

predictand = predictand.reindex(lat=new_lat)

# Remove days with nans in the predictor
predictor = deep4downscaling.trans.remove_days_with_nans(predictor)

# Align both datasets in time
predictor, predictand = deep4downscaling.trans.align_datasets(predictor, predictand, 'time')

# Split data into training and test sets
years_train = ('1980', '2010')
years_test = ('2011', '2020')

x_train = predictor.sel(time=slice(*years_train))
y_train = predictand.sel(time=slice(*years_train))

x_test = predictor.sel(time=slice(*years_test))
y_test = predictand.sel(time=slice(*years_test))

# Standardize the predictors
x_train_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)

# Set valid mask for the predictand
y_mask = deep4downscaling.trans.compute_valid_mask(y_train)

# Stack the predictand and the mask
y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

# Filter the predictand and the mask to only include valid grid points
y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)
y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                         drop=True)

# Convert the data to numpy arrays
x_train_stand_arr = deep4downscaling.trans.xarray_to_numpy(x_train_stand)
y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stack_filt)

# Create Dataset
train_dataset = deep4downscaling.deep.utils.StandardDataset(x=x_train_stand_arr,
                                                            y=y_train_arr)

# Split into training and validation sets
train_dataset, valid_dataset = random_split(train_dataset,
                                            [0.9, 0.1])

# Create DataLoaders
batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True)

# Set model name
model_name = f'deepesdv2_{uncertainty_approach}'

# Number of output grid points
n_out = y_train_arr.shape[1]

patch_size = 2

# Prepare decoder grid coordinates for NeRF positional encoding
# Normalize lat/lon to [-1, 1] based on the region's extent
d0 = y_train_stack_filt['lat'].values.astype(np.float64)
d1 = y_train_stack_filt['lon'].values.astype(np.float64)

lat_min, lat_max = d0.min(), d0.max()
lon_min, lon_max = d1.min(), d1.max()

d0_norm = 2.0 * (d0 - lat_min) / (lat_max - lat_min) - 1.0
d1_norm = 2.0 * (d1 - lon_min) / (lon_max - lon_min) - 1.0

grid_coords = torch.tensor(np.stack([d0_norm, d1_norm], axis=-1),
                            dtype=torch.float32)
print(f"NeRF positional encoding (decoder): using {grid_coords.shape[0]} grid coordinates "
      f"(normalized to [-1, 1] based on region)")

# Prepare encoder grid coordinates (patch centers) for NeRF positional encoding
first_var = next(iter(predictor.data_vars))
pred_spatial_dims = [d for d in predictor[first_var].dims if d != 'time']
dim0 = predictor[pred_spatial_dims[0]].values.astype(np.float64)
dim1 = predictor[pred_spatial_dims[1]].values.astype(np.float64)
n_h = len(dim0) // patch_size
n_w = len(dim1) // patch_size

patch_coords = np.empty((n_h * n_w, 2))
for i in range(n_h):
    for j in range(n_w):
        patch_coords[i * n_w + j, 0] = dim0[i * patch_size:(i + 1) * patch_size].mean()
        patch_coords[i * n_w + j, 1] = dim1[j * patch_size:(j + 1) * patch_size].mean()

enc_lat_min, enc_lat_max = patch_coords[:, 0].min(), patch_coords[:, 0].max()
enc_lon_min, enc_lon_max = patch_coords[:, 1].min(), patch_coords[:, 1].max()

patch_coords[:, 0] = 2.0 * (patch_coords[:, 0] - enc_lat_min) / (enc_lat_max - enc_lat_min) - 1.0
patch_coords[:, 1] = 2.0 * (patch_coords[:, 1] - enc_lon_min) / (enc_lon_max - enc_lon_min) - 1.0

encoder_grid_coords = torch.tensor(patch_coords, dtype=torch.float32)
print(f"NeRF positional encoding (encoder): using {encoder_grid_coords.shape[0]} patch centers "
      f"(normalized to [-1, 1] based on region)")

# Create model
model = deep4downscaling.deep.models.NoisyDeepESDv2(x_shape=x_train_stand_arr.shape,
                                                    n_out=n_out,
                                                    patch_size=patch_size,
                                                    dim=128,
                                                    depth=4,
                                                    num_heads=4,
                                                    mlp_dim=256,
                                                    decoder_depth=1,
                                                    noise_channels=3,
                                                    noise_dim=128,
                                                    query_dim=128,
                                                    decoder_cross_attn_mlp=False,
                                                    grid_coords=grid_coords,
                                                    encoder_grid_coords=encoder_grid_coords,
                                                    nerf_num_frequencies=8,
                                                    share_nerf_encoding=False,
                                                    members_for_training=2,
                                                    last_relu=True)

from torchsummary import summary

print("Model Summary:")
model.to(device)
if isinstance(model, torch.nn.DataParallel):
    summary(model.module, input_size=x_train_stand_arr.shape[1:], device=device)
else:
    summary(model, input_size=x_train_stand_arr.shape[1:], device=device)
    
# Wrap the model for multi-GPU training if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

# Set hyperparameters
num_epochs = 10000
learning_rate = 0.0001
patience_early_stopping = 40

# Set loss function
loss_function = deep4downscaling.deep.loss.CRPSLoss(ignore_nans=True)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
train_loss, val_loss = deep4downscaling.deep.train.standard_training_loop(model=model, 
                                                                          model_name=model_name, 
                                                                          model_path=MODELS_PATH,
                                                                          device=device, 
                                                                          num_epochs=num_epochs,
                                                                          loss_function=loss_function, 
                                                                          optimizer=optimizer,
                                                                          train_data=train_dataloader,
                                                                          valid_data=valid_dataloader,
                                                                          patience_early_stopping=patience_early_stopping,
                                                                          mixed_precision=True)

# Load the model weights into the ViT architecture
model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', weights_only=True))

# Standardize the test data
x_test_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_test)

# Compute predictions (use the y_train-based mask to match the filtered model output)
pred_test = deep4downscaling.deep.pred.compute_preds_standard(x_data=x_test_stand, model=model, device=device,
                                                              ensemble_size=10, var_target='pr', mask=y_mask, batch_size=16)

# Save the predictions
pred_test.to_netcdf(f'{PREDS_PATH}/{model_name}.nc')
