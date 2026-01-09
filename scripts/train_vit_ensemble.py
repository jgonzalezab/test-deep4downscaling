# Set paths
DATA_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/input'
FIGURES_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/figures'
MODELS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/models'

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

# Stack the predictand
y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))

# Deep Ensembles parameters 
n_ensemble_members = 5  # Number of ensemble members 
use_adversarial_training = True  # Use adversarial training
epsilon = 0.01  # Perturbation magnitude for adversarial training (1% of input range)

# Set loss function: NLLGaussianLoss for proper scoring rule
reload(deep4downscaling.deep.loss)
loss_function = deep4downscaling.deep.loss.NLLGaussianLoss(ignore_nans=True)

# Convert the data to numpy arrays
x_train_stand_arr = deep4downscaling.trans.xarray_to_numpy(x_train_stand)
y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stack)

# Create Dataset
train_dataset = deep4downscaling.deep.utils.StandardDataset(x=x_train_stand_arr,
                                                            y=y_train_arr)

# Split into training and validation sets
train_dataset, valid_dataset = random_split(train_dataset,
                                            [0.9, 0.1])

# Create DataLoaders
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True)
# Set device
device = 'cuda'

# Set hyperparameters
num_epochs = 1000
learning_rate = 0.0001
patience_early_stopping = 60

# Store trained models
ensemble_models = []

# Train each ensemble member independently
for member_idx in range(n_ensemble_members):
    
    # Create model with stochastic output for NLLGaussianLoss
    reload(deep4downscaling.deep.models)
    model = deep4downscaling.deep.models.ViT(x_shape=x_train_stand_arr.shape,
                                             y_shape=y_train_arr.shape,
                                             patch_size=2,
                                             dim=768,
                                             depth=12,
                                             num_heads=12,
                                             mlp_dim=3072,
                                             orog=None,
                                             last_relu=False,
                                             stochastic=True)
    
    # Wrap the model for multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Print model summary only for the first member
    if member_idx == 0:
        from torchsummary import summary
        summary(model if not isinstance(model, torch.nn.DataParallel) else model.module, 
                x_train_stand_arr.shape[1:])
    
    # Set model name
    model_name = f'vit_ensemble_member_{member_idx + 1}'
    
    # Initialize optimizer (fresh initialization for each member)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Select training function based on adversarial training flag
    if use_adversarial_training:
        train_loss, val_loss = deep4downscaling.deep.train.adversarial_training_loop(model=model, 
                                                                                     model_name=model_name, 
                                                                                     model_path=MODELS_PATH,
                                                                                     device=device, 
                                                                                     num_epochs=num_epochs,
                                                                                     loss_function=loss_function, 
                                                                                     optimizer=optimizer,
                                                                                     train_data=train_dataloader,
                                                                                     valid_data=valid_dataloader,
                                                                                     patience_early_stopping=patience_early_stopping,
                                                                                     mixed_precision=True,
                                                                                     epsilon=epsilon)
    else:
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
    
    # Load the model weights
    model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', 
                          weights_only=True))
    
    # Store the trained model
    ensemble_models.append(model)
    
    print(f"\nMember {member_idx + 1} training completed!")

# Standardize test data
x_test_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_test)

# Compute mask
y_mask = deep4downscaling.trans.compute_valid_mask(y_test)

# Compute ensemble predictions (combines mean and variance from all members)
# pred_test = deep4downscaling.deep.pred.compute_preds_deep_ensemble(x_data=x_test_stand, 
#                                                                    models=ensemble_models,
#                                                                    device=device, 
#                                                                    var_target='pr',
#                                                                    mask=y_mask, 
#                                                                    batch_size=8)

pred_test = deep4downscaling.deep.pred.compute_preds_gaussian(x_data=x_test_stand, 
                                                              model=ensemble_models[0],
                                                              device=device, 
                                                              var_target='pr',
                                                              mask=y_mask,
                                                              ensemble_size=4,
                                                              batch_size=8)

# Visualize predictions
reload(utils)
utils.evaluate_and_save_plots(target=y_test,
                              pred=pred_test,
                              output_path=f'{FIGURES_PATH}/evaluation_vit_ensemble_pr.pdf',
                              time_to_plot=800)