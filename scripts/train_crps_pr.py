# Set paths
DATA_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/input'
FIGURES_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/figures'
MODELS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/models'

# Import libraries
import xarray as xr
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

# Load predictand
predictand_filename = f'{DATA_PATH}/pr_AEMET.nc'
predictand = xr.open_dataset(predictand_filename)

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

# Set loss function
reload(deep4downscaling.deep.loss)
loss_function = deep4downscaling.deep.loss.CRPSLoss(ignore_nans=False)

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
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True)

# Create models
model_name = 'deepesd_crps_pr'

reload(deep4downscaling.deep.models)
model = deep4downscaling.deep.models.NoisyDeepESD(x_shape=x_train_stand_arr.shape,
                                                  y_shape=y_train_arr.shape,
                                                  num_channels_noise=3,
                                                  filters_last_conv=1,
                                                  members_for_training=2,
                                                  last_relu=False)

# Set device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Model summary
from torchsummary import summary
summary(model.to(device), x_train_stand_arr.shape[1:])

# Set hyperparameters
num_epochs = 400
learning_rate = 0.0001
patience_early_stopping = 100
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train model
train_loss, val_loss = deep4downscaling.deep.train.standard_training_loop(
                            model=model, model_name=model_name, model_path=MODELS_PATH,
                            device=device, num_epochs=num_epochs,
                            loss_function=loss_function, optimizer=optimizer,
                            train_data=train_dataloader,
                            mixed_precision=True)

# Load the model weights into the DeepESD architecture
model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', 
                      weights_only=True))

# Standardize
x_test_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_test)

# Compute predictions
pred_test = deep4downscaling.deep.pred.compute_preds_standard(
                                x_data=x_test_stand, model=model,
                                device=device, var_target='pr',
                                ensemble_size=4,
                                mask=y_mask, batch_size=16)

# Visualize predictions
reload(utils)
utils.evaluate_and_save_plots(target=y_test,
                              pred=pred_test,
                              output_path=f'{FIGURES_PATH}/evaluation_crps_pr.pdf',
                              time_to_plot=800,
                              n_members=4)