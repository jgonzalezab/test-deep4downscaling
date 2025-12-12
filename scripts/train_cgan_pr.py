# Set paths
DATA_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/input'
FIGURES_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/figures'
MODELS_PATH = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/models'

# Import libraries
import xarray as xr
import torch
from torch.utils.data import DataLoader, random_split

import sys; sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')
import deep4downscaling.viz
import deep4downscaling.trans
import deep4downscaling.deep.loss
import deep4downscaling.deep.utils
import deep4downscaling.deep.models
import deep4downscaling.deep.train
import deep4downscaling.deep.pred
import deep4downscaling.metrics
import deep4downscaling.metrics_ccs

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
loss_function = deep4downscaling.deep.loss.MseLoss(ignore_nans=True)

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
gen_name = 'cgan_generator_pr'
disc_name = 'cgan_discriminator_pr'

generator = deep4downscaling.deep.models.DeepESDpr(x_shape=x_train_stand_arr.shape,
                                                   y_shape=y_train_arr.shape,
                                                   filters_last_conv=1,
                                                   stochastic=False)
discriminator = deep4downscaling.deep.models.DeepESD_Discriminator(x_shape=x_train_stand_arr.shape,
                                                                   y_shape=y_train_arr.shape,
                                                                   filters_last_conv=25)

# Set hyperparameters
num_epochs = 100
learning_rate_gen = 1e-4
learning_rate_disc = 1e-4
lambda_adv = 1.0
lambda_recon = 1.0
freq_train_gen = 1
freq_train_disc = 1
save_checkpoint_every = 10
resume_checkpoint = 20

optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate_gen)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_disc)

# Set device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Train cGAN
train_logs = deep4downscaling.deep.train.standard_cgan_training_loop(
                generator=generator, discriminator=discriminator,
                gen_name=gen_name, disc_name=disc_name, model_path=MODELS_PATH,
                loss_function=loss_function, optimizer_G=optimizer_G,
                optimizer_D=optimizer_D, num_epochs=num_epochs, device=device,
                train_data=train_dataloader, valid_data=valid_dataloader,
                lambda_adv=lambda_adv, lambda_recon=lambda_recon,
                freq_train_gen=freq_train_gen, freq_train_disc=freq_train_disc,
                save_checkpoint_every=save_checkpoint_every,
                resume_checkpoint=resume_checkpoint)

# Load the model weights into the DeepESD architecture
generator.load_state_dict(torch.load(f'{MODELS_PATH}/{gen_name}.pt', 
                      weights_only=True))

# Standardize
x_test_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_test)

# Compute predictions
pred_test = deep4downscaling.deep.pred.compute_preds_standard(
                                x_data=x_test_stand, model=generator,
                                device=device, var_target='pr',
                                mask=y_mask, batch_size=16)

# Visualize the predictions
deep4downscaling.viz.simple_map_plot(data=pred_test.isel(time=100),
                                     colorbar='hot_r', var_to_plot='pr',
                                     output_path=f'{FIGURES_PATH}/prediction_test_day.pdf')

# Compute metrics
bias_rel_rx1day = deep4downscaling.metrics.bias_rel_rx1day(target=y_test, pred=pred_test,
                                                           var_target='pr', season='winter') 

# Visualize the metrics
deep4downscaling.viz.simple_map_plot(data=bias_rel_rx1day,
                                     colorbar='BrBG', var_to_plot='pr',
                                     vlimits=(-60, 60),
                                     output_path=f'{FIGURES_PATH}/bias_rel_rx1day.pdf')