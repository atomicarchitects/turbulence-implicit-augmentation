import numpy as np

from utils import get_timesteps_available, full_filename
from data import NumpyTimeSeriesDataset
import global_config as global_config



dataset_name = 'decaying_boxfilter_fs8_lowres'
timesteps = get_timesteps_available(global_config.numpy_dir, dataset_name)
input_files = [full_filename(global_config.numpy_dir, dataset_name, box_number, timestep) for box_number in range(32) for timestep in timesteps]
target_files = [full_filename(global_config.numpy_dir, dataset_name, box_number, timestep) for box_number in range(32) for timestep in timesteps]

dataset = NumpyTimeSeriesDataset(
    input_file_list=input_files,
    target_file_list=target_files,
    means_stds=None,
)

print(dataset_name)
print(dataset.get_means_stds())

dataset_name = 'decaying_spectralfilter_fs8_lowres'
timesteps = get_timesteps_available(global_config.numpy_dir, dataset_name)
input_files = [full_filename(global_config.numpy_dir, dataset_name, box_number, timestep) for box_number in range(32) for timestep in timesteps]
target_files = [full_filename(global_config.numpy_dir, dataset_name, box_number, timestep) for box_number in range(32) for timestep in timesteps]

dataset = NumpyTimeSeriesDataset(
    input_file_list=input_files,
    target_file_list=target_files,
    means_stds=None,
)

print(dataset_name)
print(dataset.get_means_stds())

from src.utils import load_box_timeseries

data = load_box_timeseries('forced_spectralfilter_fs4_highres', box_number=1, numpy_dir=global_config.numpy_dir)

print(data[:,0,0,0,0].mean(), data[:,0,0,0,0].std())
print(data[0,0,:,:,:].mean(), data[0,0,:,:,:].std())
print(data.shape)
# Remove mean flow first
u = data - data.mean(axis=(0,2,3,4), keepdims=True)

# 1. Basic ergodic check (time vs ensemble average)
print(f"Ergodic: |{u[:,0,32,32,32].mean():.6f} - {u[0,0,:,:,:].mean():.6f}| = {abs(u[:,0,32,32,32].mean() - u[0,0,:,:,:].mean()):.6f}")

# 2. Multiple random points check
print(f"Multi-point ergodic spread: {np.std([u[:,0,np.random.randint(64),np.random.randint(64),np.random.randint(64)].mean() for _ in range(10)]):.6f}")

# 3. Variance ergodicity
print(f"Variance ergodic: |{u[:,0,32,32,32].var():.6f} - {u[0,0,:,:,:].var():.6f}| = {abs(u[:,0,32,32,32].var() - u[0,0,:,:,:].var()):.6f}")

# 4. All components together
print(f"All components: {[abs(u[:,i,32,32,32].mean() - u[0,i,:,:,:].mean()) for i in range(3)]}")

# 5. Autocorrelation decay check
print(f"Autocorr[10]/Autocorr[0]: {np.corrcoef(u[:-10,0,32,32,32], u[10:,0,32,32,32])[0,1]:.4f}")