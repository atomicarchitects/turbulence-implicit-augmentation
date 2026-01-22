import matplotlib.pyplot as plt
import os
import numpy as np  
import pyvista as pv
import sys
sys.path.append("./src/")
import global_config
from utils import get_timesteps_available
from tqdm import tqdm

# GPU rendering setup
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'  # Use first GPU (adjust if needed)

pv.start_xvfb()  # NEED THIS for off-screen rendering


def central_difference_derivative(field, axis, dx=1):
   return (np.roll(field, -1, axis) - np.roll(field, 1, axis)) / (2 * dx)

def compute_q_criterion(grid):
   # Vorticity tensor components (anti-symmetric part)
   omega_12 = 0.5 * (grid.cell_data['∂U/∂y'] - grid.cell_data['∂V/∂x'])
   omega_13 = 0.5 * (grid.cell_data['∂U/∂z'] - grid.cell_data['∂W/∂x'])
   omega_23 = 0.5 * (grid.cell_data['∂V/∂z'] - grid.cell_data['∂W/∂y'])
   
   # Strain rate tensor components (symmetric part)
   s_11 = grid.cell_data['∂U/∂x']
   s_22 = grid.cell_data['∂V/∂y']
   s_33 = grid.cell_data['∂W/∂z']
   s_12 = 0.5 * (grid.cell_data['∂U/∂y'] + grid.cell_data['∂V/∂x'])
   s_13 = 0.5 * (grid.cell_data['∂U/∂z'] + grid.cell_data['∂W/∂x'])
   s_23 = 0.5 * (grid.cell_data['∂V/∂z'] + grid.cell_data['∂W/∂y'])
   
   # Q = 0.5 * (|Ω|² - |S|²)
   omega_squared = 2 * (omega_12**2 + omega_13**2 + omega_23**2)
   strain_squared = 2 * (s_12**2 + s_13**2 + s_23**2) + s_11**2 + s_22**2 + s_33**2
   
   return 0.5 * (omega_squared - strain_squared)

def numpy_to_pyvista_all_fields(data_3d, u_mean, v_mean, w_mean):
    c, l, w, h = data_3d.shape
    grid = pv.ImageData(dimensions=(l+1, w+1, h+1))
    
    # Instantaneous velocities
    grid.cell_data['U'] = data_3d[0].flatten(order='F')
    grid.cell_data['V'] = data_3d[1].flatten(order='F')
    grid.cell_data['W'] = data_3d[2].flatten(order='F')
    
    # Time-averaged velocities
    grid.cell_data['U_mean'] = u_mean.flatten(order='F')
    grid.cell_data['V_mean'] = v_mean.flatten(order='F')
    grid.cell_data['W_mean'] = w_mean.flatten(order='F')
    
    # Fluctuating components
    u_prime = data_3d[0] - u_mean
    v_prime = data_3d[1] - v_mean
    w_prime = data_3d[2] - w_mean
    
    grid.cell_data['u_prime'] = u_prime.flatten(order='F')
    grid.cell_data['v_prime'] = v_prime.flatten(order='F')
    grid.cell_data['w_prime'] = w_prime.flatten(order='F')
    
    # Velocity magnitudes
    grid.cell_data['U_mag'] = np.linalg.norm(data_3d, axis=0).flatten(order='F')
    grid.cell_data['u_prime_mag'] = np.sqrt(u_prime**2 + v_prime**2 + w_prime**2).flatten(order='F')
      
    # Turbulent kinetic energy (TKE)
    grid.cell_data['k'] = (0.5 * (u_prime**2 + v_prime**2 + w_prime**2)).flatten(order='F')
    
    grid.cell_data['∂U/∂x'] = central_difference_derivative(data_3d[0], axis=0).flatten(order='F')
    grid.cell_data['∂U/∂y'] = central_difference_derivative(data_3d[0], axis=1).flatten(order='F')
    grid.cell_data['∂U/∂z'] = central_difference_derivative(data_3d[0], axis=2).flatten(order='F')

    grid.cell_data['∂V/∂x'] = central_difference_derivative(data_3d[1], axis=0).flatten(order='F')
    grid.cell_data['∂V/∂y'] = central_difference_derivative(data_3d[1], axis=1).flatten(order='F')
    grid.cell_data['∂V/∂z'] = central_difference_derivative(data_3d[1], axis=2).flatten(order='F')

    grid.cell_data['∂W/∂x'] = central_difference_derivative(data_3d[2], axis=0).flatten(order='F')
    grid.cell_data['∂W/∂y'] = central_difference_derivative(data_3d[2], axis=1).flatten(order='F')
    grid.cell_data['∂W/∂z'] = central_difference_derivative(data_3d[2], axis=2).flatten(order='F')

    grid.cell_data['Q'] = compute_q_criterion(grid)
    
    return grid

def compute_anisotropy_tensor_magnitude_timeseries(data, u_mean, v_mean, w_mean):
    """
    Compute the magnitude of the Reynolds stress anisotropy tensor b_ij
    using time-averaged statistics across the whole time series.
    b_ij = <u'_i u'_j> / (2k) - (1/3) * delta_ij
    |b| = sqrt(b_ij * b_ij)
    """
    # Fluctuating components for all timesteps
    u_prime = data[:, 0] - u_mean
    v_prime = data[:, 1] - v_mean
    w_prime = data[:, 2] - w_mean
    
    # Time-averaged Reynolds stresses
    R11 = np.mean(u_prime * u_prime, axis=0)
    R22 = np.mean(v_prime * v_prime, axis=0)
    R33 = np.mean(w_prime * w_prime, axis=0)
    R12 = np.mean(u_prime * v_prime, axis=0)
    R13 = np.mean(u_prime * w_prime, axis=0)
    R23 = np.mean(v_prime * w_prime, axis=0)
    
    # Turbulent kinetic energy
    k = 0.5 * (R11 + R22 + R33)
    k_safe = np.maximum(k, 1e-10)
    
    # Anisotropy tensor: b_ij = R_ij/(2k) - (1/3)*delta_ij
    b11 = R11 / (2 * k_safe) - 1/3
    b22 = R22 / (2 * k_safe) - 1/3
    b33 = R33 / (2 * k_safe) - 1/3
    b12 = R12 / (2 * k_safe)
    b13 = R13 / (2 * k_safe)
    b23 = R23 / (2 * k_safe)
    
    # Magnitude: sqrt(b_ij * b_ij)
    b_mag = np.sqrt(b11**2 + b22**2 + b33**2 + 2*(b12**2 + b13**2 + b23**2))
    
    return b_mag

def full_filename(numpy_dir, prefix, box_number, timestep):
    if 'channel' in prefix:
        return os.path.join(numpy_dir, prefix + f'_box{box_number}_time{timestep}.npy')
    else:
        return os.path.join(numpy_dir, prefix + f'_boxnum{box_number}_time{timestep}.npy')

def load_box_timeseries(prefix, box_number=1,numpy_dir='numpy_individual_arrays'):
    timesteps = get_timesteps_available(numpy_dir,prefix)
    image_shape = np.load(full_filename(numpy_dir, prefix, box_number, timesteps[0])).shape

    array = np.zeros((len(timesteps),*image_shape))
    for i, timestep in enumerate(timesteps):
        array[i] = np.load(full_filename(numpy_dir, prefix, box_number, timestep))
    return array

# Animation parameters
scalar_field = 'b_mag'  # Change this to any field you want

colormap = 'magma'
vmin = 0
vmax = 0.816
n_colors = 128
prefix = 'channel_nearwall'
box_number = 0
feature_edge_width = 30


for prefix in ['channel_nearwall', 'channel_middle']:
    for box_number in [0,1,2]:
        data = load_box_timeseries(numpy_dir = global_config.numpy_dir, prefix = prefix, box_number=box_number)
        if 'nearwall' in prefix:
            feature_edge_color = 'blue'
        else:
            feature_edge_color = 'red'

        plotter = pv.Plotter(notebook=False, off_screen=True)

        u_mean = np.mean(data[:, 0], axis=0)
        v_mean = np.mean(data[:, 1], axis=0)
        w_mean = np.mean(data[:, 2], axis=0)
        b_mag = compute_anisotropy_tensor_magnitude_timeseries(data, u_mean, v_mean, w_mean)



        epsilon = 0.000001 # small offset so slices are just inside the box

        t = 0
        grid = numpy_to_pyvista_all_fields(data[t], u_mean=u_mean, v_mean=v_mean, w_mean=w_mean)
        grid.cell_data['b_mag'] = b_mag.flatten(order='F')
        bounds = grid.bounds

        # Z slice
        slice_data = grid.slice(normal='z', origin=[0, 0, bounds[5] - epsilon])
        plotter.add_mesh(slice_data, scalars=scalar_field, 
                        clim=[vmin, vmax],
                        cmap=colormap,
                        n_colors=n_colors,
                        show_scalar_bar=False,
                        render=False,
                        lighting=False,
                        )
        plotter.add_mesh(slice_data.extract_feature_edges(), 
                        color=feature_edge_color,      # or 'gray', 'darkgray', etc.
                        line_width=feature_edge_width)     # increase from 0.5
        # X slice
        slice_data = grid.slice(normal='x', origin=[bounds[0] + epsilon, 0, 0]) 
        plotter.add_mesh(slice_data, scalars=scalar_field, 
                        clim=[vmin, vmax],
                        cmap=colormap,
                        n_colors=n_colors,
                        show_scalar_bar=False,
                        render=False,
                        lighting=False,
                        )
        plotter.add_mesh(slice_data.extract_feature_edges(), 
                        color=feature_edge_color,      # or 'gray', 'darkgray', etc.
                        line_width=feature_edge_width)     # increase from 0.5
        # Y slice
        slice_data = grid.slice(normal='y', origin=[0, bounds[3] - epsilon, 0])
        plotter.add_mesh(slice_data, scalars=scalar_field, 
                        clim=[vmin, vmax],
                        cmap=colormap,
                        n_colors=n_colors,
                        show_scalar_bar=False,
                        render=False,
                        lighting=False,
                        )

        # Make edges darker - use a darker color
        plotter.add_mesh(slice_data.extract_feature_edges(), 
                        color=feature_edge_color,      # or 'gray', 'darkgray', etc.
                        line_width=feature_edge_width)     # increase from 0.5
        # Save as PDF

        plotter.camera_position = 'iso'
        plotter.camera.azimuth = 180
        plotter.camera.elevation = 110
        plotter.camera.roll = -180

        plotter.camera.azimuth = 90.3

        plotter.camera.up = (0, 1, 0)  # force y to be vertical

        #plotter.camera.elevation = -12  # negative to flip the vertical view
        plotter.camera.zoom(1.0)  # zoom in/out to fit better
        plotter.screenshot(os.path.join('outputs/plots_final', f'anisotropy_{prefix}_box{box_number}.png'), 
                        window_size=[3000, 3000],  # resolution
                        return_img=False)

        plotter.close()
        print("PDF saved")#