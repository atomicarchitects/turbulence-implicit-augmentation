import matplotlib.pyplot as plt
import os
import numpy as np  
import pyvista as pv
from utils import load_box_timeseries
import global_config



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


# Animation parameters
scalar_field = 'U_mag'  # Change this to any field you want

colormap = 'plasma'
n_colors = 32
prefix = 'channel_full_rich'
box_number = 0

data = load_box_timeseries(numpy_dir = global_config.numpy_dir, prefix = prefix, box_number=box_number)[0]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
u_mag = np.sqrt(data[0,:,:,100]**2 + data[1,:,:,100]**2 + data[2,:,:,100]**2)
ax.imshow(u_mag, cmap=colormap,vmin=0.2, vmax = 1.2)
ax.set_title(f'{scalar_field}')
ax.axis('off')
plt.savefig(f'{prefix}_box{box_number}.png')
plt.show()