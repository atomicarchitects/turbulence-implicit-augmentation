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

data = load_box_timeseries(numpy_dir = global_config.numpy_dir, prefix = prefix, box_number=box_number)

plotter = pv.Plotter(notebook=False,off_screen=True,window_size=[700, 300])
plotter.open_gif(f'turbulence_animation_{prefix}_box{box_number}.gif')
u_mean = np.mean(data[:, 0], axis=0)
v_mean = np.mean(data[:, 1], axis=0)
w_mean = np.mean(data[:, 2], axis=0)
vmin = 0.8
vmax = 1.2

epsilon = 0.0001 # small offset so slices are just inside the box

for t in range(data.shape[0]):
    plotter.clear()
    grid = numpy_to_pyvista_all_fields(data[t], u_mean=u_mean, v_mean=v_mean, w_mean=w_mean)
    bounds = grid.bounds
    
    slice_data = grid.slice(normal='z', origin=[0, 0, bounds[5] - epsilon])
    plotter.add_mesh(slice_data, scalars=scalar_field, 
                    clim=[vmin, vmax],
                    cmap=colormap,
                    n_colors=n_colors,
                    show_scalar_bar=False,
                    render=False,
                    )
    plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=.5)
    slice_data = grid.slice(normal='x', origin=[bounds[1] - epsilon, 0, 0]) 
    plotter.add_mesh(slice_data, scalars=scalar_field, 
                    clim=[vmin, vmax],
                    cmap=colormap,
                    n_colors=n_colors,
                    show_scalar_bar=False,
                    render=False,
                    )
    plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=.5)
    
    slice_data = grid.slice(normal='y', origin=[0, bounds[3] - epsilon, 0])
    plotter.add_mesh(slice_data, scalars=scalar_field, 
                    clim=[vmin, vmax],
                    cmap=colormap,
                    n_colors=n_colors,
                    show_scalar_bar=False,
                    render=False,
                    )
    plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=.5)
    
    plotter.add_text(f'Time: {t}', position='upper_left')
    
    plotter.write_frame()
    
plotter.close()
print("Animation saved as turbulence_animation.gif")