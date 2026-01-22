from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *
import os
import numpy as np
from glob import glob
import time
from functools import wraps
import random
random.seed(42)

# Domain dimensions (channel)
# Lx = 8*np.pi
# Ly = 2
# Lz = 3*np.pi
auth_token = os.getenv('JHTDB_AUTH_TOKEN')
jhtdb_dataset_title = 'channel'
save_prefix = 'channel_nearwall'
output_path = f'/data/NFS/potato/turbulence/numpy_4000'
n_procs = 10
Nxy = 64
Nz = 64
num_boxes = 3
T_start = 0 # for channel dataset
T_end = 25.9935 # for channel dataset
num_timesteps = 4000
spatial_interpolation = 'lag8' # no temporal interpolation is used 
y_max = 0.95 # 0.3 for middle, 0.95 for nearwall
side_length = 0.6
domain_x_range = (0, 8*np.pi)
domain_z_range = (0, 3*np.pi)

def generate_boxes_fixedy(side_length, x_range, z_range, num_boxes, y_max):
    boxes = []
    ymin, ymax = y_max - side_length, y_max
    print(f'Generating {num_boxes} boxes with ymin {ymin}, ymax {ymax}')

    for box_num in range(num_boxes):
        if box_num == 0: # first box in centre
            xmin = (x_range[1]-x_range[0])/2 - side_length/2
            zmin = (z_range[1]-z_range[0])/2 - side_length/2
            xmax, zmax = xmin + side_length, zmin + side_length
            boxes.append((xmin, xmax, ymin, ymax, zmin, zmax))
            continue
        for _ in range(10000000):
            xmin = random.uniform(x_range[0], x_range[1] - side_length)
            zmin = random.uniform(z_range[0], z_range[1] - side_length)
            xmax, zmax = xmin + side_length, zmin + side_length
            
            # Check no overlap with existing boxes
            if all(xmax <= box[0] or xmin >= box[1] or zmax <= box[4] or zmin >= box[5] 
                for box in boxes):
                boxes.append((xmin, xmax, ymin, ymax, zmin, zmax))
                break
    
    return boxes

def create_query_points_from_box(Nxy, Nz, xmin, xmax, ymin, ymax, zmin, zmax):
    """Create all query points for the domain"""
    x_points = np.linspace(xmin, xmax, Nxy)
    y_points = np.linspace(ymin, ymax, Nxy)
    z_points = np.linspace(zmin, zmax, Nz)
    X, Y, Z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return points, (Nxy, Nxy, Nz), (x_points, y_points, z_points)

boxes = generate_boxes_fixedy(side_length = side_length,
                       x_range = domain_x_range,
                       z_range = domain_z_range,
                       num_boxes = num_boxes,
                       y_max = y_max)


if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset = turb_dataset(dataset_title=jhtdb_dataset_title, 
                        output_path=output_path, 
                        auth_token=auth_token)


times = np.linspace(T_start, T_end, num_timesteps)

from multiprocessing import Pool

def process_time_step(args):
    t_idx, current_time, dataset, spatial_interpolation, query_points, nx, ny, nz, output_path, save_prefix, box_num = args
    if os.path.exists(f'{output_path}/{save_prefix}_box{box_num}_time{t_idx}.npy'):
        print(f'{output_path}/{save_prefix}_box{box_num}_time{t_idx}.npy already exists, skipping')
        return
    try:
        result_vel = getData(dataset, 
                        'velocity',
                        current_time,
                        'none',
                        spatial_interpolation,
                        'field',
                        query_points)
        vel_data = np.array(result_vel).reshape(nx, ny, nz, 3).transpose(3, 0, 1, 2)
        np.save(f'{output_path}/{save_prefix}_box{box_num}_time{t_idx}.npy', vel_data)
    except Exception as e:
        print(f'Error processing time step {t_idx} for box {box_num}: {e}')
        return

looking_good = False
while not looking_good:
    for box_num, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(boxes):
        query_points, (nx, ny, nz), (x_points, y_points, z_points) = create_query_points_from_box(Nxy, Nz, xmin, xmax, ymin, ymax, zmin, zmax)
        
        args_list = [(t_idx, current_time, dataset, spatial_interpolation, query_points, 
                    nx, ny, nz, output_path, save_prefix, box_num) 
                    for t_idx, current_time in enumerate(times)]

        with Pool(n_procs) as pool:
            pool.map(process_time_step, args_list)

        print('Verifying that all files exist....')
        all_boxes_are_there = True
    for box_num, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(boxes):
        print(f'Box {box_num} with xmin {xmin}, xmax {xmax}, ymin {ymin}, ymax {ymax}, zmin {zmin}, zmax {zmax}')
        for t_idx, current_time in enumerate(times):
            if not os.path.exists(f'{output_path}/{save_prefix}_box{box_num}_time{t_idx}.npy'):
                print(f'Uh oh! {output_path}/{save_prefix}_box{box_num}_time{t_idx}.npy does not exist!!')
                all_boxes_are_there = False

    if all_boxes_are_there:
        looking_good = True 
        print('All boxes are there!')    

#if looks_good: print(f'LGTM :)')



        