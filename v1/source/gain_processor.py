# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

def load_and_process_gain(filepath, X_cavity, Y_cavity, scaling_factor=100.0):
    data = pd.read_csv(filepath, sep="\t", header=0)
    data.columns = ["x", "z", "y", "gain"]
    # Proper Scaling
    data[["x", "y"]] /= 10.0
    data["gain"] *= scaling_factor
    x_unique = np.unique(data['x'])
    y_unique = np.unique(data['y'])
    xmean = (x_unique.max() + x_unique.min()) / 2
    ymean = (y_unique.max() + y_unique.min()) / 2
    data = data.sort_values(by=['x', 'y', 'z'])
    z = data['z'].values
    g = data['gain'].values
    dz = np.diff(z)
    avg_g = (g[:-1] + g[1:]) / 2.0
    traps = dz * avg_g
    diff_mask = (data['x'].values[:-1] != data['x'].values[1:]) | (data['y'].values[:-1] != data['y'].values[1:])
    traps[diff_mask] = 0
    data['trap_area'] = np.concatenate([[0], traps])
    cumulative_gain = data.groupby(['y', 'x'], sort=False)['trap_area'].sum().reset_index(name='gain')
    gain_2d = cumulative_gain.pivot(index="y", columns="x", values="gain").values
    gain_2d = np.nan_to_num(gain_2d, nan=0.0)
    X, Y = np.meshgrid(x_unique, y_unique)
    gain_grid = griddata(np.stack([X.ravel() - xmean, Y.ravel() - ymean], axis=-1), 
                         (gain_2d).ravel(), 
                         (X_cavity, Y_cavity), 
                         method='linear', fill_value=0)
    gain_grid = np.nan_to_num(gain_grid, nan=0.0)
    return np.exp(gain_grid), gain_grid