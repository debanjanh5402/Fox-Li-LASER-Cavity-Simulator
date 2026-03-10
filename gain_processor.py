# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

def load_and_process_gain(filepath, X_cavity, Y_cavity, scaling_factor=1.0):

    data = pd.read_csv(filepath, sep="\t", header=0)
    data.columns = ["x", "z", "y", "gain"]
    data[["x", "y"]] /= 10.0
    x = data['x']; y = data['y']

    x_unique = np.unique(x); xmean = (x_unique.max()+x_unique.min())/2
    y_unique = np.unique(y); ymean = (y_unique.max()+y_unique.min())/2
    data = data.sort_values(by=['x', 'y', 'z'])

    def integrate_group(group):
        return np.trapezoid(group['gain'], x=group['z'])
    
    cumulative_gain = data.groupby(['x', 'y']).apply(integrate_group, include_groups=False).reset_index(name='gain')

    gain_2d = cumulative_gain.pivot(index="y", columns="x", values="gain")
    gain_2d = np.nan_to_num(gain_2d, nan=0.0)

    X, Y = np.meshgrid(x_unique, y_unique)

    gain_grid = griddata((X.flatten()-xmean, Y.flatten()-ymean), 
                         gain_2d.flatten(), (X_cavity, Y_cavity), 
                         method='linear', fill_value=0)
    
    gain_grid = np.nan_to_num(gain_grid, nan=0.0)
    exp_gain_grid = np.exp(gain_grid)

    return exp_gain_grid, gain_grid