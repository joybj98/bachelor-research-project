#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:37:47 2022

@author: waterlab
"""

import xarray as xr
import numpy as np
import sys
import os

# np.set_printoptions(threshold=sys.maxsize)


bathe = xr.open_dataarray(
    '/home/waterlab/Wang/bachelor_thesis/downloads/bathymetry/gebco_2021_n60.0_s-10.0_w95.0_e180.0.nc')

# print(bathe)
# print(bathe.sel(lon=20,lat=10,method = 'nearest'))
write_dir = '/home/waterlab/Wang/WAVE_MODELS/SWAN/swan4120 wang/inputfiles/wang/bathy.txt'

data = bathe.data

if os.path.exists(write_dir):
    os.remove(write_dir)


with open(write_dir, 'a') as f:
    for lat_row in data:
        f.write(' '.join(str(number) for number in lat_row.tolist()))
        f.write('\n')


# print(bathe.data[0,10])
