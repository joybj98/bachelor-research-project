#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:06:04 2022

@author: waterlab
"""

import xarray as xr
import numpy as np
import sys
import os

MIROC6 = '/home/waterlab/Wang/bachelor_thesis/interpolated_gcms_mon/MIROC6/MIROC6.nc'
write_dir = '/home/waterlab/Wang/WAVE_MODELS/SWAN/swan4120 wang/inputfiles/wang/MIROC6.txt'

ds = xr.open_dataset(MIROC6, engine='h5netcdf')
ds = ds.isel(time=0)

uas = ds.isel(lon=3, lat=5).uas
print(uas)

uas = ds.uas.data
vas = ds.vas.data


print(uas.shape)

if os.path.exists(write_dir):
    os.remove(write_dir)


with open(write_dir, 'a') as f:
    for lat_row in uas:
        f.write(' '.join(str(number) for number in lat_row.tolist()))

        f.write('\n')
    for lat_row in vas:

        f.write(' '.join(str(number) for number in lat_row.tolist()))
        f.write('\n')


# print(uas[3, 5])
# print(uas, vas)
