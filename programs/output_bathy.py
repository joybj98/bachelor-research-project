#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:37:47 2022

@author: waterlab
"""

import xarray as xr
import numpy as np
# import sys
import os

# np.set_printoptions(threshold=sys.maxsize)


bathy = xr.open_dataarray(
    '/home/waterlab/Wang/bachelor_thesis/downloads/bathymetry/gebco_2021_n60.0_s-10.0_w95.0_e180.0.nc')

# print(bathe)
# print(bathe.sel(lon=20,lat=10,method = 'nearest'))
write_dir = '/home/waterlab/Wang/WAVE_MODELS/SWAN/swan4120_wang/inputfiles/wang/bathy.txt'
bathy = bathy.coarsen(lat=10, lon=10).mean()

# print(np.array(bathy.lon[1:])-np.array(bathy.lon[:-1]))

print(float(bathy.lon[0]), float(bathy.lat[0]), 0,
      len(bathy.lon)-1, len(bathy.lat)-1, sep='\n')
print(np.array(bathy.lon[1:])-np.array(bathy.lon[:-1]),
      np.array(bathy.lat[1:])-np.array(bathy.lat[:-1]), sep='\n')
'''
^^^^ these are input for command file
  [xpinp] [ypinp] [alpinp] [mxinp] [myinp] [dxinp] [dyinp]
'''
data = bathy.data

if os.path.exists(write_dir):
    os.remove(write_dir)


with open(write_dir, 'a') as f:
    f.write('\n'.join(' '.join(str(number)
            for number in lat_row.tolist()) for lat_row in data))
