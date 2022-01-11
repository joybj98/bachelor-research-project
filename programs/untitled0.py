#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:39:26 2022

@author: waterlab
"""

# import xarray as xr
import numpy as np

# dsa = xr.open_dataset(
#     '/home/waterlab/Wang/bachelor_thesis/downloads/bathymetry/gebco_2021_n60.0_s-10.0_w95.0_e180.0.nc', )

# dsb = xr.open_dataarray(
#     '/home/waterlab/Wang/bachelor_thesis/downloads/bathymetry/gebco_2021_tid_n60.0_s-10.0_w95.0_e180.0.nc',)

# lon = np.array(dsa.lon)
# lat = np.array(dsa.lat)
# dlon = lon[1:]-lon[:-1]
# dlat = lat[1:]-lat[:-1]

a = np.array(range(12)).reshape(3, 4)
a_ = a[:, [2, 3, 0, 1, 1, 1, 1, 1, 1]]
