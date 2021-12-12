#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:24:13 2021

@author: waterlab
"""

import xarray as xr
from glob import glob
import os


def combine_netcdfs(files):
    for model in glob(files):

        variable = []

        for var in glob(model + "/*"):

            paths = sorted(glob(var + "/*.nc"))
            sets = [xr.open_dataset(p, engine='h5netcdf') for p in paths]
            out = xr.concat(sets, "time").sortby(
                "time").transpose("lat", "lon", "time")
            variable.append(out)

        if len(variable) == 2:
            out = xr.merge(variable)

            out.to_netcdf(model + "/"+os.path.basename(model) + ".nc")
            print(f"out {os.path.basename(model)}")
            print(out)


files = "../interpolated_gcms_mon/*"

combine_netcdfs(files)
# JRA_dir = '../downloads/JRA/rawdownloads/'

# uas = xr.open_dataset(JRA_dir + "uas_new.nc",
#                       engine='h5netcdf').transpose("lat", "lon", "time")
# vas = xr.open_dataset(JRA_dir + "vas_new.nc",
#                       engine='h5netcdf').transpose("lat", "lon", "time")

# old = xr.open_dataset(JRA_dir + "../JRA.nc", engine='h5netcdf')

# new = xr.merge([uas, vas]).drop_vars(("hybrid", "step", "valid_time"))

# new = xr.concat((new, old), dim="time").sortby("time")
# new.to_netcdf(JRA_dir + "../JRAq.nc", engine='h5netcdf')


# for file in glob('../downloads/JRA/rawdownloads/**/*.idx', recursive=True):
#     os.remove(file)
# out = None
# print("s")
# out = xr.concat([xr.open_dataset(filepath, engine='cfgrib')
#                 for filepath in glob(JRA_dir + "/*")], dim="time")
# print("h")

# out = out.rename({"v": "vas", "latitude": "lat", "longitude": "lon"})
# out = out.sortby('time').sortby("lat")

# out.to_netcdf(JRA_dir + '/../vas_new.nc', engine="h5netcdf")
# print("over")
