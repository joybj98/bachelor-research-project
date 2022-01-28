#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:37:31 2021

@author: waterlab
"""

import xarray as xr
from glob import glob
import numpy as np
from time import time

modelName = 'GISS'
path = f"../downloads/{modelName}/"

variables = ['uas', 'vas']


def main():

    print("started")

    uas_files, vas_files = [], []

    for file in glob(path+'*1412.nc'):
        ds = xr.open_dataset(file, engine='netcdf4')
        print(ds)
        if 'uas' in ds:
            uas_files.append(file)
        if 'vas' in ds:
            vas_files.append(file)

    assert len(uas_files) == len(
        vas_files), f'{len(uas_files)},{len(vas_files)}'

    print('start computing...')
    t = time()

    try:
        uas_sum, vas_sum = xr.zeros_like(xr.open_dataset(file, engine='netcdf4').uas), xr.zeros_like(
            xr.open_dataset(file, engine='netcdf4').uas)
    except AttributeError:
        uas_sum, vas_sum = xr.zeros_like(xr.open_dataset(file, engine='netcdf4').vas), xr.zeros_like(
            xr.open_dataset(file, engine='netcdf4').vas)

    uas_sum = uas_sum.rename('uas')
    vas_sum = vas_sum.rename('vas')
    # print(uas_sum, v as_sum)

    for file in uas_files:
        uas = xr.open_dataset(file, engine='netcdf4').uas
        uas_sum += uas

    for file in vas_files:
        vas = xr.open_dataset(file, engine='netcdf4').vas
        vas_sum += vas

    print("opened", f"{time()-t:.04f} sec ")
    t = time()

    uas = uas_sum/len(uas_files)
    vas = vas_sum/len(vas_files)
    # print(uas, vas)
    new = xr.merge([uas, vas]).transpose('lat', 'lon', 'time')

    print("created", f"{time()-t:.04f} sec")
    t = time()

    tests = list(np.random.permutation(uas_files)[:3])
    tests += list(np.random.permutation(vas_files)[:3])
    for test in tests:
        test = xr.open_dataset(uas_files[3], engine='netcdf4').uas.transpose(
            'lat', 'lon', 'time').values.ravel()

        if 'uas' in tests:
            res = np.corrcoef(test, new.uas.data.ravel())[0][1]
            assert res > 0.8, f"tested R for uas is {res}"
        if 'vas' in tests:
            res = np.corrcoef(test, new.vas.data.ravel())[0][1]
            assert res > 0.8, f"tested R for vas is {res}"

    new.to_netcdf(path+f"{modelName}-2014.nc", engine="h5netcdf")
    print("writed", f"{time()-t:.04f} sec")


def concat():
    a = path+'GISS-2000.nc'
    b = path+'GISS-2014.nc'

    a = xr.open_dataset(a, engine='h5netcdf')
    b = xr.open_dataset(b, engine='h5netcdf')

    out = xr.concat([a, b], dim='time')
    out.to_netcdf(path+f"{modelName}.nc", engine="h5netcdf")


if __name__ == "__main__":

    concat()
