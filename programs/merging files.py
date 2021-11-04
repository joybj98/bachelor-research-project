#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:33:20 2021

@author: wangshiyuan
"""

import glob
import xarray as xr
import os
import matplotlib.pyplot as plt

# Regarding to this function, see 
# http://xarray.pydata.org/en/stable/user-guide/io.html#reading-multi-file-datasets
# I somehow simlified the method in the URL


def combine_grib1(files, dim):
    def process_one_path(path):
        with xr.open_dataset(path, engine = 'cfgrib') as ds:
            ds.load()
            print(f'{os.path.basename(path)} has been appended')
            return ds
    for idx in glob.glob(files + '.idx'):
        os.remove(idx)
    paths = sorted(glob.glob(files + '.glob1'))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined




files = '/Users/wangshiyuan/Downloads/1981-2005_u/u_new/*'

combined = combine_grib1(files, dim = 'time')
print(combined)

