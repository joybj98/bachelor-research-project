#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:09:06 2022

@author: shiyaun wang
"""

import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import os
from glob import glob

LATBOUND = [-10, 60]
LONBOUND = [95, 180]

# LATBOUND = [50, 60]
# LONBOUND = [160, 180]

STARTDATE = '1960-01-01'
ENDDATE = '2018-01-01'


def droppingVariablesOfds(ds, distToBound=0):

    return ds.sel(lat=slice(LATBOUND[0]-distToBound, LATBOUND[1]+distToBound), lon=slice(LONBOUND[0]-distToBound, LONBOUND[1]+distToBound))


def preprocessed(ds, modelType='', startdate=None, enddate=None, variableType=None, drop=False):
    def JRA_preprocess(ds):
        if "u" in ds:
            ds = ds.rename({"u": "uas"})
        if "v" in ds:
            ds = ds.rename({"v": "vas"})
        ds = ds.rename(
            {'latitude': 'lat', 'longitude': 'lon'})
        ds = ds.sortby('lat')
        ds = droppingVariablesOfds(ds)
        ds = ds.transpose('lat', 'lon', 'time')
        return ds

    def MRI_preprocess(ds):
        # ds = ds.drop_vars(['time_bnds', 'lon_bnds', 'lat_bnds'])
        ds["time"] = ds.time.to_index().map(lambda t: t.replace(day=1, hour=0))

        ds = ds.transpose('lat', 'lon', 'time')

        return ds

    def MIROC6_preprpocess(ds):
        return MRI_preprocess(ds)

    def CNRM_ESM_preprocess(ds):
        # ds = ds.drop_vars(['time_bounds'])
        ds["time"] = ds.time.to_index().map(lambda t: t.replace(day=1, hour=0))

        ds = ds.transpose('lat', 'lon', 'time')
        return ds

    def INM_preprocess(ds):
        return MRI_preprocess(ds)

    def GISS_preprocess(ds):
        return MRI_preprocess(ds)

    def unknown_preprocess(ds):
        try:
            ds = MRI_preprocess(ds)
        except Exception:

            raise ValueError(
                'The model cannot be preprocessed. Please do preprocessing by yourself')
        else:
            # print(ds)
            print('had better not preprocess by this')

        return ds

    model_list = {'JRA': JRA_preprocess,
                  'INM': INM_preprocess,
                  'GISS': GISS_preprocess,
                  'MRI': MRI_preprocess,
                  'MIROC6': MIROC6_preprpocess,
                  'CNRM-ESM2-1': CNRM_ESM_preprocess}

    def main_preprocess(ds, modelType=modelType, startdate=startdate, enddate=enddate, variableType=variableType):
        if startdate and enddate:
            ds = ds.sel(time=slice(startdate, enddate))

        if variableType:
            ds = ds[[variableType]]

        if drop:
            ds = droppingVariablesOfds(ds)
        # print(ds)

        ds = model_list.get(modelType, unknown_preprocess)(ds)

        return ds
    print(f'{modelType} has been preprocessed')

    return main_preprocess(ds)


def findNNearest(grid_ds, value_da, N=4):

    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers is 6371
        return c

    X, Y = np.meshgrid(value_da.lat, value_da.lon)
    grids = np.vstack(list(zip(X.ravel(), Y.ravel())))
    kd_tree = cKDTree(grids)

    Xt, Yt = np.meshgrid(grid_ds.lat, grid_ds.lon)

    targets = np.vstack(list(zip(Xt.ravel(), Yt.ravel())))

    _, indexes = kd_tree.query(targets, k=30*N)
    '''
    found 10*N nearest in points by lat and lon for each target
    indexes.shape is (len(targets), k)
    '''

    points_data = [[[None]*2]*N]*len(targets)
    weights_data = [[None]*N]*len(targets)
    for i, _ in enumerate(zip(targets, grids[indexes])):
        target, points = _[0], _[1]
        '''
        for every target in grid_ds, find the indexes of N lat-lon nearest points. And compute the their weights.
        '''

        # print(len(points))
        dists = [haversine(target[0], target[1], p[0], p[1]) for p in points]

        dists, nearestPoints = (list(t)
                                for t in zip(*sorted(zip(dists, points))[:N]))
        '''
        for 3N of lat-lon nearest points, compute the distance on the sphere of
        earth. Then find N nearest points by sphere distance.
        '''

        weights = 1/(np.array(dists)**2)

        weights = weights/sum(weights)  # to make the sum of weight to 1.

        weights_data[i] = weights
        points_data[i] = nearestPoints

    return targets, points_data, weights_data


def IDWInterpolation(grid_ds, value_da, N=4, exponent=2):

    targets, locs_all, weights_all = findNNearest(grid_ds, value_da, N=N)

    lat = grid_ds.lat
    lon = grid_ds.lon

    time = value_da.time

    if 'variable' in value_da.coords:
        variable = value_da['variable']
    else:
        variable = ['nothing']

    values = np.array([np.nan]*len(lat)*len(lon)*len(time)*len(variable)
                      ).reshape((len(lat), len(lon), len(time), len(variable)))

    def getValue(value_da, locs, weights):
        '''


        Parameters
        ----------
        value_da : xr.DataArray
            the dataArray that to be interpolated. 

            The interpolation would be implemented for every time step
            (and every varibale) by the same method (and weights)

        locs : list or np.ndarray
            locations of N nearest points. the shape should be (N,2)
        weights : list or np.ndarray
            weights of N nearest points. The shape should be (N,)

        Returns
        -------
        prediction : np.ndarray
            the shape would be len(time)

        '''

        locs = np.array(locs)
        values = list(value_da.sel(
            lat=lat, lon=lon, method='nearest') for lat, lon in locs[:, :])

        prediction = sum([w*v for w, v in zip(weights, values)])

        return prediction

    print('all iters:', len(locs_all))
    values = np.array([getValue(value_da, locs, weights)
                       for locs, weights in tqdm(zip(locs_all, weights_all))])

    # print(values.shape) -> (lon*lat, variable, time)

    values = values.reshape((len(lon), len(lat), len(variable), len(time)))
    values = np.transpose(values, (1, 0, 3, 2))

    res = xr.DataArray(
        values, coords=[lat, lon, time, variable], dims=['lat', 'lon', 'time', 'variable'])
    return res


def getName(path):
    return os.path.splitext(os.path.basename(path))[0]


def main(filepath, grid_ds, write=False):

    modelType = getName(filepath)
    # print(modelType)

    value_ds = xr.open_dataset(filepath, engine='h5netcdf')
    value_ds = preprocessed(value_ds, modelType, STARTDATE, ENDDATE)
    value_da = value_ds.to_array()

    res = IDWInterpolation(grid_ds, value_da)
    res = res.to_dataset('variable')

    write_dir = f'../interpolated_gcms_mon/{modelType}/'

    if write:

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        res.to_netcdf(write_dir+f'{modelType}.nc', engine='h5netcdf')

    return res


def checking(t, variable):
    '''
    Let's check some samples of the interpotaion result manually

    '''
    # import cartopy.crs as ccrs
    # fig, axis = plt.subplots(2)

    write_dir = '../interpolated_gcms_mon/INM/INM.nc'
    write_ds = xr.open_dataset(write_dir, engine='h5netcdf')
    # write_ds = write_ds.to_array()
    w = write_ds[variable].isel(time=t)

    # plt.title('sdfsfe')
    # write_ds['vas'].isel(time=t).plot(ax=axis[1, 1])

    value_ds = xr.open_dataset(gcm_dir+'INM/INM.nc', engine='h5netcdf')
    value_ds = preprocessed(value_ds, 'INM', STARTDATE, ENDDATE)
    value_ds = droppingVariablesOfds(value_ds).drop('height')
    v = value_ds[variable].isel(time=t)

    xr.plot.pcolormesh(w, vmax=9, vmin=-9, cmap='RdBu_r',
                       xlim=(95, 180), ylim=(-10, 60), figsize=(8, 6))

    # value_ds['vas'].isel(time=t).plot(ax=axis[1, 0])
    plt.title('Interpolated')
    plt.show()


if __name__ == '__main__':

    gcm_dir = '../downloads/GCMs/'

    grid_dir = '../downloads/JRA/JRA.nc'

    grid_ds = xr.open_dataset(grid_dir, engine='h5netcdf')['uas']

    # grid_ds = preprocessed(grid_ds, 'JRA')
    # grid_ds = droppingVariablesOfds(grid_ds)

    # for filepath in glob(gcm_dir+'*/*.nc'):

    #     modelType = getName(filepath)
    #     print(filepath)

    #     if os.path.exists(f'../interpolated_gcms_mon/{modelType}/{modelType}_new.nc'):
    #         continue

    # res = main(filepath, grid_ds, write=True)
    # print(res)
    checking(24, 'uas')
    # for i in range(24):
    #     checking(, 'uas')
