#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:09:06 2022

@author: waterlab
"""


import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm


LATBOUND = [-10, 60]
LONBOUND = [95, 180]

# LATBOUND = [50, 60]
# LONBOUND = [160, 180]

STARTDATE = '1960-01-01'
ENDDATE = '2018-01-01'


def droppingVariablesOfds(ds, distToBound=0):

    return ds.sel(lat=slice(LATBOUND[0]-distToBound, LATBOUND[1]+distToBound), lon=slice(LONBOUND[0]-distToBound, LONBOUND[1]+distToBound))


def preprocessed(ds, modelType='', startdate=None, enddate=None, variableType=None):
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
        # print('here\n\n\n\n')
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

    def unknown_preprocess(ds):

        raise ValueError(
            'The model cannot be preprocessed. Please do preprocessing by yourself')

        return ds

    model_list = {'JRA': JRA_preprocess,
                  'MRI': MRI_preprocess,
                  'MIROC6': MIROC6_preprpocess,
                  'CNRM-ESM2-1': CNRM_ESM_preprocess}

    def main_preprocess(ds, modelType=modelType, startdate=startdate, enddate=enddate, variableType=variableType):
        if startdate and enddate:
            ds = ds.sel(time=slice(startdate, enddate))
            # print('/n/n/n/n', 'cutdate',)

        if variableType:
            ds = ds[[variableType]]
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

    _, indexes = kd_tree.query(targets, k=3*N)
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
            the dataArray that to be interpolated. the interpolation would 
            implemented for every time step (and every varibale) by the same method (and weights)
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

    # test = np.array(getValue(value_da, locs_all[0], weights_all[0]))
    # print(test)
    # assert False

    print('all iters:', len(locs_all))
    values = np.array([getValue(value_da, locs, weights)
                       for locs, weights in tqdm(zip(locs_all, weights_all))])

    values = values.reshape((len(lon), len(lat), len(time), len(variable)))
    values = np.transpose(values, (1, 0, 2, 3))

    # new = np.empty((len(lat), len(lon), len(time)))
    # k = 0
    # for i in range(len(lon)):
    #     for j in range(len(lat)):
    #         new[j][i] = values[k]
    #         k += 1

    res = xr.DataArray(
        values, coords=[lat, lon, time, variable], dims=['lat', 'lon', 'time', 'variable'])
    return res


if __name__ == '__main__':

    value_ds = xr.open_dataset(
        '/home/waterlab/Wang/bachelor_thesis/downloads/GCMs/MRI/MRI.nc', engine='h5netcdf')

    grid_dir = '/home/waterlab/Wang/bachelor_thesis/downloads/JRA/rawdownloads/uas/anl_mdl.033_ugrd.reg_tl319.196101_196112.wang528867'

    value_ds = preprocessed(value_ds, 'MRI', STARTDATE, ENDDATE)
    value_da = value_ds.to_array()
    # print(value_da['variable'])

    grid_ds = xr.open_dataset(grid_dir, engine='cfgrib')
    grid_ds = preprocessed(grid_ds, 'JRA')
    grid_ds = droppingVariablesOfds(grid_ds)

    res = IDWInterpolation(grid_ds, value_da)
    res.isel(time=0, variable=0).plot()

    res.to_netcdf(
        '/home/waterlab/Wang/bachelor_thesis/interpolated_gcms_mon/MRI/MRI.nc', engine='h5netcdf')
    # value_da = droppingVariablesOfds(value_da)
    # value_da.isel(time=0).plot()
