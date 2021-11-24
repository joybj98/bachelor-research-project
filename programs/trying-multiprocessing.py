#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:23:36 2021

@author: wangshiyuan
"""
import xarray as xr

import pandas as pd
import numpy as np
import datetime
from time import time
from math import sqrt
from glob import glob
import os
import matplotlib.pyplot as plt

from multiprocessing import Pool

'''===================basic settings=================='''

N = 4
# means the value of each pp would be determined by 4 nearest pp's value.
EXPONENT = 2
LATBOUND = [0, 50]
LONBOUND = [95, 155]


STARTDATE = '1970-01-01'
ENDDATE = '2015-12-31'

Evaluating_indices = ['mean', 'bias', 'R', 'RMSE', 'stdev']


grid_dir = '../downloads/1981-2005_u/u_new/1981-01-01.glob1'
rp_dir = '../downloads/GCMs/*/*.nc'

write_dir = '../output/preprocessed_gcms'


'''===================for testing=================='''

'''----0-----'''
'''values in each Map'''
# the default setting shoud be [None, None]
bnd = [None, None]
bnd = [10, 10]
# these limit the program only to predict few points for each time step

'''-----1-----'''
# the default setting shoud be theirselves.
# the default should be like: STARTDATE = STARTDATE
STARTDATE = '1981-01-01'
ENDDATE = '1981-01-01'

'''-----2-----'''
timeMeasurement = True
# determine whether to measure the run time or a specific function or not

'''-----3-----'''
outputDataType = 'dataset'
# can only set as 'dataarray' or 'dataset'


def main(args):

    # grid_ds = xr.open_dataset(grid_dir, engine='cfgrib')
    # grid_ds = xr.open_mfdataset(
    #     grid_dir, concat_dim="time", combine="nested", coords='minimal', engine='cfgrib')
    # grid_ds = preprocessed(grid_ds, modelType='JRA')

    # args = ((filepath, os.path.basename(os.path.dirname(filepath)))
    #         for filepath in glob(rp_dir))
    # print('here1')
    grid_ds, file, modelType = args[0], args[1], args[2]
    # print('here2')
    rp_ds = xr.open_dataset(file, engine='h5netcdf')
    # print('here3')

    rp_ds = preprocessed(rp_ds, modelType=modelType)

    out = interpolatedDataset(grid_ds, rp_ds)

    print(out)
    return out

    # print(comparision(grid_ds, [out, out]))


def measureRunTime(func):
    def timer(*args, **kwargs):
        start = time()
        ret = func(*args, **kwargs)
        end = time()
        elapsed = end - start
        print(f'func \'{func.__name__ }\' run time {elapsed} seconds')
        return ret
    return timer if timeMeasurement is True else func


class Point():
    def __init__(self, lat=None, lon=None, source=None):
        self.lat = float(lat)
        self.lon = float(lon)
        self.dist = float('inf')
        self.value = None
        self.loc = np.array([lat, lon])
        # self.idx = []
        self.source = source

    def getValue(self):
        if self.source and self.lat and self.lon:
            value = float(self.source.sel(
                lat=self.lat, lon=self.lon, method='nearest', tolerance=0.01).uas)
        else:
            raise ValueError(
                'need to know the lat, lon, and source to get the value. Now the three are {if self.source}, {if self.lat}, {if self.lon}!')
        return value


class ModelPair():
    def __init__(self, referModelData_1_dim, selfModelData_1_dim):
        self.refer = referModelData_1_dim
        self.model = selfModelData_1_dim

    def error(self):
        return np.array(self.model - self.refer)

    def bias(self):
        return self.error().mean()

    def R(self):
        return np.corrcoef(self.refer, self.model)[0][1]
    # using R[0][1] is because R is a 2 by 2 matrix. which includes Rxx,Rxy,Ryx,Ryy. And we only use Rxy (equals to Ryx)

    def RMSE(self):
        return sqrt(np.square(self.error()).sum()/len(self.refer))

    def stdev(self):
        return np.std(self.error())

    def mean(self):
        return np.array(self.model).mean()

    def indices(self, **kwargs):
        dictionary = {
            'mean': self.mean(),
            'bias': self.bias(),
            'R': self.R(),
            "RMSE": self.RMSE(),
            'stdev': self.stdev()
        }

        return dictionary


def newVal(weights, values):
    return (weights @ values)/sum(weights)


def droppingVariablesOfds(ds, distToBound=0):
    lat_idxs = [i for i in range(len(ds.lat)) if LATBOUND[0]-distToBound <= float(
        ds.lat[i]) <= LATBOUND[1]+distToBound]
    lon_idxs = [i for i in range(len(ds.lon)) if LONBOUND[0]-distToBound <= float(
        ds.lon[i]) <= LONBOUND[1]+distToBound]

    return ds.isel(lat=lat_idxs, lon=lon_idxs)


def findNearestPoints(rp_ds, pp_loc, N=N):
    # given a rp map at one time step, and a exact location to predict, reuturn an array concluds 4 nearest Points
    # pp_loc is also a one-dimensional array and has two numbers (lat, lon)
    points, res = [], []
    lat, lon = pp_loc[0], pp_loc[1]
    # print(pp_loc)
    rp_ds = rp_ds.sel(lat=slice(lat-2, lat+2), lon=slice(lon-2, lon+2))

    # this is because we don't neet to check evey point in the grid to find 4 nearest points

    for lat in rp_ds.lat:
        for lon in rp_ds.lon:
            point = Point(lat, lon, source=rp_ds)

    # for i in lat_idxs:
    #     for j in lon_idxs:
    #         point = Point(float(rp_ds.lat[i]), float(rp_ds.lon[j]))
            point.dist = np.linalg.norm(point.loc - pp_loc)
    #         point.idx = [i, j]
    #         # print(point.idx)
            points.append(point)

            # print(f' for {pp_loc} appened the rp at {point.loc}')

    if len(points) == N:
        for point in points:
            point.value = point.getValue()
        res = points
    else:

        points = sorted(points, key=lambda point: point.dist)

        for i in range(N):
            # print(points[i].lat, points[i].lon, points[i].source)
            points[i].value = points[i].getValue()
            res.append(points[i])
            # print(
            #     f'for pp at {pp_loc} found No.{i} nearest rp, at {res[i].loc} with distance {res[i].dist}')
    return res


@ measureRunTime
def interpolatedMap(grid_ds, rp_ds, time, bnd: list = bnd, exponent=EXPONENT):
    # given (a grid map at one time step) and (a rp map at one time step), pridict the value of each point on the grid

    # grid_ds = grid_ds.isel(time=0)
    xbnd, ybnd = bnd[0], bnd[1]

    data = np.zeros((len(grid_ds.lat), len(grid_ds.lon)))

    for i, lat in enumerate(grid_ds.lat):

        for j, lon in enumerate(grid_ds.lon):

            pp_loc = np.array((float(lat), float(lon)))

            nearestNPoints = findNearestPoints(rp_ds, pp_loc)

            if nearestNPoints[0].dist == 0.0:
                # to avoid dividing by zero
                data[i][j] = nearestNPoints[0].value
            else:
                values = [point.value for point in nearestNPoints]
                # print(values)
                dists = [point.dist for point in nearestNPoints]
                weights = 1 / np.power(dists, exponent)
                data[i][j] = newVal(weights, values)

            # print(
            #     f'the value at {pp_loc} has been predicted to {data[i][j]}, in [{i}][{j}]. And its 4 nearest is {values}')
            # print(f"[{i}][{j}] is over with value {data[i][j]}")
            # print(f"[{i}][{j}] is over")

            if ybnd:
                if j >= ybnd:
                    break

        if xbnd:
            if i >= xbnd:
                break
        print(f"[{i}] is over")
    data = data[:, :, np.newaxis]
    if outputDataType == 'dataarray':

        new = xr.DataArray(data, coords=[
            grid_ds.lat, grid_ds.lon, time], dims=['lat', 'lon', 'time'])
    elif outputDataType == 'dataset':
        new = xr.Dataset(
            {
                "uas": (["lat", "lon", "time"], data),
            },
            coords={
                "lat": (["lat"], grid_ds.lat.to_index()),
                "lon": (["lon"], grid_ds.lon.to_index()),
                'time': (['time'], time)
            },
        )
    else:
        new = None
        raise ValueError(
            'need to set the outputDataType to either \'dataarray\' or \'dataset\'')

    print(f'interpolation for {time.strftime("%Y-%m-%d")[0]} is over')
    return new


@ measureRunTime
def interpolatedDataset(grid_ds, rp_ds, startTime=STARTDATE, endTime=ENDDATE):

    timeIdx = 0
    rp_ds = rp_ds.sel(time=slice(startTime, endTime))
    times = rp_ds['time'].to_index()
    concacted = interpolatedMap(grid_ds, rp_ds.sel(
        time=times[0]), times[:1])
    for timeIdx in range(1, len(times)):

        new = interpolatedMap(grid_ds, rp_ds.sel(
            time=times[timeIdx]), times[timeIdx:timeIdx+1])
        concacted = xr.concat([concacted, new], dim='time')

    return concacted


def comparision(refer_ds, interpolated_ds_list: list, startTime=STARTDATE, endTime=ENDDATE):

    referData_1_dim = np.array(refer_ds.sel(
        time=slice(startTime, endTime)).uas).ravel()

    modelData_list = [np.array(ds.sel(time=slice(startTime, endTime)).uas).ravel()
                      for ds in interpolated_ds_list]

    # the modelData_list includes lists of data of different models
    # which means a list of Models' data's list

    # while a modelData_1_dim is an unit in modelData_list

    output_index = [idx for idx in Evaluating_indices]
    output_data = np.array([])
    # print(refer_ds)
    # print(interpolated_ds_list[0])
    for modelData_1_dim in modelData_list:

        pair = ModelPair(referData_1_dim, modelData_1_dim)

        indices_list = np.array(
            [v for v in pair.indices().values()]).reshape(-1, 1)

        output_data = np.append(output_data, indices_list,
                                axis=1) if output_data.size > 0 else np.array(indices_list)

    df = pd.DataFrame(
        output_data, index=output_index, columns=list(range(len(modelData_list))))

    return df


def preprocessed(ds, modelType=''):
    def JRA_preprocess(ds):
        ds = ds.rename(
            {'u': 'uas', 'latitude': 'lat', 'longitude': 'lon'})
        ds = droppingVariablesOfds(ds)
        ds = ds.transpose('lat', 'lon', 'time')
        ds = ds.sel(time=datetime.time(12))
        ds = ds.sortby('lat')
        return ds

    def MRI_preprocess(ds):
        ds = ds.drop_vars(['time_bnds', 'lon_bnds', 'lat_bnds'])
        # ds = droppingVariablesOfds(ds,  mode='GCM')
        # ds = droppingVariablesOfds(ds, 5)
        ds = ds.transpose('lat', 'lon', 'time')

        return ds

    def MIROC6_preprpocess(ds):
        return MRI_preprocess(ds)

    def CNRM_ESM_preprocess(ds):
        ds = ds.drop_vars(['time_bounds'])
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

    def main_preprocess(ds, modelType=modelType):
        ds = model_list.get(modelType, unknown_preprocess)(ds)
        return ds
    print(f'{modelType} has been preprocessed')

    return main_preprocess(ds)


if __name__ == '__main__':
    for file in glob('../downloads/1981-2005_u/u_new/*.idx'):
        os.remove(file)

    grid_ds = xr.open_dataset(grid_dir, engine='cfgrib')
    grid_ds = preprocessed(grid_ds, modelType='JRA')
    grid_ds = grid_ds.isel(time=0)

    arg_list = []
    for filepath in glob(rp_dir):
        args = [grid_ds, filepath, os.path.basename(os.path.dirname(filepath))]
        arg_list.append(args)

    # print(args)
    t1 = time()
    pool = Pool()

    res = pool.map(main, arg_list)
    t2 = time()
    print(res)
    print(f'run time is {t2-t1:.04f} sec')
