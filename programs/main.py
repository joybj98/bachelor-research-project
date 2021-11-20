#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:23:36 2021

@author: wangshiyuan
"""
import xarray as xr
import os
import glob
import pandas as pd
import numpy as np
import datetime
from time import time
from math import sqrt

import matplotlib.pyplot as plt
# from tqdm import tqdm


'''===================basic settings=================='''

N = 4
# means the value of each pp would be determined by N nearest pp's value.
EXPONENT = 2
LATBOUND = [0, 50]
LONBOUND = [95, 155]


STARTDATE = '1970-01-01'
ENDDATE = '2015-12-31'

Evaluating_indices = ['mean', 'bias', 'R', 'RMSE', 'stdev']


reanalysis_dir = '../downloads/1981-2005_u/u_new/*.glob1'
rp_dir = '../downloads/MIROC6/*.nc'

write_dir = '../output/MRI'


'''===================for testing=================='''

'''----0-----'''
'''values in each Map'''
# the default setting shoud be [None, None]
bnd = [None, None]
# bnd = [10, 10]
# these limit the program only to predict few points for each time step

'''-----1-----'''
# the default setting shoud be theirselves.
# the default should be like: STARTDATE = STARTDATE
STARTDATE = '1981-01-01'
ENDDATE = '1981-01-02'

'''-----2-----'''
timeMeasurement = True
# determine whether to measure the run time or a specific function or not

'''-----3-----'''
outputDataType = 'dataset'
# can only set as 'dataarray' or 'dataset'


def measureRunTime(func):
    def timer(*args, **kwargs):

        start = time()
        ret = func(*args, **kwargs)
        end = time()
        print(f'func \'{func.__name__ }\' run time {end-start:.3f} seconds')
        return ret
    return timer if timeMeasurement is True else func


class Point():
    def __init__(self, lat=None, lon=None, source=None):
        self.lat = float(lat)
        self.lon = float(lon)
        self.dist = float('inf')
        self.value = None
        self.loc = np.array([self.lat, self.lon])
        # self.idx = []
        self.source = source

    def getValue(self):
        if self.source and self.lat and self.lon:
            value = float(self.source.sel(
                lat=self.lat, lon=self.lon, method='nearest', tolerance=0.01).uas)
        else:
            raise ValueError(
                'need to know the lat, lon, and source to get the value!')
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


def droppingVariablesOfds(ds):
    lat_idxs = [i for i in range(len(ds.lat)) if LATBOUND[0] <= float(
        ds.lat[i]) <= LATBOUND[1]]
    lon_idxs = [i for i in range(len(ds.lon)) if LONBOUND[0] <= float(
        ds.lon[i]) <= LONBOUND[1]]

    return ds.isel(lat=lat_idxs, lon=lon_idxs)


def findNearestPoints(gcm, pp_loc, N=N):
    # given a rp map at one time step, and a exact location to predict, reuturn an array concluds 4 nearest Points
    # pp_loc is also a one-dimensional array and has two numbers (lat, lon)
    points, res = [], []
    # res is the return list of this function
    # conluding 4 nearest Point
    lat, lon = pp_loc[0], pp_loc[1]
    # print(pp_loc)
    gcm.ds = gcm.ds.sel(lat=slice(lat-2, lat+2), lon=slice(lon-2, lon+2))

    # this is because we don't neet to check evey point in the reanalysis map to find 4 nearest points
    # it seems stupid because we can merely use the its idx to select a specific point and get its value
    # print(gcm.ds)
    for lat in gcm.ds.lat:
        for lon in gcm.ds.lon:
            point = Point(lat, lon, source=gcm.ds)
            # print(point.lat, point.lon)

    # for i in lat_idxs:
    #     for j in lon_idxs:
    #         point = Point(float(gcm.ds.lat[i]), float(gcm.ds.lon[j]))
            point.dist = np.linalg.norm(point.loc - pp_loc)
    #         point.idx = [i, j]
    #         # print(point.idx)
            points.append(point)

            # print(f' for {pp_loc} appened the rp at {point.loc}')

    if len(points) == N:
        for point in points:
            point.value = point.getValue()
            # print(f'value is {point.value}')
        res = points
    else:

        points = sorted(points, key=lambda point: point.dist)

        for i in range(N):
            points[i].value = points[i].getValue()
            res.append(points[i])
            # print(
            #     f'for pp at {pp_loc} found No.{i} nearest rp, at {res[i].loc} with distance {res[i].dist}')
    return res


@ measureRunTime
def interpolatedMap(reanalysis, gcm, time, bnd: list = bnd, exponent=EXPONENT):
    # given (a reanalysis map) and (a rp map at one time step), pridict the value of each point on the reanalysis

    reanalysis.ds = reanalysis.ds.isel(time=0)
    xbnd, ybnd = bnd[0], bnd[1]

    data = np.zeros((len(reanalysis.ds.lat), len(reanalysis.ds.lon)))

    # i = 0
    # for lat in reanalysis.ds.lat:
    for i, lat in enumerate(reanalysis.ds.lat):
        # j = 0
        for j, lon in enumerate(reanalysis.ds.lon):

            pp_loc = np.array((float(lat), float(lon)))

            nearestNPoints = findNearestPoints(gcm.ds, pp_loc)

            if nearestNPoints[0].dist == 0.0:
                # to avoid dividing by zero
                data[i][j] = nearestNPoints[0].value
            else:
                values = [point.value for point in nearestNPoints]

                dists = [point.dist for point in nearestNPoints]
                # print(values, dists)
                weights = 1 / np.power(dists, exponent)
                data[i][j] = newVal(weights, values)

            # print(
            #     f'the value at {pp_loc} has been predicted to {data[i][j]}, in [{i}][{j}]. And its 4 nearest is {values}')
            # print(f"[{i}][{j}] is over with value {data[i][j]}")
            # print(f"[{i}][{j}] is over")

            if ybnd and j >= ybnd:
                break
            # j += 1

        if xbnd and i >= xbnd:
            break
        # i += 1
        print(f"[{i}] is over")
    data = data[:, :, np.newaxis]
    if outputDataType == 'dataarray':

        new = xr.DataArray(data, coords=[
            reanalysis.ds.lat, reanalysis.ds.lon, time], dims=['lat', 'lon', 'time'])
    elif outputDataType == 'dataset':
        new = xr.Dataset(
            {
                "uas": (["lat", "lon", "time"], data),
            },
            coords={
                "lat": (["lat"], reanalysis.ds.lat.to_index()),
                "lon": (["lon"], reanalysis.ds.lon.to_index()),
                'time': (['time'], time)
            },
        )
    else:
        new = None
    print(f'interpolation for {time.strftime("%Y-%m-%d")[0]} is over')
    return new


@ measureRunTime
def interpolatedDataset(reanalysis, gcm, startTime=STARTDATE, endTime=ENDDATE):

    # reanalysis.ds = Args.reanalysis.ds
    # gcm.ds= Args.gcm.ds
    # startTime , endTime= Args.startTime, Args.endTime

    timeIdx = 0
    gcm.ds = gcm.ds.sel(time=slice(startTime, endTime))
    times = gcm.ds['time'].to_index()
    concacted = interpolatedMap(reanalysis.ds, gcm.ds.sel(
        time=times[0]), times[:1])
    for timeIdx in range(1, len(times)):

        new = interpolatedMap(reanalysis.ds, gcm.ds.sel(
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


class DsBase():
    def __init__(self, ds, source=None):
        self.ds = ds
        self.source = source


class JRA_55(DsBase):
    def __init__(self, ds):
        super().__init__(ds)
        self.source = "JRA-55"

    def preprocess(self):
        self.ds = self.ds.rename(
            {'u': 'uas', 'latitude': 'lat', 'longitude': 'lon'})
        self.ds = droppingVariablesOfds(self.ds)
        self.ds = self.ds.transpose('lat', 'lon', 'time')
        self.ds = self.ds.sel(time=datetime.time(12))
        self.ds = self.ds.sortby('lat')


class MRI(DsBase):
    def __init__(self, ds):
        super().__init__(ds)
        self.source = 'MRI'

    def preprocess(self):
        self.ds = self.ds.drop_vars(['time_bnds', 'lon_bnds', 'lat_bnds'])
        self.ds = self.ds.transpose('lat', 'lon', 'time')


class MIROC6(DsBase):
    def __init__(self, ds):
        super().__init__(ds)
        self.source = 'MIROC6'

    def preprocess(self):
        self.ds = self.ds.drop_vars(['lon_bnds', 'lat_bnds'])
        self.ds = self.ds.transpose('lat', 'lon', 'time')
        self.ds = self.ds.sel(time=datetime.time(12))


def main():
    ds = xr.open_mfdataset(
        reanalysis_dir, concat_dim='time', combine='nested',  coords='minimal', engine='cfgrib')

    reanalysis = JRA_55(ds)
    reanalysis.preprocess()
    print(reanalysis.ds)

    ds = xr.open_mfdataset(rp_dir, combine='nested',
                           coords='minimal', engine='netcdf4')
    gcm = MIROC6(ds)
    gcm.preprocess()
    print(gcm.ds)


if __name__ == '__main__':
    for file in glob.glob('../downloads/1981-2005_u/u_new/*.idx'):
        os.remove(file)

    main()
