#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:23:36 2021

@author: wangshiyuan
"""
import xarray as xr

import pandas as pd
import numpy as np
import datetime as dt
from time import time
from glob import glob
import os
# import matplotlib.pyplot as plt

from multiprocessing import Pool

'''===================basic settings=================='''

N = 4
# means the value of each pp would be determined by 4 nearest pp's value.
EXPONENT = 2
LATBOUND = [-10, 60]
LONBOUND = [95, 180]


STARTDATE = '1970-01-01'
ENDDATE = '2015-12-31'

grid_dir = '/home/waterlab/Wang/bachelor_thesis/downloads/JRA/rawdownloads/uas/anl_mdl.033_ugrd.reg_tl319.196101_196112.wang528867'

rp_dir = '../downloads/GCMs/MRI/*.nc'

write_dir_base = '../interpolated_gcms_mon'


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
STARTDATE = '1960-01-01'
ENDDATE = '2018-01-01'

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
        print(
            f'func \'{func.__name__ }\' run time {end - start:04f} seconds, {(end-start)/60 :03f} min')
        return ret
    return timer if timeMeasurement is True else func


class Point():
    def __init__(self, lat=None, lon=None, source=None, variableType=None):
        self.lat = float(lat)
        self.lon = float(lon)
        self.dist = float('inf')
        self.value = None
        self.loc = np.array([lat, lon])
        # self.idx = []
        self.source = source
        self.type = variableType

    def getValue(self):
        if self.source and self.lat and self.lon and self.type:
            value = float(self.source.sel(
                lat=self.lat, lon=self.lon, method='nearest', tolerance=0.01)[self.type])
        else:
            raise ValueError(
                'need to know the lat, lon, source and variableType to get the value. Now the three are {bool(self.source)}, {bool(self.lat)}, {bool(self.lon)}, {bool(self.type)!')
        return value


def newVal(weights, values):
    return (weights @ values)/sum(weights)


def droppingVariablesOfds(ds, distToBound=0):
    lat_idxs = [i for i in range(len(ds.lat)) if LATBOUND[0]-distToBound <= float(
        ds.lat[i]) <= LATBOUND[1]+distToBound]
    lon_idxs = [i for i in range(len(ds.lon)) if LONBOUND[0]-distToBound <= float(
        ds.lon[i]) <= LONBOUND[1]+distToBound]

    return ds.isel(lat=lat_idxs, lon=lon_idxs)


def findNearestPoints(rp_ds, pp_loc, N=N, variableType="uas"):
    # given a rp map at one time step, and a exact location to predict, reuturn an array concluds 4 nearest Points
    # pp_loc is also a one-dim
    # means the value of each pp would be determined by 4 nearest ensional array and has two numbers (lat, lon)
    points, res = [], []
    lat, lon = pp_loc[0], pp_loc[1]
    # print(pp_loc)

    rp_ds = rp_ds.sel(lat=slice(lat-2, lat+2), lon=slice(lon-2, lon+2))

    # this is because we don't neet to check evey point in the grid to find 4 nearest points

    for lat in rp_ds.lat:
        for lon in rp_ds.lon:
            point = Point(lat, lon, source=rp_ds, variableType=variableType)

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


# @ measureRunTime
def interpolatedMap(grid_ds, rp_ds, time, bnd: list = bnd, exponent=EXPONENT, modelType=None, variableType="uas"):
    # given (a grid map at one time step) and (a rp map at one time step), pridict the value of each point on the grid

    # grid_ds = grid_ds.isel(time=0)
    xbnd, ybnd = bnd[0], bnd[1]
    # pool = Pool()
    # pool.map(main, arg_list)
    # t2 = tim

    data = np.zeros((len(grid_ds.lat), len(grid_ds.lon)))

    for i, lat in enumerate(grid_ds.lat):

        for j, lon in enumerate(grid_ds.lon):

            pp_loc = np.array((float(lat), float(lon)))

            nearestNPoints = findNearestPoints(
                rp_ds, pp_loc, variableType=variableType)

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

            if ybnd and j >= ybnd:
                break

        if xbnd and i >= xbnd:
            break
        if i % 10 == 0 or i == 1:
            print(f"gcm {modelType} [{i} of {len(data)}] is over")

    data = data[:, :, np.newaxis]
    if outputDataType == 'dataarray':

        new = xr.DataArray(data, coords=[
            grid_ds.lat, grid_ds.lon, time], dims=['lat', 'lon', 'time'])
    elif outputDataType == 'dataset':
        new = xr.Dataset(
            {
                variableType: (["lat", "lon", "time"], data),
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
def interpolatedDataset(grid_ds, rp_ds, startTime=STARTDATE, endTime=ENDDATE, modelType=None, variableType="uas"):

    timeIdx = 0
    rp_ds = rp_ds.sel(time=slice(startTime, endTime))

    # print(rp_ds)
    times = rp_ds['time'].to_index()

    concacted = interpolatedMap(grid_ds, rp_ds.sel(
        time=times[0]), times[:1], modelType=modelType, variableType=variableType)
    for timeIdx in range(1, len(times)):

        new = interpolatedMap(grid_ds, rp_ds.sel(
            time=times[timeIdx]), times[timeIdx:timeIdx+1], modelType=modelType, variableType=variableType)
        concacted = xr.concat([concacted, new], dim='time')
        print(
            f"interpolation and concaction for {startTime} to {times[timeIdx].strftime('%Y-%m-%d')} is over")
    return concacted


def preprocessed(ds, modelType='', startdate=None, enddate=None, variableType=None):
    def JRA_preprocess(ds):
        if "u" in ds:
            ds = ds.rename({"u": "uas"})
        if "v" in ds:
            ds = ds.rename({"v": "vas"})
        ds = ds.rename(
            {'latitude': 'lat', 'longitude': 'lon'})
        ds = droppingVariablesOfds(ds)
        ds = ds.transpose('lat', 'lon', 'time')
        ds = ds.sortby('lat')
        return ds

    def MRI_preprocess(ds):

        # ds = ds.drop_vars(['time_bnds', 'lon_bnds', 'lat_bnds'])
        ds["time"] = ds.time.to_index().map(lambda t: t.replace(day=1, hour=0))

        # ds = droppingVariablesOfds(ds,  mode='GCM')
        # ds = droppingVariablesOfds(ds, 5)

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

        if variableType:
            ds = ds[[variableType]]

        ds = model_list.get(modelType, unknown_preprocess)(ds)

        return ds
    print(f'{modelType} has been preprocessed')

    return main_preprocess(ds)


def main(args):
    # interpolate one month of data
    grid_ds, filepath, modelType, startdate, enddate, variableType = args[
        0], args[1], args[2], args[3], args[4], args[5]

    rp_ds = xr.open_dataset(filepath, engine='h5netcdf')
    # print(rp_ds, startdate, enddate)

    rp_ds = preprocessed(rp_ds, modelType=modelType,
                         startdate=startdate, enddate=enddate, variableType=variableType)
    # print(rp_ds)
    print(args[2:])
    out = interpolatedDataset(
        grid_ds, rp_ds, modelType=modelType, startTime=startdate, endTime=enddate, variableType=variableType)

    write_dir = write_dir_base + f"/{modelType}/{variableType}"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    out.to_netcdf(
        write_dir + f"/{modelType}_{startdate.year}_{variableType}.nc", engine="h5netcdf")


if __name__ == '__main__':
    for file in glob('../downloads/JRA/rawdownloads/**/*.idx', recursive=True):
        os.remove(file)

    grid_ds = xr.open_dataset(grid_dir, engine='cfgrib')
    grid_ds = preprocessed(grid_ds, modelType='JRA')
    grid_ds = grid_ds.isel(time=0)

    arg_list = []

    for filepath in glob(rp_dir):
        ds = xr.open_dataset(filepath, engine="h5netcdf")

        modelType = os.path.basename(os.path.dirname(filepath))
        ds = preprocessed(ds, modelType=modelType,
                          startdate=STARTDATE, enddate=ENDDATE)
        if len(ds.time) == 0:
            continue

        timeIdx = ds.time.to_index()
        startdate_list = pd.date_range(timeIdx[0], timeIdx[-1], freq="YS")
        enddate_list = pd.date_range(timeIdx[0], timeIdx[-1], freq="Y")
        # print(startdate_list, enddate_list)
        for startdate, enddate in zip(startdate_list, enddate_list):
            for variableType in ds.data_vars:

                args = (grid_ds, filepath, modelType,
                        startdate, enddate, variableType)
                if not os.path.exists(write_dir_base + f"/{modelType}/{variableType}/{modelType}_{startdate.year}_{variableType}.nc"):

                    arg_list.append(args)

    # print(grid_ds)
    # for args in arg_list:
    #     print(args[2:], "\n\n")
    #     # print(args[3:5])
    print(len(arg_list))
    # print(f"the time would be about {len(arg_list)*35/40/60 :.2f} hours")

    t1 = time()

    for args in arg_list:
        main(args)

        if (time() - t1) > 12*60*60:
            break

    # Pool().map(main, arg_list)
    t2 = time()

    print(
        f'run time is {t2-t1:.04f} sec, {(t2-t1)/60/60:.2f} hours, for {len(arg_list)} loops. while the predicted time is {len(arg_list)*35/40/60 :.2f} hours')
