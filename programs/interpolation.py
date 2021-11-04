#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:23:36 2021

@author: wangshiyuan
"""
import xarray as xr
# import os
import pandas as pd
import numpy as np
import datetime
from time import time
from math import sqrt
# import matplotlib.pyplot as plt


'''===================basic settings=================='''

N = 4
# means the value of each pp would be determined by 4 nearest pp's value.
EXPONENT = 2
LATBOUND = [0, 50]
LONBOUND = [95, 155]


STARTDATE = '1970-01-01'
ENDDATE = '2015-12-31'

descriptionDict = {
    'mean': True,
    'bias': True,
    'R': True,
    'RMSE': True,
    'stdev': True, }


grid_dir = 'downloads/1981-2005_u/u_new/1981-01-01.glob1'
rp_dir = 'downloads/MRI/uas_day_MRI-ESM2-0_esm-hist_r1i2p1f1_gn_19500101-19991231.nc'

write_dir = 'output/MRI'


'''===================for testing=================='''

'''----0-----'''
'''values in each Map'''
# the default setting shoud be [None, None]
bnd = [2, 2]
# these limit the program only to predict few points for each time step

'''-----1-----'''
# the default setting shoud be theirselves.
# the default should be like: STARTDATE = STARTDATE
STARTDATE = '1981-01-01'
ENDDATE = '1981-01-05'

'''-----2-----'''
timeMeasurement = False
# determine whether to measure the run time or a specific function or not

'''-----3-----'''
outputDataType = 'dataset'
# can only set as 'dataarray' or 'dataset'


class Point():
    def __init__(self, lat=None, lon=None, source=None):
        self.lat = float(lat)
        self.lon = float(lon)
        self.dist = 0
        self.value = 0
        self.loc = np.array([lat, lon])
        self.idx = []
        self.source = source

    # def getValue(self):
    #     if self.idx:
    #         self.value = source.


class ModelPair():
    def __init__(self, referModel_dataList, selfModel_dataList):
        self.refer = referModel_dataList
        self.model = selfModel_dataList

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


def measureRunTime(func):
    def timer(*args, **kwargs):
        start = time()
        ret = func(*args, **kwargs)
        end = time()
        elapsed = end - start
        print(f'func \'{func.__name__ }\' run time {elapsed} seconds')
        return ret
    return timer if timeMeasurement is True else func


def newVal(weights, values):
    return (weights @ values)/sum(weights)


def droppingVariablesOfds(ds):
    lat_idxs = [i for i in range(len(ds.lat)) if LATBOUND[0] <= float(
        ds.lat[i]) <= LATBOUND[1]]
    lon_idxs = [i for i in range(len(ds.lon)) if LONBOUND[0] <= float(
        ds.lon[i]) <= LONBOUND[1]]

    return ds.isel(lat=lat_idxs, lon=lon_idxs)


def findNearestPoints(rp_ds, pp_loc, N=N):
    # given a rp map at one time step, and a exact location to predict, reuturn an array concluds 4 nearest Points
    # pp_loc is also a one-dimensional array and has two numbers (lat, lon)
    points, res = [], []
    # res is the return list of this function
    # conluding 4 nearest Point
    lat_idxs = [idx for idx in range(len(rp_ds.lat)) if abs(
        pp_loc[0] - float(rp_ds.lat[idx])) < 2]
    lon_idxs = [idx for idx in range(len(rp_ds.lat)) if abs(
        pp_loc[1] - float(rp_ds.lon[idx])) < 2]

    # this is because we don't neet to check evey point in the grid to find 4 nearest points
    # it seems stupid because we can merely use the its idx to select a specific point and get its value

    for i in lat_idxs:
        for j in lon_idxs:
            point = Point(float(rp_ds.lat[i]), float(rp_ds.lon[j]))
            point.dist = np.linalg.norm(point.loc - pp_loc)
            point.idx = [i, j]
            # print(point.idx)
            points.append(point)

            # print(f' for {pp_loc} appened the rp at {point.loc}')

    if len(points) == N:
        for point in points:
            point.value = float(rp_ds.isel(
                lat=point.idx[0], lon=point.idx[1]).uas)
        res = points
    else:

        points = sorted(points, key=lambda point: point.dist)
        # dists = [point.dist for point in points]

        # temp = zip(dists, points)
        # dists, points = zip(*sorted(temp))

        for i in range(N):
            points[i].value = float(rp_ds.isel(
                lat=points[i].idx[0], lon=points[i].idx[1]).uas)
            res.append(points[i])
            # print(
            #     f'for pp at {pp_loc} found No.{i} nearest rp, at {res[i].loc} with distance {res[i].dist}')
    return res


def interpolatedMap(grid_ds, rp_ds, time, bnd: list = bnd, exponent=EXPONENT):
    # given (a grid map) and (a rp map at one time step), pridict the value of each point on the grid

    grid_ds = grid_ds.isel(time=0)
    xbnd, ybnd = bnd[0], bnd[1]

    data = np.zeros((len(grid_ds.lat), len(grid_ds.lon)))
    i = 0

    for lat in grid_ds.lat:
        j = 0
        for lon in grid_ds.lon:

            pp_loc = np.array((float(lat), float(lon)))

            nearestNPoints = findNearestPoints(rp_ds, pp_loc)

            if nearestNPoints[0].dist == 0.0:
                # to avoid dividing by zero
                data[i][j] = float(rp_ds.isel(
                    lat=nearestNPoints[0].idx[0], lon=nearestNPoints[0].idx[1]).uas)
            else:
                values = [point.value for point in nearestNPoints]

                dists = [point.dist for point in nearestNPoints]
                weights = 1 / np.power(dists, exponent)
                data[i][j] = newVal(weights, values)

            # print(
            #     f'the value at {pp_loc} has been predicted to {data[i][j]}, in [{i}][{j}]. And its 4 nearest is {values}')
            # print(f"[{i}][{j}] is over with value {data[i][j]}")
            print(f"[{i}][{j}] is over")

            j += 1

            if ybnd:
                if j >= ybnd:
                    break
        i += 1

        if xbnd:
            if i >= xbnd:
                break
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
                "lat": (["lat"], grid_ds.lat),
                "lon": (["lon"], grid_ds.lon),
                'time': (['time'], time)
            },
        )
    else:
        new = None
    return new


@ measureRunTime
def interpolatedDataset(grid_ds, rp_ds, startTime=STARTDATE, endTime=ENDDATE):

    timeIdx = 0
    rp_ds = rp_ds.sel(time=slice(startTime, endTime))
    times = rp_ds['time'].to_index()
    concacted = interpolatedMap(grid_ds, rp_ds.sel(
        time=times[0]), times[:1])
    for timeIdx in range(len(times)-1):

        new = interpolatedMap(grid_ds, rp_ds.sel(
            time=times[timeIdx+1]), times[timeIdx+1:timeIdx+2])
        concacted = xr.concat([concacted, new], dim='time')

    return concacted


def comparision(refer_ds, interpolated_ds_list: list, startTime=STARTDATE, endTime=ENDDATE):

    refer_datalist = np.array(refer_ds.sel(
        time=slice(startTime, endTime)).uas).ravel()

    modelData_list = [np.array(ds.sel(time=slice(startTime, endTime)).uas).ravel()
                      for ds in interpolated_ds_list]

    # the list includes lists of data of different models
    # which means a list of Models' data's list

    # while a model_datalist is an unit in  modelData_list

    output_index = [k for k in descriptionDict.keys()]
    output_data = np.array([])

    for model_datalist in modelData_list:

        pair = ModelPair(refer_datalist, model_datalist)

        description_list = np.array(
            [v for v in pair.indices().values()]).reshape(-1, 1)

        output_data = np.append(output_data, description_list,
                                axis=1) if output_data.size > 0 else np.array(description_list)

    df = pd.DataFrame(
        output_data, index=output_index, columns=list(range(len(modelData_list))))

    return df


grid_ds = xr.open_dataset(grid_dir, engine='cfgrib')
grid_ds = grid_ds.rename({'u': 'uas', 'latitude': 'lat', 'longitude': 'lon'})
grid_ds = droppingVariablesOfds(grid_ds)
grid_ds = grid_ds.transpose('lat', 'lon', 'time')
grid_ds = grid_ds.sel(time=datetime.time(12))
# print(grid_ds)


rp_ds = xr.open_dataset(rp_dir, engine='netcdf4')
rp_ds = rp_ds.drop_vars(['time_bnds', 'lon_bnds', 'lat_bnds'])
rp_ds = rp_ds.transpose('lat', 'lon', 'time')
# print(grid_ds.to_array())
print(grid_ds)

# p1 = Point(0, 5)
# p2 = Point(0, 3)
# p3 = Point(0, 4)
# points = [p1, p2, p3]
# print(points)
# points = sorted(points, key=lambda x: x.lon)
# print(points)


# print(comparision(grid_ds, [interpolatedDataset(grid_ds, rp_ds)]))
# print(comparision(grid_ds, [interpolatedDataset(grid_ds, rp_ds)]))
