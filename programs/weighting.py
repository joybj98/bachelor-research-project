
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:34:55 2021

@author: wangshiyuan
"""

import xarray as xr
from math import sqrt
import pandas as pd
import numpy as np
import datetime
from spicy import linalg
import matplotlib.pyplot as plt

LATBOUND = [0, 50]
LONBOUND = [95, 155]
STARTDATE = '1981-01-01'
ENDDATE = '1981-12-31'

Evaluating_indices = ['mean', 'bias', 'R', 'RMSE', 'stdev']

JRA_dir = '../downloads/JRA/1981-01-01.glob1'
MRI_dir = '/home/waterlab/Wang/bachelor_thesis/interpolated_gcms/MRI/uas/MRI_1981-01-01 00:00:00_uas.nc'
MIROC6_dir = '/home/waterlab/Wang/bachelor_thesis/interpolated_gcms/MIROC6/uas/MIROC6_1981-01-01 00:00:00_uas.nc'
CNRM_dir = '/home/waterlab/Wang/bachelor_thesis/interpolated_gcms/CNRM-ESM2-1/uas/uas_day_CNRM-ESM2-1_historical_r1i1p1f2_gr_19500101-20141231.nc'


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

    def MSE(self):
        return np.square(self.error()).sum()/len(self.refer)

    def RMSE(self):
        return sqrt(self.MSE())

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


def droppingVariablesOfds(ds, distToBound=0):
    lat_idxs = [i for i in range(len(ds.lat)) if LATBOUND[0]-distToBound <= float(
        ds.lat[i]) <= LATBOUND[1]+distToBound]
    lon_idxs = [i for i in range(len(ds.lon)) if LONBOUND[0]-distToBound <= float(
        ds.lon[i]) <= LONBOUND[1]+distToBound]

    return ds.isel(lat=lat_idxs, lon=lon_idxs)


def preprocessed(ds, modelType='', startdate=None, enddate=None):
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
        ds = ds.drop_vars(['time_bou8nds'])

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

    def main_preprocess(ds, modelType=modelType, startdate=startdate, enddate=enddate):
        if startdate and enddate:
            ds = ds.sel(time=slice(startdate, enddate))

        ds = model_list.get(modelType, unknown_preprocess)(ds)

        return ds
    print(f'{modelType} has been preprocessed')

    return main_preprocess(ds)


def comparision(referData, modelData_list: list):

    # the modelData_list includes lists of data of different models
    # which means a list of Models' data's list

    output_index = [idx for idx in Evaluating_indices]
    output_data = np.array([])
    # print(refer_ds)
    # print(interpolated_ds_list[0])
    for modelData in modelData_list:

        pair = ModelPair(referData, modelData)

        indices_list = np.array(
            [v for v in pair.indices().values()]).reshape(-1, 1)

        output_data = np.append(output_data, indices_list,
                                axis=1) if output_data.size > 0 else np.array(indices_list)

    df = pd.DataFrame(
        output_data, index=output_index, columns=list(range(len(modelData_list))))

    return df


def normalized(data):
    return (data-data.min())/(data.max() - data.min())


def NewbatchGradientDesent(X, Y, weights, alpha, iters):

    def feasibleSpace(dim):
        I = np.identity(dim-1, dtype="int")
        ones = np.ones(dim-1).reshape(1, dim-1)

        return np.append(-I, ones, axis=0)

    def projectionMatrix(A):
        return A @ linalg.inv(A.T @ A) @ A.T

    dim = len(weights)
    weights[-1] = 1-sum(weights[:-1])
    costHistory = [0]*iters
    prediction = weights @ X

    for i in range(iters):

        decent = (alpha/len(Y)) * (X@(prediction-Y))
        projectedDecent = projectionMatrix(feasibleSpace(dim))@decent

        weights = weights-projectedDecent
        if weights.min() > 0:
            reached_bound = False
        else:
            if reached_bound:
                break
            else:
                weights = np.array(weights) + projectedDecent * \
                    (weights.min()/projectedDecent[np.argmin(weights)])
                reached_bound = True

        prediction = weights @ X
        costHistory[i] = ModelPair(Y, weights@X).MSE()

    # print(costHistory)
    # print(weights)
    return prediction, weights, costHistory


def newMiniBatchGradientDecent(X, Y, weights, iters=50, batchSize=1000):
    def learningSchedule(t):
        t0, t1 = 200, len(Y)
        return t0/(t+t1)

    def feasibleSpace(dim):
        I = np.identity(dim-1, dtype="int")
        ones = np.ones(dim-1).reshape(1, dim-1)

        return np.append(-I, ones, axis=0)

    def projectionMatrix(A):
        return A @ linalg.inv(A.T @ A) @ A.T

    dim = len(weights)
    weights[-1] = 1-sum(weights[:-1])
    t, c = 0, 0
    costHistory = [0]*iters
    prediction = weights @ X
    np.random.seed(42)
    for epoch in range(iters):
        shuffledIdx = np.random.permutation(len(Y))
        X_shuffled = np.array(X[:][shuffledIdx])
        Y_shuffled = Y[shuffledIdx]
        # for data in X:
        #     shuffledData = data[shuffledIdx]
        #     X_shuffled.append(shuffledData)

        for i in range(0, len(Y), batchSize):
            t += 1
            xi = X_shuffled[:][i, i+batchSize]
            yi = Y_shuffled[i, i+batchSize]

            eta = learningSchedule(t)
            decent = 2*eta/batchSize * (xi@(prediction-yi))
            projectedDecent = projectionMatrix(feasibleSpace(dim)) @ decent

            weights = weights-projectedDecent
        # if weights.min() > 0:
        #     reached_bound = False
        # else:
        #     if reached_bound:
        #         break
        #     else:
        #         weights = np.array(weights) + projectedDecent * \
        #             (weights.min()/projectedDecent[np.argmin(weights)])
        #         reached_bound = True

            prediction = weights @ X
            costHistory[c] = ModelPair(Y, weights@X).MSE()
            c += 1

    # print(costHistory)
    # print(weights)
    return prediction, weights, costHistory


def batchGradientDesent(X, Y, weights, alpha, iters):
    costHistory = [0]*iters
    prediction = weights @ X
    print(X[:-1].shape)
    for i in range(iters):
        weights = weights - (alpha/len(Y)) * (X @ (prediction - Y))
        costHistory[i] = ModelPair(Y, weights@X).MSE()
        prediction = weights @ X
    # print(costHistory)
    # print(weights)
    return prediction, weights, costHistory


def optimization(startdate, enddate):

    JRA = xr.open_dataset(JRA_dir, engine="cfgrib")
    JRA = preprocessed(JRA, "JRA").uas.values.ravel()

    MRI = xr.open_dataset(MRI_dir, engine="h5netcdf").uas.sel(
        time=slice(startdate, enddate)).values.ravel()
    MIROC6 = xr.open_dataset(MIROC6_dir, engine="h5netcdf").uas.sel(
        time=slice(startdate, enddate)).values.ravel()

    # print(JRA.size, MRI.size, MIROC6.size)
    fake = MRI * 1.01-np.cos(MRI)
    weights = np.array([10, 10, 10])

    res, weights, _ = batchGradientDesent(
        np.array([MRI, MIROC6, fake]), JRA, weights, 0.005, 100)

    return res, weights


def main():
    JRA = xr.open_dataset(JRA_dir, engine="cfgrib")
    JRA = preprocessed(JRA, "JRA").uas.values.ravel()
    MRI = xr.open_dataset(MRI_dir, engine="h5netcdf").uas.sel(
        time=slice("1981-01-01", "1981-01-10")).values.ravel()
    MIROC6 = xr.open_dataset(MIROC6_dir, engine="h5netcdf").uas.sel(
        time=slice("1981-01-01", "1981-01-10")).values.ravel()
    fake = MRI * 1.01-np.cos(MRI)
    data_list = [MRI, MIROC6, fake]

    res, weights = optimization("1981-01-01", "1981-01-10")

    data_list.append(res)

    print(comparision(JRA, data_list))


main()
