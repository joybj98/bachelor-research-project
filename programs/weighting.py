
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:34:55 2021

@author: wangshiyuan
"""

import xarray as xr
from time import time
from math import sqrt
import pandas as pd
import numpy as np
import datetime
from scipy import linalg
from scipy.stats import entropy
import matplotlib.pyplot as plt


LATBOUND = [-10, 60]
LONBOUND = [95, 180]
STARTDATE = '1981-01-01'
ENDDATE = '1981-12-31'

Evaluating_indices = ['mean', 'bias', 'R', 'RMSE', 'stdev']

JRA_dir = '../downloads/JRA/'
MRI_dir = "../interpolated_gcms_mon/MRI/"
MIROC6_dir = "../interpolated_gcms_mon/MIROC6/"
CNRM_dir = "../interpolated_gcms_mon/CNRM-ESM2-1/"


class ModelPair():
    def __init__(self, referModelData_1dim, selfModelData_1dim):
        self.refer = referModelData_1dim
        self.model = selfModelData_1dim

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


# class ModelPair(referData, modelData):
#     def

def droppingVariablesOfds(ds, distToBound=0):
    # lat_idxs = [i for i in range(len(ds.lat)) if LATBOUND[0]-distToBound <= float(
    #     ds.lat[i]) <= LATBOUND[1]+distToBound]
    # lon_idxs = [i for i in range(len(ds.lon)) if LONBOUND[0]-distToBound <= float(
    #     ds.lon[i]) <= LONBOUND[1]+distToBound]

    return ds.sel(lat=slice(LATBOUND[0], LATBOUND[1]), lon=slice(LONBOUND[0], LONBOUND[1]))


def preprocessed(ds, modelType='', startdate=None, enddate=None):
    def JRA_preprocess(ds):
        ds = ds.rename(
            {'u': 'uas', 'latitude': 'lat', 'longitude': 'lon'})
        ds = droppingVariablesOfds(ds)
        ds = ds.transpose('lat', 'lon', 'time')
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
    P = projectionMatrix(feasibleSpace(dim))
    for i in range(iters):

        decent = (alpha/len(Y)) * (X@(prediction-Y))
        P = projectionMatrix()
        projectedDecent = P@decent

        weights = weights-projectedDecent
        if weights.min() > 0:
            reached_bound = 0
        else:
            if reached_bound >= 3:
                break
            else:
                weights = np.array(weights) + projectedDecent * \
                    (weights.min()/projectedDecent[np.argmin(weights)])
                reached_bound += 1

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
    P = projectionMatrix(feasibleSpace(dim))
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
            projectedDecent = P @ decent

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


def optimized_weighting(startdate, enddate):

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


class ReferData():
    def __init__(self, referData):
        self.data = referData
        self.roll = self.data.rolling(time=20*12)

    def rollingMax(self):
        return self.roll.max()

    def rollingMin(self):
        return self.roll.min()

    def epsilon(self):
        return self.roll.max()-self.roll.min()


class REAWeighting():

    def __init__(self, referData, modelData_list):
        self.refer = ReferData(referData)
        self.models = modelData_list

    def B_T(self):

        B_T = [(model-self.refer).rolling(time=12*20).mean()
               for model in self.models]

        self.B_T = B_T
        return self.B_T

    def D_T(self):
        model_mean = self.models.mean(axis=0)

        D_T = [(model-model_mean).rolling(time=12*20).mean()
               for model in self.models]
        self.D_T = D_T

        return self.D_T

    def REA(self):

        epsilon = self.refer.epsilon()

        def RB(self):
            return epsilon/abs(self.B_T)

        def RD(self):
            return epsilon/abs(self.D_T)

        def R(self):
            return RB*RD

        return 0


def test():
    y = xr.Dataset(
        {
            "x": (["lat", "lon", "time"], np.array(range(12*2*10*10)).reshape(10, 10, -1))
        },
        coords={
            "lat": (["lat"], np.array(range(10))),
            "lon": (["lon"], np.array(range(10))),
            'time': (['time'], np.array(range(24)))
        })
    a = (y*1.01).rename({"x": "a"})
    b = (a + 1).rename({"a": "b"})
    c = (a - 1).rename({"a": "c"})
    lst = xr.merge([a, b, c])
    z = REAWeighting(y, lst)
    print(z.D_T())
    # a = a.x[0, 0, :]


def main():
    JRA = xr.open_dataset(JRA_dir + "/JRA.nc", engine="h5netcdf")
    # JRA = droppingVariablesOfds(JRA)
    # print(JRA)

    # MRI = xr.open_dataset(MRI_dir + "/MRI.nc", engine="h5netcdf")
    MIROC6 = xr.open_dataset(MIROC6_dir + "/MIROC6.nc", engine="h5netcdf")
    CNRM = xr.open_dataset(CNRM_dir + "/CNRM-ESM2-1.nc", engine="h5netcdf")
    # JRA = xr.merge([JRA.uas.drop_duplicates("time").to_dataset(),
    # JRA.vas.drop_duplicates("time").to_dataset()])
    # print(JRA)
    # JRA.to_netcdf(JRA_dir + "/JRAa.nc", engine="h5netcdf")
    print((MIROC6-JRA).rolling(time=20*12, center=True, min_periods=120).mean())
    # modelList=(MRI,MIROC6,CNRM)
    # print(JRA, MRI, MIROC6, CNRM)
    # new = xr.Dataset(
    #     {
    #         "x": (["lat", "lon", "time"], np.array(range(3*4*5)).reshape(3, 4, 5))
    #     },
    #     coords={
    #         "lat": (["lat"], np.array(range(3))),
    #         "lon": (["lon"], np.array(range(4))),
    #         'time': (['time'], np.array(range(5)))
    #     }
    # )
    # print(new.rolling(time=3, center=True, min_periods=2).max().x.data)
    # JRA = referData(JRA)
    # print(JRA.epsilon().isel(lat=0, lon=0).uas.data)

    # REA_weighting(JRA, modelList)
    #

    # for i, data in enumerate(r.mean().data):
    #     if data > -10000:
    #         print(i)
    #         break
    # print(r.mean().data)


if __name__ == "__main__":
    t = time()
    test()
    # JRA = preprocessed(JRA, "JRA").uas.values.ravel()
    # MRI = xr.open_dataset(MRI_dir, engine="h5netcdf").uas.sel(
    #     time=slice("1981-01-01", "1981-01-10")).values.ravel()
    # MIROC6 = xr.open_dataset(MIROC6_dir, engine="h5netcdf").uas.sel(
    #     time=slice("1981-01-01", "1981-01-10")).values.ravel()
    # fake = MRI * 1.01-np.cos(MRI)
    # data_list = [MRI, MIROC6, fake]

    # res, weights optimization("1981-01-01", "1981-01-10")

    # data_list.append(res)ff

    # print(comparision(JRA, data_list))

    # print(comparision(JRA, data_list))
    print(f"over. time is {time()-t:.2f}")
