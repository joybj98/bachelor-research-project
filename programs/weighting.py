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

'''

修改一下ModelPair 让他在里面用ravel（）

'''

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
    def __init__(self, referModel, Model):
        '''


        Parameters
        ----------
        referModel : np.ndarray or xr.DataArray

        Model : np.ndarray or xr.DataArray



        '''

        self.refer = referModel if type(
            referModel) is np.ndarray else referModel.data.ravel()
        self.model = Model if type(Model) is np.ndarray else Model.data.ravel()

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

    def indices(self):
        dictionary = {
            'mean': self.mean(),
            'bias': self.bias(),
            'R': self.R(),
            "RMSE": self.RMSE(),
            'stdev': self.stdev()
        }

        return dictionary


def droppingVariablesOfds(ds, distToBound=0):
    return ds.sel(lat=slice(LATBOUND[0]-distToBound, LATBOUND[1]+distToBound), lon=slice(LONBOUND[0]-distToBound, LONBOUND[1]+distToBound))


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


def getVarToDim(ds):
    '''


    Parameters
    ----------
    ds : xarray.Dataset which have multiple variables

    Returns
    -------
    res : xarray.DataArray
          only as one variable named 'variables'.

          Increased a dimension. different coordinate in the new dimension shows different variables in the old dataset.

    '''

    newdim = pd.Index([str(variable) for variable in ds], name='var')

    res = xr.concat([ds[variable] for variable in ds], newdim)
    res = res.rename('variables')
    return res


def comparision(referData, modelData_list: list):

    # the modelData_list includes lists of data of different models
    # which means a list of Models' data's list

    output_index = [idx for idx in Evaluating_indices]
    output_data = np.array([])

    for i, modelData in enumerate(modelData_list):

        if type(referData) is list and len(referData) == len(modelData_list):
            pair = ModelPair(referData[i], modelData)
        else:

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


class OptimizedWeighting():
    def __init__(self, referData, modelData):
        self.refer = referData.data.ravel()
        self.models = np.array([model.data.ravel() for model in modelData])
        self.weights = np.array([1/len(modelData)]*len(modelData))

    def NewbatchGradientDesent(self, alpha, iters):

        X = self.models
        Y = self.refer
        weights = np.array(self.weights)

        def feasibleSpace(dim):
            I = np.identity(dim-1, dtype="int")
            ones = np.ones(dim-1).reshape(1, dim-1)

            return np.append(-I, ones, axis=0)

        def projectionMatrix(A):
            return A @ linalg.inv(A.T @ A) @ A.T

        dim = len(X)
        # print(dim)
        weights[-1] = 1-sum(weights[:-1])
        costHistory = [0]*iters

        prediction = weights @ X
        P = projectionMatrix(feasibleSpace(dim))
        reached_bound = 0
        for i in range(iters):

            decent = (alpha/len(Y)) * (X@(prediction-Y))

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
        print(weights)

        print(costHistory)
        # print(weights)
        return prediction, weights


class REAWeighting():

    '''
    REFERS TO:

    Giorgi, F., & Mearns, L. O. (2002). Calculation of Average, Uncertainty Range, and Reliability of Regional Climate Changes from AOGCM Simulations via the “Reliability Ensemble Averaging” (REA) Method, Journal of Climate, 15(10), 1141-1158. Retrieved Dec 13, 2021, from https://journals.ametsoc.org/view/journals/clim/15/10/1520-0442_2002_015_1141_coaura_2.0.co_2.xml


    '''

    def __init__(self, referData, modelData):
        '''


        Parameters
        ----------
        referData : xarray.DataArray

        modelData : list whose units are xarray.DataArray

        Because we are using DataArray.rolling method, we use DataArray instead of np.ndarray initially.

        '''
        self.refer = referData
        self.models = modelData
        self.weights = None
        self.res = np.array(None)
        self.computed = False

    def initializeRB(self):
        def getB(self):
            B = [(model-self.refer).rolling(time=12*20).mean()
                 for model in self.models]
            return B

        def getEpsilon(self):
            r = self.refer.rolling(time=12*20)

            epsilon = r.max() - r.min()
            return epsilon

        B = getB(self)
        self.epsilon = getEpsilon(self)

        def getRB(self):
            def getRB_i(B_i):
                RB_i = (self.epsilon/abs(B_i)).mean()

                return RB_i if RB_i < 1 else 1

            return list(map(getRB_i, B))

        self.RB = getRB(self)

    def _getWeights(self):

        epsilon = self.epsilon
        RB = self.RB

        def D(self):

            model_mean = self.res if self.res.any() else sum(
                self.models)/len(self.models)
            # print(model_mean, '\n\n\n\n')

            D = [(model-model_mean).rolling(time=12*20).mean()
                 for model in self.models]

            return D

        def RD(self):
            def getRD_i(D_i):
                RD_i = (epsilon/abs(D_i)).mean()
                return RD_i if RD_i < 1 else 1

            return list(map(getRD_i, D(self)))

        def R(self):
            # print(RB(self), '\n\n\n', RD(self), '\n')
            return [RB_i * RD_i for RB_i, RD_i in zip(RB, RD(self))]

        R = np.array(R(self))
        self.weights = R/sum(R)
        return self.weights

    def _getWeightedMean(self):
        weights = self._getWeights()

        # print(self.weights.shape, self.data.shape)
        weighted = 0
        for w, model in zip(weights, self.models):
            weighted += w*model

        return weighted

    def _computing(self, iters=100, breakBound=1/5000):
        self.initializeRB()
        # Now we have self.RB and self.epsilon

        MSE = float('inf')

        for _ in range(iters):

            res = self._getWeightedMean()
            # print(res, '\n\n\n\n')
            temp = ModelPair(self.refer.data.ravel(),
                             res.data.ravel()).MSE()
            if abs(MSE-temp) > breakBound:
                # print(MSE-temp)
                MSE = temp
                self.res = res
            else:
                break
            print(f'iter is {_}')

            self.computed = True

    def getRes(self):
        if not self.computed:
            self._computing()
        return self.res

    def getWeights(self):
        if not self.computed:
            self._computing()
        return self.weights


def evaluate(referData, modelData):

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    time = referData.time.to_index()
    periods = list(split(time, 12))

    for testPeriod in periods:

        rest = [timeStep for timeStep in time if timeStep not in testPeriod]
        y_test = referData.sel(time=testPeriod)
        x_test = [model.sel(time=testPeriod) for model in modelData]

        y_opt = referData.sel(time=rest)
        x_opt = [model.sel(time=rest) for model in modelData]

        REA = REAWeighting(y_opt, x_opt)
        weights = REA.getWeights()
        x_weighted = sum([w*x for w, x in zip(weights, x_test)])

        print(comparision(y_test.data.ravel(), [x_weighted.data.ravel()]))


def test():

    time = pd.date_range('1970-01-01', freq='MS', periods=12*45)
    y = xr.Dataset(
        {
            "x": (["lat", "lon", "time"], np.array(range(12*45*100*150)).reshape(100, 150, -1)),
            'y': (["lat", "lon", "time"], np.array(range(-100, 12*45*100*150-100)).reshape(100, 150, -1)),
        },
        coords={
            "lat": (["lat"], np.array(range(100))),
            "lon": (["lon"], np.array(range(150))),
            'time': (['time'], time)
        })

    y = getVarToDim(y)

    # print(y.isel(lat=0, lon=0).data.ravel()[::10])

    a = y*1.1
    b = a*0.9
    c = a + 0.5*a.mean()

    lst = [a, b, c]
    print(type(lst))

    # evaluate( y, lst)


def main():
    JRA = xr.open_dataset(JRA_dir + "/JRA.nc", engine="h5netcdf")

    MIROC6 = xr.open_dataset(MIROC6_dir + "/MIROC6.nc", engine="h5netcdf")
    CNRM = xr.open_dataset(CNRM_dir + "/CNRM-ESM2-1.nc", engine="h5netcdf")


if __name__ == "__main__":
    t = time()
    test()

    print(f"over. time is {time()-t:.2f}")
