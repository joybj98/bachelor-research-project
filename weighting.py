#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:34:55 2021

@author: wangshiyuan
"""

import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import linalg
import xarray as xr
from time import time
from math import sqrt
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=False)
# import datetime
# from scipy.stats import entropy


LATBOUND = [-10, 60]
LONBOUND = [95, 180]
STARTDATE = '1960-01-01'
ENDDATE = '2014-12-31'

Evaluating_indices = ['modelName', 'startDate',
                      'endDate', 'mean', 'bias', 'R', 'RMSE', 'stdev', 'ci99',  'ci95', 'w1', 'w2', 'w3']

JRA_dir = '../downloads/JRA/'
MRI_dir = "../interpolated_gcms_mon/MRI/"
MIROC6_dir = "../interpolated_gcms_mon/MIROC6/"
CNRM_dir = "../interpolated_gcms_mon/CNRM-ESM2-1/"


class ModelPair():
    def __init__(self, referModel, model):
        '''


        Parameters
        ----------
        referModel : xr.DataArray
                    make sure the order of axises is the same.

        Model : xr.DataArray



        '''

        self.refer = referModel.ravel() if type(
            referModel) is np.ndarray else referModel.data.ravel()
        self.model = model.ravel() if type(model) is np.ndarray else model.data.ravel()
        self.startDate = model.time.to_index()[0].strftime('%Y-%m')
        self.endDate = model.time.to_index()[-1].strftime('%Y-%m')
        self.modelName = model.name

    def error(self):
        return np.array(self.model - self.refer)

    def bias(self):
        return self.error().mean()

    def R(self):
        return np.corrcoef(self.refer, self.model)[0][1]
    # using R[0][1] is because R is a 2 by 2 matrix. which includes Rxx,Rxy,Ryx,Ryy. And we only use Rxy (equals to Ryx)

    def MSE(self):
        return np.square(self.error()).mean()

    def RMSE(self):
        return sqrt(self.MSE())

    def stdev(self):
        return np.std(self.error())

    def mean(self):
        return np.array(self.model).mean()

    def confidenceIterval(self, alpha):
        # plt.plot(sorted(self.error()))
        # plt.show()

        _ = st.t.interval(alpha=alpha, df=len(
            self.error())-1, loc=self.error().mean(), scale=st.sem(self.error()))
        # print(_)
        return _

    def indices(self):
        dictionary = {
            'modelName': self.modelName,
            'startDate': self.startDate,
            'endDate': self.endDate,
            'mean': self.mean(),
            'bias': self.bias(),
            'R': self.R(),
            "RMSE": self.RMSE(),
            'stdev': self.stdev(),
            'ci99': self.confidenceIterval(0.99),
            'ci95': self.confidenceIterval(0.95),
        }

        return dictionary


def droppingVariablesOfds(ds, distToBound=0):
    return ds.sel(lat=slice(LATBOUND[0]-distToBound, LATBOUND[1]+distToBound), lon=slice(LONBOUND[0]-distToBound, LONBOUND[1]+distToBound))


def weightOfMean(n):
    return np.array([1/n]*n)


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


def normalized(data):
    return (data-data.min())/(data.max() - data.min())


class OptimizedWeighting():
    def __init__(self, referData, modelData: list, alpha=0.1, iters=60, initializing_iters=20):
        '''


        Parameters
        ----------
        referData : xr.dataArray
            make it to an 1-dimensional np.ndarray
        modelData : list
            a list of xr.dataArray
        alpha : TYPE, optional
            learning rate The default is 0.05.
        iters : TYPE, optional
            times of Descenting from a set of initial weights. The default is 30.

        '''

        self.refer = referData.data.ravel()
        self.models = np.array([model.data.ravel() for model in modelData])
        self.weights = np.array([1/len(modelData)]*len(modelData))
        self.alpha = alpha

        self.iters = iters
        self.initializing_iters = initializing_iters

    def _newBatchGradientDesent(self, initial_weights):

        X = self.models
        Y = self.refer
        alpha = self.alpha
        iters = self.iters
        weights = np.array(initial_weights)

        def feasibleSpace(dim):
            I = np.identity(dim-1, dtype="int")
            ones = np.ones(dim-1).reshape(1, dim-1)

            return np.append(-I, ones, axis=0)

        def projectionMatrix(A):
            return A @ linalg.inv(A.T @ A) @ A.T

        def MSE(refer, model):
            return np.square(model-refer).mean()

        dim = len(X)
        weights[-1] = 1-sum(weights[:-1])
        costHistory = [0]*iters
        prediction = weights @ X
        P = projectionMatrix(feasibleSpace(dim))

        reached_bound = 0
        for i in range(iters):
            Descent = (alpha/len(Y)) * (X@(prediction-Y))
            projectedDescent = P@Descent
            weights = weights-projectedDescent

            if weights.min() > 0:
                reached_bound = 0
            else:
                if reached_bound >= 4:
                    break
                else:
                    weights = np.array(weights) + projectedDescent * \
                        (weights.min()/projectedDescent[np.argmin(weights)])
                    reached_bound += 1

            prediction = weights @ X
            costHistory[i] = MSE(Y, weights@X)
            print(i, weights)
            print(costHistory[i])

        # print(weights)

        return prediction, weights, [c for c in costHistory[::-1] if c != 0][0]

    def _newMiniBatchGradientDescent(self, initial_weights, batchSize=2000):

        X = self.models
        Y = self.refer
        iters = self.iters
        weights = np.array(initial_weights)

        all_iters = (len(Y)//batchSize)*iters
        init_alpha = 1
        end_alpha = 0.2*init_alpha

        t0, t1 = init_alpha*end_alpha*all_iters / \
            (init_alpha-end_alpha), end_alpha*all_iters/(init_alpha-end_alpha)

        def learningSchedule(t):
            return t0/(t+t1)

        def feasibleSpace(dim):
            I = np.identity(dim-1, dtype="int")
            ones = np.ones(dim-1).reshape(1, dim-1)

            return np.append(-I, ones, axis=0)

        def projectionMatrix(A):
            return A @ linalg.inv(A.T @ A) @ A.T

        def MSE(refer, model):
            return np.square(model-refer).mean()

        dim = len(weights)
        weights[-1] = 1-sum(weights[:-1])
        t = 0
        costHistory = [0]*iters
        # prediction = weights @ X
        P = projectionMatrix(feasibleSpace(dim))

        for epoch in range(iters):
            shuffledIdx = np.random.permutation(len(Y))
            X_shuffled = np.array(X[:, shuffledIdx])
            # print(X_shuffled)
            # assert False
            Y_shuffled = Y[shuffledIdx]

            for i in range(0, len(Y), batchSize):
                t += 1
                xi = X_shuffled[:, i:i+batchSize]
                yi = Y_shuffled[i: i+batchSize]

                alpha = learningSchedule(t)
                prediction = weights @ xi
                Descent = (alpha/len(Y)) * (xi@(prediction-yi))
                projectedDescent = P @ Descent

                weights = weights-projectedDescent

            costHistory[epoch] = MSE(Y, weights@X)
            print(epoch, ': ', costHistory[epoch])

        return prediction, weights, [c for c in costHistory[::-1] if c != 0][0]

    def getWeights_BGD(self):
        print('start BGD...')
        iters = self.initializing_iters
        minMSE = float('inf')
        for i in range(iters):

            # weights = np.array([1/len(self.models)]*(len(self.models)-1)) + \
            #     np.random.uniform(-1, 1, size=2)*0.1
            weights = np.random.rand(3)
            weights = weights/sum(weights)
            print(f'for {i} w is')
            print(weights)

            res, weights, MSE = self._newBatchGradientDesent(weights)
            if MSE < minMSE:
                print(f'update! from {minMSE} to {MSE}')

                minMSE = MSE
                print(MSE)
                bestWeights = weights

        return bestWeights

    def getWeights_miniBGD(self):
        print('start miniBGD...')
        iters = self.initializing_iters
        minMSE = float('inf')
        if iters == 1:

            weights = np.array([1/len(self.models)]*len(self.models))
            print('weights are', weights)
            res, weights, MSE = self._newMiniBatchGradientDescent(weights)
            return weights

        for i in range(iters):
            weights = np.random.rand(len(self.models))
            weights = weights/sum(weights)
            print(f'for {i} w is')
            print(weights)

            res, weights, MSE = self._newMiniBatchGradientDescent(weights)
            if MSE < minMSE:
                print(f'update! from {minMSE} to {MSE}')
                minMSE = MSE
                bestWeights = weights

        return bestWeights


class REAWeighting():

    '''
    REFERS TO:

    Giorgi, F., & Mearns, L. O. (2002). Calculation of Average, Uncertainty Range, and Reliability of Regional Climate Changes from AOGCM Simulations via the “Reliability Ensemble Averaging” (REA) Method, Journal of Climate, 15(10), 1141-1158. Retrieved Dec 13, 2021, from https://journals.ametsoc.org/view/journals/clim/15/10/1520-0442_2002_015_1141_coaura_2.0.co_2.xml


    '''

    def __init__(self, referData_weighting, modelData_weighitng, modelData_testing):
        '''


        Parameters
        ----------
        referData_weighting : xr.DataArray

        modelData_weighitng : xr.DataArray
            data for weighting periods (historical results)
        modelData_testing : xr.DatAarray
            model data for test periods (future projection)

        Returns
        -------
        None.

        '''

        self.refer = referData_weighting
        self.models = modelData_weighitng
        self.prediction = modelData_testing

        self.weights = [1/len(self.models)]*len(self.models)
        self.res = np.array(None)
        self.computed = False

    def initializeRB(self):
        def getB(self):
            B = [abs(model-self.refer).mean(dim=('time', 'lat', 'lon'))
                 for model in self.models]
            return B

        def getEpsilon(self):
            mean = self.refer.rolling(time=12*20).mean()

            return mean.max(dim='time')-mean.min(dim='time')

        B = getB(self)
        self.epsilon = getEpsilon(self)
        # print(B)
        # print(self.epsilon)

        def getRB(self):
            def getRB_i(B_i):
                RB_i_z = (self.epsilon/abs(B_i)).mean(dim=('lat', 'lon'))
                RB_i_z = xr.where(RB_i_z < 1, RB_i_z, 1)
                RB_i = np.prod(RB_i_z.data)

                # print('\n\n\n\n', RB_i)
                # print(RB_i)

                return RB_i

            return list(map(getRB_i, B))

        self.RB = getRB(self)
        print(self.RB)

    def _getWeights(self):

        epsilon = self.epsilon
        RB = self.RB

        def D(self):

            model_mean = sum(
                [w*model for w, model in zip(self.weights, self.prediction)])

            D = [abs(model-model_mean).mean(dim=('time', 'lat', 'lon'))
                 for model in self.prediction]

            return D

        def RD(self):
            def getRD_i(D_i):
                RD_i_z = (epsilon/abs(D_i)).mean(dim=('lat', 'lon'))
                RD_i_z = xr.where(RD_i_z < 1, RD_i_z, 1)
                RD_i = np.prod(RD_i_z.data)

                return RD_i

            return list(map(getRD_i, D(self)))

        def R(self):
            return [RB_i * RD_i for RB_i, RD_i in zip(RB, RD(self))]

        R = np.array(R(self))
        self.weights = R/sum(R)
        return self.weights

    def _getWeightedMean(self):
        weights = self._getWeights()
        weighted = 0
        for w, model in zip(weights, self.models):
            weighted += w*model

        return weighted

    def _compute(self, iters=30, breakBound=1/10000):
        self.initializeRB()
        # Now we have self.RB and self.epsilon

        MSE = float('inf')

        for _ in range(iters):

            res = self._getWeightedMean()
            # print(self.weights)

            temp = ModelPair(self.refer, res).MSE()
            print(MSE-temp)
            if abs(MSE-temp) >= breakBound:

                MSE = temp
                self.res = res
            else:
                break
            print(f'iter is {_}', f'weights {self.weights}')

        self.computed = True

    def getRes(self):
        if not self.computed:
            self._compute()
        return self.res

    def getWeights(self):
        if not self.computed:
            self._compute()
        return self.weights


def compare(referData, modelData):
    '''


    Parameters
    ----------
    referData, modelData : xr.dataArray or np.ndarray or list of these two types

    Returns
    -------
    output : np.ndarray

    axis 0 shows different indices, axis 1 shows different models

    '''

    output = np.array([])

    for i, md in enumerate(modelData):
        if type(referData) is list:

            pair = ModelPair(referData[i], modelData)
        else:
            pair = ModelPair(referData, md)

        indices = np.array(
            [v for v in pair.indices().values()], dtype=object).reshape(-1, 1)

        output = np.append(
            output, indices, axis=1) if output.size > 0 else np.array(indices)

    # print(output)
    return output


def evaluate(referData, modelData):

    def split(a, n):
        '''
        To split a into n parts with as same length as possible.

        Parameters
        ----------
        a : iterable
        n : int

        '''
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    n_splits = 12
    n_models = 3+3
    time = referData.time.to_index()
    result = np.array([])

    for p, testPeriod in enumerate(split(time, n_splits)):

        rest = [timeStep for timeStep in time if timeStep not in testPeriod]

        y_test = referData.sel(time=testPeriod)
        x_test = [model.sel(time=testPeriod) for model in modelData]

        y_weight = referData.sel(time=rest)
        x_weight = [model.sel(time=rest) for model in modelData]

        method = OptimizedWeighting(
            y_weight, x_weight, iters=30, initializing_iters=1)
        weights_optmini = method.getWeights_miniBGD()
        # method = OptimizedWeighting(y_weight, x_weight)
        # weights_opt = method.getWeights_BGD()

        method = REAWeighting(y_weight, x_weight, x_test)
        weights_rea = method.getWeights()

        x_optmini = sum([w*x for w, x in zip(weights_optmini, x_test)])
        x_optmini.name = 'optmini'
        # x_opt = sum([w*x for w, x in zip(weights_opt, x_test)])
        # x_opt.name = 'opt'
        x_rea = sum([w*x for w, x in zip(weights_rea, x_test)])
        x_rea.name = 'REA'
        x_mean = sum(x_test)/len(x_test)
        x_mean.name = 'mean'

        result_period = compare(
            y_test, [x_optmini, x_rea, x_mean]+x_test)

        all_weights = [weights_optmini, weights_rea,
                       weightOfMean(len(weights_optmini))]
        all_weights = np.array(all_weights).transpose()

        def addWightstoResult(result, weights):

            empty = np.empty(
                (weights.shape[0], result.shape[1]-weights.shape[1]))
            empty[::] = np.nan
            # print(empty.shape, weights.shape)
            weights = np.append(weights, empty, axis=1)
            return np.append(result, weights, axis=0)

        result_period = addWightstoResult(result_period, all_weights)

        result = np.append(result, result_period,
                           axis=1) if result.size > 0 else result_period

    output = []

    for i in range(n_models):

        modelres = pd.DataFrame(
            result[:, i::n_models], index=Evaluating_indices, columns=list(range(result.shape[1]//n_models)))

        output.append(modelres)

    return output


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
    z = REAWeighting(y, lst)

    print(z.getRes())

    # evaluate( y, lst)


def main():
    JRA = xr.open_dataset(JRA_dir + "/JRA.nc",
                          engine="h5netcdf")[['uas', 'vas']]

    MIROC6 = xr.open_dataset(MIROC6_dir + "/MIROC6.nc",
                             engine="h5netcdf")[['uas', 'vas']]
    CNRM = xr.open_dataset(CNRM_dir + "/CNRM-ESM2-1.nc",
                           engine="h5netcdf")[['uas', 'vas']]
    MRI = xr.open_dataset(MRI_dir + '/MRI.nc',
                          engine="h5netcdf")[['uas', 'vas']]

    # Fake = MIROC6 + abs(JRA).mean()*np.sin(10*CNRM)
    # MRI = Fake

    # MRI.uas.isel(time=100).plot()
    # assert False
    JRA = 0*MIROC6 + JRA
    MRI = 0*MIROC6 + MRI

    JRA, MIROC6, CNRM, MRI = (ds.to_array() for ds in [JRA, MIROC6, CNRM, MRI])

    JRA, MIROC6, CNRM, MRI = JRA.rename('JRA'), MIROC6.rename(
        'MIROC6'), CNRM.rename('CNRM'), MRI.rename('MRI')

    modelData = [MIROC6, CNRM, MRI]

    def plot():
        from mpl_toolkits.mplot3d import Axes3D
        # from matplotlib import cm
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        weights = np.random.rand(100, 3)
        weights = weights/weights.sum(axis=1)[:, np.newaxis]

        md = np.array([m.data.ravel() for m in modelData])
        J = JRA.data.ravel()

        # print(weights.shape, md.shape)

        z = np.square(weights@md-[J]*weights.shape[0]).mean(axis=1)

        print(z.shape)

        ax.view_init(45, 60)

        img = ax.scatter(weights[:, 0], weights[:, 1],
                         weights[:, 2], c=z, cmap=plt.hot())
        fig.colorbar(img)
        plt.show()
    # plot()

    output = evaluate(JRA, modelData)
    for modelres in output:
        print(modelres)

        modelres.to_excel(f'../output/{modelres.iloc[0,0]}.xlsx')


if __name__ == "__main__":
    t = time()
    main()

    print(f"over. time is {time()-t:.2f} sec")
