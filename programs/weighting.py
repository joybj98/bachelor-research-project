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
import os
from glob import glob
import sys
np.set_printoptions(threshold=False)
# import datetime
# from scipy.stats import entropy


STARTDATE = '1960-01-01'
ENDDATE = '2014-12-31'

Evaluating_Indices = ['modelName', 'startDate',
                      'endDate', 'mean', 'stdev', 'max', 'min', 'interval', '5p', '25p', '50p', '75p', '95p', 'bias', 'e5p', 'e25p', 'e50p', 'e75p', 'e95p', 'ae5p', 'ae25p', 'ae50p', 'ae75p', 'ae95p', 'R', 'RMSE', 'RSD', 'eci99a',  'eci99b', 'eci95a',  'eci95b']

JRA_dir = '../downloads/JRA/'

model_dir = "../interpolated_gcms_mon/"

testing = True
testing = False


def plot(JRA, modelData):
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


class ModelPair():
    def __init__(self, referModel, model, givenModelName=None):
        '''


        Parameters
        ----------
        referModel : xr.DataArray
                    make sure the order of axises are the same.

        Model : xr.DataArray



        '''

        self.refer = referModel.ravel() if type(
            referModel) is np.ndarray else referModel.data.ravel()
        self.model = model.ravel() if type(model) is np.ndarray else model.data.ravel()
        self.startDate = model.time.to_index()[0].strftime('%Y-%m')
        self.endDate = model.time.to_index()[-1].strftime('%Y-%m')
        self.error = np.array(self.model-self.refer)
        self.AE = abs(self.error)

        if givenModelName:
            self.modelName = givenModelName
        elif model.name:
            self.modelName = model.name
        else:
            try:
                self.modelName = str(model.modelName.values)
            except:
                self.modelName = None

    def bias(self):
        return self.error.mean()

    def stdev(self):
        return np.std(self.model)

    def interval(self):
        return self.model.max() - self.model.min()

    def R(self):
        return np.corrcoef(self.refer, self.model)[0][1]
    # using R[0][1] is because R is a 2 by 2 matrix. which includes Rxx,Rxy,Ryx,Ryy. And we only use Rxy (equals to Ryx)

    def MSE(self):
        return np.square(self.error).mean()

    def RMSE(self):
        return sqrt(self.MSE())

    def RSD(self):
        return self.stdev()/np.std(self.refer)

    def mean(self):
        return np.array(self.model).mean()

    def confidenceInterval(self, data, alpha):
        # plt.plot(sorted(self.error()))
        # plt.show()

        _ = st.t.interval(alpha=alpha, df=len(data)-1,
                          loc=data.mean(), scale=st.sem(data))
        # print(_)
        return _

    def percentiles(self, data, ps: list):

        return [np.percentile(data, p) for p in ps]

    def indices(self):
        '''
        Note that the Evaluating_indices must be a subset of this dictionary.
        '''
        dictionary = {
            'modelName': self.modelName,
            'startDate': self.startDate,
            'endDate': self.endDate,
            'mean': self.mean(),
            'stdev': self.stdev(),
            'interval': self.interval(),
            'ci99a': self.confidenceInterval(self.model, 0.99)[0],
            'ci99b': self.confidenceInterval(self.model, 0.99)[0],
            'ci95a': self.confidenceInterval(self.model, 0.95)[0],
            'ci95b': self.confidenceInterval(self.model, 0.95)[1],
            'bias': self.bias(),
            'R': self.R(),
            "RMSE": self.RMSE(),
            'RSD': self.RSD(),
            'eci99a': self.confidenceInterval(self.error, 0.99)[0],
            'eci95a': self.confidenceInterval(self.error, 0.95)[0],
            'eci99b': self.confidenceInterval(self.error, 0.99)[1],
            'eci95b': self.confidenceInterval(self.error, 0.95)[1],
            'max': self.model.max(),
            'min': self.model.min(),


        }
        ps = [5, 25, 50, 75, 95]
        keys = [str(p)+'p' for p in ps]
        ekeys = ['e'+k for k in keys]
        aekeys = ['ae'+k for k in keys]

        values = self.percentiles(self.model, ps) + \
            self.percentiles(self.error, ps) + self.percentiles(self.AE, ps)

        new = {k: v for k, v in zip(keys+ekeys+aekeys, values)}

        dictionary.update(new)

        # print(dictionary)

        res = [dictionary[key] for key in Evaluating_Indices]

        return res


def weightOfMean(n):
    return np.array([1/n]*n)


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

        if testing:
            self.iters = 3
            self.initializing_iters = 1

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
        init_alpha = 2
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

        reached_bound = 0

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
                '''
                checking for bound constrains
                '''
                if weights.min() > 0:
                    reached_bound = 0
                else:
                    if reached_bound >= 4:
                        idx = np.argmin(weights)
                        weights = np.array(weights) + projectedDescent * \
                            (-weights.min()/projectedDescent[idx])

                        weights = weights[:idx] + weights[idx+1:]
                        weights = weights/sum(weights)
                        dim -= 1
                        P = projectionMatrix(feasibleSpace(dim))
                        self.models = self.models[np.nonzero(self.weights)]
                        # to decrease the dimension

                        self.weights[self.weights != 0][idx]
                        nonzero_idx = np.nonzero(self.weights)
                        real_idx = nonzero_idx(idx)
                        self.weights[real_idx] = 0
                        # to find the idx-th nonzero element in self.weights
                        # cannot use a[a!=0][idx]=0

                        assert np.count_nonzero(self.weights) == dim
                        self.models = self.models[np.nonzero(self.weights)]

                    else:
                        weights = np.array(weights) + projectedDescent * \
                            (-weights.min() /
                             projectedDescent[np.argmin(weights)])

                        '''
                        roll back the weights so that the min of weights to 0
                        '''
                        reached_bound += 1

            costHistory[epoch] = MSE(Y, weights@X)
            print(epoch, ': ', costHistory[epoch])
            assert np.array(np.nonzero(self.weights)).size == len(weights)
            self.weights[self.weights != 0] = weights
            print(self.weights)

        return prediction, self.weights, [c for c in costHistory[::-1] if c != 0][0]

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
            # weights = np.random.rand(len(self.models))
            # weights = weights/sum(weights)
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

    def __init__(self, referData_weighting, modelData_weighitng, modelData_testing, iters=30):
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
        self.iters = iters
        if testing:
            self.iters = 3

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

    def _compute(self, breakBound=1/10000):

        self.initializeRB()
        # Now we have self.RB and self.epsilon

        MSE = float('inf')

        for _ in range(self.iters):

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
            [v for v in pair.indices()], dtype=object).reshape(-1, 1)

        output = np.append(
            output, indices, axis=1) if output.size > 0 else np.array(indices)

    # print(output)
    return output


def evaluate(referData, modelData):
    '''


    '''

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

    n_splits = 10
    n_models = len(modelData)+3
    time = referData.time.to_index()
    result = np.array([])
    cnt = 0
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
        x_optmini.name = 'MEW'
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

        if testing:
            cnt += 1
            if cnt > 1:
                break

    for r in result:
        print(r)

    output = []
    for i in range(n_models):

        all_indices = Evaluating_Indices + \
            [f'w{i+1}' for i in range(len(modelData))]

        modelres = pd.DataFrame(
            result[:, i::n_models], index=all_indices, columns=list(range(result.shape[1]//n_models)))

        output.append(modelres)

    return output


def test():
    '''
    for testing and checking runtime at the start of this program.
    now I dont use this at all.
    '''

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

    y = y.to_array()

    # print(y.isel(lat=0, lon=0).data.ravel()[::10])

    a = y*1.1
    b = a*0.9
    c = a + 0.5*a.mean()

    lst = [a, b, c]
    z = REAWeighting(y, lst)

    print(z.getRes())


def main():
    '''
    ======================= settings ==============================
    '''
    testing = True
    testing = False

    # ALL_VARIABLEs == ['uas', 'vas']
    FOCUSING_VARIABLE = ['vas']

    LATBOUND = [-10, 60]
    LONBOUND = [95, 180]

    # LATBOUND = [-10, 60]
    # LONBOUND = [120, 152]

    '''
    all domain:
    LATBOUND = [-10,60]
    LONBOUND = [95,180]
    '''

    FILENAME = 'V'

    HAVEFAKE = False

    np.random.seed(42)

    # FORMAT == 'xlsx'

    FORMAT = 'xlsx'

    '''
    ========================= Preprocessing ===================================
    '''

    all_path = [JRA_dir + 'JRA.nc'] + sorted(glob(model_dir+'*/*.nc'))

    def getName(path):
        return str(os.path.splitext(os.path.basename(path))[0])

    all_name = pd.Index([getName(path)
                         for path in all_path], name='modelName')

    def findtime():
        with xr.open_dataset(all_path[1], engine='h5netcdf') as ds:
            time = ds.sel(time=slice(STARTDATE, ENDDATE)).time
            time.load()
        return time

    time = findtime()

    def process_one_path(path):
        with xr.open_dataset(path, engine='h5netcdf') as ds:
            ds = transformed(ds)
            ds.load()
        return ds

    def transformed(ds):
        ds = ds[FOCUSING_VARIABLE]
        ds = ds.sel(time=slice(STARTDATE, ENDDATE))
        ds = ds.assign_coords(time=time)

        ds = ds.sel(lat=slice(LATBOUND[0], LATBOUND[1]), lon=slice(
            LONBOUND[0], LONBOUND[1]))

        return ds.to_array()

    all_data = [process_one_path(p) for p in all_path]

    '''
    all_data[0] is JRA, else are GCMs
    '''

    if HAVEFAKE:
        all_data[4] = all_data[5] + \
            abs(all_data[0]).mean()*np.sin(10*all_data[1])

    all_models = xr.concat(all_data, all_name)

    # print(type(all_models.modelName[1].values))
    print('concat over')
    print(all_models)

    # for p in [ModelPair(all_models[0], m) for m in all_models]:
    #     print(p.indices())
    # assert False

    '''
    ==================== Process and Write ========================
    '''
    output = evaluate(all_models[0], all_models[1:])

    if FORMAT == 'xlsx':

        write_dir = f'../output/{FILENAME}.xlsx'

        with open(write_dir, 'w+'):
            pass

        with pd.ExcelWriter(write_dir, mode='w') as writer:
            for modelres in output:
                modelres.to_excel(writer, sheet_name=f'{modelres.iloc[0,0]}')

    elif FORMAT == 'nc':

        nc = xr.Dataset.from_dataframe(output[0])
        print(nc)
        assert False

        write_dir = f'../output/{FILENAME}.nc'


if __name__ == "__main__":
    t = time()

    main()

    print(f"over. time is {time()-t:.2f} sec")
