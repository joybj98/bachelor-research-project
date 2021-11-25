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


STARTDATE = '1981-01-01'
ENDDATE = '1981-12-31'

Evaluating_indices = ['mean', 'bias', 'R', 'RMSE', 'stdev']


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
