#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:44:09 2022

@author: wangshiyuan
"""
import pandas as pd
import xarray as xr
from glob import glob
from scipy.stats import ttest_rel, ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm


from weighting import Evaluating_Indices


ATTRS = Evaluating_Indices[1:3]

Evaluating_Indices = Evaluating_Indices[3:] + [f'w{i}'for i in range(1, 6)]
MODELS = ['MEW', 'REA', 'mean'] + [f'G{i+1}' for i in range(5)]
CASES = list(range(1, 11))

n_models = 8
n_cases = 10
n_indices = len(Evaluating_Indices)

ExpNames = ['ALL', 'U', 'V', 'SO', 'SJ', 'SJ_U', 'SJ_V', 'SM', 'SL']

PATH = '/Users/wangshiyuan/Documents/bachelor-research-project/output/'


def excel_to_nc():
    path = PATH

    def processOneFile(filename, write=False):

        cases = CASES
        models = MODELS
        indices = Evaluating_Indices

        dic = pd.read_excel(path+f'{filename}.xlsx', sheet_name=None, index_col=0,
                            header=None, names=list(range(n_cases)), nrows=n_indices, skiprows=4)

        array = np.array([df.iloc[:, :].T.to_numpy()
                          for df in dic.values()])

        assert array.shape == (n_models, n_cases, n_indices), f'{array.shape}'

        ds = xr.DataArray(array,
                          coords=[models, cases, indices],
                          dims=['model', 'case', 'index']
                          )
        print(ds)
        if write:
            ds.to_netcdf(path+f'{filename}.nc', engine='h5netcdf')

    for exp in ExpNames:
        processOneFile(exp, write=True)

    exps = sorted(ExpNames)

    ds = xr.concat([xr.open_dataarray(p, engine='h5netcdf')
                   for p in sorted(glob(path + '*.nc'))], dim=exps)

    ds = ds.rename({'concat_dim': 'exp'})

    ds.to_netcdf(path+'all_data.nc', engine='h5netcdf')


def weights_evaluating():
    '''
    ======== settings =============
    '''
    # expName = 'S2'

    filepath = PATH + 'all_data.nc'

    # write = False

    '''
    =============== preprocessing ============
    '''

    with xr.open_dataarray(filepath, engine='h5netcdf').sel(model=['MEW', 'REA'], index=[f'w{i}' for i in range(1, 6)]) as ds:

        ds.load()

    # cases = ds.case.data

    xtick = ds.case.data
    # ytick = [f'MEW_{str(exp)}' for exp in ds.exp.data] + \
    #     [f'REA_{str(exp)}' for exp in ds.exp.data]
    # models = ds.model.data
    # exps = ds.exp.data
    ytick = ds.index.data[::-1]

    # ds = ds.mean(dim='case')
    data = ds.isel(exp=0, model=0).T[::-1, :]
    # print(ds)

    # data = np.vstack((ds.isel(model=0).data, ds.isel(model=1)))

    fig, ax = plt.subplots()
    _ = ax.pcolormesh(
        xtick, ytick, data, shading='auto', cmap='Reds', vmin=0, vmax=0.75)

    ax.set_xticks(xtick)
    ax.set_xticklabels([f'case {i}' for i in xtick])
    # ax.set_yticks(np.arange(len(weights)), labels=weights)

# Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
    for i in range(len(ytick)):
        for j in range(len(xtick)):
            text = ax.text(j+1, i, round(float(data[i, j]), 3),
                           ha="center", va="center", color="k")

    # ax.set_title("sdfsdfsdf")
    fig.colorbar(_,)
    fig.tight_layout()
    plt.show()


def t_testing(expName, index, write=False):
    '''
    ======== settings =============
    '''

    filepath = PATH + 'all_data.nc'

    referModelName = 'mean'

    # index = 'RMSE'

    write = False

    p_value = 0.01

    '''
    =============== preprocessing ============
    '''

    fig, ax = plt.subplots()
    label = index
    if label == 'RMSE':
        label = label + ' ' + r'$(m  ·  s^{-1})$'

    ax.set_ylabel(label)

    ax.set_xlabel('cases')

    '''
    ========= evaluating ============
    '''
    with xr.open_dataarray(filepath, engine='h5netcdf').sel(index=index, exp=expName) as ds:

        ds.load()

    refer = ds.sel(model=referModelName)

    for model in ds.model:
        if model == referModelName:
            continue

        evaluated = ds.sel(model=model)
        print(evaluated)
        ax.plot(evaluated.values, label=model.values)

        t, p = ttest_rel(evaluated, refer)

        print(f'for {model.model.values} and {refer.model.values}')

        if p > p_value:
            print('no significant different')
        else:
            print('has significant different')
            if t > 0:
                print('Bigger')
            elif t < 0:
                print('Smaller')

        print(t, p)
        print('\n\n')

    '''
    ========== plotting ===========
    '''

    ax.plot(refer.values, linewidth=3, label=referModelName, color='k')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    if write:

        fig.savefig(PATH+f'{expName}_{index}.png')

    # return exp


if __name__ == '__main__':

    # excel_to_nc()
    weights_evaluating()
    # t_testing('SJ_V', 'R')
    # for exp in ExpNames:

    #     t_testing(exp, 'R')
    #     t_testing(exp, 'RMSE')
