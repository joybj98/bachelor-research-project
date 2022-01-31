#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:44:09 2022

@author: wangshiyuan
"""
import pandas as pd
import xarray as xr
from scipy.stats import ttest_rel, ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm


from weighting import Evaluating_Indices


ATTRS = Evaluating_Indices[1:3]

Evaluating_Indices = Evaluating_Indices[3:]
MODELS = ['MEW', 'REA', 'mean'] + [f'G{i+1}' for i in range(5)]
CASES = list(range(1, 11))

n_models = 8
n_cases = 10
n_indices = len(Evaluating_Indices)

ExpNames = ['ALL', 'U', 'V']

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


def main():
    '''
    ======== settings =============
    '''
    expName = 'ALL'

    filepath = PATH + f'{expName}.nc'

    referModelName = 'mean'

    index = 'R'

    write = False

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
    with xr.open_dataarray(filepath, engine='h5netcdf').sel(index=index) as ds:

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

        if p > 0.01:
            print('no significant different')
        else:
            print('has significant different')
            if t > 0:
                print('former is bigger than latter')
            elif t < 0:
                print('latter is bigger than former')

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


if __name__ == '__main__':

    # excel_to_nc()
    main()
