#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:44:09 2022

@author: wangshiyuan
"""
import pandas as pd
from scipy.stats import ttest_rel
import numpy as np


def main():

    filepath = '/Users/wangshiyuan/Documents/bachelor-research-project/output/ALL.xlsx'
    df = pd.read_excel(filepath, sheet_name=None,
                       index_col=0, header=None, names=list(range(10)), nrows=1, skiprows=10)

    mew = df['REA'].to_numpy().ravel()
    mean = df['mean'].to_numpy().ravel()

    print(mew, mean)
    print(mew-mean)

    t, p = ttest_rel(mew, mean)
    if p > 0.05:
        print('no significant different')
    else:
        print('has significant different')
        if t > 0:
            print('former is bigger than latter')
        elif t < 0:
            print('latter is bigger than former')

    print(t, p)
    pass


if __name__ == '__main__':
    main()
