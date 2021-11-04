#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:31:42 2021

@author: wangshiyuan
"""

import os
import glob
import shutil as st
import xarray as xr

# imput an ds, output its first time coordinate    
# seems like a stupid way, but I do not know better ways
def readTime(ds):
    string = str(ds.coords['time'][0])
    i = 0
    while (string[i] != '1' and string[i] != '2'):
        i += 1
        time = ''
    for j in range (10):
        time = time + string[i]
        i += 1
    return time



path = '/Users/wangshiyuan/Downloads/1981-2005_u'
# in the '1981-2005_u' folder, there are folders for each year, and in the 
# yearly folders, there are files for every 3 days.



for folderpath in glob.glob(path + '/*'):
    
    for file in glob.glob(folderpath + '/*.idx'):
        os.remove(file)
        # After moving and renaming the files, *.idx would be created
        # don't know the reason, but if we don't delete them, the program
        # will have bugs
        
    for file in glob.glob(folderpath + '/*.wang500703'):
        filename = os.path.basename(file)
        
        ds = xr.open_dataset(file, engine = 'cfgrib')
        timeString = readTime(ds)
        ds.close()
        if not os.path.exists(path + '/u_new'):
            os.mkdir(path + '/u_new')
            # create a new folder in the same path
        if not os.path.exists(path + '/u_new/' + timeString):
            st.move(file, path + '/u_new/' + timeString)
            print(f'file for {timeString} has been moved')    



