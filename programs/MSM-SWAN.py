'''

**IMPORTANT NOTE**

MSM data alignment
NS: 47.6[deg] - 22.4[deg], 0.05[deg], 505 nodes
EW: 120[deg] - 150[deg],  0.0625[deg], 481 nodes

For MSM dimension matrix (505, 481), data is aligned like...

(120.0000, 47.60)   (120.0625, 47.60)   .   .   (150.0000, 47.60)
(120.0000, 47.55)
(120.0000, 47.50)
.
.
.
(120.0000, 22.40)                               (150.0000, 22.40)

For SWAN's READinp option=1, your input data should be aligned like

    0,y  1,y  2,y  ...  x,y
     .                   .
     .                   .
     .                   .
    0,1  1,1  2,1  ...  x,1
    0,0  1,0  2,0  ...  x,0

Hence, MSM.nc must be put as it is when converted into SWAN's input.

'''
import numpy as np
import netCDF4 as nc


def main(data_list, output_name):
    data_entry = len(data_list)
    msm = nc.Dataset(data_list[0], 'r')
    # msm.variables.keys()
    col = len(msm.variables['lon'])
    row = len(msm.variables['lat'])
    time = len(msm.variables['time'])
    wind_col = col
    wind_row = row * 2 * time * data_entry
    wind = np.zeros((wind_row, wind_col))

    for n in range(data_entry):
        msm = nc.Dataset(data_list[n])
        U10 = msm.variables['u']
        V10 = msm.variables['v']
        for i in range(time):
            wind[(row*(i*2))+(n*row*time*2): (row*(i*2+1)) +
                 (n*row*time*2)][:] = U10[i][:][:]
            wind[(row*(i*2+1))+(n*row*time*2): (row*(i*2+2)) +
                 (n*row*time*2)][:] = V10[i][:][:]

    print(wind)

    # np.savetxt(output_name, wind, fmt="%.3f", delimiter='\t')
    # return wind


# execution
if __name__ == '__main__':
    # (EDIT) TYPE MSM DATA AS A LIST (EACH DATA SHOULD BE SEPARATED BY COMMA).
    data_list = ['./0908.nc', './0909.nc']
    # (EDIT) SPECIFY THE OUTPUT DATA NAME (DON'T FORGET ".txt").
    output_name = "WIND_MSM_.txt"
    # DON'T EDIT IT.
    main(data_list, output_name)
    # (EDIT) SPECIFY THE OUTPUT DATA NAME (DON'T FORGET ".txt").
    output_name = "WIND_MSM_.txt"
    # DON'T EDIT IT.
    main(data_list, output_name)
