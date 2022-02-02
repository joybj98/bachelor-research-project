import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib.offsetbox import AnchoredText


def main():

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([95, 180, -10, 60], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    x1, y1 = [127, 127, 180, 180, 127], [0, 31.2, 31.2, 0, 0]
    x2, y2 = [123, 123, 154, 154, 123], [20.5, 45.5, 45.5, 20.5, 20.5]

    x3, y3 = [95, 95, 150, 150, 95], [-10, 30, 30, -10, -10]

    x4, y4 = [95, 95, 122, 122, 95], [28, 60, 60, 28, 28]

    ax.plot(x1, y1, marker='.', color='y')
    ax.fill(x1, y1, color='y', alpha=0.5, label=r'$S_O$')
    ax.plot(x3, y3, marker='.', color='b')
    ax.fill(x3, y3, color='b', alpha=0.5, label=r'$S_M$')
    ax.plot(x2, y2, marker='.', color='r')
    ax.fill(x2, y2, color='r', alpha=0.5, label=r'$S_J$')
    ax.plot(x4, y4, marker='.', color='g')
    ax.fill(x4, y4, color='g', alpha=0.5, label=r'$S_L$')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=4)

    # text = AnchoredText('asdfasdf',loc=4, prop={'size': 12}, frameon=True)

    plt.show()


if __name__ == '__main__':
    path = '/Users/wangshiyuan/Documents/bachelor-research-project/interpolated_gcms_mon/CNRM-ESM2-1/CNRM-ESM2-1.nc'

    # ds = xr.open_dataset(path, engine = 'h5netcdf')
    # for time in range(10):
    #     p = ds.isel(time = 0).uas.plot(
    #     subplot_kws = dict(projection = ccrs.Orthographic(140,20),facecolor = 'gray'),
    #     transform = ccrs.PlateCarree())

    #     p.axes.coastlines()

    main()
