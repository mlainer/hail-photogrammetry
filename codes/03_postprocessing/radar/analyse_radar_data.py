import pyart
from pyart.aux_io.metranet_cartesian_reader import read_cartesian_metranet
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import num2date, date2num, Dataset
from pyart.graph import GridMapDisplay
from mpl_toolkits.basemap import Basemap

MZC_path = '/scratch/mlainer/hail/radar/MZC/'
BZC_path = '/scratch/mlainer/hail/radar/BZC/'
plot_path = '/scratch/mlainer/hail/plot_radar_hail/'

bzc_files = glob.glob(BZC_path+'*.845')
bzc_files.sort()

mzc_files = glob.glob(MZC_path+'*.845')
mzc_files.sort()

for i, file_bzc in enumerate(bzc_files):
    #file_mzc = mzc_files[i]
    data_bzc = read_cartesian_metranet(file_bzc, additional_metadata=None, chy0=255.,chx0=-160., reader='C')
    #data_mzc = read_cartesian_metranet(file_mzc, additional_metadata=None, chy0=255.,chx0=-160., reader='C')

    #print(data.fields)
    #break
    display = pyart.graph.GridMapDisplayBasemap(data_bzc)

    # create the figure
    font = {'size': 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=[10, 8])

    # Add Basic Title
    #title = 'Basic Plot with Overlay Example Title'
    #     Xleft%, ybot%
    #fig.text(0.5, 0.9, title, horizontalalignment='center', fontsize=24)

    # panel sizes      xleft%, ybot%, xright% ,ytop%
    map_panel_axes = [0.05, 0.15, 0.9, 0.7]
    colorbar_panel_axes = [0.15, 0.09, 0.7, .010]

    # parameters
    level = 0
    vmin = 10
    vmax = 100
    lat = 47.08729
    lon = 8.4941

    # panel 1, basemap, radar reflectivity and NARR overlay
    ax1 = fig.add_axes(map_panel_axes)

    #ax1 = display.plot_basemap(resolution='h',auto_range=False, min_lat=45.7, max_lat=47.9, min_lon=5.7, max_lon=10.6)
    display.plot_basemap(resolution='h',auto_range=False, min_lat=46.972, max_lat=47.0, min_lon=8.03839, max_lon=8.079)

    #display.plot_grid('maximum_expected_severe_hail_size', level=level, cmap='jet',vmin=vmin, vmax=vmax, title_flag=True,
    #                colorbar_flag=True, colorbar_label='MESHS [cm]', axislabels_flag=False)

    display.plot_grid('probability_of_hail', level=level, cmap='jet',vmin=vmin, vmax=vmax, title_flag=True,
                    colorbar_flag=True, colorbar_label='POH [%]', axislabels_flag=False)
    
    ax1.scatter(lon,lat,s=2000, color='black')

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plt.savefig(plot_path+'BZC'+str(i)+'.png', facecolor='white',bbox_inches='tight')