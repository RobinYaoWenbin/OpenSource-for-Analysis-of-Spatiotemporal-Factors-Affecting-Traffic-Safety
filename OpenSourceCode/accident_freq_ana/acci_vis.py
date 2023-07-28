import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../mgtwr/")
from mgtwr_local.sel import SearchGTWRParameter
from mgtwr_local.model import GTWR
from mgtwr_local.sel import SearchGWRParameter
from mgtwr_local.model import GWR
import statsmodels.api as sm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime, timezone, timedelta
from matplotlib.pyplot import MultipleLocator
import copy
import time, datetime
import logging
import time
import seaborn as sns
import math
from math import radians, cos, sin, asin, sqrt
import geopandas as gpd
from coord_convert.transform import wgs2gcj, wgs2bd, gcj2wgs, gcj2bd, bd2wgs, bd2gcj
from shapely.geometry import Point,Polygon,shape
import shapely.geometry
from random import sample
import random
# random.seed(0)
import transbigdata as tbd
from matplotlib import ticker
plt.rc('font',family='Times New Roman')
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 200)

def conv284(ser):
    return gcj2wgs(ser[0], ser[1])

class acci_vis:
    def __init__(self):
        pass
    @staticmethod
    def temporal_vis(acci_freq_file,grid_data_file,dev_file,data_save_file):
        df = pd.read_csv(acci_freq_file,encoding='gbk')
        df = df.groupby(['ts'])['acci_freq'].sum().reset_index()
        df.to_csv(data_save_file,index=False,encoding='gbk')
        fig, axs = plt.subplots(figsize=(10, 8), dpi=600, constrained_layout=True)
        tick_spacing = 3
        fontsize = 18
        linewidth = 3
        axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        axs.set_xlabel('Time of day(24 hours)', fontsize=fontsize)
        axs.set_ylabel('Accident frequency', fontsize=fontsize)
        axs.plot(df['ts'], df['acci_freq'], 'o-', linewidth=linewidth)
        plt.xticks(rotation=90, size=fontsize)
        plt.yticks(size=fontsize)
        plt.show()

    @staticmethod
    def spatial_vis(acci_freq_file,grid_data_file,dev_file,data_save_file,image_save_path):
        df = pd.read_csv(acci_freq_file, encoding='gbk')
        grid_df = gpd.read_file(grid_data_file)
        dev_df = pd.read_excel(dev_file)
        dev_df.dropna(how='all', inplace=True, axis=0)
        dev_df.dropna(how='all', inplace=True, axis=1)
        dev_df['lnglat84'] = dev_df[['citydogcj02lng', 'citydogcj02lat']].apply(conv284, axis=1)
        dev_df['lng84'] = dev_df[['lnglat84']].applymap(lambda x: x[0])
        dev_df['lat84'] = dev_df[['lnglat84']].applymap(lambda x: x[1])
        df = df.groupby(['LONCOL','LATCOL'])['acci_freq'].sum().reset_index()
        grid_df = pd.merge(grid_df , df , how='left' , on = ['LONCOL','LATCOL'])
        grid_df.fillna(0,inplace=True)
        tbd.set_mapboxtoken(
            'please give ak token here')
        tbd.set_imgsavepath(image_save_path)
        bounds = [min(dev_df['lng84']) - 0.01, min(dev_df['lat84']) - 0.01,
                  max(dev_df['lng84']) + 0.01, max(dev_df['lat84']) + 0.01]
        print(bounds)
        fig = plt.figure(1, (4, 8), dpi=300)
        ax = plt.subplot(111)
        plt.sca(ax)
        # Add map basemap
        tbd.plot_map(plt, bounds, zoom=13, style=4)
        # Add scale bar and north arrow
        tbd.plotscale(ax, bounds=bounds, textsize=1, compasssize=1, accuracy='auto', rect=[0.9, 0.9], zorder=4)
        plt.axis('off')
        plt.xlim(bounds[0], bounds[2])
        plt.ylim(bounds[1], bounds[3])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        grid_df.plot(column='acci_freq', ax=ax, legend=True, alpha=0.7, cmap='OrRd', cax=cax)
        plt.show()
        grid_df.to_file(data_save_file)

if __name__ == "__main__":
    # acci_vis.temporal_vis(acci_freq_file="../../data/各时间片各栅格事故频率数据/pdo_freq_ts_data.csv",
    #                       grid_data_file="../../data/ResearchAreaGrid/grid/research_grid.shp",
    #                       dev_file="../../data/distinct_camera.xlsx",
    #                       data_save_file="../../data/各时间片各栅格事故频率数据/pdo_ts_freq.csv")

    acci_vis.spatial_vis(acci_freq_file="../../data/各时间片各栅格事故频率数据/pdo_freq_ts_data.csv",
                          grid_data_file="../../data/ResearchAreaGrid/grid/research_grid.shp",
                          dev_file="../../data/distinct_camera.xlsx",
                          data_save_file="../../data/各时间片各栅格事故频率数据/pdo_dis/pdo_freq_dis.shp",
                         image_save_path="../../data/各时间片各栅格事故频率数据/")