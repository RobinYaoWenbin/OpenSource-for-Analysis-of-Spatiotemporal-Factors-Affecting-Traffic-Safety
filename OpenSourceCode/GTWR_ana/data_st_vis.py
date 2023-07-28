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

class vis:
    def __init__(self):
        pass
    @staticmethod
    def temporal_vis(beta_file , column_name,data_save_file):
        df = gpd.read_file(beta_file)
        df_tem = df.groupby(['ts'])[column_name].mean().reset_index()
        df_tem.to_csv(data_save_file,index=False,encoding='gbk')
        fig, axs = plt.subplots(figsize=(10, 8), dpi=600, constrained_layout=True)
        tick_spacing = 3
        fontsize = 18
        linewidth = 3
        axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        axs.set_xlabel('Time of day(24 hours)', fontsize=fontsize)
        axs.set_ylabel('Coefficient', fontsize=fontsize)
        axs.plot(df_tem['ts'], df_tem[column_name], 'o-', linewidth=linewidth)
        plt.xticks(rotation=90, size=fontsize)
        plt.yticks(size=fontsize)
        plt.show()
    @staticmethod
    def spatial_vis(beta_file , column_name,dev_file , data_save_file,image_save_path):
        dev_df = pd.read_excel(dev_file)
        dev_df.dropna(how='all', inplace=True, axis=0)
        dev_df.dropna(how='all', inplace=True, axis=1)
        dev_df['lnglat84'] = dev_df[['citydogcj02lng', 'citydogcj02lat']].apply(conv284, axis=1)
        dev_df['lng84'] = dev_df[['lnglat84']].applymap(lambda x: x[0])
        dev_df['lat84'] = dev_df[['lnglat84']].applymap(lambda x: x[1])
        df = gpd.read_file(beta_file)
        df_spa = df.groupby(['LONCOL','LATCOL'])[column_name].mean().reset_index()
        df_spa = pd.merge(df[['LONCOL','LATCOL','geometry']] , df_spa , how='inner' , on=['LONCOL','LATCOL'])
        tbd.set_mapboxtoken(
            'give your ak here ')
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
        df_spa.plot(column=column_name, ax=ax, legend=True,alpha=0.05, cmap='OrRd',cax=cax)
        plt.show()
        df_spa.to_file(data_save_file)

if __name__ == "__main__":
    # vis.temporal_vis(beta_file="../../data/GTWR_result/pdo_betas/betas.shp",
    #                  column_name='avg_freq_b',
    #                  data_save_file="../../data/GTWR_result/pdo_result/avg_freq_temporal_vis.csv")

    vis.spatial_vis(beta_file="../../data/GTWR_result/pdo_betas/betas.shp",
                    column_name='avg_freq_b',
                    dev_file="../../data/distinct_camera.xlsx",
                    data_save_file="../../data/GTWR_result/pdo_result/avg_freq_spatial_vis/avg_freq_spatial_vis.shp",
                    image_save_path="../../data/GTWR_result/tile_file/")




