import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# sys.path.append("../lib/")
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

def conv284(ser):
    return gcj2wgs(ser[0], ser[1])

def get_ts(x):
    tmp_h = int(x[11:13])
    return tmp_h

class acci_freq:
    def __init__(self):
        pass
    @staticmethod
    def split_grid(dev_file,grid_data_file,grid_params_file):
        dev_df = pd.read_excel(dev_file)
        dev_df.dropna(how='all', inplace=True, axis=0)
        dev_df.dropna(how='all', inplace=True, axis=1)
        dev_df['lnglat84'] = dev_df[['citydogcj02lng', 'citydogcj02lat']].apply(conv284, axis=1)
        dev_df['lng84'] = dev_df[['lnglat84']].applymap(lambda x: x[0])
        dev_df['lat84'] = dev_df[['lnglat84']].applymap(lambda x: x[1])
        # Only the accident data of Xiaoshan District are taken
        poly_context = {'type': 'MULTIPOLYGON',
                        'coordinates': [[[[min(dev_df['lng84']), min(dev_df['lat84'])],
                                          [min(dev_df['lng84']), max(dev_df['lat84'])],
                                          [max(dev_df['lng84']), max(dev_df['lat84'])],
                                          [max(dev_df['lng84']), min(dev_df['lat84'])]]]]}
        poly_shape = shapely.geometry.asShape(poly_context)
        d = {'geometry': poly_shape}
        poly_shape = gpd.GeoDataFrame(d)
        poly_shape.plot()
        plt.show()
        grid_df , params = tbd.area_to_grid(location=poly_shape,accuracy=1000 , method='rect', params='auto')
        grid_df.to_file(filename=grid_data_file)
        np.save(grid_params_file, params)

    @staticmethod
    def get_freq(grid_data_file,grid_param_file,pdo_data_file,inj_data_file,pdo_freq_data_file,inj_freq_data_file):
        grid_df = gpd.read_file(grid_data_file)
        grid_params = np.load(grid_param_file, allow_pickle=True).item()
        df_pdo = pd.read_csv(pdo_data_file, encoding='gbk')
        df_inj = pd.read_csv(inj_data_file, encoding='gbk')
        lon_list , lat_list = tbd.GPS_to_grid(lon=df_pdo['actual_gps_x'] , lat=df_pdo['actual_gps_y'],params=grid_params)
        df_pdo['LONCOL'] = lon_list ; df_pdo['LATCOL'] = lat_list
        df_pdo['ts'] = df_pdo[['sgfssj']].applymap(get_ts)
        df_pdo_freq = df_pdo.groupby(['LONCOL','LATCOL','ts'])['ywsw'].count().reset_index().rename(columns={'ywsw':'acci_freq'})
        df_pdo_freq.to_csv(pdo_freq_data_file,index=False,encoding='gbk')
        lon_list, lat_list = tbd.GPS_to_grid(lon=df_inj['actual_gps_x'], lat=df_inj['actual_gps_y'], params=grid_params)
        df_inj['LONCOL'] = lon_list
        df_inj['LATCOL'] = lat_list
        df_inj['ts'] = df_inj[['sgfssj']].applymap(get_ts)
        df_inj_freq = df_inj.groupby(['LONCOL', 'LATCOL', 'ts'])['ywsw'].count().reset_index().rename(
            columns={'ywsw': 'acci_freq'})
        df_inj_freq.to_csv(inj_freq_data_file, index=False, encoding='gbk')

if __name__ == "__main__":
    # acci_freq.split_grid(dev_file="../../data/distinct_camera.xlsx",
    #                      grid_data_file="../../data/ResearchAreaGrid/grid/research_grid.shp",
    #                      grid_params_file="../../data/ResearchAreaGrid/grid_params.npy")

    acci_freq.get_freq(grid_data_file="../../data/ResearchAreaGrid/grid/research_grid.shp",
                       grid_param_file="../../data/ResearchAreaGrid/grid_params.npy",
                       pdo_data_file="../../data/事故数据概述分析/pdo_data.csv",
                       inj_data_file="../../data/事故数据概述分析/inj_data.csv",
                       pdo_freq_data_file="../../data/各时间片各栅格事故频率数据/pdo_freq_ts_data.csv",
                       inj_freq_data_file="../../data/各时间片各栅格事故频率数据/inj_freq_ts_data.csv")


