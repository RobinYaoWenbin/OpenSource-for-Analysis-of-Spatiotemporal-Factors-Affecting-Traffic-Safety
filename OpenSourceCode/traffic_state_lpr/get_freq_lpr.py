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

class get_traf_state:
    def __init__(self):
        pass
    @staticmethod
    def match_dev_grid(dev_file,grid_data_file,grid_param_file,dev_grid_match_file):
        dev_df = pd.read_excel(dev_file)
        dev_df.dropna(how='all', inplace=True, axis=0)
        dev_df.dropna(how='all', inplace=True, axis=1)
        dev_df['lnglat84'] = dev_df[['citydogcj02lng', 'citydogcj02lat']].apply(conv284, axis=1)
        dev_df['lng84'] = dev_df[['lnglat84']].applymap(lambda x: x[0])
        dev_df['lat84'] = dev_df[['lnglat84']].applymap(lambda x: x[1])
        grid_df = gpd.read_file(grid_data_file)
        grid_params = np.load(grid_param_file, allow_pickle=True).item()
        lon_list, lat_list = tbd.GPS_to_grid(lon=dev_df['lng84'], lat=dev_df['lat84'], params=grid_params)
        dev_df['LONCOL'] = lon_list
        dev_df['LATCOL'] = lat_list
        dev_df = dev_df[['camera_id','LONCOL','LATCOL']]
        dev_df.to_csv(dev_grid_match_file,index=False,encoding='gbk')

    @staticmethod
    def get_freq(dev_grid_match_file,lpr_data_file,volumn_file):
        dev_grid_match_df = pd.read_csv(dev_grid_match_file,encoding='gbk')
        df_lpr = pd.read_csv(lpr_data_file)
        df_lpr.drop(columns=['Unnamed: 0'], inplace=True)
        df_lpr.rename(columns={'0': 'car_num', '1': 'cap_date', '2': 'type', '3': 'dev_id', '4': 'dir', '5': 'road_id','6': 'turn_id'}, inplace=True)
        df_lpr = df_lpr[['car_num' , 'cap_date' ,  'dev_id']]
        df_lpr = pd.merge(df_lpr , dev_grid_match_df , how='inner' , left_on='dev_id',right_on='camera_id')
        df_lpr['ts'] = df_lpr[['cap_date']].applymap(get_ts)
        df_vol = df_lpr.groupby(['LONCOL', 'LATCOL', 'ts'])['car_num'].count().reset_index().rename(columns={'car_num': 'volumn'})
        df_vol.to_csv(volumn_file,index=False,encoding='gbk')

    @staticmethod
    def get_freq_auto(lpr_data_file_path , dev_grid_match_file , data_save_path):
        file_list = os.listdir(lpr_data_file_path)
        for i in range(len(file_list)):
            tmp_file = file_list[i]
            print("正在处理{0}".format(tmp_file))
            tmp_save_file = data_save_path+"volumn_data_" + tmp_file
            get_traf_state.get_freq(dev_grid_match_file=dev_grid_match_file,lpr_data_file=lpr_data_file_path+tmp_file,volumn_file=tmp_save_file)

    @staticmethod
    def get_avg_freq(daily_freq_file_path,avg_freq_file):
        file_list = os.listdir(daily_freq_file_path)
        df = pd.DataFrame([])
        for i in range(len(file_list)):
            tmp_file = file_list[i]
            print("正在处理{0}".format(tmp_file))
            tmp_df = pd.read_csv(daily_freq_file_path+tmp_file,encoding='gbk')
            if i == 0:
                df = tmp_df
                df.rename(columns={'volumn':0},inplace=True)
            else:
                df = pd.merge(df,tmp_df,how='outer',on=['LONCOL','LATCOL','ts'])
                df.rename(columns={'volumn': i}, inplace=True)
        df.fillna(0,inplace=True)
        df['avg_freq'] = df[[0,1,2,3,4]].mean(axis=1)
        df = df[['LONCOL','LATCOL','ts','avg_freq']]
        df.to_csv(avg_freq_file,index=False,encoding='gbk')

if __name__ == "__main__":
    # get_traf_state.match_dev_grid(dev_file="../../data/distinct_camera.xlsx",
    #                               grid_data_file="../../data/ResearchAreaGrid/grid/research_grid.shp",
    #                               grid_param_file="../../data/ResearchAreaGrid/grid_params.npy",
    #                               dev_grid_match_file="../../data/各时间片各栅格交通状态数据/dev_grid_match.csv")

    # get_traf_state.get_freq(dev_grid_match_file="../../data/各时间片各栅格交通状态数据/dev_grid_match.csv",
    #                         lpr_data_file="../../data/萧山市车牌识别数据/0318.csv",
    #                         volumn_file="../../data/各时间片各栅格交通状态数据/各天的状态数据/volumn_data_0319.csv")

    # get_traf_state.get_freq_auto(lpr_data_file_path="../../data/萧山市车牌识别数据/" ,
    #                              dev_grid_match_file="../../data/各时间片各栅格交通状态数据/dev_grid_match.csv",
    #                              data_save_path="../../data/各时间片各栅格交通状态数据/各天的状态数据/")

    get_traf_state.get_avg_freq(daily_freq_file_path="../../data/各时间片各栅格交通状态数据/各天的状态数据/",
                                avg_freq_file="../../data/各时间片各栅格交通状态数据/avg_traffic_freq.csv")