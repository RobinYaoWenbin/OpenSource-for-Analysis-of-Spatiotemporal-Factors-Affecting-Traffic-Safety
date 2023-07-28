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

class data_agg:
    def __init__(self):
        pass
    @staticmethod
    def merge(y_file,x_file_list,grid_data_file,merge_data_save_file):
        grid_df = gpd.read_file(grid_data_file)
        timeslot = list(range(0,24))
        tmp_df = pd.DataFrame({'ts':timeslot})
        grid_df = pd.merge(grid_df,tmp_df,how="cross")
        y_df = pd.read_csv(y_file,encoding='gbk')
        grid_df = pd.merge(grid_df , y_df , how='left' , on = ['LONCOL','LATCOL','ts'])
        for i in range(len(x_file_list)):
            tmp_file = x_file_list[i]
            tmp_df = pd.read_csv(tmp_file,encoding='gbk')
            grid_df = pd.merge(grid_df , tmp_df , how='left' , on = ['LONCOL','LATCOL','ts'])
        grid_df.fillna(0,inplace=True)
        grid_df.to_file(merge_data_save_file)
    @staticmethod
    def merge_staic_variable(x_file_list,grid_data_file,merge_data_save_file):
        df = gpd.read_file(grid_data_file)
        for i in range(len(x_file_list)):
            tmp_file = x_file_list[i]
            tmp_df = pd.read_csv(tmp_file,encoding='gbk')
            tmp_df.drop(columns=['Unnamed: 0'] , inplace=True)
            df = pd.merge(df , tmp_df , how='left' , on = ['LONCOL','LATCOL'])
        df.fillna(0,inplace=True)
        df.to_file(merge_data_save_file)


if __name__ == "__main__":
    # x_file_list = ["../../data/各时间片各栅格交通状态数据/avg_traffic_freq.csv"]
    # data_agg.merge(y_file="../../data/各时间片各栅格事故频率数据/inj_freq_ts_data.csv",
    #                x_file_list=x_file_list,
    #                grid_data_file="../../data/ResearchAreaGrid/grid/research_grid.shp",
    #                merge_data_save_file="../../data/merge_data/inj_data/inj_merge_data.shp")

    x_file_list = ["../../code/OSM道路设施/osm数据道路设施情况.csv","../../code/POI建成环境/建成环境.csv"]
    data_agg.merge_staic_variable(x_file_list=x_file_list,
                   grid_data_file="../../data/merge_data/inj_data/inj_merge_data.shp",
                   merge_data_save_file="../../data/merge_data/inj_data_all/inj_merge_data.shp")






