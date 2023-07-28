import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats.mstats import winsorize
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

def repair_traffic_freq(ser,grid_fill_df,ts_freq_df):
    tmp_df = pd.DataFrame([ser])
    tmp_df = pd.merge(tmp_df,grid_fill_df,how='inner',on=['LONCOL','LATCOL'])
    if len(tmp_df) == 0:
        return ser['avg_freq']
    else:
        return list(ts_freq_df[ ts_freq_df['ts'] == tmp_df['ts'][0]  ]['avg_freq'])[0]

class clearn:
    def __init__(self):
        pass
    @staticmethod
    def clearn_traffic_state(merge_data,data_save_file):
        df = gpd.read_file(merge_data)
        grid_fill_df = df[(df['avg_freq'] == 0) & (df['acci_freq'] != 0) ][['LONCOL','LATCOL']].drop_duplicates().reset_index(drop=True)
        ts_freq_df = df.groupby(['ts'])['avg_freq'].mean().reset_index()
        df['avg_freq'] = df.apply(repair_traffic_freq , args=(grid_fill_df,ts_freq_df,) , axis=1)
        df.to_file(data_save_file)
    @staticmethod
    def field_rename(merge_data,data_save_file):
        df = gpd.read_file(merge_data)
        df.rename(columns={'acci_freq':'y','avg_freq':'x1','in_degree':'x2','out_degree':'x3','node_betwe':'x4',
                           'pagerank':'x5','street_den':'x6','intersecti':'x7','circuity_a':'x8','edge_densi':'x9',
                           'orientatio':'x10','bike_stree':'x11','cr1':'x12','cr2':'x13','cr3':'x14','cr4':'x15',
                           'cr5':'x16','cr6':'x17','normalized':'x18'},inplace=True)
        df.to_file(data_save_file)
    @staticmethod
    def winsor_data(merge_data , explain_var_list , data_save_file):
        df = gpd.read_file(merge_data)
        for i in range(len(explain_var_list)):
            tmp_var = explain_var_list[i]
            df[tmp_var] = winsorize(df[tmp_var], limits=[0.1, 0.1])
        df.to_file(data_save_file)

if __name__ == "__main__":
    # clearn.clearn_traffic_state(merge_data="../../data/merge_data/pdo_data/pdo_merge_data.shp",
    #                             data_save_file="../../data/merge_data/pdo_data_clearn/pdo_merge_data.shp")
    # clearn.field_rename(merge_data="../../data/merge_data/pdo_data_clearn/pdo_merge_data.shp",
    #                     data_save_file="../../data/merge_data/pdo_data_rename/pdo_merge_data.shp")

    clearn.winsor_data(merge_data="../../data/merge_data/inj_data_rename/inj_merge_data.shp" ,
                       explain_var_list=['x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12',
                                         'x13','x14','x15','x16','x17','x18'] ,
                       data_save_file="../../data/merge_data/inj_data_rename/inj_merge_data.shp")




















