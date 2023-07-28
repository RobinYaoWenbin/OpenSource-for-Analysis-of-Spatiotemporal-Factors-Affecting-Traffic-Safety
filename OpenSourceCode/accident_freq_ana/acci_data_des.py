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

def getdate(x):
    return x[0:4] + x[5:7] + x[8:10]

def conv284(ser):
    return gcj2wgs(ser[0], ser[1])

class acci_data_intro:
    def __init__(self):
        pass
    @staticmethod
    def data_des(input_file,dev_file,save_freq_data_file,pdo_data_file,inj_data_file):
        df = pd.read_csv(input_file,encoding='utf-8',sep='#')
        df = df[['actual_gps_x','actual_gps_y','sgfssj','ywsw','injurystr']]
        df['date'] = df[['sgfssj']].applymap(getdate)
        df = df[(df['date']!='20211109')]
        df = df[(df['actual_gps_x']>118) & (df['actual_gps_x']<121) & (df['actual_gps_y']>29) & (df['actual_gps_y']<31)]
        print("有无伤亡字段显示无伤亡但injury字段显示有受伤的行数为{0}".format( len(df[ (df['ywsw']=='无伤亡')& (df['injurystr']=='有受伤')]) ))
        print("The total number of rows of accident data is {0}".format(len(df)))
        df.drop(columns=['injurystr'],inplace=True)
        df.dropna(inplace=True)
        print("The total number of rows of accident data after removing null values is {0}".format(len(df)))
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
        df = tbd.clean_outofshape(df, shape=poly_shape, col=['actual_gps_x', 'actual_gps_y'], accuracy=500)
        print("Accident data within the study area : {0} ".format(len(df)))
        # Draw the number of accidents per day
        fontsize = 18
        linewidth = 3
        freq_df = df.groupby(['date'])['ywsw'].count().reset_index().rename(columns={'ywsw':'all_freq'})
        fig, axs = plt.subplots(figsize=(10, 8), dpi=600, constrained_layout=True)
        tick_spacing = 7
        axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        axs.set_xlabel('Date',fontsize=fontsize)
        axs.set_ylabel('Accident frequency',fontsize=fontsize)
        axs.plot(freq_df['date'], freq_df['all_freq'], 'o-',linewidth = linewidth)
        plt.xticks(rotation=90,size=fontsize)
        plt.yticks(size=fontsize)
        plt.show()
        df_pdo = df[df['ywsw'] == '无伤亡']
        df_inj = df[df['ywsw'] == '有伤亡']
        print("The number of accidents with casualties is {0}, the number of accidents without casualties is {1}".format( len(df_inj) , len(df_pdo) ))
        freq_pdo_df = df_pdo.groupby(['date'])['ywsw'].count().reset_index().rename(columns={'ywsw': 'pdo_freq'})
        fig, axs = plt.subplots(figsize=(10, 8), dpi=600, constrained_layout=True)
        tick_spacing = 7
        axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        axs.set_xlabel('Date', fontsize=fontsize)
        axs.set_ylabel('Accident frequency', fontsize=fontsize)
        axs.plot(freq_pdo_df['date'], freq_pdo_df['pdo_freq'], 'o-',linewidth = linewidth)
        plt.xticks(rotation=90, size=fontsize)
        plt.yticks(size=fontsize)
        plt.show()
        freq_inj_df = df_inj.groupby(['date'])['ywsw'].count().reset_index().rename(columns={'ywsw': 'inj_freq'})
        fig, axs = plt.subplots(figsize=(10, 8), dpi=600, constrained_layout=True)
        tick_spacing = 7
        axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        axs.set_xlabel('Date', fontsize=fontsize)
        axs.set_ylabel('Accident frequency', fontsize=fontsize)
        axs.plot(freq_inj_df['date'], freq_inj_df['inj_freq'], 'o-',linewidth = linewidth)
        plt.xticks(rotation=90, size=fontsize)
        plt.yticks(size=fontsize)
        plt.show()
        df_final = pd.merge(freq_df ,freq_pdo_df ,how='inner',on='date' )
        df_final = pd.merge(df_final , freq_inj_df ,how='inner',on='date')
        df_final.to_csv(save_freq_data_file,index=False,encoding='gbk')
        df_pdo.to_csv(pdo_data_file,index=False,encoding='gbk')
        df_inj.to_csv(inj_data_file,index=False,encoding='gbk')

    @staticmethod
    def acci_dis(pdo_data_file,inj_data_file,image_save_path):
        df_pdo = pd.read_csv(pdo_data_file,encoding='gbk')
        df_inj = pd.read_csv(inj_data_file,encoding='gbk')
        df_all = df_pdo.append(df_inj,ignore_index=True)
        tbd.set_mapboxtoken(
            'please give the ak token here ')
        tbd.set_imgsavepath(image_save_path)
        bounds = [min(df_all['actual_gps_x']) - 0.01, min(df_all['actual_gps_y']) - 0.01, max(df_all['actual_gps_x']) + 0.01, max(df_all['actual_gps_y']) + 0.01]
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
        label_set = ['无伤亡','有伤亡']
        for i in range(len(label_set)):
            tmp = df_all[df_all['ywsw'] == label_set[i]]
            if label_set[i] == '有伤亡':
                tmp_label = "KSI"
            elif label_set[i] == '无伤亡':
                tmp_label = "PDO"
            plt.scatter(tmp['actual_gps_x'], tmp['actual_gps_y'], s=2 ,label = tmp_label)
        plt.legend(frameon=True,loc="lower left",fontsize='small')
        plt.show()
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
        plt.scatter(df_all['actual_gps_x'], df_all['actual_gps_y'], s=2)
        plt.show()

if __name__ == "__main__":
    # acci_data_intro.data_des(input_file="../../data/事故数据/dwd_c_apu_jq_mark1003_1109.txt",
    #                          dev_file="../../data/distinct_camera.xlsx",
    #                          save_freq_data_file="../../data/事故数据概述分析/各月事故频率数据.csv",
    #                          pdo_data_file="../../data/事故数据概述分析/pdo_data.csv",
    #                          inj_data_file="../../data/事故数据概述分析/inj_data.csv")

    acci_data_intro.acci_dis(pdo_data_file="../../data/事故数据概述分析/pdo_data.csv",
                             inj_data_file="../../data/事故数据概述分析/inj_data.csv",
                             image_save_path="../../data/事故数据概述分析/tile_image/")
