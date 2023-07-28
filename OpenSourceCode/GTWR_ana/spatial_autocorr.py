import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# sys.path.append("../lib/")
from datetime import datetime, timezone, timedelta
import esda
import libpysal as lps
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

class autocorr:
    def __init__(self):
        pass
    @staticmethod
    def start(data_file,test_var):
        df = gpd.read_file(data_file)
        wq = lps.weights.Queen.from_dataframe(df)  # 使用Quuen式邻接矩阵
        wq.transform = 'r'  # 标准化矩阵
        y = df[test_var]
        print("正在检验的变量为:",test_var)
        mi = esda.moran.Moran(y, wq)
        print("Moran's I 值为：", mi.I)
        print("随机分布假设下Z检验值为：", mi.z_rand)
        print("随机分布假设下Z检验的P值为：", mi.p_rand)
        print("正态分布假设下Z检验值为：", mi.z_norm)
        print("正态分布假设下Z检验的P值为：", mi.p_norm)

if __name__ == "__main__":
    autocorr.start(data_file="../../data/merge_data/pdo_data_norm/pdo_merge_data.shp",
                   test_var='y')



