import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

class norm:
    def __init__(self):
        pass
    @staticmethod
    def start(data_file,output_file):
        df = gpd.read_file(data_file)
        ss = StandardScaler()
        data = ss.fit_transform(df[['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12',
                                'x13','x14','x15','x16','x17','x18']])
        tmp_df = pd.DataFrame(data,columns=['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12',
                                'x13','x14','x15','x16','x17','x18'])
        df = df[['LONCOL','LATCOL','ts','geometry']].join(tmp_df)
        df.to_file(output_file)

if __name__ == "__main__":
    norm.start(data_file="../../data/merge_data/pdo_data_rename/pdo_merge_data.shp",
               output_file="../../data/merge_data/pdo_data_norm/pdo_merge_data.shp")