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
from random import sample
import random
# random.seed(0)
plt.rc('font',family='Times New Roman')
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

def delquote(x):
    x = str(x)
    x = x.split("'")
    if len(x) == 1:
        return None
    else:
        return x[1]

class tablehead:
    def __init__(self):
        pass
    @staticmethod
    def add2file(header , input_file , output_file):
        head_names = []
        header_list = header.split(",")
        for i in range(len(header_list)):
            tmp_head = header_list[i]
            tmp_head = tmp_head.split("`")
            tmp_head = tmp_head[1]
            head_names.append(tmp_head)
        df = pd.read_csv(input_file,names=head_names,encoding='utf-8')
        header_names = list(df.columns)
        for i in range(len(header_names)):
            tmp_column = header_names[i]
            df[tmp_column] = df[[tmp_column]].applymap(delquote)
        df.to_csv(output_file,index=False,encoding='utf-8')

if __name__ == "__main__":
    header = '''
      give header
    '''
    tablehead.add2file(header=header,
                       input_file="E:\\study_e\\交通事故安全\\data\\事故数据\\convert_stg_c_sys_department_in_hz.csv",
                       output_file="E:\\study_e\\交通事故安全\\data\\事故数据\\final_stg_c_sys_department_in_hz.csv")
