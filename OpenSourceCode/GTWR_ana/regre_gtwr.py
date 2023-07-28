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

def summaryGTWR(Results):
    XNames = ["X" + str(i) for i in range(Results.k)]

    summary = "%s\n" % ('Spatiotemporal Weighted Regression (GTWR) Results')
    summary += '-' * 75 + '\n'

    if Results.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + Results.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + Results.kernel)

    summary += "%-62s %12.3f\n" % ('Model tau used:', Results.tau)

    summary += "%-62s %12.3f\n" % ('Spatial Bandwidth used:', Results.bw_s)

    summary += "%-62s %12.3f\n" % ('Temporal Bandwidth used:', Results.bw_t)

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'

    if Results.kernel == 'gaussian':

        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', Results.RSS)
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Results.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Results.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(Results.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Results.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Results.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Results.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Results.bic)
        summary += "%-62s %12.3f\n" % ('R2:', Results.R2)
    else:
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Results.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Results.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Results.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Results.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Results.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Results.bic)
        # summary += "%-60s %12.6f\n" % ('Percent deviance explained:', 0)

    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', Results.adj_alpha[1])
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):', Results.critical_tval(Results.adj_alpha[1]))

    summary += "\n%s\n" % ('Summary Statistics For GTWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD', 'Min', 'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    for i in range(Results.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
        XNames[i], np.mean(Results.betas[:, i]), np.std(Results.betas[:, i]), np.min(Results.betas[:, i]),
        np.median(Results.betas[:, i]), np.max(Results.betas[:, i]))

    summary += '=' * 75 + '\n'

    return summary

def summaryGWR(Results):
    XNames = ["X" + str(i) for i in range(Results.k)]

    summary = "%s\n" % ('Geographically weighted regression (GWR) Results')
    summary += '-' * 75 + '\n'

    if Results.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + Results.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + Results.kernel)

    summary += "%-62s %12.3f\n" % ('Temporal Bandwidth used:', Results.bw)

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'

    if Results.kernel == 'gaussian':

        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', Results.RSS)
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Results.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Results.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(Results.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Results.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Results.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Results.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Results.bic)
        summary += "%-62s %12.3f\n" % ('R2:', Results.R2)
    else:
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Results.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Results.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Results.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Results.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Results.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Results.bic)
        # summary += "%-60s %12.6f\n" % ('Percent deviance explained:', 0)

    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', Results.adj_alpha[1])
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):', Results.critical_tval(Results.adj_alpha[1]))

    summary += "\n%s\n" % ('Summary Statistics For GWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD', 'Min', 'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    for i in range(Results.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
        XNames[i], np.mean(Results.betas[:, i]), np.std(Results.betas[:, i]), np.min(Results.betas[:, i]),
        np.median(Results.betas[:, i]), np.max(Results.betas[:, i]))

    summary += '=' * 75 + '\n'

    return summary

class regression:
    def __init__(self):
        pass
    @staticmethod
    def cal_VIF(data_file,explain_var_list):
        df = gpd.read_file(data_file)
        coords = df[['LONCOL', 'LATCOL']].values
        t = df[['ts']].values
        y = df[['y']].values
        X = df[explain_var_list]
        X = sm.add_constant(X, prepend=True, has_constant='skip')
        print( pd.Series( [variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns) )

    @staticmethod
    def gtwr_start(data_file,explain_var_list,result_data_save_file):
        df = gpd.read_file(data_file)
        coords = df[['LONCOL','LATCOL']].values
        t = df[['ts']].values
        y = df[['y']].values
        X = df[explain_var_list].values
        sel = SearchGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
        bw, tau = sel.search(tau_max=20, verbose=True, time_cost=True)
        gtwr = GTWR(coords, t, X, y, bw, tau, kernel='gaussian', fixed=True).fit()
        print( summaryGTWR(gtwr) )
        explain_var_list.insert(0,'const')
        for i in range(len(explain_var_list)):
            explain_var_list[i] = explain_var_list[i]+'_betas'
        gtwr_result_df = pd.DataFrame(gtwr.betas, columns=explain_var_list)
        df = df.join(gtwr_result_df)
        print(df.head(2))
        df.to_file(result_data_save_file)

    @staticmethod
    def gwr_start(data_file,explain_var_list):
        df = gpd.read_file(data_file)
        coords = df[['LONCOL', 'LATCOL']].values
        t = df[['ts']].values
        y = df[['y']].values
        X = df[explain_var_list].values
        sel = SearchGWRParameter(coords, X, y, kernel='gaussian', fixed=True)
        bw = sel.search(bw_max=40, verbose=True, time_cost=True)
        gwr = GWR(coords, X, y, bw, kernel='gaussian', fixed=True).fit()
        print( summaryGWR(gwr) )

    @staticmethod
    def ols_start(data_file,explain_var_list):
        df = gpd.read_file(data_file)
        y = df[['y']].values
        X = df[explain_var_list].values
        X = sm.add_constant(X, prepend=True, has_constant='skip')
        model = sm.OLS(y, X)
        result = model.fit()
        print(result.summary())

if __name__ == "__main__":
    regression.gtwr_start(data_file="../../data/merge_data/pdo_data_clearn/pdo_merge_data.shp",
                          explain_var_list=['avg_freq'],
                          result_data_save_file="../../data/GTWR_result/betas/betas.shp")
    # regression.gwr_start(data_file="../../data/merge_data/pdo_data_clearn/pdo_merge_data.shp",
    #                      explain_var_list=['avg_freq'])
    # regression.ols_start(data_file="../../data/merge_data/pdo_data_clearn/pdo_merge_data.shp",
    #                      explain_var_list=['avg_freq'])
    # regression.cal_VIF(data_file="../../data/merge_data/pdo_data_clearn/pdo_merge_data.shp",
    #                    explain_var_list=['avg_freq'])



