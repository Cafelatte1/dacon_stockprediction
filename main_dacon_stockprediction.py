import os
import multiprocessing
import copy
import pickle
import warnings
from datetime import datetime, timedelta
from time import time, sleep, mktime
from matplotlib import font_manager as fm, rc, rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import random

import numpy as np
from numpy import array, nan, random as rnd, where
import pandas as pd
from pandas import DataFrame as dataframe, Series as series, isna, read_csv
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
from scipy.stats import f_oneway

from sklearn import preprocessing as prep
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split as tts, GridSearchCV as GridTuner, StratifiedKFold, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from sklearn.pipeline import make_pipeline

from sklearn import linear_model as lm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn import neighbors as knn
from sklearn import ensemble

# ===== tensorflow =====
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import metrics as tf_metrics
from tensorflow.keras import callbacks as tf_callbacks
from tqdm.keras import TqdmCallback
import tensorflow_addons as tfa
import keras_tuner as kt
from keras_tuner import HyperModel

# # ===== timeseries =====
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from sklearn.model_selection import TimeSeriesSplit
# from tensorflow.keras.preprocessing import timeseries_dataset_from_array as make_ts_tensor

# # ===== NLP =====
# from selenium import webdriver
# from konlpy.tag import Okt
# from KnuSentiLex.knusl import KnuSL

# ===== import functions =====
import sys
sys.path.append("projects/DA_Platform")
from DA_v4 import *

# global setting
warnings.filterwarnings(action='ignore')
rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

font_path = 'myfonts/NanumSquareB.ttf'
font_obj = fm.FontProperties(fname=font_path, size=12).get_name()
rc('font', family=font_obj)

folder_path = "projects/dacon_stockprediction/"
seed_everything()

# ===== task specific functions =====
from pykrx import stock
import yfinance as yf
def getBreakthroughPoint(df, col1, col2, patient_days, fill_method="fb"):
    '''
    :param df: dataframe (including col1, col2)
    :param col1: obj
    :param col2: obj moving average
    :param patient_days: patient days detected as breakthrough point
    :return: signal series
    '''
    sigPrice = []
    flag = -1  # A flag for the trend upward/downward

    for i in range(0, len(df)):
        if df[col1][i] > df[col2][i] and flag != 1:
            tmp = df['Close'][i:(i + patient_days + 1)]
            if len(tmp) == 1:
                sigPrice.append("buy")
                flag = 1
            else:
                if (tmp.iloc[1:] > tmp.iloc[0]).all():
                    sigPrice.append("buy")
                    flag = 1
                else:
                    sigPrice.append(nan)
        elif df[col1][i] < df[col2][i] and flag != 0:
            tmp = df['Close'][i:(i + patient_days + 1)]
            if len(tmp) == 1:
                sigPrice.append("sell")
                flag = 0
            else:
                if (tmp.iloc[1:] < tmp.iloc[0]).all():
                    sigPrice.append("sell")
                    flag = 0
                else:
                    sigPrice.append(nan)
        else:
            sigPrice.append(nan)

    sigPrice = series(sigPrice)
    for idx, value in enumerate(sigPrice):
        if not isna(value):
            if value == "buy":
                sigPrice.iloc[1:idx] = "sell"
            else:
                sigPrice.iloc[1:idx] = "buy"
            break
    # if fill_method == "bf":
    #
    # elif fill_method == ""
    sigPrice.ffill(inplace=True)
    return sigPrice
def stochastic(df, n=14, m=5, t=5):
    #데이터 프레임으로 받아오기 때문에 불필요

    #n 일중 최저가
    ndays_high = df['High'].rolling(window=n, min_periods=n).max()
    ndays_low = df['Low'].rolling(window=n, min_periods=n).min()
    fast_k = ((df['Close'] - ndays_low) / (ndays_high - ndays_low) * 100)
    slow_k = fast_k.ewm(span=m, min_periods=m).mean()
    slow_d = slow_k.ewm(span=t, min_periods=t).mean()
    df = df.assign(fast_k=fast_k, fast_d=slow_k, slow_k=slow_k, slow_d=slow_d)
    return df

# ===== raw data loading =====
# Get Stock List
# 종목 코드 로드
start_date = '20210104'
end_date = '20211029'
stock_list = read_csv(folder_path + "stock_list.csv")
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime('%V')
business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

# 모든 종목
stock_list.set_index("종목명", inplace=True)
selected_codes = stock_list.index.tolist()
stock_list = stock_list.loc[selected_codes]["종목코드"]
stock_dic = dict.fromkeys(selected_codes)

# # original data loading
# for stock_name, stock_code in tqdm(stock_list.items()):
#     print("=====", stock_name, "=====")
#
#     # 종목 주가 데이터 로드
#     try:
#         stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
#         stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
#         investor_df = stock.get_market_trading_volume_by_date(start_date, end_date, stock_code)[["기관합계", "외국인합계"]].reset_index()
#         kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()
#         stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
#         investor_df.columns = ["Date", "inst", "fore"]
#         kospi_df.columns = ["Date", "kospi"]
#         # 영업일과 주가 정보를 outer 조인
#         train_x = pd.merge(business_days, stock_df, how='left', on="Date")
#         train_x = pd.merge(train_x, investor_df, how='left', on="Date")
#         train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
#         # 앞의 일자로 nan값 forward fill
#         train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)
#         # 첫 날이 na 일 가능성이 있으므로 backward fill 수행
#         train_x.iloc[:, 1:] = train_x.iloc[:, 1:].bfill(axis=0)
#     except:
#         stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
#         stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
#         kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()
#         stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
#         kospi_df.columns = ["Date", "kospi"]
#         # 영업일과 주가 정보를 outer 조인
#         train_x = pd.merge(business_days, stock_df, how='left', on="Date")
#         train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
#         # 종가데이터에 생긴 na 값을 선형보간 및 정수로 반올림
#         # 앞의 일자로 nan값 forward fill
#         train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)
#         # 첫 날이 na 일 가능성이 있으므로 backward fill 수행
#         train_x.iloc[:, 1:] = train_x.iloc[:, 1:].bfill(axis=0)
#
#     stock_dic[stock_name]["df"] = train_x.copy()

# easyIO(stock_dic, folder_path + "dataset/stock_df_ori_" + start_date + "_" + end_date + ".pickle", op="w")

# pension_fund_dic = dict.fromkeys(stock_dic.keys())
# for i in pension_fund_dic.keys(): pension_fund_dic[i] = []
#
# for i in tqdm(stock_dic["삼성전자"]["df"]["Date"].dt.strftime("%Y%m%d").values):
#     # i = stock_dic["삼성전자"]["df"]["Date"].dt.strftime("%Y%m%d").values[0]
#     try:
#         pension_fund_df = stock.get_market_net_purchases_of_equities_by_ticker(i, i, "KOSPI", "연기금")[["종목명", "순매수거래대금"]]
#         for j in stock_dic.keys():
#             # break
#             tmp_value = pension_fund_df["순매수거래대금"][pension_fund_df["종목명"] == j].values
#             pension_fund_dic[j].append(0 if len(tmp_value) == 0 else tmp_value[0])
#     except:
#         for j in stock_dic.keys():
#             pension_fund_dic[j].append(nan)
#
# easyIO(pension_fund_dic, folder_path + "dataset/pension_fund_dic_" + start_date + "_" + end_date + ".pickle", op="w")
#
# print(len(stock_dic.keys()) == len(stock_list))
#
# stock_dic["삼성전자"]["df"]
# stock_dic["삼성전자"]["target_list"][0]

stock_dic = easyIO(None, folder_path + "dataset/stock_df_ori_" + start_date + "_" + end_date + ".pickle", op="r")
pension_fund_dic = easyIO(None, folder_path + "dataset/pension_fund_dic_" + start_date + "_" + end_date + ".pickle", op="r")
forex_index_data = yf.download(["USDKRW=X", "USDAUD=X", "USDJPY=X", "EURUSD=X", "CNY=X", "^GSPC", "^DJI", "^IXIC", "^STOXX50E",
                                "^SOX",  "000001.SS", "000300.SS", "MME=F", "^TNX"],
                               start='2020-12-04', end='2021-11-29', rounding=True)
tmp_forex_index = forex_index_data["Close"]
tmp_forex_index.index = pd.to_datetime(tmp_forex_index.index)
tmp_forex_index = tmp_forex_index[(tmp_forex_index.index >= pd.to_datetime(start_date)) & (tmp_forex_index.index <= pd.to_datetime(end_date))]
tmp_forex_index.columns = ["sse_composite_index", "csi300_index", "usdtocny", "eurtousd", "msci_emerging", "usdtoaud", "usdtojpy", "usdtokrw",
                           "dow", "snp500", "nasdaq", "semicon_index", "euro50", "us10y_tsy"]
tmp_forex_index.reset_index(drop=False, inplace=True)

# print(stock_dic["삼성전자"]["df"])
# print(stock_dic["KODEX 200"]["df"])

# ===== feature engineering =====
non_stock = []
corr_list = []
timeunit_gap_forviz = 1
metric_days = 14
cat_vars = []
bin_vars = []
cat_vars.append("weekday")
cat_vars.append("weeknum")
bin_vars.append("mfi_signal")
num_pvalue_check = None
cat_pvalue_check = series(0, index=["weekday", "weeknum", "mfi_signal"])

# # 연기금 순매수가 모두 0인 기업 count
# pension_fund_cnt = 0

idx_assign_flag = True
for stock_name, stock_data in stock_dic.items():
    train_x = stock_data["df"].copy()

    # 환율 및 관련 인덱스 feature 추가
    train_x = pd.merge(train_x, tmp_forex_index, how="left", on="Date")
    # 과거 일자로 forward fill 수행
    train_x = train_x.ffill()
    # 첫 날이 nan 일 가능성이 있으므로 backward fill 수행
    train_x = train_x.bfill()

    # # 연기금 순매수 feature 추가
    # train_x["pension_fund"] = pension_fund_dic[stock_name]
    # # 과거 일자로 forward fill 수행
    # train_x["pension_fund"] = train_x["pension_fund"].ffill()
    # # 첫 날이 nan 일 가능성이 있으므로 backward fill 수행
    # train_x["pension_fund"] = train_x["pension_fund"].bfill()
    # # 연기금 순매수가 없으면 해당 feature 드랍
    # if train_x["pension_fund"].sum() == 0:
    #     train_x.drop("pension_fund", axis=1, inplace=True)
    #     pension_fund_cnt += 1
    # # 연기금 순매수가 0 초과면 연속 순매수 일자 feature 생성
    # else:
    #     cnt_consecutive = 0
    #     tmp_consecutive = []
    #     for i in train_x["pension_fund"]:
    #         if i > 0: cnt_consecutive += 1
    #         else: cnt_consecutive = 0
    #         tmp_consecutive.append(cnt_consecutive)
    #     train_x["consec_pension_fund"] = tmp_consecutive

    # 요일 및 주차 파생변수 추가
    train_x['weekday'] = train_x["Date"].apply(lambda x: x.weekday())
    train_x['weeknum'] = train_x["Date"].apply(lambda x: week_of_month(x))

    # 거래대금 파생변수 추가
    train_x['trading_amount'] = train_x["Close"] * train_x["Volume"]

    # 월별 주기성 특징을 잡기 위한 sin 및 cos 변환 파생변수 추가
    day_to_sec = 24 * 60 * 60
    month_to_sec = 20 * day_to_sec
    timestamp_s = train_x["Date"].apply(datetime.timestamp)
    timestamp_freq = round((timestamp_s / month_to_sec).diff(20)[20], 1)

    train_x['dayofmonth_freq_sin'] = np.sin((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))
    train_x['dayofmonth_freq_cos'] = np.cos((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))

    # 1. OBV 파생변수 추가
    # 매수 신호: obv > obv_ema
    # 매도 신호: obv < obv_ema
    obv = [0]
    for i in range(1, len(train_x.Close)):
        if train_x.Close[i] >= train_x.Close[i - 1]:
            obv.append(obv[-1] + train_x.Volume[i])
        elif train_x.Close[i] < train_x.Close[i - 1]:
            obv.append(obv[-1] - train_x.Volume[i])
    train_x['obv'] = obv
    train_x['obv'][0] = nan
    train_x['obv_ema'] = train_x['obv'].ewm(com=metric_days, min_periods=metric_days).mean()

    # Stochastic 파생변수 추가
    # fast_d = moving average on fast_k
    train_x[["fast_k", "fast_d"]] = stochastic(train_x, n=metric_days)[["fast_k", "fast_d"]]

    # 2. MFI 파생변수 추가
    # MFI = 100 - (100 / 1 + MFR)
    # MFR = 14일간의 양의 MF / 14일간의 음의 MF
    # MF = 거래량 * (당일고가 + 당일저가 + 당일종가) / 3
    # MF 컬럼 만들기
    train_x["mf"] = train_x["Volume"] * ((train_x["High"]+train_x["Low"]+train_x["Close"]) / 3)
    # 양의 MF와 음의 MF 표기 컬럼 만들기
    p_n = []
    for i in range(len(train_x['mf'])):
        if i == 0 :
            p_n.append(nan)
        else:
            if train_x['mf'][i] >= train_x['mf'][i-1]:
                p_n.append('p')
            else:
                p_n.append('n')
    train_x['p_n'] = p_n
    # 14일간 양의 MF/ 14일간 음의 MF 계산하여 컬럼 만들기
    mfr = []
    for i in range(len(train_x['mf'])):
        if i < metric_days-1:
            mfr.append(nan)
        else:
            train_x_=train_x.iloc[(i-metric_days+1):i]
            a = (sum(train_x_['mf'][train_x['p_n'] == 'p']) + 1) / (sum(train_x_['mf'][train_x['p_n'] == 'n']) + 10)
            mfr.append(a)
    train_x['mfr'] = mfr
    # 최종 MFI 컬럼 만들기
    train_x['mfi'] = 100 - (100 / (1 + train_x['mfr']))
    train_x["mfi_signal"] = train_x['mfi'].apply(lambda x: "buy" if x > 50 else "sell")

    # 이동평균 추가
    train_x["close_mv5"] = train_x["Close"].rolling(5, min_periods=5).mean()
    train_x["close_mv10"] = train_x["Close"].rolling(10, min_periods=10).mean()
    train_x["close_mv20"] = train_x["Close"].rolling(20, min_periods=20).mean()

    train_x["volume_mv5"] = train_x["Volume"].rolling(5, min_periods=5).mean()
    train_x["volume_mv10"] = train_x["Volume"].rolling(10, min_periods=10).mean()
    train_x["volume_mv20"] = train_x["Volume"].rolling(20, min_periods=20).mean()

    train_x["trading_amount_mv5"] = train_x["trading_amount"].rolling(5, min_periods=5).mean()
    train_x["trading_amount_mv10"] = train_x["trading_amount"].rolling(10, min_periods=10).mean()
    train_x["trading_amount_mv20"] = train_x["trading_amount"].rolling(20, min_periods=20).mean()

    train_x["kospi_mv5"] = train_x["kospi"].rolling(5, min_periods=5).mean()
    train_x["kospi_mv10"] = train_x["kospi"].rolling(10, min_periods=10).mean()
    train_x["kospi_mv20"] = train_x["kospi"].rolling(20, min_periods=20).mean()

    try:
        train_x["inst_mv5"] = train_x["inst"].rolling(5, min_periods=5).mean()
        train_x["inst_mv10"] = train_x["inst"].rolling(10, min_periods=10).mean()
        train_x["inst_mv20"] = train_x["inst"].rolling(20, min_periods=20).mean()

        # 기관 연속 순매수 일자 feature 생성
        cnt_consecutive = 0
        tmp_consecutive = []
        for i in train_x["inst"]:
            if i > 0:
                cnt_consecutive += 1
            else:
                cnt_consecutive = 0
            tmp_consecutive.append(cnt_consecutive)
        train_x["consec_inst"] = tmp_consecutive

        train_x["fore_mv5"] = train_x["fore"].rolling(5, min_periods=5).mean()
        train_x["fore_mv10"] = train_x["fore"].rolling(10, min_periods=10).mean()
        train_x["fore_mv20"] = train_x["fore"].rolling(20, min_periods=20).mean()

        # 외국인 연속 순매수 일자 feature 생성
        cnt_consecutive = 0
        tmp_consecutive = []
        for i in train_x["fore"]:
            if i > 0:
                cnt_consecutive += 1
            else:
                cnt_consecutive = 0
            tmp_consecutive.append(cnt_consecutive)
        train_x["consec_fore"] = tmp_consecutive
    except:
        pass

    # 과거데이터 추가
    tmp_df = dataframe()
    tmp_cols = []
    for i in range(1,6,1):
        tmp_df = pd.concat([tmp_df, train_x["Close"].shift(i).to_frame()], axis=1)
        tmp_cols.append("close_" + str(i) + "shift")
    tmp_df.columns = tmp_cols
    train_x = pd.concat([train_x, tmp_df], axis=1)

    # 지표계산을 위해 쓰인 컬럼 drop
    train_x.drop(["mf", "p_n", "mfr", "Open", "High", "Low"], inplace=True, axis=1)

    # 컬럼이름 소문자 변환 및 정렬
    train_x.columns = train_x.columns.str.lower()
    train_x = pd.concat([train_x[["date"]], train_x.iloc[:,1:].sort_index(axis=1)], axis=1)

    # create target list
    target_list = []
    target_list.append(train_x["close"].copy())
    target_list.append(train_x["close"].shift(-1))
    target_list.append(train_x["close"].shift(-2))
    target_list.append(train_x["close"].shift(-3))
    target_list.append(train_x["close"].shift(-4))
    target_list.append(train_x["close"].shift(-5))
    for idx, value in enumerate(target_list):
        value.name = "target_shift" + str(idx)

    # if "inst" not in train_x.columns:
    #     pass
    # else:
    #     # <visualization>
    #     # 시각화용 데이터프레임 생성
    #     train_bi = pd.concat([target_list[timeunit_gap_forviz], train_x], axis=1)[:-timeunit_gap_forviz]
    #
    #     # 기업 평균 상관관계를 측정하기 위한 연산
    #     corr_obj = train_bi.drop(bin_vars + cat_vars, axis=1).corr().round(3)
    #     corr_rows = corr_obj.index.tolist()
    #     corr_cols = corr_obj.columns.tolist()
    #     corr_list.append(corr_obj.to_numpy().round(3)[..., np.newaxis])
    #
    #     # # 상관관계 시각화
    #     # fig, ax = plt.subplots(figsize=(16, 9))
    #     # graph = sns.heatmap(corr_obj, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize": 8, "fontweight": "bold"})
    #     # plt.xticks(rotation=45)
    #     # fig.subplots_adjust(left=0.15, bottom=0.2)
    #     # ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
    #     # ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
    #     # plt.subplots_adjust(bottom=0.17, right=1)
    #     # plt.title('Correlation Visualization on ' + stock_name, fontsize=15, fontweight="bold", pad=15)
    #     # createFolder("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/")
    #     # plt.savefig("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/" + stock_name + ".png", dpi=300)
    #     # plt.close()
    #
    #     # # feature 와 target 간 시각화
    #     # # ===== scatter plot on numerical feature =====
    #     # for i in train_x.columns:
    #     #     if i == "date":
    #     #         pass
    #     #     elif i in cat_vars + bin_vars:
    #     #         fig, ax = plt.subplots(figsize=(12, 6))
    #     #         graph = sns.boxplot(x=train_bi[i], y=train_bi["target_shift" + str(timeunit_gap_forviz)], palette=sns.hls_palette())
    #     #         change_width(ax, 0.2)
    #     #         graph.set_title(i + " on " + stock_name, fontsize=15, fontweight="bold", pad=15)
    #     #         plt.show()
    #     #         createFolder('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name)
    #     #         plt.savefig('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name + "/" + i + ".png", dpi=300)
    #     #         plt.close()
    #     #     else:
    #     #         fig, ax = plt.subplots(figsize=(12, 6))
    #     #         graph = sns.regplot(x=train_bi[i], y=train_bi["target_shift" + str(timeunit_gap_forviz)], color="green",
    #     #                             scatter_kws={'s': 15}, line_kws={"color": "orange"})
    #     #         graph.set_title(i + " on " + stock_name, fontsize=15, fontweight="bold", pad=15)
    #     #         plt.show()
    #     #         createFolder('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name)
    #     #         plt.savefig('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name + "/" + i +".png", dpi=300)
    #     #         plt.close()
    #
    #     ols_x = train_x.copy()
    #     ols_x = ols_x[:-timeunit_gap_forviz]
    #     ols_y = target_list[timeunit_gap_forviz][:-timeunit_gap_forviz]
    #
    #     ols_y = ols_y[ols_x.isna().any(axis=1) == False]
    #     ols_x = ols_x[ols_x.isna().any(axis=1) == False]
    #     ols_x = ols_x.drop(["date"] + cat_vars + bin_vars, axis=1)
    #     model_ols = sm.OLS(ols_y.to_frame(), sm.add_constant(ols_x, prepend=False))
    #     model_ols = model_ols.fit()
    #
    #     if idx_assign_flag:
    #         num_pvalue_check = series([1 if i <= 0.05 else 0 for i in model_ols.pvalues.drop("const")],
    #                                   index=model_ols.pvalues.drop("const").index) / len(stock_list)
    #         idx_assign_flag = False
    #     else:
    #         num_pvalue_check += series([1 if i <= 0.05 else 0 for i in model_ols.pvalues.drop("const")],
    #                                    index=model_ols.pvalues.drop("const").index) / len(stock_list)
    #
    #     # categorical, binary 변수에 대한 분산분석 (target 과의 상관관계 파악)
    #     # 귀무가설(H0) : 두 변수는 상관관계가 없다
    #     # 대립가설(H1) : 두 변수는 상관관계가 있다
    #
    #     pvalue_check_cat = train_bi.groupby("weekday")["target_shift" + str(timeunit_gap_forviz)].apply(list)
    #     cat_pvalue_check["weekday"] += 1 / len(stock_list) if f_oneway(*pvalue_check_cat)[1] <= 0.05 else 0
    #     pvalue_check_cat = train_bi.groupby("weeknum")["target_shift" + str(timeunit_gap_forviz)].apply(list)
    #     cat_pvalue_check["weeknum"] += 1 / len(stock_list) if f_oneway(*pvalue_check_cat)[1] <= 0.05 else 0
    #     pvalue_check_cat = train_bi.groupby("mfi_signal")["target_shift" + str(timeunit_gap_forviz)].apply(list)
    #     cat_pvalue_check["mfi_signal"] += 1 / len(stock_list) if f_oneway(*pvalue_check_cat)[1] <= 0.05 else 0

    # onehot encoding
    onehot_encoder = MyOneHotEncoder()
    train_x = onehot_encoder.fit_transform(train_x, cat_vars + bin_vars)

    stock_dic[stock_name]["df"] = train_x.copy()
    stock_dic[stock_name]["target_list"] = target_list

print(stock_dic["삼성전자"]["df"])
print(stock_dic["삼성전자"]["target_list"])

# # print(pension_fund_cnt)
#
# # numerical vars pvalue test
# print(num_pvalue_check)
# num_pvalue_check.sort_values(ascending=False)
# num_pvalue_check.to_csv(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/num_features.csv")
# # categorical vars F-test
# print(cat_pvalue_check)
# cat_pvalue_check.to_csv(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/cat_features.csv")
#
# # data check
# print(stock_dic["삼성전자"]["df"])
#
# # # 상관관계 평균 시각화
# corr_mean = dataframe(np.concatenate(corr_list, axis=2).mean(axis=2), index=corr_rows, columns=corr_cols).round(3)
# corr_std = dataframe(np.concatenate(corr_list, axis=2).std(axis=2), index=corr_rows, columns=corr_cols).round(3)
# print(corr_mean)
# print(corr_std)
#
# fig, ax = plt.subplots(figsize=(16, 9))
# graph = sns.heatmap(corr_mean, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize":8, "fontweight": "bold"})
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
# ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
# plt.title('Mean Correlation Visualization', fontsize=15, fontweight="bold", pad=15)
# plt.subplots_adjust(bottom=0.17, right=1)
# corr_mean.to_csv(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/상관계수_평균.csv")
# plt.savefig(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/상관계수_평균.png", dpi=300)
# plt.close()
#
# fig, ax = plt.subplots(figsize=(16, 9))
# sns.heatmap(corr_std, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize":8, "fontweight": "bold"})
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
# ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
# plt.title('Std. Correlation Visualization', fontsize=15, fontweight="bold", pad=15)
# plt.subplots_adjust(bottom=0.17, right=1)
# corr_std.to_csv(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/상관계수_표준편차.csv")
# plt.savefig(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/상관계수_표준편차.png", dpi=300)
# plt.close()
#
# fig, ax = plt.subplots(figsize=(16, 9))
# mean_to_std = (corr_mean/corr_std).round(3)
# sns.heatmap(mean_to_std, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize":8, "fontweight": "bold"})
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
# ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
# plt.title('Mean divided by Std. Correlation Visualization', fontsize=15, fontweight="bold", pad=15)
# plt.subplots_adjust(bottom=0.17, right=1)
# mean_to_std.to_csv(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/상관계수_표준편차대비평균.csv")
# plt.savefig(folder_path + "result/timegap_" + str(timeunit_gap_forviz) + "/상관계수_표준편차대비평균.png", dpi=300)
# plt.close()

# <feature selection and feature scaling>
# selected_features = None
cat_vars_oh = ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4",
               "weeknum_1", "weeknum_2", "weeknum_3", "weeknum_4", "weeknum_5"]
bin_vars_oh = ["mfi_signal_buy", "mfi_signal_sell"]
forex_index_vars = ["sse_composite_index", "csi300_index", "usdtocny", "eurtousd", "msci_emerging",
                    "usdtoaud", "usdtojpy", "usdtokrw", "dow", "snp500", "nasdaq", "semicon_index", "euro50", "us10y_tsy"]
forex_index_vars = series(forex_index_vars).str.lower().to_list()
print(forex_index_vars)

# feature_seed = 1
# selected_features = None
# logtrans_vec = []

# # corr mean >= 0.2
# feature_seed = 2
# selected_features = ["date", "close", "close_1shift", "close_2shift", "close_3shift", "close_4shift",
#                      "close_mv5", "close_mv10", "close_mv20", "fast_d", "fast_k", "inst_mv20",
#                      "kospi", "kopsi_mv5", "kospi_mv10", "kospi_mv20", "obv", "obv_ema",
#                      "trading_amount", "trading_amount_mv5", "trading_amount_mv10", "trading_amount_mv20",
#                      "volume", "volume_mv5", "volume_mv10", "volume_mv20"]
# logtrans_vec = []

# # corr mean >= 0.4
# feature_seed = 3
# selected_features = ["date", "close", "close_1shift", "close_2shift", "close_3shift", "close_4shift",
#                      "close_mv5", "close_mv10", "close_mv20", "obv",
#                      "trading_amount_mv5", "trading_amount_mv10", "trading_amount_mv20"]
# logtrans_vec = []

# # only close
# feature_seed = 4
# selected_features = ["date", "close"]
# logtrans_vec = []

# feature_seed = 5
# selected_features = ["date", "close", "close_1shift", "close_2shift", "close_3shift", "close_4shift"]
# logtrans_vec = []

# feature_seed = 6
# selected_features = ["date", "close", "close_1shift", "close_2shift", "close_3shift", "close_4shift",
#                      "close_mv5", "trading_amount_mv20"]
# logtrans_vec = []

# feature_seed = 7
# selected_features = ["date", "close"] + ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 8
# selected_features = ["date", "close"] + cat_vars_oh
# logtrans_vec = []

# feature_seed = 9
# selected_features = ["date", "close", "close_5shift"] + ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 10
# selected_features = ["date", "close", "close_1shift", "close_2shift", "close_5shift"] + ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 11
# selected_features = ["date", "close", "close_5shift", "dow", "snp500", "nasdaq", "semicon_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 12
# selected_features = ["date", "close", "close_5shift", "semicon_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 13
# selected_features = ["date", "close", "close_5shift", "nasdaq", "semicon_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 14
# selected_features = ["date", "close", "semicon_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 15
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 16
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "usdtocny"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 17
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "usdtoaud"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 18
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "sse_composite_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 19
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "sse_composite_index", "msci_emerging"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 20
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "sse_composite_index", "msci_emerging", "snp500"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 21
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "sse_composite_index", "msci_emerging", "nasdaq"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 22
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "usdtojpy",
# "sse_composite_index", "msci_emerging"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 23
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw",
#                      "sse_composite_index", "csi300_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 24
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw",
#                      "csi300_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 25
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw",
#                      "sse_composite_index", "us10y_tsy"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 26
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "eurtousd",
#                      "sse_composite_index"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 27
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "eurtousd",
#                      "sse_composite_index", "euro50"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []

# feature_seed = 28
# selected_features = ["date", "close", "close_5shift", "semicon_index", "usdtokrw", "eurtousd",
#                      "sse_composite_index", "msci_emerging"] + \
#                     ["weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"]
# logtrans_vec = []



# easyIO(stock_dic, folder_path + "dataset/stock_df_fe_" + start_date + "_" + end_date + ".pickle", op="w")
stock_dic = easyIO(None, folder_path + "dataset/stock_df_fe_" + start_date + "_" + end_date + ".pickle", op="r")
stock_dic["삼성전자"]["df"].head(20)
print(stock_dic["삼성전자"]["target_list"])

print(stock_dic["삼성전자"]["df"][selected_features])

# 정성적 판단 best set -> 26

feature_name = "feature_seed_" + str(feature_seed)
createFolder(folder_path + "result/" + feature_name + "/")
createFolder(folder_path + "submission/" + feature_name + "/")

for stock_name, stock_data in stock_dic.items():
    train_x = stock_data["df"].copy()

    # <feature selection>
    if selected_features is not None:
        tmp_list = [i for i in selected_features if i in train_x.columns]
        if len(tmp_list) != 0:
            train_x = train_x[tmp_list]

    train_x = train_x.dropna()
    train_x.reset_index(drop=True, inplace=True)

    # create target list
    target_list = []
    target_list.append(train_x["close"].copy())
    target_list.append(train_x["close"].shift(-1))
    target_list.append(train_x["close"].shift(-2))
    target_list.append(train_x["close"].shift(-3))
    target_list.append(train_x["close"].shift(-4))
    target_list.append(train_x["close"].shift(-5))
    for idx, value in enumerate(target_list):
        value.name = "target_shift" + str(idx)

    # feature 분포 시각화
    # ===== hist plot on numerical feature =====
    # for i in train_x.columns:
    #     if i == "date" or i in cat_vars + bin_vars:
    #         pass
    #     else:
    #         plt.figure(figsize=(12, 6))
    #         graph = sns.histplot(x=train_bi[i], bins=50, color="orange")
    #         graph.set_title("Distribution on " + stock_name + " (skewness : " + str(train_bi[i].skew().round(3)) + ")", fontsize=15, fontweight="bold", pad=15)
    #         graph.set_xlabel(graph.get_xlabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         graph.set_ylabel(graph.get_ylabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         plt.show()
    #         createFolder('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name)
    #         plt.savefig('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name + "/dist_" + i + ".png", dpi=300)
    #         plt.close()

    # <feature scaling>
    # log transform
    for i in logtrans_vec:
        if i in train_x.columns:
            train_x[i] = train_x[i].apply(np.log1p)

    # transformation 후 재 시각화
    # ===== hist plot on numerical feature =====
    # for i in logtrans_vec:
    #     if i in train_x.columns:
    #         plt.figure(figsize=(12, 6))
    #         graph = sns.histplot(x=train_x[i], bins=50, color="orange")
    #         graph.set_title("After log scaling distribution on " + stock_name + " (skewness : " + str(train_x[i].skew().round(3)) + ")", fontsize=15,
    #                         fontweight="bold", pad=15)
    #         graph.set_xlabel(graph.get_xlabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         graph.set_ylabel(graph.get_ylabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         plt.show()
    #         createFolder('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name)
    #         plt.savefig('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name + "/dist_logTrans_" + i + ".png", dpi=300)
    #         plt.close()

    stock_dic[stock_name]["df"] = train_x.copy()
    stock_dic[stock_name]["target_list"] = target_list

print(stock_dic["삼성전자"]["df"].shape)
print(stock_dic["삼성전자"]["target_list"][0].shape)


# ===== train, val, test split and auto prediction  =====
# train 2021/1/6 ~ 2021/9/5
# validation 2021/9/6 ~ 2021/9/17
# test 2021/9/27 ~ 2021/10/1

# 학습 전 필요 변수 초기화
kfolds_spliter = TimeSeriesSplit(n_splits=5, test_size=1, gap=0)

targetType = "numeric"
targetTask = None
class_levels = [0,1]
cut_off = 0

ds = None
result_val = None
result_test = None


model_names = ["Linear"]
# model_names = ["Linear", "LGB_GOSS"]

# ===== Automation Predict =====
# validation data evaluation
fit_runningtime = time()
# 데이터를 저장할 변수 설정
total_perf = None
for stock_name, stock_data in stock_dic.items():
    stock_data["perf_list"] = dict.fromkeys(model_names)
    stock_data["pred_list"] = dict.fromkeys(model_names)
    total_perf = dict.fromkeys(model_names)
    for i in model_names:
        stock_data["perf_list"][i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        stock_data["pred_list"][i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        total_perf[i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        for j in total_perf[i].keys():
            total_perf[i][j] = series(0, index=["MAE", "MAPE", "NMAE", "RMSE", "NRMSE", "R2", "Running_Time"])

target_timegap = 5
seqLength = 5
eval_from = "20211029"
val_year = 2021; val_month = 10; val_day = 22
test_year = 2021; test_month = 10; test_day = 29
for time_ngap in range(1,target_timegap+1):
    print(F"=== Target on N+{time_ngap} ===")
    for stock_name, stock_data in stock_dic.items():
        # remove date
        full_x = stock_data["df"]
        full_y = stock_data["target_list"][time_ngap]
        tmp_date = full_x["date"]
        arima_target = stock_data["target_list"][0]
        arima_date = stock_data["df"]["date"]

        # train_x_lstm = full_x[tmp_date <= datetime(val_year, val_month, val_day)][:-time_ngap]
        # train_y_lstm = full_y[tmp_date <= datetime(val_year, val_month, val_day)][seqLength - 1:-time_ngap]
        train_x = full_x[tmp_date <= datetime(val_year, val_month, val_day)][:-time_ngap]
        train_y = full_y[tmp_date <= datetime(val_year, val_month, val_day)][:-time_ngap]

        val_x = full_x[tmp_date == datetime(val_year, val_month, val_day)]
        val_y = full_y[tmp_date == datetime(val_year, val_month, val_day)]
        # val_x_lstm = full_x[(which(tmp_date == datetime(val_year, val_month, val_day)) - seqLength + 1): \
        #                     (which(tmp_date == datetime(val_year, val_month, val_day)) + target_timegap + 1)][:seqLength]
        # val_y_lstm = full_y[(which(tmp_date == datetime(val_year, val_month, val_day)) - seqLength + 1): \
        #                     (which(tmp_date == datetime(val_year, val_month, val_day)) + target_timegap + 1)][seqLength - 1:seqLength]
        arima_train = arima_target[arima_date <= datetime(val_year, val_month, val_day)]

        test_x = full_x[tmp_date == datetime(test_year, test_month, test_day)]
        # test_x_lstm = full_x[(which(tmp_date == datetime(test_year, test_month, test_day))-seqLength+1): \
        #                     (which(tmp_date == datetime(test_year, test_month, test_day))+target_timegap+1)][:seqLength]
        arima_full = arima_target[arima_date <= datetime(test_year, test_month, test_day)]

        # full_x_lstm = full_x[tmp_date <= datetime(test_year, test_month, test_day)][:-time_ngap]
        # full_y_lstm = full_y[tmp_date <= datetime(test_year, test_month, test_day)][seqLength - 1:-time_ngap]
        full_x = full_x[tmp_date <= datetime(test_year, test_month, test_day)][:-time_ngap]
        full_y = full_y[tmp_date <= datetime(test_year, test_month, test_day)][:-time_ngap]

        full_x.drop("date", axis=1, inplace=True)
        # full_x_lstm.drop("date", axis=1, inplace=True)
        train_x.drop("date", axis=1, inplace=True)
        # train_x_lstm.drop("date", axis=1, inplace=True)
        val_x.drop("date", axis=1, inplace=True)
        # val_x_lstm.drop("date", axis=1, inplace=True)
        test_x.drop("date", axis=1, inplace=True)
        # test_x_lstm.drop("date", axis=1, inplace=True)

        # <선형회귀>
        if "Linear" in model_names:
            tmp_runtime = time()
            print("Linear Regression on", stock_name)
            # evaludation on validation set
            model = doLinear(train_x, train_y, val_x, val_y,
                             targetType=targetType, targetTask=targetTask,
                             kfolds_prediction=False)
            print(model["performance"])
            stock_data["perf_list"]["Linear"][time_ngap] = model["performance"]
            tmp_perf = series(model["performance"])
            # prediction on test set
            model = doLinear(full_x, full_y, test_x, None,
                             targetType=targetType, targetTask=targetTask,
                             kfolds_prediction=False)
            stock_data["pred_list"]["Linear"][time_ngap] = model["pred"]
            tmp_runtime = time() - tmp_runtime
            total_perf["Linear"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # <엘라스틱넷>
        if "ElasticNet" in model_names:
            tmp_runtime = time()
            print("ElasticNet on", stock_name)
            # evaludation on validation set
            model = doElasticNet(train_x, train_y, val_x, val_y,
                                 targetType=targetType, targetTask=targetTask,
                                 kfolds_prediction=False, kfolds=kfolds_spliter,
                                 model_export=True)
            print(model["performance"])
            stock_data["perf_list"]["ElasticNet"][time_ngap] = model["performance"]
            tmp_perf = series(model["performance"])
            # prediction on test set
            model = doElasticNet(full_x, full_y, test_x, None,
                                 targetType=targetType, targetTask=targetTask,
                                 preTrained=model["model"])
            stock_data["pred_list"]["ElasticNet"][time_ngap] = model["pred"]
            tmp_runtime = time() - tmp_runtime
            total_perf["ElasticNet"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # # <KNN>
        # if "KNN" in model_names:
        #     tmp_runtime = time()
        #     print("KNN on", stock_name)
        #     # evaludation on validation set
        #     list(range(1,int(train_x.shape[0]*0.05),2))
        #     model = doKNN(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter)
        #     print(model["performance"])
        #     stock_data["perf_list"]["KNN"][time_ngap] = model["performance"]
        #     tmp_perf = series(model["performance"])
        #     # prediction on test set
        #     model = doKNN(full_x, full_y, test_x, None, kfolds=kfolds_spliter)
        #     stock_data["pred_list"]["KNN"][time_ngap] = model["pred"]
        #     tmp_runtime = time() - tmp_runtime
        #     total_perf["KNN"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # <LightGBM Gradient-based One-Side Sampling>
        if "LGB_GOSS" in model_names:
            tmp_runtime = time()
            print("LightGBM GOSS on", stock_name)
            # evaludation on validation sets
            model = doLGB(train_x, train_y, val_x, val_y, boostingType="goss",
                          targetType=targetType, targetTask=targetTask,
                          leavesSeq=[pow(2, i) - 1 for i in [6]], mcwSeq=[1e-3],
                          subsampleSeq=[0.8], colsampleSeq=[0.8], gammaSeq=[0.0],
                          kfolds_prediction=False, kfolds=kfolds_spliter, model_export=True)
            print(model["performance"])
            stock_data["perf_list"]["LGB_GOSS"][time_ngap] = model["performance"]
            tmp_perf = series(model["performance"])
            # prediction on test set
            model = doLGB(full_x, full_y, test_x, None, boostingType="goss",
                          targetType=targetType, targetTask=targetTask,
                          preTrained=model["model"])
            stock_data["pred_list"]["LGB_GOSS"][time_ngap] = model["pred"]
            tmp_runtime = time() - tmp_runtime
            total_perf["LGB_GOSS"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # # <ARIMA>
        # if "ARIMA" in model_names:
        #     tmp_runtime = time()
        #     print("ARIMA on", stock_name)
        #     # order=(p: Auto regressive, q: Difference, d: Moving average)
        #     # 일반적 하이퍼파라미터 공식
        #     # 1. p + q < 2
        #     # 2. p * q = 0
        #     # 근거 : 실제로 대부분의 시계열 자료에서는 하나의 경향만을 강하게 띄기 때문 (p 또는 q 둘중 하나는 0)
        #     model = ARIMA(arima_train, order=(1, 2, 0))
        #     model_fit = model.fit()
        #     pred = array([model_fit.forecast(time_ngap).iloc[-1]])
        #     tmp_mae = metrics.mean_absolute_error(val_y, pred)
        #     tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
        #     tmp_perf = {"MAE": tmp_mae,
        #                 "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
        #                 "NMAE": tmp_mae / val_y.abs().mean(),
        #                 "RMSE": tmp_rmse,
        #                 "NRMSE": tmp_rmse / val_y.abs().mean(),
        #                 "R2": metrics.r2_score(val_y, pred)}
        #     print(tmp_perf)
        #     stock_data["perf_list"]["ARIMA"][time_ngap] = tmp_perf
        #
        #     # prediction on test data
        #     model = ARIMA(arima_full, order=(1, 2, 0))
        #     model_fit = model.fit()
        #
        #     stock_data["pred_list"]["ARIMA"][time_ngap] = array([model_fit.forecast(time_ngap).iloc[-1]])
        #     # recode running time
        #     tmp_runtime = time() - tmp_runtime
        #     total_perf["ARIMA"][time_ngap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))

        # # <LSTM>
        # tmp_runtime = time()
        # print("LSTM V1 on", stock_name)
        # model = doMLP(train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, mlpName="MLP_LSTM_V1",
        #               hiddenLayers=128, epochs=100, batch_size=2, seqLength=seqLength, model_export=True)
        # print(model["performance"])
        # stock_data["perf_list"]["MLP_LSTM_V1"][time_ngap] = model["performance"]
        # tmp_perf = series(model["performance"])
        #
        # model = doMLP(full_x_lstm, full_y_lstm, test_x_lstm, test_y_lstm,
        #               seqLength=seqLength, preTrained=model["model"])
        # stock_data["pred_list"]["MLP_LSTM_V1"][time_ngap] = model["pred"]
        # tmp_runtime = time() - tmp_runtime
        # total_perf["MLP_LSTM_V1"][time_ngap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))

    for i in model_names:
        total_perf[i][time_ngap] /= len(stock_dic.keys())
fit_runningtime = time() - fit_runningtime

# prediction value check
print(stock_dic["삼성전자"]["pred_list"])
print(fit_runningtime)


# 성능평가 테이블 생성
perf_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1,6)])
runningtime_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1,6)])
for i in list(total_perf.keys()):
    if array(list(total_perf[i].values())).sum() == 0:
        pass
    else:
        perf_table.loc[i] = dataframe(total_perf[i]).loc["NMAE"].values
        runningtime_table.loc[i] = dataframe(total_perf[i]).loc["Running_Time"].values

# NMAE = MAPE
perf_table = perf_table.iloc[:,:target_timegap]
perf_table = perf_table * 100
perf_table.loc["best_model"] = perf_table.min(axis=0)
perf_table["avg"] = perf_table.iloc[:,:5].mean(axis=1)
perf_table["std"] = perf_table.iloc[:,:5].std(axis=1)
perf_table["running_time"] = runningtime_table.mean(axis=1).append(series({"best_model": -1}))
print(perf_table)
perf_table.to_csv(folder_path + "result/" + feature_name + "/perf_" + feature_name + ".csv", index=True)

# smoothing_alpha = 1
# effic_table = (1/(perf_table.iloc[:,:target_timegap] + 1e-1 * smoothing_alpha))
# effic_table = effic_table.apply(lambda x: x/(perf_table["running_time"] + 1e-5 * smoothing_alpha))
# effic_table = effic_table * 100
# effic_table["avg"] = effic_table.iloc[:,:5].mean(axis=1)
# effic_table["std"] = effic_table.iloc[:,:5].std(axis=1)
# print(effic_table.mean(axis=1))
# print(effic_table)
# effic_table.to_csv(folder_path + "result/" + feature_name + "/effi_" + feature_name + ".csv", index=True)

# perf_table.to_csv("projects/dacon_stockprediction/eval_result/perf_" + feature_test_seed + "_" + eval_from + ".csv")
# effic_table.to_csv("projects/dacon_stockprediction/eval_result/effic_" + feature_test_seed + "_" + eval_from + ".csv")



# export submission file
submission = read_csv("projects/dacon_stockprediction/sample_submission.csv")
for for_model in model_names:
    for i in submission.columns[1:]:
        tmp_list = []
        for j in stock_dic[stock_list.index[stock_list == i][0]]["pred_list"][for_model].values():
            tmp_list.append(j[0])
        submission[i][:5] = tmp_list
    submission.to_csv(folder_path + "submission/" + feature_name + "/" + feature_name + "_" + for_model + "" + ".csv", index=False)

# test set prediction
# for i in submission.columns[1:]:
#     print(stock_dic[stock_list.index[stock_list == i][0]]["pred_list"])
#     submission[i] = [i.round()[0] for i in stock_dic[stock_list.index[stock_list == i][0]]["pred_list"]["LGB_GOSS"].values()]
#
# print(submission.isna().sum().sum())
# submission.to_csv("projects/dacon_stockprediction/submission/submission_lightgbm_goss.csv", index=False)

feature_perf_dic = dict.fromkeys(model_names, dataframe())
cnt = 1
while True:
    try:
        perf_tmp = read_csv("projects/dacon_stockprediction/result/feature_seed_" + str(cnt) + "/perf_feature_seed_" + str(cnt) + ".csv")
    except:
        break
    for k in feature_perf_dic.keys():
        tmp_idx = which(perf_tmp.iloc[:,0] == k)
        if type(tmp_idx) != np.ndarray:
            feature_perf_dic[k] = pd.concat([feature_perf_dic[k], perf_tmp.iloc[tmp_idx,1:].to_frame().T], axis=0)
            tmp_name = list(feature_perf_dic[k].index); tmp_name[-1] = "feature_seed_" + str(cnt)
            feature_perf_dic[k].index = tmp_name
    cnt += 1


for k in feature_perf_dic.keys():
    print("=====", k, "=====")
    print(feature_perf_dic[k])
    print()
    print("<best feature set>")
    print(feature_perf_dic[k]["avg"].index[feature_perf_dic[k]["avg"] == feature_perf_dic[k]["avg"].min()][0])
    print(); print()




