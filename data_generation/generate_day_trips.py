# %%
from tensorflow.keras import backend as K
K.clear_session()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import datetime as dt
import importlib
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark import SparkConf
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, concatenate, GlobalAveragePooling1D
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Model
import IPython
from copy import deepcopy
from tqdm import trange, tqdm

mpl.rcParams['figure.facecolor'] = 'white'

import warnings

import pandas as pd
import swifter
pd.set_option('display.max_columns', None)
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.get_logger().setLevel('INFO')
import pyspark
print(pyspark.__version__)
spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.autoBroadcastJoinThreshold", -1)\
        .config("spark.driver.maxResultSize", 0)\
        .config("spark.shuffle.spill", "true")\
        .getOrCreate()

# %%
# Get data for the specified day
def get_apc_data_for_date(filter_date):
    print(f"Running this get_apc_data_for_date({filter_date})...")
    # filepath = os.path.join('data', 'processed', 'apc_weather_gtfs.parquet')
    filepath = '/home/jptalusan/mta_stationing_problem/data/processed/apc_weather_gtfs_20220921.parquet'
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    plot_date = filter_date.strftime('%Y-%m-%d')
    get_columns = ['trip_id', 'transit_date', 'arrival_time', 'scheduled_time', 'vehicle_id',
                   'block_abbr', 'stop_sequence', 'stop_name', 'stop_id_original',
                   'load', 
                   'darksky_temperature', 
                   'darksky_humidity', 
                   'darksky_precipitation_probability', 
                   'route_direction_name', 'route_id', 'gtfs_direction_id',
                   'dayofweek',  'year', 'month', 'hour', 'zero_load_at_trip_end',
                   'sched_hdwy']
    get_str = ", ".join([c for c in get_columns])
    query = f"""
    SELECT {get_str}
    FROM apc
    WHERE (transit_date == '{plot_date}')
    ORDER BY arrival_time
    """
    apcdata = spark.sql(query)
    apcdata = apcdata.na.fill(value=0,subset=["zero_load_at_trip_end"])
    apcdata = apcdata.withColumn("route_id_dir", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name))
    apcdata = apcdata.withColumn("day", F.dayofmonth(apcdata.arrival_time))
    apcdata = apcdata.drop("route_direction_name")
    apcdata = apcdata.withColumn("load", F.when(apcdata.load < 0, 0).otherwise(apcdata.load))
    return apcdata

# %%
def prepare_input_data(input_df, ohe_encoder, label_encoders, num_scaler, columns, keep_columns=[], target='y_class'):
    num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy', 'traffic_speed']
    cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', 'time_window']
    ohe_columns = ['dayofweek', 'route_id_dir', 'is_holiday', 'is_school_break', 'zero_load_at_trip_end']

    # OHE
    input_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(input_df[ohe_columns]).toarray()
    # input_df = input_df.drop(columns=ohe_columns)

    # Label encoder
    for cat in cat_columns:
        encoder = label_encoders[cat]
        input_df[cat] = encoder.transform(input_df[cat])
    
    # Num scaler
    input_df[num_columns] = num_scaler.transform(input_df[num_columns])
    input_df['y_class']  = input_df.y_class.astype('int')

    if keep_columns:
        columns = keep_columns + columns
    # Rearrange columns
    input_df = input_df[columns]
    
    return input_df

def assign_data_to_bins(df, TARGET='load'):
    bins = pd.IntervalIndex.from_tuples([(-1, 6.0), (6.0, 12.0), (12.0, 55.0), (55.0, 75.0), (75.0, 100.0)])
    mycut = pd.cut(df[TARGET].tolist(), bins=bins)
    df['y_class'] = mycut.codes
    return df

# %%
TIMEWINDOW = 15
def add_features(df):
    df = df[df.arrival_time.notna()]
    df = df[df.sched_hdwy.notna()]
    df = df[df.darksky_temperature.notna()]

    df['day'] = df["arrival_time"].dt.day
    df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

    # Adding extra features
    # Holidays
    fp = os.path.join('data', 'US Holiday Dates (2004-2021).csv')
    holidays_df = pd.read_csv(fp)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    holidays_df['is_holiday'] = True
    df = df.merge(holidays_df[['Date', 'is_holiday']], left_on='transit_date', right_on='Date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(False)
    df = df.drop(columns=['Date'])
        
    # School breaks
    fp = os.path.join('data', 'School Breaks (2019-2022).pkl')
    school_break_df = pd.read_pickle(fp)
    school_break_df['is_school_break'] = True
    df = df.merge(school_break_df[['Date', 'is_school_break']], left_on='transit_date', right_on='Date', how='left')
    df['is_school_break'] = df['is_school_break'].fillna(False)
    df = df.drop(columns=['Date'])

    # Traffic
    # Causes 3M data points to be lost
    fp = os.path.join('data', 'triplevel_speed.pickle')
    speed_df = pd.read_pickle(fp)
    speed_df = speed_df.rename({'route_id_direction':'route_id_dir'}, axis=1)
    speed_df = speed_df[['transit_date', 'trip_id', 'route_id_dir', 'traffic_speed']]
    df = df.merge(speed_df, how='left', 
                    left_on=['transit_date', 'trip_id', 'route_id_dir'], 
                    right_on=['transit_date', 'trip_id', 'route_id_dir'])
    # df = df[~df['traffic_speed'].isna()]
    df['traffic_speed'].bfill(inplace=True)

    df['minute'] = df['arrival_time'].dt.minute
    df['minuteByWindow'] = df['minute'] // TIMEWINDOW
    df['temp'] = df['minuteByWindow'] + (df['hour'] * 60 / TIMEWINDOW)
    df['time_window'] = np.floor(df['temp']).astype('int')
    df = df.drop(columns=['minute', 'minuteByWindow', 'temp'])

    # HACK
    df = df[df['hour'] != 3]
    df = df[df['stop_sequence'] != 0]

    df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

    df = assign_data_to_bins(df, TARGET='load')
    return df

# %%
DATE = '2021-08-23'
date_to_predict = dt.datetime.strptime(DATE, '%Y-%m-%d')
# date_to_predict
# date_to_predict = dt.date(2021, 8, 23)
apcdata = get_apc_data_for_date(date_to_predict)
df = apcdata.toPandas()
df = add_features(df)

# %%
# Load model
latest = tf.train.latest_checkpoint('models/same_day/school_zero_load')
columns = joblib.load('models/same_day/LL_X_columns.joblib')
label_encoders = joblib.load('models/same_day/LL_Label_encoders.joblib')
ohe_encoder = joblib.load('models/same_day/LL_OHE_encoder.joblib')
num_scaler = joblib.load('models/same_day/LL_Num_scaler.joblib')

raw_df = deepcopy(df)
input_df = prepare_input_data(df, ohe_encoder, label_encoders, num_scaler, columns, target='y_class')

# %%
def setup_simple_lstm_generator(num_features, num_classes, learning_rate=1e-4):
    # define model
    model = tf.keras.Sequential()
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"],
    )

    input_shape = (None, None, num_features)
    model.build(input_shape)
    return model

def generate_simple_lstm_predictions(input_df, model, past, future):
    past_df = input_df[0:past]
    future_df = input_df[past:]
    predictions = []
    if future == None:
        future = len(future_df)
    for f in range(future):
        pred = model.predict(past_df.to_numpy().reshape(1, *past_df.shape))
        y_pred = np.argmax(pred)
        predictions.append(y_pred)
        
        # Add information from future
        last_row = future_df.iloc[[0]]
        last_row['y_class'] = y_pred
        past_df = pd.concat([past_df[1:], last_row])
        
        # Move future to remove used row
        future_df = future_df[1:]
    return predictions

# %%
# import random

# percentiles = [(0, 6.0), (6.0, 12.0), (12.0, 55.0), (55.0, 75.0), (75.0, 100.0)]

# NUM_CLASSES = 5
# FUTURE = None
# PAST = 5

# NUM_TRIPS = 2
# if NUM_TRIPS == None:
#     rand_trips = df.trip_id.unique().tolist()
# else:
#     rand_trips = random.sample(df.trip_id.unique().tolist(), NUM_TRIPS)


# from multiprocessing import Process, Queue, cpu_count, Manager
# from time import sleep

# queue = Queue()

# def mp_worker(L, queue):

#     while queue.qsize() > 0 :
#         trip_id = queue.get()
#         _df = df.query("trip_id == @trip_id")
#         _input_df = input_df.loc[_df.index]
#         model = tf.keras.Sequential()
#         model.add(LSTM(256, return_sequences=True))
#         model.add(LSTM(256))
#         model.add(Dropout(0.2))
#         model.add(Dense(128, activation='relu'))
#         model.add(Dropout(0.2))
#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(5, activation='softmax'))

#         # compile model
#         model.compile(
#             loss="sparse_categorical_crossentropy",
#             optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#             metrics=["sparse_categorical_accuracy"],
#         )

#         input_shape = (None, None, _input_df.shape[1])
#         model.build(input_shape)
#         print(trip_id)
#         y_pred = generate_simple_lstm_predictions(_input_df, model, PAST, FUTURE)
#         print(y_pred)
#         L.append(_input_df)
#         # try:
#         #     loads = [random.randint(percentiles[yp][0], percentiles[yp][1]) for yp in y_pred]
            
#         #     _raw_df = raw_df.loc[_df.index]
#         #     y_true = _raw_df[0:PAST]['load'].tolist()
#         #     a = y_true + loads
#         #     _raw_df['sampled_loads'] = a
            
#         #     L.append(_raw_df)
#         # except:
#         #     print(f"FAILED:{trip_id}")
#         #     L.append(pd.DataFrame())
#         #     continue


# def mp_handler():
#     with Manager() as manager:
#         L = manager.list()
        
#         # Spawn two processes, assigning the method to be executed 
#         # and the input arguments (the queue)
#         processes = [Process(target=mp_worker, args=(L,queue,)) for _ in range(cpu_count() - 1)]

#         for process in processes:
#             process.start()

#         for process in processes:
#             process.join()

#         # trip_res = pd.concat(L)
#         # return trip_res

# print(rand_trips)
# for trip_id in rand_trips:
#     queue.put(trip_id)

# # trip_res = 
# mp_handler()

# %%
import random

percentiles = [(0, 6.0), (6.0, 12.0), (12.0, 55.0), (55.0, 75.0), (75.0, 100.0)]

NUM_CLASSES = 5
FUTURE = None
PAST = 5

NUM_TRIPS = None
if NUM_TRIPS == None:
    rand_trips = df.trip_id.unique().tolist()
else:
    rand_trips = random.sample(df.trip_id.unique().tolist(), NUM_TRIPS)

model = setup_simple_lstm_generator(input_df.shape[1], NUM_CLASSES)
model.load_weights(latest)

trip_res = []
load_arr = []
for trip_id in tqdm(rand_trips):
    _df = df.query("trip_id == @trip_id")
    try:
        _input_df = input_df.loc[_df.index]
        y_pred = generate_simple_lstm_predictions(_input_df, model, PAST, FUTURE)
        loads = [random.randint(percentiles[yp][0], percentiles[yp][1]) for yp in y_pred]
        
        _raw_df = raw_df.loc[_df.index]
        y_true = _raw_df[0:PAST]['load'].tolist()
        a = y_true + loads
        _raw_df['sampled_loads'] = a
        
        trip_res.append(_raw_df)
    except:
        print(f"FAILED:{trip_id}")
        continue

trip_res = pd.concat(trip_res)

# %%
_columns = ['trip_id', 'transit_date', 'arrival_time', 'scheduled_time', 'vehicle_id', 'block_abbr', 'stop_sequence', 'stop_id_original', 'route_id_dir', 'zero_load_at_trip_end', 'sampled_loads']
trip_res = trip_res[_columns]

fp = 'results/sampled_loads.pkl'
trip_res.to_pickle(fp)


