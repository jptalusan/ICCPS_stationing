from tensorflow.keras import backend as K
K.clear_session()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import datetime as dt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
import pickle
import random
import json
import joblib
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

mpl.rcParams['figure.facecolor'] = 'white'

import warnings

import pandas as pd
pd.set_option('display.max_columns', None)
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.get_logger().setLevel('INFO')
import pyspark

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.autoBroadcastJoinThreshold", -1)\
        .config("spark.driver.maxResultSize", 0)\
        .config("spark.shuffle.spill", "true")\
        .getOrCreate()
        
def get_apc_data_for_date(filter_date):
    print("Running this...")
    filepath = '/home/jptalusan/mta_stationing_problem/data/processed/apc_weather_gtfs_20220921.parquet'
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    plot_date = filter_date.strftime('%Y-%m-%d')
    get_columns = ['trip_id', 'transit_date', 'arrival_time', 'scheduled_time',
                'block_abbr', 'stop_sequence', 'stop_id_original',
                'vehicle_id', 'vehicle_capacity',
                'load', 
                'darksky_temperature', 
                'darksky_humidity', 
                'darksky_precipitation_probability', 
                'route_direction_name', 'route_id', 'overload_id',
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
    apcdata = apcdata.withColumn("route_id_dir", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name))
    apcdata = apcdata.withColumn("day", F.dayofmonth(apcdata.arrival_time))
    apcdata = apcdata.drop("route_direction_name")
    apcdata = apcdata.withColumn("load", F.when(apcdata.load < 0, 0).otherwise(apcdata.load))
    apcdata = apcdata.na.fill(value=0,subset=["zero_load_at_trip_end"])
    return apcdata

def prepare_input_data(input_df, ohe_encoder, label_encoders, num_scaler, columns, keep_columns=[], target='y_class'):
    num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy']
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

TIMEWINDOW = 15
def add_features(df):
    df = df[df.arrival_time.notna()]
    df = df.fillna(method="bfill")

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

    df['minute'] = df['arrival_time'].dt.minute
    df['minuteByWindow'] = df['minute'] // TIMEWINDOW
    df['temp'] = df['minuteByWindow'] + (df['hour'] * 60 / TIMEWINDOW)
    df['time_window'] = np.floor(df['temp']).astype('int')
    df = df.drop(columns=['minute', 'minuteByWindow', 'temp'])

    # HACK
    # df = df[df['hour'] != 3]
    # df = df[df['stop_sequence'] != 0]

    df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

    df = assign_data_to_bins(df, TARGET='load')
    return df

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
    pred_probs = []
    if future == None:
        future = len(future_df)
    for f in range(future):
        pred = model.predict(past_df.to_numpy().reshape(1, *past_df.shape))
        pred_probs.append(pred)
        y_pred = np.argmax(pred)
        predictions.append(y_pred)
        
        # Add information from future
        last_row = future_df.iloc[[0]]
        last_row['y_class'] = y_pred
        past_df = pd.concat([past_df[1:], last_row])
        
        # Move future to remove used row
        future_df = future_df[1:]
    return predictions, pred_probs

def compute_ons_offs(s):
    curr_load = s['sampled_loads']
    next_load = s['next_load']
    if next_load > curr_load:
        ons = next_load - curr_load
        offs = 0
    elif next_load < curr_load:
        ons = 0
        offs = curr_load - next_load
    else:
        ons = 0
        offs = 0
        
    return ons, offs

def fix_time(d, x):
    if x[0:2] == '24':
        arrival_time = '00'+x[2:]
        return pd.to_datetime(d + ' ' + arrival_time) + pd.Timedelta(days=1)
    if x[0:2] == '25':
        arrival_time = '01'+x[2:]
        return pd.to_datetime(d + ' ' + arrival_time) + pd.Timedelta(days=1)
    if x[0:2] == '26':
        arrival_time = '02'+x[2:]
        return pd.to_datetime(d + ' ' + arrival_time) + pd.Timedelta(days=1)
    if x[0:2] == '27':
        arrival_time = '03'+x[2:]
        return pd.to_datetime(d + ' ' + arrival_time) + pd.Timedelta(days=1)
    return pd.to_datetime(d + ' ' + x)

def merge_overload_regular_bus_trips(regular, overload):
    m = regular.merge(overload, how='left', on=['trip_id', 'transit_date', 'scheduled_time', 'block_abbr', 'stop_sequence', 'stop_id_original', 'route_id_dir', 'route_id'])
    
    m['arrival_time'] = np.max(m[['arrival_time_x', 'arrival_time_y']], axis=1)
    
    m['zero_load_at_trip_end'] = m['zero_load_at_trip_end_x']
    
    m.loc[~m['arrival_time_x'].isnull(), "load"] = m['load_x']
    # m.loc[~m['arrival_time_x'].isnull(), "ons"] = m['ons_x']
    # m.loc[~m['arrival_time_x'].isnull(), "offs"] = m['offs_x']
    
    m.loc[~m['arrival_time_y'].isnull(), "load"] = m['load_y']
    # m.loc[~m['arrival_time_y'].isnull(), "ons"] = m['ons_y']
    # m.loc[~m['arrival_time_y'].isnull(), "offs"] = m['offs_y']
    
    m['vehicle_id'] = m['vehicle_id_x']
    m['vehicle_capacity'] = m['vehicle_capacity_x']
    m['overload_id'] = m['overload_id_x']
    m = m[m.columns.drop(list(m.filter(regex='_x')))]
    m = m[m.columns.drop(list(m.filter(regex='_y')))]
    # m = m[regular.columns]
    return m

# Load model
latest = tf.train.latest_checkpoint('models/no_speed')
columns = joblib.load('models/LL_X_columns.joblib')
label_encoders = joblib.load('models/LL_Label_encoders.joblib')
ohe_encoder = joblib.load('models/LL_OHE_encoder.joblib')
num_scaler = joblib.load('models/LL_Num_scaler.joblib')

ohe_columns = ['dayofweek', 'route_id_dir', 'is_holiday', 'is_school_break', 'zero_load_at_trip_end']
percentiles = [(0, 6.0), (6.0, 12.0), (12.0, 55.0), (55.0, 75.0), (75.0, 100.0)]
NUM_CLASSES = 5
FUTURE = None
PAST = 5
NUM_TRIPS = None

def generate_traffic_data_for_date(DATE, GTFS_MONTH, CHAINS):
    # date_to_predict = dt.datetime.strptime(DATE, '%Y-%m-%d')
    # apcdata = get_apc_data_for_date(date_to_predict)
    # df = apcdata.toPandas()

    # # HACK
    # # a = df.query("trip_id == '259845' and vehicle_id == '1818'").sort_values('stop_sequence')
    # # b = df.query("trip_id == '259845' and vehicle_id == '2008'").sort_values('stop_sequence')
    # # m1 = merge_overload_regular_bus_trips(a, b)

    # # a = df.query("trip_id == '259635' and vehicle_id == '2019'").sort_values('stop_sequence')
    # # b = df.query("trip_id == '259635' and vehicle_id == '1914'").sort_values('stop_sequence')
    # # m2 = merge_overload_regular_bus_trips(a, b)

    # df = df.query("overload_id == 0")
    # # overload_trips = df.query("overload_id > 0").trip_id.unique()
    # # tdf = tdf[~tdf['trip_id'].isin(overload_trips)]
    # # df = pd.concat([tdf, m1, m2])
    # df = df.dropna(subset=['arrival_time'])
    # # df = df.fillna(method='ffill').fillna(method='bfill')

    # # HACK
    # df = df.query("route_id != 95")
    # df = df.query("route_id != 89")
    # df = df[~df['stop_id_original'].isin(['PEARL', 'JOHASHEN', 'ROS10AEN'])]

    # df = add_features(df)
    # raw_df = deepcopy(df)

    # # HACK
    # # df.loc[df['time_window'].isin([6, 7, 8]), 'time_window'] = 16

    # input_df = prepare_input_data(df, ohe_encoder, label_encoders, num_scaler, columns, target='y_class')
    # input_df = input_df.drop(columns=ohe_columns)

    # if NUM_TRIPS == None:
    #     rand_trips = df.trip_id.unique().tolist()
    # else:
    #     rand_trips = random.sample(df.trip_id.unique().tolist(), NUM_TRIPS)

    # model = setup_simple_lstm_generator(input_df.shape[1], NUM_CLASSES)
    # model.load_weights(latest)

    # trip_res = []
    # for trip_id in tqdm(rand_trips):
    #     _df = df.query("trip_id == @trip_id")
    #     try:
    #         _input_df = input_df.loc[_df.index]
    #         _, y_pred_probs = generate_simple_lstm_predictions(_input_df, model, PAST, FUTURE)
            
    #         # Introducing stochasticity
    #         y_pred = [np.random.choice(len(ypp.flatten()), size=1, p=ypp.flatten())[0] for ypp in y_pred_probs]
        
    #         loads = [random.randint(percentiles[yp][0], percentiles[yp][1]) for yp in y_pred]
            
    #         _raw_df = raw_df.loc[_df.index]
    #         y_true = _raw_df[0:PAST]['load'].tolist()
    #         a = y_true + loads
    #         _raw_df['sampled_loads'] = a
            
    #         y_true_classes = _raw_df[0:PAST]['y_class'].tolist()
    #         _raw_df['y_pred_classes'] = y_true_classes + y_pred
    #         _raw_df['y_pred_probs'] = [[-1] * NUM_CLASSES]*len(y_true_classes) + [ypp[0] for ypp in y_pred_probs]
            
    #         trip_res.append(_raw_df)
    #     except:
    #         print(f"FAILED:{trip_id}")
    #         continue

    # trip_res = pd.concat(trip_res)
    # _columns = ['trip_id', 'transit_date', 'arrival_time', 'scheduled_time', 'block_abbr', 
    #             'stop_sequence', 'stop_id_original', 'route_id_dir', 'zero_load_at_trip_end', 
    #             'y_pred_classes', 'y_pred_probs', 'sampled_loads', 'vehicle_id', 'vehicle_capacity']
    # trip_res_df = trip_res[_columns]

    # # fp = 'results/sampled_loads.pkl'
    # fp = f'results/sampled_loads_{DATE.replace("-","")}.pkl'
    # trip_res_df.to_pickle(fp)

    DEFAULT_CAPACITY = 10.0

    overall_vehicle_plan = {}
    # start_time = '08:00:00'
    # end_time = '12:00:00'
    fp = f'results/sampled_loads_{DATE.replace("-","")}.pkl'
    trip_res_df = pd.read_pickle(fp)

    # start_datetime = dt.datetime.strptime(f"{DATE} {start_time}", "%Y-%m-%d %H:%M:%S")
    # end_datetime = dt.datetime.strptime(f"{DATE} {end_time}", "%Y-%m-%d %H:%M:%S")

    # arr = []
    # for trip_id, trip_df in trip_res_df.groupby('trip_id'):
    #     if (trip_df.scheduled_time.min() >= start_datetime) and (trip_df.scheduled_time.max() <= end_datetime):
    #         arr.append(trip_df)

    # trip_res_df = pd.concat(arr)

    # TODO: run again with vehicle_capacity (above)
    for vehicle_id, vehicle_df in trip_res_df.groupby('vehicle_id'):
        vehicle_df = vehicle_df.dropna(subset=['arrival_time']).sort_values(['scheduled_time'])
        # vehicle_capacity = vehicle_df.iloc[0].vehicle_capacity
        vehicle_capacity = DEFAULT_CAPACITY
        # if np.isnan(vehicle_capacity):
        #     vehicle_capacity = DEFAULT_CAPACITY
        # TODO: This is not the baseline behavior
        starting_depot = 'MCC5_1'
        service_type = 'regular'
        blocks = [block for block in vehicle_df.block_abbr.unique().tolist()]
        trips = []
        for block in blocks:
            block_df = vehicle_df.query("block_abbr == @block")
            for trip in block_df.trip_id.unique().tolist():
                trips.append((str(block), str(trip)))
        overall_vehicle_plan[vehicle_id] = {'vehicle_capacity': vehicle_capacity, 'trips': trips, 'starting_depot': starting_depot, 'service_type': service_type}

    OVERLOAD_BUSES = 5
    for vehicle_id in range(41, 41 + OVERLOAD_BUSES):
        overall_vehicle_plan[str(vehicle_id)] = {'vehicle_capacity': 55.0, 'trips': [], "starting_depot": "MCC5_1", 'service_type': "overload"}
        
    with open(f'results/vehicle_plan_{DATE.replace("-", "")}.json', 'w') as fp:
        json.dump(overall_vehicle_plan, fp, sort_keys=True, indent=2)
        
    overall_block_plan = {}
    for block_abbr, block_df in trip_res_df.groupby('block_abbr'):
        block_df = block_df.dropna(subset=['arrival_time']).sort_values(['scheduled_time'])
        trip_ids = block_df.trip_id.unique().tolist()
        start_time = block_df[block_df['trip_id'] == trip_ids[0]].iloc[0]['scheduled_time'].strftime('%Y-%m-%d %H:%M:%S')
        end_time = block_df[block_df['trip_id'] == trip_ids[-1]].iloc[-1]['scheduled_time'].strftime('%Y-%m-%d %H:%M:%S')
        overall_block_plan[block_abbr] = {'trip_ids': trip_ids,
                                        'start_time': start_time,
                                        'end_time': end_time}

    overall_trip_plan = {}
    for trip_id, trip_df in trip_res_df.groupby('trip_id'):
        trip_df = trip_df.dropna(subset=['arrival_time']).sort_values(['scheduled_time'])
        route_id_dir = trip_df.iloc[0].route_id_dir
        block_abbr = int(trip_df.iloc[0].block_abbr)
        route_id = int(route_id_dir.split("_")[0])
        route_direction = route_id_dir.split("_")[1]
        zero_load_at_trip_end = trip_df.iloc[-1].zero_load_at_trip_end.tolist()
        scheduled_time = trip_df.scheduled_time.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        stop_sequence = trip_df.stop_sequence.tolist()
        stop_sequence = list(range(0, len(stop_sequence)))
        stop_id_original = trip_df.stop_id_original.tolist()
        
        overall_trip_plan[trip_id] = {'route_id': route_id, 
                                    'block_abbr': block_abbr,
                                    'route_direction': route_direction, 
                                    'scheduled_time': scheduled_time, 
                                    'stop_sequence': stop_sequence, 
                                    'stop_id_original': stop_id_original,
                                    'zero_load_at_trip_end':zero_load_at_trip_end,
                                    'last_stop_sequence': stop_sequence[-1],
                                    'last_stop_id': stop_id_original[-1]}

    with open(f'results/trip_plan_{DATE.replace("-", "")}.json', 'w') as fp:
        json.dump(overall_trip_plan, fp, sort_keys=True, indent=2)
        
    # fp = f'results/sampled_loads_{DATE.replace("-","")}.pkl'
    # trip_res = pd.read_pickle(fp)

    for chain in tqdm(range(CHAINS)):
        loads = [random.randint(percentiles[yp][0], percentiles[yp][1]) for yp in trip_res_df.y_pred_classes]
        trip_res_df['sampled_loads'] = loads

        sampled_ons_offs = []
        for trip_id, trip_id_df in trip_res_df.groupby(['transit_date', 'trip_id']):
            tdf = trip_id_df.sort_values('stop_sequence').reset_index(drop=True)
            tdf['stop_sequence'] = list(range(1, len(tdf) + 1))
            tdf['ons'] = 0
            tdf['offs'] = 0
            tdf['next_load'] = tdf['sampled_loads'].shift(-1)
            
            # Intermediate stops
            tdf[['ons', 'offs']] = tdf.apply(compute_ons_offs, axis=1, result_type="expand")
            
            # first and last stops
            tdf.at[0, 'ons'] = tdf.iloc[0]['sampled_loads']
            tdf.at[len(tdf) - 1, 'offs'] = tdf.iloc[-1]['sampled_loads']
            sampled_ons_offs.append(tdf)
            
        df = pd.concat(sampled_ons_offs)
        df['key_pair'] = list(zip(df.route_id_dir, 
                                df.block_abbr,
                                df.stop_sequence,
                                df.stop_id_original, 
                                df.scheduled_time))
        df = df.set_index('key_pair')
        drop_cols = ['trip_id', 'route_id_dir', 'block_abbr', 'stop_id_original', 'stop_id', 'scheduled_time', 
                        'transit_date', 'arrival_time', 'zero_load_at_trip_end', 'y_pred_classes', 'y_pred_probs',
                        'vehicle_capacity', 'vehicle_id', 'stop_sequence']
        drop_cols = [dc for dc in drop_cols if dc in df.columns]
        df = df.drop(drop_cols, axis=1)
        sampled_ons_offs_dict = df.to_dict('index')
        
        if chain == 0:
            with open(f'results/sampled_ons_offs_dict_{DATE.replace("-", "")}.pkl', 'wb') as handle:
                pickle.dump(sampled_ons_offs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pass
        else:
            Path(f'results/chains/{DATE.replace("-","")}').mkdir(parents=True, exist_ok=True)
            with open(f'results/chains/{DATE.replace("-","")}/ons_offs_dict_chain_{DATE.replace("-","")}_{chain - 1}.pkl', 'wb') as handle:
                pickle.dump(sampled_ons_offs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # stop_times_fp = f'data/GTFS/{GTFS_MONTH}/stop_times.txt'
    # stop_times_df = pd.read_csv(stop_times_fp)
    # stop_times_df['date'] = DATE
    
    # stop_times_df['scheduled_time'] = stop_times_df.apply(lambda x: fix_time(x.date, x.arrival_time), axis=1)

    # stop_times_df['key_pair'] = list(zip(stop_times_df.trip_id, stop_times_df.stop_id, stop_times_df.scheduled_time))
    # stop_times_df = stop_times_df.set_index('key_pair')

    # drop_cols = ['departure_time', 'stop_id', 'stop_sequence', 'stop_headsign', 'trip_id',
    #             'pickup_type', 'drop_off_type', 'shape_dist_traveled', 'scheduled_time', 'date']
    # drop_cols = [dc for dc in drop_cols if dc in stop_times_df.columns]
    # time_point_dict = stop_times_df.drop(drop_cols, axis=1).to_dict('index')
    # with open(f'results/time_point_dict_{DATE.replace("-", "")}.pkl', 'wb') as handle:
    #     pickle.dump(time_point_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # GTFS_MONTH = 'OCT2021'
    
    CHAINS = 21
    GTFS_MONTH = 'JAN2022'
    # dates = ['2021-03-05']
    # dates = ['2021-10-18', '2021-11-23', '2021-12-15', '2022-01-27', '2022-02-25', '2022-03-26', '2022-04-02']
    # dates = ['2021-06-07', '2021-07-13', '2021-08-25', '2021-05-07']
    dates = ['2021-06-07', '2021-07-13', '2021-08-25', '2021-05-07']
    for date in tqdm(dates):
        generate_traffic_data_for_date(date, GTFS_MONTH, CHAINS)
    