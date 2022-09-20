import argparse
import datetime as dt
    
from Environment.enums import LogType

GMT5 = 18000
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
PASSENGER_TIME_TO_LEAVE = 30 #minutes
EARLY_PASSENGER_DELTA_MIN = 1

def convert_pandas_dow_to_pyspark(pandas_dow):
    return (pandas_dow + 1) % 7 + 1

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }
   
def str_timestamp_to_datetime(timestamp_str):
    return dt.datetime.strptime(timestamp_str, DATETIME_FORMAT)
def str_timestamp_to_seconds(timestamp_str):
    return dt.datetime.strptime(timestamp_str, DATETIME_FORMAT).timestamp()
def seconds_epoch_to_str(timestamp_seconds):
    return dt.datetime.fromtimestamp(timestamp_seconds).strftime(DATETIME_FORMAT)
def datetime_to_str(_datetime):
    return _datetime.strftime(DATETIME_FORMAT)

def time_since_midnight_in_seconds(datetime_time):
    # t = dt.time(10, 10, 35)
    t = datetime_time
    td = dt.datetime.combine(dt.datetime.min, t) - dt. datetime.min
    seconds = td.total_seconds() # Python 2.7+
    return seconds

def log(logger, curr_time, message, type=LogType.DEBUG):
    if type == LogType.DEBUG:
        # self.logger.debug(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        logger.debug(f"[{datetime_to_str(curr_time)}] {message}")
    if type == LogType.ERROR:
        # self.logger.error(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        logger.error(f"[{datetime_to_str(curr_time)}] {message}")
    if type == LogType.INFO:
        # self.logger.info(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        logger.info(f"[{datetime_to_str(curr_time)}] {message}")