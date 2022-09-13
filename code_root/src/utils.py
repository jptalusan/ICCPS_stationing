import argparse
import datetime as dt

GMT5 = 18000
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

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
    
def str_timestamp_to_seconds(timestamp_str):
    return dt.datetime.strptime(timestamp_str, DATETIME_FORMAT).timestamp()
def seconds_epoch_to_str(timestamp_seconds):
    return dt.datetime.fromtimestamp(timestamp_seconds).strftime(DATETIME_FORMAT)