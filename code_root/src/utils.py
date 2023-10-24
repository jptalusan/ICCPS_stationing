import argparse
import datetime as dt
from Environment.enums import LogType

GMT5 = 18000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
PASSENGER_TIME_TO_LEAVE = 30  # minutes
DECISION_INTERVAL = 15  # minutes
EARLY_PASSENGER_DELTA_MIN = 1
VEHICLE_CAPACITY = 40
OVERAGE_THRESHOLD = 0.05


def convert_pandas_dow_to_pyspark(pandas_dow):
    return (pandas_dow + 1) % 7 + 1


def namespace_to_dict(namespace):
    return {k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v for k, v in vars(namespace).items()}


def str_timestamp_to_datetime(timestamp_str):
    return dt.datetime.strptime(timestamp_str, DATETIME_FORMAT)


def str_timestamp_to_seconds(timestamp_str):
    return dt.datetime.strptime(timestamp_str, DATETIME_FORMAT).timestamp()


def seconds_epoch_to_str(timestamp_seconds):
    return dt.datetime.fromtimestamp(timestamp_seconds).strftime(DATETIME_FORMAT)


def datetime_to_str(_datetime):
    # return _datetime.strftime(DATETIME_FORMAT)
    return _datetime.strftime("%H:%M:%S")


def time_since_midnight_in_seconds(datetime_time):
    # t = dt.time(10, 10, 35)
    t = datetime_time
    td = dt.datetime.combine(dt.datetime.min, t) - dt.datetime.min
    seconds = td.total_seconds()  # Python 2.7+
    return seconds


def log(logger, curr_time=None, message=None, type=LogType.DEBUG):
    if logger is None:
        return

    if type == LogType.DEBUG:
        # self.logger.debug(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        if curr_time:
            logger.debug(f"[{datetime_to_str(curr_time)}] {message}")
        else:
            logger.debug(f"{message}")
    if type == LogType.ERROR:
        # self.logger.error(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        if curr_time:
            logger.error(f"[{datetime_to_str(curr_time)}] {message}")
        else:
            logger.debug(f"{message}")
    if type == LogType.INFO:
        # self.logger.info(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        if curr_time:
            logger.info(f"[{datetime_to_str(curr_time)}] {message}")
        else:
            logger.debug(f"{message}")


def get_tod(timestamp):
    h = timestamp.hour
    if h < 6:
        return "early_am"
    elif h >= 6 and h < 9:
        return "rush_am"
    elif h >= 9 and h < 13:
        return "mid_am"
    elif h >= 13 and h < 17:
        return "mid_pm"
    elif h >= 17 and h < 19:
        return "rush_pm"
    elif h >= 20 and h < 24:
        return "night"
    else:
        return None


import os
import argparse
import tarfile
import json

# Import smtplib for the actual sending function.
import smtplib
from datetime import datetime

# Here are the email package modules we'll need.
from email.message import EmailMessage


def emailer(config_path):
    # Create the container email message.
    msg = EmailMessage()
    msg["Subject"] = "MCTS Stationing finished"
    me = "jptalusan@gmail.com"
    recipients = ["jptalusan@gmail.com"]
    msg["From"] = me
    msg["To"] = ", ".join(recipients)
    msg.preamble = "You will not see this in a MIME-aware mail reader.\n"

    # Open the files in binary mode.  You can also omit the subtype
    # if you want MIMEImage to guess it.

    now = datetime.now()
    log_name = now.strftime("%Y-%m-%d")

    with open(config_path) as f:
        config = json.load(f)

    output_tar_file = f"MCTS {config.get('starting_date_str', log_name)}.tar.gz"

    log_path = f"./results/{config['starting_date_str']}_{config['mcts_log_name']}/results.csv"

    with tarfile.open(output_tar_file, "w:gz") as tar:
        tar.add(log_path, arcname=os.path.basename(log_path))
        tar.add(config_path, arcname=os.path.basename(config_path))

    for file in [output_tar_file]:
        filename = os.path.basename(os.path.normpath(file))
        with open(file, "rb") as fp:
            img_data = fp.read()
        msg.add_attachment(img_data, maintype="application", subtype="tar+gzip", filename=filename)

    # Send the email via our own SMTP server.
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login("jptalusan@gmail.com", "gqixjuljscezarle")
        s.send_message(msg)
        s.quit()
