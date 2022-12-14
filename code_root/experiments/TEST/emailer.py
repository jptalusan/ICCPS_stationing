import smtplib
from collections import deque
import argparse
import json
import re
import datetime as dt

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

def tail(filename, n=10):
    with open(filename) as f:
        return deque(f, n)

def send_email(config):
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security which makes the connection more secure
    smtpobj.starttls()
    senderemail_id = "jptalusan@gmail.com"
    senderemail_id_password = "bhgbzkzzwgrhnpji"
    receiveremail_id = "jptalusan@gmail.com"
    # Authentication for signing to gmail account
    smtpobj.login(senderemail_id, senderemail_id_password)
    # message to be sent
    # message = f"Finished running: {config['mcts_log_name']} on digital-storm-1."
    SUBJECT = f"DONE with {config['mcts_log_name']}"
    
    first_line = ""
    with open(f'logs/no_inject_{config["mcts_log_name"]}.log') as f:
        first_line = f.readline()
    
    result = tail(f'logs/no_inject_{config["mcts_log_name"]}.log', n=4)
    result = list(result)
    result = first_line + ("").join(result)
    in_brackets = re.findall("\[(.*?)\]", result)

    start_time = dt.datetime.strptime(in_brackets[1], DATETIME_FORMAT)
    end_time = dt.datetime.strptime(in_brackets[-1], DATETIME_FORMAT)
    
    elapsed = (end_time - start_time).total_seconds()
    
    result = result + "\n" + f"Total time: {elapsed} seconds"
    
    message = 'Subject: {}\n\n{}'.format(SUBJECT, f"Testing done on digital-storm-1:\n{result}.")
    
    # sending the mail - passing 3 arguments i.e sender address, receiver address and the message
    smtpobj.sendmail(senderemail_id, receiveremail_id, message)
    
    # Hereby terminate the session
    smtpobj.quit()
    print("mail send - Using simple text message")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_level', type=str, default='DEBUG')
    parser.add_argument('-c', '--config', type=str, default='DEBUG')
    args = parser.parse_args()
    args = namespace_to_dict(args)

    config_path = f'{args["config"]}.json'
    with open(config_path) as f:
        config = json.load(f)

    if config.get("send_mail", True):
        send_email(config)