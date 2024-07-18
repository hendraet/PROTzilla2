import pandas as pd
from datetime import datetime

def convert_time_to_datetime(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    seconds_since_midnight = time_obj.second + time_obj.minute * 60 + time_obj.hour * 3600
    return seconds_since_midnight
