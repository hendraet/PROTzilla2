import pandas as pd
from datetime import datetime

def convert_time_to_datetime(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    hours_since_midnight = time_obj.hour
    return hours_since_midnight