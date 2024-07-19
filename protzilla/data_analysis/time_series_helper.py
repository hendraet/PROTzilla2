import pandas as pd
from datetime import datetime

def convert_time_to_datetime(time_str):
    """
    Convert a string time to a datetime object
    :param time_str: The time string to convert

    :return: A datetime object
    """
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    hours_since_midnight = time_obj.hour
    return hours_since_midnight