import datetime

import requests
import json

from typing import Optional, List
import datetime


def get_usgs(channels: Optional[List[str]] = None,
             start_datetime: datetime.datetime = None,
             end_datetime: datetime.datetime = None,
             sampling_period: int = 1) -> dict:
    if not isinstance(start_datetime, datetime.date):
        print('start_datetime must be a datetime')
        raise ValueError('start_datetime must be a datetime')

    if not isinstance(end_datetime, datetime.datetime):
        print('end_datetime must be a datetime')
        raise ValueError('end_datetime must be a datetime')

    if start_datetime >= end_datetime:
        print('Start must be before End')
        raise ValueError('Start must be before End')

    if end_datetime > datetime.datetime.now():
        print('Can not get data from the future.')
        raise ValueError('Can not get data from the future.')

    valid_channels = {'X', 'Y', 'Z', 'F'}  # there's more, but I don't know them off the top of my head
    if not all(c in valid_channels for c in channels):
        raise ValueError(f'All requested channels must be in {valid_channels}.')

    api_date_format = '%Y-%m-%dT%H:%M:%S.000Z'
    params = {
        'elements': ', '.join(channels),
        'endtime': end_datetime.strftime(api_date_format),
        'starttime': start_datetime.strftime(api_date_format),
        'format': 'json',
        'id': 'BOU',
        'sampling_period': sampling_period,
        'type': 'adjusted'
    }
    api_url = f'https://geomag.usgs.gov/ws/data/'
    res = requests.get(api_url, params)

    if res.status_code == 200:
        print('Returning USGS data.')
        return json.loads(res.text)
    else:
        print('Failed to get USGS data.')
        print(res.status_code)
        print(res.text)
        raise Exception((res.status_code, res.text))
