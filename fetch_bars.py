""" Fetches 1-minute and daily SPY bars from EODHD API, and stores them in csv files
"""
from dotenv import load_dotenv
load_dotenv()

import os
import datetime as dt
from zoneinfo import ZoneInfo
from pathlib import Path
 
import requests
import pandas as pd

API_KEY = os.environ.get('EODHD_API_KEY')
 
SYMBOL = 'SPY.US'
INTERVAL = '1m'
EASTERN = ZoneInfo('America/New_York')

# EODHD doesn't have reliable intraday data from before 2016
INTRADAY_START_ET = dt.datetime(2016, 1, 1, 9, 30, 0, tzinfo=EASTERN)
INTRADAY_START = INTRADAY_START_ET.astimezone(dt.timezone.utc)

DAILY_START = '1993-01-29'  # The inception date of SPY

END_ET = dt.datetime.now(EASTERN)
END = END_ET.astimezone(dt.timezone.utc)

# EODHD restricts 1-minute data requests to a maximum of 120 days per call
CHUNK_DAYS = 110

MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)

OUTPUT_DIR = f'{SYMBOL.replace('.US', '')}_bars'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTRADAY_CSV = f'{INTERVAL}.csv'
INTRADAY_FILEPATH = Path(os.path.join(OUTPUT_DIR, INTRADAY_CSV))
DAILY_CSV = '1d.csv'
DAILY_FILEPATH = Path(os.path.join(OUTPUT_DIR, DAILY_CSV))
 
INTRADAY_URL = f'https://eodhd.com/api/intraday/{SYMBOL}'
DAILY_URL = f'https://eodhd.com/api/eod/{SYMBOL}'

def to_unix(d: dt.datetime) -> int:
    """ Converts a DateTime variable to its corresponding Unix timestamp.
    """
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())
 
 
def fetch_chunk(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """ Fetches 1-minute bars from EODHD API within a specified time frame.
    """
    params = {
        'api_token': API_KEY,
        'interval': INTERVAL,
        'from': to_unix(start),
        'to': to_unix(end),
        'fmt': 'json',
    }
    resp = requests.get(INTRADAY_URL, params=params, timeout=30)
    data = resp.json()
    df = pd.DataFrame(data)
    print(f'Fetching chunk from {start} to {end}')
    return df
 
def filter_hours(df: pd.DataFrame) -> pd.DataFrame:
    """ Removes extended-hours bars from the DataFrame.
    """
    is_weekday = df['datetime'].dt.weekday < 5
    is_regular_hours = (df['datetime'].dt.time >= MARKET_OPEN) & (
        df['datetime'].dt.time < MARKET_CLOSE)
    filtered = df[is_weekday * is_regular_hours].copy()
    return filtered

def fill_empty_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """ Adds rows for minutes that are empty and fills NaN into each column.
    """
    filled_pieces = []
    for day, day_df in df.groupby(df.index.date):
        full_day_index = pd.date_range(
            start=pd.Timestamp(day, tz=df.index.tz).replace(hour=9, minute=30),
            end=pd.Timestamp(day, tz=df.index.tz).replace(hour=15, minute=59),
            freq='1min'
        )
        filled_pieces.append(day_df.reindex(full_day_index))
    return pd.concat(filled_pieces)

def save_intraday():
    """ Saves the cleaned 1-minute bars DataFrame to a csv file.
    """
    all_chunks = []
    chunk_start = INTRADAY_START
 
    while chunk_start < END:
        chunk_end = min(chunk_start + dt.timedelta(
                days=CHUNK_DAYS-1, 
                hours=23, 
                minutes=59
            ), END)

        df_chunk = fetch_chunk(chunk_start, chunk_end)
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
 
        chunk_start = chunk_end + dt.timedelta(minutes=1)
 
    df = pd.concat(all_chunks, ignore_index=True)
 
    cols_order = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = df[cols_order]

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(EASTERN)
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    df = filter_hours(df)

    df = df.set_index('datetime')
    df = fill_empty_minutes(df)
    df.index.name = 'datetime'

    df.to_csv(INTRADAY_FILEPATH)

def save_daily():
    """ Fetches and saves daily bars to a csv file.
    """
    start = DAILY_START
    end = END.strftime('%Y-%m-%d')

    params = {
            'api_token': API_KEY,
            'period': 'd',
            'from': start,
            'to': end,
            'fmt': 'json',
        }
    resp = requests.get(DAILY_URL, params=params, timeout=30)
    data = resp.json()
    df = pd.DataFrame(data)

    df.to_csv(DAILY_FILEPATH, index=False)
 
 
if __name__ == '__main__':
    save_intraday()
    save_daily()