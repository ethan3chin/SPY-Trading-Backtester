""" Synthesizes larger-interval intraday bars using 1-minute bars, 
and stores them in csv files
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
 
SYMBOL = 'SPY.US'
INTERVALS = [('5m', '5min'), ('15m', '15min'), ('30m', '30min'), ('1h', '1h')]
FORMAT = '%Y-%m-%d %H:%M:%S'
 
OUTPUT_DIR = f"{SYMBOL.replace('.US', '')}_bars"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
AGG = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
}
 
def resample_symbol(one_min: pd.DataFrame, freq: str) -> pd.DataFrame:
    """ Resamples 1-minute bars into larger intervals and aggregates 
    the columns using the respective parameters from AGG.
    """
    pieces = []
    for _, day_df in one_min.groupby(one_min.index.date):
        day_open = day_df.index.min()
        resampled = day_df.resample(freq, origin=day_open).agg(AGG)
        pieces.append(resampled)
    out = pd.concat(pieces)
    out.loc[out['open'].isna(), 'volume'] = np.nan
    return out.reset_index().rename(columns={'index': 'datetime'})
 
 
def main():
    one_min = pd.read_csv('SPY_bars/1m.csv', index_col='datetime')
    one_min.index = pd.to_datetime(one_min.index, format=FORMAT)
    one_min = one_min.sort_index()
 
    for interval_name, freq in INTERVALS:
        df = resample_symbol(one_min, freq)
        df.to_csv(Path(os.path.join(OUTPUT_DIR, f'{interval_name}.csv')), index=False)
        print(f'{interval_name}: {len(df)} bars saved')
 
 
if __name__ == '__main__':
    main()