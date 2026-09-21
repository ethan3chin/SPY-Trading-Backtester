import pandas as pd
import numpy as np

INTERVALS = ['1m', '5m', '15m', '30m', '1h', '1d']
DF = {}
for interval in INTERVALS:
    DF[interval] = pd.read_csv(f'SPY_bars/{interval}.csv', index_col='datetime')

MACD_SPANS = (26, 12, 9)
WILDER = 14
TSI_PERIODS = (25, 13)
BOLLINGER_PERIOD = 20
BOLLINGER_STDEVS = 2
MFI_WINDOW = 14

def ema(interval, n):
    df = DF[interval]
    df[f'ema_{n}'] = df['close'].ewm(span=n, ignore_na=False).mean()

def macd(interval):
    df = DF[interval]
    df['macd_long'] = df['close'].ewm(
        span=MACD_SPANS[0], 
        ignore_na=False
    ).mean()
    df['macd_short'] = df['close'].ewm(
        span=MACD_SPANS[1], 
        ignore_na=False
    ).mean()
    df['macd_line'] = df['macd_short'] - df['macd_long']
    df['signal_line'] = df['macd_line'].ewm(
        span=MACD_SPANS[2], 
        ignore_na=False
    ).mean()
    df['macd_histogram'] = df['macd_line'] - df['signal_line']
    df = df.drop(columns=['macd_long', 'macd_short'])

def atr(interval):
    df = DF[interval]
    df['diff1'] = df['high'] - df['low']
    df['diff2'] = (df['high'] - df['close'].shift(1)).abs()
    df['diff3'] = (df['low'] - df['close'].shift(1)).abs()
    df['true_range'] = df[['diff1', 'diff2', 'diff3']].max(axis=1)
    df['atr'] = df['true_range'].ewm(alpha=1/WILDER, ignore_na=False).mean()
    df = df.drop(columns=['diff1', 'diff2', 'diff3', 'true_range'])

def atr_pct(interval):
    df = DF[interval]
    df['atr%'] = df['atr'] / (df['high'] + df['low'] / 2)

def adx(interval):
    df = DF[interval]
    df['+dm'] = np.where(df['high'].diff() > 0, df['high'].diff(), 0)
    df['-dm'] = np.where(df['low'].diff() < 0, -1*df['low'].diff(), 0)
    df.loc[df['+dm'] > df['-dm'], '-dm'] = 0
    df.loc[df['-dm'] > df['+dm'], '+dm'] = 0
    df['+di'] = df['+dm'].ewm(alpha=1/WILDER, ignore_na=False).mean() / df['atr']
    df['-di'] = df['-dm'].ewm(alpha=1/WILDER, ignore_na=False).mean() / df['atr']
    df['dx'] = (df['-di'] - df['+di']).abs() / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].ewm(alpha=1/WILDER, ignore_na=False).mean()
    df = df.drop(columns=['+dm', '-dm', 'dx'])

def rsi(interval):
    df = DF[interval]
    df['gain'] = np.where(df['close'].diff() > 0, df['close'].diff(), 0)
    df['loss'] = np.where(df['close'].diff() < 0, -1*df['close'].diff(), 0)
    df['av_gain'] = df['gain'].rolling(window=WILDER).mean()
    df['av_loss'] = df['loss'].rolling(window=WILDER).mean()
    df['smooth_av_gain'] = df['av_gain'].ewm(alpha=1/WILDER, ignore_na=False).mean()
    df['smooth_av_loss'] = df['av_loss'].ewm(alpha=1/WILDER, ignore_na=False).mean()
    df['rs'] = df['smooth_av_gain'] / df['smooth_av_loss']
    df['rsi'] = 100 - (100/(1 + df['rs']))
    df = df.drop(
        columns=[
            'gain', 'loss', 'av_gain', 'av_loss', 
            'smooth_av_gain', 'smooth_av_loss', 'rs'
        ]
    )

def roc(interval, n):
    df = DF[interval]
    df['roc'] = (df['close'].pct_change(periods=n, fill_method=None)) * 100

def tsi(interval):
    df = DF[interval]
    df['pc'] = df['close'].diff()
    df['apc'] = df['pc'].abs()
    df['1s_pc'] = df['pc'].ewm(span=TSI_PERIODS[0], ignore_na=False).mean()
    df['1s_apc'] = df['apc'].ewm(span=TSI_PERIODS[0], ignore_na=False).mean()
    df['2s_pc'] = df['1s_pc'].ewm(span=TSI_PERIODS[1], ignore_na=False).mean()
    df['2s_apc'] = df['1s_apc'].ewm(span=TSI_PERIODS[1], ignore_na=False).mean()
    df['tsi'] = (df['2s_pc'] / df['2s_apc']) * 100
    df = df.drop(columns=['pc', 'apc', '1s_pc', '1s_apc', '2s_pc', '2s_apc'])

def bollinger_bands(interval):
    df = DF[interval]
    df['middle_band'] = df['close'].rolling(window=BOLLINGER_PERIOD).mean()
    df['stdev'] = df['close'].rolling(window=BOLLINGER_PERIOD).std()
    df['upper_band'] = df['middle_band'] + (df['stdev'] * BOLLINGER_STDEVS)
    df['lower_band'] = df['middle_band'] - (df['stdev'] * BOLLINGER_STDEVS)
    df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
    df = df.drop(columns='stdev')

def obv(interval):
    df = DF[interval]
    df['sign'] = np.where(df['close'].diff() > 0, 1, -1)
    df['obv_vals'] = df['volume'] * df['sign']
    if interval == '1d':
        df['obv'] = df['obv_vals'].cumsum()
    else:
        df['obv'] = df.groupby(df.index.date)['obv_vals'].cumsum()
    df = df.drop(columns=['sign', 'obv_vals'])

def vwap(interval):
    df = DF[interval]
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby(df.index.date)['pv'].cumsum()
    df['cum_volume'] = df.groupby(df.index.date)['volume'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_volume']
    df = df.drop(columns=['pv', 'cum_pv', 'cum_volume'])

def mfi(interval):
    df = DF[interval]
    df['+mf'] = np.where(df['typical_price'].diff > 1, df['pv'], 0)
    df['-mf'] = np.where(df['typical_price'].diff < 1, df['pv'], 0)
    df['+mf_sum'] = df['+mf'].rolling(window=MFI_WINDOW).sum()
    df['-mf_sum'] = df['-mf'].rolling(window=MFI_WINDOW).sum()
    df['mr'] = df['+mf_sum'] / df['-mf_sum']
    df['mfi'] = 100 - (100/(1 + df['mr']))
    df.drop(columns=['+mf', '-mf', '+mf_sum', '-mf_sum', 'mr'])

def vroc(interval, n):
    df = DF[interval]
    df['vroc'] = (df['volume'].pct_change(periods=n, fill_method=None)) * 100

def z_score(interval, n):
    df = DF[interval]
    df['mean'] = df['close'].rolling(window=n).mean()
    df['stdev'] = df['close'].rolling(window=n).std()
    df['z-score'] = (df['close'] - df['mean']) / df['stdev']
    df.drop(columns=['mean', 'stdev'])
