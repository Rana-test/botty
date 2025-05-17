
import numpy as np
import pandas as pd


def calculate_rsi_wilder(close_prices, period=14):
    """Calculates RSI using Wilder's smoothing method."""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_supertrend(df_minute, atr_period=10, multiplier=4):
    """
    Calculates Supertrend based on given ATR period and multiplier.
    Returns DataFrame with trend and entry/exit signals.
    """
    df = df_minute.copy()
    # Debug
    df_minute.to_csv('minute_data.csv', index=False)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    ############################## START HOURLY #################
    # # Aggregate to hourly candles
    # df_hourly = df.resample('1h', label='right', closed='right').agg({
    #     'open': 'first',
    #     'high': 'max',
    #     'low': 'min',
    #     'close': 'last',
    #     # 'volume': 'sum'
    # }).dropna()

    ########################## END HOURLY #####################
    ########################## START MIN #####################
    # Sort descending (latest to earliest)
    df = df.sort_index(ascending=False)

    df['minutes_from_latest'] = ((df.index[0] - df.index).total_seconds() // 60).astype(int)
    df['reverse_hour_bin'] = (df['minutes_from_latest'] // 60).astype(int)

    ohlc_agg = {
        'open': 'last',
        'high': 'max',
        'low': 'min',
        'close': 'first'
    }
    df_hourly = df.groupby('reverse_hour_bin').agg(ohlc_agg)
    df_hourly['time'] = df.groupby('reverse_hour_bin').apply(lambda x: x.index[0])
    df_hourly.set_index('time', inplace=True)
    df_hourly.sort_index(inplace=True, ascending=True)
    ###########################END MIN#########################

    df_hourly['tr'] = pd.concat([
        df_hourly['high'] - df_hourly['low'],
        (df_hourly['high'] - df_hourly['close'].shift(1)).abs(),
        (df_hourly['low'] - df_hourly['close'].shift(1)).abs()
    ], axis=1).max(axis=1)

    df_hourly['atr'] = df_hourly['tr'].ewm(alpha=1/atr_period, adjust=False).mean()

    up_list, dn_list, trend_list = [np.nan]*len(df_hourly), [np.nan]*len(df_hourly), [np.nan]*len(df_hourly)

    for i in range(len(df_hourly)):
        src = df_hourly['close'].iloc[i]
        if i == 0:
            up = src - multiplier * df_hourly['atr'].iloc[i]
            dn = src + multiplier * df_hourly['atr'].iloc[i]
            trend = 1
        else:
            up_temp = src - multiplier * df_hourly['atr'].iloc[i]
            dn_temp = src + multiplier * df_hourly['atr'].iloc[i]
            up1, dn1 = up_list[i-1], dn_list[i-1]
            prev_close = df_hourly['close'].iloc[i-1]

            up = max(up_temp, up1) if prev_close > up1 else up_temp
            dn = min(dn_temp, dn1) if prev_close < dn1 else dn_temp

            if trend_list[i-1] == -1 and src > dn1:
                trend = 1
            elif trend_list[i-1] == 1 and src < up1:
                trend = -1
            else:
                trend = trend_list[i-1]

        up_list[i], dn_list[i], trend_list[i] = up, dn, trend

    df_hourly['up'], df_hourly['dn'], df_hourly['trend'] = up_list, dn_list, trend_list
    df_hourly['ema20'] = df_hourly['close'].ewm(span=20, adjust=False).mean()
    df_hourly['ema34'] = df_hourly['close'].ewm(span=34, adjust=False).mean()
    df_hourly['rsi'] = calculate_rsi_wilder(df_hourly['close'])

    df_hourly['entry_signal'] = 0
    df_hourly.loc[
        (df_hourly['close'] < df_hourly['ema20']) &
        (df_hourly['close'] < df_hourly['ema34']) &
        (df_hourly['ema20'] < df_hourly['ema34']) &
        (df_hourly['trend'] == -1) &
        (df_hourly['rsi'] < 45),
        'entry_signal'
    ] = 1

    df_hourly.loc[
        (df_hourly['close'] > df_hourly['ema20']) &
        (df_hourly['close'] > df_hourly['ema34']) &
        (df_hourly['ema20'] > df_hourly['ema34']) &
        (df_hourly['trend'] == 1) &
        (df_hourly['rsi'] > 55),
        'entry_signal'
    ] = -1

    df_hourly['exit_signal'] = 0
    df_hourly.loc[
        (df_hourly['trend'] != df_hourly['trend'].shift(1)),
          'exit_signal'
          ] = 1
    #Debug
    df_hourly.to_csv('hourly_data.csv')

    return df_hourly
