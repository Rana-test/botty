from datetime import datetime, timedelta, time
import pandas as pd
from zoneinfo import ZoneInfo
from api_helper import get_time

def get_minute_data(api, now=None):
    """Retrieves last 30 days of 1-minute NIFTY price data from Finvasia."""
    nifty_token = "26000"
    market_open = time(9, 15)
    market_close = time(15, 30)

    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))

    def clamp_to_trading_time(dt):
        dt_time = dt.time()
        dt_date = dt.date()
        if dt_time > market_close:
            return datetime.combine(dt_date, market_close, ZoneInfo("Asia/Kolkata"))
        elif dt_time < market_open:
            prev_day = dt_date - timedelta(days=1)
            while prev_day.weekday() >= 5:
                prev_day -= timedelta(days=1)
            return datetime.combine(prev_day, market_close, ZoneInfo("Asia/Kolkata"))
        return dt.replace(second=0, microsecond=0)

    end_time = clamp_to_trading_time(now)
    start_time = end_time - timedelta(days=30)
    fmt = "%d-%m-%Y %H:%M:%S"
    start_secs = get_time(start_time.strftime(fmt))
    end_secs = get_time(end_time.strftime(fmt))

    bars = api.get_time_price_series(
        exchange='NSE', token=nifty_token,
        starttime=int(start_secs), endtime=int(end_secs), interval=1
    )

    df = pd.DataFrame(bars)
    df.rename(columns={'into': 'open', 'inth': 'high', 'intl': 'low', 'intc': 'close'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df = df[['time', 'open', 'high', 'low', 'close']]
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    df = df[(df['time'].dt.time >= market_open) & (df['time'].dt.time <= market_close)]

    return df


def get_option_chain(api, instrument_key, expiry_date):
    """Fetches option chain data using Upstox Options API."""
    response = api.get_put_call_option_chain(
        instrument_key=instrument_key,
        expiry_date=expiry_date
    )
    return response.data