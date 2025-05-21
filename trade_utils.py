import pandas as pd
import os
import math
import logging
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from api_helper import get_time
import time as sleep_time
import calendar
import numpy as np

logger = logging.getLogger(__name__)

holiday_dict ={
    '2025-05-01':'2025-04-30',
    '2025-10-02':'2025-10-01',
    '2025-12-25':'2025-12-24',
}

month_mapping = {
    '1': 'JAN', '2': 'FEB', '3': 'MAR', '4': 'APR', '5': 'MAY', '6': 'JUN',
    '7': 'JUL', '8': 'AUG', '9': 'SEP', 'O': 'OCT', 'N': 'NOV', 'D': 'DEC'
}

def calculate_total_pnl(df):
    """Calculates total and per-symbol PnL from a trade DataFrame."""
    df['flqty'] = df['flqty'].astype(float)
    df['flprc'] = df['flprc'].astype(float)
    pnl_by_symbol = {}

    for tsym, group in df.groupby('tsym'):
        buy_qty = group[group['trantype'] == 'B']['flqty'].sum()
        sell_qty = group[group['trantype'] == 'S']['flqty'].sum()
        buy_amt = (group[group['trantype'] == 'B']['flqty'] * group[group['trantype'] == 'B']['flprc']).sum()
        sell_amt = (group[group['trantype'] == 'S']['flqty'] * group[group['trantype'] == 'S']['flprc']).sum()

        matched_qty = min(buy_qty, sell_qty)
        avg_buy = buy_amt / buy_qty if buy_qty else 0
        avg_sell = sell_amt / sell_qty if sell_qty else 0
        pnl_by_symbol[tsym] = (avg_sell - avg_buy) * matched_qty

    total_pnl = sum(pnl_by_symbol.values())
    return total_pnl

def write_to_trade_book(api, trade_csv="trade_book.csv"):
    """Fetch and update trade book records from the API."""
    cols = ['trantype', 'tsym', 'order_type', 'qty', 'fillshares', 'flqty', 'flprc', 'avgprc', 'exch_tm', 'remarks', 'exchordid']
    df_existing = pd.read_csv(trade_csv, dtype=str) if os.path.exists(trade_csv) else pd.DataFrame(columns=cols)

    new_rec_df = pd.DataFrame(api.get_trade_book())
    if not new_rec_df.empty:
        new_rec_df['order_type'] = new_rec_df['tsym'].str.extract(r'([PC])\d+$')[0].map({'P': 'PE', 'C': 'CE'})
        new_rec_df = new_rec_df[cols].fillna("").astype(str)
        df_existing = df_existing.fillna("").astype(str)

        exists = (df_existing['exchordid'] == new_rec_df['exchordid'].iloc[0]).any()
        if not exists:
            df_existing = pd.concat([df_existing, new_rec_df], ignore_index=True)
            df_existing.to_csv(trade_csv, index=False, float_format="%.2f")
    
    # Update trade book if expiry date is in the past
    # Convert exchange time to datetime
    df_existing['exch_tm'] = pd.to_datetime(df_existing['exch_tm'], format='%d-%m-%Y %H:%M:%S')
    # Step 1: Extract expiry from tsym (e.g., '08MAY25' from 'NIFTY08MAY25P24200')
    def extract_expiry(tsym):
        try:
            date_str = tsym[6:13]
            return pd.to_datetime(date_str, format='%d%b%y')
        except:
            return pd.NaT
    df_existing['expiry'] = df_existing['tsym'].apply(extract_expiry)
    # Step 2: Filter for expired contracts
    today = pd.Timestamp.today().normalize()
    expired_df = df_existing[df_existing['expiry'] < today]
    # Step 3: Group by symbol and find quantity mismatches
    grouped = expired_df.groupby(['tsym', 'trantype'])['flqty'].sum().unstack(fill_value=0)

    # Step 4: Identify mismatches
    mismatch_rows = []
    for tsym, row in grouped.iterrows():
        buy_qty = row.get('B', 0)
        sell_qty = row.get('S', 0)
        if buy_qty != sell_qty:
            missing_type = 'B' if sell_qty > buy_qty else 'S'
            missing_qty = abs(sell_qty - buy_qty)
            mismatch_row = {
                'trantype': missing_type,
                'tsym': tsym,
                'order_type': tsym[-2:],  # get 'PE' or 'CE' from symbol
                'qty': missing_qty,
                'fillshares': missing_qty,
                'flqty': missing_qty,
                'flprc': 0.05,
                'avgprc': 0.05,
                'exch_tm': datetime.now(),
                'remarks': 'Mismatch',
                'exchordid': 'mismatchordid',
            }
            mismatch_rows.append(mismatch_row)

    # Step 5: Append mismatches
    if mismatch_rows:
        mismatch_df = pd.DataFrame(mismatch_rows)
        df_existing = pd.concat([df_existing, mismatch_df], ignore_index=True)

    total_pnl = calculate_total_pnl(df_existing)
    return df_existing, total_pnl

def fixed_ratio_position_size(base_size, delta, total_profit):
    """
    Dynamically adjusts lot size based on profit/loss with a fixed ratio method.
    Ensures a minimum of 1 lot.
    """
    if total_profit >= 0:
        adjustment = math.floor(math.sqrt(total_profit / delta))
    else:
        adjustment = -math.floor(math.sqrt(abs(total_profit) / delta))

    return max(1, base_size + adjustment)

def get_minute_data(api, now=None):
    nifty_token = '26000'  # NSE|26000 is the Nifty 50 index
    
    # Define trading hours
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    # Set current time if not provided
    if now is None:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    
    # Adjust latest_time to the most recent trading minute
    def adjust_to_trading_hours(dt):
        dt = dt.astimezone(ZoneInfo("Asia/Kolkata")) 
        dt_time = dt.time()
        dt_date = dt.date()
        
        if dt_time > market_close:
            # After market close, use 15:30:00 of the same day
            return datetime.combine(dt_date, market_close, ZoneInfo("Asia/Kolkata"))
        elif dt_time < market_open:
            # Before market open, use 15:30:00 of the previous trading day
            prev_day = dt_date - timedelta(days=1)
            # Check if previous day is a weekday (Monday to Friday)
            while prev_day.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
                prev_day -= timedelta(days=1)
            return datetime.combine(prev_day, market_close, ZoneInfo("Asia/Kolkata"))
        else:
            # Within trading hours, round down to the nearest minute
            return dt.replace(second=0, microsecond=0)
    
    latest_time = adjust_to_trading_hours(now)
    
    # Time 90 days ago
    start_time = latest_time - timedelta(days=30)
    
    # Desired time format
    fmt = "%d-%m-%Y %H:%M:%S"
    
    # Convert times to seconds (assuming get_time is defined elsewhere)
    start_secs = get_time(start_time.strftime(fmt))  # dd-mm-YYYY HH:MM:SS
    end_secs = get_time(latest_time.strftime(fmt))
    
    # Fetch 1-minute candle data from Finvasia API
    bars = api.get_time_price_series(
        exchange='NSE',
        token=nifty_token,
        starttime=int(start_secs),
        endtime=int(end_secs),
        interval=1  # 1-minute candles
    )
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(bars)
    df.rename(columns={
        'into': 'open',
        'inth': 'high',
        'intl': 'low',
        'intc': 'close'
    }, inplace=True)
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    
    # Select and convert columns to float
    df = df[['time', 'open', 'high', 'low', 'close']]
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    
    # Filter data to ensure it's within trading hours (9:15:00 to 15:30:00)
    df = df[(df['time'].dt.time >= market_open) & (df['time'].dt.time <= market_close)]
    # df = df.reset_index()

    return df

def exit_trade(finvasia_api, exit_orders, live=True):
    return_msgs=[]
    action = "EXIT POSITIONS"
    logging.info(f"Checking open order when trend changed")
    for _, order in exit_orders.iterrows():
        order_tsm = order['tsym']
        # order_type = order['dname'].split()[-1]
        # # ord_qty = min(abs(int(curr_pos[curr_pos['tsym']==order_tsm]['netqty'].iloc[0])),order['order_qty'])
        qty = int(order['netqty'])
        ord_act = 'S' if qty > 0 else 'B'
        qty = abs(qty)
        logging.info(f"Closing {order['order_type']} order {order_tsm}: {order}")
        # Get the current qty as per exisitng position and limit qty to that 
        ret_status, ret_msg = place_order(finvasia_api, live, order_tsm, ord_act, str(qty), 'EXIT STEMA')
        return_msgs.append(ret_msg)
        
    sleep_time.sleep(10)
    write_to_trade_book(finvasia_api)
    return return_msgs

def place_order(api, live, trading_symbol, buy_sell, qty, order_type):
    logging.info(f"Within place order")
    quantity = str(abs(int(qty)))
    tradingsymbol= trading_symbol
    prd_type = 'M'
    exchange = 'NFO' 
    # disclosed_qty= 0
    price_type = 'MKT'
    price=0
    trigger_price = None
    retention='DAY'
    email_body = ''
    if live:
        logging.info(f"Placing order: {trading_symbol}, {buy_sell}, {qty}, {order_type}")
        response = api.place_order(buy_or_sell=buy_sell, product_type=prd_type, exchange=exchange, tradingsymbol=tradingsymbol, quantity=quantity, discloseqty=quantity,price_type=price_type, price=price,trigger_price=trigger_price, retention=retention, remarks=order_type)
        if response is None or 'norenordno' not in response:
            logging.info(f"None Response")
            return False, {'subject': "Order Placement Failed", 'body': "Order Placement Failed"}
        order_id = response['norenordno']
        logging.info(f"Order_id: {order_id}")
        email_body = f"Order placed successfully : Order No: {order_id}/n"
        email_body += f'buy_or_sell={buy_sell}, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks={order_type}'
        for _ in range(10):  
            logging.info(f"Waiting for order execution confirmation")# Try for ~10 seconds
            sleep_time.sleep(1)
            orders = api.get_order_book()
            if orders:
                matching_orders = [o for o in orders if o['norenordno'] == order_id]
                if matching_orders:
                    order = matching_orders[0]
                    logging.info(f"Matching Order: {order}")
                    status = order['status']
                    logging.info(f"Order response: {status}")
                    if status == 'COMPLETE':
                        # subject = f"Order executed successfully.: Order No: {order_id}"
                        email_body+=f"Order executed successfully.: Order No: {order_id}"
                        return True, {'subject': "Order executed successfully", 'body': email_body}
                    elif status in ['REJECTED', 'CANCELLED']:
                        email_body+=f"Order {status}. Reason: {order.get('rejreason', 'Not available')}"
                        return False, {'subject': f"ORDER REJECTED : Reason: {order.get('rejreason', 'Not available')}", 'body': email_body}
            else:
                email_body = email_body+ f"Could not fetch order book./n"

        email_body = email_body+ "Timed out waiting for order update."
        logging.info(f"Order Execution Timed out")
        return True, {'subject': "Order Timed out", 'body': email_body}

    else:
        print(f'buy_or_sell={buy_sell}, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks={order_type}')
        subject = f"{order_type} order for {tradingsymbol}"
        email_body = f'buy_or_sell={buy_sell}, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks={order_type}'
        # send_email_plain(subject, email_body)
        logging.info(f"Dummy order placed: {email_body}")
        return True, {'subject': subject, 'body': email_body}

def get_last_thursday(year, month):
    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    date = datetime(year, month, last_day)

    # Go backwards to find the last Thursday
    while date.weekday() != 3:  # 3 corresponds to Thursday
        date -= timedelta(days=1)

    return date

def get_target_thursday():
    today = datetime.now()
    this_month_thursday = get_last_thursday(today.year, today.month)
    days_until = (this_month_thursday - today).days

    if days_until > 11:
        return this_month_thursday.strftime("%Y-%m-%d")
    else:
        # Move to next month
        year = today.year + (1 if today.month == 12 else 0)
        month = 1 if today.month == 12 else today.month + 1
        return get_last_thursday(year, month).strftime("%Y-%m-%d")

def find_last_thursday(year, month):
    """Finds the last Thursday of a given month and year."""
    last_day = datetime(year, month, 1).replace(day=calendar.monthrange(year, month)[1])
    while last_day.weekday() != 3:  # Thursday is weekday 3
        last_day -= timedelta(days=1)
    return last_day.day

def get_next_week_thursday(today=None):
    if today is None:
        today = datetime.now(ZoneInfo('Asia/Kolkata'))
    
    # Find start of current week (Monday)
    start_of_week = today - timedelta(days=today.weekday())
    
    # Start of next week
    start_of_next_week = start_of_week + timedelta(days=7)
    
    # Next week's Thursday
    next_week_thursday = start_of_next_week + timedelta(days=3)
    
    return next_week_thursday.strftime("%Y-%m-%d")


def convert_option_symbol(input_symbol):
    # Ensure input is valid
    if not isinstance(input_symbol, str) or len(input_symbol) < 12:
        raise ValueError("Invalid symbol format. Expected format: NIFTY2530622300PE or NIFTY25FEB22450PE")

    # Extract the underlying index (e.g., NIFTY)
    index = input_symbol[:5]

    # Extract the year
    year_prefix = input_symbol[5:7]

    # Determine if it's a weekly or monthly expiry
    remaining_part = input_symbol[7:-2]  # Excludes index, year, and option type

    # Check if the month part is in the three-letter format (e.g., FEB) for monthly options
    if remaining_part[:3].isalpha():
        # Monthly expiry format: NIFTY25FEB22450PE
        month_abbr = remaining_part[:3].upper()
        strike_price = remaining_part[3:]
        expiry_day = None  # Monthly expiry day to be determined separately
    else:
        # Weekly expiry format: NIFTY2530622300PE (where "306" means 6th March)
        month_code = remaining_part[0]  # Could be a single letter (O, N, D) or a digit (1-9)
        day = remaining_part[1:3]  # Extract the day (e.g., '06')
        strike_price = remaining_part[3:]

        # Convert month code to full month abbreviation
        if month_code in month_mapping:
            month_abbr = month_mapping[month_code]
        else:
            raise ValueError(f"Invalid month code: {month_code}")

        expiry_day = day  # Weekly expiry has an explicit day

    # Determine the final expiry format
    if expiry_day:
        # Weekly expiry format: NIFTY06MAR25P22300
        new_symbol = f"{index}{expiry_day}{month_abbr}{year_prefix}{'P' if input_symbol[-2:] == 'PE' else 'C'}{strike_price}"
    else:
        # Monthly expiry format: Find the last Thursday of the month
        expiry_year = int(f"20{year_prefix}")
        expiry_month = datetime.strptime(month_abbr, "%b").month
        expiry_day = find_last_thursday(expiry_year, expiry_month)  # Function to get last Thursday

        new_symbol = f"{index}{expiry_day}{month_abbr}{year_prefix}{'P' if input_symbol[-2:] == 'PE' else 'C'}{strike_price}"

    return new_symbol

def get_nearest_delta_options(option_chain_data, upstox_instruments, delta):
    call_symbol = None
    put_symbol = None
    min_call_diff = float("inf")
    min_put_diff = float("inf")

    for option in option_chain_data:
        # Access call and put options using dot notation
        call_option = option.call_options
        put_option = option.put_options

        # Process call options
        if call_option and call_option.option_greeks:
            call_symb = upstox_instruments[upstox_instruments['instrument_key']==call_option.instrument_key].tradingsymbol.values[0]
            call_delta = call_option.option_greeks.delta
            call_oi = float(call_option.market_data.oi)
            if call_delta is not None and abs(call_delta - delta) < min_call_diff and call_symb[-4:-2] == '00' and call_oi>10000:
                min_call_diff = abs(call_delta - delta)
                call_symbol = call_option.instrument_key
                upstox_ce_ltp = call_option.market_data.ltp
                co = call_option

        # Process put options
        if put_option and put_option.option_greeks:
            put_symb = upstox_instruments[upstox_instruments['instrument_key']==put_option.instrument_key].tradingsymbol.values[0]
            put_delta = put_option.option_greeks.delta
            put_oi = float(put_option.market_data.oi) if put_option.market_data.oi is not None else 0
            if put_delta is not None and abs(put_delta + delta) < min_put_diff and put_symb[-4:-2] == '00' and put_oi>10000:
                min_put_diff = abs(put_delta + delta)
                put_symbol = put_option.instrument_key
                upstox_pe_ltp = put_option.market_data.ltp
                po = put_option

    call_bid = float(co.market_data.bid_price)
    call_ask = float(co.market_data.ask_price)
    call_bid_ask = call_ask-call_bid
    put_bid = float(po.market_data.bid_price)
    put_ask = float(po.market_data.ask_price)
    put_bid_ask = put_ask-put_bid
    return call_symbol, put_symbol, upstox_ce_ltp, upstox_pe_ltp, po, co , co.market_data.oi, po.market_data.oi, co.option_greeks.delta, po.option_greeks.delta, call_bid_ask, put_bid_ask
    
def get_positions(api):
    try:
        pos_df = pd.DataFrame(api.get_positions())
        pos_df = pos_df[(~pos_df['dname'].isna())]
        # pos_df = pos_df[(pos_df['netqty']!="0")] # Not needed
        pos_df["PnL"] = -1 * (pos_df["netupldprc"].astype(float) - pos_df["lp"].astype(float)) * pos_df["netqty"].astype(float)
        pos_df["totsellamt"] = pos_df["totsellamt"].astype(float)
        pos_df["netqty"] = pos_df["netqty"].astype(int)
        pos_df['type'] = pos_df['dname'].apply(lambda x: x.split()[3])
        pos_df['sp'] = pos_df['dname'].apply(lambda x: x.split()[2])
        pos_df['expiry'] = pos_df['dname'].apply(lambda x: x.split()[1])  # Extract expiry date
        pos_df['expiry'] = pd.to_datetime(pos_df['expiry'], format="%d%b%y")
        current_date = pd.Timestamp.today().normalize()
        pos_df['Days_to_Expiry'] = pos_df['expiry'].apply(lambda x: (x - current_date).days)
        # pos_df['exit_breakeven_per']= pos_df.apply(lambda x: exit_params[x['Days_to_Expiry']]['distance_from_breakeven'],axis=1)
        pos_df['exit_breakeven_per']="0"
        # pos_df['exit_loss_per']= pos_df.apply(lambda x: exit_params[x['Days_to_Expiry']]['loss_multiple'],axis=1)
        pos_df['exit_loss_per']=0.5
        return pos_df
    except Exception as e:
        return None

def get_revised_qty_margin(orders, upstox_charge_api, max_margin_available):
    main_leg = orders['Main'] #={'trading_symbol':main_leg['fin_pe_symbol'], , 'trading_up_symbol':main_leg['upstox_pe_instrument_key'], 'order_action':'S', 'order_qty':str(trade_qty), 'order_type':'PUT'}
    hedge_leg = orders['Hedge'] #={'trading_symbol':hedge_leg['fin_pe_symbol'], , 'trading_up_symbol':main_leg['upstox_pe_instrument_key'], 'order_action':'B', 'order_qty':str(trade_qty), 'order_type':'PUT'}
    instruments = [
    {
        "instrument_key": main_leg['trading_up_symbol'],  # Replace with actual instrument key
        "quantity": main_leg['order_qty'],  # Quantity in lots
        "transaction_type": "SELL" if main_leg['order_action']=="S" else "BUY",
        "product": "D",  # 'D' for Delivery, 'I' for Intraday
        "price": 0  # Market price; set to 0 for market orders
    },
    {
        "instrument_key": hedge_leg['trading_up_symbol'],  # Replace with actual instrument key
        "quantity": hedge_leg['order_qty'],  # Quantity in lots
        "transaction_type": "SELL" if hedge_leg['order_action']=="S" else "BUY",
        "product": "D",  # 'D' for Delivery, 'I' for Intraday
        "price": 0  # Market price; set to 0 for market orders
    },
    ]
    margin_request = {"instruments": instruments}
    api_response = upstox_charge_api.post_margin(margin_request)
    final_margin = float(api_response.data.final_margin)
    if max_margin_available<final_margin:
        orders['Main']['order_qty']=0
        orders['Hedge']['order_qty']=0
        return orders, 99999999, 0
    else:
        margin_per_lot = 75*final_margin/float(main_leg['order_qty'])
        lots = max(0,max_margin_available//margin_per_lot)
        orders['Main']['order_qty']=lots*75
        orders['Hedge']['order_qty']=lots*75
        return orders, margin_per_lot, lots*75

def get_atm_iv(upstox_opt_api, expiry_date, current_index_price):
    strike_interval = 50
    remainder = math.fmod(current_index_price, strike_interval)
    if remainder > strike_interval / 2:
        atm_strike = math.ceil(current_index_price / strike_interval) * strike_interval
    else:
        atm_strike = math.floor(current_index_price / strike_interval) * strike_interval

    api_response = upstox_opt_api.get_put_call_option_chain(instrument_key="NSE_INDEX|Nifty 50", expiry_date=expiry_date)
    for sp in api_response.data:
        if sp.strike_price == atm_strike:
            return((float(sp.call_options.option_greeks.iv)+float(sp.put_options.option_greeks.iv))/2)
    return None   

def calc_expected_move(index_price: float, vix: float, days: int) -> float:
    daily_volatility = (vix/100) / np.sqrt(365)  # Convert annualized VIX to daily volatility
    expected_move = index_price * daily_volatility * np.sqrt(days)
    return expected_move

def sigmoid_exit_percent(dte, k=0.7):
    if dte < 0:
        return 100
    elif dte > 10:
        return 70
    else:
        return 65 +  70/ (1 + math.exp(k * dte))
    
def exit_order(pos_df, api, order_type, live=False):
    return_msgs=[]
    for i,pos in pos_df.iterrows():
        # Exit only the loss making side
        # if pos["PnL"] < 1: 
        #Exit complete leg
        trading_symbol = pos["tsym"]  # Trading symbol of the position
        netqty = int(pos["netqty"])  # Net quantity of the position
        if netqty != 0:  # Ensure position exists
            # transaction_type = "BUY" if netqty < 0 else "SELL"
            quantity = abs(netqty)  # Exit full position
            prd_type = 'M'
            exchange = 'NFO' 
            # disclosed_qty= 0
            price_type = 'MKT'
            price=0
            trigger_price = None
            retention='DAY'
            buy_sell = 'S' if netqty>0 else 'B'
            status, return_msg = place_order(api, live, trading_symbol, buy_sell, abs(netqty), order_type)
            return_msgs+= return_msg
    return return_msgs

def get_strikes(upstox_opt_api, finvasia_api, instrument, expiry,trade_qty,upstox_instruments, delta, finvasia_user_id):
    SPAN_Expiry = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d-%b-%Y").upper()
    trade_details={}
    option_chain = upstox_opt_api.get_put_call_option_chain(instrument_key=instrument,expiry_date=expiry).data
    upstox_ce_instrument_key, upstox_pe_instrument_key, upstox_ce_ltp, upstox_pe_ltp,  po, co, call_oi, put_oi, call_delta, put_delta, call_bid_ask, put_bid_ask = get_nearest_delta_options(option_chain,upstox_instruments,delta)
    trade_details['upstox_ce_instrument_key']=upstox_ce_instrument_key
    trade_details['upstox_pe_instrument_key']=upstox_pe_instrument_key
    trade_details['call_oi']=call_oi
    trade_details['put_oi']=put_oi
    trade_details['call_delta']=call_delta*100
    trade_details['put_delta']=put_delta*100
    trade_details['call_bid_ask']=call_bid_ask
    trade_details['put_bid_ask']=put_bid_ask
    upstox_ce_symbol = upstox_instruments[upstox_instruments['instrument_key']==upstox_ce_instrument_key]['tradingsymbol'].values[0]
    # trade_details['upstox_ce'] = upstox_ce_symbol
    upstox_pe_symbol = upstox_instruments[upstox_instruments['instrument_key']==upstox_pe_instrument_key]['tradingsymbol'].values[0]
    # trade_details['upstox_pe'] = upstox_pe_symbol
    instruments = []
    instruments.append({"instrument_key": upstox_ce_instrument_key, "quantity": trade_qty, "transaction_type": "SELL", "product": "D", "price": upstox_ce_ltp})
    instruments.append({"instrument_key": upstox_pe_instrument_key, "quantity": trade_qty, "transaction_type": "SELL", "product": "D", "price": upstox_pe_ltp})
    current_index_price = round(float(finvasia_api.get_quotes(exchange="NSE", token=str(26000))['lp']),2)
    atm_iv =0
    strike_interval = 100
    remainder = math.fmod(current_index_price, strike_interval)
    if remainder > strike_interval / 2:
        atm_strike = math.ceil(current_index_price / strike_interval) * strike_interval
    else:
        atm_strike = math.floor(current_index_price / strike_interval) * strike_interval
    for sp in option_chain:
        if sp.strike_price == atm_strike:
            atm_iv = (float(sp.call_options.option_greeks.iv)+float(sp.put_options.option_greeks.iv))/2

    trade_details['current_index_price']=current_index_price
    lower_range = round((int(upstox_pe_symbol[-7:-2])-current_index_price)/current_index_price*100,2)
    trade_details['lower_range']=lower_range
    upper_range = round((int(upstox_ce_symbol[-7:-2])-current_index_price)/current_index_price*100,2)
    trade_details['upper_range']=upper_range
    trading_range = int(upstox_ce_symbol[-7:-2])-int(upstox_pe_symbol[-7:-2])
    trade_details['trading_range']=trading_range//2
    trade_details['range_per']=round(trading_range/current_index_price*100,2)

    # Build instruments for finvasia
    fin_pe_symbol = convert_option_symbol(upstox_pe_symbol)
    fin_ce_symbol = convert_option_symbol(upstox_ce_symbol)
    trade_details['fin_pe_symbol']=fin_pe_symbol
    trade_details['fin_ce_symbol']=fin_ce_symbol

    span_res = finvasia_api.span_calculator(finvasia_user_id,[
            {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":SPAN_Expiry,"optt":"PE","strprc":str(fin_pe_symbol[-5:])+".00","buyqty":"0","sellqty":str(trade_qty),"netqty":"0"},
            {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":SPAN_Expiry,"optt":"CE","strprc":str(fin_ce_symbol[-5:])+".00","buyqty":"0","sellqty":str(trade_qty),"netqty":"0"}
        ])

    trade_margin=float(span_res['span_trade']) + float(span_res['expo_trade'])
    trade_details['trade_margin']=trade_margin
    finvasia_pe_ltp=float(finvasia_api.get_quotes(exchange="NFO", token=str(fin_pe_symbol))['lp'])
    trade_details['finvasia_pe_ltp']=finvasia_pe_ltp
    finvasia_ce_ltp=float(finvasia_api.get_quotes(exchange="NFO", token=str(fin_ce_symbol))['lp'])
    trade_details['finvasia_ce_ltp']=finvasia_ce_ltp
    tot_fin_premium = round(trade_qty*(finvasia_pe_ltp+finvasia_ce_ltp),2)
    trade_details['tot_fin_premium']=tot_fin_premium
    fin_mtm_per= round(tot_fin_premium/trade_margin*100,2)
    trade_details['fin_mtm_per']=str(fin_mtm_per)+"%"
    trade_details['INDIA_VIX']=round(float(finvasia_api.get_quotes(exchange="NSE", token=str(26017))['lp']),2)
    try:
        # trade_details['INDIA_VIX_RSI']=get_nifty_rsi()
        trade_details['ATM_IV']=atm_iv
    except:
        trade_details['INDIA_VIX_RSI']=-1
        trade_details['ATM_IV']=-1
    return trade_details

def check_recent_pe_ce(df, time_window_minutes=120):
    # Ensure exch_tm is datetime
    df = df.copy()
    df['exch_tm'] = pd.to_datetime(df['exch_tm'], format='%d-%m-%Y %H:%M:%S')

    # Time window filtering
    now = datetime.now()
    start_time = now - timedelta(minutes=time_window_minutes)
    recent_df = df[df['exch_tm'] >= start_time]
    # Filter for 'STOP_LOSS' in remarks
    recent_df = recent_df[recent_df['remarks'].str.upper() == 'STOP_LOSS']

    # Check presence
    has_pe = not recent_df[recent_df['order_type'] == 'PE'].empty
    has_ce = not recent_df[recent_df['order_type'] == 'CE'].empty

    return has_pe, has_ce