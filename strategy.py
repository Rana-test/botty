import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from trade_utils import (
    write_to_trade_book, 
    fixed_ratio_position_size, 
    exit_trade, 
    get_positions, 
    get_revised_qty_margin,
    get_atm_iv,
    calc_expected_move,
    sigmoid_exit_percent,
    exit_order,
    place_order,
    get_strikes,
    check_recent_pe_ce,
)
from indicators import calculate_supertrend
import pandas as pd
from trade_utils import get_target_thursday, holiday_dict
import time as sleep_time

logger = logging.getLogger(__name__)


def run_hourly_trading_strategy(
    live,
    finvasia_api,
    upstox_opt_api,
    upstox_charge_api,
    upstox_instruments,
    min_df,
    entry_confirm,
    exit_confirm,
    finvasia_user_id,
    put_neg_bias = 1,
    pos_base_lots = 7,
    pos_delta = 60000,
    current_time=None
):
    """
    Runs the hourly trading strategy using Supertrend and RSI.
    """
    return_msgs = []
    open_orders = None
    # Retrieve PnL
    trade_book_df, total_pnl = write_to_trade_book(finvasia_api)
    logger.info(f"Trade Book PnL: {total_pnl}")

    # Determine position size
    entry_trade_qty = fixed_ratio_position_size(pos_base_lots, pos_delta, total_pnl) * 75

    # logger.info("Running STEMA Strategy")
    instrument = "NSE_INDEX|Nifty 50"
    action = 'NO ACTION'

    if not current_time:
        current_time = datetime.now(ZoneInfo("Asia/Kolkata"))

    # Get market data and technical indicators
    signal_df = calculate_supertrend(min_df)
    signal_df=signal_df.reset_index()
    latest_signal = signal_df.iloc[-1]
    latest_trend = latest_signal['trend']
    entry_signal = latest_signal['entry_signal']
    exit_signal = latest_signal['exit_signal']
    rsi = latest_signal['rsi']

    pos = pd.DataFrame(finvasia_api.get_positions())
    if pos is None or pos.empty:
        has_open_order= False
    else:
        open_orders = pos[pos['netqty'].astype(int)!=0]
        has_open_order = not open_orders.empty

    #Verify exit based on open orders and trend
    if has_open_order:
        logging.info(f"Checking trend change for open orders")
        open_order_type= open_orders['dname'].apply(lambda x: x.split()[-1]).unique().item()
        # if trend is -1 and has PE open orders then exit
        if latest_trend == -1 and open_order_type == 'PE':
            exit_signal = 1
        # if trend is 1 and has CE open orders then exit
        if latest_trend == 1 and open_order_type == 'CE':
            exit_signal = 1

    logging.info(f"has_open_order: {has_open_order}")
    logging.info(f"Current Trend: {latest_trend}")
    logging.info(f"Entry Signal: {entry_signal}, Exit Signal:{exit_signal}")
    # Check if open order for Put exists and trend is -1 then trend changed
    logging.info(f"Checking rsi confirm with trend")
    rsi_exit_confirm =  (latest_trend ==1 and rsi>49) or (latest_trend == -1 and rsi<51)
    # Exit open orders if trend changes
    if abs(exit_signal)>0 and has_open_order and rsi_exit_confirm:
        exit_confirm+=exit_signal
    else:
        exit_confirm=0
    #Debugging
    # exit_confirm=3
    if exit_confirm>0 and has_open_order:
        logger.info("Exit condition met. Preparing to exit trades.")
        # Implement exit logic
        exit_confirm = 0
        open_orders['order_type'] = open_orders['dname'].apply(lambda x: x.split()[-1])
        exit_signals = []
        for _, row in open_orders.iterrows():
            if row['order_type'] == 'PE' and latest_trend == -1 and rsi_exit_confirm:
                exit_signals.append(-1)
                logging.info(f"Trend -1: exiting PE order {row['tsym']}")
            elif row['order_type'] == 'CE' and latest_trend == 1 and rsi_exit_confirm:
                exit_signals.append(1)
                logging.info(f"Trend 1: exiting CE order {row['tsym']}")
            else:
                exit_signals.append(0)
                # Debugging
                # exit_signals.append(1)

        # Add the exit signal per order to the dataframe if needed
        open_orders['exit_signal'] = exit_signals
        exit_orders = open_orders[open_orders['exit_signal'] != 0]
        if not exit_orders.empty:
            exit_trade_msgs = exit_trade(finvasia_api, exit_orders)
            return_msgs+= exit_trade_msgs
            # sleep_time.sleep(60)
            # write_to_trade_book(finvasia_api)
            # Check for open orders again after Exit # maybe - Giving gap of 1 iteration between exit and entry
            pos = pd.DataFrame(finvasia_api.get_positions())
            if pos is None or pos.empty:
                has_open_order= False
            else:
                open_orders = pos[pos['netqty'].astype(int)!=0]
                has_open_order = not open_orders.empty

    # Place new order if no open orders and combined_signal is 1 or -1
    rsi_entry_confirm =  (latest_trend ==1 and rsi>55) or (latest_trend == -1 and rsi<45)
    if not has_open_order and entry_signal != 0 and rsi_entry_confirm:
        entry_confirm+=entry_signal
    else:
        entry_confirm=0

    # Do not place order if recent PE/CE stop loss order exists in past 3 hours to avoid whipsaw
    # has_pe, has_ce = check_recent_pe_ce(trade_book_df, time_window_minutes=180)
    # if has_pe and entry_signal == -1:
    #     entry_confirm = 0
    #     logger.info("Recent PE order exists. Skipping entry.")
    # elif has_ce and entry_signal == 1:
    #     entry_confirm = 0
    #     logger.info("Recent CE order exists. Skipping entry.")

    #Debugging
    # entry_confirm=3
    if abs(entry_confirm) > 1:
        logger.info("Entry condition met. Preparing to place order.")
        entry_confirm = 0  
        action = 'MAKE ENTRY'
        order_type = 'CE' if entry_signal == 1 else 'PE'

        # Determine expiry, adjust for holidays
        expiry = get_target_thursday()
        expiry = holiday_dict.get(expiry, expiry)
        logging.info(f"Calculated Expiry: {expiry}")

        today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Kolkata")).date())

        # Get margin and collateral limits
        limits = finvasia_api.get_limits()
        min_coll = min(
            float(limits['cash']) + float(limits.get('cash_coll',0)) + float(limits['payin']) - float(limits['payout']) - float(limits.get('marginused', 0))/2,
            float(limits.get('collateral',0)) - float(limits.get('marginused', 0))/2
        )

        # Determine legs based on order type
        main_leg = get_strikes(upstox_opt_api, finvasia_api, instrument, expiry, entry_trade_qty, upstox_instruments, 0.25, finvasia_user_id)
        hedge_leg = get_strikes(upstox_opt_api, finvasia_api, instrument, expiry, entry_trade_qty, upstox_instruments, 0.17, finvasia_user_id)

        symbol_key = 'pe' if order_type == 'PE' else 'ce'
        main_key = f"fin_{symbol_key}_symbol"
        hedge_key = f"fin_{symbol_key}_symbol"
        up_main_key = f"upstox_{symbol_key}_instrument_key"
        up_hedge_key = f"upstox_{symbol_key}_instrument_key"

        orders = {
            'Main': {
                'trading_symbol': main_leg[main_key],
                'trading_up_symbol': main_leg[up_main_key],
                'order_action': 'S',
                'order_qty': str(entry_trade_qty),
                'order_type': order_type
            },
            'Hedge': {
                'trading_symbol': hedge_leg[hedge_key],
                'trading_up_symbol': hedge_leg[up_hedge_key],
                'order_action': 'B',
                'order_qty': str(entry_trade_qty),
                'order_type': order_type
            }
        }

        logging.info(f"Main Leg: {orders['Main']['trading_symbol']}")
        logging.info(f"Hedge Leg: {orders['Hedge']['trading_symbol']}")

        orders = get_revised_qty_margin(orders, upstox_charge_api, min_coll)
        base_lots = int(orders['Main']['order_qty']) // (75 * put_neg_bias)
        orders['Main']['order_qty'] = 75 * base_lots
        orders['Hedge']['order_qty'] = 75 * base_lots

        for leg in ['Hedge', 'Main']:
            order = orders[leg]
            if int(order['order_qty']) > 0:
                success, msg = place_order(
                    finvasia_api, live,
                    order['trading_symbol'],
                    order['order_action'],
                    order['order_qty'],
                    'STEMA'
                )
                logging.info(f"{leg} Order Status: {success}")
                return_msgs.append(msg)

        write_to_trade_book(finvasia_api)

    # Summary email/debug output
    subject = f"Trade Decision at {latest_signal['time']}"
    email_body = f"""
    Current Time: {latest_signal['time']}
    Current Close: {latest_signal['close']}
    20 EMA: {latest_signal['ema20']}
    34 EMA: {latest_signal['ema34']}
    Trend: {latest_trend}
    RSI: {rsi}
    Entry Signal: {entry_signal}
    Entry Confirm: {entry_confirm}
    Exit Signal: {exit_signal}
    Exit Confirm: {exit_confirm}
    Action: {action}
    Open Orders: {list(open_orders['tsym']) if open_orders is not None else "NA"}
    """
    return_msgs.append({'subject': subject, 'body': email_body})

    return return_msgs, entry_confirm, exit_confirm

def monitor_trade(finvasia_api, upstox_opt_api):
    return_msgs=[]
    # logging.info("Getting positions")
    pos_df = get_positions(finvasia_api)
    if pos_df is None:
        return {'get_pos Error': "Error getting position Info"}, []

    current_index_price = float(finvasia_api.get_quotes(exchange="NSE", token="26000")['lp'])
    total_pnl = 0
    expiry_metrics = {}

    for expiry, group in pos_df.groupby("expiry"):
        group = group[group['netqty'] != 0]
        if group.empty:
            continue
        expiry_date_str = expiry.strftime('%Y-%m-%d')
        try:
            atm_iv = get_atm_iv(upstox_opt_api, expiry_date_str, current_index_price)
        except Exception as e:
            logging.error(f"Error getting ATM IV: {e}")
            atm_iv=15
        days_to_expiry = int(group['Days_to_Expiry'].mean())
        expected_move = calc_expected_move(current_index_price, atm_iv, days_to_expiry)

        # PnL Metrics
        group = group.astype({'netupldprc': float, 'lp': float, 'netqty': float})
        current_pnl = -((group['netupldprc'] - group['lp']) * group['netqty']).sum()
        max_profit = -(group['netupldprc'] * group['netqty']).sum()

        # Breakevens
        ce_rows = group[group["type"] == "CE"]
        pe_rows = group[group["type"] == "PE"]
        ce_strike = ce_rows['sp'].astype(float).min() if not ce_rows.empty else 0
        pe_strike = pe_rows['sp'].astype(float).max() if not pe_rows.empty else 0

        upper_breakeven = ce_strike if ce_strike else float('inf')
        lower_breakeven = pe_strike if pe_strike else 0

        if ce_strike and pe_strike:
            breakeven_range = upper_breakeven - lower_breakeven
            near_breakeven = min(
                100 * (current_index_price - lower_breakeven) / current_index_price,
                100 * (upper_breakeven - current_index_price) / current_index_price
            )
        elif ce_strike:
            breakeven_range = upper_breakeven - current_index_price
            near_breakeven = 100 * breakeven_range / current_index_price
        elif pe_strike:
            breakeven_range = current_index_price - lower_breakeven
            near_breakeven = 100 * breakeven_range / current_index_price
        else:
            breakeven_range = 0
            near_breakeven = 0

        stop_loss_per = float(group['exit_loss_per'].mean())
        max_loss = -stop_loss_per * max_profit

        day_exit_pct = sigmoid_exit_percent(days_to_expiry) / 100
        exit_condition = (
            current_index_price < lower_breakeven or
            current_index_price > upper_breakeven or
            current_pnl < max_loss or
            current_pnl > day_exit_pct * max_profit
        )


        order_type = "STOP_LOSS" if current_pnl < 0 else "PROFIT_BOOK"
        #Debugging
        # exit_condition=True
        if exit_condition:
            logging.info(f"Exit condition met for {expiry_date_str}. Current PnL: {current_pnl}")
            if current_index_price < lower_breakeven:
                logging.info(f"Exit Condition -- Current Index Price: {current_index_price} < Lower Breakeven: {lower_breakeven}")
            if current_index_price > upper_breakeven:
                logging.info(f"Exit Condition -- Current Index Price: {current_index_price} > Upper Breakeven: {upper_breakeven}")
            if current_pnl < max_loss:
                logging.info(f"Exit Condition -- Current PnL: {current_pnl} < Max Loss: {max_loss}")
            if current_pnl > day_exit_pct * max_profit:
                logging.info(f"Exit Condition -- Current PnL: {current_pnl} > Day Exit Pct {day_exit_pct} * max_profit: {max_profit}")

            msgs = exit_order(group, finvasia_api, order_type= order_type, live=True)
            return_msgs += msgs
            write_to_trade_book(finvasia_api)
            breakeven_info = {
                "Lower_Breakeven": order_type,
                "Upper_Breakeven": order_type,
                "Breakeven_Range": order_type,
                "Breakeven_Range_Per": order_type
            }
        else:
            breakeven_info = {
                "Lower_Breakeven": round(lower_breakeven, 2),
                "Upper_Breakeven": round(upper_breakeven, 2),
                "Breakeven_Range": round(breakeven_range, 2),
                "Breakeven_Range_Per": round(100 * breakeven_range / current_index_price, 2)
            }

        # Store expiry metrics
        expiry_metrics[expiry] = {
            "PNL": round(current_pnl, 2),
            "CE_Strike": round(ce_strike, 2),
            "PE_Strike": round(pe_strike, 2),
            "Current_Index_Price": current_index_price,
            "ATM_IV": round(atm_iv, 2),
            "Expected_Movement": round(expected_move, 2),
            "Near_Breakeven": round(near_breakeven, 2),
            "Max_Profit": round(max_profit, 2),
            "Max_Loss": round(max_loss, 2),
            "Realized_Premium": round(0, 2),
            **breakeven_info
        }

        total_pnl += current_pnl

    metrics = {
        "Total_PNL": round(total_pnl, 2),
        "Expiry_Details": expiry_metrics
    }
    return metrics, return_msgs
