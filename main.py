import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from strategy import run_hourly_trading_strategy, monitor_trade
from market_data import get_minute_data
from trade_utils import write_to_trade_book, get_minute_data
from api_helper import ShoonyaApiPy
from trade_utils import holiday_dict

from utils import identify_session, is_within_timeframe, format_trade_metrics

# Optional: Load credentials and APIs via helper
from creds import load_credentials_and_apis
import time as sleep_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logging.info("Checking if today is a holiday")
    dt = str(datetime.now(ZoneInfo("Asia/Kolkata")).date())
    if dt in holiday_dict:
        logging.info("Exiting since today is a holiday")
        # Debug
        exit(0)
    logging.info("Inside Main")
    session = identify_session()
    logging.info(f"Identified Session: {session}")
    if not session:
        print("No active trading session.")
        return

    # Load APIs and configurations
    finvasia_api, upstox_opt_api, upstox_charge_api, upstox_instruments, session_vars_df, trade_book, email_client,finvasia_user_id = load_credentials_and_apis()

    write_to_trade_book(finvasia_api, trade_csv="trade_book.csv")
    # logger.info("Trade book updated.")

    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    # logger.info(f"Running strategy at: {now}")

    # Define confirmation triggers (could be dynamic)
    exit_confirm = session_vars_df[session_vars_df['session_var'] == 'exit_confirm']['value'].iloc[0]
    entry_confirm = session_vars_df[session_vars_df['session_var'] == 'entry_confirm']['value'].iloc[0]
    logging.info(f"Loaded session variables: {session_vars_df}")
    # Wait till 9:15
    while is_within_timeframe("08:30", "09:15"):
        sleep_time.sleep(60)
    # Start Monitoring
    counter=0
    while is_within_timeframe(session.get('start_time'), session.get('end_time')):
        logging.info(f"Monitoring Trade @ {datetime.now(ZoneInfo('Asia/Kolkata'))}")
        metrics, return_msgs = monitor_trade(finvasia_api, upstox_opt_api)
        if len(return_msgs)>0:
            for msg in return_msgs:
                email_client.send_email_plain(msg['subject'], msg['body'])
        if metrics =="STOP_LOSS":
            email_client.send_email_plain("STOP LOSS HIT - QUIT", "STOP LOSS HIT")
        else:
        #     subject = f"FINVASIA: MTM:{metrics['Total_PNL']} | NEAR_BE:{metrics['Near_Breakeven']} | RANGE:{metrics['Breakeven_Range_Per']}| MAX_PROFIT:{metrics['Max_Profit']} | MAX_LOSS: {metrics['Max_Loss']}"
            if counter % 10 == 0:
                logging.info(f"Sending status mail")
                subject = "FINVASIA STATUS"
                metrics["INDIA_VIX"] = round(float(finvasia_api.get_quotes(exchange="NSE", token=str(26017))['lp']),2)
                email_body = format_trade_metrics(metrics)
                email_client.send_email_html(subject, email_body)
            if counter % 15 ==0:
                stema_min_df = get_minute_data(finvasia_api,now=None)
                logging.info(f"Got historical data")
                # Run strategy
                now = datetime.now(ZoneInfo("Asia/Kolkata"))
                return_msgs, entry_confirm, exit_confirm = run_hourly_trading_strategy(
                    live=True,
                    finvasia_api=finvasia_api,
                    upstox_opt_api=upstox_opt_api,
                    upstox_charge_api=upstox_charge_api,
                    upstox_instruments=upstox_instruments,
                    min_df=get_minute_data(finvasia_api, now),
                    entry_confirm=entry_confirm,
                    exit_confirm=exit_confirm,
                    finvasia_user_id= finvasia_user_id,
                    put_neg_bias = 1,
                    pos_base_lots = 7,
                    pos_delta = 60000,
                    current_time=None
                )
                print(f'Number of email messages: {len(return_msgs)}')
                for msg in return_msgs:
                    email_client.send_email_plain(msg['subject'], msg['body'])
            counter+=1
        sleep_time.sleep(60)
  
    # Logout
    logging.info(f"Saving session variables")
    session_vars_df.loc[session_vars_df['session_var']=='exit_confirm','value']=exit_confirm
    session_vars_df.loc[session_vars_df['session_var']=='entry_confirm','value']=entry_confirm
    session_vars_df.to_csv('session_var.csv', index=False)
    write_to_trade_book(finvasia_api)
    finvasia_api.logout()

# main()

if __name__ == "__main__":
    main()