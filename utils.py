from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd

def is_within_timeframe(start, end):
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    start_time = now.replace(hour=int(start.split(':')[0]), minute=int(start.split(':')[1]), second=0, microsecond=0)
    end_time = now.replace(hour=int(end.split(':')[0]), minute=int(end.split(':')[1]), second=0, microsecond=0)
    return start_time <= now <= end_time

def identify_session():
    if is_within_timeframe("08:30", "12:27"):
        return {"session": "session1", "start_time": "09:15", "end_time": "12:27"}
    elif is_within_timeframe("12:30", "15:30"):
        return {"session": "session2","start_time": "12:30", "end_time": "15:30"}
    return None

def format_trade_metrics(metrics):
    total_pnl = metrics.get("Total_PNL", "N/A")
    india_vix = metrics.get("INDIA_VIX", "N/A")
    expiry_details = metrics.get("Expiry_Details", {})
    
    data = []
    for expiry, details in expiry_details.items():
        if "Error" in details:
            data.append([expiry, details["Error"], "", "", "", "", "", "", "", "", ""])
        else:
            data.append([
                expiry, details.get("PNL", "N/A"), details.get("CE_Strike", "N/A"),
                details.get("PE_Strike", "N/A"), details.get("Current_Index_Price", "N/A"),
                details.get("ATM_IV", "N/A"), details.get("Expected_Movement", "N/A"),
                details.get("Lower_Breakeven", "N/A"), details.get("Upper_Breakeven", "N/A"),
                details.get("Breakeven_Range", "N/A"), details.get("Breakeven_Range_Per", "N/A"),
                details.get("Near_Breakeven", "N/A"), details.get("Max_Profit", "N/A"), details.get("Max_Loss", "N/A"),
                details.get("Realized_Premium", "N/A")
            ])
    
    df = pd.DataFrame(data, columns=[
        "Expiry", "PNL", "CE Strike", "PE Strike", "Current Index Price", "ATM IV", "Expected Movement", "Lower Breakeven", 
        "Upper Breakeven", "Breakeven Range", "Breakeven %", "Near Breakeven", "Max Profit", "Max Loss", "Realized_Premium"
    ])
    
    table_html = df.to_html(index=False, border=1)
    
    email_body = f"""
    <html>
    <head>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
    </style>
    </head>
    <body>
        <p><strong>Total PNL:</strong> {total_pnl}</p>
        <p><strong>INDIA VIX:</strong> {india_vix}</p>
        {table_html}
    </body>
    </html>
    """
    
    return email_body



import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class EmailClient:
    def __init__(self, sender_email, receiver_email, email_password):
        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.email_password = email_password
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587

    def _send(self, subject, body, subtype='html'):
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, subtype))

        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.email_password)
            server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")
        finally:
            server.quit()

    def send_email_html(self, subject, html_body):
        self._send(subject, html_body, subtype='html')

    def send_email_plain(self, subject, text_body):
        self._send(subject, text_body, subtype='plain')

