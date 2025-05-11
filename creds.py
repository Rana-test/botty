import os
import yaml
import upstox_client
from api_helper import ShoonyaApiPy
import pyotp
### Upstox login 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import pandas as pd
from utils import EmailClient

def load_credentials():
    """Loads credentials from YAML file or environment variables."""
    if os.path.exists('creds.yml'):
        with open('creds.yml', 'r') as file:
            return yaml.safe_load(file)
    return {
        key: os.getenv(key) for key in [
            "EMAIL_USER", "EMAIL_TO", "EMAIL_PASS", "TOKEN", "userid", "password",
            "vendor_code", "api_secret", "imei", "UPSTOX_API_KEY", "UPSTOX_API_SECRET",
            "UPSTOX_URL", "UPSTOX_MOB_NO", "UPSTOX_CLIENT_PASS", "UPSTOX_CLIENT_PIN"
        ]
    }


def login_finvasia(token, userid, password, vendor_code, api_secret, imei):
    api = ShoonyaApiPy()
    twoFA = pyotp.TOTP(token).now()
    login_response = api.login(
        userid=userid,
        password=password,
        twoFA=twoFA,
        vendor_code=vendor_code,
        api_secret=api_secret,
        imei=imei
    )
    assert login_response['stat'] == 'Ok', f"Finvasia login failed: {login_response}"
    return api


def login_upstox(UPSTOX_API_KEY, UPSTOX_URL, UPSTOX_API_SECRET, UPSTOX_MOB_NO, UPSTOX_CLIENT_PASS, UPSTOX_CLIENT_PIN):
    configuration = upstox_client.Configuration()

    def wait_for_page_load(driver, timeout=30):
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script('return document.readyState') == 'complete')
        
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument("--headless") 
    driver = webdriver.Chrome(options=options)
    driver.get(UPSTOX_URL)
    wait_for_page_load(driver)
    username_input_xpath = '//*[@id="mobileNum"]'
    username_input_element = driver.find_element(By.XPATH, username_input_xpath)
    username_input_element.clear()
    username_input_element.send_keys(UPSTOX_MOB_NO)
    get_otp_button_xpath = '//*[@id="getOtp"]'
    get_otp_button_element = driver.find_element(By.XPATH, get_otp_button_xpath)
    get_otp_button_element.click()
    client_pass = pyotp.TOTP(UPSTOX_CLIENT_PASS).now()
    text_box = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "otpNum")))
    text_box.clear()
    text_box.send_keys(client_pass)
    wait = WebDriverWait(driver, 10)
    continue_button = wait.until(EC.element_to_be_clickable((By.ID, "continueBtn")))
    continue_button.click()
    # XPath for the pin input field
    text_box = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "pinCode")))
    text_box.clear()
    text_box.send_keys(UPSTOX_CLIENT_PIN)
    continue_button = wait.until(EC.element_to_be_clickable((By.ID, "pinContinueBtn")))
    continue_button.click()
    redirect_url = WebDriverWait(driver, 10).until(
        lambda d: "?code=" in d.current_url
    )
    # Retrieve the token from the URL
    token = driver.current_url.split("?code=")[1]

    url = 'https://api.upstox.com/v2/login/authorization/token'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'code': token,
        'client_id': UPSTOX_API_KEY,
        'client_secret': UPSTOX_API_SECRET,
        'redirect_uri': "https://www.google.com",
        'grant_type': 'authorization_code',
    }

    response = requests.post(url, headers=headers, data=data)

    access_token=response.json().get("access_token")

    # Configure OAuth2 access token for authorization: OAUTH2
    configuration = upstox_client.Configuration()
    configuration.access_token = access_token

    upstox_opt_api = upstox_client.OptionsApi(upstox_client.ApiClient(configuration))
    upstox_charge_api = upstox_client.ChargeApi(upstox_client.ApiClient(configuration))

    return upstox_opt_api, upstox_charge_api


def load_credentials_and_apis():
    creds = load_credentials()

    finvasia_api = login_finvasia(
        token=creds["TOKEN"],
        userid=creds["userid"],
        password=creds["password"],
        vendor_code=creds["vendor_code"],
        api_secret=creds["api_secret"],
        imei=creds["imei"],
    )

    upstox_opt_api, upstox_charge_api = login_upstox(
        UPSTOX_API_KEY=creds['UPSTOX_API_KEY'], 
        UPSTOX_URL = creds['UPSTOX_URL'],
        UPSTOX_API_SECRET=creds['UPSTOX_API_SECRET'],
        UPSTOX_MOB_NO=creds['UPSTOX_MOB_NO'],
        UPSTOX_CLIENT_PASS=creds['UPSTOX_CLIENT_PASS'],
        UPSTOX_CLIENT_PIN=creds['UPSTOX_CLIENT_PIN'],
    )

    upstox_instruments = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz")
    session_vars_df = pd.read_csv('session_var.csv')
    trade_book = pd.read_csv('trade_book.csv')
    email_client = EmailClient(
        sender_email=creds["EMAIL_USER"],
        receiver_email=creds["EMAIL_TO"],
        email_password=creds["EMAIL_PASS"]
    )

    return finvasia_api, upstox_opt_api, upstox_charge_api, upstox_instruments, session_vars_df, trade_book, email_client, creds["userid"]
