import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

SYMBOL = "XAU/USD"

TIMEFRAME = "5min"

START_PRICE = 100.0
DIRECTION = "UP"
SPACING = 1.0

TOTAL_DEGREES = 360
BARS_PER_360 = 360
SHAPE_NUM = 8

MAX_STOP_PIPS = 100
MIN_RR = 3.0
MAX_TRADES_PER_DAY = 6