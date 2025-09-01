from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os

MARKET_DATA_PATH = Path("F:\\", 'Macro', 'OSINT', 'market_data.h5')
COT_DATA_PATH = Path("F:\\", 'Macro', 'OSINT','COT', 'cot.h5' )
APP_ROOT = Path(__file__).parent