from talib import MA, SMA, EMA, MOM
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from CTAFlow.data.sierra.fast_parse import FastScidReader as FSR
import asyncio as aio
from CTAFlow.config import DLY_DATA_PATH

fsr = FSR(str(DLY_DATA_PATH/"ESH25-CME.scid"))

with fsr as f:
    data = f.to_pandas()
