import numpy as np
import pandas as pd
from data.data_client import DataClient

dclient = DataClient()

class IntradayFeatures:

    def __init__(self, ticker_symbol, close_col="Last", bid_volume="BidVolume", ask_volume="AskVolume", volume="Volume"):

        print('Loading Intraday Data')
        self.data = dclient.query_market_data(ticker_symbol)
        self.returns = np.log(self.data[close_col]) - np.log(self.data[close_col].shift(1))
        self.buy_vol = self.data[ask_volume]
        self.sell_vol = self.data[bid_volume]
        self.volume = self.data[volume]

        return

    def historical_rv(self, window=21, average=True, annualize=False):
        returns = self.returns

        dr = pd.bdate_range(returns.index.date[0], returns.index.date[-1])
        rv = np.zeros(len(dr))
        rv[:] = np.nan
        if average:
            denom = window
        else:
            denom = 1

        for idx in range(window, len(dr)):
            d_start = dr[idx - window]
            d_end = dr[idx]

            rv[idx] = np.sqrt(np.sum(returns.loc[d_start:d_end] ** 2)) / denom

        if annualize:
            rv *= np.sqrt(252)

        hrv = pd.Series(data=rv,
                        index=dr,
                        name=f'{window}_rv')

        return hrv

    def realized_semivariance(self,window=1, average=True):
        returns = self.returns
        data = []
        dr = pd.bdate_range(returns.index.date[0], returns.index.date[-1])
        denom = 1
        if average:
            denom = window

        for idx in range(window, len(dr)):
            start = dr[idx - window]
            end = dr[idx]
            rets = returns.loc[start:end]
            rs_neg = np.sqrt((rets[rets < 0].sum() ** 2)) / denom
            rs_pos = np.sqrt((rets[rets > 0].sum() ** 2)) / denom
            data.append((end, rs_pos, rs_neg))

        # Create a DataFrame from collected rows
        rs_df = pd.DataFrame(data, columns=['date', 'RS_pos', 'RS_neg'])
        rs_df.set_index('date', inplace=True)
        return rs_df

    def cumulative_delta(self, window=1):
        """Calculate cumulative delta (buy volume - sell volume) over a rolling window

        Args:
            window: Number of days to calculate cumulative delta over

        Returns:
            pd.Series: Cumulative delta values indexed by date
        """
        returns = self.returns
        buy_vol = self.buy_vol
        sell_vol = self.sell_vol

        # Get unique dates from the index
        dr = pd.bdate_range(returns.index.date[0], returns.index.date[-1])
        cumulative_delta = np.zeros(len(dr))
        cumulative_delta[:] = np.nan

        for idx in range(window, len(dr)):
            d_start = dr[idx - window]
            d_end = dr[idx]

            # Calculate delta for the window period
            buy_volume_window = buy_vol.loc[d_start:d_end].sum()
            sell_volume_window = sell_vol.loc[d_start:d_end].sum()
            cumulative_delta[idx] = buy_volume_window - sell_volume_window

        cd_series = pd.Series(data=cumulative_delta,
                             index=dr,
                             name=f'{window}d_cumulative_delta')

        return cd_series

@dataclass
class SpreadFeatures:

    def __init__(self, symbol):
        self.symbol = symbol

        self.summary = dclient.get_curve_data_summary(symbol)
        curve_data = dclient.query_curve_data(symbol)


        return
