import pandas as pd
import numpy as np
from typing import Optional, Union, List, Sequence, Dict
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler


class SJMRegime:

    def __init__(self, n_states=2, jump_penalty=80):

        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.scaler = StandardScaler()
        self.model = JumpModel(n_components=n_states, jump_penalty=jump_penalty)

        return

    def prepare_features(self, df_features: pd.DataFrame, price_column: str = "Close", resample: bool = True,
                         resample_period: str = '1d', offset: str = '-9h',
                         selected_columns: Optional[Union[str]] = None, ):
        px = df_features[price_column]
        if resample:
            id_returns = px.pct_change()
            dates = px.index.date
            rv = id_returns.groupby(dates).apply(lambda x: np.sum(x ** 2))
            px_rs = px.resample(resample_period, offset=offset).last()
            abs_daily = px_rs.pct_change()
            training_df_features = pd.DataFrame(data=[rv, abs_daily], index=pd.to_datetime(dates))
            if selected_columns:
                training_df_features[selected_columns] = df_features[selected_columns].loc[training_df_features.index]
        else:
            training_df_features = df_features[selected_columns]

        return training_df_features

    def fit(self, X, feat_weights=None):

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, feat_weights=feat_weights)

        return self

    def predict(self, X):

        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)
