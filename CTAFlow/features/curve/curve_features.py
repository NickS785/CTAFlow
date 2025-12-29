import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

CRACK_SPREAD_LOCATION = Path("F:\\Data\\intraday\\synthetic\\crack_spread.csv")
DATA_DIR = Path("F:\\monthly_contracts\\")



class CurveFeatures:
    """
    Features derived from futures curve structure for energy commodities.

    This class provides methods to calculate curve-based features such as:
    - Relative basis (spread slope comparison)
    - Butterfly spreads (curve curvature)
    - Carry, contango/backwardation metrics
    - Time spread analysis

    Usage:
    ------
    from CTAFlow.data.spread_loader import SpreadEngine
    from CTAFlow.features.curve.energy import CurveFeatures

    # Generate continuous contracts and spreads
    engine = SpreadEngine(contract_pool)
    engine.generate_continuous_df(start, end)
    engine.calculate_spreads()

    # Create curve features
    curve_features = CurveFeatures(engine)
    rb = curve_features.relative_basis((1, 2), (2, 3))
    """

    def __init__(self, spread_engine=None, continuous_df: Optional[pd.DataFrame] = None,
                 spread_df: Optional[pd.DataFrame] = None):
        """
        Initialize CurveFeatures.

        Parameters:
        -----------
        spread_engine : SpreadEngine, optional
            SpreadEngine instance with data and spreads already computed
        continuous_df : pd.DataFrame, optional
            DataFrame with continuous contract prices (M1, M2, M3, ...)
            Used if spread_engine is not provided
        spread_df : pd.DataFrame, optional
            DataFrame with spreads (Spread_M1_M2, Spread_M2_M3, ...)
            Used if spread_engine is not provided
        """
        if spread_engine is not None:
            self.continuous_df = spread_engine.data
            self.spread_df = spread_engine.spreads
        else:
            self.continuous_df = continuous_df
            self.spread_df = spread_df

        if self.continuous_df is None:
            logger.warning("No continuous contract data provided")
        if self.spread_df is None:
            logger.warning("No spread data provided")

    def relative_basis(self, pair_1: Tuple[int, int], pair_2: Tuple[int, int]) -> pd.Series:
        """
        Calculate relative basis between two calendar spreads.

        Relative Basis = (Spread1 / TimeDiff1) - (Spread2 / TimeDiff2)

        This measures the difference in slope between two parts of the curve.
        Useful for identifying curve shape changes and relative value opportunities.

        Parameters:
        -----------
        pair_1 : Tuple[int, int]
            First spread pair, e.g., (1, 2) for M1-M2
        pair_2 : Tuple[int, int]
            Second spread pair, e.g., (2, 3) for M2-M3

        Returns:
        --------
        pd.Series with relative basis values

        Example:
        --------
        # Compare front spread slope vs back spread slope
        rb = curve_features.relative_basis((1, 2), (2, 3))

        # Positive: Front is steeper than back (curve flattening)
        # Negative: Back is steeper than front (curve steepening)
        """
        if self.spread_df is None:
            raise ValueError("No spread data available. Provide spread_df or SpreadEngine with spreads.")

        # Parse column names
        s1_name = f"Spread_M{pair_1[0]}_M{pair_1[1]}"
        s2_name = f"Spread_M{pair_2[0]}_M{pair_2[1]}"

        if s1_name not in self.spread_df.columns:
            raise ValueError(f"Spread {s1_name} not found in data")
        if s2_name not in self.spread_df.columns:
            raise ValueError(f"Spread {s2_name} not found in data")

        # Time differences (in months)
        dt1 = abs(pair_1[1] - pair_1[0])
        dt2 = abs(pair_2[1] - pair_2[0])

        # Annualized spreads
        term1 = self.spread_df[s1_name] / dt1
        term2 = self.spread_df[s2_name] / dt2

        return term1 - term2

    def butterfly(self, m1: int, m2: int, m3: int) -> pd.Series:
        """
        Calculate butterfly spread (curve curvature).

        Butterfly = (M1-M2) - (M2-M3) = M1 - 2*M2 + M3

        Measures the convexity of the futures curve.

        Parameters:
        -----------
        m1, m2, m3 : int
            Contract months (e.g., 1, 2, 3 for M1, M2, M3)

        Returns:
        --------
        pd.Series with butterfly values

        Interpretation:
        ---------------
        Positive: Front spread > back spread (curve flattening/backwardation)
        Negative: Back spread > front spread (curve steepening/contango)
        """
        if self.spread_df is None:
            raise ValueError("No spread data available")

        s1_name = f"Spread_M{m1}_M{m2}"
        s2_name = f"Spread_M{m2}_M{m3}"

        if s1_name not in self.spread_df.columns or s2_name not in self.spread_df.columns:
            raise ValueError(f"Required spreads not found: {s1_name}, {s2_name}")

        return self.spread_df[s1_name] - self.spread_df[s2_name]

    def condor(self, m1: int, m2: int, m3: int, m4: int) -> pd.Series:
        """
        Calculate condor spread (second-order curvature).

        Condor = (M1-M2) - 2*(M2-M3) + (M3-M4)

        Measures how the butterfly is changing along the curve.

        Parameters:
        -----------
        m1, m2, m3, m4 : int
            Contract months

        Returns:
        --------
        pd.Series with condor values
        """
        if self.spread_df is None:
            raise ValueError("No spread data available")

        s1 = f"Spread_M{m1}_M{m2}"
        s2 = f"Spread_M{m2}_M{m3}"
        s3 = f"Spread_M{m3}_M{m4}"

        for s in [s1, s2, s3]:
            if s not in self.spread_df.columns:
                raise ValueError(f"Required spread not found: {s}")

        return self.spread_df[s1] - 2 * self.spread_df[s2] + self.spread_df[s3]

    def carry_return(self, front_month: int = 1, back_month: int = 2) -> pd.Series:
        """
        Calculate implied carry return.

        Carry = (Back - Front) / Front

        Measures the percentage return from rolling contracts.

        Parameters:
        -----------
        front_month : int
            Front contract (default: 1 for M1)
        back_month : int
            Back contract (default: 2 for M2)

        Returns:
        --------
        pd.Series with carry returns

        Interpretation:
        ---------------
        Positive: Contango (back > front) - cost to roll long
        Negative: Backwardation (front > back) - benefit to roll long
        """
        if self.continuous_df is None:
            raise ValueError("No continuous contract data available")

        front_col = f"M{front_month}"
        back_col = f"M{back_month}"

        if front_col not in self.continuous_df.columns or back_col not in self.continuous_df.columns:
            raise ValueError(f"Required contracts not found: {front_col}, {back_col}")

        return (self.continuous_df[back_col] - self.continuous_df[front_col]) / self.continuous_df[front_col]

    def curve_slope(self, front: int = 1, back: int = 6) -> pd.Series:
        """
        Calculate overall curve slope.

        Slope = (M_back - M_front) / (back - front)

        Measures the average price change per month along the curve.

        Parameters:
        -----------
        front : int
            Front contract month
        back : int
            Back contract month

        Returns:
        --------
        pd.Series with slope values per unit month
        """
        if self.continuous_df is None:
            raise ValueError("No continuous contract data available")

        front_col = f"M{front}"
        back_col = f"M{back}"

        if front_col not in self.continuous_df.columns or back_col not in self.continuous_df.columns:
            raise ValueError(f"Required contracts not found: {front_col}, {back_col}")

        return (self.continuous_df[back_col] - self.continuous_df[front_col]) / (back - front)

    def all_features(self, lookbacks: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate all curve features with rolling statistics.

        Parameters:
        -----------
        lookbacks : List[int]
            Rolling window sizes for statistics

        Returns:
        --------
        pd.DataFrame with all curve features including:
            - Relative basis (multiple pairs)
            - Butterflies
            - Condor
            - Carry returns
            - Curve slope
            - Rolling statistics for each
        """
        features = pd.DataFrame(index=self.spread_df.index if self.spread_df is not None else self.continuous_df.index)

        # Relative basis features
        if self.spread_df is not None and len(self.spread_df.columns) >= 2:
            try:
                rb_12_23 = self.relative_basis((1, 2), (2, 3))
                features['relative_basis_M1M2_M2M3'] = rb_12_23

                # Rolling statistics
                for window in lookbacks:
                    features[f'rb_ma{window}'] = rb_12_23.rolling(window).mean()
                    features[f'rb_std{window}'] = rb_12_23.rolling(window).std()
                    features[f'rb_zscore{window}'] = (rb_12_23 - rb_12_23.rolling(window).mean()) / rb_12_23.rolling(window).std()
            except ValueError as e:
                logger.warning(f"Could not calculate relative basis: {e}")

        # Butterfly features
        if self.spread_df is not None:
            try:
                bf = self.butterfly(1, 2, 3)
                features['butterfly_M1_M2_M3'] = bf

                for window in lookbacks:
                    features[f'butterfly_ma{window}'] = bf.rolling(window).mean()
                    features[f'butterfly_zscore{window}'] = (bf - bf.rolling(window).mean()) / bf.rolling(window).std()
            except ValueError as e:
                logger.warning(f"Could not calculate butterfly: {e}")

        # Carry features
        if self.continuous_df is not None:
            try:
                carry = self.carry_return(1, 2)
                features['carry_M1_M2'] = carry

                for window in lookbacks:
                    features[f'carry_ma{window}'] = carry.rolling(window).mean()
            except ValueError as e:
                logger.warning(f"Could not calculate carry: {e}")

        # Curve slope
        if self.continuous_df is not None:
            try:
                slope = self.curve_slope(1, 6)
                features['curve_slope_M1_M6'] = slope

                for window in lookbacks:
                    features[f'slope_ma{window}'] = slope.rolling(window).mean()
            except ValueError as e:
                logger.warning(f"Could not calculate curve slope: {e}")

        logger.info(f"Generated {features.shape[1]} curve features")
        return features

