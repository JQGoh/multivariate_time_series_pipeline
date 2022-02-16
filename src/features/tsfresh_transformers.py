# -*- coding: utf-8 -*-

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series

from src.enums import TsFreshEnum

logger = logging.getLogger(__name__)


class TsfreshRollingMixin(object):
    """The Mixin expects the inherited class having the following attributes

    Attributes
    ----------
    input_column_names: list of str
            List of column names of input data

    column_id : "id"
         See https://tsfresh.readthedocs.io/en/latest/text/data_formats.html#data-formats
        If None, assume this is a single series data set. We shall manually create an extra
        column and label this time series. Best set as "id"

    column_sort : "sort"
        See https://tsfresh.readthedocs.io/en/latest/text/data_formats.html#data-formats

    rolling_window_size : int
        This is used for tsfresh.utilities.dataframe_functions.roll_time_series.
        It mainly helps to derive features based on a fixed rolling window size, instead of
        deriving the tsfresh features by considering whole time series length.
        Recall that tsfresh transforms a sequence into a scalar value, but we want to derive
        the rolling window statistics for each time point.

    min_timeshift : int
        This is used for tsfresh.utilities.dataframe_functions.roll_time_series.
        Throw away all extracted forecast windows smaller or equal than this. Must be larger
        than or equal 0.
    """

    def fit(self, X):
        assert isinstance(
            self.input_column_names, list
        ), "input_column_names must be a list"
        assert isinstance(X, pd.DataFrame), "Input X must be of pd.DataFrame type"
        self.revise_input_column_names(X)
        return self

    def revise_input_column_names(self, X):
        # see tsfresh.feature_extraction.data._check_colname
        # "Dict keys are not allowed to contain '__'"
        # "Dict keys are not allowed to end with '_'"

        # all input column names should be revised to avoid raised error
        self.columns_mapper = {
            column: column.replace("__", "--").rstrip("_") for column in X.columns
        }

        self.update_columns = False
        if any([k != v for k, v in self.columns_mapper.items()]):
            self.update_columns = True
            logger.info(
                "Modify the column name to avoid '__' character and ends with '_' "
                "character"
            )
            logger.info("Original column names: {}".format(self.columns_mapper.keys()))
            logger.info(
                "Modified column names: {}".format(self.columns_mapper.values())
            )

        if self.update_columns:
            if getattr(self, "kind_to_fc_parameters", None) is not None:
                # Update kind_to_fc_parameters for tsfresh.feature_extraction.extract_features
                # to use acceptable column names
                self.kind_to_fc_parameters = {
                    self.columns_mapper.get(k, k): v
                    for k, v in self.kind_to_fc_parameters.items()
                }

            if self.column_id is not None:
                self.column_id = self.columns_mapper.get(self.column_id, self.column_id)
            else:
                self.column_id = TsFreshEnum.ID

            if self.column_sort is not None:
                self.column_sort = self.columns_mapper.get(
                    self.column_sort, self.column_sort
                )

            self.input_column_names = [
                self.columns_mapper.get(name, name) for name in self.input_column_names
            ]

    def prepare_df(self, X):
        """Validate the basic settings and revise the column names to be compatible with
        tsfresh requirement
        """
        assert isinstance(X, pd.DataFrame), "Input X must be of pd.DataFrame type"
        df = X

        if self.update_columns:
            df.columns = [self.columns_mapper[column] for column in df.columns]

        if self.column_id not in df.columns and self.column_id == TsFreshEnum.ID:
            # A constant 'id' name to specify the whole series
            df[TsFreshEnum.ID] = TsFreshEnum.ID
        return df

    def get_combined(self, derived_features):
        # derived_features expect to have multi-index
        assert isinstance(self.df, pd.DataFrame)

        # combine the derived rolling features to the input features
        joined_columns = (
            [self.column_id, self.column_sort]
            if self.column_sort is not None
            else [self.column_id, TsFreshEnum.SORT]
        )

        df_combined = self.df.merge(
            derived_features,
            left_on=joined_columns,
            right_index=True,
            how="left",
        )
        transformed = df_combined[derived_features.columns].values

        return transformed

    def get_roll_time_series(self, X):
        self.df = self.prepare_df(X)

        # df_rolled generated by roll_time_series will automatically have
        #  - a new column 'id' that has tuple values of (column_id, column_sort)
        #  - a new column 'sort' if column_sort not specified
        df_rolled = roll_time_series(
            self.df,
            column_id=self.column_id,
            column_sort=self.column_sort if self.column_sort is not None else None,
            min_timeshift=self.min_timeshift,
            max_timeshift=self.rolling_window_size - 1,
        )

        if self.column_id in df_rolled.columns and self.column_id != TsFreshEnum.ID:
            # self.column_id info will no longer be used, discarded
            df_rolled.drop(columns=self.column_id, inplace=True)

        return df_rolled

    def get_feature_names_out(self, input_features=None):
        """
        Parameters
        ----------
        input_features : array-like of str or None, default=None
           Not used, present here for API consistency by convention.
           https://github.com/scikit-learn/scikit-learn/blob
         /0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/feature_extraction/text.py#L1433-L1450
        """
        if getattr(self, "derived_names", None) is not None:
            return self.derived_names
        else:
            raise ValueError(
                "No derived features from {}. No transformed has been "
                "performed?".format(self.__class__)
            )


class RollingLagsTrasformer(BaseEstimator, TransformerMixin, TsfreshRollingMixin):
    def __init__(
        self,
        input_column_names=None,
        column_id=None,
        column_sort=None,
        min_timeshift=0,
        rolling_window_size=10,
        orders=None,
    ):
        self.input_column_names = input_column_names
        self.min_timeshift = min_timeshift
        self.rolling_window_size = rolling_window_size
        self.column_id = column_id
        self.column_sort = column_sort
        self.orders = orders
        assert isinstance(self.orders, list) and all(
            [order > 0 for order in self.orders]
        )

    def transform(self, X):
        df_rolled = self.get_roll_time_series(X)

        dfs = []
        for order in self.orders:
            dfs.append(self.get_lag(df_rolled, order))
        df_features = pd.concat(dfs, axis=1)

        self.derived_names = df_features.columns
        logger.info("lags features transformation completed")
        return self.get_combined(df_features)

    def get_lag(self, df_rolled, order):
        df_features = (
            df_rolled[self.input_column_names + [TsFreshEnum.ID]]
            .groupby(TsFreshEnum.ID)
            .nth(-1 * order)
        )
        df_features.set_index(
            pd.MultiIndex.from_tuples(df_features.index), inplace=True
        )

        new_column_names = [
            col + " (lag {})".format(order) for col in df_features.columns
        ]
        df_features.columns = new_column_names
        return df_features


class TSFreshRollingTransformer(BaseEstimator, TransformerMixin, TsfreshRollingMixin):
    def __init__(
        self,
        default_fc_parameters=None,
        kind_to_fc_parameters=None,
        input_column_names=None,
        column_id=None,
        column_sort=None,
        min_timeshift=0,
        rolling_window_size=10,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.input_column_names = input_column_names
        self.min_timeshift = min_timeshift
        self.rolling_window_size = rolling_window_size
        self.column_id = column_id
        self.column_sort = column_sort

    def transform(self, X):
        df_rolled = self.get_roll_time_series(X)

        df_features = extract_features(
            df_rolled,
            default_fc_parameters=self.default_fc_parameters,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            column_id=TsFreshEnum.ID,
            column_sort=self.column_sort
            if self.column_sort is not None
            else TsFreshEnum.SORT,
        )
        self.derived_names = [
            col + "(window {})".format(self.rolling_window_size)
            for col in df_features.columns
        ]

        logger.info("Tsfresh extract_features transformation completed")
        return self.get_combined(df_features)
