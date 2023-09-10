import pandas as pd
from darts import TimeSeries
from sklearn.base import BaseEstimator, TransformerMixin


class DartsTransformerMixin(object):
    def __init__(
        self, darts_transformers, time_col, value_cols=None, group_cols=None, **kwargs
    ):
        """
        Parameters
        ----------
        darts_transformers:
            Transformers supported by Darts, (import path: darts.dataprocessing.transformers.*)
            whereby the transformer class inherits BaseDataTransformer.

        time_col: str
            The datetime column.

        value_cols: list of str or str
            The feature values to be processed, included as Darts Timeseries object.

        group_cols: list of str or str
            The list of columns or single column that serve as series id.
        """
        super(DartsTransformerMixin, self).__init__(**kwargs)
        self.names = None
        self.darts_transformers = darts_transformers
        self.time_col = time_col
        self.value_cols = value_cols
        self.group_cols = group_cols

    def get_feature_names_out(self, input_features=None):
        """
        Parameters
        ----------
        input_features : array-like of str or None, default=None
           Not used, present here for API consistency by convention.
           https://github.com/scikit-learn/scikit-learn/blob
         /0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/feature_extraction/text.py#L1433-L1450
        """
        return self.names

    def _get_darts_timeseries(self, X):
        if self.group_cols:
            darts_series = TimeSeries.from_group_dataframe(
                df=X,
                time_col=self.time_col,
                value_cols=self.value_cols,
                group_cols=self.group_cols,
            )
        else:
            darts_series = TimeSeries.from_dataframe(
                df=X,
                time_col=self.time_col,
                value_cols=self.value_cols,
            )
        return darts_series

    def _get_dataframe(self, darts_timeseries):
        if isinstance(darts_timeseries, list):
            dfs = []
            for series in darts_timeseries:
                df = series.pd_dataframe().reset_index()

                # restore the series ID info
                for column, series_id in zip(
                    series.static_covariates.loc["global_components"].index,
                    series.static_covariates.loc["global_components"].values,
                ):
                    df[column] = series_id
                dfs.append(df)

            return pd.concat(dfs, axis=0)
        else:
            return darts_timeseries.pd_dataframe().reset_index()


class DartsTransformers(
    BaseEstimator,
    TransformerMixin,
    DartsTransformerMixin,
):
    def fit(self, X):
        if hasattr(self.darts_transformers, "fit"):
            darts_series = self._get_darts_timeseries(X)
            self.darts_transformers.fit(darts_series)
        return self

    def transform(self, X):
        darts_series = self._get_darts_timeseries(X)
        transformed_series = self.darts_transformers.transform(darts_series)
        transformed = self._get_dataframe(transformed_series)
        self.names = transformed.columns
        return transformed
