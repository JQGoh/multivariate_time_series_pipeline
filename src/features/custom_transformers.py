# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        return pd.DataFrame(X, columns=self.columns)

    def get_feature_names_out(self, input_features=None):
        """
        Parameters
        ----------
        input_features : array-like of str or None, default=None
           Not used, present here for API consistency by convention.
           https://github.com/scikit-learn/scikit-learn/blob
         /0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/feature_extraction/text.py#L1433-L1450
        """
        return self.columns


class DatetimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """Take a series of datetime values and derive its hour, minute values. Original datetime
    values kept and casted in integer format"""

    def fit(self, X):
        assert len(X.shape) == 1
        assert pd.core.dtypes.common.is_datetime64_ns_dtype(pd.Series(X))
        self.datetime_name = X.name
        return self

    def transform(self, X):
        return np.vstack([X.dt.hour, X.dt.minute, X.view(int)]).T

    def get_feature_names_out(self, input_features=None):
        """
        Parameters
        ----------
        input_features : array-like of str or None, default=None
           Not used, present here for API consistency by convention.
           https://github.com/scikit-learn/scikit-learn/blob
         /0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/feature_extraction/text.py#L1433-L1450
        """
        return ["hour", "minute", self.datetime_name]


class FeatureNamesMixin(object):
    def __init__(self, names, **kwargs):
        super(FeatureNamesMixin, self).__init__(**kwargs)
        self.names = names

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


class TransformWithFeatureNamesFactory(object):
    # Design mimics the suggestion from
    # https://realpython.com/factory-method-python/#basic-implementation-of-factory-method
    def __init__(self):
        self._creators = {}

    def register_format(self, format, transformer):
        class TransformerFactoryClass(FeatureNamesMixin, transformer):
            pass

        self._creators[format] = TransformerFactoryClass

    def get_transformer(self, format):
        transformer = self._creators.get(format)
        if transformer is None:
            raise ValueError("Transformer of format {} is not available".format(format))
        return transformer
