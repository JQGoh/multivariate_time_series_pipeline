# -*- coding: utf-8 -*-

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.enums import DataMetadata, FilePathEnum
from src.features.custom_transformers import (DataFrameTransformer,
                                              DatetimeFeatureTransformer,
                                              TransformWithFeatureNamesFactory)
from src.features.tsfresh_transformers import (RollingLagsTrasformer,
                                               TSFreshRollingTransformer)


def main():
    mock_data = pd.read_csv(
        FilePathEnum.MOCK_DATA,
        infer_datetime_format=DataMetadata.DATETIME_FORMAT,
        parse_dates=[DataMetadata.DATETIME],
    )

    # Take 10% of most recent data as test set
    grouped = mock_data.groupby(DataMetadata.ID)
    max_row_counts = grouped.count()[DataMetadata.TARGET].max()
    test_row_count = int(max_row_counts * 0.1)
    test = grouped.tail(test_row_count)
    train = mock_data[~mock_data.index.isin(test.index)]

    # raw test set saved for later use
    test.to_csv(FilePathEnum.TEST_DATA, index=None)

    # NOTE: Using scikit-learn==1.0.2, we shall encounter the following AttributeError for
    # SimpleImputer, without using TransformWithFeatureNamesFactory()
    # However, newer version such as scikit-learn==1.3.0, we don't need TransformWithFeatureNamesFactory()
    # AttributeError: Estimator MedianImputer does not provide get_feature_names_out.
    # Did you mean to call pipeline[:-1].get_feature_names_out()?
    transformer_factory = TransformWithFeatureNamesFactory()
    transformer_factory.register_format("SimpleImputer", SimpleImputer)

    print("Setting imputer, datetime pipeline to impute input feature values")
    pre_imputer_class = "Pre-MedianImputer"
    derive_datetime_class = "DeriveDatetime"
    impute_and_datetime_transformer = ColumnTransformer(
        transformers=[
            (
                pre_imputer_class,
                transformer_factory.get_transformer("SimpleImputer")(
                    names=DataMetadata.NUMERIC_FEATURES,
                    missing_values=np.nan,
                    strategy="median",
                ),
                DataMetadata.NUMERIC_FEATURES,
            ),
            (
                derive_datetime_class,
                DatetimeFeatureTransformer(),
                DataMetadata.DATETIME,
            ),
        ],
        verbose=True,
        remainder="passthrough",
    )
    impute_and_datetime_pipeline = Pipeline(
        steps=[
            ("ImputeAndDatetime", impute_and_datetime_transformer),
        ],
    )

    train_datetime = impute_and_datetime_pipeline.fit_transform(train)
    datetime_transformed_names = impute_and_datetime_pipeline.get_feature_names_out()

    print(
        "Setting multivariate time series pipeline that includes Tsfresh derived rolling features"
    )
    features_for_rolling = [
        feature
        for feature in datetime_transformed_names
        if feature.startswith(pre_imputer_class)
    ]
    # one-hot encoded applied to hour component only
    one_hot_features = [derive_datetime_class + "__hour"]
    kind_to_fc_params = {
        feature: {
            "median": None,
            "maximum": None,
            "minimum": None,
            "c3": [{"lag": 10}],
        }
        for feature in features_for_rolling
    }

    tsfresh_rolling_class = "TSFreshRolling"
    rolling_lags_class = "RollingLags"
    tsfresh_pipeline = Pipeline(
        steps=[
            (
                "CastToDataFrame",
                DataFrameTransformer(columns=datetime_transformed_names),
            ),
            (
                "tsfresh",
                ColumnTransformer(
                    transformers=[
                        (
                            tsfresh_rolling_class,
                            TSFreshRollingTransformer(
                                input_column_names=features_for_rolling,
                                kind_to_fc_parameters=kind_to_fc_params,
                                rolling_window_size=30,
                                column_id="remainder" + "__" + DataMetadata.ID,
                                column_sort=derive_datetime_class
                                + "__"
                                + DataMetadata.DATETIME,
                            ),
                            features_for_rolling
                            + [
                                derive_datetime_class + "__" + DataMetadata.DATETIME,
                                "remainder" + "__" + DataMetadata.ID,
                            ],
                        ),
                        (
                            rolling_lags_class,
                            RollingLagsTrasformer(
                                input_column_names=features_for_rolling,
                                rolling_window_size=30,
                                column_id="remainder" + "__" + DataMetadata.ID,
                                column_sort=derive_datetime_class
                                + "__"
                                + DataMetadata.DATETIME,
                                orders=[1, 2, 3],
                            ),
                            features_for_rolling
                            + [
                                derive_datetime_class + "__" + DataMetadata.DATETIME,
                                "remainder" + "__" + DataMetadata.ID,
                            ],
                        ),
                        (
                            "OneHot",
                            OneHotEncoder(),
                            one_hot_features,
                        ),
                        (
                            "PassThrough",
                            "passthrough",
                            [
                                derive_datetime_class + "__" + DataMetadata.DATETIME,
                                "remainder" "__" + DataMetadata.ID,
                            ],
                        ),
                    ],
                    verbose=True,
                    remainder="passthrough",
                ),
            ),
        ]
    )
    train_tsfresh = tsfresh_pipeline.fit_transform(train_datetime)
    tsfresh_transformed_names = tsfresh_pipeline.get_feature_names_out()

    print(
        "Setting post time series derived features pipeline to impute any other "
        "remaining missing feature values followed by standard scaling."
    )
    numeric_names_for_transform = [
        col
        for col in tsfresh_transformed_names
        if col.startswith(tsfresh_rolling_class) or col.startswith(rolling_lags_class)
    ]
    impute_scaler_pipeline = Pipeline(
        steps=[
            (
                "Post-MedianImputer",
                transformer_factory.get_transformer("SimpleImputer")(
                    names=numeric_names_for_transform,
                    missing_values=np.nan,
                    strategy="median",
                ),
            ),
            (
                "StandardScaler",
                StandardScaler(),
            ),
        ]
    )
    post_process_pipeline = Pipeline(
        steps=[
            (
                "DataFrameConverter",
                DataFrameTransformer(columns=tsfresh_transformed_names),
            ),
            (
                "PostProcess",
                ColumnTransformer(
                    transformers=[
                        (
                            "ImputeAndScaler",
                            impute_scaler_pipeline,
                            numeric_names_for_transform,
                        )
                    ],
                    verbose=True,
                    remainder="passthrough",
                ),
            ),
        ]
    )

    train_post_processed = post_process_pipeline.fit_transform(train_tsfresh)
    post_transformed_names = post_process_pipeline.get_feature_names_out()

    print("Save the derived time series multivariate features")
    train_df = pd.DataFrame(train_post_processed, columns=post_transformed_names)
    train_df.to_csv(FilePathEnum.TRAIN_FEATURES, index=None)

    # target column is in fact not transformed but can be loaded for later use
    # similarly, record some other useful special columns
    target_name = next(
        name for name in post_transformed_names if name.endswith(DataMetadata.TARGET)
    )
    series_id = next(
        name for name in post_transformed_names if name.endswith(DataMetadata.ID)
    )
    post_processed_datetime = next(
        name for name in post_transformed_names if name.endswith(DataMetadata.DATETIME)
    )
    cloudpickle.dump(
        (
            impute_and_datetime_pipeline,
            tsfresh_pipeline,
            post_process_pipeline,
            post_processed_datetime,
            series_id,
            target_name,
        ),
        open(FilePathEnum.PIPELINE, "wb"),
    )


if __name__ == "__main__":
    main()
