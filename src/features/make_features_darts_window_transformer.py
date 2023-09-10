# -*- coding: utf-8 -*-

import cloudpickle
import pandas as pd
from darts.dataprocessing.transformers.scaler import Scaler
from darts.dataprocessing.transformers.window_transformer import \
    WindowTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.enums import DataMetadata, FilePathEnum
from src.features.darts_transformers import DartsTransformers


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

    # Purposely offset series_id="id2" with large values to demonstrate
    # that Scaler(StandardScaler()) is scaling based on the individual statistics
    id2_idx = train[train[DataMetadata.ID] == "id2"].index
    train.loc[id2_idx, DataMetadata.GLOBAL_INTENSITY] = (
        train.loc[id2_idx, DataMetadata.GLOBAL_INTENSITY] + 20000
    )

    # Darts transformers
    window_transformers = WindowTransformer(
        transforms=[
            {
                "function": "sum",
                "mode": "rolling",
                "window": pd.Timedelta(minutes=30),
            },
            {
                "function": "median",
                "mode": "rolling",
                "window": pd.Timedelta(minutes=10),
            },
        ],
    )
    dart_windows = DartsTransformers(
        darts_transformers=window_transformers,
        time_col=DataMetadata.DATETIME,
        value_cols=DataMetadata.GLOBAL_INTENSITY,
        group_cols=[DataMetadata.ID],
    )
    darts_scaler = DartsTransformers(
        darts_transformers=Scaler(StandardScaler()),
        time_col=DataMetadata.DATETIME,
        group_cols=[DataMetadata.ID],
    )
    # pipeline includes all Darts based transformers
    darts_window_scale_pipeline = Pipeline(
        steps=[
            ("WindowsOperations", dart_windows),
            ("StandardScaler", darts_scaler),
        ],
    )

    # pipeline includes sklearn based transformers
    sklearn_preprocess_pipeline = Pipeline(
        steps=[
            ("MedianImputer", SimpleImputer(strategy="median")),
            ("Quantile", QuantileTransformer()),
        ],
    )

    feature_engineering = ColumnTransformer(
        transformers=[
            (
                "DartsWindowScale",
                darts_window_scale_pipeline,
                # Need to provide datetime, seriesID, referenced columns
                [DataMetadata.DATETIME, DataMetadata.ID, DataMetadata.GLOBAL_INTENSITY],
            ),
            (
                "SklearnImputeQuantile",
                sklearn_preprocess_pipeline,
                DataMetadata.NUMERIC_FEATURES,
            ),
        ],
        verbose=True,
        remainder="passthrough",
    )
    pipeline = Pipeline(
        steps=[
            ("FeatureEngineering", feature_engineering),
        ],
    )

    # fit-transform and save the pipeline
    transformed_train = pipeline.fit_transform(train)
    transformed_train_names = pipeline.get_feature_names_out()
    transformed_train_df = pd.DataFrame(
        transformed_train, columns=transformed_train_names
    )
    cloudpickle.dump(
        pipeline,
        open(FilePathEnum.WITH_DARTS_PIPELINE, "wb"),
    )

    # Test the loaded pipeline to transform on test set data
    loaded_pipeline = cloudpickle.load(open(FilePathEnum.WITH_DARTS_PIPELINE, "rb"))
    # Prepare test data, now we load the fitted pipeline to transform
    loaded_raw_test = pd.read_csv(
        FilePathEnum.TEST_DATA,
        infer_datetime_format=DataMetadata.DATETIME_FORMAT,
        parse_dates=[DataMetadata.DATETIME],
    )
    transformed_test = loaded_pipeline.transform(loaded_raw_test)
    transformed_test_names = loaded_pipeline.get_feature_names_out()
    transformed_test_df = pd.DataFrame(transformed_test, columns=transformed_test_names)
    # validate that the transformed feature names are consistent
    assert (transformed_train_names == transformed_test_names).all()

    # List the distribution statistics to demonstrate the the scaling is done per-series basis
    print(train.groupby(DataMetadata.ID)[DataMetadata.GLOBAL_INTENSITY].describe())
    transformed_train_df[
        "DartsWindowScale__rolling_sum_0 days 00:30:00_Global_intensity"
    ] = transformed_train_df[
        "DartsWindowScale__rolling_sum_0 days 00:30:00_Global_intensity"
    ].astype(
        float
    )
    print(
        transformed_train_df.groupby("DartsWindowScale__id")[
            "DartsWindowScale__rolling_sum_0 days 00:30:00_Global_intensity"
        ].describe()
    )


if __name__ == "__main__":
    main()
