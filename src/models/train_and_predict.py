# -*- coding: utf-8 -*-

import cloudpickle
import matplotlib.pyplot as plt
import pandas as pd
from darts import concatenate
from darts.models import RegressionModel
from darts.timeseries import TimeSeries

from src.enums import DataMetadata, FilePathEnum


def main():
    (
        impute_and_datetime_pipeline,
        tsfresh_pipeline,
        post_process_pipeline,
        post_processed_datetime,
        series_id,
        target_column,
    ) = cloudpickle.load(open(FilePathEnum.PIPELINE, "rb"))

    # Prepare train data, which has been processed by pipeline
    loaded_train = pd.read_csv(FilePathEnum.TRAIN_FEATURES)
    # cast to human readable datetime format
    loaded_train[post_processed_datetime] = pd.to_datetime(
        loaded_train[post_processed_datetime]
    )
    loaded_train.set_index(post_processed_datetime, inplace=True)

    # Prepare test data, now we load the fitted pipeline to transform
    loaded_raw_test = pd.read_csv(
        FilePathEnum.TEST_DATA,
        infer_datetime_format=DataMetadata.DATETIME_FORMAT,
        parse_dates=[DataMetadata.DATETIME],
    )
    # it is straight forward to transform raw data by following the right sequence of pipeline
    stage1 = impute_and_datetime_pipeline.transform(loaded_raw_test)
    stage2 = tsfresh_pipeline.transform(stage1)
    test = post_process_pipeline.transform(stage2)
    test = pd.DataFrame(test, columns=post_process_pipeline.get_feature_names_out())
    test[post_processed_datetime] = pd.to_datetime(test[post_processed_datetime])
    test.set_index(post_processed_datetime, inplace=True)

    # Separate target and keep the features in loaded_train, test
    train_target = loaded_train[[target_column, series_id]].copy()
    train_covariates = loaded_train.drop(columns=[target_column])
    test_target = test[[target_column, series_id]].copy()
    test_covariates = test.drop(columns=[target_column])

    # Cast data into Dart's time series
    train_target_groupby = train_target.groupby(series_id)
    loaded_train_groupby = train_covariates.groupby(series_id)
    test_target_groupby = test_target.groupby(series_id)
    test_covariates_groupby = test_covariates.groupby(series_id)
    group_keys = list(train_target_groupby.groups.keys())

    # Note: TimeSeries.from_dataframe does not support column of string type. Drop the identifier
    train_target_list = [x.drop(columns=series_id) for _, x in train_target_groupby]
    train_target_timeseries = [
        TimeSeries.from_series(series) for series in train_target_list
    ]
    train_covariates_list = [x.drop(columns=series_id) for _, x in loaded_train_groupby]
    train_covariates_timeseries = [
        TimeSeries.from_dataframe(df) for i, df in enumerate(train_covariates_list)
    ]

    # Prepend data to provide sufficient history for test
    history_count = 1
    predict_count = 20
    test_covariates_list = [
        x.drop(columns=series_id) for _, x in test_covariates_groupby
    ]
    test_covariates_list = [
        pd.concat([x.tail(history_count), y])
        for x, y in zip(train_covariates_list, test_covariates_list)
    ]
    test_covariates_timeseries = [
        TimeSeries.from_dataframe(df) for df in test_covariates_list
    ]
    # ground truth
    test_target_list = [
        x.head(predict_count).drop(columns=series_id) for _, x in test_target_groupby
    ]
    test_target_timeseries = [
        TimeSeries.from_series(series) for series in test_target_list
    ]

    regr_model = RegressionModel(
        lags=None, lags_past_covariates=history_count, lags_future_covariates=None
    )
    regr_model.fit(
        series=train_target_timeseries,
        past_covariates=train_covariates_timeseries,
        future_covariates=None,
    )

    predictions = regr_model.predict(
        n=predict_count,
        series=train_target_timeseries,
        past_covariates=test_covariates_timeseries,
    )
    pred_actual = concatenate(predictions + test_target_timeseries, axis="component")
    pred_actual.plot()
    legend_text = ["prediction: " + x for x in group_keys] + [
        "ground truth: " + x for x in group_keys
    ]
    plt.legend(legend_text)
    plt.title("Predictions of Global Active Power")
    plt.savefig(FilePathEnum.FIGURE, format="png")


if __name__ == "__main__":
    main()
