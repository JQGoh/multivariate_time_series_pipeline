# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from src.enums import DataMetadata, FilePathEnum
from urllib import request


def main():
    """Download the raw data from the original source and prepare the demo data"""

    if not FilePathEnum.DOWNLOADED_ZIP.is_file():
        print("Download raw data from source")
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
            "household_power_consumption.zip"
        )
        with request.urlopen(url) as response, open(
            FilePathEnum.DOWNLOADED_ZIP, "wb"
        ) as out_file:
            data = response.read()
            out_file.write(data)

    df = prepare_demo_data(FilePathEnum.DOWNLOADED_ZIP)
    df.to_csv(FilePathEnum.MOCK_DATA, index=None)


def prepare_demo_data(rawfile_path):
    df = pd.read_csv(rawfile_path, delimiter=";", compression="zip")
    df[DataMetadata.DATETIME] = df["Date"] + " " + df["Time"]
    df.drop(["Date", "Time"], axis=1, inplace=True)

    numeric_columns = [DataMetadata.TARGET] + DataMetadata.NUMERIC_FEATURES
    for col in numeric_columns:
        # clean numeric data by omitting "?"
        df[col] = df[col].replace("?", np.nan).astype(np.float64)

    # Remove rows with empty feature values
    features = list(set(df.columns) - {DataMetadata.DATETIME})
    removed_idx = df[df[features].isnull().all(axis=1)].index
    df.drop(removed_idx, inplace=True)

    df[DataMetadata.DATETIME] = pd.to_datetime(
        df[DataMetadata.DATETIME], format="%d/%m/%Y %H:%M:%S"
    )
    df.sort_values([DataMetadata.DATETIME], inplace=True)

    # Now we have a reasonably 'cleaned' data to prepare mock data
    # Take the first 5 days of year 2008 and 2009 to mimic two different series
    series1 = df[
        (pd.Timestamp("2008-01-01") <= df[DataMetadata.DATETIME])
        & (df[DataMetadata.DATETIME] < pd.Timestamp("2008-01-06"))
    ].copy()
    series1[DataMetadata.ID] = "id1"
    series2 = df[
        (pd.Timestamp("2009-01-01") <= df[DataMetadata.DATETIME])
        & (df[DataMetadata.DATETIME] < pd.Timestamp("2009-01-06"))
    ].copy()
    series2[DataMetadata.ID] = "id2"
    df_2series = pd.concat([series1, series2], axis=0)

    # reset the two series to have the same year=2008
    df_2series[DataMetadata.DATETIME] = df_2series[DataMetadata.DATETIME].map(
        lambda x: x.replace(year=2008)
    )

    # Purposely assign some missing values
    random_state = 30
    # each numeric feature picks 5% samples randomly and assign np.Nan
    for feature in DataMetadata.NUMERIC_FEATURES:
        indices = df_2series.sample(frac=0.05, random_state=random_state).index
        random_state = random_state + 1
        df_2series.loc[indices, feature] = np.nan

    return df_2series


if __name__ == "__main__":
    main()
