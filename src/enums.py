# -*- coding: utf-8 -*-
from pathlib import Path


class FilePathEnum(object):
    PROJECT_DIR_POSIX = project_dir = Path(__file__).resolve().parents[1]  # PosixPath

    DOWNLOADED_ZIP = Path(PROJECT_DIR_POSIX).joinpath(
        "data/raw/household_power_consumption.zip"
    )
    FIGURE = Path(PROJECT_DIR_POSIX).joinpath("reports/figures/predictions_global_active_power.png")
    MOCK_DATA = Path(PROJECT_DIR_POSIX).joinpath("data/interim/mock_data.csv")
    PIPELINE = Path(PROJECT_DIR_POSIX).joinpath("data/processed/data_processing_pipelines.pkl")
    TEST_DATA = Path(PROJECT_DIR_POSIX).joinpath("data/processed/test_data.csv")
    TRAIN_FEATURES = Path(PROJECT_DIR_POSIX).joinpath("data/processed/train_features.csv")


class TsFreshEnum(object):
    # id passed as column_id/the derived feature due to rolling of time series
    ID = "id"
    SORT = "sort"

class DataMetadata(object):
    """Data metadata"""

    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    DATETIME = "datetime"
    TARGET = "Global_active_power"
    # synthetic series id
    ID = TsFreshEnum.ID

    GLOBAL_REACTIVE_POWER = "Global_reactive_power"
    GLOBAL_INTENSITY = "Global_intensity"
    SUB_METERING_1 = "Sub_metering_1"
    SUB_METERING_2 = "Sub_metering_2"
    SUB_METERING_3 = "Sub_metering_3"
    VOLTAGE = "Voltage"

    # Column sets
    NUMERIC_FEATURES = [
        GLOBAL_REACTIVE_POWER,
        GLOBAL_INTENSITY,
        SUB_METERING_1,
        SUB_METERING_2,
        SUB_METERING_3,
        VOLTAGE,
    ]
