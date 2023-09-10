Multivariate Time Series Pipeline
==============================

A demonstration of building a tractable, feature engineering pipeline for multivariate time series. Read more in the article [Building a Tractable, Feature Engineering Pipeline for Multivariate Time Seires](https://www.kdnuggets.com/2022/03/building-tractable-feature-engineering-pipeline-multivariate-time-series.html) published on the KDnuggets.

## Quick Start
1. Install the required Python packages. Installation in a virtual environment is recommended, see [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) for an example.
```
pip install -r requirements.txt
```
2. Set the Python path to `multivariate_time_series_pipeline`.
```
export PYTHONPATH=$PYTHONPATH:{your_local_path/multivariate_time_series_pipeline}
```  
whereby **your_local_path** is the path containing the `multivariate_time_series_pipeline` repository.

3. Execute the Python scripts at the `multivariate_time_series_pipeline` path with the following 
scripts in sequence:  
   - `src/data/make_dataset.py`: Download the raw data and prepare a mock data for demo.
   - `src/features/make_features.py`: Perform time series feature engineering.
   - `src/models/train_and_predict.py`: Train a simple linear regression model for time series predictions using [darts](https://github.com/unit8co/darts) package.
4. A successful time series prediction expects to replicate a chart as shown below 
![predictions_global_active_power.png](reports/figures/predictions_global_active_power.png) 

## Overview
This project is inspired by the need of:  
  * Build a time series feature engineering pipeline using the [Scikit-learn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) such that the pipeline can be used repeatedly for different use cases with minimal customization.
  * Get a clear idea of the types of transformations performed to obtain the features based on the feature names. Sklearn pipeline may also shuffle the column order of derived features upon the transformation and thus it is not straight forward to track the derived features based on the matrix values. The output feature names at pipeline's stages should illustrate the types of transformations performed and the users can then select the relevant intermediate features for further feature processings.  
  * Configure the desired features using a time series feature derivation library such as [tsfresh](https://tsfresh.readthedocs.io/en/latest/) during the intermediate stage of pipeline transformation. The time series derived features in particular focuses on the rolling based feature derivation.

### Key Ideas
This project leverages [mixin (multiple inheritance)](https://easyaspython.com/mixins-for-fun-and-profit-cb9962760556) and [factory method](https://realpython.com/factory-method-python/) to implement components used in the customized pipeline. The key components include the customized Python class objects in [custom_transformers.py](src/features/custom_transformers.py) and [tsfresh_transformers.py](src/features/tsfresh_transformers.py). The [make_features.py](src/features/make_features.py) file demonstrates the usage of these components to perform time series feature engineering. 

## Changelog
  * (2023-09-10) Updated dependencies to simplify the installation process. Darts version 0.25.0 removed optional dependencies such as LightGBM and etc., see [PR1589](https://github.com/unit8co/darts/pull/1589) for further details. Included a short note to remark that Sklearn SimpleImputer does support get_feature_names_out() method and thus the earlier work using ``TransformWithFeatureNamesFactory`` is no longer required for scikit-learn==1.3.0.
  * (2023-09-10) Added [DartsTransformers](https://github.com/JQGoh/multivariate_time_series_pipeline/blob/ec6a11d000bd5183d798f3bd242cc841b5deac05/src/features/darts_transformers.py#L79) and provided an example generating features using Darts-based transformers. Note that in the [``make_features.py``](https://github.com/JQGoh/multivariate_time_series_pipeline/blob/master/src/features/make_features.py), operations such as SimpleImputer with strategy="median" in fact refers to the the median value computed across all the time series, instead of per-series basis. For feature transformations performed in a per-series basis, it is best to follow the practices such as [Scaler(StandardScaler())](https://github.com/JQGoh/multivariate_time_series_pipeline/blob/ec6a11d000bd5183d798f3bd242cc841b5deac05/src/features/make_features_darts_window_transformer.py#L63) in Darts, which does scaling on the list of time series. In [``make_features_darts_window_transformer.py``](https://github.com/JQGoh/multivariate_time_series_pipeline/blob/master/src/features/make_features_darts_window_transformer.py) example, we include Sklearn transformers as part of the demonstration example, but do note that the transformation is not done in a per-series manner.
    - Caveat: For columns processed by the Sklearn ColumnTransformer, by right the columns such as [datetime and series ID](https://github.com/JQGoh/multivariate_time_series_pipeline/blob/ec6a11d000bd5183d798f3bd242cc841b5deac05/src/features/make_features_darts_window_transformer.py#L89) will be discarded upon the pipeline's transform. However, we cast the dataframe into Darts.Timeseries object and re-cast the Darts.Timeseries back to the dataframe during the transformation, and this requires us to keep the datetime and series ID columns.