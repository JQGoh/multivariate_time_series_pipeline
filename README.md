Multivariate Time Series Pipeline
==============================

Demonstration of building a configurable multivariate time series pipeline which performs feature engineering for time series forecasting.

## Quick Start
1. Install the required Python packages. Installation in a virtual environment is recommended, see [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) for an example.
```
pip install -r requirements.txt
```
2. Set the Python path to `multivariate_time_series_pipeline`.
```
export PYTHONPATH=$PYTHONPATH:{your_local_path/multivariate_time_series_pipeline}
```  
where by **your_local_path** is the path containing the `multivariate_time_series_pipeline` repository.

3. Execute the Python scripts at the `multivariate_time_series_pipeline` path with the following 
scripts in sequence:  
   - `src/data/make_dataset.py`: Download the raw data and prepare a mock data for demo.
   - `src/features/make_features.py`: Perform time series feature engineering.
   - `src/models/train_and_predict.py`: Train a simple linear regression model for time series predictions using [darts](https://github.com/unit8co/darts) package.
4. A successful time series prediction expects to replicate a chart as shown below 
![predictions_global_active_power.png](reports/figures/predictions_global_active_power.png) 

## Overview
This project is inspired by the need of:  
  * Build a feature engineering pipeline of time series derived features using the [Sklearn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) such that a pipeline can be used repeatedly for feature processing.  
  * Get a clear idea of the types of transformations performed to obtain the features based on their column names. 
  * Sklearn pipeline's transformation may shuffle the column order of derived features and thus it is not straight forward to track derived features by reading the matrix values.  
  * Configure the desired features using a time series feature derivation library such as [tsfresh](https://tsfresh.readthedocs.io/en/latest/) during the intermediate stage of pipeline transformation. The time series derived features in particular focuses on the rolling based feature derivation.

[make_features.py](src/features/make_features.py) file mainly illustrates the time series feature derivation using the constructed pipeline. A number of customized Python class objects are available in [custom_transformers.py](src/features/custom_transformers.py) and [tsfresh_transformers.py](src/features/tsfresh_transformers.py), which leverage [mixin (a simple type of multiple inheritance)](https://www.ianlewis.org/en/mixins-and-python) and [factory method](https://realpython.com/factory-method-python/) to implement the customized pipeline.   
