Multivariate Time Series Pipeline
==============================

Demonstration of building a multivariate time series pipeline which performs feature engineering for time series forecasting.

## Quick Start
1. Install the required Python packages.  
```
pip install -r requirements.txt
```
2. Set the Python path to "multivariate_time_series_pipeline".
```
export PYTHONPATH=$PYTHONPATH:{your_local_path/multivariate_time_series_pipeline}
```
where by **your_local_path** is the path containing the repository 
`multivariate_time_series_pipeline` repository.
3. Execute the Python scripts at the `multivariate_time_series_pipeline` path with the following 
scripts in sequence:  
   - `src/data/make_dataset.py`: Download the raw data and prepare a mock data for demonstration purpose.
   - `src/featuers/make_features.py`: Perform time series feature engineering.
   - `src/models/train_and_predict.py`: Train a simple linear regression model for time series predictions using [darts](https://github.com/unit8co/darts) package.
4. A successful time series prediction expects to replicate a chart like 
![predictions_global_active_power.png](reports/figures/predictions_global_active_power.png) 

