import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split

def create_training_data(time_series_length, rate_changes):
    time_series_data = []
    labels = []
    for n in range(0, len(rate_changes), time_series_length):
        each_slot = rate_changes[n:n + time_series_length + 1]
        time_series_data.append(each_slot[0:time_series_length])
        try:
            labels.append(each_slot[time_series_length])
        except IndexError:
            pass
    return (time_series_data, labels)
    
    
main():
  facility = 'facility.csv'
  sensor_data = 'sensor_read.csv'
  raw_data = pd.read_csv(sensor_data)
  facilities = pd.read_csv(facility)
  
  #Feature selection
  raw = pd.merge(raw_data, facilities, how='left', left_on='facility_id', right_on='id')
  raw['read_at'] = pd.to_datetime(raw['read_at'], infer_datetime_format = True)
  min_date = raw['read_at'].min()
  max_date = raw['read_at'].max()
  #couting the number of hours passed from midnight, day and month details
  raw['hour_since_midnight'] = raw['read_at'].dt.hour
  raw['day'] = raw['read_at'].dt.day
  raw['month'] = raw['read_at'].dt.month
  raw['day_of_week'] = raw['read_at'].dt.dayofweek
  raw.sort_values(['facility_id', 'read_at'], inplace = True, axis=0)
  #Calculating the rate of change of trash bin fill levels
  raw['rate_of_change'] = raw['value'].diff() / ( (raw['read_at'] - raw['read_at'].shift(1)).dt.seconds / 60.0)
  raw.sort_values(['read_at'], inplace = True)
  raw.dropna()
  #For model test: Select one trashbin
  #one_bin = raw.loc[raw['facility_id'] == 'D9CE43FDD7CBD9105C8A36F58D0265C3']
  one_bin = raw
  one_bin = one_bin.dropna(axis=0,how='any')
  facility1 = one_bin.sort_values('read_at',  ascending=True)
  facility1 = facility1.drop(['id_x','created_at','lat', 'day',  'name', 'month', 'day_of_week', 'lng', 'id_y', 'value', 'sensor_id', 'auto_id', 'battery_level_x', 'sensor_type', 'facility_id', 'battery_level_y', 'percentage', 'type', 'updated_at', 'fill_level_threshold'], axis = 1)
  facility1 = facility1.sort_values('read_at',  ascending=True)
  facility1['actual_interval'] = (facility1['read_at'].shift(-1) - facility1['read_at']).dt.seconds / 60.0
  #Removing records where the time-interval between two successive readings is greater than 45 minutes
  facility1_filt_temp = facility1.loc[facility1['actual_interval'] < 45]
  temp_1 = facility1_filt_temp.loc[facility1_filt_temp['rate_of_change'] >= 0]
  facility1_filtered = temp_1.dropna()
  rc = facility1_filtered['rate_of_change'].tolist()
  #Convert our data into time series data and generate labels (t+4) for the same
  facility1_data, labels = create_training_data(4, rc)
  facility1_data = facility1_data[:-1]
  print np.shape(facility1_data)
  #Generate train and test data
  test_split = 0.2
  data_shape = np.shape(facility1_data)
  X_train, X_test, y_train, y_test = facility1_data[0:int(data_shape[0] * (1-test_split))], facility1_data[int(data_shape[0] * test_split):], labels[0:int(data_shape[0] * (1-test_split))], labels[int(data_shape[0] * test_split):]
