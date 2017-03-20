import pandas as pd
import numpy as np
import datetime as dt
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

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

def train_xgb_model(X_Y_training, X_Y_validation):
    num_boost_round = 5000
    watchlist = [(X_Y_training, 'train'), (X_Y_validation, 'eval')]
    params = {
       "objective": "reg:linear",
       "booster": "gbtree",
       "eval_metric": "rmse",
       "eta": 0.1,
       #Tweak this parameter
       "max_depth": 12,
       "subsample": 0.6,
       "colsample_bytree": 0.6,
       "min_child_weights": 0.1,
       "silent": 1,
       "gamma": 0,
       "seed": 0,
       #To avoid overfitting
       "tree_method": 'exact'
    }
    model = xgb.train(params, X_Y_training, num_boost_round, evals = watchlist,
                      early_stopping_rounds = 100, verbose_eval = True)
    return model

def main():
    TEST_SPLIT = 0.3
    facility = 'facility.csv'
    sensor_data = 'sensor_read.csv'
    raw_data = pd.read_csv(sensor_data)
    facilities = pd.read_csv(facility)
    #Features selection
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
    facilites = one_bin.sort_values('read_at',  ascending=True)
    facilites = facilites.drop(['id_x','created_at','lat', 'day',  'name', 'month', 'day_of_week', 'lng', 'id_y', 'value', 'sensor_id', 'auto_id', 'battery_level_x', 'sensor_type', 'facility_id', 'battery_level_y', 'percentage', 'type', 'updated_at', 'fill_level_threshold'], axis = 1)
    facilites = facilites.sort_values('read_at',  ascending=True)
    facilites['actual_interval'] = (facilites['read_at'].shift(-1) - facilites['read_at']).dt.seconds / 60.0
    #Removing records where the time-interval between two successive readings is greater than 45 minutes
    facilities_filt_temp = facilites.loc[facilites['actual_interval'] < 45]
    temp_1 = facilities_filt_temp.loc[facilities_filt_temp['rate_of_change'] > 0]
    temp_1 = temp_1.loc[temp_1['rate_of_change'] < 1.1]
    facilities_filtered = temp_1.dropna()
    rc = facilities_filtered['rate_of_change'].tolist()
    #Convert our data into time series data and generate labels (t+4) for the same
    facility_data, labels = create_training_data(4, rc)
    facility_data = facility_data[:-1]
    print np.shape(facility_data)
    #Generate train and test data
    data_shape = np.shape(facility_data)
    X_train, X_test, y_train, y_test = facility_data[0:int(data_shape[0] * (1-TEST_SPLIT))], facility_data[int(data_shape[0] * TEST_SPLIT):], labels[0:int(data_shape[0] * (1-TEST_SPLIT))], labels[int(data_shape[0] * TEST_SPLIT):]
    shape = np.shape(X_train)
    indices = range(0, shape[0])
    training_idx, validation_idx = indices[:shape[0] // 5 * 4], indices[shape[0] // 5 * 4:]
    X_training, X_validation = [X_train[i] for i in training_idx], [X_train[i] for i in validation_idx]
    Y_training, Y_validation = [y_train[i] for i in training_idx], [y_train[i] for i in validation_idx]
    X_Y_training = xgb.DMatrix(X_training, Y_training)
    X_Y_validation = xgb.DMatrix(X_validation, Y_validation)
    model_xgb = train_xgb_model(X_Y_training, X_Y_validation)
    Xtest = xgb.DMatrix(X_test)
    Ypreds = model_xgb.predict(Xtest)
    print "RMSE Error is: "
    print sqrt(mean_squared_error(y_test, Ypreds))
    l1, = plt.plot(y_test[100:200], 'r', label='line1')
    l2, = plt.plot(Ypreds[100:200], 'g', label='line2')
    plt.ylabel("actual rate of change")
    plt.xlabel("time steps")
    plt.legend(handles=[l1, l2])
    plt.show()

if __name__=='__main__':
    main()
