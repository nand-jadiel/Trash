import pandas as pd
import numpy as np
import datetime as dt
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing

def create_time_series_data(fac_ID, time_series_length, pd_data_frame):
    facility_records = pd_data_frame.loc[pd_data_frame['facility_id'] == fac_ID]
    df_shape = facility_records.shape
    number_of_samples = df_shape[0]
    req_columns = facility_records.drop(['value', 'facility_id', 'actual_date', 'actual_time', 'actual_interval'], axis=1)
    min_since_opening = req_columns['minutes_since_park_opening'].tolist()
    rate_changes = req_columns['rate_of_change'].tolist()
    time_stamps = req_columns['actual_timestamp'].tolist()
    day_of_week = req_columns['day_of_the_week'].tolist()
    minutes = req_columns['minutes_since_park_opening'].tolist()
    time_series_data = []
    labels = []
    for n in range(0, number_of_samples - time_series_length, time_series_length):
        each_slot = rate_changes[n:n + time_series_length + 1]
        time_record = time_stamps[n:n + time_series_length + 1]
        delta_t = [(pd.to_datetime(time_record[time_series_length]) - pd.to_datetime(time_record[t])).seconds / float(60) for t in range(0, time_series_length)]
        day_name = day_of_week[n:n + time_series_length + 1]
        minu = minutes[n:n + time_series_length + 1]
        try:
            #Packet format -
            #minutes from park opening to reading, day of the week, delta time to raget reading
            temp = [[i[2], i[3], int(i[0])] for i in zip(delta_t, each_slot[0: time_series_length], day_name[0: time_series_length], minu[0: time_series_length])]
            temp2 =  [s for t in temp for s in t] + each_slot
            time_series_data.append(temp2)
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
       "max_depth": 10,
       "subsample": 0.6,
       "colsample_bytree": 0.6,
       "min_child_weights": 0.1,
       "silent": 1,
       "gamma": 0,
       "seed": 0,
       "tree_method": 'exact'
    }

    model = xgb.train(params, X_Y_training, num_boost_round, evals = watchlist,
                      early_stopping_rounds = 100, verbose_eval = True)
    return model

def main():
    TEST_SPLIT = 0.3
    TIME_STEPS = 4
    facility = 'facility.csv'
    sensor_data = 'sensor_read.csv'
    raw_data = pd.read_csv(sensor_data)
    facilities = pd.read_csv(facility)
    #Features selection
    raw = pd.merge(raw_data, facilities, how='left', left_on='facility_id', right_on='id')
    raw['read_at'] = pd.to_datetime(raw['read_at'], infer_datetime_format = True)
    raw.sort_values(['facility_id', 'read_at'], inplace = True)
    raw_values = raw.dropna(axis=0, how='any')
    facilites = raw_values.drop(['id_x','created_at','lat',  'name', 'lng', 'id_y', 'sensor_id', 'auto_id', 'battery_level_x', 'sensor_type', 'battery_level_y', 'percentage', 'type', 'updated_at', 'fill_level_threshold'], axis = 1)
    facilites['date'] = facilites['read_at'].dt.date
    facilites['timestamp'] = facilites['read_at'].dt.time
    facilites['actual_timestamp'] = pd.to_datetime(facilites['read_at']) + pd.DateOffset(hours = -6)
    facilites['actual_date'] = facilites['actual_timestamp'].dt.date
    facilites['actual_time'] = facilites['actual_timestamp'].dt.time
    facilites['day_of_the_week'] = facilites['actual_timestamp'].dt.weekday
    facilites = facilites.drop(['date', 'timestamp'], axis = 1)
    facilites['actual_interval'] = (facilites['actual_timestamp'].shift(-1) - facilites['actual_timestamp']).dt.seconds / 60.0
    facilites = facilites.loc[facilites['actual_interval'] < 60]
    facilites['rate_of_change'] = facilites['value'].diff() / ((facilites['actual_timestamp'] - facilites['actual_timestamp'].shift(1)).dt.seconds / 60.0)
    facilites = facilites.dropna(axis=0, how='any')
    facilites['minutes_since_park_opening'] = (facilites['actual_timestamp'].dt.hour - 8) * 60 + (facilites['actual_timestamp'].dt.minute)
    facilites = facilites.drop(['read_at'], axis=1)
    all_facilities = facilites['facility_id'].tolist()
    facilites_filtered = facilites.loc[facilites['rate_of_change'] > 0.0]
    facility_names = np.unique(facilites_filtered['facility_id'].tolist())
    print "Number of facilities: ", len(facility_names)

    #Creating time-series input data
    time_steps = []
    targets = []
    time_series_length = TIME_STEPS
    df = facilites_filtered.dropna(axis=0, how='any')
    for each_fac in facility_names:
        data, lab = create_time_series_data(each_fac, time_series_length, df)
        time_steps = time_steps + data
        targets = targets + lab

    data_shape = np.shape(time_steps)
    print "Overall Data Shape: ", data_shape
    split_index = int(data_shape[0] * (1-TEST_SPLIT))
    print "Splitting the dataset at: ", split_index
    X_train, X_test, y_train, y_test = time_steps[0:split_index], time_steps[split_index:], targets[0:split_index], targets[split_index:]
    print "Train Data Shape: ", np.shape(X_train)
    print "Test Data shape: ", np.shape(X_test)
    print "Sample from the Training data: ", X_train[0]
    print "Sample from the Training-Targets data: ", y_train[0]
    shape = np.shape(X_train)
    indices = range(0, shape[0])
    training_idx, validation_idx = indices[:shape[0] // 5 * 4], indices[shape[0] // 5 * 4:]
    X_training, X_validation = [X_train[i] for i in training_idx], [X_train[i] for i in validation_idx]
    Y_training, Y_validation = [y_train[i] for i in training_idx], [y_train[i] for i in validation_idx]
    print "Validation Data Shape: ", np.shape(X_validation)
    print "Actual Training Data Shape: ", np.shape(X_training)
    X_Y_training = xgb.DMatrix(X_training, Y_training)
    X_Y_validation = xgb.DMatrix(X_validation, Y_validation)

    model_xgb = train_xgb_model(X_Y_training, X_Y_validation)
    Xtest = xgb.DMatrix(X_test)
    Ypreds = model_xgb.predict(Xtest)
    print "RMSE Error is: "
    print sqrt(mean_squared_error(y_test, Ypreds))
    l1, = plt.plot(y_test[900:1000], 'r', label='Ground Truth')
    l2, = plt.plot(Ypreds[900:1000], 'g', label='Predictions')
    plt.ylabel("actual rate of change")
    plt.xlabel("time steps")
    plt.legend(handles=[l1, l2])
    plt.show()


if __name__=='__main__':
    main()
