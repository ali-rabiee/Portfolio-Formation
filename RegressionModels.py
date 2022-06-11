import math # Mathematical functions 
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates # Formatting dates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Lambda
from keras.layers import LeakyReLU, Input, Dense, GRU, Dropout, LSTM, Layer, BatchNormalization
from keras import initializers, regularizers, constraints
from keras.engine.input_layer import Input
from keras import backend as K
from keras.models import Model
import time
from get_stock import get_stock
from feature_eng import get_indicators

def evaluate_model(y_pred, y_gt):
    # Mean Square Error (MSE)
    MSE = mean_squared_error(y_gt, y_pred)
    print(f'Mean Square Error (MSE): {np.round(MSE, 2)}')

    # Mean Absolute Error (MAE)
    MAE = mean_absolute_error(y_gt, y_pred)
    print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

    # Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean((np.abs(np.subtract(y_gt, y_pred)/ y_gt))) * 100
    print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(y_gt, y_pred)/ y_gt)) ) * 100
    print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %') 


def prepare_data(ticker, df, predicted_col, start, end, look_back=7, use_indicators=False): 
    """
    inputs:
    ticker = name of the ticker in string format
    predicted_col = the column that you want to predict
    start = start Date of test set
    end = end date of test set
    use_indicators = False: use Close, Open, Volume, High, Low as features
    use_indicators = True: use indicators as features
    """
    # Add predictions of DQN to test set
    DQN_prd_test = pd.read_csv(f"./Output/results/DQN predictions/{ticker}_3act_preds_test.csv")
    DQN_prd_valid = pd.read_csv(f"./Output/results/DQN predictions/{ticker}_3act_preds_valid.csv")
    DQN_prd = DQN_prd_valid.append(DQN_prd_test)
    st = str(DQN_prd.iloc[0]['Date'])
    ed = str(DQN_prd.iloc[-1]['Date'])
    # print(df[(df['Date'] >= st) & (df['Date'] <= ed)]['Date'])
    try:
        DQN_prd = DQN_prd['ensemble'].to_list()
        df[(df['Date'] >= st) & (df['Date'] <= ed)]['Action'] = DQN_prd
    except:
        pass
    # Select Features
    if not use_indicators:
        df = df[['Date', 'Close', 'Open', 'Volume', 'High', 'Low', 'Reward', 'Action']]
    else:
        df = get_indicators(df, predicted_col)
        df = df.dropna()
        df = df[['Date', 'Close', 'Open', 'Volume', 'High', 'Low', 'Reward', 'ma7', 'ema', 'Action']]

    features_num = len(df.columns) - 1
    features_name = df.columns
    # print(f"Number of Features: {features_num}")
    # print(f"Features: {features_name}")

    # Select the column that you want to predict
    df['Predictions'] = df[predicted_col]

    # Separate the Training and the test sets
    training_set = df[df['Date'] < start]
    test_set = df[(df['Date'] >= start) & (df['Date'] <= end)]
    dataset_size = len(df)
    training_size = len(training_set)
    test_size = len(test_set)

    print(f"dataset size = {dataset_size}")
    print(f"train size = {training_size}")
    print(f"test size = {test_size}")
    
    # Select Features
    features_train = training_set.iloc[:, 1:].values
    features_test = test_set.iloc[:, 1:].values

    # Feature Scaling
    sc = MinMaxScaler()
    training_set_scaled = sc.fit_transform(features_train)
    test_set_scaled = sc.fit_transform(features_test)

    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    predictions_scaled = scaler_pred.fit_transform(df[['Predictions']])

    # Creating the train data structure
    X_train = []
    y_train = []

    for i in range(look_back, training_size):
        x = training_set_scaled[i - look_back:i,:features_num]
        
        if training_set_scaled[i][features_num-1] == 1:
            act_col = np.ones((1, look_back), dtype=float)
        else:
            act_col = np.zeros((1, look_back), dtype=float)

        x = np.insert(x, features_num, act_col, axis=1)

        X_train.append(x) 
        y_train.append(training_set_scaled[i, features_num])

    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Creating the test data structure
    X_test = []
    y_test = []
  
    for i in range(look_back, test_size):
        x = test_set_scaled[i - look_back:i,:features_num]
        
        if test_set_scaled[i][features_num-1] == 1:
            act_col = np.ones((1, look_back), dtype=float)
        else:
            act_col = np.zeros((1, look_back), dtype=float)

        x = np.insert(x, features_num, act_col, axis=1)

        X_test.append(x)
        y_test.append(test_set_scaled[i, features_num]) 

    X_test, y_test = np.array(X_test), np.array(y_test) 
    
    return X_train, y_train, X_test, y_test, features_num, scaler_pred



class LSTM_model:
    def __init__(self, units, input_shape):
        
        self.units = units
        self.input_shape0 = input_shape[0]
        self.input_shape1 = input_shape[1]

    def build(self):
        # build the model
        self.model = Sequential()
        #Adding the first LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=self.units, return_sequences=True, input_shape=(self.input_shape0, self.input_shape1)))
        self.model.add(Dropout(0.2))
        # Adding a second LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding a third LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding a fourth LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=self.units))
        self.model.add(Dropout(0.2))
        # Adding the output layer
        self.model.add(Dense(units=1))
        
        # Compiling the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        return self.model


class GRU_model:
    def __init__(self, units, input_shape):
        
        self.units = units
        self.input_shape0 = input_shape[0]
        self.input_shape1 = input_shape[1]

    def build(self):
        # build the model
        self.model = Sequential()
        #Adding the first GRU layer and some Dropout regularisation
        self.model.add(GRU(units=self.units, return_sequences=True, input_shape=(self.input_shape0, self.input_shape1)))
        self.model.add(Dropout(0.2))
        # Adding a second GRU layer and some Dropout regularisation
        self.model.add(GRU(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding a third GRU layer and some Dropout regularisation
        self.model.add(GRU(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding a fourth GRU layer and some Dropout regularisation
        self.model.add(GRU(units=self.units))
        self.model.add(Dropout(0.2))
        # Adding the output layer
        self.model.add(Dense(units=1))
        
        # Compiling the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        return self.model


class LSTM_GRU_model:
    def __init__(self, units, input_shape):
        
        self.units = units
        self.input_shape0 = input_shape[0]
        self.input_shape1 = input_shape[1]

    def build(self):
        # build the model
        self.model = Sequential()
        #Adding the first GRU layer and some Dropout regularisation
        self.model.add(GRU(units=self.units, return_sequences=True, input_shape=(self.input_shape0, self.input_shape1)))
        self.model.add(Dropout(0.2))
        # Adding a second LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding a third GRU layer and some Dropout regularisation
        self.model.add(GRU(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding a fourth LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=self.units))
        self.model.add(Dropout(0.2))
        # Adding the output layer
        self.model.add(Dense(units=1))
        
        # Compiling the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        return self.model