# !pip install yfinance

import pandas_datareader.data as web
import datetime
import pandas as pd 
from functools import reduce
import yfinance as yf

def get_stock(ticker, start, end, interval):
    '''
    Getting data from Yahoo Finance
    ticker: Symbol of the stock
    start: start of the range string type
    end: end of the range string type
    interval: time interval of the data --> 60m, 1d, 1wk, 1mo, 3mo 
    '''
    # Create start and end objects using datetime
    start_obj = datetime.datetime.strptime(start, '%Y-%m-%d')
    start_copy = datetime.datetime.strptime(start, '%Y-%m-%d')
    end_obj = datetime.datetime.strptime(end, '%Y-%m-%d')
    end_copy = start_obj + datetime.timedelta(days=365)
    # Initial data
    data = yf.download(tickers=ticker, start=str(start_copy)[:10], end=str(end_copy)[:10], interval=interval)

    while (end_copy <= end_obj):
        # Download the dataset from yahoo finance
        df = yf.download(tickers=ticker, start=str(start_copy)[:10], end=str(end_copy)[:10], interval=interval)
        
        # print("start", str(start_copy)[:10])
        # print("end", str(end_copy)[:10])
        
        # Append the new data to the dataset
        data = data.append(df)
      
        if end_copy == end_obj:
            break

        # Update the begining point and ending points
        start_copy = start_copy + datetime.timedelta(days=365)
        if (end_copy + datetime.timedelta(days=365) <= end_obj):
            end_copy = end_copy + datetime.timedelta(days=365)
        else:
            end_copy = end_obj

    # Reset index
    data.reset_index(inplace=True, drop=False)
    
    Date = []
    Time = []
    
    if "Datetime" in data.columns:
        data["Date"] = data["Datetime"]
        date = data["Date"].to_list()
            
        for i in date:
            Datetime = str(i)
            date = Datetime[:10]
            time = Datetime[11:16]
            # Change the date format
            # date = str(datetime.datetime.strptime(date, '%Y-%m-%d'))[10]
            # date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y')
            Date.append(date)
            Time.append(time)
    
    else:

        if "Date" not in data.columns:
            data.rename(columns={"index": "Date"}, inplace=True)

        Time = ["00:00"] * len(data)
        data["Time"] = Time
        for i in data['Date']:
            date = str(i)[:10]
            # Change the date format
            # date = str(datetime.datetime.strptime(date, '%Y-%m-%d'))[10]
            Date.append(date)
            # Date.append(datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y'))
        

    # Update the Dataset
    data["Date"] = Date
    data["Time"] = Time

    # Change Date and Time column position
    # Time = data.pop('Time')
    # data.insert(0, 'Time', Time)    
    # Date = data.pop('Date')
    # data.insert(0, 'Date', Date)


    # Drop duplicate rows
    data = data.drop_duplicates(["Date", "Time"])
    # Create Datetime column
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

    # Drop out of range rows
    for i in data["Datetime"]:
        if i < start_obj or i > end_obj:
            data.drop(data.index[data['Datetime'] == i], inplace=True)

    # Drop NaN
    data.dropna(inplace=True)
    # Drop Unnamed Columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Reset index
    data.reset_index(drop=True, inplace=True)
    # reorder columns
    data = data[['Date', 'Time', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    return data



# Adding rewards to dataset 
def get_reward(data):
    df = data.copy()
    reward = []
    action = []
    for i in range(len(df)):
        reward_day = (df["Close"][i] - df["Open"][i]) / df["Open"][i]
        reward_day = round(reward_day, 2)
        action_day = 1
        if reward_day < 0:
            reward_day = 0
            action_day = 0
        reward.append(reward_day)
        action.append(action_day)
    df["Reward"] = reward
    df["Action"] = action
    
    return df
