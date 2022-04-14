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
        print("start", str(start_copy)[:10])
        print("end", str(end_copy)[:10])
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
    # Rename the Datetime column"
    if "Datetime" in data.columns:
        data.rename(columns={"Datetime": "Date"}, inplace=True)
    if "index" in data.columns:
        data.rename(columns={"index": "Date"}, inplace=True)

    # Keep the useful columns and drop the others
    data = data[["Date", "Open", "Close"]]
    # Drop duplicate rows
    data = data.drop_duplicates(["Date"])
    # Drop NaN
    data.dropna(inplace=True)
    return data



