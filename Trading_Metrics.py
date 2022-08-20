

import pandas as pd
import numpy as np
# Trading Function
def trade(df, date, initial_capital, long=True):

    capital = initial_capital
    stock = initial_capital / df["Open"][0]
    gain = 0

    local_df = df[df['Date'] == date]
    local_df.reset_index(drop=True, inplace=True)

    close = local_df['Close'][0]
    open = local_df['Open'][0]

    if long:
        stock_temp = capital / open
        capital = round((stock_temp * close), 2)

    else:
        capital_temp = stock * open
        stock = capital_temp / close
        capital = round((stock * close), 2)

    return capital

def metrics(df, initial_capital):
    
    df_local = df.copy() 
    
    # Return
    df_local = df.assign(Return=pd.Series(df.Capital - initial_capital))

    # Calculate rate of return
    df_local = df_local.assign(ROR=pd.Series(df.Capital - initial_capital)/ initial_capital * 100)

    # Calculate Drawdown
    max_ror = max(df_local.ROR)
    DD_list = []
    for ror in df_local.ROR:
        DD_list.append(max_ror - ror)
    df_local = df_local.assign(DD=pd.Series(DD_list))   
    
    # Calculate Maximum DrawDown
    MDD = round(max(abs(df_local.DD)),2)
    
    # Add RRR column
    df_local['RRR'] = (df_local['ROR'] / MDD) * 100

    # Calculate Sortino Ratio and Sharpe Ratio
    mean = df_local.ROR[-1:].values[0]
    std_neg = df_local[df_local["ROR"] < 0].ROR.std()
    std = df_local.ROR.std()
    sortino = mean / std_neg
    sharpe = mean / std
    
    print(f"Sortino Ratio = {sortino} \n")
    print(f"Sharpe Ratio = {sharpe} \n")
    print(f"MDD = {MDD} \n")
    
    return df_local

def buy_and_hold(portfo, start, end, initial_capital):
    df = pd.read_csv(f'./datasets/AAPLday.csv')
    df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    Dates = df['Date'].to_list()
    stocks = []
    prices = []

    for item in portfo:
        ticker = item[0]
        df = pd.read_csv(f"./datasets/{ticker}day.csv")
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        close_price = df['Close'].to_list()
        prices.append(close_price)

        weight = item[1]
        capital = weight * initial_capital
        stock_amount = capital / close_price[0]
        stocks.append(stock_amount)

    returns = {}

    for i, date in enumerate(Dates):
        if date == start:
            returns[date] = initial_capital
        
        ret = 0
        for j, stock_amount in enumerate(stocks):
            close_price = prices[j][i]
            ret += close_price * stock_amount

        returns[date] = ret
    return returns
