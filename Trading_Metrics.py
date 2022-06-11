
# Google Drive Connection

import pandas as pd
import numpy as np
# Trading Function
def trade(df, actions, initial_capital, long=True):

  capital = initial_capital
  stock = initial_capital / df["Open"][0]
  gain = 0
  local_df = pd.DataFrame(columns = ["Date", "Open", "Close"])

  for i in range(len(actions)):
      local_df = local_df.append(df.loc[df['Date'] == actions["Date"][i]])
  return local_df
  local_df.reset_index(drop=True, inplace=True)
  date = actions.loc[:, 'Date'].tolist()
  open = local_df.loc[:, 'Open'].tolist()
  close = local_df.loc[:, 'Close'].tolist()
  actions = actions.loc[:, 'ensemble'].tolist()
  capital_list = []
  gain_list = []

  for i in range(len(actions)):
    if long:
      if actions[i] == 1: # Action is Long
        stock_temp = capital / open[i]
        capital = round((stock_temp * close[i]), 2)
    else:
      if actions[i] == 2: # Action is Short
        capital_temp = stock * open[i]
        stock = capital_temp / close[i]
        capital = round((stock * close[i]), 2)
    
    capital_list.append(capital)
    gain = capital - initial_capital
    gain_list.append(gain)

  local_df = local_df.assign(Capital=pd.Series(capital_list))
  local_df = local_df.assign(Gain=pd.Series(gain_list))

  return local_df

def metrics(df, initial_capital):
    
    df_local = df.copy() 
    
    # Calculate rate of return
    df_local = df.assign(ROR=pd.Series(df.Capital - initial_capital)/ initial_capital * 100)

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
    std_neg = df_local[df_local["ROR"]<0].ROR.std()
    std = df_local.ROR.std()
    sortino = mean / std_neg
    sharpe = mean / std
    
    print(f"Sortino Ratio = {sortino} \n")
    print(f"Sharpe Ratio = {sharpe} \n")
    print(f"MDD = {MDD} \n")
    
    return df_local


