def get_indicators(dataset, col):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset[col].rolling(window=7).mean()
    # dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    
    # Create MACD
    # dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    # dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    # dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    # print(type(dataset['Close']))
    # dataset['20sd'] = dataset['Close'].rolling(20).std()

    # dataset['20sd'] = pd.stats.moments.rolling_std(dataset['Close'],20)
    # dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    # dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset[col].ewm(com=0.5).mean()
    
    # Create Momentum
    # dataset['momentum'] = dataset[col] - 1
    # dataset['log_momentum'] = np.log2(dataset['momentum']) 
    
    return dataset