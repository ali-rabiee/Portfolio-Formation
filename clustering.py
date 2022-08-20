
from optparse import Values
import pandas as pd
import numpy as np
import glob
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
import datetime

def calculate_correlation(base, numdays, tickers):
    """ Define a def that prepare needed data for detecting the correlation between different stocks"""
    
    # Path of datasets in Google Drive
    path = "./datasets" # use your path
    all_files = glob.glob(path + "/*hour.csv")
    
    # Initial lists for time series(stocks) and their names
    mySeries = []
    namesofMySeries = []

    # Prepare the data
    for filename in all_files:
        # remove path from the name
        stock_name = filename.replace('hour.csv', '').replace('./datasets/', '')

        if stock_name not in tickers:
            continue

        namesofMySeries.append(stock_name)

        # Read data
        df = pd.read_csv(filename, index_col=None, header=0) 
        df = df.loc[:,["Date", "Datetime", "Close"]]
        
        # Finding the last numdays dates
        date_list = [str(base - datetime.timedelta(days=x))[:10] for x in range(numdays)]

        # Define a new DataFrame for saving the result
        df_new = pd.DataFrame(columns=["Date", "Datetime", "Close"])

        # Select the last numdays rows  
        for i in date_list:
            df_new = df_new.append(df[df['Date']==i]) 

        # Drop Date column
        df_new.drop('Date', axis=1, inplace=True)
        # Set the Datetime columns as index
        df_new.set_index("Datetime", inplace=True)
        # Ordered the data according to our date index
        df_new.sort_index(inplace=True)
        
        # Append time series 
        mySeries.append(df_new)
      

    # Check if our data is uniform in length.
    series_lengths = {len(series) for series in mySeries}

    # Remove series with zero lenght
    mySeries = [series for series in mySeries if len(series) != 0]

    # Find the longest series of the series and elongate others according to that
    max_len = max(series_lengths)
    longest_series = None
    for series in mySeries:
        if len(series) == max_len:
            longest_series = series

    # Reindexed the series that are not as long as the longest one and fill the empty dates with np.nan
    problems_index = []
    for i in range(len(mySeries)):
        if len(mySeries[i])!= max_len:
            problems_index.append(i)
            mySeries[i] = mySeries[i].reindex(longest_series.index)   

    # Used linear interpolation to fill the gap but for series that have more missing value
    for i in problems_index:
        mySeries[i].interpolate(limit_direction="both",inplace=True)

    # Normalize the Datasets
    for i in range(len(mySeries)):
        scaler = MinMaxScaler()
        mySeries[i] = MinMaxScaler().fit_transform(mySeries[i])
        mySeries[i]= mySeries[i].reshape(len(mySeries[i]))

    # Define the SOM model for clustering task
    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    som = MiniSom(som_x, som_y, len(mySeries[0]), sigma=0.3, learning_rate=0.1)
    som.random_weights_init(mySeries)
    som.train(mySeries, 50000)

    win_map = som.win_map(mySeries)

    # Mapping 
    cluster_map = []
    for idx in range(len(mySeries)):
        winner_node = som.winner(mySeries[idx])
        cluster_map.append((namesofMySeries[idx], f"Cluster {winner_node[0]*som_y + winner_node[1] + 1}"))

    clusters = pd.DataFrame(cluster_map,columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")

    data = dict(zip(namesofMySeries, mySeries))
    df_tickers = pd.DataFrame(data) 

    return som_x, som_y, win_map, df_tickers, clusters

def visualize_clusters(som_x, som_y, win_map):
    
    # Visualize time-series with average point
    fig, axs = plt.subplots(som_x,som_y,figsize=(12,12))
    fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")
    plt.show()

    # Cluster distribution
    cluster_c = []
    cluster_n = []
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                cluster_c.append(len(win_map[cluster]))
            else:
                cluster_c.append(0)
            cluster_number = x*som_y+y+1
            cluster_n.append(f"Cluster {cluster_number}")

    plt.figure(figsize=(25,5))
    plt.title("Cluster Distribution for SOM")
    plt.bar(cluster_n,cluster_c)
    plt.show()