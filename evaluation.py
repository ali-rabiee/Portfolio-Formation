import pandas as pd
import numpy as np

def full_ensemble(df):
    m1 = df.eq(1).all(axis=1)
    m2 = df.eq(2).all(axis=1)
    local_df = df.copy()
    local_df['ensemble'] = np.select([m1, m2], [1, 2], 0)
    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)
    return local_df

def perc_ensemble(df, thr=0.7):
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
    return pd.DataFrame(np.select([c1, c2], [1, 2], 0), index=df.index, columns=['ensemble'])

def ensemble(numWalks, ticker, data_type='test', numDel=0, use_walks=True):
    rewSum = 0
    posSum = 0
    negSum = 0
    covSum = 0
    numSum = 0
    rewLongSum = 0
    rewShortSum = 0
    values=[]
    columns = ["Iteration", "Reward_Long%", "Reward_Short%", "Reward%", "Wins%", "Losses%", "Coverage%", "Accuracy"]
    dax = pd.read_csv(f"./datasets/{ticker}day.csv", index_col='Date')

    for j in range(0, numWalks):
        
        if use_walks:
                df = pd.read_csv("./Output/ensemble/ensembleFolder/walk"+str(j)+"ensemble_"+data_type+f"_{ticker}.csv",index_col='Date')
                for deleted in range(1, numDel):
                    del df['iteration'+str(deleted)]
                df = perc_ensemble(df)
        
        else:
                df = pd.read_csv(f"./Output/results/DQN predictions/{ticker}_preds_ensemble_{data_type}.csv", index_col='Date')
                
        num = 0
        rew = 0
        rew_short = 0
        rew_long = 0 
        pos = 0
        neg = 0
        cov = 0

        for date, i in df.iterrows():
            num += 1

            if date in dax.index:
                if (i['ensemble']==1):
                    pos+= 1 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0
                    neg+= 0 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
                    rew+=(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                    rew_long+=(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                    cov+=1

                elif (i['ensemble']==2):
                    neg+= 0 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
                    pos+= 1 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0
                    rew+=-(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                    rew_short+=-(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                    cov+=1
                    
        values.append([str(round(j,2)), str(round(rew_long,2)), str(round(rew_short,2)), str(round(rew,2)),str(round((pos/num)*100,2)),str(round((neg/num)*100,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "")])
        
        rewSum+=rew
        rewLongSum+=rew_long
        rewShortSum+=rew_short
        posSum+=pos
        negSum+=neg
        covSum+=cov
        numSum+=num


    values.append(["sum", str(round(rewLongSum,2)), str(round(rewShortSum,2)), str(round(rewSum,2)),str(round(posSum,2)),str(round(negSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "")])
    
    return pd.DataFrame(values, columns= columns)