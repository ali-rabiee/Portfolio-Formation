'''
Visualization of the results 
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import datetime
import random

# to show Bar Plots:
def bar_plot(list1, list2, dates, bar_names, title):
    data = {
        'Date' : [],
        'Daily return' : [],
        'Returns' : [],
    }
    for i in range(len(list1)):
        data['Date'].append(dates[i])
        data['Daily return'].append(list1[i])
        data['Date'].append(dates[i])
        data['Daily return'].append(list2[i])
        data['Returns'].append(bar_names[0])
        data['Returns'].append(bar_names[1])
    
    df = pd.DataFrame(data)

    fig = px.bar(df, x="Date", y="Daily return",
                 color='Returns', barmode='group',
                 height=500, title=title)
    fig.update_layout(
        font_color = 'blue'
    )
    fig.show()



# to show Line Charts:
def line_chart(returns, dates):
    
    data = {
        'date' : dates,
        'return' : returns,
    }
    
    df = pd.DataFrame(data)
    
    fig = px.line(df, x='date', y="return")
    fig.show()



# to show Pie Charts:
def pie_chart(group, percentage):
    data = {
            'group': group,
            'percentage' : percentage,
            }
    
    df = pd.DataFrame(data)
    
    fig = px.pie(df, values='percentage', names='group')
    fig.update_layout(
    margin = dict(l=300, r=300, t=30, b=30),)
    fig.show()