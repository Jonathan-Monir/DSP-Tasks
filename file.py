# import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np
import streamlit as st
import pandas as pd

def Continuous_plot(x, y, plot_name):
    df = pd.DataFrame({'x': x, 'y': y})
    fig = px.line(df, x='x', y='y', title='Continuous signal')
    fig.update_traces(mode='lines')
    
    return fig

file_path = "signal1.txt"

def clean_text(file_path):
    signals = list()
    with open(file_path, 'r') as file:
        for line in file:
            signals.append(line.strip())
            
    x = [item.split(' ')[0] for item in signals[3:]]
    y = [item.split(' ')[1] for item in signals[3:]]
    z = 0
    if signals[0] == '1':
        z = [item.split(' ')[2] for item in signals[3:]]
    return x, y, z
        
    
x = clean_text(file_path)[0]
y = clean_text(file_path)[1]
Continuous_plot(x,y,'hi').show()