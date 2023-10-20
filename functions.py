import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Function to clean text
def clean_text(file_list):
    signals = list()
    for line in file_list:
        signals.append(line.strip())
            
    x = [item.split(' ')[0] for item in signals[3:]]
    y = [item.split(' ')[1] for item in signals[3:]]
    z = 0
    if signals[0] == '1':
        z = [item.split(' ')[2] for item in signals[3:]]
    return x, y, z

# Function for making Discrete plot
def Discrete_plot(x, y, plot_name):
    fig = px.scatter(x=x, y=y, title=plot_name)

    for i in range(len(x)):
        fig.add_trace(go.Scatter(x=[x[i], x[i]], y=[0, y[i], None], mode='lines', name=f'Line {i}', line=dict(color='gray', dash='dash')))

    fig.update_layout(
        xaxis_title='n',
        yaxis_title='X(n)',
        showlegend=True
    )
    
    return fig

# Function for making Continuous plot
def Continuous_plot(x, y, plot_name):
    df = pd.DataFrame({'x': x, 'y': y})
    fig = px.line(df, x='x', y='y', title=plot_name)
    fig.update_traces(mode='lines')
    
    return fig