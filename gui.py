# import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np
import streamlit as st
import pandas as pd
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


def Continuous_plot(x, y, plot_name):
    df = pd.DataFrame({'x': x, 'y': y_input})
    fig = px.line(df, x='x', y='y', title='Continuous signal')
    fig.update_traces(mode='lines')
    
    return fig

def Discrete_signal_generator(trig_func, Amplitude, Analog_freq, Sample_freq, phase_shift, n):
    
    trig_func = getattr(math,trig_func)
    w = 2 * math.pi * (Analog_freq/Sample_freq)
    y = Amplitude * trig_func(w*n + phase_shift)
    
    return y 

import math



# inputs
trig_func = st.radio('Trig function',['sin','cos'])
x = range(st.number_input("number of samples",min_value=1,max_value=10000,value=200))
amp = st.number_input("Amplitude",min_value=1,max_value=1000,value=10)
Analog_freq = st.number_input("Analog frequency",min_value=0,max_value=10000,value=5)
Sample_freq = st.number_input("Sample frequency",min_value=1,max_value=10000,value=200)
phase_shift = st.number_input("Phase shift",min_value=float(0),max_value=float(100),step=0.2)

fs_min = 2 * Analog_freq

if Sample_freq < fs_min:
    raise Exception(f"Warning: The chosen sampling frequency ({Sample_freq} Hz) is below the minimum required ({fs_min} Hz).")
    
vectorized_function = np.vectorize(Discrete_signal_generator)
y_input = vectorized_function(trig_func, amp, Analog_freq, Sample_freq, phase_shift, x)
st.plotly_chart(Discrete_plot(x, y_input, "Discrete Signal"))


st.plotly_chart(Continuous_plot(x, y_input, "Continuous Signal"),use_container_width=False)

