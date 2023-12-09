import streamlit as st
import math
import numpy as np
if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
else:
    st.warning("Please add input signals")
    
if "output_signals" in st.session_state:
    output_signals = st.session_state["output_signals"]

else:
    st.warning("Please add output signals")

if "input_signals" in st.session_state:
    signal1 = input_signals[0][1]
    signal2 = input_signals[1][1]


operation = st.selectbox("select operation",["normalize cross correlation","time delay","files"])