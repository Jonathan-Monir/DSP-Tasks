import streamlit as st
if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
if "output_signals" in st.session_state["output_signals"]:
    output_signals = st.session_state["output_signals"]
    
