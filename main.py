import streamlit as st
import numpy as np
from scipy.fft import fft
from scipy.fft import ifft
import math
import matplotlib.pyplot as plt
def format_samples(signal):
    return "\n".join([f"{x} {y}" for x, y in enumerate(signal)])

def parse_signal_data(data):
    x_values, y_values = [], []
    for line in data:
        if ',' in line:
            parts = line.strip().split(',')
        else:
            parts = line.strip().split()
        if len(parts) == 2:
            x, y = parts
            x = x.replace('f', '')
            y = y.replace('f', '')
            try:
                x_values.append(float(x))
                y_values.append(float(y))
            except ValueError:
                # Handle invalid data here (e.g., skip or log it)
                pass
    return x_values, y_values

def SignalCompareAmplitude(SignalInput, SignalOutput):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
                return False
    return True

def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))

def SignalComparePhaseShift(SignalInput, SignalOutput):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A = RoundPhaseShift(SignalInput[i])
            B = RoundPhaseShift(SignalOutput[i])
            if abs(A - B) > 0.0001:
                return False
    return True
    

def main():
    st.image("logo.png")
if __name__ == "__main__":
    main()