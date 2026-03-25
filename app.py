import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.linalg import eigh

HBAR = 1.0
MASS = 1.0

def build_hamiltonian(x, V):
    dx = x[1] - x[0]
    n = len(x)
    kp = HBAR**2 / (2 * MASS * dx**2)

    H = np.diag(2 * kp + V) \
        + np.diag(-kp * np.ones(n - 1), 1) \
        + np.diag(-kp * np.ones(n - 1), -1)

    return H

def solve(x, V, num_states=5):
    H = build_hamiltonian(x, V)
    energies, states = eigh(H, subset_by_index=[0, num_states - 1])
    return energies, states

st.title("Quantum Mechanics Simulator")

system = st.selectbox("Potential", ["Harmonic Oscillator", "Particle in a Box"])

N = st.slider("Grid points", 100, 800, 300)
num_states = st.slider("States", 2, 8, 4)

if system == "Harmonic Oscillator":
    x = np.linspace(-5, 5, N)
    omega = 1.0
    V = 0.5 * omega**2 * x**2
else:
    x = np.linspace(0, 2, N)
    V = np.zeros_like(x)

energies, states = solve(x, V, num_states)

st.write("### Energies")
st.write(energies)

fig = go.Figure()

for i in range(num_states):
    fig.add_trace(go.Scatter(
        x=x,
        y=states[:, i] + energies[i],
        name=f"n={i}"
    ))

st.plotly_chart(fig)
