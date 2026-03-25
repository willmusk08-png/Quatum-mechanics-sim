"""
Quantum Mechanics Simulator
===========================
Interactive 1D Schrödinger equation solver with:
  • Six quantum potentials (Particle-in-a-Box, Harmonic Oscillator, Double Well,
    Finite Square Well, Morse, Kronig-Penney)
  • Time-dependent wave-packet evolution via eigenstate expansion
  • Superposition builder with automatic normalisation
  • Momentum-space analysis via FFT
  • Expectation values, uncertainties, and dipole matrix analysis
  • Analytical benchmarking where available
  • Grid-convergence/error analysis
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import eigh
from scipy.fft import fft, fftfreq, fftshift
HBAR = 1.0
MASS = 1.0
st.set_page_config(
    page_title="Quantum Mechanics Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background-color: #050a14;
        background-image:
            linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    section[data-testid="stSidebar"] {
        background: #090f1e !important;
        border-right: 1px solid rgba(0,212,255,0.12);
    }
    section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

    h1 {
        font-family: 'Space Mono', monospace !important;
        font-size: 2.1rem !important;
        background: linear-gradient(100deg, #00d4ff, #7c5cff, #00ffb3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: #00d4ff !important;
    }

    p, li { color: #8ba8c4; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(0,212,255,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4a6d8c;
        padding: 0.6rem 1.4rem;
        border-bottom: 2px solid transparent;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
        background: transparent !important;
    }

    [data-testid="metric-container"] {
        background: rgba(0,212,255,0.04);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }
    [data-testid="metric-container"] label {
        color: #4a7a99 !important;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-family: 'Space Mono', monospace;
    }

    .stAlert {
        background: rgba(0,212,255,0.06) !important;
        border-color: rgba(0,212,255,0.25) !important;
        color: #8ba8c4 !important;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(5,10,20,0.6)",
    font=dict(family="DM Sans", color="#8ba8c4", size=12),
    xaxis=dict(
        gridcolor="rgba(0,212,255,0.08)",
        linecolor="rgba(0,212,255,0.2)",
        tickcolor="rgba(0,212,255,0.2)",
        zerolinecolor="rgba(0,212,255,0.15)",
    ),
    yaxis=dict(
        gridcolor="rgba(0,212,255,0.08)",
        linecolor="rgba(0,212,255,0.2)",
        tickcolor="rgba(0,212,255,0.2)",
        zerolinecolor="rgba(0,212,255,0.15)",
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,212,255,0.15)", borderwidth=1),
    margin=dict(l=60, r=20, t=50, b=50),
)

STATE_COLORS = [
    "#00d4ff", "#7c5cff", "#00ffb3", "#ff6b6b",
    "#ffd93d", "#ff8e00", "#a8ff78", "#ff5e99",
    "#74c0fc", "#c77dff",
]


def merge_plot_layout(**overrides):
    """Safely merge nested Plotly layout keys."""
    layout = {k: (v.copy() if isinstance(v, dict) else v) for k, v in PLOT_LAYOUT.items()}
    for key in ("font", "xaxis", "yaxis", "legend", "margin"):
        if key in overrides:
            merged = dict(layout.get(key, {}))
            merged.update(overrides.pop(key))
            layout[key] = merged
    layout.update(overrides)
    return layout


def build_hamiltonian(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    n = len(x)
    kp = HBAR**2 / (2 * MASS * dx**2)
    H = np.diag(2 * kp + V) + np.diag(-kp * np.ones(n - 1), 1) + np.diag(-kp * np.ones(n - 1), -1)
    return H


def normalize_states(x: np.ndarray, states: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    norms = np.sqrt(np.sum(np.abs(states) ** 2, axis=0) * dx)
    return states / norms


def solve_schrodinger(x: np.ndarray, V: np.ndarray, num_states: int):
    H = build_hamiltonian(x, V)
    energies, states = eigh(H, subset_by_index=[0, num_states - 1])
    states = normalize_states(x, states)
    return energies, states

def make_particle_in_box(L: float, N: int):
    x = np.linspace(0, L, N + 2)[1:-1]
    V = np.zeros_like(x)
    return x, V
def make_harmonic(x_max: float, N: int, omega: float):
    x = np.linspace(-x_max, x_max, N)
    V = 0.5 * MASS * omega**2 * x**2
    return x, V


def make_double_well(x_max: float, N: int, a: float, b: float):
    x = np.linspace(-x_max, x_max, N)
    V = a * x**4 - b * x**2
    V -= V.min()
    return x, V


def make_finite_square_well(x_max: float, N: int, V0: float, width: float):
    x = np.linspace(-x_max, x_max, N)
    V = np.where(np.abs(x) < width / 2, 0.0, V0)
    return x, V


def make_morse(x_max: float, N: int, D: float, alpha: float, x_eq: float):
    x = np.linspace(0.1, x_max, N)
    V = D * (1 - np.exp(-alpha * (x - x_eq))) ** 2
    return x, V


def make_kronig_penney(x_max: float, N: int, V0: float, period: float, barrier_frac: float):
    x = np.linspace(-x_max, x_max, N)
    barrier_w = period * barrier_frac
    phase = np.mod(x - x.min(), period)
    V = np.where(phase < barrier_w, V0, 0.0)
    return x, V

def analytical_pib(n_vals, L):
    n_vals = np.asarray(n_vals)
    return n_vals**2 * np.pi**2 * HBAR**2 / (2 * MASS * L**2)


def analytical_ho(n_vals, omega):
    n_vals = np.asarray(n_vals)
    return HBAR * omega * (n_vals + 0.5)


def morse_approx(n_vals, D, alpha):
    n = np.asarray(n_vals)
    omega = alpha * np.sqrt(2 * D / MASS)
    return HBAR * omega * (n + 0.5) - (HBAR * omega) ** 2 * (n + 0.5) ** 2 / (4 * D)

def expectation(x: np.ndarray, psi: np.ndarray, op: np.ndarray) -> float:
    dx = x[1] - x[0]
    return float(np.real(np.sum(np.conj(psi) * op * psi) * dx))


def expectation_x(x, psi):
    return expectation(x, psi, x)


def expectation_x2(x, psi):
    return expectation(x, psi, x**2)


def expectation_p(x, psi):
    dx = x[1] - x[0]
    dpsi = np.gradient(psi, dx)
    return float(np.real(-1j * HBAR * np.sum(np.conj(psi) * dpsi) * dx))


def expectation_p2(x, psi):
    dx = x[1] - x[0]
    d2psi = np.gradient(np.gradient(psi, dx), dx)
    return float(np.real(-HBAR**2 * np.sum(np.conj(psi) * d2psi) * dx))


def delta_x(x, psi):
    x1 = expectation_x(x, psi)
    x2 = expectation_x2(x, psi)
    return np.sqrt(max(x2 - x1**2, 0.0))


def delta_p(x, psi):
    p1 = expectation_p(x, psi)
    p2 = expectation_p2(x, psi)
    return np.sqrt(max(p2 - p1**2, 0.0))


def momentum_space(x: np.ndarray, psi: np.ndarray):
    dx = x[1] - x[0]
    n = len(x)
    phi = fftshift(fft(psi)) * dx / np.sqrt(2 * np.pi)
    p = fftshift(fftfreq(n, d=dx)) * 2 * np.pi * HBAR
    return p, phi


def build_superposition(states: np.ndarray, coefficients: np.ndarray, x: np.ndarray):
    psi = states @ coefficients
    dx = x[1] - x[0]
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi


def time_evolve(states: np.ndarray, energies: np.ndarray, coefficients: np.ndarray, t: float):
    phases = np.exp(-1j * energies * t / HBAR)
    return states @ (coefficients * phases)


def transition_dipole_matrix(x: np.ndarray, states: np.ndarray):
    dx = x[1] - x[0]
    n = states.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = abs(np.sum(np.conj(states[:, i]) * x * states[:, j]) * dx)
    return M


def suggested_warnings(system: str, x: np.ndarray, V: np.ndarray, energies: np.ndarray, num_states: int):
    warnings = []
    dx = x[1] - x[0]
    if dx > 0.08:
        warnings.append("Grid spacing is fairly coarse. Increase grid points for smoother eigenstates and lower energy error.")
    if system == "Harmonic Oscillator" and max(abs(x[0]), abs(x[-1])) < 5:
        warnings.append("Oscillator domain is somewhat tight. Higher-energy states may be clipped at the boundaries.")
    if system == "Finite Square Well" and energies[-1] > 0.85 * np.max(V):
        warnings.append("Highest requested state is approaching the barrier height; bound-state interpretation becomes less clean.")
    if system == "Particle in a Box" and num_states > 7 and len(x) < 300:
        warnings.append("Many excited box states on a modest grid can show visible discretisation error.")
    return warnings
def fig_eigenstates(x, V, energies, states, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=V, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=2, dash="dot"),
        name="V(x)", fill="tozeroy", fillcolor="rgba(255,255,255,0.03)",
    ))

    spacing = np.min(np.diff(energies)) if len(energies) > 1 else max(energies[0], 1.0)
    scale = 0.40 * max(spacing, 0.2)

    for i, E in enumerate(energies):
        color = STATE_COLORS[i % len(STATE_COLORS)]
        psi = states[:, i]
        psi_shifted = scale * psi + E

        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([psi_shifted, np.full_like(x, E)[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=psi_shifted, mode="lines",
            line=dict(color=color, width=1.9),
            name=f"n={i}, E={E:.4f}",
            hovertemplate=f"x=%{{x:.3f}}<br>ψ+E=%{{y:.4f}}<extra>n={i}, E={E:.4f}</extra>",
        ))
        fig.add_hline(
            y=E,
            line=dict(color=color, width=0.8, dash="dash"),
            annotation_text=f"E{i}={E:.3f}",
            annotation_font=dict(size=9, color=color),
            annotation_position="right",
        )

    fig.update_layout(**merge_plot_layout(
        title=dict(text=title, font=dict(family="Space Mono", size=13, color="#00d4ff")),
        xaxis_title="x (a₀)",
        yaxis_title="Energy / ψ(x) offset",
        height=460,
    ))
    return fig

def fig_state_profile(x, psi, label, color="#00d4ff", mode="|ψ|²"):
    if mode == "ψ(x)":
        y = np.real(psi)
        y_title = "Re[ψ(x)]"
        fill = None
        fillcolor = None
    elif mode == "Im[ψ(x)]":
        y = np.imag(psi)
        y_title = "Im[ψ(x)]"
        fill = None
        fillcolor = None
    else:
        y = np.abs(psi) ** 2
        y_title = "|ψ(x)|²"
        fill = "tozeroy"
        fillcolor = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=2),
        fill=fill, fillcolor=fillcolor,
        name=mode,
    ))
    fig.update_layout(**merge_plot_layout(
        title=dict(text=f"{mode}  —  {label}", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        xaxis_title="x (a₀)",
        yaxis_title=y_title,
        height=300,
    ))
    return fig


def fig_momentum(p, phi, label, color="#7c5cff"):
    prob_p = np.abs(phi) ** 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p, y=prob_p, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor="rgba(124,92,255,0.12)",
        name="|φ(p)|²",
    ))
    fig.update_layout(**merge_plot_layout(
        title=dict(text=f"Momentum-Space Density — {label}", font=dict(family="Space Mono", size=13, color="#7c5cff")),
        xaxis_title="p (ℏ/a₀)",
        yaxis_title="|φ(p)|²",
        height=300,
    ))
    return fig


def fig_time_evolution(x, psi_t_list, t_list, frame_duration_ms=90, title="Wave-Packet Evolution"):
    frames = []
    for t, psi_t in zip(t_list, psi_t_list):
        prob = np.abs(psi_t) ** 2
        frames.append(go.Frame(
            data=[go.Scatter(
                x=x, y=prob, mode="lines",
                line=dict(color="#00d4ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
            )],
            name=f"{t:.2f}",
            layout=go.Layout(title_text=f"{title} — t = {t:.2f}"),
        ))

    fig = go.Figure(
        data=[go.Scatter(
            x=x, y=np.abs(psi_t_list[0]) ** 2, mode="lines",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
            name="|ψ(x,t)|²",
        )],
        frames=frames,
    )
    fig.update_layout(**merge_plot_layout(
        title=dict(text=f"{title} (Press ▶)", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        xaxis_title="x (a₀)",
        yaxis_title="|ψ(x,t)|²",
        height=420,
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=1.08, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                        label=f.name, method="animate") for f in frames],
            currentvalue=dict(prefix="t = ", font=dict(color="#00d4ff", size=11)),
            pad=dict(t=40), x=0.05, len=0.92,
            bgcolor="rgba(0,212,255,0.1)",
            bordercolor="rgba(0,212,255,0.2)",
        )],
    ))
    return fig


def fig_dipole_matrix(M, labels):
    fig = go.Figure(go.Heatmap(
        z=M, x=labels, y=labels,
        colorscale=[[0, "#050a14"], [0.5, "#7c5cff"], [1, "#00d4ff"]],
        showscale=True,
        hovertemplate="⟨%{y}|x̂|%{x}⟩ = %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**merge_plot_layout(
        title=dict(text="Transition Dipole Matrix |⟨m|x̂|n⟩|", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        height=380,
    ))
    fig.update_xaxes(title="State n", **PLOT_LAYOUT["xaxis"])
    fig.update_yaxes(title="State m", **PLOT_LAYOUT["yaxis"])
    return fig


def fig_energy_comparison(n_vals, numerical, analytical, system_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Energy Levels", "Absolute Error"), horizontal_spacing=0.12)
    xlab = [f"n={n}" for n in n_vals]

    fig.add_trace(go.Bar(name="Numerical", x=xlab, y=numerical, marker_color="#00d4ff", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(name="Analytical", x=xlab, y=analytical, marker_color="#7c5cff", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=xlab, y=np.abs(np.array(numerical) - np.array(analytical)),
        mode="lines+markers", line=dict(color="#00ffb3", width=2), marker=dict(size=7, color="#00ffb3"),
        name="Error",
    ), row=1, col=2)

    for row, col in [(1, 1), (1, 2)]:
        fig.update_xaxes(**PLOT_LAYOUT["xaxis"], row=row, col=col)
        fig.update_yaxes(**PLOT_LAYOUT["yaxis"], row=row, col=col)

    fig.update_layout(**merge_plot_layout(
        title=dict(text=f"{system_name} — Numerical vs Analytical", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        height=380,
        barmode="group",
    ))
    return fig


def fig_convergence(grid_sizes, errors, system_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid_sizes, y=errors, mode="lines+markers",
        line=dict(color="#00ffb3", width=2), marker=dict(size=8, color="#00ffb3"),
        name="Max |ΔE|",
        hovertemplate="N=%{x}<br>max |ΔE|=%{y:.3e}<extra></extra>",
    ))
    fig.update_layout(**merge_plot_layout(
        title=dict(text=f"Grid Convergence — {system_name}", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        xaxis_title="Grid points N",
        yaxis_title="max |ΔE| (first few states)",
        height=340,
    ))
    return fig


st.title("Quantum Mechanics Simulator")
st.markdown(
    "<p style='color:#4a7a99;font-size:0.85rem;margin-top:-0.5rem;'>Numerical solutions to the 1D time-independent & time-dependent Schrödinger equation</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### System")
    system = st.selectbox(
        "Potential",
        [
            "Particle in a Box",
            "Harmonic Oscillator",
            "Double Well",
            "Finite Square Well",
            "Morse Potential",
            "Kronig-Penney (Periodic)",
        ],
    )

    num_states = st.slider("Eigenstates", 2, 10, 5)
    grid_points = st.slider("Grid points", 150, 1200, 500, step=50)

    st.markdown("---")
    st.markdown("### Visualization")
    state_plot_mode = st.selectbox("State plot mode", ["|ψ|²", "ψ(x)", "Im[ψ(x)]"])
    packet_frame_ms = st.slider("Animation frame duration (ms)", 40, 200, 95, 5)
    packet_speed_factor = st.slider("Simulation speed factor", 0.25, 2.00, 0.60, 0.05)

    st.markdown("---")
    st.markdown("### ⚙ Parameters")

    if system == "Particle in a Box":
        L = st.slider("Box length L", 0.5, 8.0, 2.0, 0.1)
        x, V = make_particle_in_box(L, grid_points)
        n_vals = np.arange(1, num_states + 1)
        E_analytical = analytical_pib(n_vals, L)
        convergence_builder = lambda N: make_particle_in_box(L, N)

    elif system == "Harmonic Oscillator":
        omega = st.slider("ω (frequency)", 0.5, 4.0, 1.0, 0.1)
        x_max = st.slider("Domain half-width", 3.0, 12.0, 7.0, 0.5)
        x, V = make_harmonic(x_max, grid_points, omega)
        n_vals = np.arange(num_states)
        E_analytical = analytical_ho(n_vals, omega)
        convergence_builder = lambda N: make_harmonic(x_max, N, omega)

    elif system == "Double Well":
        a = st.slider("a (quartic coeff)", 0.1, 2.0, 0.25, 0.05)
        b = st.slider("b (quadratic coeff)", 0.5, 4.0, 1.5, 0.1)
        x_max = st.slider("Domain half-width", 3.0, 10.0, 5.0, 0.5)
        x, V = make_double_well(x_max, grid_points, a, b)
        n_vals = np.arange(num_states)
        E_analytical = None
        convergence_builder = None

    elif system == "Finite Square Well":
        V0 = st.slider("Outside barrier height V₀", 1.0, 30.0, 10.0, 0.5)
        width = st.slider("Well width", 0.5, 5.0, 2.0, 0.1)
        x_max = st.slider("Domain half-width", 3.0, 12.0, 6.0, 0.5)
        x, V = make_finite_square_well(x_max, grid_points, V0, width)
        n_vals = np.arange(num_states)
        E_analytical = None
        convergence_builder = None

    elif system == "Morse Potential":
        D_e = st.slider("Dissociation energy D", 5.0, 30.0, 10.0, 0.5)
        alpha = st.slider("α (width parameter)", 0.5, 3.0, 1.0, 0.1)
        x_eq = st.slider("Equilibrium xₑ", 0.5, 4.0, 1.5, 0.1)
        x_max = st.slider("Domain x_max", 4.0, 15.0, 8.0, 0.5)
        x, V = make_morse(x_max, grid_points, D_e, alpha, x_eq)
        n_vals = np.arange(num_states)
        E_analytical = morse_approx(n_vals, D_e, alpha)
        convergence_builder = lambda N: make_morse(x_max, N, D_e, alpha, x_eq)

    else:
        V0 = st.slider("Barrier height V₀", 1.0, 20.0, 5.0, 0.5)
        period = st.slider("Lattice period", 0.5, 4.0, 1.5, 0.1)
        barrier_frac = st.slider("Barrier width fraction", 0.1, 0.9, 0.3, 0.05)
        x_max = st.slider("Domain half-width", 3.0, 12.0, 6.0, 0.5)
        x, V = make_kronig_penney(x_max, grid_points, V0, period, barrier_frac)
        n_vals = np.arange(num_states)
        E_analytical = None
        convergence_builder = None

solve_start = time.perf_counter()
energies, states = solve_schrodinger(x, V, num_states)
solve_time_ms = 1000 * (time.perf_counter() - solve_start)
dx = x[1] - x[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Hamiltonian size", f"{len(x)} × {len(x)}")
m2.metric("Grid spacing Δx", f"{dx:.4f}")
m3.metric("Solve time", f"{solve_time_ms:.1f} ms")
m4.metric("Potential span", f"{float(np.max(V)-np.min(V)):.3f}")

for msg in suggested_warnings(system, x, V, energies, num_states):
    st.warning(msg)

(tab_eigen, tab_packet, tab_super, tab_analysis, tab_info) = st.tabs(
    ["  Eigenstates  ", "  Wave Packet  ", "  Superposition  ", "  Analysis  ", "  About  "]
)


with tab_eigen:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.plotly_chart(fig_eigenstates(x, V, energies, states, system), use_container_width=True, config={"displayModeBar": False})

    with col_right:
        selected = st.select_slider(
            "Inspect state",
            options=list(range(num_states)),
            format_func=lambda n: f"n = {n}  (E = {energies[n]:.4f})",
        )
        psi_sel = states[:, selected]
        color = STATE_COLORS[selected % len(STATE_COLORS)]

        st.plotly_chart(
            fig_state_profile(x, psi_sel, f"n={selected}", color=color, mode=state_plot_mode),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        p_sel, phi_sel = momentum_space(x, psi_sel)
        mask_sel = (p_sel > -20) & (p_sel < 20)
        st.plotly_chart(
            fig_momentum(p_sel[mask_sel], phi_sel[mask_sel], f"n={selected}", color="#7c5cff"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        dx_sel = delta_x(x, psi_sel)
        dp_sel = delta_p(x, psi_sel)
        c1, c2, c3 = st.columns(3)
        c1.metric("⟨x⟩", f"{expectation_x(x, psi_sel):.4f}")
        c2.metric("Δx", f"{dx_sel:.4f}")
        c3.metric("ΔxΔp / ℏ", f"{dx_sel * dp_sel / HBAR:.4f}")

    st.markdown("---")
    col_tbl, col_chart = st.columns([1, 2])

    with col_tbl:
        st.markdown("#### Energy Levels")
        labels = [f"n={n}" for n in n_vals]
        table = {"State": labels, "E (num)": [f"{e:.6f}" for e in energies]}
        if E_analytical is not None:
            err = np.abs(energies - E_analytical)
            table["E (analytic)"] = [f"{e:.6f}" for e in E_analytical]
            table["|ΔE|"] = [f"{e:.2e}" for e in err]
        st.dataframe(table, use_container_width=True)

    with col_chart:
        if E_analytical is not None:
            st.plotly_chart(fig_energy_comparison(n_vals, energies, E_analytical, system), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Closed-form analytical energies are not available for this potential. Numerical results are shown above.")

with tab_packet:
    st.markdown("#### Gaussian Wave-Packet Dynamics")
    st.markdown("Construct a Gaussian packet, project it onto eigenstates, and animate **ψ(x,t) = Σ cₙ φₙ(x)e^{-iEₙt/ℏ}**.")

    wp_col1, wp_col2 = st.columns(2)
    with wp_col1:
        x0 = st.slider("Centre x₀", float(x.min() * 0.7), float(x.max() * 0.7), float(np.mean(x)), 0.05)
        sigma = st.slider("Width σ", 0.1, 3.0, 0.5, 0.05)
    with wp_col2:
        k0 = st.slider("Momentum k₀", -10.0, 10.0, 2.0, 0.1)
        base_t_max = st.slider("Base max time", 0.5, 20.0, 5.0, 0.5)
        n_frames = st.slider("Animation frames", 20, 90, 45, 5)

    effective_t_max = base_t_max * packet_speed_factor
    dx_local = x[1] - x[0]
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * k0 * x)
    gauss /= np.sqrt(np.sum(np.abs(gauss) ** 2) * dx_local)

    c_n = np.array([np.sum(np.conj(states[:, i]) * gauss) * dx_local for i in range(num_states)], dtype=complex)
    t_list = np.linspace(0.0, effective_t_max, n_frames)
    psi_t_list = [time_evolve(states, energies, c_n, t) for t in t_list]
    occ = np.abs(c_n) ** 2

    st.plotly_chart(
        fig_time_evolution(x, psi_t_list, t_list, frame_duration_ms=packet_frame_ms, title="Wave-Packet Evolution"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    occ_fig = go.Figure(go.Bar(
        x=[f"n={i}" for i in range(num_states)],
        y=occ,
        marker_color=STATE_COLORS[:num_states],
        opacity=0.85,
        text=[f"{v:.3f}" for v in occ],
        textposition="outside",
        textfont=dict(color="#8ba8c4", size=10),
    ))
    occ_fig.update_layout(**merge_plot_layout(
        title=dict(text="Eigenstate Decomposition |cₙ|²", font=dict(family="Space Mono", size=12, color="#00d4ff")),
        height=260,
        yaxis_range=[0, max(float(np.max(occ)) * 1.25, 0.1)],
        xaxis_title="State",
        yaxis_title="|cₙ|²",
    ))
    st.plotly_chart(occ_fig, use_container_width=True, config={"displayModeBar": False})

with tab_super:
    st.markdown("#### Superposition State Builder")
    st.markdown("Set coefficients cₙ for each eigenstate. The simulator normalises automatically.")

    raw_coeffs = []
    coeffs_per_row = 5
    for row_start in range(0, num_states, coeffs_per_row):
        cols = st.columns(min(coeffs_per_row, num_states - row_start))
        for offset, col in enumerate(cols):
            i = row_start + offset
            default = 1.0 if i == 0 else 0.0
            raw_coeffs.append(col.number_input(f"c_{i}", value=default, step=0.1, key=f"coeff_{i}"))

    c_arr = np.array(raw_coeffs, dtype=complex)
    norm_c = np.sqrt(np.sum(np.abs(c_arr) ** 2))

    if norm_c < 1e-12:
        st.warning("At least one coefficient must be non-zero.")
    else:
        c_arr /= norm_c
        psi_super = build_superposition(states, c_arr, x)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.plotly_chart(
                fig_state_profile(x, psi_super, "Superposition state", color="#00ffb3", mode=state_plot_mode),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with s_col2:
            p, phi = momentum_space(x, psi_super)
            mask = (p > -20) & (p < 20)
            st.plotly_chart(fig_momentum(p[mask], phi[mask], "Superposition state"), use_container_width=True, config={"displayModeBar": False})

        st.markdown("**Animate this superposition state**")
        t_end = st.slider("Evolution time", 0.5, 15.0, 4.0, 0.5, key="super_t")
        n_fr = st.slider("Frames", 20, 70, 35, 5, key="super_fr")
        t_lst = np.linspace(0.0, t_end * packet_speed_factor, n_fr)
        psi_t_lst = [time_evolve(states, energies, c_arr, t) for t in t_lst]
        st.plotly_chart(
            fig_time_evolution(x, psi_t_lst, t_lst, frame_duration_ms=packet_frame_ms, title="Superposition evolution"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        dx_s = delta_x(x, psi_super)
        dp_s = delta_p(x, psi_super)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⟨x⟩", f"{expectation_x(x, psi_super):.4f}")
        m2.metric("Δx", f"{dx_s:.4f}")
        m3.metric("Δp", f"{dp_s:.4f}")
        m4.metric("ΔxΔp / ℏ", f"{dx_s * dp_s / HBAR:.4f}")


with tab_analysis:
    st.markdown("#### Per-Eigenstate Analysis")

    rows = []
    unc_vals = []
    for i in range(num_states):
        psi = states[:, i]
        dx_i = delta_x(x, psi)
        dp_i = delta_p(x, psi)
        unc_vals.append(dx_i * dp_i / HBAR)
        rows.append({
            "State": f"n={i}",
            "Energy": f"{energies[i]:.6f}",
            "⟨x⟩": f"{expectation_x(x, psi):.4f}",
            "⟨p⟩": f"{expectation_p(x, psi):.4f}",
            "Δx": f"{dx_i:.4f}",
            "Δp": f"{dp_i:.4f}",
            "ΔxΔp/ℏ": f"{dx_i * dp_i / HBAR:.5f}",
        })
    st.dataframe(rows, use_container_width=True)

    unc_fig = go.Figure()
    unc_fig.add_hline(
        y=0.5,
        line=dict(color="#ff6b6b", dash="dash", width=1.5),
        annotation_text="Heisenberg limit  ΔxΔp = ℏ/2",
        annotation_font=dict(color="#ff6b6b", size=10),
    )
    unc_fig.add_trace(go.Scatter(
        x=[f"n={i}" for i in range(num_states)],
        y=unc_vals,
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=10, color=STATE_COLORS[:num_states], line=dict(width=1.5, color="#050a14")),
        text=[f"{v:.4f}" for v in unc_vals],
        textposition="top center",
        textfont=dict(color="#8ba8c4", size=10),
        name="ΔxΔp/ℏ",
    ))
    unc_fig.update_layout(**merge_plot_layout(
        title=dict(text="Heisenberg Uncertainty Product ΔxΔp/ℏ", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        height=320,
        xaxis_title="State",
        yaxis_title="ΔxΔp / ℏ",
    ))
    st.plotly_chart(unc_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### Momentum-Space Probability Densities")
    p_cols = st.columns(min(num_states, 4))
    for i, col in enumerate(p_cols[:num_states]):
        psi = states[:, i]
        p, phi = momentum_space(x, psi)
        mask = (p > -15) & (p < 15)
        color = STATE_COLORS[i % len(STATE_COLORS)]
        mini_fig = go.Figure(go.Scatter(
            x=p[mask], y=np.abs(phi[mask]) ** 2, mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
        ))
        mini_fig.update_layout(**merge_plot_layout(
            title=dict(text=f"n={i}", font=dict(size=11, color=color)),
            height=200,
            showlegend=False,
            margin=dict(l=30, r=10, t=30, b=30),
        ))
        mini_fig.update_xaxes(title="p", **PLOT_LAYOUT["xaxis"])
        mini_fig.update_yaxes(title="|φ|²", **PLOT_LAYOUT["yaxis"])
        col.plotly_chart(mini_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### Transition Dipole Matrix |⟨m|x̂|n⟩|")
    M = transition_dipole_matrix(x, states)
    labels = [f"n={i}" for i in range(num_states)]
    st.plotly_chart(fig_dipole_matrix(M, labels), use_container_width=True, config={"displayModeBar": False})
    st.caption("Non-zero off-diagonal elements indicate electric-dipole-allowed transitions.")

    st.markdown("---")
    st.markdown("#### Grid Convergence / Error Analysis")
    if E_analytical is None or convergence_builder is None:
        st.info("Grid-convergence benchmarking is shown when a closed-form analytical reference is available.")
    else:
        test_sizes = [200, 300, 450, 600, 800, 1000]
        test_sizes = [n for n in test_sizes if n >= max(150, num_states * 20)]
        max_errors = []
        with st.spinner("Computing convergence sweep..."):
            for N in test_sizes:
                x_test, V_test = convergence_builder(N)
                E_test, _ = solve_schrodinger(x_test, V_test, num_states)
                max_errors.append(float(np.max(np.abs(E_test - E_analytical))))
        st.plotly_chart(fig_convergence(test_sizes, max_errors, system), use_container_width=True, config={"displayModeBar": False})

with tab_info:
    st.markdown("#### What this application does")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
**Numerical method**
- Finite-difference discretisation of the kinetic-energy operator on a uniform grid
- Hamiltonian assembled as a symmetric tridiagonal matrix
- Dense `scipy.linalg.eigh` eigendecomposition for the lowest eigenpairs
- Wavefunctions normalised by numerical quadrature

**Potentials implemented**
- Particle in a Box
- Harmonic Oscillator
- Double Well
- Finite Square Well
- Morse Potential
- Kronig-Penney periodic lattice
            """
        )

    with col_b:
        st.markdown(
            """
**Physics computed**
- Energy eigenvalues and eigenfunctions
- Time evolution of Gaussian packets and arbitrary superpositions
- Momentum-space densities from FFTs
- Expectation values ⟨x⟩, ⟨p⟩, ⟨x²⟩, ⟨p²⟩
- Heisenberg uncertainty products
- Electric-dipole transition matrix elements
- Numerical vs analytical benchmarking and grid convergence
            """
        )

    st.markdown("---")
    st.markdown(
        """
**Stack:** Python · NumPy · SciPy · Plotly · Streamlit

All quantities are shown in dimensionless units with ℏ = m = 1.
        """
    )
