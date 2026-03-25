"""
Quantum Mechanics Simulator
============================
Interactive 1D Schrödinger equation solver with:
  • Six quantum potentials (Particle-in-Box, Harmonic Oscillator, Double Well,
    Finite Square Well, Morse, Kronig-Penney)
  • Time-dependent wave-packet evolution (eigenstate expansion)
  • Superposition state builder
  • Momentum-space (Fourier) analysis
  • Expectation values & Heisenberg uncertainty
  • Transition dipole-moment matrix
  • Fully interactive Plotly visualizations
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.linalg import eigh
from scipy.fft import fft, fftfreq, fftshift

# ── Constants (dimensionless / atomic units) ──────────────────────────────────
HBAR = 1.0
MASS = 1.0

# ── Page config & custom CSS ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Mechanics Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Dark background with subtle grid */
    .stApp {
        background-color: #050a14;
        background-image:
            linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #090f1e !important;
        border-right: 1px solid rgba(0,212,255,0.12);
    }
    section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label { color: #7aadcc !important; font-size:0.78rem; letter-spacing:0.06em; text-transform: uppercase; }

    /* Main title */
    h1 {
        font-family: 'Space Mono', monospace !important;
        font-size: 2.1rem !important;
        background: linear-gradient(100deg, #00d4ff, #7c5cff, #00ffb3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        padding-bottom: 0.1rem;
    }

    h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: #00d4ff !important;
        letter-spacing: -0.01em;
    }

    p, li { color: #8ba8c4; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid rgba(0,212,255,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4a6d8c;
        border: none;
        background: transparent;
        padding: 0.6rem 1.4rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
        background: transparent !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(0,212,255,0.04);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }
    [data-testid="metric-container"] label { color: #4a7a99 !important; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-family:'Space Mono',monospace; font-size:1.3rem; }

    /* Dataframe */
    .dataframe { background: #090f1e !important; color: #8ba8c4 !important; }

    /* Divider */
    hr { border-color: rgba(0,212,255,0.1) !important; }

    /* Slider accent */
    .stSlider [data-baseweb="slider"] div[role="slider"] { background:#00d4ff !important; border-color:#00d4ff !important; }

    /* Expander */
    .streamlit-expanderHeader { color: #4a7a99 !important; font-size:0.8rem; font-family:'Space Mono',monospace; }

    /* Code blocks */
    code { color: #00ffb3 !important; background: rgba(0,255,179,0.07) !important; }

    /* Info / warning boxes */
    .stAlert { background: rgba(0,212,255,0.06) !important; border-color: rgba(0,212,255,0.25) !important; color: #8ba8c4 !important; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Plotly theme ──────────────────────────────────────────────────────────────
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
]

# ── Hamiltonian solver ────────────────────────────────────────────────────────

def build_hamiltonian(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    n  = len(x)
    kp = HBAR**2 / (2 * MASS * dx**2)
    H  = np.diag(2 * kp + V) + np.diag(-kp * np.ones(n - 1), 1) + np.diag(-kp * np.ones(n - 1), -1)
    return H


def normalize_states(x: np.ndarray, states: np.ndarray) -> np.ndarray:
    dx    = x[1] - x[0]
    norms = np.sqrt(np.sum(np.abs(states) ** 2, axis=0) * dx)
    return states / norms


def solve_schrodinger(x: np.ndarray, V: np.ndarray, num_states: int):
    H = build_hamiltonian(x, V)
    energies, states = eigh(H, subset_by_index=[0, num_states - 1])
    states = normalize_states(x, states)
    return energies, states

# ── Potentials ────────────────────────────────────────────────────────────────

def make_particle_in_box(L: float, N: int):
    x = np.linspace(0, L, N + 2)[1:-1]
    V = np.zeros_like(x)
    return x, V


def make_harmonic(x_max: float, N: int, omega: float):
    x = np.linspace(-x_max, x_max, N)
    V = 0.5 * MASS * omega**2 * x**2
    return x, V


def make_double_well(x_max: float, N: int, a: float, b: float):
    """V = a*x⁴ - b*x²"""
    x = np.linspace(-x_max, x_max, N)
    V = a * x**4 - b * x**2
    V -= V.min()
    return x, V


def make_finite_square_well(x_max: float, N: int, V0: float, width: float):
    x = np.linspace(-x_max, x_max, N)
    V = np.where(np.abs(x) < width / 2, 0.0, V0)
    return x, V


def make_morse(x_max: float, N: int, D: float, alpha: float, x_eq: float):
    """D(1 - e^{-α(x-x_e)})²"""
    x = np.linspace(0.1, x_max, N)
    V = D * (1 - np.exp(-alpha * (x - x_eq))) ** 2
    return x, V


def make_kronig_penney(x_max: float, N: int, V0: float, period: float, barrier_frac: float):
    """Periodic square barrier lattice."""
    x = np.linspace(-x_max, x_max, N)
    barrier_w = period * barrier_frac
    V = np.where(np.mod(x, period) < barrier_w, V0, 0.0)
    return x, V

# ── Analytical solutions ──────────────────────────────────────────────────────

def analytical_pib(n_vals, L):
    return (np.array(n_vals) ** 2 * np.pi**2 * HBAR**2) / (2 * MASS * L**2)


def analytical_ho(n_vals, omega):
    return HBAR * omega * (np.array(n_vals) + 0.5)


def morse_approx(n_vals, D, alpha):
    omega = alpha * np.sqrt(2 * D / MASS)
    x_e   = omega / (2 * D)           # anharmonicity
    n     = np.array(n_vals)
    return HBAR * omega * (n + 0.5) - (HBAR * omega)**2 / (4 * D) * (n + 0.5) ** 2

# ── Quantum analysis helpers ──────────────────────────────────────────────────

def expectation(x: np.ndarray, psi: np.ndarray, op: np.ndarray) -> float:
    dx = x[1] - x[0]
    return float(np.real(np.sum(np.conj(psi) * op * psi) * dx))


def expectation_x(x, psi):
    return expectation(x, psi, x)


def expectation_x2(x, psi):
    return expectation(x, psi, x**2)


def expectation_p(x, psi):
    """⟨p⟩ via finite-difference gradient."""
    dx = x[1] - x[0]
    dpsi_dx = np.gradient(psi, dx)
    return float(np.real(-1j * HBAR * np.sum(np.conj(psi) * dpsi_dx) * dx))


def expectation_p2(x, psi):
    dx = x[1] - x[0]
    d2psi = np.gradient(np.gradient(psi, dx), dx)
    return float(np.real(-HBAR**2 * np.sum(np.conj(psi) * d2psi) * dx))


def delta_x(x, psi):
    x1 = expectation_x(x, psi)
    x2 = expectation_x2(x, psi)
    return np.sqrt(max(x2 - x1**2, 0))


def delta_p(x, psi):
    p2 = expectation_p2(x, psi)
    p1 = expectation_p(x, psi)
    return np.sqrt(max(p2 - p1**2, 0))


def momentum_space(x: np.ndarray, psi: np.ndarray):
    """FFT → φ(p) normalised."""
    dx = x[1] - x[0]
    n  = len(x)
    phi = fftshift(fft(psi)) * dx / np.sqrt(2 * np.pi)
    p   = fftshift(fftfreq(n, d=dx)) * 2 * np.pi * HBAR
    return p, phi


def build_superposition(states: np.ndarray, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalised superposition state from real coefficients."""
    psi = states @ coefficients
    dx  = x[1] - x[0]
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi


def time_evolve(states: np.ndarray, energies: np.ndarray, coefficients: np.ndarray, t: float) -> np.ndarray:
    """ψ(x,t) = Σ cₙ φₙ(x) e^{-iEₙt/ℏ}"""
    phases = np.exp(-1j * energies * t / HBAR)
    c_t    = coefficients * phases
    return states @ c_t


def transition_dipole_matrix(x: np.ndarray, states: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    n  = states.shape[1]
    M  = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = float(
                np.abs(np.sum(np.conj(states[:, i]) * x * states[:, j]) * dx)
            )
    return M

# ── Plotting helpers ──────────────────────────────────────────────────────────

def fig_eigenstates(x, V, energies, states, title):
    fig = go.Figure()

    # Potential
    fig.add_trace(go.Scatter(
        x=x, y=V, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=2, dash="dot"),
        name="V(x)", fill="tozeroy",
        fillcolor="rgba(255,255,255,0.03)"
    ))

    # Scale wavefunctions relative to energy-level spacing
    if len(energies) > 1:
        spacing = np.min(np.diff(energies))
    else:
        spacing = max(energies[0], 1.0)
    scale = 0.40 * spacing

    for i, (E, color) in enumerate(zip(energies, STATE_COLORS[:len(energies)])):
        psi = states[:, i]
        psi_shifted = scale * psi + E

        # Fill between wavefunction and energy level
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

        # Wavefunction curve
        fig.add_trace(go.Scatter(
            x=x, y=psi_shifted, mode="lines",
            line=dict(color=color, width=1.8),
            name=f"n={i}  E={E:.4f}",
            hovertemplate="x=%{x:.3f}<br>ψ+E=%{y:.4f}<extra>n=%d E=%.4f</extra>" % (i, E),
        ))

        # Energy level dashes
        fig.add_hline(
            y=E, line=dict(color=color, width=0.8, dash="dash"),
            annotation_text=f"E{i}={E:.3f}",
            annotation_font=dict(size=9, color=color),
            annotation_position="right",
        )

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=title, font=dict(family="Space Mono", size=13, color="#00d4ff")),
        xaxis_title="x (a₀)",
        yaxis_title="Energy / ψ(x) offset",
        height=460,
    )
    return fig


def fig_probability(x, psi, label, color="#00d4ff"):
    prob = np.abs(psi) ** 2
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=prob, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
        name="|ψ|²",
    ))
    # Classical turning points (where prob is appreciable)
    peak = prob.max()
    fig.add_hline(y=peak * 0.05, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"Probability Density  |ψ|²  —  {label}", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        xaxis_title="x (a₀)",
        yaxis_title="|ψ(x)|²",
        height=300,
    )
    return fig


def fig_momentum(p, phi, label, color="#7c5cff"):
    prob_p = np.abs(phi) ** 2
    fig    = go.Figure()
    fig.add_trace(go.Scatter(
        x=p, y=prob_p, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor="rgba(124,92,255,0.12)",
        name="|φ(p)|²",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"Momentum-Space Density  —  {label}", font=dict(family="Space Mono", size=13, color="#7c5cff")),
        xaxis_title="p (ℏ/a₀)",
        yaxis_title="|φ(p)|²",
        height=300,
    )
    return fig


def fig_time_evolution(x, psi_t_list, t_list, energies, states, coefficients):
    """Animated probability-density time evolution."""
    frames = []
    for t, psi_t in zip(t_list, psi_t_list):
        prob = np.abs(psi_t) ** 2
        frames.append(
            go.Frame(
                data=[go.Scatter(x=x, y=prob, mode="lines",
                                 line=dict(color="#00d4ff", width=2),
                                 fill="tozeroy", fillcolor="rgba(0,212,255,0.12)")],
                name=f"t={t:.2f}",
                layout=go.Layout(title_text=f"Wave-Packet Evolution  —  t = {t:.3f} ℏ/Eₕ"),
            )
        )

    prob0 = np.abs(psi_t_list[0]) ** 2
    fig   = go.Figure(
        data=[go.Scatter(x=x, y=prob0, mode="lines",
                         line=dict(color="#00d4ff", width=2),
                         fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
                         name="|ψ(x,t)|²")],
        layout=go.Layout(
            **PLOT_LAYOUT,
            title=dict(text="Wave-Packet Evolution  (Press ▶)", font=dict(family="Space Mono", size=13, color="#00d4ff")),
            xaxis_title="x (a₀)",
            yaxis_title="|ψ(x,t)|²",
            height=420,
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.02, y=1.08, xanchor="left",
                buttons=[
                    dict(label="▶  Play",  method="animate",
                         args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True, mode="immediate")]),
                    dict(label="⏸  Pause", method="animate",
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
        ),
        frames=frames,
    )
    return fig


def fig_dipole_matrix(M, labels):
    fig = go.Figure(go.Heatmap(
        z=M, x=labels, y=labels,
        colorscale=[[0, "#050a14"], [0.5, "#7c5cff"], [1, "#00d4ff"]],
        showscale=True,
        hovertemplate="⟨%{y}|x̂|%{x}⟩ = %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Transition Dipole Matrix  |⟨m|x̂|n⟩|", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        height=380,
        xaxis=dict(**PLOT_LAYOUT["xaxis"], title="State n"),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], title="State m"),
    )
    return fig


def fig_energy_comparison(n_vals, numerical, analytical, system_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Energy Levels", "Absolute Error"),
                        horizontal_spacing=0.12)

    # Side-by-side bar chart
    fig.add_trace(go.Bar(name="Numerical", x=[f"n={n}" for n in n_vals], y=numerical,
                         marker_color="#00d4ff", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(name="Analytical", x=[f"n={n}" for n in n_vals], y=analytical,
                         marker_color="#7c5cff", opacity=0.85), row=1, col=1)

    # Error
    error = np.abs(np.array(numerical) - np.array(analytical))
    fig.add_trace(go.Scatter(x=[f"n={n}" for n in n_vals], y=error,
                             mode="lines+markers",
                             line=dict(color="#00ffb3", width=2),
                             marker=dict(size=7, color="#00ffb3"),
                             name="Error"), row=1, col=2)

    for r, c in [(1,1),(1,2)]:
        fig.update_xaxes(**PLOT_LAYOUT["xaxis"], row=r, col=c)
        fig.update_yaxes(**PLOT_LAYOUT["yaxis"], row=r, col=c)

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"{system_name} — Numerical vs Analytical", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        height=380, barmode="group",
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# ── UI ─────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

st.title("Quantum Mechanics Simulator")
st.markdown(
    "<p style='color:#4a7a99;font-size:0.85rem;margin-top:-0.5rem;'>Numerical solutions to the 1D time-independent & time-dependent Schrödinger equation</p>",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚛ System")

    system = st.selectbox(
        "Potential",
        ["Particle in a Box", "Harmonic Oscillator", "Double Well",
         "Finite Square Well", "Morse Potential", "Kronig-Penney (Periodic)"],
    )

    num_states  = st.slider("Eigenstates", 2, 10, 5)
    grid_points = st.slider("Grid points", 150, 1200, 500, step=50)

    st.markdown("---")
    st.markdown("### ⚙ Parameters")

    # Per-system parameters
    if system == "Particle in a Box":
        L     = st.slider("Box length L", 0.5, 8.0, 2.0, 0.1)
        x, V  = make_particle_in_box(L, grid_points)
        n_vals      = np.arange(1, num_states + 1)
        E_analytical = analytical_pib(n_vals, L)

    elif system == "Harmonic Oscillator":
        omega   = st.slider("ω (frequency)", 0.5, 4.0, 1.0, 0.1)
        x_max   = st.slider("Domain half-width", 3.0, 12.0, 7.0, 0.5)
        x, V    = make_harmonic(x_max, grid_points, omega)
        n_vals       = np.arange(num_states)
        E_analytical  = analytical_ho(n_vals, omega)

    elif system == "Double Well":
        a     = st.slider("a (quartic coeff)", 0.1, 2.0, 0.25, 0.05)
        b     = st.slider("b (quadratic coeff)", 0.5, 4.0, 1.5, 0.1)
        x_max = st.slider("Domain half-width", 3.0, 10.0, 5.0, 0.5)
        x, V  = make_double_well(x_max, grid_points, a, b)
        n_vals = np.arange(num_states)
        E_analytical = None

    elif system == "Finite Square Well":
        V0    = st.slider("Well depth V₀", 1.0, 30.0, 10.0, 0.5)
        width = st.slider("Well width", 0.5, 5.0, 2.0, 0.1)
        x_max = st.slider("Domain half-width", 3.0, 12.0, 6.0, 0.5)
        x, V  = make_finite_square_well(x_max, grid_points, V0, width)
        n_vals = np.arange(num_states)
        E_analytical = None

    elif system == "Morse Potential":
        D_e   = st.slider("Dissociation energy D", 5.0, 30.0, 10.0, 0.5)
        alpha  = st.slider("α (width parameter)", 0.5, 3.0, 1.0, 0.1)
        x_eq   = st.slider("Equilibrium x_e", 0.5, 4.0, 1.5, 0.1)
        x_max  = st.slider("Domain x_max", 4.0, 15.0, 8.0, 0.5)
        x, V   = make_morse(x_max, grid_points, D_e, alpha, x_eq)
        n_vals  = np.arange(num_states)
        E_analytical = morse_approx(n_vals, D_e, alpha)

    else:  # Kronig-Penney
        V0          = st.slider("Barrier height V₀", 1.0, 20.0, 5.0, 0.5)
        period      = st.slider("Lattice period", 0.5, 4.0, 1.5, 0.1)
        barrier_frac = st.slider("Barrier width fraction", 0.1, 0.9, 0.3, 0.05)
        x_max       = st.slider("Domain half-width", 3.0, 12.0, 6.0, 0.5)
        x, V        = make_kronig_penney(x_max, grid_points, V0, period, barrier_frac)
        n_vals = np.arange(num_states)
        E_analytical = None

# ── Solve ──────────────────────────────────────────────────────────────────────
energies, states = solve_schrodinger(x, V, num_states)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_eigen, tab_packet, tab_super, tab_analysis, tab_info = st.tabs(
    ["  Eigenstates  ", "  Wave Packet  ", "  Superposition  ", "  Analysis  ", "  About  "]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Eigenstates
# ════════════════════════════════════════════════════════════════════════════════
with tab_eigen:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.plotly_chart(
            fig_eigenstates(x, V, energies, states, system),
            use_container_width=True, config={"displayModeBar": False},
        )

    with col_right:
        selected = st.select_slider(
            "Inspect state", options=list(range(num_states)),
            format_func=lambda n: f"n = {n}  (E = {energies[n]:.4f})",
        )
        psi_sel = states[:, selected]
        color   = STATE_COLORS[selected % len(STATE_COLORS)]

        st.plotly_chart(
            fig_probability(x, psi_sel, f"n={selected}", color),
            use_container_width=True, config={"displayModeBar": False},
        )

        # Quick metrics
        dx_ = delta_x(x, psi_sel)
        dp_ = delta_p(x, psi_sel)
        c1, c2, c3 = st.columns(3)
        c1.metric("⟨x⟩", f"{expectation_x(x, psi_sel):.4f}")
        c2.metric("Δx", f"{dx_:.4f}")
        c3.metric("ΔxΔp / ℏ", f"{dx_*dp_/HBAR:.4f}")

    # Energy table + comparison
    st.markdown("---")
    col_tbl, col_chart = st.columns([1, 2])

    with col_tbl:
        st.markdown("#### Energy Levels")
        labels = [f"n={n}" for n in n_vals]
        table  = {"State": labels, "E (num)": [f"{e:.6f}" for e in energies]}
        if E_analytical is not None:
            err = np.abs(energies - E_analytical)
            table["E (analytic)"] = [f"{e:.6f}" for e in E_analytical]
            table["|ΔE|"]         = [f"{e:.2e}" for e in err]
        st.dataframe(table, use_container_width=True)

    with col_chart:
        if E_analytical is not None:
            st.plotly_chart(
                fig_energy_comparison(n_vals, energies, E_analytical, system),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            st.info("Closed-form analytical energies are not available for this potential. Numerical results shown above.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Time-Dependent Wave Packet
# ════════════════════════════════════════════════════════════════════════════════
with tab_packet:
    st.markdown("#### Gaussian Wave-Packet Dynamics")
    st.markdown(
        "Construct a Gaussian packet, expand it in energy eigenstates, and animate its time evolution: "
        "**ψ(x,t) = Σ cₙ φₙ(x) e^{−iEₙt/ℏ}**"
    )

    wp_col1, wp_col2 = st.columns(2)
    with wp_col1:
        x0    = st.slider("Centre x₀", float(x.min()*0.7), float(x.max()*0.7), float(np.mean(x)), 0.05)
        sigma = st.slider("Width σ", 0.1, 3.0, 0.5, 0.05)
    with wp_col2:
        k0      = st.slider("Momentum k₀", -10.0, 10.0, 2.0, 0.1)
        t_max   = st.slider("Max time", 0.5, 20.0, 5.0, 0.5)
        n_frames = st.slider("Animation frames", 20, 80, 40, 5)

    # Build packet as eigenstate superposition
    dx    = x[1] - x[0]
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * k0 * x)
    norm  = np.sqrt(np.sum(np.abs(gauss) ** 2) * dx)
    gauss /= norm

    # Project onto eigenstates
    c_n = np.array([np.sum(np.conj(states[:, i]) * gauss) * dx for i in range(num_states)])

    t_list    = np.linspace(0, t_max, n_frames)
    psi_t_list = [time_evolve(states, energies, c_n, t) for t in t_list]

    # Occupation probabilities
    occ = np.abs(c_n) ** 2
    st.plotly_chart(
        fig_time_evolution(x, psi_t_list, t_list, energies, states, c_n),
        use_container_width=True,
    )

    st.markdown("**Eigenstate occupation probabilities  |cₙ|²**")
    occ_fig = go.Figure(go.Bar(
        x=[f"n={i}" for i in range(num_states)], y=occ,
        marker_color=STATE_COLORS[:num_states], opacity=0.85,
        text=[f"{v:.3f}" for v in occ], textposition="outside",
        textfont=dict(color="#8ba8c4", size=10),
    ))
    occ_fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Eigenstate Decomposition  |cₙ|²", font=dict(family="Space Mono", size=12, color="#00d4ff")),
        height=260, yaxis_range=[0, max(occ) * 1.25],
        xaxis_title="State", yaxis_title="|cₙ|²",
    )
    st.plotly_chart(occ_fig, use_container_width=True, config={"displayModeBar": False})

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Superposition Builder
# ════════════════════════════════════════════════════════════════════════════════
with tab_super:
    st.markdown("#### Superposition State Builder")
    st.markdown("Set real coefficients cₙ for each eigenstate. The app normalises automatically.")

    coeff_cols = st.columns(min(num_states, 5))
    raw_coeffs = []
    for i, col in enumerate(coeff_cols[:num_states]):
        default = 1.0 if i == 0 else 0.0
        raw_coeffs.append(col.number_input(f"c_{i}", value=default, step=0.1, key=f"coeff_{i}"))

    # Pad if fewer columns shown than states
    while len(raw_coeffs) < num_states:
        raw_coeffs.append(0.0)

    c_arr = np.array(raw_coeffs, dtype=complex)
    norm_c = np.sqrt(np.sum(np.abs(c_arr) ** 2))
    if norm_c < 1e-10:
        st.warning("At least one coefficient must be non-zero.")
    else:
        c_arr /= norm_c
        psi_super = states @ c_arr

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.plotly_chart(
                fig_probability(x, psi_super, "Superposition State", "#00ffb3"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with s_col2:
            p, phi = momentum_space(x, psi_super)
            mask   = (p > -20) & (p < 20)
            st.plotly_chart(
                fig_momentum(p[mask], phi[mask], "Superposition State"),
                use_container_width=True, config={"displayModeBar": False},
            )

        # Time evolution of superposition
        st.markdown("**Animate this superposition state**")
        t_end = st.slider("Evolution time", 0.5, 15.0, 4.0, key="super_t")
        n_fr  = st.slider("Frames", 20, 60, 30, key="super_fr")
        t_lst = np.linspace(0, t_end, n_fr)
        psi_t_lst = [time_evolve(states, energies, c_arr, t) for t in t_lst]
        st.plotly_chart(
            fig_time_evolution(x, psi_t_lst, t_lst, energies, states, c_arr),
            use_container_width=True,
        )

        # Uncertainty metrics
        st.markdown("---")
        st.markdown("**Uncertainty Budget**")
        dx_s = delta_x(x, psi_super)
        dp_s = delta_p(x, psi_super)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⟨x⟩", f"{expectation_x(x, psi_super):.4f}")
        m2.metric("Δx", f"{dx_s:.4f}")
        m3.metric("Δp", f"{dp_s:.4f}")
        m4.metric("ΔxΔp / ℏ", f"{dx_s*dp_s/HBAR:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown("#### Per-Eigenstate Analysis")

    # Uncertainty table for all states
    rows = []
    for i in range(num_states):
        psi = states[:, i]
        dx_ = delta_x(x, psi)
        dp_ = delta_p(x, psi)
        rows.append({
            "State":  f"n={i}",
            "Energy": f"{energies[i]:.6f}",
            "⟨x⟩":    f"{expectation_x(x, psi):.4f}",
            "⟨p⟩":    f"{expectation_p(x, psi):.4f}",
            "Δx":     f"{dx_:.4f}",
            "Δp":     f"{dp_:.4f}",
            "ΔxΔp/ℏ": f"{dx_*dp_/HBAR:.5f}",
        })

    st.dataframe(rows, use_container_width=True)

    # Uncertainty scatter
    unc_vals = []
    for i in range(num_states):
        psi  = states[:, i]
        dxi_ = delta_x(x, psi)
        dpi_ = delta_p(x, psi)
        unc_vals.append(dxi_ * dpi_ / HBAR)

    unc_fig = go.Figure()
    unc_fig.add_hline(y=0.5, line=dict(color="#ff6b6b", dash="dash", width=1.5),
                      annotation_text="Heisenberg limit  ΔxΔp = ℏ/2",
                      annotation_font=dict(color="#ff6b6b", size=10))
    unc_fig.add_trace(go.Scatter(
        x=[f"n={i}" for i in range(num_states)], y=unc_vals,
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=10, color=STATE_COLORS[:num_states], line=dict(width=1.5, color="#050a14")),
        text=[f"{v:.4f}" for v in unc_vals],
        textposition="top center",
        textfont=dict(color="#8ba8c4", size=10),
        name="ΔxΔp/ℏ",
    ))
    unc_fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Heisenberg Uncertainty Product  ΔxΔp/ℏ  (min = 0.5)", font=dict(family="Space Mono", size=13, color="#00d4ff")),
        height=320, xaxis_title="State", yaxis_title="ΔxΔp / ℏ",
    )
    st.plotly_chart(unc_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # Momentum-space column per state
    st.markdown("#### Momentum-Space Probability Densities")
    p_cols = st.columns(min(num_states, 4))
    for i, col in enumerate(p_cols[:num_states]):
        psi = states[:, i]
        p, phi = momentum_space(x, psi)
        mask    = (p > -15) & (p < 15)
        mini_fig = go.Figure(go.Scatter(
            x=p[mask], y=np.abs(phi[mask])**2, mode="lines",
            line=dict(color=STATE_COLORS[i % len(STATE_COLORS)], width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(STATE_COLORS[i%len(STATE_COLORS)][1:3],16)},"
                      f"{int(STATE_COLORS[i%len(STATE_COLORS)][3:5],16)},"
                      f"{int(STATE_COLORS[i%len(STATE_COLORS)][5:7],16)},0.10)",
        ))
        mini_fig.update_layout(
            **PLOT_LAYOUT,
            title=dict(text=f"n={i}", font=dict(size=11, color=STATE_COLORS[i % len(STATE_COLORS)])),
            height=200, margin=dict(l=30, r=10, t=30, b=30),
            showlegend=False,
        )
        mini_fig.update_xaxes(**PLOT_LAYOUT["xaxis"], title="p")
        mini_fig.update_yaxes(**PLOT_LAYOUT["yaxis"], title="|φ|²")
        col.plotly_chart(mini_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### Transition Dipole Matrix  |⟨m|x̂|n⟩|")
    M      = transition_dipole_matrix(x, states)
    labels = [f"n={i}" for i in range(num_states)]
    st.plotly_chart(fig_dipole_matrix(M, labels), use_container_width=True, config={"displayModeBar": False})
    st.caption("Non-zero off-diagonal elements indicate optically allowed transitions (electric dipole selection rule).")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ════════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("#### What this application does")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Numerical Method**
- Finite-difference discretisation of the kinetic energy operator on a uniform grid
- Hamiltonian assembled as a symmetric tridiagonal matrix
- Sparse-compatible `scipy.linalg.eigh` eigendecomposition (O(N³) dense, upgradeable)
- Wavefunctions normalised via numerical quadrature

**Potentials**
- Particle in a Box (infinite walls, Dirichlet BCs)
- Harmonic Oscillator — analytic ℏω(n+½) benchmark
- Double Well — polynomial quartic ax⁴−bx² (tunnelling dynamics)
- Finite Square Well — piecewise constant
- Morse Potential — D(1−e^{−α(x−x_e)})² with anharmonic correction
- Kronig-Penney — periodic lattice (band structure precursor)
        """)

    with col_b:
        st.markdown("""
**Physics Computed**
- Energy eigenvalues & eigenfunctions {Eₙ, φₙ(x)}
- Time-dependent evolution: ψ(x,t) = Σ cₙ φₙ e^{−iEₙt/ℏ}
- Momentum-space wavefunctions φ(p) via FFT
- Expectation values ⟨x⟩, ⟨p⟩, ⟨x²⟩, ⟨p²⟩
- Heisenberg uncertainty product ΔxΔp/ℏ ≥ ½
- Electric-dipole transition matrix ⟨m|x̂|n⟩
- Gaussian wave-packet decomposition |cₙ|²

**Benchmarks**
- Numerical vs analytic energies tabulated for Particle-in-Box,
  Harmonic Oscillator, and Morse (perturbation expansion)
        """)

    st.markdown("---")
    st.markdown("""
**Stack:** Python · NumPy · SciPy · Plotly · Streamlit

*All quantities in dimensionless atomic units (ℏ = m = 1).
Grid convergence can be improved by increasing the grid-point count in the sidebar.*
    """)
