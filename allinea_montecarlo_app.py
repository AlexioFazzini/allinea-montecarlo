"""ALLINEA Monte Carlo Stress‑Test – Streamlit app
Author: ChatGPT – April 2025
Description: interactive tool to estimate the probability of hitting a savings/wealth
            goal within a chosen horizon using Monte Carlo simulation, with a basic
            stress‑layer and downloadable results.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit page config & branding
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ALLINEA Monte Carlo Stress‑Test", layout="wide")

st.title("ALLINEA – Monte Carlo Stress‑Test")
st.write(
    "Valuta la probabilità di raggiungere un obiettivo di patrimonio entro un dato orizzonte, "
    "simulando migliaia di scenari di mercato, con la possibilità di applicare stress test."
)

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar – Input parameters
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parametri di base")

    init_cap = st.number_input(
        "Capitale iniziale (€)", min_value=0, max_value=10_000_000, value=100_000, step=10_000
    )
    annual_contrib = st.number_input(
        "Contributo annuo (€)", min_value=0, max_value=500_000, value=10_000, step=1_000
    )
    years = st.slider("Orizzonte (anni)", min_value=1, max_value=40, value=20)
    target = st.number_input(
        "Obiettivo di patrimonio (€)", min_value=0, max_value=20_000_000, value=500_000, step=10_000
    )

    st.markdown("---")
    st.header("Ipotesi di mercato")
    exp_return = st.slider("Rendimento atteso (μ)", 0.00, 0.15, 0.05, 0.005, format="%.3f")
    volatility = st.slider("Volatilità (σ)", 0.00, 0.30, 0.10, 0.005, format="%.3f")
    inflation = st.slider("Inflazione media", 0.00, 0.10, 0.02, 0.005, format="%.3f")

    st.markdown("---")
    st.header("Impostazioni avanzate")
    n_sims = st.number_input("Numero simulazioni", 1000, 200_000, 10_000, 1000)
    fat_tail = st.checkbox("Attiva code grasse (Student‑t df=5)")

    run_button = st.button("Esegui Stress‑Test", type="primary")

# ────────────────────────────────────────────────────────────────────────────────
# Monte Carlo engine
# ────────────────────────────────────────────────────────────────────────────────

def simulate_paths(mu: float, sigma: float, years: int, init_cap: float, contrib: float,
                   inflation: float = 0.02, n_sims: int = 10_000, fat_tail: bool = False,
                   df: int = 5) -> pd.DataFrame:
    """Return DataFrame (years+1 × n_sims) with wealth evolution."""
    rng = np.random.default_rng()
    if fat_tail:
        scale = sigma / np.sqrt(df / (df - 2))
        shocks = rng.standard_t(df, size=(years, n_sims)) * scale + mu
    else:
        shocks = rng.normal(mu, sigma, size=(years, n_sims))

    real_shocks = (1 + shocks) / (1 + inflation) - 1  # adjust for inflation

    paths = np.empty((years + 1, n_sims))
    paths[0] = init_cap
    for t in range(1, years + 1):
        paths[t] = (paths[t - 1] + contrib) * (1 + real_shocks[t - 1])
    return pd.DataFrame(paths)


# ────────────────────────────────────────────────────────────────────────────────
# Main logic – run simulation and display results
# ────────────────────────────────────────────────────────────────────────────────

if run_button:
    paths = simulate_paths(
        mu=exp_return,
        sigma=volatility,
        years=years,
        init_cap=init_cap,
        contrib=annual_contrib,
        inflation=inflation,
        n_sims=n_sims,
        fat_tail=fat_tail,
    )

    final_vals = paths.iloc[-1]
    prob_success = (final_vals >= target).mean()

    st.subheader("Risultati principali")
    col1, col2 = st.columns(2)
    col1.metric("Probabilità di successo", f"{prob_success:.1%}")
    col2.metric("Wealth finale – mediana", f"€ {final_vals.median():,.0f}")

    # Fan‑chart
    percentiles = paths.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
    years_axis = np.arange(0, years + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(years_axis, percentiles[0.1], percentiles[0.9], alpha=0.2, label="10°‑90°")
    ax.fill_between(years_axis, percentiles[0.25], percentiles[0.75], alpha=0.4, label="25°‑75°")
    ax.plot(years_axis, percentiles[0.5], linewidth=2, label="Mediana")
    ax.axhline(target, linestyle="--", linewidth=1.2, label="Obiettivo")
    ax.set_xlabel("Anno")
    ax.set_ylabel("Patrimonio (€)")
    ax.set_title("Evoluzione del patrimonio – fan‑chart Monte Carlo")
    ax.legend()
    st.pyplot(fig)

    # Download CSV of percentiles
    csv_buffer = io.StringIO()
    percentiles.to_csv(csv_buffer, index_label="Anno")
    st.download_button(
        label="Scarica percentili (.csv)",
        data=csv_buffer.getvalue(),
        file_name="percentili_montecarlo.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        "⚠️ *Disclaimer*: simulazione a scopo puramente illustrativo. I risultati si basano su ipotesi di mercato ipotetiche e non garantiscono performance future."
    )

# ────────────────────────────────────────────────────────────────────────────────
# Requirements (per requirements.txt)
# ────────────────────────────────────────────────────────────────────────────────
# streamlit>=1.33
# numpy
# pandas
# matplotlib
