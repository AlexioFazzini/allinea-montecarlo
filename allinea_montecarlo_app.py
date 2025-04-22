"""ALLINEA Monte Carlo Stress‑Test – Streamlit app
Author: ChatGPT – April 2025
Description: interactive tool to estimate the probability of hitting a savings/wealth
            goal within a chosen horizon using Monte Carlo simulation, with optional
            withdrawal shock and downloadable CSV / PDF report.
"""

import io
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Try import FPDF; if unavailable user must add to requirements.txt
try:
    from fpdf import FPDF
except ImportError:  # graceful fallback
    FPDF = None

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit page config & branding
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ALLINEA Monte Carlo Stress‑Test", layout="wide")

st.title("ALLINEA – Monte Carlo Stress‑Test")
st.write(
    "Valuta la probabilità di raggiungere un obiettivo di patrimonio entro un dato orizzonte, "
    "simulando migliaia di scenari di mercato e opzionalmente un prelievo straordinario."
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
    st.header("Ipotesi di mercato (in % annuo)")
    exp_return_pct = st.slider("Rendimento atteso (%)", 0.0, 15.0, 5.0, 0.1, format="%.1f")
    volatility_pct = st.slider("Volatilità (%)", 0.0, 30.0, 10.0, 0.1, format="%.1f")
    inflation_pct = st.slider("Inflazione media (%)", 0.0, 10.0, 2.0, 0.1, format="%.1f")

    # Convert to decimals
    exp_return = exp_return_pct / 100
    volatility = volatility_pct / 100
    inflation = inflation_pct / 100

    st.markdown("---")
    st.header("Shock di prelievo opzionale")
    use_withdrawal = st.checkbox("Simula un prelievo futuro")
    if use_withdrawal:
        withdrawal_year = st.slider("Anno del prelievo", min_value=1, max_value=years - 1, value=int(years / 2))
        withdrawal_amount = st.number_input(
            "Importo prelievo (€)", min_value=1_000, max_value=10_000_000, value=50_000, step=1_000
        )
    else:
        withdrawal_year, withdrawal_amount = None, 0.0

    st.markdown("---")
    st.header("Impostazioni avanzate")
    n_sims = st.number_input("Numero simulazioni", 1000, 200_000, 10_000, 1000)
    fat_tail = st.checkbox("Attiva code grasse (Student‑t df=5)")

    run_button = st.button("Esegui Stress‑Test", type="primary")

# ────────────────────────────────────────────────────────────────────────────────
# Monte Carlo engine
# ────────────────────────────────────────────────────────────────────────────────

def simulate_paths(
    mu: float,
    sigma: float,
    years: int,
    init_cap: float,
    contrib: float,
    inflation: float = 0.02,
    n_sims: int = 10_000,
    fat_tail: bool = False,
    withdrawal_year: int | None = None,
    withdrawal_amount: float = 0.0,
    df: int = 5,
) -> pd.DataFrame:
    """Return DataFrame (years+1 × n_sims) with wealth evolution.

    Withdrawal is modelled at *start of the specified year* (after contribution, before growth).
    """
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
        balance = paths[t - 1] + contrib
        if withdrawal_year is not None and t == withdrawal_year:
            balance = np.maximum(balance - withdrawal_amount, 0)  # no shortfall allowed
        paths[t] = balance * (1 + real_shocks[t - 1])
    return pd.DataFrame(paths)


# ────────────────────────────────────────────────────────────────────────────────
# Helper – build PDF report
# ────────────────────────────────────────────────────────────────────────────────

def build_pdf(prob_success: float, median_wealth: float, target: float, fig) -> bytes:
    if FPDF is None:
        st.error("Modulo FPDF non installato. Aggiungi 'fpdf' a requirements.txt e ripeti.")
        return b""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "ALLINEA – Monte Carlo Stress‑Test", ln=True, align="C")

    pdf.set_font("Helvetica", size=12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Data: {date.today().isoformat()}")

    pdf.ln(4)
    pdf.multi_cell(0, 8, f"Probabilità di successo: {prob_success:.1%}")
    pdf.multi_cell(0, 8, f"Patrimonio finale – Mediana: € {median_wealth:,.0f}")
    pdf.multi_cell(0, 8, f"Obiettivo: € {target:,.0f}")

    # Save matplotlib fig to bytes and embed
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
    img_buf.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(img_buf.read())
        tmp.flush()
        pdf.image(tmp.name, x=10, w=190)

    pdf.ln(2)
    pdf.set_font("Helvetica", size=8)
    pdf.multi_cell(0, 6, "Simulazione illustrativa basata su ipotesi di mercato e volatilità ipotetiche. Non costituisce garanzia di risultato.")

    return pdf.output(dest="S").encode("latin-1")


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
        withdrawal_year=withdrawal_year,
        withdrawal_amount=withdrawal_amount,
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
    if use_withdrawal:
        ax.axvline(withdrawal_year, color="black", linestyle=":", linewidth=1, label="Prelievo")
        ax.text(withdrawal_year + 0.05, target * 0.95, f"-€ {withdrawal_amount:,.0f}", rotation=90, va="top", fontsize=8)
    ax.set_xlabel("Anno")
    ax.set_ylabel("Patrimonio (€)")
    ax.set_title("Evoluzione del patrimonio – fan‑chart Monte Carlo")
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

    # Download PDF report
    if FPDF is not None:
        pdf_bytes = build_pdf(prob_success, final_vals.median(), target, fig)
        st.download_button(
            label="Scarica report PDF",
            data=pdf_bytes,
            file_name="report_montecarlo.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Per la funzione PDF occorre aggiungere 'fpdf' a requirements.txt e rilanciare.")

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
# fpdf
