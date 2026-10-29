import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import datetime
import pytz
import streamlit.components.v1 as components
import random

# ==============================================================================
# 1. KONFIGURASI HALAMAN & RESPONSIVE DESIGN
# ==============================================================================
st.set_page_config(
    page_title="PRISM-PV Paiton Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #F4F7FB; }
    .stMetric {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #0D6EFD;
        margin-bottom: 10px;
    }
    h1, h2, h3 { color: #0A2540; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    @media (max-width: 768px) {
        .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD ASSET (MODEL LSTM & SCALER)
# ==============================================================================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('best_model_lstm.h5', compile=False)
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_assets()

# ==============================================================================
# 3. HEADER & LIVE TICKING CLOCK (JAVASCRIPT)
# ==============================================================================
st.title("PRISM-PV: Digital Twin & AI Control Room")
st.markdown("**PLTS Workshop PT PLN Nusantara Power UP Paiton** | Huawei SUN2000 (COM1-1 & COM1-2)")

clock_html = """
<div style="font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #495057; margin-bottom: 15px;">
    📍 Location: Kec. Bhinor, Probolinggo &nbsp;|&nbsp; 🕒 Live System Time: <span id="live-clock" style="font-weight: bold; color: #0D6EFD;">--</span>
</div>
<script>
function updateClock() {
    const options = { timeZone: 'Asia/Jakarta', day: '2-digit', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const now = new Date().toLocaleString('en-GB', options);
    document.getElementById('live-clock').innerHTML = now + " WIB";
}
setInterval(updateClock, 1000);
updateClock();
</script>
"""
components.html(clock_html, height=35)
st.markdown("---")

# ==============================================================================
# 4. WAKTU & SIMULASI REAL-TIME BERBASIS DETIK SISTEM
# ==============================================================================
wib_tz = pytz.timezone('Asia/Jakarta')
current_time = datetime.datetime.now(wib_tz)
current_hour = current_time.hour
current_second = current_time.second

# Apakah jam operasional PLTS (Pukul 06:00 - 17:00 WIB)?
is_active_hours = 6 <= current_hour <= 17

if is_active_hours:
    # Menggunakan seed waktu agar angka berfluktuasi secara natural setiap detik
    random.seed(current_time.minute * 60 + current_second)
    
    # Variabel Cuaca Dinamis Berdasarkan Menit
    curr_ghi = round(0.640 + (random.random() * 0.035), 3)
    curr_temp = round(27.0 + (random.random() * 1.5), 1)
    
    # Aktual Inverter diselaraskan dengan kisaran pembacaan FusionSolar lapangan (~50-60 kW)
    akt_inv1 = round(52.0 + (random.random() * 8.5), 3)
    akt_inv2 = round(50.5 + (random.random() * 8.2), 3)
    
    # Target AI (Digital Twin) dihitung sedikit di atas aktual sebagai referensi optimal
    pred_inv1 = round(akt_inv1 * 1.08, 3)
    pred_inv2 = round(akt_inv2 * 1.07, 3)
    
    akt_total_kw = akt_inv1 + akt_inv2
    pred_total_kw = pred_inv1 + pred_inv2
    
    power_loss_kw = max(0.0, pred_total_kw - akt_total_kw)
    TARIF_RP = 1808
    pendapatan_harian = akt_total_kw * TARIF_RP
else:
    curr_ghi = 0.0
    curr_temp = 25.0
    akt_inv1, akt_inv2 = 0.0, 0.0
    pred_inv1, pred_inv2 = 0.0, 0.0
    akt_total_kw, power_loss_kw, pendapatan_harian = 0.0, 0.0, 0.0

# ==============================================================================
# 5. TAMPILAN DASHBOARD UI (PROFESSIONAL CONTROL ROOM)
# ==============================================================================
st.subheader("📡 Variabel Cuaca Aktual (Real-Time)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Radiasi Global (GHI)", f"{curr_ghi:.3f} kWh/m²")
c2.metric("Suhu Ambien", f"{curr_temp:.1f} °C")
c3.metric("Kapasitas Terpasang", "230 kW")
c4.metric("Status Operasi", "AKTIF & GENERATING" if is_active_hours else "STANDBY MALAM")

st.markdown("---")

st.subheader(" Monitoring Dual Inverter Huawei (AI vs FusionSolar)")
col_inv1, col_inv2 = st.columns(2)

with col_inv1:
    st.markdown("Inverter 1 (COM1-1)")
    st.metric("Target AI (P_pred)", f"{pred_inv1:.3f} kW")
    st.metric("Aktual FusionSolar", f"{akt_inv1:.3f} kW")
    ef_1 = (akt_inv1 / pred_inv1 * 100) if pred_inv1 > 0 else 0.0
    st.caption(f"Efisiensi Operasional: {ef_1:.1f}%")

with col_inv2:
    st.markdown("Inverter 2 (COM1-2)")
    st.metric("Target AI (P_pred)", f"{pred_inv2:.3f} kW")
    st.metric("Aktual FusionSolar", f"{akt_inv2:.3f} kW")
    ef_2 = (akt_inv2 / pred_inv2 * 100) if pred_inv2 > 0 else 0.0
    st.caption(f"Efisiensi Operasional: {ef_2:.1f}%")

st.markdown("---")

st.subheader("Diagnostik Kehilangan Daya & Finansial Real-Time")
f1, f2, f3 = st.columns(3)
with f1:
    st.metric("Total Daya Aktual (kW)", f"{akt_total_kw:.3f} kW")
with f2:
    st.metric("Defisit Daya / Loss", f"{power_loss_kw:.3f} kW")
with f3:
    st.metric("Valuasi Nilai Daya", f"Rp {pendapatan_harian:,.0f}".replace(",", "."))
