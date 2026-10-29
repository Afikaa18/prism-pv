import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# 1. KONFIGURASI HALAMAN & RESPONSIVE DESIGN (DESKTOP & MOBILE)
# ==============================================================================
st.set_page_config(
    page_title="PRISM-PV Paiton Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS agar metrik terlihat rapi di HP (Mobile) maupun Laptop
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
    .loss-metric {
        border-left: 5px solid #DC3545; /* Merah untuk Loss */
    }
    .success-metric {
        border-left: 5px solid #198754; /* Hijau untuk Aktual */
    }
    h1, h2, h3 { color: #0A2540; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    /* Penyesuaian padding untuk layar HP */
    @media (max-width: 768px) {
        .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
""", unsafe_allow_html=True)

# Auto-Refresh setiap 60 detik (60000 ms) agar Dashboard bersifat Real-Time Live
st_autorefresh(interval=60000, key="data_refresh_paiton")

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
# 3. FUNGSI PENARIKAN DATA CUACA (KOORDINAT KEC. BHINOR, PAITON)
# ==============================================================================
@st.cache_data(ttl=300) # Cache 5 menit agar tidak di-banned API
def get_paiton_realtime_weather():
    # Koordinat Presisi: PLTS Workshop PLN NP UP Paiton, Kec. Bhinor (-7.712, 113.582)
    url = "https://api.open-meteo.com/v1/forecast?latitude=-7.712&longitude=113.582&hourly=temperature_2m,shortwave_radiation&timezone=Asia%2FJakarta"
    res = requests.get(url).json()
    df_rt = pd.DataFrame({
        'Statistical period': pd.to_datetime(res['hourly']['time']),
        'Average Temperature (℃)': res['hourly']['temperature_2m'],
        'Global Irradiation (kWh/m²)': np.maximum(0.0, np.array(res['hourly']['shortwave_radiation']) / 1000.0)
    })
    return df_rt

# ==============================================================================
# 4. WAKTU PRESISI REAL-TIME (WIB - PAITON TIMING)
# ==============================================================================
wib_tz = pytz.timezone('Asia/Jakarta')
current_time = datetime.datetime.now(wib_tz)
formatted_now = current_time.strftime("%d %B %Y | %H:%M:%S WIB")

# HEADER DASHBOARD
st.title("⚡ PRISM-PV: Digital Twin & AI Control Room")
st.markdown("**PLTS Workshop PT PLN Nusantara Power UP Paiton** | Huawei SUN2000 (COM1-1 & COM1-2)")
st.caption(f"📍 Location: Kec. Bhinor, Probolinggo | 🕒 Live System Time: **{formatted_now}**")
st.markdown("---")

# ==============================================================================
# 5. DATA PIPELINE & AI PREDICTION (NOWCASTING)
# ==============================================================================
with st.spinner("Mengkalibrasi telemetri cuaca satelit..."):
    df_rt_full = get_paiton_realtime_weather()
    
    # Menarik data cuaca 24 jam terakhir hingga jam saat ini
    current_hour_trunc = current_time.replace(minute=0, second=0, microsecond=0).tz_localize(None)
    df_recent = df_rt_full[df_rt_full['Statistical period'] <= current_hour_trunc].tail(24).copy()
    
    if len(df_recent) < 24:
        df_recent = df_rt_full.tail(24).copy()

    # Ekstraksi Cuaca Detik Ini (Baris Terakhir)
    curr_ghi = df_recent.iloc[-1]['Global Irradiation (kWh/m²)']
    curr_temp = df_recent.iloc[-1]['Average Temperature (℃)']

    # Siklus Temporal (Trigonometri)
    df_recent['hour'] = df_recent['Statistical period'].dt.hour
    df_recent['month'] = df_recent['Statistical period'].dt.month
    df_recent['hour_sin'] = np.sin(2 * np.pi * df_recent['hour'] / 24.0)
    df_recent['hour_cos'] = np.cos(2 * np.pi * df_recent['hour'] / 24.0)
    df_recent['month_sin'] = np.sin(2 * np.pi * df_recent['month'] / 12.0)
    df_recent['month_cos'] = np.cos(2 * np.pi * df_recent['month'] / 12.0)

    features = ['Global Irradiation (kWh/m²)', 'Average Temperature (℃)', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

    # Prediksi AI (Seharusnya Produksi Berapa?)
    X_input_scaled = scaler_X.transform(df_recent[features])
    X_input_3d = X_input_scaled.reshape(1, 24, 6)
    pred_scaled = model.predict(X_input_3d, verbose=0)
    
    # Total Prediksi Output Workshop (Target AI)
    target_ai_total = max(0.0, float(scaler_y.inverse_transform(pred_scaled)[0][0]))
    
    # Jika malam hari (Radiasi 0), paksa AI target menjadi 0
    if curr_ghi <= 0.01:
        target_ai_total = 0.0

# ==============================================================================
# 6. PENDEKATAN AKTUAL & PERHITUNGAN KERUGIAN (LOSSES PHYSICS)
# ==============================================================================
# Di lapangan asli, ini ditarik dari API FusionSolar. 
# Sebagai simulasi cerdas, kita asumsikan efisiensi aktual = 93% dari Target AI (karena rugi kabel/debu)
aktual_total = target_ai_total * 0.93 if target_ai_total > 0 else 0.0

# Pemecahan 2 Inverter
pred_inv1, pred_inv2 = target_ai_total * 0.52, target_ai_total * 0.48
akt_inv1, akt_inv2 = aktual_total * 0.52, aktual_total * 0.48

# Kalkulasi Celah Kerugian (Energy & Financial Loss)
energy_loss = max(0.0, target_ai_total - aktual_total)
TARIF_RP = 1808
kerugian_rp = energy_loss * TARIF_RP
pendapatan_rp = aktual_total * TARIF_RP

# ==============================================================================
# 7. TAMPILAN RESPONSIVE (DASHBOARD UI)
# ==============================================================================

# SECTION A: Cuaca & Lingkungan
st.subheader("📡 Variabel Cuaca Aktual (Real-Time)")
# Gunakan st.columns yang otomatis membungkus (wrap) di layar HP
c1, c2, c3, c4 = st.columns(4)
c1.metric("Radiasi Global (GHI)", f"{curr_ghi:.3f} kWh/m²")
c2.metric("Suhu Ambien", f"{curr_temp:.1f} °C")
c3.metric("Potensi Daya Maksimal", "230 kW")
c4.metric("Status Operasi", "AKTIF" if curr_ghi > 0 else "STANDBY MALAM")

st.markdown("---")

# SECTION B: Inverter & Produksi
st.subheader("⚡ Monitoring Dual Inverter Huawei (AI vs FusionSolar)")
col_inv1, col_inv2 = st.columns(2)

with col_inv1:
    st.markdown("##### 🔌 Inverter 1 (COM1-1)")
    st.info(f"**Target AI:** {pred_inv1:.2f} kWh\n\n**Aktual:** {akt_inv1:.2f} kWh")
    ef_1 = (akt_inv1/pred_inv1*100) if pred_inv1 > 0 else 0.0
    st.caption(f"Efisiensi Transmisi: {ef_1:.1f}%")

with col_inv2:
    st.markdown("##### 🔌 Inverter 2 (COM1-2)")
    st.info(f"**Target AI:** {pred_inv2:.2f} kWh\n\n**Aktual:** {akt_inv2:.2f} kWh")
    ef_2 = (akt_inv2/pred_inv2*100) if pred_inv2 > 0 else 0.0
    st.caption(f"Efisiensi Transmisi: {ef_2:.1f}%")

st.markdown("---")

# SECTION C: Financial & Loss Diagnostics (The Core Value)
st.subheader("📊 Diagnostik Kerugian Daya & Pendapatan Finansial")
st.write("Menganalisis selisih antara Model Kembaran Digital (Target AI) dengan Daya Aktual Lapangan.")

# Desain kolom diagnostik
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("<div class='stMetric success-metric'>", unsafe_allow_html=True)
    st.metric("Total Produksi (Aktual)", f"{aktual_total:.2f} kWh")
    st.markdown("</div>", unsafe_allow_html=True)

with f2:
    st.markdown("<div class='stMetric loss-metric'>", unsafe_allow_html=True)
    st.metric("Terbuang/Energy Loss", f"{energy_loss:.2f} kWh")
    st.markdown("</div>", unsafe_allow_html=True)

with f3:
    st.markdown("<div class='stMetric success-metric'>", unsafe_allow_html=True)
    st.metric("Pendapatan Bersih", f"Rp {pendapatan_rp:,.0f}".replace(",", "."))
    st.markdown("</div>", unsafe_allow_html=True)

if energy_loss > 5.0:
    st.error(f"⚠️ Peringatan: Terdapat kehilangan daya signifikan sebesar **{energy_loss:.2f} kWh** (Potensi Rugi: Rp {kerugian_rp:,.0f}). Periksa kebersihan panel dari debu PLTU atau indikasi shading awan!")
elif curr_ghi > 0:
    st.success("✅ Performa Pembangkit Optimal. Tidak ada indikasi kerugian daya ekstrem.")
