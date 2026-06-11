import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CPS Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #F0FBF7 0%, #E6F7F1 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D4F3C 0%, #1A6B52 100%) !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #E8F8F2 !important;
}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stCheckbox > label,
[data-testid="stSidebar"] .stNumberInput > label {
    color: #A8E6CE !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    padding: 0 4px;
}
/* Slider track active */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="progressbar"] {
    background: #4ECBA0 !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #4ECBA0 !important;
    border-color: #4ECBA0 !important;
}
/* Selectbox in sidebar */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(78,203,160,0.4) !important;
    color: white !important;
}
/* Checkbox */
[data-testid="stSidebar"] .stCheckbox [data-baseweb="checkbox"] div {
    border-color: #4ECBA0 !important;
}
[data-testid="stSidebar"] .stCheckbox [data-baseweb="checkbox"] div[data-checked="true"] {
    background: #4ECBA0 !important;
    border-color: #4ECBA0 !important;
}

/* Section headers in sidebar */
.sidebar-section {
    background: rgba(78,203,160,0.15);
    border-left: 3px solid #4ECBA0;
    padding: 6px 12px;
    margin: 16px 0 10px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4ECBA0 !important;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0D4F3C;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #4A7A68;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 400;
}

/* Predict button */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #4ECBA0, #1A8C6E) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    letter-spacing: 0.02em !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.88 !important;
}

/* Result card */
.result-card {
    background: white;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    box-shadow: 0 8px 32px rgba(13,79,60,0.10);
    text-align: center;
    border: 1px solid rgba(78,203,160,0.2);
}
.score-value {
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #0D4F3C, #4ECBA0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.score-label {
    font-size: 0.85rem;
    color: #7DA89A;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-top: 0.3rem;
}
.category-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 1rem;
}
.badge-sangat-baik { background:#D1FAE5; color:#065F46; }
.badge-baik        { background:#A7F3D0; color:#065F46; }
.badge-cukup       { background:#FEF3C7; color:#92400E; }
.badge-perlu       { background:#FEE2E2; color:#991B1B; }

/* Gauge container */
.gauge-wrap {
    display: flex;
    justify-content: center;
    margin: 1rem 0;
}

/* Info cards */
.info-card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(13,79,60,0.07);
    border-left: 4px solid #4ECBA0;
    margin-bottom: 1rem;
}
.info-card h4 {
    margin: 0 0 0.3rem;
    font-size: 0.75rem;
    color: #7DA89A;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.info-card p {
    margin: 0;
    font-size: 1rem;
    color: #1A3D31;
    font-weight: 500;
}

/* Analisis section title */
.analysis-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: #7DA89A;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.8rem 0 0.8rem;
}

/* Factor row card */
.factor-card {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(13,79,60,0.06);
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    border: 1px solid rgba(78,203,160,0.12);
}
.factor-icon {
    font-size: 1.4rem;
    line-height: 1;
    margin-top: 2px;
    flex-shrink: 0;
}
.factor-body { flex: 1; }
.factor-name {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7DA89A;
    margin-bottom: 2px;
}
.factor-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1A3D31;
    margin-bottom: 4px;
}
.factor-tip {
    font-size: 0.82rem;
    color: #4A7A68;
    line-height: 1.5;
}
.factor-status {
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
/* status colors */
.status-ok   { color: #059669; }
.status-warn { color: #D97706; }
.status-bad  { color: #DC2626; }

/* Overall tip banner */
.overall-banner {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.7;
    font-weight: 500;
}
.banner-great { background:#D1FAE5; color:#065F46; border-left: 4px solid #059669; }
.banner-good  { background:#A7F3D0; color:#065F46; border-left: 4px solid #10B981; }
.banner-ok    { background:#FEF3C7; color:#92400E; border-left: 4px solid #F59E0B; }
.banner-bad   { background:#FEE2E2; color:#991B1B; border-left: 4px solid #EF4444; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #7DA89A;
}
.empty-state .icon { font-size: 4rem; margin-bottom: 1rem; }
.empty-state h3 { font-size: 1.3rem; font-weight: 600; color: #4A7A68; margin: 0; }
.empty-state p { font-size: 0.9rem; margin-top: 0.5rem; }

/* Divider */
.green-divider {
    height: 3px;
    background: linear-gradient(90deg, #4ECBA0, transparent);
    border-radius: 2px;
    margin: 1.5rem 0;
}

/* Streamlit default element tweaks */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(78,203,160,0.4) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Assets ───────────────────────────────────────────────────────────────────
BOOSTER_PATH  = "booster.ubj"
SCALER_PATH   = "scaler.pkl"
METADATA_PATH = "metadata.pkl"

@st.cache_resource
def load_assets():
    for path in [BOOSTER_PATH, SCALER_PATH, METADATA_PATH]:
        if not os.path.exists(path):
            st.error(f"File tidak ditemukan: {path}")
            st.stop()
    model = XGBRegressor()
    model.load_model(BOOSTER_PATH)
    scaler   = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, scaler, metadata

model, scaler, metadata = load_assets()
MODEL_COLUMNS = metadata["columns"]

SLEEP_RISK_MAP = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

def build_model_input(raw):
    data = {col: 0 for col in MODEL_COLUMNS}
    data["sleep_duration_hrs"]          = raw["sleep_duration_hrs"]
    data["sleep_quality_score"]         = raw["sleep_quality_score"]
    data["rem_percentage"]              = raw["rem_percentage"]
    data["deep_sleep_percentage"]       = raw["deep_sleep_percentage"]
    data["sleep_latency_mins"]          = raw["sleep_latency_mins"]
    data["wake_episodes_per_night"]     = raw["wake_episodes_per_night"]
    data["felt_rested"]                 = int(raw["felt_rested"])
    data["sleep_disorder_risk_ord"]     = SLEEP_RISK_MAP[raw["sleep_disorder_risk"]]
    data["caffeine_mg_before_bed"]      = raw["caffeine_mg_before_bed"]
    data["screen_time_before_bed_mins"] = raw["screen_time_before_bed_mins"]
    data["exercise_day"]                = int(raw["exercise_day"])
    data["stress_score"]                = raw["stress_score"]
    # defaults for unused but required columns
    data["age"]                         = 28
    data["bmi"]                         = 22.0
    data["nap_duration_mins"]           = 0
    data["alcohol_units_before_bed"]    = 0
    data["steps_that_day"]              = 7000
    data["work_hours_that_day"]         = 8
    data["heart_rate_resting_bpm"]      = 70
    data["sleep_aid_used"]              = 0
    data["shift_work"]                  = 0
    data["room_temperature_celsius"]    = 24
    data["weekend_sleep_diff_hrs"]      = 0
    data["gender_Male"]                 = 1
    data["mental_health_condition_Healthy"] = 1
    data["chronotype_Neutral"]          = 1
    data["season_Summer"]               = 1
    return pd.DataFrame([data], columns=MODEL_COLUMNS)

def score_category(s):
    if s >= 85: return "Sangat Baik", "badge-sangat-baik", "#065F46"
    if s >= 70: return "Baik",        "badge-baik",        "#065F46"
    if s >= 55: return "Cukup",       "badge-cukup",       "#92400E"
    return              "Perlu Perhatian", "badge-perlu",  "#991B1B"

def gauge_svg(score):
    pct   = min(max(score / 100, 0), 1)
    angle = pct * 180
    r     = 90
    cx, cy = 110, 110
    import math
    rad   = math.radians(180 - angle)
    nx    = cx + r * math.cos(rad)
    ny    = cy - r * math.sin(rad)
    # arc color
    if score >= 85:   arc_color = "#4ECBA0"
    elif score >= 70: arc_color = "#1A8C6E"
    elif score >= 55: arc_color = "#F59E0B"
    else:             arc_color = "#EF4444"
    return f"""
    <svg width="220" height="130" viewBox="0 0 220 130" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#EF4444"/>
          <stop offset="50%" style="stop-color:#F59E0B"/>
          <stop offset="100%" style="stop-color:#4ECBA0"/>
        </linearGradient>
      </defs>
      <!-- bg arc -->
      <path d="M 20,110 A 90,90 0 0,1 200,110" fill="none" stroke="#E5E7EB" stroke-width="14" stroke-linecap="round"/>
      <!-- colored arc -->
      <path d="M 20,110 A 90,90 0 0,1 200,110" fill="none" stroke="url(#arcGrad)" stroke-width="14" stroke-linecap="round"
            stroke-dasharray="{pct * 283} 283"/>
      <!-- needle -->
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="#0D4F3C" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="6" fill="#0D4F3C"/>
      <!-- labels -->
      <text x="18"  y="128" font-size="10" fill="#9CA3AF" font-family="Inter,sans-serif">0</text>
      <text x="104" y="24"  font-size="10" fill="#9CA3AF" font-family="Inter,sans-serif" text-anchor="middle">50</text>
      <text x="198" y="128" font-size="10" fill="#9CA3AF" font-family="Inter,sans-serif" text-anchor="end">100</text>
    </svg>"""

def analyze_factors(raw):
    """
    Returns list of dicts:
      icon, name, value_str, status (ok/warn/bad), tip
    """
    factors = []

    # 1. Durasi Tidur
    dur = raw["sleep_duration_hrs"]
    if dur >= 7 and dur <= 9:
        factors.append(dict(icon="😴", name="Durasi Tidur", value=f"{dur} jam",
            status="ok",   tip="Durasi tidurmu ideal. Pertahankan konsistensi jam tidur setiap malam."))
    elif dur < 7:
        deficit = round(7 - dur, 1)
        factors.append(dict(icon="😴", name="Durasi Tidur", value=f"{dur} jam",
            status="bad",  tip=f"Kurang {deficit} jam dari ideal. Coba tidur lebih awal 30–60 menit secara bertahap hingga mencapai 7–8 jam."))
    else:
        factors.append(dict(icon="😴", name="Durasi Tidur", value=f"{dur} jam",
            status="warn", tip="Tidur >9 jam bisa menurunkan kualitas tidur malam berikutnya. Coba batasi hingga 8–9 jam."))

    # 2. Kualitas Tidur
    q = raw["sleep_quality_score"]
    if q >= 70:
        factors.append(dict(icon="⭐", name="Kualitas Tidur", value=f"{q:.0f} / 100",
            status="ok",   tip="Kualitas tidurmu baik. Jaga rutinitas tidur yang konsisten."))
    elif q >= 50:
        factors.append(dict(icon="⭐", name="Kualitas Tidur", value=f"{q:.0f} / 100",
            status="warn", tip="Kualitas tidur cukup, bisa ditingkatkan. Hindari makan berat 2–3 jam sebelum tidur dan pastikan kamar gelap & sejuk."))
    else:
        factors.append(dict(icon="⭐", name="Kualitas Tidur", value=f"{q:.0f} / 100",
            status="bad",  tip="Kualitas tidur rendah. Coba terapkan sleep hygiene: matikan lampu, hindari gadget, dan tidur di jam yang sama setiap hari."))

    # 3. REM Sleep
    rem = raw["rem_percentage"]
    if rem >= 20:
        factors.append(dict(icon="🌙", name="REM Sleep", value=f"{rem:.0f}%",
            status="ok",   tip="Porsi REM sleep cukup. REM yang baik mendukung memori dan kreativitas."))
    elif rem >= 15:
        factors.append(dict(icon="🌙", name="REM Sleep", value=f"{rem:.0f}%",
            status="warn", tip="REM sleep sedikit di bawah ideal (20–25%). Kurangi alkohol dan kafein karena keduanya menekan fase REM."))
    else:
        factors.append(dict(icon="🌙", name="REM Sleep", value=f"{rem:.0f}%",
            status="bad",  tip="REM sleep rendah. Ini bisa disebabkan stres, kafein, atau kurang tidur. Prioritaskan durasi tidur yang cukup."))

    # 4. Deep Sleep
    deep = raw["deep_sleep_percentage"]
    if deep >= 15:
        factors.append(dict(icon="💤", name="Deep Sleep", value=f"{deep:.0f}%",
            status="ok",   tip="Deep sleep cukup. Fase ini penting untuk pemulihan fisik dan konsolidasi memori."))
    elif deep >= 10:
        factors.append(dict(icon="💤", name="Deep Sleep", value=f"{deep:.0f}%",
            status="warn", tip="Deep sleep agak rendah. Olahraga teratur dan menghindari kafein sore hari dapat membantu meningkatkannya."))
    else:
        factors.append(dict(icon="💤", name="Deep Sleep", value=f"{deep:.0f}%",
            status="bad",  tip="Deep sleep sangat rendah. Hindari alkohol, tidur di suhu ruangan 18–22°C, dan coba tidak tidur siang terlalu lama."))

    # 5. Waktu Tertidur
    lat = raw["sleep_latency_mins"]
    if lat <= 20:
        factors.append(dict(icon="⏱️", name="Waktu Tertidur", value=f"{lat} menit",
            status="ok",   tip="Waktu tertidur normal (di bawah 20 menit). Tanda tubuhmu siap beristirahat."))
    elif lat <= 40:
        factors.append(dict(icon="⏱️", name="Waktu Tertidur", value=f"{lat} menit",
            status="warn", tip=f"Butuh {lat} menit untuk tertidur. Coba teknik napas 4-7-8 atau meditasi singkat sebelum tidur."))
    else:
        factors.append(dict(icon="⏱️", name="Waktu Tertidur", value=f"{lat} menit",
            status="bad",  tip=f"Terlalu lama tertidur ({lat} menit). Hindari berbaring di tempat tidur sambil main HP dan kurangi kafein setelah jam 2 siang."))

    # 6. Terbangun per Malam
    wake = raw["wake_episodes_per_night"]
    if wake <= 1:
        factors.append(dict(icon="🔔", name="Terbangun Malam", value=f"{wake}x",
            status="ok",   tip="Jarang terbangun. Tidurmu nyenyak dan tidak terganggu."))
    elif wake <= 3:
        factors.append(dict(icon="🔔", name="Terbangun Malam", value=f"{wake}x",
            status="warn", tip=f"Terbangun {wake}x per malam masih bisa ditoleransi. Pastikan tidak minum banyak cairan menjelang tidur."))
    else:
        factors.append(dict(icon="🔔", name="Terbangun Malam", value=f"{wake}x",
            status="bad",  tip=f"Sering terbangun ({wake}x) sangat mengganggu siklus tidur. Periksa apakah ada gangguan lingkungan (cahaya, suara) atau pertimbangkan konsultasi dokter."))

    # 7. Tingkat Stres
    s = raw["stress_score"]
    if s <= 4:
        factors.append(dict(icon="🧘", name="Tingkat Stres", value=f"{s:.1f} / 10",
            status="ok",   tip="Stres terkendali. Pertahankan kebiasaan relaksasi yang sudah kamu lakukan."))
    elif s <= 7:
        factors.append(dict(icon="🧘", name="Tingkat Stres", value=f"{s:.1f} / 10",
            status="warn", tip=f"Stres cukup tinggi ({s:.1f}/10). Coba journaling, jalan santai 15 menit, atau mengurangi notifikasi HP sebelum tidur."))
    else:
        factors.append(dict(icon="🧘", name="Tingkat Stres", value=f"{s:.1f} / 10",
            status="bad",  tip=f"Stres sangat tinggi ({s:.1f}/10) dan berdampak besar pada performa kognitif. Pertimbangkan berbicara dengan orang terpercaya atau profesional."))

    # 8. Kafein
    caf = raw["caffeine_mg_before_bed"]
    if caf == 0:
        factors.append(dict(icon="☕", name="Kafein Sebelum Tidur", value="Tidak ada",
            status="ok",   tip="Tidak mengonsumsi kafein sebelum tidur. Bagus untuk kualitas tidur."))
    elif caf <= 80:
        factors.append(dict(icon="☕", name="Kafein Sebelum Tidur", value=f"{caf} mg",
            status="warn", tip=f"Kafein {caf}mg masih dalam batas aman, tapi idealnya nol. Hindari kopi/teh setelah jam 2 siang."))
    else:
        factors.append(dict(icon="☕", name="Kafein Sebelum Tidur", value=f"{caf} mg",
            status="bad",  tip=f"Kafein {caf}mg sebelum tidur mengganggu fase tidur dalam. Kafein bertahan 6–8 jam dalam tubuh — stop konsumsi setelah jam 2 siang."))

    # 9. Screen Time
    sc = raw["screen_time_before_bed_mins"]
    if sc <= 30:
        factors.append(dict(icon="📱", name="Screen Time Sebelum Tidur", value=f"{sc} menit",
            status="ok",   tip="Screen time sebelum tidur sudah terbatas. Cahaya biru dari layar bisa menekan produksi melatonin."))
    elif sc <= 60:
        factors.append(dict(icon="📱", name="Screen Time Sebelum Tidur", value=f"{sc} menit",
            status="warn", tip=f"Screen time {sc} menit cukup tinggi. Coba ganti 30 menit terakhir dengan membaca buku fisik atau mendengarkan musik tenang."))
    else:
        factors.append(dict(icon="📱", name="Screen Time Sebelum Tidur", value=f"{sc} menit",
            status="bad",  tip=f"Screen time {sc} menit sebelum tidur sangat mengganggu produksi melatonin. Aktifkan mode malam di HP dan stop pakai layar 1 jam sebelum tidur."))

    # 10. Olahraga
    if raw["exercise_day"]:
        factors.append(dict(icon="🏃", name="Olahraga Hari Ini", value="Ya",
            status="ok",   tip="Olahraga meningkatkan kualitas tidur dan performa kognitif. Pertahankan!"))
    else:
        factors.append(dict(icon="🏃", name="Olahraga Hari Ini", value="Tidak",
            status="warn", tip="Tidak olahraga hari ini. Aktivitas fisik ringan 20–30 menit (jalan cepat, stretching) sudah cukup untuk meningkatkan kualitas tidur."))

    return factors


def overall_banner(score):
    if score >= 85:
        return "banner-great", "🎉 Performa kognitifmu sangat baik hari ini! Pola tidur dan gaya hidupmu sangat mendukung kemampuan berpikir, konsentrasi, dan produktivitas. Pertahankan!"
    if score >= 70:
        return "banner-good", "👍 Performa kognitifmu baik. Ada beberapa area kecil yang bisa ditingkatkan — lihat analisis di bawah untuk saran spesifik."
    if score >= 55:
        return "banner-ok", "⚠️ Performa kognitifmu cukup, namun ada beberapa faktor yang perlu diperhatikan. Perbaikan kecil pada tidur dan gaya hidup bisa memberi dampak besar."
    return "banner-bad", "🔴 Performa kognitifmu rendah. Tidur yang kurang berkualitas dan gaya hidup yang kurang sehat kemungkinan jadi penyebabnya. Ikuti saran di bawah untuk mulai memperbaiki."

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 0.5rem; text-align:center;'>
      <span style='font-size:2rem;'>🧠</span>
      <div style='font-size:1.1rem; font-weight:800; color:white; margin-top:4px; letter-spacing:-0.01em;'>CPS Predictor</div>
      <div style='font-size:0.75rem; color:#A8E6CE; margin-top:2px;'>Cognitive Performance Score</div>
    </div>
    <hr style='border-color:rgba(78,203,160,0.2); margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">😴 Pola Tidur</div>', unsafe_allow_html=True)

    sleep_duration = st.slider("Durasi Tidur (jam)", 0.0, 14.0, 7.0, 0.5)
    sleep_quality  = st.slider("Kualitas Tidur (0–100)", 0.0, 100.0, 75.0, 1.0)
    rem_pct        = st.slider("REM Sleep (%)", 0.0, 40.0, 20.0, 1.0)
    deep_pct       = st.slider("Deep Sleep (%)", 0.0, 40.0, 20.0, 1.0)
    latency        = st.slider("Waktu Tertidur (menit)", 0, 90, 15, 1)
    wake_ep        = st.slider("Terbangun per Malam", 0, 10, 1, 1)
    felt_rested    = st.checkbox("Merasa Segar Setelah Bangun", value=True)
    sleep_disorder = st.selectbox("Risiko Gangguan Tidur", ["Healthy", "Mild", "Moderate", "Severe"])

    st.markdown('<div class="sidebar-section">⚡ Gaya Hidup</div>', unsafe_allow_html=True)

    stress        = st.slider("Tingkat Stres (0–10)", 0.0, 10.0, 3.0, 0.5)
    caffeine      = st.slider("Kafein Sebelum Tidur (mg)", 0, 400, 50, 10)
    screen_time   = st.slider("Screen Time Sebelum Tidur (menit)", 0, 180, 30, 5)
    exercise      = st.checkbox("Olahraga Hari Ini", value=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    predict_btn = st.button("🔍  Prediksi Sekarang")

# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>Cognitive Performance Score</h1>
  <p>Prediksi kemampuan kognitif berdasarkan pola tidur & gaya hidup kamu hari ini</p>
</div>
""", unsafe_allow_html=True)

if predict_btn:
    raw = {
        "sleep_duration_hrs":          sleep_duration,
        "sleep_quality_score":         sleep_quality,
        "rem_percentage":              rem_pct,
        "deep_sleep_percentage":       deep_pct,
        "sleep_latency_mins":          latency,
        "wake_episodes_per_night":     wake_ep,
        "felt_rested":                 felt_rested,
        "sleep_disorder_risk":         sleep_disorder,
        "stress_score":                stress,
        "caffeine_mg_before_bed":      caffeine,
        "screen_time_before_bed_mins": screen_time,
        "exercise_day":                exercise,
    }

    try:
        input_df = build_model_input(raw)
        scaled   = scaler.transform(input_df)
        pred     = float(np.round(model.predict(scaled)[0], 1))
        label, badge_cls, _ = score_category(pred)
        factors = analyze_factors(raw)
        banner_cls, banner_msg = overall_banner(pred)

        # ── Row 1: Score card + banner ──
        col_main, col_side = st.columns([1, 1.4], gap="large")

        with col_main:
            st.markdown(f"""
            <div class="result-card">
              <div class="score-label">Cognitive Performance Score</div>
              <div class="gauge-wrap">{gauge_svg(pred)}</div>
              <div class="score-value">{pred}</div>
              <div style='margin-top:0.8rem;'>
                <span class="category-badge {badge_cls}">{label}</span>
              </div>
              <div style='margin-top:1.2rem; font-size:0.82rem; color:#9CA3AF;'>
                Estimasi model machine learning — bukan diagnosis medis
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_side:
            st.markdown(f'<div class="overall-banner {banner_cls}">{banner_msg}</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="analysis-title">📊 Analisis Per Faktor</div>',
                        unsafe_allow_html=True)

            status_icon = {"ok": "✅", "warn": "⚠️", "bad": "❌"}
            status_cls  = {"ok": "status-ok", "warn": "status-warn", "bad": "status-bad"}

            for f in factors:
                si = status_icon[f["status"]]
                sc_cls = status_cls[f["status"]]
                st.markdown(f"""
                <div class="factor-card">
                  <div class="factor-icon">{f['icon']}</div>
                  <div class="factor-body">
                    <div class="factor-name">{f['name']}</div>
                    <div class="factor-value">{f['value']}</div>
                    <div class="factor-tip">{f['tip']}</div>
                  </div>
                  <div class="factor-status {sc_cls}">{si}</div>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediksi gagal: {e}")

else:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">🌿</div>
      <h3>Siap memprediksi performa kognitif kamu?</h3>
      <p>Isi data tidur & gaya hidup di panel kiri, lalu tekan <strong>Prediksi Sekarang</strong></p>
    </div>
    """, unsafe_allow_html=True)
