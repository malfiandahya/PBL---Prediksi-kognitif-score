import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Prediksi Cognitive Performance Score",
    page_icon="🧠",
    layout="wide"
)

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
    model.load_model(BOOSTER_PATH)          # format native XGBoost, bebas versi sklearn

    scaler   = joblib.load(SCALER_PATH)     # StandardScaler saja, tidak ada ColumnTransformer
    metadata = joblib.load(METADATA_PATH)
    return model, scaler, metadata


model, scaler, metadata = load_assets()
MODEL_COLUMNS        = metadata["columns"]
CATEGORICAL_OPTIONS  = metadata["categorical_options"]

SLEEP_RISK_MAP = {
    "Healthy":  0,
    "Mild":     1,
    "Moderate": 2,
    "Severe":   3,
}


def build_model_input(raw_input: dict) -> pd.DataFrame:
    """Ubah input user → DataFrame 59 kolom sesuai urutan training."""
    data = {col: 0 for col in MODEL_COLUMNS}

    numeric_values = {
        "age":                        raw_input["age"],
        "bmi":                        raw_input["bmi"],
        "sleep_duration_hrs":         raw_input["sleep_duration_hrs"],
        "sleep_quality_score":        raw_input["sleep_quality_score"],
        "rem_percentage":             raw_input["rem_percentage"],
        "deep_sleep_percentage":      raw_input["deep_sleep_percentage"],
        "sleep_latency_mins":         raw_input["sleep_latency_mins"],
        "wake_episodes_per_night":    raw_input["wake_episodes_per_night"],
        "caffeine_mg_before_bed":     raw_input["caffeine_mg_before_bed"],
        "alcohol_units_before_bed":   raw_input["alcohol_units_before_bed"],
        "screen_time_before_bed_mins":raw_input["screen_time_before_bed_mins"],
        "exercise_day":               int(raw_input["exercise_day"]),
        "steps_that_day":             raw_input["steps_that_day"],
        "nap_duration_mins":          raw_input["nap_duration_mins"],
        "stress_score":               raw_input["stress_score"],
        "work_hours_that_day":        raw_input["work_hours_that_day"],
        "heart_rate_resting_bpm":     raw_input["heart_rate_resting_bpm"],
        "sleep_aid_used":             int(raw_input["sleep_aid_used"]),
        "shift_work":                 int(raw_input["shift_work"]),
        "room_temperature_celsius":   raw_input["room_temperature_celsius"],
        "weekend_sleep_diff_hrs":     raw_input["weekend_sleep_diff_hrs"],
        "felt_rested":                int(raw_input["felt_rested"]),
        "sleep_disorder_risk_ord":    SLEEP_RISK_MAP[raw_input["sleep_disorder_risk"]],
    }

    for key, value in numeric_values.items():
        if key in data:
            data[key] = value

    one_hot_features = {
        "gender":                raw_input["gender"],
        "occupation":            raw_input["occupation"],
        "country":               raw_input["country"],
        "chronotype":            raw_input["chronotype"],
        "mental_health_condition": raw_input["mental_health_condition"],
        "season":                raw_input["season"],
        "day_type":              raw_input["day_type"],
    }

    for feature, selected_value in one_hot_features.items():
        encoded_col = f"{feature}_{selected_value}"
        if encoded_col in data:
            data[encoded_col] = 1

    return pd.DataFrame([data], columns=MODEL_COLUMNS)


def score_category(score: float) -> str:
    if score >= 85:
        return "Sangat Baik"
    if score >= 70:
        return "Baik"
    if score >= 55:
        return "Cukup"
    return "Perlu Perhatian"


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("🧠 Prediksi Cognitive Performance Score")
st.write(
    "Aplikasi ini memprediksi **Cognitive Performance Score (CPS)** berdasarkan pola tidur, "
    "aktivitas fisik, kondisi kesehatan, dan gaya hidup pengguna."
)

with st.sidebar:
    st.header("Tentang Aplikasi")
    st.write(
        "Model menerima data input dari pengguna, lalu mengubahnya ke format fitur yang sama "
        "seperti saat training sebelum menghasilkan prediksi CPS."
    )
    st.info("Pastikan file booster.ubj, scaler.pkl, dan metadata.pkl berada dalam folder yang sama dengan app.py.")

st.divider()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Data Diri")
        age          = st.number_input("Usia", min_value=10, max_value=100, value=25, step=1)
        gender       = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
        bmi          = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
        occupation   = st.selectbox("Pekerjaan", CATEGORICAL_OPTIONS["occupation"])
        country      = st.selectbox("Negara", CATEGORICAL_OPTIONS["country"])
        mental_health_condition = st.selectbox(
            "Kondisi Kesehatan Mental", CATEGORICAL_OPTIONS["mental_health_condition"]
        )

    with col2:
        st.subheader("Pola Tidur")
        sleep_duration_hrs      = st.slider("Durasi Tidur (jam)", 0.0, 14.0, 7.0, 0.1)
        sleep_quality_score     = st.slider("Skor Kualitas Tidur", 0.0, 100.0, 75.0, 1.0)
        rem_percentage          = st.slider("REM Sleep (%)", 0.0, 60.0, 20.0, 0.5)
        deep_sleep_percentage   = st.slider("Deep Sleep (%)", 0.0, 60.0, 20.0, 0.5)
        sleep_latency_mins      = st.slider("Waktu untuk Tertidur (menit)", 0, 180, 20, 1)
        wake_episodes_per_night = st.slider("Terbangun per Malam", 0, 15, 2, 1)
        sleep_disorder_risk     = st.selectbox(
            "Risiko Gangguan Tidur", CATEGORICAL_OPTIONS["sleep_disorder_risk"]
        )
        sleep_aid_used = st.checkbox("Menggunakan Bantuan Tidur")
        felt_rested    = st.checkbox("Merasa Segar Setelah Bangun", value=True)

    with col3:
        st.subheader("Gaya Hidup")
        caffeine_mg_before_bed       = st.slider("Kafein sebelum Tidur (mg)", 0, 500, 50, 5)
        alcohol_units_before_bed     = st.slider("Alkohol sebelum Tidur (unit)", 0.0, 10.0, 0.0, 0.5)
        screen_time_before_bed_mins  = st.slider("Screen Time sebelum Tidur (menit)", 0, 300, 60, 5)
        exercise_day                 = st.checkbox("Olahraga Hari Ini", value=True)
        steps_that_day               = st.number_input("Jumlah Langkah Hari Ini", min_value=0, max_value=50000, value=7000, step=500)
        nap_duration_mins            = st.slider("Durasi Tidur Siang (menit)", 0, 240, 20, 5)
        stress_score                 = st.slider("Skor Stres", 0.0, 10.0, 4.0, 0.1)
        work_hours_that_day          = st.slider("Jam Kerja Hari Ini", 0.0, 18.0, 8.0, 0.5)
        heart_rate_resting_bpm       = st.slider("Detak Jantung Istirahat (bpm)", 40, 130, 70, 1)
        shift_work                   = st.checkbox("Bekerja Shift")
        room_temperature_celsius     = st.slider("Suhu Ruangan (°C)", 10.0, 35.0, 24.0, 0.5)
        weekend_sleep_diff_hrs       = st.slider("Selisih Tidur Weekend vs Weekday (jam)", -5.0, 5.0, 0.5, 0.1)
        chronotype = st.selectbox("Chronotype", CATEGORICAL_OPTIONS["chronotype"])
        season     = st.selectbox("Musim", CATEGORICAL_OPTIONS["season"])
        day_type   = st.selectbox("Tipe Hari", CATEGORICAL_OPTIONS["day_type"])

    submitted = st.form_submit_button("Prediksi CPS")

if submitted:
    raw_input = {
        "age": age, "gender": gender, "bmi": bmi,
        "occupation": occupation, "country": country,
        "mental_health_condition": mental_health_condition,
        "sleep_duration_hrs": sleep_duration_hrs,
        "sleep_quality_score": sleep_quality_score,
        "rem_percentage": rem_percentage,
        "deep_sleep_percentage": deep_sleep_percentage,
        "sleep_latency_mins": sleep_latency_mins,
        "wake_episodes_per_night": wake_episodes_per_night,
        "sleep_disorder_risk": sleep_disorder_risk,
        "sleep_aid_used": sleep_aid_used,
        "felt_rested": felt_rested,
        "caffeine_mg_before_bed": caffeine_mg_before_bed,
        "alcohol_units_before_bed": alcohol_units_before_bed,
        "screen_time_before_bed_mins": screen_time_before_bed_mins,
        "exercise_day": exercise_day,
        "steps_that_day": steps_that_day,
        "nap_duration_mins": nap_duration_mins,
        "stress_score": stress_score,
        "work_hours_that_day": work_hours_that_day,
        "heart_rate_resting_bpm": heart_rate_resting_bpm,
        "shift_work": shift_work,
        "room_temperature_celsius": room_temperature_celsius,
        "weekend_sleep_diff_hrs": weekend_sleep_diff_hrs,
        "chronotype": chronotype,
        "season": season,
        "day_type": day_type,
    }

    input_df     = build_model_input(raw_input)
    input_scaled = scaler.transform(input_df)   # StandardScaler langsung, tanpa Pipeline

    try:
        prediction = model.predict(input_scaled)[0]
        prediction = float(np.round(prediction, 2))
        category   = score_category(prediction)

        st.success("Prediksi berhasil dibuat.")

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Prediksi Cognitive Performance Score", prediction)
        with result_col2:
            st.metric("Kategori", category)

        with st.expander("Lihat data input yang dikirim ke model"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "Catatan: Hasil prediksi merupakan estimasi dari model Machine Learning, "
            "bukan diagnosis medis atau penilaian klinis."
        )

    except Exception as error:
        st.error("Prediksi gagal dijalankan.")
        st.write("Detail error:")
        st.code(str(error))
