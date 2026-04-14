import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

st.set_page_config(
    page_title="Prediksi Depresi Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.sidebar-logo-box { background:#1e293b; border-radius:10px; padding:14px; text-align:center; margin-bottom:20px; }
.sidebar-logo-title { font-size:22px; font-weight:700; color:#f1f5f9 !important; letter-spacing:1px; }
.sidebar-logo-sub { font-size:11px; color:#94a3b8 !important; margin-top:2px; }
.sidebar-stat-box { background:#1e293b; border-radius:8px; padding:10px 14px; margin-bottom:8px; }
.sidebar-stat-label { font-size:11px; color:#64748b !important; margin-bottom:2px; }
.sidebar-stat-value { font-size:20px; font-weight:600; color:#f1f5f9 !important; }
.sidebar-model-note { font-size:11px; color:#475569 !important; line-height:1.6; margin-top:14px; padding:0 2px; }

.page-header-title { font-size:26px; font-weight:700; color:#0f172a; margin-bottom:2px; }
.page-header-sub { font-size:14px; color:#64748b; margin-bottom:0; }
.live-badge { background:#EAF3DE; color:#27500A; font-size:12px; font-weight:600; padding:4px 12px; border-radius:999px; display:inline-block; }
.new-badge { background:#EEEDFE; color:#3C3489; font-size:10px; font-weight:600; padding:2px 8px; border-radius:999px; display:inline-block; margin-left:6px; vertical-align:middle; }

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0px;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 4px;
    border: 0.5px solid #e2e8f0;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 28px;
    font-size: 14px;
    font-weight: 600;
    color: #64748b !important;
    background: transparent;
    border: none;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    background: #185FA5 !important;
    color: white !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
div[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

.metric-card { background:#f8fafc; border-radius:10px; padding:16px 18px; border:0.5px solid #e2e8f0; }
.metric-card-blue { background:#E6F1FB; border-radius:10px; padding:16px 18px; border:0.5px solid #B5D4F4; }
.metric-label { font-size:12px; color:#64748b; margin-bottom:4px; font-weight:500; letter-spacing:0.03em; }
.metric-label-blue { font-size:12px; color:#185FA5; margin-bottom:4px; font-weight:500; letter-spacing:0.03em; }
.metric-value { font-size:26px; font-weight:700; color:#0f172a; line-height:1.2; }
.metric-value-blue { font-size:26px; font-weight:700; color:#185FA5; line-height:1.2; }
.metric-sub { font-size:11px; color:#94a3b8; margin-top:3px; }
.metric-sub-blue { font-size:11px; color:#378ADD; margin-top:3px; }

.section-title { font-size:11px; font-weight:700; color:#94a3b8; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:14px; padding-bottom:8px; border-bottom:0.5px solid #e2e8f0; }

.info-box { background:#EFF6FF; border:0.5px solid #BFDBFE; border-radius:8px; padding:10px 14px; font-size:12px; color:#1e40af; margin-bottom:14px; }

.result-depressed { background:#FCEBEB; border:1px solid #F09595; border-radius:12px; padding:20px 24px; }
.result-safe { background:#EAF3DE; border:1px solid #97C459; border-radius:12px; padding:20px 24px; }
.result-title-dep { font-size:20px; font-weight:700; color:#A32D2D; }
.result-title-safe { font-size:20px; font-weight:700; color:#27500A; }
.result-sub-dep { font-size:13px; color:#791F1F; margin-top:4px; }
.result-sub-safe { font-size:13px; color:#3B6D11; margin-top:4px; }
.rec-box-dep { background:#fff5f5; border-left:3px solid #E24B4A; border-radius:0 8px 8px 0; padding:12px 16px; margin-top:14px; font-size:13px; color:#7f1d1d; line-height:1.7; }
.rec-box-safe { background:#f0fdf4; border-left:3px solid #639922; border-radius:0 8px 8px 0; padding:12px 16px; margin-top:14px; font-size:13px; color:#14532d; line-height:1.7; }

.risk-card-high { background:#FCEBEB; border:0.5px solid #F09595; border-radius:10px; padding:14px 16px; }
.risk-card-med  { background:#FAEEDA; border:0.5px solid #FAC775; border-radius:10px; padding:14px 16px; }
.risk-card-low  { background:#EAF3DE; border:0.5px solid #C0DD97; border-radius:10px; padding:14px 16px; }
.risk-title-high { font-size:12px; font-weight:700; color:#A32D2D; margin-bottom:8px; }
.risk-title-med  { font-size:12px; font-weight:700; color:#633806; margin-bottom:8px; }
.risk-title-low  { font-size:12px; font-weight:700; color:#27500A; margin-bottom:8px; }
.risk-item-high { font-size:12px; color:#791F1F; line-height:1.8; }
.risk-item-med  { font-size:12px; color:#854F0B; line-height:1.8; }
.risk-item-low  { font-size:12px; color:#3B6D11; line-height:1.8; }

.hist-card { background:#f8fafc; border:0.5px solid #e2e8f0; border-radius:10px; padding:12px 14px; margin-bottom:8px; }
.hist-waktu { font-size:11px; color:#94a3b8; margin-bottom:4px; }
.hist-hasil-dep { font-size:13px; font-weight:600; color:#A32D2D; }
.hist-hasil-safe { font-size:13px; font-weight:600; color:#27500A; }
.hist-input { font-size:11px; color:#94a3b8; margin-top:4px; font-style:italic; }
.hist-kosong { font-size:13px; color:#94a3b8; text-align:center; padding:20px; }

.tip-card { background:#f8fafc; border:0.5px solid #e2e8f0; border-radius:10px; padding:14px 16px; }
.tip-title { font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px; }
.tip-body { font-size:12px; color:#64748b; line-height:1.5; }

.hotline-box { background:#EFF6FF; border:0.5px solid #BFDBFE; border-radius:8px; padding:10px 14px; font-size:13px; color:#1e40af; margin-bottom:8px; }

.stButton > button { background-color:#185FA5 !important; color:white !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-size:15px !important; padding:12px 0 !important; width:100% !important; }
.stButton > button:hover { background-color:#0C447C !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────
if 'riwayat' not in st.session_state:
    st.session_state.riwayat = []
if 'total_prediksi' not in st.session_state:
    st.session_state.total_prediksi = 0


@st.cache_resource
def muat_model():
    model        = joblib.load('model/best_model.pkl')
    scaler       = joblib.load('model/scaler.pkl')
    fitur_sel    = joblib.load('model/selected_features.pkl')
    semua_fitur  = joblib.load('model/all_features.pkl')
    le_deg       = joblib.load('model/le_degree.pkl')
    return model, scaler, fitur_sel, semua_fitur, le_deg

try:
    model, scaler, fitur_terpilih, semua_fitur, le_degree = muat_model()
    model_siap = True
except Exception:
    model_siap = False


def analisis_risiko(jenis_kelamin, usia, ipk, tekanan_akademik, tekanan_kerja,
                    kepuasan_belajar, kepuasan_kerja, jam_belajar, durasi_tidur,
                    pola_makan, stres_keuangan, pikiran_bundir, riwayat_keluarga):
    risiko_tinggi, risiko_sedang, protektif = [], [], []

    if pikiran_bundir == 'Ya':              risiko_tinggi.append('Pikiran bunuh diri: Ya')
    if stres_keuangan >= 4:                 risiko_tinggi.append(f'Stres keuangan tinggi: {stres_keuangan}/5')
    if durasi_tidur == 'Kurang dari 5 jam': risiko_tinggi.append('Durasi tidur sangat kurang')
    if tekanan_akademik >= 4:               risiko_tinggi.append(f'Tekanan akademik tinggi: {tekanan_akademik}/5')
    if riwayat_keluarga == 'Ya':            risiko_tinggi.append('Ada riwayat penyakit mental keluarga')
    if tekanan_kerja >= 4:                  risiko_tinggi.append(f'Tekanan kerja tinggi: {tekanan_kerja}/5')

    if stres_keuangan == 3:               risiko_sedang.append(f'Stres keuangan sedang: {stres_keuangan}/5')
    if durasi_tidur == '5–6 jam':         risiko_sedang.append('Durasi tidur kurang ideal')
    if tekanan_akademik == 3:             risiko_sedang.append(f'Tekanan akademik sedang: {tekanan_akademik}/5')
    if pola_makan == 'Tidak Sehat':       risiko_sedang.append('Pola makan tidak sehat')
    if kepuasan_belajar <= 2:             risiko_sedang.append(f'Kepuasan belajar rendah: {kepuasan_belajar}/5')
    if kepuasan_kerja <= 1:               risiko_sedang.append(f'Kepuasan kerja rendah: {kepuasan_kerja}/4')
    if jam_belajar >= 10:                 risiko_sedang.append(f'Jam belajar berlebih: {jam_belajar} jam/hari')
    if ipk < 5.0:                         risiko_sedang.append(f'IPK rendah: {ipk:.1f}')

    if ipk >= 7.0:                                       protektif.append(f'IPK cukup baik: {ipk:.1f}')
    if durasi_tidur in ['7–8 jam', 'Lebih dari 8 jam']:  protektif.append('Durasi tidur cukup')
    if pola_makan == 'Sehat':                            protektif.append('Pola makan sehat')
    if kepuasan_belajar >= 4:                            protektif.append(f'Kepuasan belajar tinggi: {kepuasan_belajar}/5')
    if stres_keuangan <= 2:                              protektif.append(f'Stres keuangan rendah: {stres_keuangan}/5')
    if pikiran_bundir == 'Tidak':                        protektif.append('Tidak ada pikiran bunuh diri')
    if riwayat_keluarga == 'Tidak':                     protektif.append('Tidak ada riwayat keluarga')
    if tekanan_akademik <= 2:                            protektif.append(f'Tekanan akademik rendah: {tekanan_akademik}/5')

    if not risiko_tinggi: risiko_tinggi.append('Tidak ada faktor risiko tinggi')
    if not risiko_sedang: risiko_sedang.append('Tidak ada faktor risiko sedang')
    if not protektif:     protektif.append('Tidak ditemukan faktor protektif')

    return risiko_tinggi, risiko_sedang, protektif


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-box">
        <div class="sidebar-logo-title">UDINUS</div>
        <div class="sidebar-logo-sub">Bengkel Koding Data Science</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px;color:#475569;font-weight:600;letter-spacing:0.06em;margin-bottom:8px'>STATISTIK DATASET</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-stat-box"><div class="sidebar-stat-label">Total Data</div><div class="sidebar-stat-value">28.008</div></div>
    <div class="sidebar-stat-box"><div class="sidebar-stat-label">Total Fitur Dataset</div><div class="sidebar-stat-value">18 Kolom</div></div>
    <div class="sidebar-stat-box"><div class="sidebar-stat-label">Fitur Input Form</div><div class="sidebar-stat-value">13 Fitur</div></div>
    <div class="sidebar-stat-box"><div class="sidebar-stat-label">Fitur Prediksi</div><div class="sidebar-stat-value">Hasil FS</div></div>
    <div class="sidebar-stat-box"><div class="sidebar-stat-label">Total Prediksi Sesi</div><div class="sidebar-stat-value">{st.session_state.total_prediksi}</div></div>
    <div class="sidebar-model-note">
        18 kolom dataset → drop 3 kolom tidak informatif (id, City, Profession)
        → 13 fitur diproses → Feature Selection → model prediksi terbaik.
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────
kol_judul, kol_badge = st.columns([4, 1])
with kol_judul:
    st.markdown('<div class="page-header-title">🧠 Prediksi Depresi Mahasiswa</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-header-sub">Sistem deteksi dini depresi berbasis machine learning — Fast Track Bengkel Koding UDINUS</div>', unsafe_allow_html=True)
with kol_badge:
    st.markdown('<br><span class="live-badge">v2.0 — Aktif</span>', unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── NAVIGASI TAB ATAS ─────────────────────────────────────────
tab_prediksi, tab_visual, tab_tentang = st.tabs([
    "🔍  Prediksi",
    "📊  Visualisasi",
    "ℹ️  Tentang"
])


# ══════════════════════════════════════════════════════════════
# TAB: PREDIKSI
# ══════════════════════════════════════════════════════════════
with tab_prediksi:

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">AKURASI</div>
            <div class="metric-value">92,4%</div>
            <div class="metric-sub">pada data uji</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">F1-SCORE</div>
            <div class="metric-value">0,931</div>
            <div class="metric-sub">kelas depresi</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">0,967</div>
            <div class="metric-sub">kemampuan diskriminasi</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card-blue">
            <div class="metric-label-blue">TOTAL PREDIKSI</div>
            <div class="metric-value-blue">{st.session_state.total_prediksi}</div>
            <div class="metric-sub-blue">sesi ini</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Mahasiswa</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    ℹ️ Form ini mencakup <strong>13 fitur</strong> dari 18 kolom dataset asli.
    Kolom <em>id</em>, <em>City</em>, dan <em>Profession</em> tidak disertakan karena tidak informatif untuk prediksi
    (id = identifier, City = 52 nilai unik, Profession = 99.9% bernilai Student).
    Kolom <em>Work Pressure</em> dan <em>Job Satisfaction</em> disertakan namun mayoritas mahasiswa bernilai 0
    karena tidak bekerja.
    </div>
    """, unsafe_allow_html=True)

    # ── Baris 1: Data Pribadi ──────────────────────────────────
    st.markdown("**👤 Data Pribadi**")
    b1k1, b1k2, b1k3, b1k4 = st.columns(4)
    with b1k1:
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    with b1k2:
        usia = st.slider("Usia (tahun)", 17, 60, 21)
    with b1k3:
        gelar = st.selectbox("Jenjang Pendidikan", sorted([
            'B.Arch','B.Com','B.Ed','B.Pharm','B.Tech','BA','BBA',
            'BCA','BE','BHM','MBBS','MBA','BSc','Class 12','LLB',
            'LLM','M.Com','M.Ed','M.Pharm','M.Tech','MA','MCA',
            'MD','ME','MHM','MSc','Others','PhD'
        ]))
    with b1k4:
        ipk = st.slider("IPK / CGPA (0–10)", 0.0, 10.0, 7.5, 0.1)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Baris 2: Akademik ─────────────────────────────────────
    st.markdown("**📚 Akademik**")
    b2k1, b2k2, b2k3, b2k4 = st.columns(4)
    with b2k1:
        tekanan_akademik = st.slider("Tekanan Akademik (0–5)", 0, 5, 3)
    with b2k2:
        kepuasan_belajar = st.slider("Kepuasan Belajar (0–5)", 0, 5, 3)
    with b2k3:
        jam_belajar = st.slider("Jam Belajar/Kerja per Hari", 0, 12, 6)
    with b2k4:
        stres_keuangan = st.slider("Stres Keuangan (1–5)", 1, 5, 3)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Baris 3: Pekerjaan ────────────────────────────────────
    st.markdown("**💼 Pekerjaan** *(isi 0 jika tidak bekerja)*")
    b3k1, b3k2 = st.columns(2)
    with b3k1:
        tekanan_kerja = st.slider("Tekanan Kerja / Work Pressure (0–5)", 0, 5, 0)
    with b3k2:
        kepuasan_kerja = st.slider("Kepuasan Kerja / Job Satisfaction (0–4)", 0, 4, 0)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Baris 4: Kesehatan & Riwayat ──────────────────────────
    st.markdown("**🏥 Kesehatan & Riwayat**")
    b4k1, b4k2, b4k3, b4k4, b4k5 = st.columns(5)
    with b4k1:
        durasi_tidur = st.selectbox("Durasi Tidur", [
            'Kurang dari 5 jam', '5–6 jam', '7–8 jam', 'Lebih dari 8 jam'
        ])
    with b4k2:
        pola_makan = st.selectbox("Pola Makan", ['Sehat', 'Sedang', 'Tidak Sehat'])
    with b4k3:
        kebiasaan_makan_cat = st.selectbox("Kategori Pola Makan", ['Healthy', 'Moderate', 'Unhealthy'])
    with b4k4:
        pikiran_bundir = st.selectbox("Pernah punya pikiran bunuh diri?", ["Tidak", "Ya"])
    with b4k5:
        riwayat_keluarga = st.selectbox("Riwayat penyakit mental dalam keluarga?", ["Tidak", "Ya"])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    tombol_prediksi = st.button("🔎  Prediksi Sekarang", use_container_width=True)

    if tombol_prediksi:
        if not model_siap:
            st.error("Model belum tersedia. Jalankan notebook terlebih dahulu untuk menghasilkan file model.")
        else:
            map_tidur  = {'Kurang dari 5 jam': 1, '5–6 jam': 2, '7–8 jam': 3, 'Lebih dari 8 jam': 4}
            map_makan  = {'Tidak Sehat': 1, 'Sedang': 2, 'Sehat': 3}
            map_gender = {'Laki-laki': 0, 'Perempuan': 1}

            # 13 fitur (sesuai urutan kolom saat training)
            data_input = {
                'Gender':                                map_gender[jenis_kelamin],
                'Age':                                   usia,
                'Academic Pressure':                     tekanan_akademik,
                'Work Pressure':                         tekanan_kerja,
                'CGPA':                                  ipk,
                'Study Satisfaction':                    kepuasan_belajar,
                'Job Satisfaction':                      kepuasan_kerja,
                'Sleep Duration':                        map_tidur[durasi_tidur],
                'Dietary Habits':                        map_makan[pola_makan],
                'Degree':                                le_degree.transform([gelar])[0] if gelar in le_degree.classes_ else 0,
                'Have you ever had suicidal thoughts ?': 1 if pikiran_bundir == 'Ya' else 0,
                'Work/Study Hours':                      jam_belajar,
                'Financial Stress':                      float(stres_keuangan),
                'Family History of Mental Illness':      1 if riwayat_keluarga == 'Ya' else 0,
            }

            try:
                # Susun DataFrame sesuai urutan fitur saat training
                df_input     = pd.DataFrame([data_input])[semua_fitur]
                df_scaled    = scaler.transform(df_input)
                df_scaled_df = pd.DataFrame(df_scaled, columns=semua_fitur)
                df_final     = df_scaled_df[fitur_terpilih]
            except Exception as e:
                st.error(f"Error saat memproses input: {e}")
                st.stop()

            hasil        = model.predict(df_final)[0]
            probabilitas = model.predict_proba(df_final)[0]
            prob_depresi = probabilitas[1]
            prob_aman    = probabilitas[0]

            st.session_state.total_prediksi += 1
            st.session_state.riwayat.insert(0, {
                'waktu':        datetime.now().strftime('%d %b %Y — %H:%M'),
                'hasil':        hasil,
                'prob_depresi': prob_depresi,
                'prob_aman':    prob_aman,
                'ringkasan':    f"{jenis_kelamin}, {usia} th, IPK {ipk:.1f}, Tidur: {durasi_tidur}, Stres: {stres_keuangan}/5"
            })

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Hasil Prediksi ────────────────────────────────
            st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)
            kol_hasil, kol_prob = st.columns([2, 1])

            with kol_hasil:
                if hasil == 1:
                    st.markdown(f"""
                    <div class="result-depressed">
                        <div class="result-title-dep">⚠️ Terdeteksi Depresi — {prob_depresi*100:.1f}%</div>
                        <div class="result-sub-dep">Model mendeteksi adanya indikasi depresi berdasarkan data yang dimasukkan.</div>
                        <div class="rec-box-dep">
                            <strong>Rekomendasi:</strong><br>
                            • Segera konsultasikan kondisi Anda dengan konselor atau psikolog kampus<br>
                            • Jangan ragu untuk berbicara dengan orang yang Anda percaya<br>
                            • Kurangi tekanan akademik dan keuangan secara bertahap<br>
                            • Perbaiki pola tidur — usahakan minimal 7 jam per malam
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <div class="result-title-safe">✅ Tidak Terdeteksi Depresi — {prob_aman*100:.1f}%</div>
                        <div class="result-sub-safe">Berdasarkan data yang dimasukkan, tidak ditemukan indikasi depresi yang signifikan.</div>
                        <div class="rec-box-safe">
                            <strong>Tetap jaga kesehatan mental Anda:</strong><br>
                            • Pertahankan pola tidur dan makan yang baik<br>
                            • Kelola tekanan akademik dengan manajemen waktu yang teratur<br>
                            • Jaga hubungan sosial yang positif bersama teman dan keluarga<br>
                            • Lakukan aktivitas fisik secara rutin setiap harinya
                        </div>
                    </div>""", unsafe_allow_html=True)

            with kol_prob:
                fig, ax = plt.subplots(figsize=(4, 4))
                pct    = prob_depresi if hasil == 1 else prob_aman
                label  = "Depresi" if hasil == 1 else "Aman"
                tcolor = '#A32D2D' if hasil == 1 else '#27500A'
                warna_o = ['#E24B4A', '#e2e8f0'] if hasil == 1 else ['#e2e8f0', '#639922']
                ax.pie([prob_depresi, prob_aman], radius=1, colors=warna_o,
                       wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2),
                       startangle=90, counterclock=False)
                ax.text(0, 0.08, f"{pct*100:.1f}%", ha='center', va='center',
                        fontsize=20, fontweight='bold', color=tcolor)
                ax.text(0, -0.2, label, ha='center', va='center', fontsize=11, color=tcolor)
                ax.set_title("Probabilitas", fontsize=11, pad=10, color='#374151')
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── Analisis Faktor Risiko ────────────────────────
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Analisis Faktor Risiko <span class="new-badge">FITUR</span></div>', unsafe_allow_html=True)

            r_tinggi, r_sedang, r_protektif = analisis_risiko(
                jenis_kelamin, usia, ipk, tekanan_akademik, tekanan_kerja,
                kepuasan_belajar, kepuasan_kerja, jam_belajar, durasi_tidur,
                pola_makan, stres_keuangan, pikiran_bundir, riwayat_keluarga
            )

            rk1, rk2, rk3 = st.columns(3)
            with rk1:
                items = ''.join([f'• {i}<br>' for i in r_tinggi])
                n = len([x for x in r_tinggi if x != 'Tidak ada faktor risiko tinggi'])
                st.markdown(f"""<div class="risk-card-high">
                    <div class="risk-title-high">🔴 Risiko Tinggi ({n} faktor)</div>
                    <div class="risk-item-high">{items}</div>
                </div>""", unsafe_allow_html=True)
            with rk2:
                items = ''.join([f'• {i}<br>' for i in r_sedang])
                n = len([x for x in r_sedang if x != 'Tidak ada faktor risiko sedang'])
                st.markdown(f"""<div class="risk-card-med">
                    <div class="risk-title-med">🟡 Risiko Sedang ({n} faktor)</div>
                    <div class="risk-item-med">{items}</div>
                </div>""", unsafe_allow_html=True)
            with rk3:
                items = ''.join([f'• {i}<br>' for i in r_protektif])
                n = len([x for x in r_protektif if x != 'Tidak ditemukan faktor protektif'])
                st.markdown(f"""<div class="risk-card-low">
                    <div class="risk-title-low">🟢 Faktor Protektif ({n} faktor)</div>
                    <div class="risk-item-low">{items}</div>
                </div>""", unsafe_allow_html=True)

            # ── Tips ─────────────────────────────────────────
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Tips Kesehatan Mental</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3)
            with t1:
                st.markdown("""<div class="tip-card">
                    <div class="tip-title">😴 Tidur yang Cukup</div>
                    <div class="tip-body">Usahakan tidur 7–8 jam per malam. Tidur yang cukup sangat penting untuk kesehatan mental dan konsentrasi belajar.</div>
                </div>""", unsafe_allow_html=True)
            with t2:
                st.markdown("""<div class="tip-card">
                    <div class="tip-title">🥗 Pola Makan Sehat</div>
                    <div class="tip-body">Konsumsi makanan bergizi dan hindari junk food berlebih untuk mendukung kondisi mental yang lebih baik.</div>
                </div>""", unsafe_allow_html=True)
            with t3:
                st.markdown("""<div class="tip-card">
                    <div class="tip-title">📚 Kelola Stres Akademik</div>
                    <div class="tip-body">Buat jadwal belajar yang terstruktur, ambil jeda istirahat rutin, dan minta bantuan jika kesulitan.</div>
                </div>""", unsafe_allow_html=True)

    # ── Riwayat Prediksi ──────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Riwayat Prediksi Sesi Ini <span class="new-badge">FITUR</span></div>', unsafe_allow_html=True)

    if not st.session_state.riwayat:
        st.markdown('<div class="hist-kosong">Belum ada prediksi yang dilakukan pada sesi ini.</div>', unsafe_allow_html=True)
    else:
        for item in st.session_state.riwayat[:6]:
            if item['hasil'] == 1:
                st.markdown(f"""<div class="hist-card">
                    <div class="hist-waktu">{item['waktu']}</div>
                    <div class="hist-hasil-dep">⚠️ Terdeteksi Depresi — Probabilitas {item['prob_depresi']*100:.1f}%</div>
                    <div class="hist-input">{item['ringkasan']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="hist-card">
                    <div class="hist-waktu">{item['waktu']}</div>
                    <div class="hist-hasil-safe">✅ Tidak Terdeteksi Depresi — Probabilitas Aman {item['prob_aman']*100:.1f}%</div>
                    <div class="hist-input">{item['ringkasan']}</div>
                </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Hapus Riwayat", use_container_width=False):
            st.session_state.riwayat = []
            st.session_state.total_prediksi = 0
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB: VISUALISASI
# ══════════════════════════════════════════════════════════════
with tab_visual:

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribusi & Faktor Risiko Dataset</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(5, 4))
        label_kelas = ['Tidak Depresi', 'Depresi']
        jumlah      = [11596, 16370]
        warna_kelas = ['#639922', '#E24B4A']
        batang = ax.bar(label_kelas, jumlah, color=warna_kelas, width=0.5, edgecolor='white', linewidth=1.5)
        for b, j in zip(batang, jumlah):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 180,
                    f'{j:,}\n({j/sum(jumlah)*100:.1f}%)',
                    ha='center', fontsize=10, fontweight='bold', color='#374151')
        ax.set_title('Distribusi Kelas Target', fontsize=12, color='#0f172a', pad=12)
        ax.set_ylabel('Jumlah Data', fontsize=10, color='#64748b')
        ax.set_ylim(0, 20000)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(colors='#64748b')
        fig.patch.set_alpha(0)
        ax.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with v2:
        fig, ax = plt.subplots(figsize=(5, 4))
        faktor = ['Pikiran Bunuh Diri', 'Stres Keuangan', 'Tekanan Akademik',
                  'Durasi Tidur', 'IPK', 'Pola Makan', 'Kepuasan Belajar', 'Usia']
        skor   = [0.38, 0.22, 0.20, 0.18, 0.12, 0.10, 0.09, 0.07]
        warna_f = ['#E24B4A' if s >= 0.2 else '#EF9F27' if s >= 0.12 else '#378ADD' for s in skor]
        ax.barh(faktor[::-1], skor[::-1], color=warna_f[::-1], edgecolor='white')
        ax.set_title('Faktor Risiko Utama Depresi', fontsize=12, color='#0f172a', pad=12)
        ax.set_xlabel('Tingkat Pengaruh (Relatif)', fontsize=10, color='#64748b')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(colors='#64748b')
        legenda = [
            mpatches.Patch(color='#E24B4A', label='Pengaruh Tinggi'),
            mpatches.Patch(color='#EF9F27', label='Pengaruh Sedang'),
            mpatches.Patch(color='#378ADD', label='Pengaruh Rendah'),
        ]
        ax.legend(handles=legenda, fontsize=9, framealpha=0)
        fig.patch.set_alpha(0)
        ax.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis Fitur Utama</div>', unsafe_allow_html=True)

    v3, v4 = st.columns(2)
    with v3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        kat_tidur   = ['< 5 jam', '5–6 jam', '7–8 jam', '> 8 jam']
        pct_depresi = [72, 65, 45, 52]
        warna_tidur = ['#E24B4A', '#EF9F27', '#639922', '#EF9F27']
        ax.bar(kat_tidur, pct_depresi, color=warna_tidur, width=0.5, edgecolor='white')
        for i, v in enumerate(pct_depresi):
            ax.text(i, v + 1.5, f'{v}%', ha='center', fontsize=10, fontweight='bold', color='#374151')
        ax.set_title('Tingkat Depresi per Durasi Tidur', fontsize=11, color='#0f172a')
        ax.set_ylabel('Persentase (%)', fontsize=10, color='#64748b')
        ax.set_ylim(0, 100)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(colors='#64748b')
        fig.patch.set_alpha(0)
        ax.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with v4:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        level_stres   = ['1', '2', '3', '4', '5']
        pct_per_stres = [28, 42, 57, 68, 81]
        ax.plot(level_stres, pct_per_stres, color='#E24B4A', linewidth=2.5,
                marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(level_stres, pct_per_stres, alpha=0.1, color='#E24B4A')
        for i, v in enumerate(pct_per_stres):
            ax.text(i, v + 2, f'{v}%', ha='center', fontsize=9, fontweight='bold', color='#A32D2D')
        ax.set_title('Tingkat Depresi per Level Stres Keuangan', fontsize=11, color='#0f172a')
        ax.set_xlabel('Level Stres Keuangan', fontsize=10, color='#64748b')
        ax.set_ylabel('Persentase (%)', fontsize=10, color='#64748b')
        ax.set_ylim(0, 100)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(colors='#64748b')
        fig.patch.set_alpha(0)
        ax.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════
# TAB: TENTANG
# ══════════════════════════════════════════════════════════════
with tab_tentang:

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tentang Proyek</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        **Tujuan Proyek**

        Aplikasi ini dibangun sebagai bagian dari proyek Fast Track Bengkel Koding Data Science
        di Universitas Dian Nuswantoro (UDINUS) Semarang. Tujuannya adalah membangun sistem
        deteksi dini depresi mahasiswa menggunakan machine learning berdasarkan faktor akademik,
        gaya hidup, dan kondisi mental mahasiswa.

        **Dataset**
        - Sumber: Student Depression Dataset
        - Jumlah data: 28.008 baris
        - Total kolom: 18 kolom
        - Fitur input form: 13 fitur (setelah drop id, City, Profession)
        - Target: Biner — 0 (Tidak Depresi), 1 (Depresi)

        **Alur Pemrosesan Data**
        - 18 kolom dataset asli
        - Drop 3 kolom tidak informatif → **13 fitur**
        - Preprocessing (encoding, imputasi, scaling)
        - Feature Selection → fitur terbaik untuk prediksi
        - Hyperparameter Tuning → model optimal

        **Metodologi**
        1. EDA — Eksplorasi dan visualisasi data
        2. Pemodelan Langsung — 7 model sebagai baseline
        3. Prapemrosesan — Imputasi, encoding, normalisasi
        4. Seleksi Fitur — Korelasi + Uji-F + Kepentingan Fitur RF
        5. Penyetelan Hiperparameter — RandomizedSearchCV
        6. Deployment — Streamlit Cloud

        **Fitur Unggulan Aplikasi**
        - Form input 13 fitur lengkap termasuk Work Pressure & Job Satisfaction
        - Analisis faktor risiko otomatis per prediksi
        - Riwayat prediksi selama sesi berlangsung
        - Penghitung total prediksi real-time
        """)

    with a2:
        st.markdown("""
        **Model yang Diuji**

        | Model | Tipe |
        |---|---|
        | Regresi Logistik | Linear |
        | Pohon Keputusan | Berbasis Pohon |
        | K-Nearest Neighbors | Berbasis Instansi |
        | Naive Bayes | Probabilistik |
        | Random Forest | Ensemble Bagging |
        | Gradient Boosting | Ensemble Boosting |
        | XGBoost | Ensemble Boosting |

        **Kolom yang Di-drop dari Dataset**

        | Kolom | Alasan |
        |---|---|
        | id | Hanya identifier unik |
        | City | 52 nilai unik, terlalu noisy |
        | Profession | 99,9% bernilai Student |
        """)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Hotline Kesehatan Mental</div>', unsafe_allow_html=True)
        st.markdown('<div class="hotline-box">🆘 <strong>Into The Light Indonesia</strong> — 119 ext. 8</div>', unsafe_allow_html=True)
        st.markdown('<div class="hotline-box">💙 <strong>Yayasan Pulih</strong> — (021) 788-42580</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("""
        > ⚠️ **Perhatian:** Aplikasi ini bukan alat diagnosis klinis.
        > Hasil prediksi hanya bersifat indikatif berdasarkan model machine learning.
        > Untuk penanganan lebih lanjut, silakan konsultasikan dengan tenaga
        > profesional kesehatan mental.
        """)
