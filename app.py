from streamlit_option_menu import option_menu
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Aplikasi Prediksi Obesitas",
    page_icon="🍔",
    layout="centered"
)

@st.cache_resource
def load_model():
    loaded_model = joblib.load('models/model_random_forest.joblib')
    return loaded_model

def load_label_encoders():
    encoders = joblib.load('models/label_encoders.joblib')
    return encoders

def load_css(file_name: str):
    with open(file_name) as f:
        css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def is_valid_input(
    gender, family_history_with_overweight, favc, caec, smoke, scc, calc,
    mtrans, fcvc, ncp, faf, tue, ch2o
):
    """
    Memeriksa apakah input sudah diisi dengan benar.
    """
    # Cek radio
    radio_valid = all([
        family_history_with_overweight in ['Ya', 'Tidak'],
        favc in ['Ya', 'Tidak'],
        smoke in ['Ya', 'Tidak'],
        scc in ['Ya', 'Tidak'],
    ])
    # Cek selectbox
    select_valid = all([
        gender in ['Laki-laki', 'Perempuan'],
        calc in ['Tidak pernah', 'Kadang-kadang', 'Sering', "Selalu"],
        caec in ['Tidak pernah', 'Kadang-kadang', 'Sering', "Selalu"],
        mtrans in ['Mobil pribadi', 'Sepeda', 'Sepeda motor', 'Transportasi umum', 'Berjalan kaki'],
        ncp in ['1—2', '3', '>3'],
        fcvc in ['Tidak pernah', 'Kadang-kadang', 'Selalu'],
        ch2o in ['<1', '1—2', '>2'],
        faf in ['Tidak pernah', '1—2 hari', '2—4 hari', '4—5 hari'],
        tue in ['0—2 jam 0', '3—5 jam 1', '>5 jam 2'],
    ])
    return radio_valid and select_valid

def preprocess_input(
    encoders,
    gender, age, height, weight,
    family_history_with_overweight, favc, fcvc, ncp,
    caec, smoke, ch2o, scc, faf, tue,
    calc, mtrans
):
    """
    Melakukan preprocessing dan encoding terhadap data input dari form.
    """
    # Mapping manual
    fcvc_mapping = {"Tidak pernah": 1, "Kadang-kadang": 2, "Selalu": 3}
    ncp_mapping = {'1—2': 1, '3': 2, '>3': 3}
    ch2o_mapping = {'<1': 1, '1—2': 2, '>2': 3}
    faf_mapping = {'Tidak pernah': 0, '1—2 hari': 1, '2—4 hari': 2, '4—5 hari': 3}
    tue_mapping = {'0—2 jam 0': 0, '3—5 jam 1': 1, '>5 jam 2': 2}

    gender_mapping = {"Laki-laki": 'Male', "Perempuan": 'Female'}
    family_history_mapping = {"Ya": 'yes', "Tidak": 'no'}
    favc_mapping = {"Ya": 'yes', "Tidak": 'no'}
    caec_mapping = {
        "Tidak pernah": "no",
        "Kadang-kadang": "Sometimes",
        "Sering": "Frequently",
        "Selalu": "Always"
    }
    smoke_mapping = {"Ya": 'yes', "Tidak": 'no'}
    scc_mapping = {"Ya": 'yes', "Tidak": 'no'}
    calc_mapping = {
        "Tidak pernah": "no",
        "Kadang-kadang": "Sometimes",
        "Sering": "Frequently",
        "Selalu": "Always"
    }
    mtrans_mapping = {
        "Mobil pribadi": 'Automobile',
        "Sepeda motor": 'Motorbike',
        "Sepeda": 'Bike',
        "Transportasi umum": 'Public_Transportation',
        "Berjalan kaki": 'Walking'
    }

    try:
        # Konversi teks ke numerik
        age = int(age)
        height = float(height) / 100.0  # cm -> meter
        weight = float(weight)

        # Terjemahkan label ke bahasa Inggris
        gender_enc = gender_mapping[gender]
        family_history_enc = family_history_mapping[family_history_with_overweight]
        favc_enc = favc_mapping[favc]
        caec_enc = caec_mapping[caec]
        smoke_enc = smoke_mapping[smoke]
        scc_enc = scc_mapping[scc]
        calc_enc = calc_mapping[calc]
        mtrans_enc = mtrans_mapping[mtrans]

        # Label encoding dengan dict yang di-load
        data_dict = {
            'Gender': encoders['Gender'].transform([gender_enc])[0],
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history_with_overweight': encoders['family_history_with_overweight'].transform([family_history_enc])[0],
            'FAVC': encoders['FAVC'].transform([favc_enc])[0],
            'FCVC': fcvc_mapping[fcvc],
            'NCP': ncp_mapping[ncp],
            'CAEC': encoders['CAEC'].transform([caec_enc])[0],
            'SMOKE': encoders['SMOKE'].transform([smoke_enc])[0],
            'CH2O': ch2o_mapping[ch2o],
            'SCC': encoders['SCC'].transform([scc_enc])[0],
            'FAF': faf_mapping[faf],
            'TUE': tue_mapping[tue],
            'CALC': encoders['CALC'].transform([calc_enc])[0],
            'MTRANS': encoders['MTRANS'].transform([mtrans_enc])[0]
        }

    except Exception as e:
        st.error(f"Error saat preprocessing input: {e}")
        return None

    # Buat dataframe satu baris untuk diprediksi
    df = pd.DataFrame([data_dict])
    return df

try:
    load_css('styles.css') # Load CSS
except:
    pass

# Muat model dan encoders
model = load_model()
label_encoders = load_label_encoders()

# ======================== NAVIGASI ========================
with st.sidebar :
# st.sidebar.title("DAFTAR HALAMAN")
    page = option_menu(
                " ",  # Judul menu
                ["Latar Belakang", "Obesity Buddy: App Prediksi Obesitas", "Informasi Kelompok 10"],  # Opsi menu
                icons=["info-circle", "clipboard-data", "people"],  # Ikon opsi menu
                menu_icon="list",  # Ikon menu utama (hamburger menu)
                default_index=0,  # Indeks default yang dipilih
                orientation="vertical",  # Sidebar vertikal
            )


# ======================== HALAMAN: LATAR BELAKANG ========================
if page == "Latar Belakang":
    st.title("Latar Belakang")
    st.write(
        """
        **Obesitas** adalah kondisi medis yang terjadi akibat kelebihan berat badan berbahaya bagi kesehatan, yang diukur dengan Indeks Massa Tubuh (BMI). Menurut WHO, seseorang dianggap obesitas jika BMI lebih dari 30, yang dapat meningkatkan risiko penyakit kardiovaskular, diabetes tipe 2, hipertensi, dan gangguan mental. Di Indonesia, prevalensi obesitas meningkat dari 11,7% (2010) menjadi 15,4% (2013).

        Obesitas dipengaruhi oleh faktor genetik, pola hidup, diet buruk, kurang aktivitas fisik, serta kebiasaan tidur yang buruk. Untuk mencegahnya, deteksi dini sangat penting, dan perubahan pola makan serta peningkatan aktivitas fisik dapat mengurangi risikonya.

        Dengan penerapan teknologi **AI** dan algoritma **Random Forest**, aplikasi ini memanfaatkan data dari kuisioner kesehatan, pola makan, dan aktivitas fisik untuk memprediksi tingkat obesitas dan memberikan rekomendasi kesehatan yang lebih personal. Random Forest menggabungkan hasil banyak decision tree untuk meningkatkan akurasi prediksi, membantu mengelola berat badan dengan lebih efektif.
        """
    )


# ======================== HALAMAN: PREDIKSI OBESITAS ========================
elif page == "Obesity Buddy: App Prediksi Obesitas":
    st.title("Obesity Buddy: App Prediksi Obesitas")
    st.markdown(
        "Isi semua pertanyaan di bawah ini, lalu klik **Generate Hasil** untuk melihat prediksi tingkat obesitas."
    )

    with st.form(key='kuisioner_form'):
        kolom_kanan, kolom_kiri = st.columns(2)

        with kolom_kanan:
            gender = st.selectbox(
                'Jenis Kelamin',
                ['Pilih...', 'Laki-laki', 'Perempuan'],
                index=0
            )
            age = st.text_input(
                'Umur (tahun)',
                value='',
                placeholder='Masukkan umur Anda'
            )
            height = st.text_input(
                'Tinggi Badan (cm)',
                value='',
                placeholder='Masukkan tinggi badan Anda dalam cm'
            )
            weight = st.text_input(
                'Berat Badan (kg)',
                value='',
                placeholder='Masukkan berat badan Anda dalam kg'
            )
            family_history_with_overweight = st.radio(
                'Riwayat keluarga dengan obesitas?',
                ['Ya', 'Tidak'],
                horizontal=True
            )
            favc = st.radio(
                'Sering mengonsumsi makanan tinggi kalori?',
                ['Ya', 'Tidak'],
                horizontal=True
            )
            fcvc = st.selectbox(
                'Seberapa sering makan sayuran?',
                ['Pilih...', 'Tidak pernah', 'Kadang-kadang', 'Selalu'],
                index=0
            )
            ncp = st.selectbox(
                'Berapa kali makan utama per hari?',
                ['Pilih...', '1—2', '3', '>3'],
                index=0
            )

        with kolom_kiri:
            caec = st.selectbox(
                'Makan di antara waktu makan utama?',
                ['Pilih...', 'Tidak pernah', 'Kadang-kadang', 'Sering', 'Selalu'],
                index=0
            )
            smoke = st.radio(
                'Apakah Anda merokok?',
                ['Ya', 'Tidak'],
                horizontal=True
            )
            ch2o = st.selectbox(
                'Konsumsi air per hari (liter)?',
                ['Pilih...', '<1', '1—2', '>2'],
                index=0
            )
            scc = st.radio(
                'Memantau kalori harian?',
                ['Ya', 'Tidak'],
                horizontal=True
            )
            faf = st.selectbox(
                'Frekuensi aktivitas fisik/minggu?',
                ['Pilih...', 'Tidak pernah', '1—2 hari', '2—4 hari', '4—5 hari'],
                index=0
            )
            tue = st.selectbox(
                'Jam penggunaan perangkat teknologi/hari?',
                ['Pilih...', '0—2 jam 0', '3—5 jam 1', '>5 jam 2'],
                index=0
            )
            calc = st.selectbox(
                'Frekuensi konsumsi alkohol?',
                ['Pilih...', 'Tidak pernah', 'Kadang-kadang', 'Sering', 'Selalu'],
                index=0
            )
            mtrans = st.selectbox(
                'Transportasi harian?',
                ['Pilih...', 'Mobil pribadi', 'Sepeda', 'Sepeda motor', 'Transportasi umum', 'Berjalan kaki'],
                index=0
            )

        submit_button = st.form_submit_button(label='Generate Hasil')

    # Handle submit
    if submit_button:
        # Validasi input
        if is_valid_input(
            gender, family_history_with_overweight, favc, caec, smoke, scc, calc,
            mtrans, fcvc, ncp, faf, tue, ch2o
        ) and all([
            age.strip() != '', height.strip() != '', weight.strip() != ''
        ]):
            # Lakukan preprocessing
            input_data = preprocess_input(
                label_encoders,
                gender, age, height, weight,
                family_history_with_overweight, favc, fcvc, ncp,
                caec, smoke, ch2o, scc, faf, tue,
                calc, mtrans
            )

            if input_data is not None:
                try:
                    # Prediksi
                    prediction = model.predict(input_data)[0]
                    predicted_label = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]

                    st.markdown("<h3 class='result'>Hasil Prediksi:</h3>", unsafe_allow_html=True)
                    st.markdown(
                        f"<p class='result'>Tingkat Obesitas Anda: <b>{predicted_label}</b></p>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
        else:
            st.error("Lengkapi seluruh pilihan dan masukkan angka/teks dengan benar sebelum prediksi.")


# ======================== 9. HALAMAN: INFORMASI KELOMPOK 10 ========================
elif page == "Informasi Kelompok 10":
    st.title("Tentang Anggota Kelompok 10")

    st.markdown(
        """
        <style>
        .round-image {
            border-radius: 50%;
            width: 150px;
            height: 150px;
            object-fit: cover;
        }
        .image-container {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        .image-container div {
            text-align: center;
        }
        .image-container img {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="image-container">
            <div>
                <img src="https://via.placeholder.com/150" class="round-image" />
                <p>Anggota 1</p>
                <p>Putri Manika Rukmamaya(091)</p>
            </div>
            <div>
                <img src="https://via.placeholder.com/150" class="round-image" />
                <p>Anggota 2</p>
                <p>Maysahayu Artika Maharani(214)</p>
            </div>
            <div>
                <img src="https://via.placeholder.com/150" class="round-image" />
                <p>Anggota 3</p>
                <p>Gesang Nur Zamroji(145)</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

