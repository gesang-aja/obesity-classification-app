app.py
from streamlit_option_menu import option_menu
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


icon_image = Image.open('image/chubby.png')
st.set_page_config(
    page_title="Obesity Buddy",
    page_icon=icon_image,
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
        gender in ['Perempuan', 'Laki-laki'],
        calc in ['Selalu', 'Sering', 'Kadang-kadang', 'Tidak'],
        caec in ['Selalu', 'Sering', 'Kadang-kadang', 'Tidak'],
        mtrans in ['Transportasi Umum', 'Sepeda', 'Sepeda Motor', 'Mobil Pribadi', 'Jalan Kaki'],
        fcvc in ['Tidak Pernah', 'Kadang-kadang', 'Selalu'],
        ncp in ['Antara 1 atau 2', 'Tiga', 'Lebih dari 3'],
        ch2o in ['Kurang dari satu liter', 'Antara 1 dan 2 L', 'Lebih dari 2 L'],
        faf in ['Tidak Pernah', '1 atau 2 hari', '2 atau 4 hari', '4 atau 5 hari'],
        tue in ['0—2 jam', '3—5 jam', 'Lebih dari 5 jam'],
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
    fcvc_mapping = {"Tidak Pernah": 1, "Kadang-kadang": 2, "Selalu": 3}
    ncp_mapping = {'Antara 1 atau 2': 1, 'Tiga': 2, 'Lebih dari 3': 3}
    ch2o_mapping = {'Kurang dari satu liter': 1, 'Antara 1 dan 2 L': 2, 'Lebih dari 2 L': 3}
    faf_mapping = {'Tidak Pernah': 0, '1 atau 2 hari': 1, '2 atau 4 hari': 2, '4 atau 5 hari': 3}
    tue_mapping = {'0—2 jam': 0, '3—5 jam': 1, 'Lebih dari 5 jam': 2}

    gender_mapping = {'Perempuan': 'Female', 'Laki-laki': 'Male'}
    family_history_mapping = {'Ya': 'yes', 'Tidak': 'no'}
    favc_mapping = {'Ya': 'yes', 'Tidak': 'no'}
    caec_mapping = {'Selalu': 'Always', 'Sering': 'Frequently', 'Kadang-kadang': 'Sometimes', 'Tidak': 'no'}
    smoke_mapping = {'Ya': 'yes', 'Tidak': 'no'}
    scc_mapping = {'Ya': 'yes', 'Tidak': 'no'}
    calc_mapping = {'Selalu': 'Always', 'Sering': 'Frequently', 'Kadang-kadang': 'Sometimes', 'Tidak': 'no'}
    mtrans_mapping = {'Transportasi Umum': 'Public_Transportation', 'Jalan Kaki': 'Walking', 'Mobil Pribadi': 'Automobile', 'Sepeda Motor': 'Motorbike', 'Sepeda': 'Bike'}

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
    load_css('static/css/styles.css') # Load CSS
except:
    pass

# Muat model dan encoders
model = load_model()
label_encoders = load_label_encoders()

# ======================== NAVIGASI ========================
with st.sidebar:
    col1, col2 = st.columns([1, 4])  # Atur proporsi kolom (1:4)

    # Tambahkan logo di kolom pertama
    with col1:
        st.image(icon_image, width=50)  # Sesuaikan ukuran logo

    # Tambahkan nama aplikasi di kolom kedua
    with col2:
        st.markdown(
            """
            <style>
            @keyframes glowing {
                0% { text-shadow: 0 0 5px rgba(247, 74, 6, 0.5), 0 0 10px rgba(247, 74, 6, 0.5); }
                100% { text-shadow: 0 0 20px rgba(247, 74, 6, 1), 0 0 30px rgba(247, 74, 6, 1); }
            }
            .animated-text {
                animation: glowing 1.5s infinite;
                color: #f74a06;
                text-align: left;  
                font-size: 36px; 
            }
            </style>
            <h2 class="animated-text">Obesity Buddy</h2>
            """,
            unsafe_allow_html=True
        )

    # Tambahkan opsi menu
    page = option_menu(
        "   ",  # Judul menu
        ["Meet Your Buddy", "Buddy Scan", "Buddy Insights", "Visualisasi Dataset", "Informasi Kelompok 10"],  # Opsi menu
        icons=["info-circle", "clipboard-data", "bar-chart-line", "bar-chart", "people"],  # Ikon opsi menu
        menu_icon="list",  # Ikon menu utama (hamburger menu)
        default_index=0,  # Indeks default yang dipilih
        orientation="vertical",  # Sidebar vertikal
    )

# ======================== HALAMAN: MEET YOUR BUDDY ========================
if page == "Meet Your Buddy":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f7a06a, #f74a06); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h1 style="color: white; text-align: center;">Meet Your Buddy!</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #ffffff, #f0f0f0); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h3 style="background: -webkit-linear-gradient(#f7a06a, #f74a06); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; text-align: center;">Obesity Buddy: Aplikasi Prediksi Obesitas</h3>
        </div>
        """,
        unsafe_allow_html=True
    ) ##f74a06

    st.markdown('<hr style="border: 2px solid #f74a06;">', unsafe_allow_html=True)

    st.write(
        """
        **Obesitas** bukan hanya soal penampilan, tetapi juga masalah kesehatan serius yang bisa mempengaruhi kualitas hidup. Menurut **World Health Organization (WHO)**, seseorang dianggap obesitas jika **Indeks Massa Tubuh (BMI)** mereka lebih dari 30, yang meningkatkan risiko terkena penyakit jantung, diabetes tipe 2, hipertensi, dan bahkan gangguan mental. Di Indonesia, prevalensi obesitas telah meningkat dari **11,7% (2010)** menjadi **15,4% (2013)**, menunjukkan bahwa ini adalah masalah yang semakin berkembang.

        Faktor penyebab obesitas sangat beragam, mulai dari genetik, pola makan tidak sehat, kurangnya aktivitas fisik, hingga kebiasaan tidur buruk. Namun, Anda tidak perlu khawatir! Deteksi dini adalah langkah pertama untuk mencegahnya, dan perubahan gaya hidup dapat membantu Anda mengurangi risikonya secara signifikan.

        Dengan menggunakan teknologi **AI** canggih dan algoritma **Random Forest**, **Obesity Buddy** siap menjadi teman setia Anda dalam perjalanan menuju tubuh yang lebih sehat. Aplikasi ini memanfaatkan data dari kuisioner kesehatan, pola makan, dan aktivitas fisik untuk memprediksi tingkat obesitas Anda dengan akurasi tinggi dan memberikan rekomendasi kesehatan yang personal.

        **Random Forest** menggabungkan banyak decision tree untuk menghasilkan prediksi yang lebih akurat, sehingga Anda bisa mendapatkan panduan yang lebih efektif dalam mengelola berat badan dan mencapai gaya hidup yang lebih sehat.
        """
    )

    st.markdown('<hr style="border: 2px solid #f74a06;">', unsafe_allow_html=True)


# ======================== HALAMAN: BUDDY SCAN ========================
elif page == "Buddy Scan":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f7a06a, #f74a06); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h1 style="color: white; text-align: center;">Buddy Scan</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #ffffff, #f0f0f0); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <div class='description' style='font-size: 16px; color: black;'>
                Silakan isi pertanyaan di bawah ini. Lalu, klik tombol "Lihat Hasil" untuk mendapatkan prediksi tingkat obesitas Anda.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form(key='kuisioner_form'):
        kolom_kiri, kolom_kanan = st.columns(2)

        with kolom_kiri:
            # 1. Jenis Kelamin
            gender = st.selectbox(
                'Jenis Kelamin',
                ['Pilih', 'Perempuan', 'Laki-laki'],
                index=0,
                key='input_gender'
            )

            # 2. Age
            age = st.text_input(
                'Umur (tahun)',
                value='',
                key='input_age',
                placeholder='Masukkan umur Anda'
            )

            # 3. Height
            height = st.text_input(
                'Tinggi Badan (cm)',
                value='',
                key='input_height',
                placeholder='Masukkan tinggi badan Anda dalam cm'
            )

            # 4. Weight
            weight = st.text_input(
                'Berat Badan (kg)',
                value='',
                key='input_weight',
                placeholder='Masukkan berat badan Anda dalam kg'
            )

            # 5. Family History with Overweight
            family_history_with_overweight = st.radio(
                'Apakah Anda memiliki riwayat keluarga dengan obesitas?',
                ['Ya', 'Tidak'],
                key='input_family_history_with_overweight'
            )

            # 6. FAVC (Frequent Consumption of High Caloric Food)
            favc = st.radio(
                'Apakah Anda sering mengonsumsi makanan tinggi kalori?',
                ['Ya', 'Tidak'],
                key='input_favc'
            )

            # 7. FCVC (Frequent Consumption of Vegetables)
            fcvc = st.selectbox(
                'Seberapa sering Anda makan sayur?',
                ['Pilih', 'Tidak Pernah', 'Kadang-kadang', 'Selalu'],
                index=0,
                key='input_fcvc'
            )

            # 8. NCP (Number of Main Meals per Day)
            ncp = st.selectbox(
                'Berapa kali Anda makan setiap hari?',
                ['Pilih', 'Antara 1 atau 2', 'Tiga', 'Lebih dari 3'],
                index=0,
                key='input_ncp'
            )

        with kolom_kanan:
            # 9. CAEC (Consumption of Any Food Between Meals)
            caec = st.selectbox(
                'Seberpa sering Anda "nyemil"?',
                ['Pilih', 'Selalu', 'Sering', 'Kadang-kadang', 'Tidak'],
                index=0,
                key='input_caec'
            )

            # 10. SMOKE (Smoking Habit)
            smoke = st.radio(
                'Apakah Anda merokok?',
                ['Ya', 'Tidak'],
                key='input_smoke'
            )

            # 11. CH2O (Daily Water Consumption in Liters)
            ch2o = st.selectbox(
                'Berapa banyak air yang Anda konsumsi setiap hari? (liter)',
                ['Pilih', 'Kurang dari satu liter', 'Antara 1 dan 2 L', 'Lebih dari 2 L'],
                index=0,
                key='input_ch2o'
            )

            # 12. SCC (Calorie Monitoring)
            scc = st.radio(
                'Apakah Anda memantau kalori harian?',
                ['Ya', 'Tidak'],
                key='input_scc'
            )

            # 13. FAF (Frequency of Physical Activity)
            faf = st.selectbox(
                'Seberapa sering Anda melakukan aktivitas fisik?',
                ['Pilih', 'Tidak Pernah', '1 atau 2 hari', '2 atau 4 hari', '4 atau 5 hari'],
                index=0,
                key='input_faf'
            )

            # 14. TUE (Time Using Technology Devices per Day)
            tue = st.selectbox(
                'Berapa jam Anda menggunakan perangkat teknologi (telepon seluler, video game, televisi, komputer, dll.) per hari?',
                ['Pilih', '0—2 jam', '3—5 jam', 'Lebih dari 5 jam'],
                index=0,
                key='input_tue'
            )

            # 15. CALC (Alcohol Consumption Frequency)
            calc = st.selectbox(
                'Seberapa sering Anda minum alkohol?',
                ['Pilih', 'Selalu', 'Sering', 'Kadang-kadang', 'Tidak'],
                index=0,
                key='input_calc'
            )

            # 16. MTRANS (Mode of Transportation)
            mtrans = st.selectbox(
                'Jenis transportasi apa Anda gunakan sehari-hari?',
                ['Pilih', 'Transportasi Umum', 'Sepeda', 'Sepeda Motor', 'Mobil Pribadi', 'Jalan Kaki'],
                index=0,
                key='input_mtrans'
            )

        submit_button = st.form_submit_button(label='Lihat Hasil')

    # Handle submit
    if submit_button:
        # Validasi input
        if is_valid_input(
            gender, family_history_with_overweight, favc, caec, smoke, scc, calc,
            mtrans, fcvc, ncp, faf, tue, ch2o
        ) and all([
            age.strip() != '', height.strip() != '', weight.strip() != ''
        ]):
            try:
                # Konversi input tinggi dan berat badan ke angka
                height_cm = float(height)
                weight_kg = float(weight)

                # Hitung BMI
                height_m = height_cm / 100
                bmi = weight_kg / (height_m ** 2)

                # Rentang berat badan normal (BMI: 18.5 - 24.9)
                min_normal_weight = 18.5 * (height_m ** 2)
                max_normal_weight = 24.9 * (height_m ** 2)

                # Kategorisasi BMI
                if bmi < 18.5:
                    bmi_category = "Kekurangan berat badan"
                    recommendation = (
                        "Anda disarankan untuk meningkatkan berat badan dengan mengonsumsi makanan bergizi lebih banyak "
                        "dan berkonsultasi dengan ahli gizi."
                    )
                elif 18.5 <= bmi <= 24.9:
                    bmi_category = "Normal"
                    recommendation = (
                        "Pertahankan gaya hidup sehat Anda untuk menjaga berat badan ideal."
                    )
                elif 25 <= bmi <= 29.9:
                    bmi_category = "Kelebihan berat badan"
                    recommendation = (
                        f"Pertimbangkan untuk menurunkan berat badan melalui olahraga rutin dan pola makan sehat. "
                        f"Target berat badan Anda: {min_normal_weight:.1f} kg - {max_normal_weight:.1f} kg."
                    )
                else:
                    bmi_category = "Obesitas"
                    recommendation = (
                        f"Anda disarankan untuk menurunkan berat badan secara bertahap. "
                        f"Target berat badan Anda: {min_normal_weight:.1f} kg - {max_normal_weight:.1f} kg. "
                        f"Segera konsultasikan dengan dokter atau ahli gizi untuk program yang tepat."
                    )

                # Lakukan preprocessing
                input_data = preprocess_input(
                    label_encoders,
                    gender, age, height, weight,
                    family_history_with_overweight, favc, fcvc, ncp,
                    caec, smoke, ch2o, scc, faf, tue,
                    calc, mtrans
                )

                if input_data is not None and model is not None:
                    # Prediksi
                    prediction = model.predict(input_data)[0]
                    predicted_label = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]

                    # Tampilkan hasil prediksi dan BMI dalam card
                    st.markdown(
                        """
                        <div style="background: linear-gradient(135deg, #f7a06a, #f74a06); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                            <h3 style="color: black;">🔎 Hasil Prediksi 🔍</h3>
                            <p style="color: white;">📈 Tingkat Obesitas Anda: <b style="color: yellow;">{predicted_label}</b></p>
                            <p style="color: white;">⚖️ BMI Anda: <b style="color: yellow;">{bmi:.2f} ({bmi_category})</b></p>
                            <p style="color: white;">📏 Berat badan normal untuk tinggi Anda: <b style="color: yellow;">{min_normal_weight:.1f} kg</b> - <b style="color: yellow;">{max_normal_weight:.1f} kg</b></p>
                            <p style="color: white;">💡 <b>Rekomendasi:</b> <span style="color: yellow;">{recommendation}</span></p>
                        </div>
                        """.format(predicted_label=predicted_label, bmi=bmi, bmi_category=bmi_category, min_normal_weight=min_normal_weight, max_normal_weight=max_normal_weight, recommendation=recommendation),
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
        else:
            st.error("Lengkapi seluruh pilihan dan masukkan angka/teks dengan benar sebelum prediksi.")



# ======================== HALAMAN: BUDDY INSIGHTS ========================
elif page == "Buddy Insights":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f7a06a, #f74a06); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h1 style="color: white; text-align: center;">Buddy Insights</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
 
    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)
    
    insight_data = {
        "Apa itu Obesitas dan BMI❓": {
            "Penjelasan": (
                "**Obesitas** adalah kondisi medis yang ditandai dengan penumpukan lemak tubuh berlebihan.\n"
                "**Indeks Massa Tubuh (BMI)** adalah ukuran yang digunakan untuk menilai apakah berat badan seseorang berada dalam kisaran yang sehat."
            ),
            "Kategori BMI Menurut WHO": (
                "- **Kekurangan Berat Badan**: BMI kurang dari 18.5\n"
                "- **Normal**: BMI antara 18.5 dan 24.9\n"
                "- **Kelebihan Berat Badan**: BMI antara 25 dan 29.9\n"
                "- **Obesitas**: BMI 30 atau lebih"
            )
        },
        "Jenis-Jenis Obesitas": {
            "Kategori Obesitas": (
                "- **Insufficient Weight**: BMI kurang dari 18.5\n"
                "- **Normal Weight**: BMI antara 18.5 dan 24.9\n"
                "- **Overweight Level I**: BMI antara 25 dan 29.9\n"
                "- **Overweight Level II**: BMI antara 30 dan 34.9\n"
                "- **Obesity Type I**: BMI antara 35 dan 39.9\n"
                "- **Obesity Type II**: BMI antara 40 dan 49.9\n"
                "- **Obesity Type III**: BMI 50 atau lebih"
            )
        },
        "Tips Menjaga Berat Badan Sehat": {
            "Tips": (
                "- **Konsumsi Makanan Bergizi**: Perbanyak sayuran, buah-buahan, biji-bijian, dan protein tanpa lemak.\n"
                "- **Olahraga Rutin**: Lakukan aktivitas fisik minimal 150 menit per minggu.\n"
                "- **Atur Pola Makan**: Hindari makanan tinggi gula dan lemak jenuh.\n"
                "- **Minum Air Cukup**: Pastikan Anda minum setidaknya 2 liter air per hari.\n"
                "- **Kontrol Stres**: Stres dapat mempengaruhi pola makan dan kesehatan secara keseluruhan.\n"
                "- **Tidur Cukup**: Tidur yang cukup membantu metabolisme dan kesehatan mental."
            )
        }
    }

    for title, content in insight_data.items():
        with st.expander(f"🌟 {title}"):
            for key, value in content.items():
                st.markdown(f"**{key}**: {value}")



# ======================== HALAMAN: VISUALISASI DATASET ========================
elif page == "Visualisasi Dataset":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f7a06a, #f74a06); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h1 style="color: white; text-align: center;">📊 Visualisasi Dataset 📊</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Muat dataset dari file ObesityDataSet_raw_and_data_sinthetic.csv
    dataset_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
    dataset = pd.read_csv(dataset_path)
    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)

    st.write("Berikut adalah beberapa baris dari dataset:")
    st.write(dataset.head())
    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)

    st.write("Deskripsi statistik dari dataset:")
    st.write(dataset.describe())
    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)

    # Periksa nama kolom dalam dataset
    st.write("Nama kolom dalam dataset:")
    st.write(dataset.columns)
    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)

    # Visualisasi distribusi umur
    st.write("Visualisasi distribusi umur:")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset, x='Age', kde=True, color='orange')
    plt.xlabel('Umur')
    plt.ylabel('Frekuensi')
    plt.title('Distribusi Umur')
    st.pyplot(plt)
    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)

    # # Visualisasi distribusi BMI
    # if 'BMI' in dataset.columns:
    #     st.write("Visualisasi distribusi BMI:")
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(dataset['BMI'], bins=30, color='orange', edgecolor='black')
    #     plt.xlabel('BMI')
    #     plt.ylabel('Frequency')
    #     plt.title('Distribusi BMI')
    #     st.pyplot(plt)
    # else:
    #     st.error("Kolom 'BMI' tidak ditemukan dalam dataset.")

    # Konversi kolom kategorikal menjadi numerik
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    dataset_encoded = dataset.copy()
    for column in categorical_columns:
        dataset_encoded[column] = dataset_encoded[column].astype('category').cat.codes

    # Visualisasi korelasi fitur
    st.write("Visualisasi korelasi fitur:")
    plt.figure(figsize=(12, 8))
    correlation_matrix = dataset_encoded.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriks Korelasi Fitur')
    st.pyplot(plt)
    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)

    
# ======================== HALAMAN: INFORMASI KELOMPOK 10 ========================
elif page == "Informasi Kelompok 10":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f7a06a, #f74a06); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h1 style="color: white; text-align: center;">Anggota Kelompok 1️⃣0️⃣ </h1>
        </div> 
        """,
        unsafe_allow_html=True
    )

    st.markdown('<hr style="border: 2px solid #f74a06; margin-top: 40px;">', unsafe_allow_html=True)

    # Menggunakan Streamlit untuk menampilkan gambar dan teks dengan rata tengah
    col1, col2, col3 = st.columns(3)

    with col1:
            image_putri = Image.open('image/manika.png')  # Pastikan nama file sesuai dan berada di folder 'image'
            st.image(image_putri, caption='', use_container_width=True)
            st.markdown(
                "<p style='text-align: center;'><strong>Putri Manika Rukmamaya</strong><br>23031554091</p>",
                unsafe_allow_html=True
            )
        
    with col2:
            image_maysahayu = Image.open('image/rani.png')  # Pastikan nama file sesuai dan berada di folder 'image'
            st.image(image_maysahayu, caption='', use_container_width=True)
            st.markdown(
                "<p style='text-align: center;'><strong>Maysahayu Artika Maharani</strong><br>23031554214</p>",
                unsafe_allow_html=True
            )

    with col3:
            image_gesang = Image.open('image/gesang.png')  # Pastikan nama file sesuai dan berada di folder 'image'
            st.image(image_gesang, caption='', use_container_width=True)
            st.markdown(
                "<p style='text-align: center;'><strong>Gesang Nur Zamroji</strong><br>23031554145</p>",
                unsafe_allow_html=True
            )
        

