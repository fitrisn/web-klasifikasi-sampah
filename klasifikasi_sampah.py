import streamlit as st
from PIL import Image
import numpy as np
from gtts import gTTS
import tensorflow as tf
import os
import gdown

# ==================== AMBIL MODEL DARI GOOGLE DRIVE ====================
@st.cache_resource
def load_model_from_drive():
    url = "https://drive.google.com/uc?id=1tXa-NIqw_ZZ05ZXOC9cMzNuAa7fCCBfp"
    output = "model_organik_anorganik.h5"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)

model = load_model_from_drive()
class_names = ['Anorganik', 'Organik']

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(page_title="SiPilah - Edukasi Sampah", layout="centered")
st.image("header.png", use_container_width=True)

# ==================== SIDEBAR ====================
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content { background-color: #E6F7FF; }
    </style>
""", unsafe_allow_html=True)
st.sidebar.title("🌈 SiPilah")
page = st.sidebar.radio("Ayo Pilih Halamannya:", [
    "🏡 Beranda", 
    "🏠 Tentang Sampah", 
    "📸 Klasifikasi Sampah"])

# ==================== HALAMAN BERANDA ====================
if page == "🏡 Beranda":
    st.markdown("""
        <div style='background-color: #FFF3CD; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: #FF8C00;'>👋 Halo Teman Kecil!</h1>
            <h4>Selamat datang di <b>SiPilah</b>! Tempat seru untuk belajar memilah sampah 🌍</h4>
        </div>
    """, unsafe_allow_html=True)
    st.success("✨ Yuk mulai petualangan menjaga bumi di sini!")
    st.balloons()

# ==================== HALAMAN EDUKASI ====================
elif page == "🏠 Tentang Sampah":
    st.title("📚 Yuk Kenali Sampah!")
    st.markdown("""
    Mari kita belajar bersama tentang sampah dan bagaimana cara menjaga bumi tetap bersih dan sehat! 🌍✨

    ### 1️⃣ Apa Itu Sampah?
    Sampah adalah benda yang kita buang karena sudah tidak digunakan lagi. Tapi tahukah kamu? Walaupun dibuang, sampah tetap harus diperhatikan supaya tidak mencemari lingkungan.

    🧃 Contoh benda yang jadi sampah:
    - Bungkus jajanan 🍭
    - Kulit buah seperti pisang 🍌
    - Botol plastik bekas minuman 🧴
    - Kertas bekas menggambar atau menulis 📄

    👉 Sampah bisa ditemukan di rumah, sekolah, taman, bahkan di jalan. Makanya penting banget kita tahu cara mengelolanya!

    ### 2️⃣ Jenis-Jenis Sampah
    Sampah itu ada dua jenis utama. Yuk kenali satu per satu!

    #### 🟢 Sampah Organik
    Sampah yang berasal dari alam dan bisa membusuk dengan cepat. Biasanya berasal dari tumbuhan atau makanan.

    🍃 Contoh sampah organik:
    - Sisa makanan seperti nasi dan sayur 🍚
    - Daun-daun kering yang jatuh dari pohon 🍂
    - Kulit buah seperti apel dan pisang 🍌
    - Cangkang telur 🥚
    """)
    st.image("organik.png", caption="Contoh Sampah Organik 🍃", use_container_width=True)
    st.markdown("""
    💡 Sampah organik bisa diubah menjadi kompos! Kompos adalah pupuk alami yang bisa membantu tanaman tumbuh subur. Jadi, sampah organik bisa sangat bermanfaat!

    #### 🔵 Sampah Anorganik
    Sampah yang tidak mudah membusuk dan biasanya dibuat oleh manusia. Sampah ini bisa bertahan lama di bumi kalau tidak dikelola dengan baik.

    🧴 Contoh sampah anorganik:
    - Botol plastik bekas minuman 🧴
    - Kaleng minuman atau makanan 🥫
    - Kertas bekas 📄
    - Bungkus snack atau permen 🍫
    """)
    st.image("anorganik.png", caption="Contoh Sampah Anorganik 🧴", use_container_width=True)
    st.markdown("""
    💡 Sampah anorganik bisa didaur ulang! Artinya, bisa diubah menjadi barang baru seperti tas, mainan, atau bahkan pakaian!

    ### 3️⃣ Kenapa Sampah Harus Dipilah?
    Memilah sampah artinya kita memisahkan sampah sesuai jenisnya. Ini penting banget, lho!

    🌟 Manfaat memilah sampah:
    - 🌍 Bumi jadi lebih bersih dan sehat
    - 🐰 Hewan tidak makan sampah yang berbahaya
    - ♻️ Sampah bisa dimanfaatkan kembali
    - 🧠 Kita jadi lebih peduli dan pintar menjaga lingkungan

    ### 4️⃣ Bagaimana Cara Memilah Sampah?
    Yuk, belajar cara memilah sampah dengan mudah!

    👀 Langkah pertama:
    Lihat dulu, ini sampah makanan atau plastik?

    🗂️ Langkah kedua:
    Pisahkan ke tempat yang benar:
    - Sampah Organik 🍃 → Masukkan ke tempat kompos
    - Sampah Anorganik 🧴 → Masukkan ke tempat daur ulang

    👨‍👩‍👧‍👦 Langkah ketiga:
    Ajak teman, kakak, adik, dan orang tua untuk ikut memilah sampah bersama. Bisa jadi kegiatan seru di rumah atau sekolah!

    ### 🎉 Ayo Jadi Pahlawan Lingkungan!
    Kamu bisa jadi pahlawan yang menjaga bumi tetap bersih! Mulai dari hal kecil seperti membuang sampah di tempatnya dan memilahnya dengan benar. 🌱

    ### 🎖 Misi Anak Hebat:
    - Selalu buang sampah di tempatnya ✅
    - Pisahkan organik dan anorganik ✅
    - Ajak teman belajar bareng ✅

    🦸‍♀️🦸‍♂️ Yuk, jadi Superhero Sampah! Karena bumi butuh bantuan kita semua.
    """)

    st.success("🌟 Hebat! Kamu sudah belajar tentang sampah 🎉")


# ==================== HALAMAN KLASIFIKASI ====================
elif page == "📸 Klasifikasi Sampah":
    st.title("🔍 Yuk Tebak Jenis Sampahmu!")

    opsi_input = st.radio("Pilih cara input gambar:", ("📁 Upload Gambar", "📷 Ambil dari Kamera"))

    def klasifikasikan_gambar(image):
        image = image.convert("RGB")
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        label = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return label, confidence

    def play_audio_tts(text):
        tts = gTTS(text, lang='id')
        tts.save("output.mp3")
        with open("output.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
        os.remove("output.mp3")

    def tampilkan_penjelasan(label, confidence):
        st.subheader(f"✅ Ini adalah sampah **{label}**")
        st.write(f"Keyakinan model: {confidence:.2f}%")
        if label == "Organik":
            text = "Sampah organik bisa membusuk. Contohnya daun, sisa makanan, kulit buah. Bisa dijadikan kompos."
            st.info("""
            🍃 Sampah **organik** bisa membusuk.
            Contoh: daun, sisa makanan, kulit buah.
            Bisa dijadikan kompos!
            """)
        else:
            text = "Sampah anorganik sulit membusuk. Contohnya plastik, botol, kaleng, kertas. Bisa didaur ulang."
            st.info("""
            🧴 Sampah **anorganik** sulit membusuk.
            Contoh: plastik, botol, kaleng, kertas.
            Bisa didaur ulang jadi barang baru!
            """)

        play_audio_tts(text)
        st.success("👏 Keren! Kamu pintar memilah sampah!")

    if opsi_input == "📁 Upload Gambar":
        uploaded_file = st.file_uploader("Unggah gambar sampah", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang kamu unggah', use_container_width=True)
            label, confidence = klasifikasikan_gambar(image)
            tampilkan_penjelasan(label, confidence)

    elif opsi_input == "📷 Ambil dari Kamera":
        picture = st.camera_input("Ambil gambar dari kamera")
        if picture:
            image = Image.open(picture)
            st.image(image, caption='Foto dari kamera', use_container_width=True)
            label, confidence = klasifikasikan_gambar(image)
            tampilkan_penjelasan(label, confidence)