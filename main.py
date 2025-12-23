import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Uygulama Başlığı ---
st.title("🛡️ DEL-CUT Kalite Kontrol")
st.write("Önce RED (Hatalı) referans fotoğrafını, sonra kontrol edilecek parçayı yükleyin.")

# --- Kenar Çubuğu: Referans Görsel Yükleme ---
st.sidebar.header("1. ADIM: Referans Yükle")
uploaded_ref = st.sidebar.file_uploader("RED kabul edilen görseli seç", type=["jpg", "png", "jpeg"])

def calculate_texture_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
    return laplacian_var

def compare_histograms(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# --- Ana Akış ---
if uploaded_ref is not None:
    file_bytes_ref = np.asarray(bytearray(uploaded_ref.read()), dtype=np.uint8)
    ref_img = cv2.imdecode(file_bytes_ref, 1)
    
    # Görüntüyü RGB'ye çevirip göster (OpenCV BGR okur)
    st.sidebar.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption="Referans (RED)", use_container_width=True)
    
    ref_score = calculate_texture_score(ref_img)
    st.sidebar.markdown(f"**Referans Puanı:** `{ref_score:.1f}`")

    # --- Test Görseli Yükleme ---
    st.markdown("---")
    st.header("2. ADIM: Parça Kontrolü")
    uploaded_test = st.file_uploader("Üretilen parçanın fotoğrafını çek/yükle", type=["jpg", "png", "jpeg"])
    
    if uploaded_test is not None:
        file_bytes_test = np.asarray(bytearray(uploaded_test.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes_test, 1)
        
        st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), caption="Test Edilen Parça", use_container_width=True)
        
        test_score = calculate_texture_score(test_img)
        similarity = compare_histograms(ref_img, test_img)
        
        st.markdown("### 📊 Sonuçlar")
        st.write(f"Test Parçası Puanı: **{test_score:.1f}**")
        st.write(f"Benzerlik Oranı: **%{similarity*100:.1f}**")
        
        # Karar Eşiği
        threshold = ref_score * 0.85 
        
        if test_score >= threshold and similarity > 0.45:
            st.error("🚨 DİKKAT: RED OLABİLİR")
            st.write("Bu parça, yüklediğiniz hatalı referansa çok benziyor.")
        else:
            st.success("✅ FARKLI GÖRÜNÜYOR")
            st.write("Referans alınan hataya rastlanmadı.")
            
else:
    st.info("👈 Lütfen önce sol menüden (veya mobilde üstteki oktan) REFERANS görseli yükleyin.")
