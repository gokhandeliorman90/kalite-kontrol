import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Ayarlar ---
st.set_page_config(page_title="DEL-CUT AI", page_icon="🛡️")

st.title("🛡️ DEL-CUT Gelişmiş Kalite Kontrol")
st.markdown("""
Bu sistem **Çoklu Referans** mantığıyla çalışır. 
Makine öğrenmesi hassasiyetini artırmak için **mümkün olduğunca çok sayıda** ve **farklı** hatalı (RED) parça fotoğrafı yükleyin.
""")

# --- 1. BÖLÜM: EĞİTİM VERİSİ (RED ÖRNEKLERİ) ---
st.sidebar.header("📂 1. ADIM: Hata Tanıtımı")
st.sidebar.info("Sisteme ne kadar çok 'Hatalı' örnek gösterirseniz o kadar akıllı olur.")

# Burada accept_multiple_files=True diyerek çoklu seçimi açıyoruz
uploaded_refs = st.sidebar.file_uploader(
    "RED (Hatalı) örneklerin hepsini seçip yükleyin", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

def calculate_features(image):
    """Görselden parmak izi (Doku ve Renk özellikleri) çıkarır"""
    # 1. Doku Analizi (Laplacian)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    texture_score = cv2.Laplacian(blur, cv2.CV_64F).var()
    
    # 2. Renk/Histogram Analizi (HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return texture_score, hist

# Referans Havuzu Oluştur
reference_data = []

if uploaded_refs:
    st.sidebar.success(f"✅ {len(uploaded_refs)} adet RED örneği işlendi.")
    
    for uploaded_file in uploaded_refs:
        # Dosyayı oku
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Özellikleri çıkar ve havuza at
        score, hist = calculate_features(img)
        reference_data.append({
            "score": score,
            "hist": hist,
            "name": uploaded_file.name
        })
else:
    st.sidebar.warning("⚠️ Lütfen en az 1 adet referans görsel yükleyin.")

# --- 2. BÖLÜM: TEST (KALİTE KONTROL) ---
st.markdown("---")
st.header("🔍 2. ADIM: Parça Kontrolü")

uploaded_test = st.file_uploader("Üretilen parçanın fotoğrafını yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_test is not None:
    if not reference_data:
        st.error("Lütfen önce sol menüden RED örneklerini yükleyin!")
    else:
        # Test resmini hazırla
        file_bytes_test = np.asarray(bytearray(uploaded_test.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes_test, 1)
        
        # Test resmini göster
        st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), caption="Test Edilen Numune", width=300)
        
        # Test özelliklerini çıkar
        test_score, test_hist = calculate_features(test_img)
        
        # --- KARAR MOTORU ---
        # Test parçasını, yüklenen TÜM referanslarla tek tek kıyasla.
        # Eğer herhangi bir RED örneğine çok benziyorsa, RED ver.
        
        match_found = False
        max_similarity = 0.0
        matched_ref_name = ""
        
        st.write("---")
        st.write("📊 **Analiz Raporu:**")
        
        for ref in reference_data:
            # Histogram Benzerliği (0 ile 1 arası, 1=Aynı)
            sim = cv2.compareHist(ref["hist"], test_hist, cv2.HISTCMP_CORREL)
            
            # Eğer benzerlik %50'den fazlaysa ve pürüzlülük (score) yakınsa
            if sim > max_similarity:
                max_similarity = sim
                matched_ref_name = ref["name"]
            
            # KRİTİK EŞİK: %55 Benzerlik ve Benzer Doku
            # (Hassasiyeti artırmak için 0.55 yaptık)
            if sim > 0.55 and test_score >= (ref["score"] * 0.8):
                match_found = True
                break # Bir tane bile eşleşme bulursak yeterli
        
        # --- SONUÇ EKRANI ---
        if match_found:
            st.error("🚨 SONUÇ: RED (Hatalı)")
            st.markdown(f"""
            **Tespit:** Bu parça, veritabanındaki **RED** örneklerle uyuşuyor.
            - En çok benzediği örnek: *{matched_ref_name}*
            - Benzerlik Oranı: **%{max_similarity*100:.1f}**
            """)
        else:
            st.success("✅ SONUÇ: KABUL / Temiz")
            st.markdown(f"""
            **Tespit:** Bu parça yüklediğiniz hatalı örneklere benzemiyor.
            - Hatalı örneklere en yakın benzerlik: **%{max_similarity*100:.1f}** (Güvenli bölgede)
            """)
