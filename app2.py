import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== KAMUS & LEXICON (SESUAI METODOLOGI) ==================
# Dictionary Normalisasi: mengubah kata gaul/slang menjadi kata baku (dictionary-based)
NORMALIZATION_DICT = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "engga": "tidak", "nggak": "tidak",
    "brg": "barang", "sdh": "sudah", "dgn": "dengan", "thx": "terima kasih",
    "tks": "terima kasih", "makasih": "terima kasih", "mksh": "terima kasih", "bgt": "banget",
    "kalo": "kalau", "kl": "kalau", "tp": "tapi", "dr": "dari",
    "bs": "bisa", "sy": "saya", "ak": "aku", "aq": "aku",
    "mantul": "mantap", "mantap": "bagus", "jos": "bagus", 
    "ok": "oke", "oke": "bagus", "good": "bagus", "best": "bagus",
    "jelek": "buruk", "parah": "buruk", "ancur": "buruk", "rusak": "buruk",
    "dtg": "datang", "sampe": "sampai", "nyampe": "sampai", "cepet": "cepat",
    "kirim": "pengiriman", "kurir": "pengiriman", "packing": "kemasan",
    "seller": "penjual", "respon": "tanggapan", "bintang": "rating",
    "gan": "juragan", "sis": "kakak", "kak": "kakak", "om": "paman",
    "admin": "penjual", "olshop": "toko online", "bhn": "bahan",
    "adem": "sejuk", "mlar": "melar", "size": "ukuran", "pas": "sesuai",
    "trmksh": "terima kasih", "thankyou": "terima kasih", "recommended": "rekomendasi",
    "recomended": "rekomendasi", "rekomend": "rekomendasi", "mantab": "mantap",
    "bgus": "bagus", "bgs": "bagus", "krn": "karena", "utk": "untuk",
    "buat": "untuk", "krm": "kirim", "pngrmn": "pengiriman", "cpt": "cepat",
    "cpet": "cepat", "lma": "lama", "telat": "terlambat", "mrh": "murah",
    "mhl": "mahal", "ori": "asli", "kw": "palsu", "fake": "palsu",
    "rsk": "rusak", "ssuai": "sesuai", "krg": "kurang", "byk": "banyak",
    "sgt": "sangat", "bener": "benar", "krenn": "keren", "gokil": "keren",
    "bngt": "banget", "bngd": "banget", "puass": "puas", "puasss": "puas",
    "kcwa": "kecewa", "jlek": "jelek", "jelk": "jelek", "bruk": "buruk"
}

# Lexicon Positif (untuk pelabelan berbasis leksikon)
POSITIVE_WORDS = {
    "bagus", "baik", "cepat", "rapi", "aman", "sesuai", "mantap", "puas", 
    "oke", "keren", "nyaman", "suka", "awet", "murah", "ramah", "lengkap",
    "lucu", "halus", "lembut", "tebal", "asli", "original", "rekomendasi",
    "top", "memuaskan", "berfungsi", "pas", "cocok", "sejuk",
    "adem", "dingin", "menyerap", "keringat", "elegan", "mewah", "premium",
    "rapih", "kilat", "gesit", "responsif", "sopan", "jujur", "amanah",
    "bonus", "hadiah", "terjangkau", "hemat", "diskon", "promo", "bersih",
    "wangi", "harum", "cantik", "ganteng", "kece", "modis", "trendy",
    "terima kasih", "thanks", "love", "senang", "sukses", "berkualitas",
    "recommended", "sempurna", "istimewa", "terbaik", "favorit", "worth",
    "gercep", "excellent", "seneng", "super", "juara", "aman", "terpercaya"
}

# Lexicon Negatif (untuk pelabelan berbasis leksikon)
NEGATIVE_WORDS = {
    "jelek", "buruk", "lambat", "lama", "rusak", "cacat", "pecah", "penyok",
    "kecewa", "salah", "beda", "tipis", "kasar", "kotor", "mahal", "bohong",
    "palsu", "kw", "robek", "bolong", "batal", "retur", "komplain", "parah",
    "nyesel", "kurang", "tidak", "panas", "gerah", "gatal", "sempit", 
    "longgar", "kebesaran", "kekecilan", "luntur", "pudar", "kusam",
    "bau", "ape", "lecek", "kusut", "benang", "jahitan", "lepas", "copot",
    "penipuan", "penipu", "lamban", "lelet", "jutek", "galak", "sombong",
    "ribet", "susah", "baret", "gores", "bekas", "mengecewakan",
    "hancur", "ampun", "zonk", "terrible", "awful", "worst", "bad",
    "mengecewakan", "terlambat", "slow", "najis", "sampah", "php",
    "berbeda", "error", "pecah", "sobek", "busuk", "horrible"
}

# Stopwords: kata umum yang tidak memiliki makna sentimen
STOPWORDS = {
    "yang", "di", "dan", "itu", "ini", "dari", "ke", "untuk", "dengan", "nya",
    "saya", "aku", "kami", "kita", "bisa", "ada", "adalah", "juga", "karena",
    "tapi", "namun", "atau", "jadi", "jika", "kalau", "sudah", "lagi", "akan",
    "pada", "masih", "saja", "yg", "dg", "rt", "dgn", "ny", "d", "k",
    "kalo", "biar", "bikin", "bilang", "gak", "ga", "krn", "nih",
    "sih", "si", "tau", "tdk", "tuh", "utk", "ya", "jd", "jgn", "sdh", 
    "aja", "n", "t", "nyg", "hehe", "pen", "u", "nan", "loh", "rt",
    "oleh", "se", "te", "an", "kan", "dia", "mereka", "ia", "telah",
    "sedang", "pernah", "belum", "bukan", "jangan", "bila", "maka",
    "walaupun", "meskipun", "agar", "supaya", "semoga", "dalam", "kepada",
    "terhadap", "antara", "tentang", "hingga", "sambil", "demi", "sebelum",
    "sesudah", "saat", "ketika", "sewaktu", "begitu", "seperti", "bagai",
    "ibarat", "umpama", "laksana", "seolah", "serupa", "macam", "secara",
    "setiap", "seluruh", "semua", "para", "sang", "si", "sri", "banget",
    "sekali", "sangat", "very", "nih", "deh", "dong", "kok",
    "sih", "lho", "kah", "pun"
}

# ================== FUNGSI PREPROCESSING BERTAHAP (SESUAI METODOLOGI) ==================
def cleaning(text):
    """
    Tahap 1 - Cleaning: menghapus tanda baca, angka, dan karakter non-alfanumerik
    Input: "Barang bagus!!! cepat sampai... thx 123"
    Output: "Barang bagus cepat sampai thx"
    """
    if not isinstance(text, str):
        return ""
    # Menghapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def case_folding(text):
    """
    Tahap 2 - Case Folding: mengubah seluruh teks menjadi huruf kecil
    Input: "Barang BAGUS Cepat Sampai"
    Output: "barang bagus cepat sampai"
    """
    return text.lower()

def normalization(text):
    """
    Tahap 3 - Normalization: mengubah kata gaul/slang menjadi kata baku menggunakan kamus dictionary-based
    Input: "brg bgus bgt, cpt sampe, thx"
    Output: "barang bagus banget cepat sampai terima kasih"
    """
    words = text.split()
    normalized = [NORMALIZATION_DICT.get(word, word) for word in words]
    return ' '.join(normalized)

def stopword_removal(text):
    """
    Tahap 4 - Stopword Removal: menghapus kata umum yang tidak memiliki makna sentimen
    Input: "barang yang bagus banget dan cepat sampai"
    Output: "barang bagus banget cepat sampai"
    """
    words = text.split()
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return ' '.join(filtered)

def preprocess_text_step_by_step(text):
    """
    Pipeline preprocessing BERTAHAP dengan menyimpan hasil setiap tahap
    Return: dictionary berisi hasil setiap tahapan
    """
    original = str(text)
    
    # Tahap 1: Cleaning
    step1_cleaning = cleaning(original)
    
    # Tahap 2: Case Folding
    step2_case_folding = case_folding(step1_cleaning)
    
    # Tahap 3: Normalization
    step3_normalization = normalization(step2_case_folding)
    
    # Tahap 4: Stopword Removal
    step4_stopword_removal = stopword_removal(step3_normalization)
    
    return {
        'original': original,
        'step1_cleaning': step1_cleaning,
        'step2_case_folding': step2_case_folding,
        'step3_normalization': step3_normalization,
        'step4_stopword_removal': step4_stopword_removal,
        'final': step4_stopword_removal
    }

def preprocess_text(text):
    """Pipeline preprocessing lengkap (untuk pemrosesan cepat)"""
    text = cleaning(text)
    text = case_folding(text)
    text = normalization(text)
    text = stopword_removal(text)
    return text

# ================== PELABELAN HYBRID APPROACH (SESUAI METODOLOGI) ==================
def lexicon_sentiment(text):
    """Penentuan label awal menggunakan pendekatan lexicon-based sentiment"""
    words = set(text.split())
    pos_count = len(words.intersection(POSITIVE_WORDS))
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    
    if pos_count > neg_count:
        return 'Positif'
    elif neg_count > pos_count:
        return 'Negatif'
    else:
        return 'Netral'

def hybrid_labeling(text_preprocessed, rating):
    """Validasi silang dengan rating untuk koreksi label (Hybrid Approach)"""
    lexicon_label = lexicon_sentiment(text_preprocessed)
    
    try:
        rating = float(rating)
    except:
        return lexicon_label
    
    # Aturan koreksi berdasarkan rating
    # Jika sentimen leksikon = Positif dan rating ≤ 3 → label dikoreksi menjadi Netral
    if lexicon_label == 'Positif' and rating <= 3:
        return 'Netral'
    # Jika sentimen leksikon = Negatif dan rating ≥ 4 → label dikoreksi menjadi Netral
    elif lexicon_label == 'Negatif' and rating >= 4:
        return 'Netral'
    else:
        return lexicon_label

# ================== FUNGSI DETEKSI KOLOM RATING ==================
def detect_rating_column(df):
    """Deteksi otomatis kolom rating dari DataFrame"""
    # Kandidat nama kolom rating
    rating_candidates = ['rating', 'bintang', 'star', 'rate', 'nilai', 'score']
    
    # Cek apakah ada kolom dengan nama yang sesuai
    for col in df.columns:
        if col.lower() in rating_candidates:
            return col
    
    # Jika tidak ditemukan, cari kolom numerik dengan nilai 1-5
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].min() >= 1 and df[col].max() <= 5:
                return col
    
    return None

# ================== UI UTAMA ==================
st.title("📊 Analisis Sentimen Ulasan Tokopedia")
st.markdown("""
**Metode Penelitian:** Pendekatan Kuantitatif dengan Waterfall yang Dimodifikasi untuk Data Science  
**Tahapan:** Pengumpulan Data → Preprocessing → Pelabelan Hybrid → Ekstraksi Fitur TF-IDF → Modeling SVM → Evaluasi & Visualisasi
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 1. Upload Data")
    uploaded_file = st.file_uploader("Upload file CSV ulasan Tokopedia", type=["csv"])
    
    st.divider()
    st.header("⚙️ 2. Pengaturan Model")
    st.info("**Split Data:** 80% Training, 20% Testing")
    st.caption("Model: SVM dengan kernel Linear")

if uploaded_file:
    # ================== PENGUMPULAN DATA ==================
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset berhasil dimuat: **{len(df)}** baris data")
    
    # Deteksi kolom rating otomatis
    rating_col = detect_rating_column(df)
    
    if rating_col is None:
        st.error("⚠️ Kolom 'rating' tidak ditemukan! Pastikan dataset memiliki kolom rating dengan nilai 1-5")
        st.stop()
    else:
        st.info(f"✅ Kolom rating terdeteksi: **{rating_col}**")
    
    # Pilih kolom teks
    st.subheader("Konfigurasi Kolom Dataset")
    text_col = st.selectbox("Pilih Kolom Ulasan (Teks):", df.columns, index=0)
    
    # ================== PREPROCESSING DATA BERTAHAP ==================
    st.subheader("📝 Tahap Preprocessing Data")
    
    with st.expander("🔍 Lihat Detail Proses Preprocessing (Klik untuk expand)", expanded=False):
        st.markdown("""
        **Tahapan Preprocessing sesuai Metodologi:**
        1. **Cleaning** - Menghapus tanda baca, angka, dan karakter non-alfanumerik
        2. **Case Folding** - Mengubah seluruh teks menjadi huruf kecil
        3. **Normalization** - Mengubah kata gaul/slang menjadi kata baku menggunakan kamus
        4. **Stopword Removal** - Menghapus kata umum yang tidak memiliki makna sentimen
        """)
        
        # Ambil contoh data untuk demonstrasi
        sample_idx = st.number_input("Pilih indeks baris untuk melihat contoh preprocessing:", 
                                      min_value=0, max_value=len(df)-1, value=0)
        sample_text = df[text_col].iloc[sample_idx]
        
        st.write("**Teks Original:**")
        st.code(sample_text, language=None)
        
        # Proses bertahap
        steps = preprocess_text_step_by_step(sample_text)
        
        st.write("**Tahap 1 - Cleaning:**")
        st.code(steps['step1_cleaning'], language=None)
        st.caption("✓ Menghapus tanda baca, angka, dan karakter non-alfanumerik")
        
        st.write("**Tahap 2 - Case Folding:**")
        st.code(steps['step2_case_folding'], language=None)
        st.caption("✓ Mengubah semua huruf menjadi huruf kecil")
        
        st.write("**Tahap 3 - Normalization:**")
        st.code(steps['step3_normalization'], language=None)
        st.caption("✓ Mengubah kata gaul/slang menjadi kata baku")
        
        st.write("**Tahap 4 - Stopword Removal:**")
        st.code(steps['step4_stopword_removal'], language=None)
        st.caption("✓ Menghapus kata umum yang tidak bermakna sentimen")
        
        st.success("✅ Hasil Akhir Preprocessing siap digunakan untuk analisis sentimen")
    
    # Proses preprocessing untuk seluruh dataset
    with st.spinner("⏳ Melakukan preprocessing untuk seluruh dataset..."):
        # Simpan hasil preprocessing bertahap untuk kolom tambahan (opsional)
        df['step1_cleaning'] = df[text_col].astype(str).apply(cleaning)
        df['step2_case_folding'] = df['step1_cleaning'].apply(case_folding)
        df['step3_normalization'] = df['step2_case_folding'].apply(normalization)
        df['step4_stopword_removal'] = df['step3_normalization'].apply(stopword_removal)
        
        # Hasil final preprocessing
        df['text_preprocessed'] = df['step4_stopword_removal']
        
        # ================== PELABELAN DATA (HYBRID APPROACH) ==================
        df['sentiment'] = df.apply(lambda row: hybrid_labeling(row['text_preprocessed'], row[rating_col]), axis=1)
    
    st.success("✅ Preprocessing & Pelabelan Hybrid selesai!")
    
    # ================== TAB ANALISIS ==================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribusi Sentimen", 
        "☁️ WordCloud", 
        "📋 Tabel Interaktif", 
        "🤖 Model SVM & Evaluasi",
        "🔮 Prediksi Baru"
    ])
    
    # --- TAB 1: DISTRIBUSI SENTIMEN ---
    with tab1:
        st.subheader("📊 Grafik Distribusi Sentimen")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("Total Ulasan", len(df))
        col_metric2.metric("Rata-rata Rating", f"{df[rating_col].mean():.2f} ⭐")
        sentiment_dist = df['sentiment'].value_counts()
        col_metric3.metric("Sentimen Dominan", sentiment_dist.index[0])
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Pie Chart
            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            sentiment_counts = df['sentiment'].value_counts()
            ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=colors)
            ax_pie.set_title('Proporsi Sentimen', fontsize=14, fontweight='bold')
            st.pyplot(fig_pie)
        
        with col_viz2:
            # Bar Chart
            fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='sentiment', order=['Positif', 'Negatif', 'Netral'], 
                         palette=colors, ax=ax_bar)
            ax_bar.set_title('Distribusi Sentimen', fontsize=14, fontweight='bold')
            ax_bar.set_xlabel('Sentimen', fontsize=12)
            ax_bar.set_ylabel('Jumlah', fontsize=12)
            for container in ax_bar.containers:
                ax_bar.bar_label(container)
            st.pyplot(fig_bar)
        
        # Sentimen per Rating
        st.subheader("Sentimen Berdasarkan Rating")
        fig_cross, ax_cross = plt.subplots(figsize=(10, 5))
        ct = pd.crosstab(df[rating_col], df['sentiment'])
        ct.plot(kind='bar', ax=ax_cross, color=colors)
        ax_cross.set_title('Distribusi Sentimen per Rating', fontsize=14, fontweight='bold')
        ax_cross.set_xlabel('Rating', fontsize=12)
        ax_cross.set_ylabel('Jumlah', fontsize=12)
        ax_cross.legend(title='Sentimen')
        plt.xticks(rotation=0)
        st.pyplot(fig_cross)
    
    # --- TAB 2: WORDCLOUD ---
    with tab2:
        st.subheader("☁️ WordCloud per Sentimen")
        
        sentiment_filter = st.selectbox("Pilih Sentimen:", ['Positif', 'Negatif', 'Netral'])
        
        filtered_text = ' '.join(df[df['sentiment'] == sentiment_filter]['text_preprocessed'])
        
        if filtered_text.strip():
            colormap = 'RdYlGn' if sentiment_filter == 'Positif' else 'Reds' if sentiment_filter == 'Negatif' else 'Greys'
            wc = WordCloud(width=1200, height=500, background_color='white',
                          colormap=colormap, max_words=100).generate(filtered_text)
            
            fig_wc, ax_wc = plt.subplots(figsize=(14, 7))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            ax_wc.set_title(f'WordCloud - Sentimen {sentiment_filter}', fontsize=16, fontweight='bold')
            st.pyplot(fig_wc)
        else:
            st.warning(f"⚠️ Tidak ada data untuk sentimen {sentiment_filter}")
    
    # --- TAB 3: TABEL INTERAKTIF ---
    with tab3:
        st.subheader("📋 Tabel Interaktif Hasil Analisis")
        
        # Pilihan tampilan
        view_option = st.radio("Pilih Tampilan:", 
                               ["Hasil Akhir (Final)", "Tahapan Preprocessing Lengkap"],
                               horizontal=True)
        
        # Filter sentimen
        sentiment_filter_table = st.multiselect(
            "Filter Sentimen:", 
            df['sentiment'].unique(), 
            default=list(df['sentiment'].unique())
        )
        
        display_df = df[df['sentiment'].isin(sentiment_filter_table)]
        
        if view_option == "Hasil Akhir (Final)":
            # Tampilan sederhana
            display_cols = [text_col, 'text_preprocessed', rating_col, 'sentiment']
            
            # Tambahkan kolom opsional jika ada
            optional_cols = ['nama_produk', 'product_name', 'kategori', 'category', 'jumlah_terjual', 'sold']
            for col in optional_cols:
                if col in df.columns and col not in display_cols:
                    display_cols.insert(0, col)
        else:
            # Tampilan lengkap dengan tahapan preprocessing
            display_cols = [text_col, 'step1_cleaning', 'step2_case_folding', 
                           'step3_normalization', 'step4_stopword_removal', 
                           rating_col, 'sentiment']
        
        st.dataframe(display_df[display_cols], use_container_width=True, height=400)
        
        # Download hasil
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Analisis", csv, "hasil_analisis_sentimen.csv", "text/csv")
    
    # --- TAB 4: MODEL SVM & EVALUASI ---
    with tab4:
        st.subheader("🤖 Pemodelan dengan Support Vector Machine (SVM)")
        st.markdown("""
        **Ekstraksi Fitur:** TF-IDF (Term Frequency–Inverse Document Frequency)  
        **Algoritma:** Support Vector Machine dengan kernel Linear  
        **Split Data:** 80:20 (Training:Testing)  
        **Evaluasi:** Confusion Matrix, Accuracy, Precision, Recall
        """)
        
        if st.button("🚀 Latih Model SVM", type="primary"):
            with st.spinner("⏳ Melatih model SVM dengan kernel Linear..."):
                # ================== EKSTRAKSI FITUR TF-IDF ==================
                vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                X = vectorizer.fit_transform(df['text_preprocessed'])
                y = df['sentiment']
                
                # ================== PEMBAGIAN DATA 80:20 ==================
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # ================== TRAINING SVM ==================
                svm_model = SVC(kernel='linear', random_state=42)
                svm_model.fit(X_train, y_train)
                
                # Prediksi
                y_pred = svm_model.predict(X_test)
                
                # Simpan ke session state
                st.session_state['svm_model'] = svm_model
                st.session_state['vectorizer'] = vectorizer
                
                # ================== EVALUASI MODEL ==================
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred, labels=['Positif', 'Negatif', 'Netral'])
                cr = classification_report(y_test, y_pred, output_dict=True)
                
                st.success(f"✅ Model berhasil dilatih dengan **Accuracy: {accuracy:.4f}**")
                
                # Visualisasi Evaluasi
                col_eval1, col_eval2 = st.columns(2)
                
                with col_eval1:
                    st.subheader("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Positif', 'Negatif', 'Netral'],
                               yticklabels=['Positif', 'Negatif', 'Netral'], ax=ax_cm)
                    ax_cm.set_xlabel('Predicted', fontsize=12)
                    ax_cm.set_ylabel('Actual', fontsize=12)
                    ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                    st.pyplot(fig_cm)
                
                with col_eval2:
                    st.subheader("Classification Report")
                    report_df = pd.DataFrame(cr).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                
                # Metrik per kelas
                st.subheader("Performa Model per Sentimen (Accuracy, Precision, Recall)")
                metrics_data = []
                for label in ['Positif', 'Negatif', 'Netral']:
                    if label in cr:
                        metrics_data.append({
                            'Sentimen': label,
                            'Precision': f"{cr[label]['precision']:.4f}",
                            'Recall': f"{cr[label]['recall']:.4f}",
                            'F1-Score': f"{cr[label]['f1-score']:.4f}",
                            'Support': int(cr[label]['support'])
                        })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # TF-IDF Top Features
                st.subheader("Top 10 Fitur TF-IDF Paling Berpengaruh")
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = X.toarray().mean(axis=0)
                top_features = pd.DataFrame({
                    'Fitur': feature_names,
                    'Skor TF-IDF': tfidf_scores
                }).sort_values('Skor TF-IDF', ascending=False).head(10)
                
                fig_tfidf, ax_tfidf = plt.subplots(figsize=(10, 6))
                ax_tfidf.barh(top_features['Fitur'], top_features['Skor TF-IDF'], color='steelblue')
                ax_tfidf.set_xlabel('Skor TF-IDF Rata-rata', fontsize=12)
                ax_tfidf.set_ylabel('Fitur', fontsize=12)
                ax_tfidf.set_title('Top 10 Fitur TF-IDF', fontsize=14, fontweight='bold')
                ax_tfidf.invert_yaxis()
                st.pyplot(fig_tfidf)
    
    # --- TAB 5: PREDIKSI BARU ---
    with tab5:
        st.subheader("🔮 Prediksi Sentimen Ulasan Baru")
        
        if 'svm_model' not in st.session_state:
            st.warning("⚠️ Model belum dilatih! Silakan latih model di tab **Model SVM & Evaluasi** terlebih dahulu.")
        else:
            st.info("💡 Masukkan teks ulasan baru untuk memprediksi sentimennya")
            
            # Input ulasan baru
            new_review = st.text_area("Masukkan ulasan:", 
                                      placeholder="Contoh: Barang bagus banget, cepat sampai, packing rapi. Terima kasih!",
                                      height=100)
            
            col_pred1, col_pred2 = st.columns([1, 3])
            
            with col_pred1:
                predict_button = st.button("🚀 Prediksi Sentimen", type="primary", use_container_width=True)
            
            if predict_button and new_review:
                with st.spinner("⏳ Memproses ulasan..."):
                    # Preprocessing
                    preprocessed = preprocess_text(new_review)
                    
                    # Transform dengan TF-IDF
                    vectorizer = st.session_state['vectorizer']
                    new_review_tfidf = vectorizer.transform([preprocessed])
                    
                    # Prediksi
                    svm_model = st.session_state['svm_model']
                    prediction = svm_model.predict(new_review_tfidf)[0]
                    
                    # Tampilkan hasil
                    st.divider()
                    st.subheader("Hasil Prediksi")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.write("**Teks Original:**")
                        st.info(new_review)
                        
                        st.write("**Teks Setelah Preprocessing:**")
                        st.code(preprocessed)
                    
                    with col_res2:
                        # Tampilkan prediksi dengan warna
                        if prediction == 'Positif':
                            st.success(f"### Sentimen: {prediction} 😊")
                        elif prediction == 'Negatif':
                            st.error(f"### Sentimen: {prediction} 😞")
                        else:
                            st.warning(f"### Sentimen: {prediction} 😐")
                        
                        # Tampilkan kata kunci yang terdeteksi
                        words = set(preprocessed.split())
                        pos_words = words.intersection(POSITIVE_WORDS)
                        neg_words = words.intersection(NEGATIVE_WORDS)
                        
                        st.write("**Kata Kunci Terdeteksi:**")
                        if pos_words:
                            st.write(f"✅ Positif: {', '.join(list(pos_words)[:5])}")
                        if neg_words:
                            st.write(f"❌ Negatif: {', '.join(list(neg_words)[:5])}")
            
            elif predict_button and not new_review:
                st.error("⚠️ Silakan masukkan teks ulasan terlebih dahulu!")

else:
    st.info("👈 Silakan upload file CSV dataset ulasan Tokopedia di sidebar untuk memulai analisis")
    
    # Tampilkan informasi kelompok
    st.subheader("👥 Nama Kelompok")
    st.markdown("""
    - **Shelly Ananda** (2411500891)
    - **M. Richo Irsyad F.** (2411500958)
    - **Nazif Hamza Effendy** (2411501527)
    """)
