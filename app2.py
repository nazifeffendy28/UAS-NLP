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
# Dictionary Normalisasi: mengubah kata gaul/slang menjadi kata baku
NORMALIZATION_DICT = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "engga": "tidak",
    "brg": "barang", "sdh": "sudah", "dgn": "dengan", "thx": "terima kasih",
    "tks": "terima kasih", "makasih": "terima kasih", "bgt": "banget",
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
    "recomended": "rekomendasi", "rekomend": "rekomendasi"
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
    "terima kasih", "thanks", "love", "senang", "sukses"
}

# Lexicon Negatif (untuk pelabelan berbasis leksikon)
NEGATIVE_WORDS = {
    "jelek", "buruk", "lambat", "lama", "rusak", "cacat", "pecah", "penyok",
    "kecewa", "salah", "beda", "tipis", "kasar", "kotor", "mahal", "bohong",
    "palsu", "kw", "robek", "bolong", "batal", "retur", "komplain", "parah",
    "nyesel", "kurang", "tidak", "panas", "gerah", "gatal", "sempit", 
    "longgar", "kebesaran", "kekecilan", "luntur", "pudar", "kusam",
    "bau", "ape", "lecek", "kusut", "benang", "jahitan", "lepas", "copot",
    "penipuan", "penipu", "lamban", "lelet", "jutek", "galak", "kasar",
    "sombong", "ribet", "susah", "baret", "gores", "bekas", "mengecewakan",
    "hancur", "ampun", "zonk", "terrible", "awful", "worst", "bad"
}

# Stopwords: kata umum yang tidak memiliki makna sentimen
STOPWORDS = {
    "yang", "di", "dan", "itu", "ini", "dari", "ke", "untuk", "dengan", "nya",
    "saya", "aku", "kami", "kita", "bisa", "ada", "adalah", "juga", "karena",
    "tapi", "namun", "atau", "jadi", "jika", "kalau", "sudah", "lagi", "akan",
    "pada", "masih", "saja", "yg", "dg", "rt", "dgn", "ny", "d", "k",
    "kalo", "biar", "bikin", "bilang", "gak", "ga", "krn", "nya", "nih",
    "sih", "si", "tau", "tdk", "tuh", "utk", "ya", "jd", "jgn", "sdh", 
    "aja", "n", "t", "nyg", "hehe", "pen", "u", "nan", "loh", "rt",
    "oleh", "se", "te", "an", "kan", "dia", "mereka", "ia", "telah",
    "sedang", "pernah", "belum", "bukan", "jangan", "bila", "maka",
    "walaupun", "meskipun", "agar", "supaya", "semoga", "dalam", "kepada",
    "terhadap", "antara", "tentang", "hingga", "sambil", "demi", "sebelum",
    "sesudah", "saat", "ketika", "sewaktu", "begitu", "seperti", "bagai",
    "ibarat", "umpama", "laksana", "seolah", "serupa", "macam", "secara",
    "setiap", "seluruh", "semua", "para", "sang", "si", "sri"
}

# ================== FUNGSI PREPROCESSING (SESUAI METODOLOGI) ==================
def cleaning(text):
    """Cleaning: menghapus tanda baca, angka, dan karakter non-alfanumerik"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def case_folding(text):
    """Case Folding: mengubah seluruh teks menjadi huruf kecil"""
    return text.lower()

def normalization(text):
    """Normalization: mengubah kata gaul/slang menjadi kata baku menggunakan kamus"""
    words = text.split()
    normalized = [NORMALIZATION_DICT.get(word, word) for word in words]
    return ' '.join(normalized)

def stopword_removal(text):
    """Stopword Removal: menghapus kata umum yang tidak memiliki makna sentimen"""
    words = text.split()
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return ' '.join(filtered)

def preprocess_text(text):
    """Pipeline preprocessing lengkap sesuai metodologi"""
    text = cleaning(text)
    text = case_folding(text)
    text = normalization(text)
    text = stopword_removal(text)
    return text

# ================== PELABELAN HYBRID (SESUAI METODOLOGI) ==================
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

def hybrid_labeling(row, rating_col):
    """Validasi silang dengan rating untuk koreksi label"""
    lexicon_label = lexicon_sentiment(row['text_preprocessed'])
    
    try:
        rating = int(row[rating_col])
    except:
        return lexicon_label
    
    # Aturan koreksi berdasarkan rating
    if lexicon_label == 'Positif' and rating <= 3:
        return 'Netral'
    elif lexicon_label == 'Negatif' and rating >= 4:
        return 'Netral'
    else:
        return lexicon_label

# ================== UI UTAMA ==================
st.title("📊 Analisis Sentimen Ulasan Tokopedia")
st.markdown("**Metodologi:** Waterfall untuk Data Science dengan SVM & TF-IDF")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV Tokopedia", type=["csv"])
    
    st.divider()
    st.header("⚙️ Pengaturan")
    st.info("Model SVM akan dilatih dengan rasio data 80:20")

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset berhasil dimuat: {len(df)} baris data")
    
    # Pilih kolom
    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("Pilih Kolom Ulasan:", df.columns, index=0)
    with col2:
        rating_col = st.selectbox("Pilih Kolom Rating:", df.columns, index=1)
    
    # Preprocessing
    with st.spinner("⏳ Melakukan preprocessing data..."):
        df['text_preprocessed'] = df[text_col].astype(str).apply(preprocess_text)
        df['sentiment'] = df.apply(lambda x: hybrid_labeling(x, rating_col), axis=1)
    
    st.success("✅ Preprocessing selesai!")
    
    # ================== TAB ANALISIS ==================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "☁️ WordCloud", 
        "📋 Data", 
        "🤖 Model SVM",
        "🔮 Prediksi"
    ])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.subheader("Statistik Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ulasan", len(df))
        col2.metric("Rata-rata Rating", f"{df[rating_col].mean():.2f} ⭐")
        sentiment_dist = df['sentiment'].value_counts()
        col3.metric("Sentimen Dominan", sentiment_dist.index[0])
        
        st.subheader("Distribusi Sentimen")
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
            sns.countplot(data=df, x='sentiment', palette=colors, ax=ax_bar)
            ax_bar.set_title('Distribusi Sentimen', fontsize=14, fontweight='bold')
            ax_bar.set_xlabel('Sentimen', fontsize=12)
            ax_bar.set_ylabel('Jumlah', fontsize=12)
            for container in ax_bar.containers:
                ax_bar.bar_label(container)
            st.pyplot(fig_bar)
        
        # Sentimen per Rating
        st.subheader("Sentimen per Rating")
        ct = pd.crosstab(df[rating_col], df['sentiment'])
        st.bar_chart(ct)
    
    # --- TAB 2: WORDCLOUD ---
    with tab2:
        st.subheader("WordCloud per Sentimen")
        
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
    
    # --- TAB 3: DATA ---
    with tab3:
        st.subheader("Tabel Data Hasil Preprocessing & Pelabelan")
        
        # Filter sentimen
        sentiment_filter_table = st.multiselect(
            "Filter Sentimen:", 
            df['sentiment'].unique(), 
            default=list(df['sentiment'].unique())
        )
        
        display_df = df[df['sentiment'].isin(sentiment_filter_table)]
        
        # Kolom yang ditampilkan
        display_cols = [text_col, 'text_preprocessed', rating_col, 'sentiment']
        
        # Tambahkan kolom opsional jika ada
        optional_cols = ['product_name', 'category', 'sold']
        for col in optional_cols:
            if col in df.columns and col not in display_cols:
                display_cols.insert(0, col)
        
        st.dataframe(display_df[display_cols], use_container_width=True, height=400)
        
        # Download hasil
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Preprocessing", csv, "hasil_preprocessing.csv", "text/csv")
    
    # --- TAB 4: MODEL SVM ---
    with tab4:
        st.subheader("🧠 Training Model SVM dengan TF-IDF")
        st.markdown("""
        **Metodologi:**
        - Ekstraksi Fitur: TF-IDF (Term Frequency–Inverse Document Frequency)
        - Algoritma: Support Vector Machine (SVM) dengan kernel Linear
        - Split Data: 80% Training, 20% Testing
        """)
        
        if st.button("🚀 Latih Model SVM", type="primary"):
            with st.spinner("⏳ Melatih model SVM..."):
                # Ekstraksi Fitur dengan TF-IDF
                vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                X = vectorizer.fit_transform(df['text_preprocessed'])
                y = df['sentiment']
                
                # Split data 80:20
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Training SVM dengan kernel Linear
                svm_model = SVC(kernel='linear', random_state=42)
                svm_model.fit(X_train, y_train)
                
                # Prediksi
                y_pred = svm_model.predict(X_test)
                
                # Simpan ke session state
                st.session_state['svm_model'] = svm_model
                st.session_state['vectorizer'] = vectorizer
                
                # Evaluasi
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred, labels=['Positif', 'Negatif', 'Netral'])
                cr = classification_report(y_test, y_pred, output_dict=True)
                
                st.success(f"✅ Model berhasil dilatih dengan Accuracy: **{accuracy:.4f}**")
                
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
                st.subheader("Performa per Sentimen")
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
                
                fig_tfidf, ax_tfidf = plt.subplots(figsize=(10, 5))
                sns.barplot(data=top_features, x='Skor TF-IDF', y='Fitur', 
                           palette='viridis', ax=ax_tfidf)
                ax_tfidf.set_title('Top 10 Fitur TF-IDF', fontsize=14, fontweight='bold')
                st.pyplot(fig_tfidf)
        
        elif 'svm_model' in st.session_state:
            st.info("✅ Model sudah tersimpan di memori. Anda dapat langsung melakukan prediksi di tab **Prediksi**.")
        else:
            st.warning("⚠️ Klik tombol 'Latih Model SVM' untuk memulai training.")
    
    # --- TAB 5: PREDIKSI MANUAL ---
    with tab5:
        st.subheader("🔮 Prediksi Sentimen Ulasan Baru")
        
        user_input = st.text_area("Masukkan teks ulasan produk:", height=150,
                                  placeholder="Contoh: Barang bagus, cepat sampai, packing rapi...")
        
        if st.button("🔍 Prediksi Sentimen") and user_input:
            # Preprocessing input
            clean_input = preprocess_text(user_input)
            
            # Prediksi Lexicon
            lexicon_pred = lexicon_sentiment(clean_input)
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.info(f"**Metode Lexicon-Based:** {lexicon_pred}")
                st.caption("Prediksi berdasarkan kamus kata positif/negatif")
            
            # Prediksi SVM
            with col_res2:
                if 'svm_model' in st.session_state:
                    vec_input = st.session_state['vectorizer'].transform([clean_input])
                    svm_pred = st.session_state['svm_model'].predict(vec_input)[0]
                    st.success(f"**Metode SVM (Machine Learning):** {svm_pred}")
                    st.caption("Prediksi menggunakan model SVM terlatih")
                else:
                    st.warning("**Metode SVM:** Model belum dilatih")
                    st.caption("Latih model di tab 'Model SVM' terlebih dahulu")
            
            # Tampilkan hasil preprocessing
            with st.expander("Lihat Hasil Preprocessing"):
                st.write("**Teks Asli:**", user_input)
                st.write("**Teks Setelah Preprocessing:**", clean_input)

else:
    st.info("📂 Silakan upload file CSV untuk memulai analisis")
    st.markdown("""
    **Format CSV yang dibutuhkan:**
    - Kolom ulasan/teks ulasan (wajib)
    - Kolom rating (wajib)
    - Kolom product_name/nama_produk (opsional)
    - Kolom category/kategori (opsional)
    - Kolom sold/jumlah_terjual (opsional)
    
    **Contoh struktur data:**
```
    ulasan,rating,product_name,category,sold
    "Barang bagus sekali",5,"Kaos Polos","fashion",1234
    "Kecewa dengan kualitas",2,"Sepatu Olahraga","olahraga",567
```
    """)
