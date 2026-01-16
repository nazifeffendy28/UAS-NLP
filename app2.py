import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Sentimen Tokopedia", layout="wide")

# Kamus normalisasi kata gaul/slang
NORMALIZATION_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'bgt': 'banget', 'bgt': 'banget', 'bngd': 'banget', 'bngt': 'banget',
    'bgs': 'bagus', 'bgus': 'bagus', 'bgt': 'banget',
    'mantap': 'mantap', 'mantul': 'mantap', 'mantab': 'mantap',
    'jelek': 'jelek', 'jlek': 'jelek', 'jelk': 'jelek',
    'ok': 'oke', 'oke': 'oke', 'okeh': 'oke',
    'rekomend': 'rekomendasi', 'rekomen': 'rekomendasi',
    'puas': 'puas', 'puass': 'puas', 'puasss': 'puas',
    'kecewa': 'kecewa', 'kcwa': 'kecewa',
    'cpt': 'cepat', 'cpet': 'cepat', 'cepet': 'cepat',
    'lama': 'lama', 'lma': 'lama', 'telat': 'terlambat',
    'murah': 'murah', 'mrh': 'murah', 'mahal': 'mahal', 'mhl': 'mahal',
    'kualitas': 'kualitas', 'kualitas': 'kualitas', 'quality': 'kualitas',
    'original': 'asli', 'ori': 'asli', 'kw': 'palsu', 'fake': 'palsu',
    'recommended': 'rekomendasi', 'recommend': 'rekomendasi',
    'tq': 'terima kasih', 'thx': 'terima kasih', 'thanks': 'terima kasih', 
    'makasih': 'terima kasih', 'mksh': 'terima kasih', 'tks': 'terima kasih',
    'brg': 'barang', 'barang': 'barang',
    'krm': 'kirim', 'pengiriman': 'pengiriman', 'pngrmn': 'pengiriman',
    'seller': 'penjual', 'penjual': 'penjual', 'toko': 'toko',
    'respon': 'responsif', 'fast': 'cepat', 'slow': 'lambat',
    'rusak': 'rusak', 'rsk': 'rusak', 'cacat': 'cacat',
    'sesuai': 'sesuai', 'ssuai': 'sesuai', 'cocok': 'cocok',
    'krg': 'kurang', 'kurang': 'kurang',
    'byk': 'banyak', 'banyak': 'banyak',
    'tp': 'tetapi', 'tapi': 'tetapi', 'tp': 'tetapi',
    'utk': 'untuk', 'untuk': 'untuk', 'buat': 'untuk',
    'sgt': 'sangat', 'sangat': 'sangat', 'bener': 'benar',
    'keren': 'keren', 'krn': 'keren', 'top': 'bagus',
    'buruk': 'buruk', 'bruk': 'buruk', 'ancur': 'hancur',
    'ampun': 'ampun', 'parah': 'parah', 'prh': 'parah',
    'sukses': 'sukses', 'sks': 'sukses', 'sukses': 'sukses',
    'keren': 'keren', 'krenn': 'keren', 'gokil': 'keren'
}

# Stopwords bahasa Indonesia
STOPWORDS = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada',
    'adalah', 'ada', 'atau', 'oleh', 'se', 'te', 'nya', 'an', 'kan', 'akan',
    'juga', 'saya', 'kamu', 'dia', 'mereka', 'kami', 'kita', 'anda', 'ia',
    'sudah', 'telah', 'akan', 'sedang', 'masih', 'pernah', 'belum', 'tidak',
    'bukan', 'jangan', 'bila', 'kalau', 'jika', 'karena', 'maka', 'namun',
    'tetapi', 'walaupun', 'meskipun', 'agar', 'supaya', 'semoga', 'dalam',
    'kepada', 'terhadap', 'antara', 'tentang', 'hingga', 'sambil', 'demi',
    'sebelum', 'sesudah', 'saat', 'ketika', 'sewaktu', 'begitu', 'seperti',
    'bagai', 'ibarat', 'umpama', 'laksana', 'seolah', 'serupa', 'macam',
    'secara', 'setiap', 'seluruh', 'semua', 'para', 'sang', 'si', 'sri'
])

# Lexicon-based sentiment (kata kunci sentimen)
POSITIVE_WORDS = set([
    'bagus', 'baik', 'mantap', 'puas', 'suka', 'cepat', 'murah', 'recommended',
    'rekomendasi', 'oke', 'original', 'asli', 'keren', 'top', 'sukses', 'sesuai',
    'cocok', 'kualitas', 'responsif', 'ramah', 'memuaskan', 'terpercaya', 'lengkap',
    'rapi', 'aman', 'nyaman', 'sempurna', 'istimewa', 'terbaik', 'favorit', 'worth',
    'gercep', 'excellent', 'mantul', 'terima kasih', 'thanks', 'love', 'senang'
])

NEGATIVE_WORDS = set([
    'jelek', 'buruk', 'kecewa', 'lambat', 'lama', 'mahal', 'rusak', 'cacat',
    'palsu', 'fake', 'tidak', 'kurang', 'mengecewakan', 'parah', 'hancur',
    'ampun', 'zonk', 'php', 'berbeda', 'beda', 'salah', 'error', 'pecah',
    'sobek', 'kotor', 'bau', 'busuk', 'najis', 'sampah', 'mengecewakan',
    'terlambat', 'slow', 'bad', 'worst', 'terrible', 'awful', 'horrible',
    'bohong', 'penipuan', 'tipu', 'palsu', 'kw', 'reject'
])

# Fungsi Preprocessing
def cleaning(text):
    """Cleaning: menghapus tanda baca, angka, dan karakter non-alfanumerik"""
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def case_folding(text):
    """Case Folding: mengubah seluruh teks menjadi huruf kecil"""
    return text.lower()

def normalization(text):
    """Normalization: mengubah kata gaul/slang menjadi kata baku"""
    words = text.split()
    normalized = [NORMALIZATION_DICT.get(word, word) for word in words]
    return ' '.join(normalized)

def stopword_removal(text):
    """Stopword Removal: menghapus kata umum yang tidak memiliki makna sentimen"""
    words = text.split()
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return ' '.join(filtered)

def preprocess_text(text):
    """Pipeline preprocessing lengkap"""
    text = cleaning(text)
    text = case_folding(text)
    text = normalization(text)
    text = stopword_removal(text)
    return text

# Fungsi Pelabelan Sentimen (Hybrid Approach)
def lexicon_sentiment(text):
    """Penentuan label awal menggunakan pendekatan lexicon-based"""
    words = set(text.split())
    pos_count = len(words.intersection(POSITIVE_WORDS))
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    
    if pos_count > neg_count:
        return 'Positif'
    elif neg_count > pos_count:
        return 'Negatif'
    else:
        return 'Netral'

def hybrid_labeling(row):
    """Validasi silang dengan rating untuk koreksi label"""
    lexicon_label = lexicon_sentiment(row['text_preprocessed'])
    rating = row['rating']
    
    # Aturan koreksi berdasarkan rating
    if lexicon_label == 'Positif' and rating <= 3:
        return 'Netral'
    elif lexicon_label == 'Negatif' and rating >= 4:
        return 'Netral'
    else:
        return lexicon_label

# UI Streamlit
st.title("📊 Analisis Sentimen Ulasan Tokopedia")
st.markdown("**Metodologi:** Waterfall untuk Data Science dengan SVM & TF-IDF")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV ulasan Tokopedia", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Dataset berhasil dimuat: {len(df)} baris data")
    
    # Validasi kolom wajib
    required_cols = ['ulasan', 'rating']
    optional_cols = ['nama_produk', 'kategori', 'jumlah_terjual']
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"⚠️ Kolom wajib tidak ditemukan. Pastikan ada kolom: {', '.join(required_cols)}")
        st.stop()
    
    # Preprocessing
    with st.spinner("⏳ Melakukan preprocessing data..."):
        df['text_preprocessed'] = df['ulasan'].apply(preprocess_text)
        
        # Pelabelan Hybrid
        df['sentiment'] = df.apply(hybrid_labeling, axis=1)
    
    # Tab untuk visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "☁️ WordCloud", "📋 Data", "🤖 Model"])
    
    with tab1:
        st.subheader("Distribusi Sentimen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            sentiment_counts = df['sentiment'].value_counts()
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors)
            ax.set_title('Proporsi Sentimen', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='sentiment', palette=['#2ecc71', '#e74c3c', '#95a5a6'], ax=ax)
            ax.set_title('Distribusi Sentimen', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sentimen', fontsize=12)
            ax.set_ylabel('Jumlah', fontsize=12)
            for container in ax.containers:
                ax.bar_label(container)
            st.pyplot(fig)
        
        # Statistik
        st.subheader("Statistik Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ulasan", len(df))
        col2.metric("Rata-rata Rating", f"{df['rating'].mean():.2f}")
        col3.metric("Modus Sentimen", df['sentiment'].mode()[0])
    
    with tab2:
        st.subheader("WordCloud per Sentimen")
        
        sentiment_filter = st.selectbox("Pilih Sentimen", ['Positif', 'Negatif', 'Netral'])
        
        filtered_text = ' '.join(df[df['sentiment'] == sentiment_filter]['text_preprocessed'])
        
        if filtered_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  colormap='RdYlGn' if sentiment_filter == 'Positif' else 'Reds',
                                  max_words=100).generate(filtered_text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'WordCloud - {sentiment_filter}', fontsize=16, fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning(f"Tidak ada data untuk sentimen {sentiment_filter}")
    
    with tab3:
        st.subheader("Tabel Data Hasil Preprocessing")
        
        # Filter
        sentiment_filter_table = st.multiselect("Filter Sentimen", 
                                                 df['sentiment'].unique(), 
                                                 default=df['sentiment'].unique())
        
        display_df = df[df['sentiment'].isin(sentiment_filter_table)]
        
        # Kolom yang ditampilkan
        display_cols = ['ulasan', 'text_preprocessed', 'rating', 'sentiment']
        if 'nama_produk' in df.columns:
            display_cols.insert(0, 'nama_produk')
        
        st.dataframe(display_df[display_cols], use_container_width=True, height=400)
        
        # Download hasil
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil", csv, "hasil_preprocessing.csv", "text/csv")
    
    with tab4:
        st.subheader("Training Model SVM dengan TF-IDF")
        
        if st.button("🚀 Train Model", type="primary"):
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
                
                # Evaluasi
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred, labels=['Positif', 'Negatif', 'Netral'])
                cr = classification_report(y_test, y_pred, output_dict=True)
                
                st.success(f"✅ Model berhasil dilatih dengan Accuracy: **{accuracy:.4f}**")
                
                # Confusion Matrix
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['Positif', 'Negatif', 'Netral'],
                                yticklabels=['Positif', 'Negatif', 'Netral'], ax=ax)
                    ax.set_xlabel('Predicted', fontsize=12)
                    ax.set_ylabel('Actual', fontsize=12)
                    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                
                with col2:
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
                            'Precision': cr[label]['precision'],
                            'Recall': cr[label]['recall'],
                            'F1-Score': cr[label]['f1-score'],
                            'Support': int(cr[label]['support'])
                        })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

else:
    st.info("Upload file CSV untuk memulai analisis")
    st.markdown("""
    **Format CSV yang dibutuhkan:**
    - `ulasan` (wajib): teks ulasan produk
    - `rating` (wajib): rating 1-5
    - `nama_produk` (opsional): nama produk
    - `kategori` (opsional): kategori produk
    - `jumlah_terjual` (opsional): jumlah produk terjual
    """)
