import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Advanced Sentiment Analysis Tokopedia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== RESOURCE LOADING & CACHING ==================
@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

# ================== KAMUS KATA LENGKAP (EXPANDED) ==================
# Dictionary Slang ke Baku
slang_dict = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "engga": "tidak",
    "brg": "barang", "sdh": "sudah", "dgn": "dengan", "thx": "terima kasih",
    "tks": "terima kasih", "makasih": "terima kasih", "bgt": "banget",
    "kalo": "kalau", "kl": "kalau", "tp": "tapi", "dr": "dari",
    "bs": "bisa", "sy": "saya", "ak": "aku", "aq": "aku",
    "mantul": "mantap betul", "mantap": "bagus", "jos": "bagus", 
    "ok": "oke", "oke": "bagus", "good": "bagus", "best": "bagus",
    "jelek": "buruk", "parah": "buruk", "ancur": "buruk", "rusak": "buruk",
    "dtg": "datang", "sampe": "sampai", "nyampe": "sampai", "cepet": "cepat",
    "kirim": "pengiriman", "kurir": "pengiriman", "packing": "kemasan",
    "seller": "penjual", "respon": "tanggapan", "bintang": "rating",
    "gan": "juragan", "sis": "kakak", "kak": "kakak", "om": "paman",
    "admin": "penjual", "olshop": "toko online", "bhn": "bahan",
    "adem": "sejuk", "mlar": "melar", "size": "ukuran", "pas": "sesuai",
    "trmksh": "terima kasih", "thankyou": "terima kasih"
}

# Lexicon Positif (Diperluas untuk Fashion & Umum)
positive_words = {
    "bagus", "baik", "cepat", "rapi", "aman", "sesuai", "mantap", "puas", 
    "oke", "keren", "nyaman", "suka", "awet", "murah", "ramah", "lengkap",
    "lucu", "halus", "lembut", "tebal", "asli", "original", "recomended",
    "recommended", "top", "memuaskan", "berfungsi", "pas", "cocok", "sejuk",
    "adem", "dingin", "menyerap", "keringat", "elegan", "mewah", "premium",
    "rapih", "kilat", "gesit", "responsif", "sopan", "jujur", "amanah",
    "bonus", "hadiah", "terjangkau", "hemat", "diskon", "promo", "bersih",
    "wangi", "harum", "cantik", "ganteng", "kece", "modis", "trendy"
}

# Lexicon Negatif (Diperluas untuk Fashion & Umum)
negative_words = {
    "jelek", "buruk", "lambat", "lama", "rusak", "cacat", "pecah", "penyok",
    "kecewa", "salah", "beda", "tipis", "kasar", "kotor", "mahal", "bohong",
    "palsu", "kw", "robek", "bolong", "batal", "retur", "komplain", "parah",
    "nyesel", "kurang", "tidak", "panas", "gerah", "gatal", "sempit", 
    "longgar", "kebesaran", "kekecilan", "luntur", "pudar", "kusam",
    "bau", "ape", "lecek", "kusut", "benang", "jahitan", "lepas", "copot",
    "penipuan", "penipu", "lamban", "lelet", "jutek", "galak", "kasar",
    "sombong", "ribet", "susah", "baret", "gores", "bekas"
}

# Stopwords
custom_stopwords = {
    "yang", "di", "dan", "itu", "ini", "dari", "ke", "untuk", "dengan", "nya",
    "saya", "aku", "kami", "kita", "bisa", "ada", "adalah", "juga", "karena",
    "tapi", "namun", "atau", "jadi", "jika", "kalau", "sudah", "lagi", "akan",
    "pada", "masih", "saja", "yg", "dg", "rt", "dgn", "ny", "d", "k",
    "kalo", "biar", "bikin", "bilang", "gak", "ga", "krn", "nya", "nih",
    "sih", "si", "tau", "tdk", "tuh", "utk", "ya", "jd", "jgn", "sdh", 
    "aja", "n", "t", "nyg", "hehe", "pen", "u", "nan", "loh", "rt"
}

# ================== FUNGSI PREPROCESSING ==================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    normalized_words = [slang_dict.get(w, w) for w in words]
    filtered_words = [w for w in normalized_words if w not in custom_stopwords]
    # Stemming opsional (bisa di-uncomment jika performa server kuat)
    # stemmed_words = [stemmer.stem(w) for w in filtered_words]
    return " ".join(filtered_words)

# ================== FUNGSI SENTIMEN (HYBRID & LABELING) ==================
def get_lexicon_sentiment(text):
    words = text.split()
    score = 0
    for word in words:
        if word in positive_words: score += 1
        elif word in negative_words: score -= 1
    return 'Positif' if score > 0 else 'Negatif' if score < 0 else 'Netral'

def hybrid_sentiment_logic(text_sentiment, rating):
    try: r = int(rating)
    except: return text_sentiment
    
    # Logika Koreksi
    if text_sentiment == 'Positif' and r <= 3: return 'Netral' # Sarkas/Salah rating
    if text_sentiment == 'Negatif' and r >= 4: return 'Netral' # Komplain tapi rating bagus
    
    # Fallback ke Rating jika teks netral
    if text_sentiment == 'Netral':
        if r >= 4: return 'Positif'
        elif r <= 2: return 'Negatif'
        
    return text_sentiment

# Label Ground Truth dari Rating (Untuk Pelatihan SVM)
def get_rating_label(rating):
    try: r = int(rating)
    except: return 'Netral'
    if r >= 4: return 'Positif'
    elif r <= 2: return 'Negatif'
    return 'Netral'

# ================== UI UTAMA ==================
st.title("🤖 Advanced Sentiment Analysis & SVM Prediction")
st.markdown("Dashboard analisis sentimen ulasan Tokopedia dengan pendekatan Hybrid (Lexicon + Rating) dan Machine Learning (SVM).")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV Tokopedia", type=["csv"])
    
    st.header("⚙️ 2. Pengaturan")
    category_filter = st.selectbox("Filter Kategori:", ["Semua", "fashion", "pertukangan", "elektronik", "olahraga", "handphone"])
    
    st.divider()
    st.header("🧠 3. Model Training")
    train_model_btn = st.button("Latih Model SVM")
    test_size_param = st.slider("Rasio Data Test (%)", 10, 50, 20) / 100

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- DATA FILTERING ---
    if category_filter != "Semua" and 'category' in df.columns:
        df = df[df['category'] == category_filter]
    
    # --- PREPROCESSING ---
    text_col = st.selectbox("Kolom Teks:", df.columns, index=1)
    rating_col = st.selectbox("Kolom Rating:", df.columns, index=2)
    
    with st.spinner("Melakukan preprocessing data..."):
        df['Cleaned_Text'] = df[text_col].astype(str).apply(clean_text)
        df['Lexicon_Sentiment'] = df['Cleaned_Text'].apply(get_lexicon_sentiment)
        df['Final_Sentiment'] = df.apply(lambda x: hybrid_sentiment_logic(x['Lexicon_Sentiment'], x[rating_col]), axis=1)
        # Label ground truth untuk SVM dari rating murni
        df['Label_True'] = df[rating_col].apply(get_rating_label) 

    # ================== TAB ANALISIS ==================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard Sentimen", 
        "🏆 Top Produk", 
        "🔠 TF-IDF & Kata Kunci", 
        "🧠 Model SVM & Evaluasi",
        "🔮 Prediksi Manual"
    ])

    # --- TAB 1: DASHBOARD SENTIMEN ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Ulasan", len(df))
        c2.metric("Rata-rata Rating", f"{df[rating_col].mean():.2f} ⭐")
        c3.metric("Sentimen Positif", f"{len(df[df['Final_Sentiment']=='Positif'])/len(df)*100:.1f}%")

        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.subheader("Distribusi Sentimen")
            fig_pie, ax_pie = plt.subplots()
            df['Final_Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax_pie, colors=['#66b3ff','#ff9999','#99ff99'])
            ax_pie.set_ylabel('')
            st.pyplot(fig_pie)
        
        with col_viz2:
            st.subheader("Sentimen per Rating")
            ct = pd.crosstab(df[rating_col], df['Final_Sentiment'])
            st.bar_chart(ct)
            
        st.subheader("WordCloud Global")
        all_text = " ".join(df['Cleaned_Text'])
        wc = WordCloud(width=800, height=300, background_color='white').generate(all_text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

    # --- TAB 2: TOP PRODUK (BUSINESS INTELLIGENCE) ---
    with tab2:
        if 'product_name' in df.columns:
            col_prod1, col_prod2 = st.columns(2)
            
            with col_prod1:
                st.subheader("🛒 Top 10 Produk Terlaris (Sold)")
                if 'sold' in df.columns:
                    # Bersihkan kolom sold jika ada teks (misal "1rb+")
                    df['sold_clean'] = pd.to_numeric(df['sold'], errors='coerce').fillna(0)
                    top_sold = df.groupby('product_name')['sold_clean'].sum().nlargest(10).sort_values()
                    fig_sold, ax_sold = plt.subplots()
                    top_sold.plot(kind='barh', ax=ax_sold, color='skyblue')
                    ax_sold.set_title("Total Terjual")
                    st.pyplot(fig_sold)
                else:
                    st.warning("Kolom 'sold' tidak ditemukan.")

            with col_prod2:
                st.subheader("💬 Top 10 Produk Paling Banyak Diulas")
                top_reviewed = df['product_name'].value_counts().nlargest(10).sort_values()
                fig_rev, ax_rev = plt.subplots()
                top_reviewed.plot(kind='barh', ax=ax_rev, color='salmon')
                ax_rev.set_title("Jumlah Ulasan")
                st.pyplot(fig_rev)
        else:
            st.error("Kolom 'product_name' tidak ditemukan dalam dataset.")

    # --- TAB 3: TF-IDF ANALYSIS ---
    with tab3:
        st.subheader("🔍 Top 10 Kata Paling Berpengaruh (TF-IDF)")
        tfidf = TfidfVectorizer(max_features=1000, stop_words=list(custom_stopwords))
        try:
            tfidf_matrix = tfidf.fit_transform(df['Cleaned_Text'])
            feature_names = tfidf.get_feature_names_out()
            
            # Sum TF-IDF scores for each word
            dense = tfidf_matrix.todense()
            denselist = dense.tolist()
            df_tfidf = pd.DataFrame(denselist, columns=feature_names)
            top_words = df_tfidf.mean().sort_values(ascending=False).head(10)
            
            fig_tfidf, ax_tfidf = plt.subplots(figsize=(10, 5))
            sns.barplot(x=top_words.values, y=top_words.index, ax=ax_tfidf, palette="viridis")
            ax_tfidf.set_title("Rata-rata Skor TF-IDF")
            st.pyplot(fig_tfidf)
            
            with st.expander("Lihat Data TF-IDF"):
                st.dataframe(top_words)
        except ValueError:
            st.error("Data teks tidak cukup untuk melakukan analisis TF-IDF.")

    # --- TAB 4: SVM MODELING ---
    with tab4:
        st.subheader("🧠 Support Vector Machine (SVM) Classification")
        st.markdown("Model akan dilatih menggunakan **Rating Asli** sebagai label kebenaran (Ground Truth).")
        st.info("Label: Rating 4-5 (Positif), 3 (Netral), 1-2 (Negatif)")

        if train_model_btn:
            with st.spinner("Melatih Model SVM... (Ini mungkin memakan waktu)"):
                # Persiapan Data
                X = df['Cleaned_Text']
                y = df['Label_True']
                
                # Split Data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=42)
                
                # Vectorization
                vectorizer = TfidfVectorizer(max_features=2000)
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                # SVM Training
                svm_model = SVC(kernel='linear')
                svm_model.fit(X_train_vec, y_train)
                
                # Prediksi
                y_pred = svm_model.predict(X_test_vec)
                
                # Simpan ke session state agar bisa dipakai di tab lain
                st.session_state['svm_model'] = svm_model
                st.session_state['vectorizer'] = vectorizer
                
                # --- HASIL EVALUASI ---
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Model berhasil dilatih! Akurasi: {acc:.2%}")
                
                col_eval1, col_eval2 = st.columns(2)
                
                with col_eval1:
                    st.write("**Confusion Matrix:**")
                    cm = confusion_matrix(y_test, y_pred, labels=['Negatif', 'Netral', 'Positif'])
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['Neg', 'Neu', 'Pos'], 
                                yticklabels=['Neg', 'Neu', 'Pos'])
                    st.pyplot(fig_cm)
                    
                with col_eval2:
                    st.write("**Classification Report:**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

        elif 'svm_model' in st.session_state:
            st.success("Model sudah tersimpan di memori.")
        else:
            st.warning("Klik tombol 'Latih Model SVM' di sidebar untuk memulai.")

    # --- TAB 5: PREDIKSI MANUAL ---
    with tab5:
        st.subheader("🔮 Coba Prediksi Sentimen")
        user_input = st.text_area("Masukkan kalimat ulasan produk:")
        
        if st.button("Prediksi") and user_input:
            # 1. Prediksi Lexicon
            clean_in = clean_text(user_input)
            lex_pred = get_lexicon_sentiment(clean_in)
            
            # 2. Prediksi SVM (Jika ada)
            svm_pred = "Model Belum Dilatih"
            if 'svm_model' in st.session_state:
                vec_in = st.session_state['vectorizer'].transform([clean_in])
                svm_pred = st.session_state['svm_model'].predict(vec_in)[0]
            
            col_res1, col_res2 = st.columns(2)
            col_res1.info(f"**Metode Lexicon/Rule:** {lex_pred}")
            if svm_pred == "Model Belum Dilatih":
                col_res2.warning(f"**Metode SVM:** {svm_pred}")
            else:
                col_res2.success(f"**Metode SVM:** {svm_pred}")

else:
    st.info("Silakan upload file CSV untuk memulai analisis.")