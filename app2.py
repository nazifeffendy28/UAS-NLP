import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(page_title="Analisis Sentimen Tokopedia", layout="wide")

# ================== KAMUS SLANG ==================
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
    "seller": "penjual", "respon": "tanggapan", "bintang": "rating"
}

positive_words = {
    "bagus","baik","cepat","rapi","aman","sesuai","mantap","puas","oke","keren",
    "nyaman","suka","awet","murah","ramah","lengkap","halus","lembut","tebal",
    "asli","original","recommended","top","memuaskan","pas","cocok","adem",
    "modis","trendy"
}

negative_words = {
    "jelek","buruk","lama","rusak","cacat","kecewa","salah","beda","tipis",
    "kasar","kotor","mahal","bohong","palsu","robek","retur","komplain",
    "parah","nyesel","panas","gerah","gatal","sempit","luntur","bau","kusut"
}

custom_stopwords = {
    "yang","di","dan","itu","ini","dari","ke","untuk","dengan","nya",
    "saya","aku","kami","kita","bisa","ada","adalah","juga","karena",
    "tapi","atau","jadi","jika","kalau","sudah","lagi","akan"
}
# ================== PREPROCESSING ==================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Case Folding
    text = text.lower()

    # 2. Cleaning (hapus angka & simbol)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenisasi
    tokens = text.split()

    # 3. Normalization (slang → baku)
    tokens = [slang_dict.get(t, t) for t in tokens]

    # 4. Stopword Removal
    tokens = [t for t in tokens if t not in stopwords]

    return " ".join(tokens)

# ================== LEXICON SENTIMENT ==================
def lexicon_sentiment(text):
    score = 0
    for w in text.split():
        if w in positive_words:
            score += 1
        elif w in negative_words:
            score -= 1
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Negatif"
    return "Netral"

# ================== HYBRID LABELING ==================
def hybrid_label(sentiment, rating):
    try:
        rating = int(rating)
    except:
        return sentiment

    if sentiment == "Positif" and rating <= 3:
        return "Netral"
    if sentiment == "Negatif" and rating >= 4:
        return "Netral"

    return sentiment

# ================== LABEL UNTUK SVM ==================
def rating_label(rating):
    try:
        rating = int(rating)
    except:
        return "Netral"

    if rating >= 4:
        return "Positif"
    elif rating <= 2:
        return "Negatif"
    return "Netral"

# ================== UI ==================
st.title("Analisis Sentimen Ulasan Tokopedia")
st.markdown("Metode Kuantitatif – Hybrid Lexicon & SVM")

file = st.file_uploader("Upload dataset CSV Tokopedia", type=["csv"])

if file:
    df = pd.read_csv(file)

    text_col = st.selectbox("Kolom teks ulasan", df.columns)
    rating_col = st.selectbox("Kolom rating", df.columns)
    product_col = st.selectbox("Kolom nama produk", df.columns)

    # ================== PREPROCESSING ==================
    df["clean_text"] = df[text_col].astype(str).apply(preprocess_text)

    # ================== HYBRID LABEL ==================
    df["lexicon_sentiment"] = df["clean_text"].apply(lexicon_sentiment)
    df["final_sentiment"] = df.apply(
        lambda x: hybrid_label(x["lexicon_sentiment"], x[rating_col]),
        axis=1
    )

    # ================== LABEL SVM ==================
    df["svm_label"] = df[rating_col].apply(rating_label)

    tab1, tab2, tab3 = st.tabs([
        "Dashboard & Visualisasi",
        "TF-IDF & Word",
        "SVM & Evaluasi"
    ])

    # ================== DASHBOARD ==================
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ulasan", len(df))
        col2.metric("Rata-rata Rating", f"{df[rating_col].mean():.2f}")
        col3.metric("Sentimen Positif", f"{(df['final_sentiment']=='Positif').mean()*100:.1f}%")

        fig, ax = plt.subplots()
        df["final_sentiment"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

        wc = WordCloud(width=900, height=400, background_color="white") \
            .generate(" ".join(df["clean_text"]))
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        st.dataframe(df[[product_col, rating_col, "final_sentiment"]].head(10))

    # ================== TF-IDF ==================
    with tab2:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df["clean_text"])

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=tfidf.get_feature_names_out()
        )

        top_words = tfidf_df.mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x=top_words.values, y=top_words.index, ax=ax)
        st.pyplot(fig)

        st.dataframe(top_words)

    # ================== SVM & EVALUASI ==================
    with tab3:
        if st.button("Latih Model SVM"):
            X_train, X_test, y_train, y_test = train_test_split(
                df["clean_text"],
                df["svm_label"],
                test_size=0.2,   # 80:20
                random_state=42
            )

            vectorizer = TfidfVectorizer(max_features=2000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            model = SVC(kernel="linear")
            model.fit(X_train_vec, y_train)

            y_pred = model.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            st.metric("Akurasi", f"{acc*100:.2f}%")

            cm = confusion_matrix(
                y_test, y_pred,
                labels=["Negatif","Netral","Positif"]
            )

            fig, ax = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d",
                xticklabels=["Neg","Neu","Pos"],
                yticklabels=["Neg","Neu","Pos"]
            )
            st.pyplot(fig)

            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Silakan upload dataset CSV terlebih dahulu")
