import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from wordcloud import WordCloud

# =======================
# KONFIGURASI HALAMAN
# =======================
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia",
    layout="wide"
)

# =======================
# KAMUS & LEXICON (SESUIAI PROPOSAL)
# =======================
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
    "gan": "juragan", "sis": "kakak", "kak": "kakak",
    "admin": "penjual", "olshop": "toko online",
    "bhn": "bahan", "adem": "sejuk", "size": "ukuran", "pas": "sesuai"
}

positive_words = {
    "bagus", "baik", "cepat", "rapi", "aman", "sesuai", "mantap", "puas",
    "oke", "keren", "nyaman", "suka", "awet", "murah", "ramah",
    "halus", "lembut", "tebal", "original", "recommended", "top",
    "memuaskan", "cocok", "sejuk", "adem", "elegan", "modis"
}

negative_words = {
    "jelek", "buruk", "lambat", "lama", "rusak", "cacat", "kecewa",
    "tipis", "kasar", "kotor", "mahal", "palsu", "robek", "bolong",
    "parah", "nyesel", "panas", "gerah", "sempit", "luntur", "bau"
}

custom_stopwords = {
    "yang", "di", "dan", "itu", "ini", "dari", "ke", "untuk", "dengan",
    "saya", "aku", "kami", "kita", "ada", "adalah", "juga", "karena",
    "tapi", "atau", "jadi", "jika", "kalau", "sudah", "lagi", "akan"
}

# =======================
# PREPROCESSING
# =======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in custom_stopwords]
    return " ".join(words)

def lexicon_sentiment(text):
    score = 0
    for word in text.split():
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Negatif"
    else:
        return "Netral"

def hybrid_label(lex_sent, rating):
    if lex_sent == "Positif" and rating <= 3:
        return "Netral"
    if lex_sent == "Negatif" and rating >= 4:
        return "Netral"
    return lex_sent

def rating_label(r):
    if r >= 4:
        return "Positif"
    elif r <= 2:
        return "Negatif"
    else:
        return "Netral"

# =======================
# UI
# =======================
st.title("Analisis Sentimen Ulasan Produk Tokopedia")
st.markdown("Pendekatan Hybrid (Lexicon + Rating) dan Support Vector Machine")

uploaded_file = st.file_uploader("Upload dataset CSV Tokopedia", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    text_col = st.selectbox("Pilih kolom teks ulasan", df.columns)
    rating_col = st.selectbox("Pilih kolom rating", df.columns)
    category_col = st.selectbox("Pilih kolom kategori", df.columns)

    kategori = st.selectbox("Filter kategori", ["Semua"] + list(df[category_col].unique()))
    if kategori != "Semua":
        df = df[df[category_col] == kategori]

    df["clean_text"] = df[text_col].apply(clean_text)
    df["lexicon_sentiment"] = df["clean_text"].apply(lexicon_sentiment)
    df["hybrid_sentiment"] = df.apply(
        lambda x: hybrid_label(x["lexicon_sentiment"], x[rating_col]), axis=1
    )
    df["label_true"] = df[rating_col].apply(rating_label)

    tab1, tab2, tab3 = st.tabs(["Dashboard", "TF-IDF & Wordcloud", "SVM & Evaluasi"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Ulasan", len(df))
        c2.metric("Rata-rata Rating", round(df[rating_col].mean(), 2))
        c3.metric("Sentimen Positif (%)",
                  round((df["hybrid_sentiment"] == "Positif").mean() * 100, 2))

        fig, ax = plt.subplots()
        df["hybrid_sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        st.pyplot(fig)

    with tab2:
        tfidf = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf.fit_transform(df["clean_text"])
        feature_names = tfidf.get_feature_names_out()
        avg_tfidf = np.mean(X_tfidf.toarray(), axis=0)
        top_words = pd.Series(avg_tfidf, index=feature_names).sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x=top_words.values, y=top_words.index, ax=ax)
        st.pyplot(fig)

        text_all = " ".join(df["clean_text"])
        wc = WordCloud(width=800, height=300, background_color="white").generate(text_all)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    with tab3:
        test_size = st.slider("Rasio data uji (%)", 10, 40, 20) / 100
        if st.button("Latih Model SVM"):
            X = df["clean_text"]
            y = df["label_true"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            vectorizer = TfidfVectorizer(max_features=2000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            svm = SVC(kernel="linear")
            svm.fit(X_train_vec, y_train)

            y_pred = svm.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            st.metric("Akurasi Model", round(acc * 100, 2))

            cm = confusion_matrix(y_test, y_pred, labels=["Negatif", "Netral", "Positif"])
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d",
                        xticklabels=["Neg", "Neu", "Pos"],
                        yticklabels=["Neg", "Neu", "Pos"],
                        ax=ax_cm)
            st.pyplot(fig_cm)

            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")
