import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

st.set_page_config(
    page_title="Advanced Sentiment Analysis Tokopedia",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in custom_stopwords]
    return " ".join(words)

def get_lexicon_sentiment(text):
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

def hybrid_sentiment_logic(text_sentiment, rating):
    try:
        r = int(rating)
    except:
        return text_sentiment
    if text_sentiment == "Positif" and r <= 3:
        return "Netral"
    if text_sentiment == "Negatif" and r >= 4:
        return "Netral"
    if text_sentiment == "Netral":
        if r >= 4:
            return "Positif"
        elif r <= 2:
            return "Negatif"
    return text_sentiment

def get_rating_label(rating):
    try:
        r = int(rating)
    except:
        return "Netral"
    if r >= 4:
        return "Positif"
    elif r <= 2:
        return "Negatif"
    return "Netral"

st.title("🤖 Hybrid Sentiment Analysis Tokopedia")

uploaded_file = st.file_uploader("Upload CSV Tokopedia", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    text_col = st.selectbox("Kolom Teks", df.columns)
    rating_col = st.selectbox("Kolom Rating", df.columns)

    df["Cleaned_Text"] = df[text_col].astype(str).apply(clean_text)
    df["Lexicon_Sentiment"] = df["Cleaned_Text"].apply(get_lexicon_sentiment)
    df["Final_Sentiment"] = df.apply(
        lambda x: hybrid_sentiment_logic(x["Lexicon_Sentiment"], x[rating_col]),
        axis=1
    )
    df["Label_True"] = df[rating_col].apply(get_rating_label)

    tab1, tab2, tab3 = st.tabs(["Dashboard", "TF-IDF", "SVM & Evaluasi"])

    with tab1:
        st.metric("Total Ulasan", len(df))
        fig, ax = plt.subplots()
        df["Final_Sentiment"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    with tab2:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df["Cleaned_Text"])
        feature_names = tfidf.get_feature_names_out()
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        top_words = df_tfidf.mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x=top_words.values, y=top_words.index, ax=ax)
        st.pyplot(fig)

    with tab3:
        if st.button("Latih Model SVM"):
            X_train, X_test, y_train, y_test = train_test_split(
                df["Cleaned_Text"], df["Label_True"],
                test_size=0.2, random_state=42
            )

            vectorizer = TfidfVectorizer(max_features=2000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            svm = SVC(kernel="linear")
            svm.fit(X_train_vec, y_train)
            y_pred = svm.predict(X_test_vec)

            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

            cm = confusion_matrix(y_test, y_pred, labels=["Negatif","Netral","Positif"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d",
                        xticklabels=["Neg","Neu","Pos"],
                        yticklabels=["Neg","Neu","Pos"])
            st.pyplot(fig)

            st.dataframe(
                pd.DataFrame(
                    classification_report(y_test, y_pred, output_dict=True)
                ).transpose()
            )
else:
    st.info("Upload file CSV untuk memulai analisis.")
