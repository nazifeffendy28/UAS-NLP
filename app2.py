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
st.set_page_config(
    page_title="Hybrid Sentiment Analysis Tokopedia",
    layout="wide"
)

# ================== KAMUS SLANG ==================
slang_dict = {
    "ga": "tidak", "gak": "tidak", "tdk": "tidak",
    "yg": "yang", "ak": "aku", "aq": "aku",
    "bgt": "banget", "tp": "tapi",
    "klo": "kalau", "ok": "oke", "mantul": "mantap"
}

# ================== STOPWORDS ==================
stopwords = {
    "yang","dan","di","ke","dari","ini","itu","nya",
    "aku","saya","untuk","dengan","ada","karena",
    "tapi","jadi","sudah","masih"
}

# ================== LEXICON ==================
positive_words = {
    "bagus","baik","cepat","rapi","aman","sesuai","mantap",
    "puas","oke","keren","nyaman","suka","awet","murah",
    "ramah","lengkap","halus","lembut","tebal","asli"
}

negative_words = {
    "jelek","buruk","lama","rusak","cacat","kecewa","salah",
    "beda","tipis","kasar","kotor","mahal","bohong","palsu",
    "robek","komplain","parah","nyesel","bau"
}

# ================== PREPROCESSING ==================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [slang_dict.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

# ================== LEXICON SENTIMENT ==================
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

# ================== HYBRID LABELING ==================
def hybrid_label(text_sentiment, rating):
    try:
        rating = int(rating)
    except:
        return text_sentiment

    if text_sentiment == "Positif" and rating <= 3:
        return "Netral"
    if text_sentiment == "Negatif" and rating >= 4:
        return "Netral"
    return text_sentiment

# ================== UI ==================
st.title("Hybrid Sentiment Analysis Tokopedia")

uploaded_file = st.file_uploader("Upload CSV Tokopedia", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    text_col = st.selectbox("Kolom Teks Ulasan", df.columns)
    rating_col = st.selectbox("Kolom Rating", df.columns)

    # ================== PREPROCESSING ==================
    df["clean_text"] = df[text_col].astype(str).apply(clean_text)

    # ================== HYBRID LABELING ==================
    df["lexicon_sentiment"] = df["clean_text"].apply(lexicon_sentiment)
    df["label"] = df.apply(
        lambda x: hybrid_label(x["lexicon_sentiment"], x[rating_col]),
        axis=1
    )

    # ================== DASHBOARD ==================
    tab1, tab2, tab3 = st.tabs([
        "Dashboard",
        "TF-IDF",
        "SVM & Evaluasi"
    ])

    with tab1:
        st.metric("Total Data", len(df))
        fig, ax = plt.subplots()
        df["label"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

        wc = WordCloud(
            width=800, height=400, background_color="white"
        ).generate(" ".join(df["clean_text"]))
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

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

    # ================== SVM ==================
    with tab3:
        if st.button("Latih Model SVM"):
            X_train, X_test, y_train, y_test = train_test_split(
                df["clean_text"],
                df["label"],
                test_size=0.2,
                random_state=42
            )

            vectorizer = TfidfVectorizer(max_features=2000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            svm = SVC(kernel="linear")
            svm.fit(X_train_vec, y_train)
            y_pred = svm.predict(X_test_vec)

            st.metric(
                "Akurasi",
                f"{accuracy_score(y_test, y_pred)*100:.2f}%"
            )

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

            report = pd.DataFrame(
                classification_report(
                    y_test, y_pred, output_dict=True
                )
            ).transpose()
            st.dataframe(report)

else:
    st.info("Upload file CSV untuk memulai.")
