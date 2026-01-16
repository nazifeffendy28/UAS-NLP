# ================== IMPORT LIBRARY ==================
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

# ================== PREPROCESSING ==================
def cleaning(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def case_folding(text):
    return text.lower()

def stopword_removal(text):
    STOPWORDS = {"yang","dan","di","ke","dari","ini","itu","saya","aku","nya","untuk"}
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

def preprocess_text(text):
    text = cleaning(text)
    text = case_folding(text)
    text = stopword_removal(text)
    return text

# ================== UI ==================
st.set_page_config(page_title="Analisis Sentimen Tokopedia", layout="wide")
st.title("📊 Analisis Sentimen Tokopedia (SVM + TF-IDF)")

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success(f"Data berhasil dimuat: {len(df)} baris")

    text_col = st.selectbox("Pilih kolom teks ulasan:", df.columns)
    rating_col = st.selectbox("Pilih kolom rating:", df.columns)

    # ================== PREPROCESS DATA ==================
    df["text_preprocessed"] = df[text_col].astype(str).apply(preprocess_text)

    # ================== LABEL SENTIMEN (BERDASARKAN RATING) ==================
    def rating_to_sentiment(r):
        if r >= 4:
            return "Positif"
        elif r <= 2:
            return "Negatif"
        else:
            return "Netral"

    df["sentiment"] = df[rating_col].apply(rating_to_sentiment)

    st.success("Preprocessing & pelabelan selesai")

    # ================== TAB ==================
    tab1, tab2 = st.tabs(["🤖 Model SVM", "🔮 Prediksi Baru"])

    # ================== TAB 1: MODEL SVM ==================
    with tab1:
        if st.button("🚀 Latih Model SVM"):
            # TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
            X = vectorizer.fit_transform(df["text_preprocessed"])
            y = df["sentiment"]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Model
            svm_model = SVC(kernel="linear")
            svm_model.fit(X_train, y_train)

            # Prediksi
            y_pred = svm_model.predict(X_test)

            # Simpan model
            st.session_state["svm_model"] = svm_model
            st.session_state["vectorizer"] = vectorizer

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Akurasi Model: {acc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            cr = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(cr).transpose())

            # ================== TF-IDF TOP FEATURES ==================
            st.subheader("Top 10 Fitur TF-IDF")

            feature_names = vectorizer.get_feature_names_out()
            scores = X.toarray().mean(axis=0)

            top_df = pd.DataFrame({
                "Fitur": feature_names,
                "Skor": scores
            }).sort_values("Skor", ascending=False).head(10)

            fig2, ax2 = plt.subplots()
            sns.barplot(data=top_df, x="Skor", y="Fitur", ax=ax2)
            st.pyplot(fig2)

    # ================== TAB 2: PREDIKSI BARU ==================
    with tab2:
        st.subheader("🔮 Prediksi Sentimen Ulasan Baru")

        if "svm_model" not in st.session_state:
            st.warning("Latih model terlebih dahulu di tab Model SVM")
        else:
            input_text = st.text_area("Masukkan ulasan:")

            if st.button("Prediksi"):
                processed = preprocess_text(input_text)
                X_new = st.session_state["vectorizer"].transform([processed])
                result = st.session_state["svm_model"].predict(X_new)[0]

                st.success(f"Hasil Prediksi Sentimen: **{result}**")
