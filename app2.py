# ================== IMPORT LIBRARY ==================
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Hybrid Sentiment Analysis Tokopedia",
    layout="wide"
)

# ================== LOAD STEMMER ==================
@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

# ================== KAMUS SLANG ==================
slang_dict = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "tdk": "tidak",
    "ak": "aku", "aq": "aku", "sy": "saya",
    "bgt": "banget", "tp": "tapi", "klo": "kalau",
    "mantul": "mantap", "ok": "oke", "dtg": "datang"
}

# ================== LEXICON (PUNYA KAMU – TIDAK DIUBAH) ==================
positive_words = {
    "bagus","baik","cepat","rapi","aman","sesuai","mantap","puas",
    "oke","keren","nyaman","suka","awet","murah","ramah","lengkap",
    "lucu","halus","lembut","tebal","asli","original","recommended",
    "top","memuaskan","pas","cocok","adem","modis","trendy"
}

negative_words = {
    "jelek","buruk","lama","rusak","cacat","kecewa","salah","beda",
    "tipis","kasar","kotor","mahal","bohong","palsu","kw","robek",
    "retur","komplain","parah","nyesel","panas","gerah","gatal",
    "sempit","kebesaran","kekecilan","luntur","bau","kusut"
}

# ================== STOPWORDS ==================
stopwords = {
    "yang","dan","di","ke","dari","ini","itu","nya","aku","saya",
    "untuk","dengan","ada","karena","tapi","jadi","sudah","masih"
}

# ================== PREPROCESSING ==================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

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
    else:
        return "Netral"

# ================== HYBRID / FUSION LOGIC ==================
def hybrid_sentiment(text_sentiment, rating):
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

# ================== UI ==================
st.title("🤖 Hybrid / Fusion Sentiment Analysis Tokopedia")
st.caption("Menggabungkan Lexicon-based Sentiment + Rating + SVM")

uploaded_file = st.file_uploader("Upload CSV Tokopedia", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ================== FILTER KATEGORI ==================
    if "category" in df.columns:
        category = st.selectbox(
            "Filter Kategori",
            ["Semua"] + sorted(df["category"].dropna().unique().tolist())
        )
        if category != "Semua":
            df = df[df["category"] == category]

    text_col = st.selectbox("Kolom Teks Ulasan", df.columns)
    rating_col = st.selectbox("Kolom Rating", df.columns)

    # ================== PREPROCESSING ==================
    df["Cleaned_Text"] = df[text_col].astype(str).apply(clean_text)
    df["Lexicon_Sentiment"] = df["Cleaned_Text"].apply(lexicon_sentiment)
    df["Final_Sentiment"] = df.apply(
        lambda x: hybrid_sentiment(x["Lexicon_Sentiment"], x[rating_col]),
        axis=1
    )

    # ================== TABS ==================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard Hybrid",
        "🏆 Top Produk",
        "🔠 TF-IDF",
        "🧠 SVM & Evaluasi",
        "🔮 Prediksi Manual"
    ])

    # ================== TAB 1 ==================
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Ulasan", len(df))
        c2.metric("Rata-rata Rating", f"{df[rating_col].mean():.2f}")
        c3.metric(
            "Sentimen Positif",
            f"{(df['Final_Sentiment']=='Positif').mean()*100:.1f}%"
        )

        fig, ax = plt.subplots()
        df["Final_Sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    # ================== TAB 2 ==================
    with tab2:
        if "product_name" in df.columns:
            top = df["product_name"].value_counts().head(10)
            st.bar_chart(top)

    # ================== TAB 3 ==================
    with tab3:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df["Cleaned_Text"])
        feature_names = tfidf.get_feature_names_out()

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        top_words = tfidf_df.mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x=top_words.values, y=top_words.index, ax=ax)
        st.pyplot(fig)

    # ================== TAB 4 ==================
    with tab4:
        if st.button("Latih Model SVM"):
            X_train, X_test, y_train, y_test = train_test_split(
                df["Cleaned_Text"],
                df["Final_Sentiment"],
                test_size=0.2,
                random_state=42
            )

            vectorizer = TfidfVectorizer(max_features=2000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            svm = SVC(kernel="linear")
            svm.fit(X_train_vec, y_train)
            y_pred = svm.predict(X_test_vec)

            st.session_state["svm"] = svm
            st.session_state["vectorizer"] = vectorizer

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Akurasi Model: {acc:.2%}")

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

            st.dataframe(
                pd.DataFrame(
                    classification_report(y_test, y_pred, output_dict=True)
                ).transpose()
            )

    # ================== TAB 5 ==================
    with tab5:
        user_text = st.text_area("Masukkan ulasan produk")

        if st.button("Prediksi"):
            clean = clean_text(user_text)
            lex = lexicon_sentiment(clean)
            st.info(f"Hasil Lexicon: **{lex}**")

            if "svm" in st.session_state:
                vec = st.session_state["vectorizer"].transform([clean])
                pred = st.session_state["svm"].predict(vec)[0]
                st.success(f"Hasil SVM: **{pred}**")
            else:
                st.warning("Model SVM belum dilatih")

else:
    st.info("Upload file CSV untuk memulai analisis.")

