# ================== IMPORT LIBRARY ==================
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Hybrid Sentiment Analysis Tokopedia",
    layout="wide"
)

# ================== KAMUS SLANG (NORMALIZATION) ==================
slang_dict = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "tdk": "tidak",
    "ak": "aku", "aq": "aku", "sy": "saya",
    "bgt": "banget", "tp": "tapi", "klo": "kalau",
    "mantul": "mantap", "ok": "oke", "dtg": "datang"
}

# ================== LEXICON SENTIMENT (TIDAK DIUBAH) ==================
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

# ================== STOPWORD REMOVAL ==================
stopwords = {
    "yang","dan","di","ke","dari","ini","itu","nya","aku","saya",
    "untuk","dengan","ada","karena","tapi","jadi","sudah","masih"
}

# ================== PREPROCESSING (CLEANING, CASE FOLDING, NORMALIZATION) ==================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                                   # Case Folding
    text = re.sub(r"[^a-z\s]", " ", text)                 # Cleaning
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]         # Normalization
    words = [w for w in words if w not in stopwords]      # Stopword Removal
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

# ================== HYBRID LABELING LOGIC (SESUIAI PROPOSAL) ==================
def hybrid_sentiment(text_sentiment, rating):
    try:
        r = int(rating)
    except:
        return text_sentiment

    if text_sentiment == "Positif" and r <= 3:
        return "Netral"
    if text_sentiment == "Negatif" and r >= 4:
        return "Netral"

    return text_sentiment

# ================== UI ==================
st.title("📊 Hybrid Sentiment Analysis Tokopedia")
st.caption("Lexicon-based + Rating Fusion + SVM (Linear Kernel)")

uploaded_file = st.file_uploader(
    "Upload dataset CSV Tokopedia (teks, rating, nama produk, kategori, terjual)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ================== FILTER KATEGORI ==================
    if "kategori" in df.columns:
        kategori = st.selectbox(
            "Filter Kategori Produk",
            ["Semua"] + sorted(df["kategori"].dropna().unique())
        )
        if kategori != "Semua":
            df = df[df["kategori"] == kategori]

    text_col = st.selectbox("Kolom Teks Ulasan", df.columns)
    rating_col = st.selectbox("Kolom Rating", df.columns)

    # ================== PREPROCESSING ==================
    df["clean_text"] = df[text_col].astype(str).apply(clean_text)

    # ================== HYBRID LABELING ==================
    df["lexicon_sentiment"] = df["clean_text"].apply(lexicon_sentiment)
    df["final_sentiment"] = df.apply(
        lambda x: hybrid_sentiment(x["lexicon_sentiment"], x[rating_col]),
        axis=1
    )

    # ================== TABS ==================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔠 TF-IDF & Wordcloud",
        "🧠 SVM & Evaluasi",
        "🔮 Prediksi Manual"
    ])

    # ================== TAB 1: DASHBOARD ==================
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Ulasan", len(df))
        c2.metric("Rata-rata Rating", f"{df[rating_col].mean():.2f}")
        c3.metric(
            "Sentimen Positif (%)",
            f"{(df['final_sentiment']=='Positif').mean()*100:.1f}%"
        )

        fig, ax = plt.subplots()
        df["final_sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    # ================== TAB 2: TF-IDF & WORDCLOUD ==================
    with tab2:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df["clean_text"])
        feature_names = tfidf.get_feature_names_out()

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        top_words = tfidf_df.mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x=top_words.values, y=top_words.index, ax=ax)
        ax.set_title("Top Kata Berdasarkan TF-IDF")
        st.pyplot(fig)

        wc = WordCloud(
            width=800,
            height=300,
            background_color="white"
        ).generate(" ".join(df["clean_text"]))

        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # ================== TAB 3: SVM & EVALUASI ==================
    with tab3:
        if st.button("Latih Model SVM (80:20)"):
            X_train, X_test, y_train, y_test = train_test_split(
                df["clean_text"],
                df["final_sentiment"],
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

    # ================== TAB 4: PREDIKSI MANUAL ==================
    with tab4:
        user_text = st.text_area("Masukkan teks ulasan")

        if st.button("Prediksi Sentimen"):
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
