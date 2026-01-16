import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================== KONFIGURASI ==================
st.set_page_config(
    page_title="Sentiment Analysis Tokopedia",
    layout="wide"
)

# ================== STEMMER ==================
@st.cache_resource
def load_stemmer():
    return StemmerFactory().create_stemmer()

stemmer = load_stemmer()

# ================== KAMUS ==================
slang_dict = {
    "ga": "tidak", "gak": "tidak", "yg": "yang",
    "bgt": "banget", "kalo": "kalau", "tdk": "tidak"
}

positive_words = {"bagus", "baik", "cepat", "mantap", "puas", "sesuai", "nyaman"}
negative_words = {"jelek", "buruk", "lama", "rusak", "kecewa", "cacat"}

custom_stopwords = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "saya", "aku", "nya", "tidak", "ada"
}

# ================== PREPROCESSING ==================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [slang_dict.get(w, w) for w in words if w not in custom_stopwords]
    return " ".join(words)

# ================== SENTIMEN ==================
def get_lexicon_sentiment(text):
    score = 0
    for w in text.split():
        if w in positive_words:
            score += 1
        elif w in negative_words:
            score -= 1
    return "Positif" if score > 0 else "Negatif" if score < 0 else "Netral"

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
st.title("📊 Analisis Sentimen Hybrid Tokopedia")

uploaded_file = st.file_uploader("Upload CSV Tokopedia", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    text_col = st.selectbox("Kolom Teks", df.columns)
    rating_col = st.selectbox("Kolom Rating", df.columns)

    df["Cleaned_Text"] = df[text_col].apply(clean_text)
    df["Lexicon"] = df["Cleaned_Text"].apply(get_lexicon_sentiment)
    df["Final_Sentiment"] = df.apply(
        lambda x: hybrid_sentiment(x["Lexicon"], x[rating_col]), axis=1
    )

    # ================== DASHBOARD ==================
    st.subheader("Distribusi Sentimen")
    fig, ax = plt.subplots()
    df["Final_Sentiment"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    # ================== WORDCLOUD ==================
    st.subheader("WordCloud")
    all_text = " ".join(df["Cleaned_Text"])
    if all_text.strip():
        wc = WordCloud(
            width=800, height=300, background_color="white"
        ).generate(all_text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.warning("Tidak cukup teks untuk WordCloud")

    # ================== TF-IDF ==================
    st.subheader("Top Kata TF-IDF")
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df["Cleaned_Text"])

    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    top_words = tfidf_df.mean().sort_values(ascending=False).head(10)

    fig_tfidf, ax_tfidf = plt.subplots()
    sns.barplot(x=top_words.values, y=top_words.index, ax=ax_tfidf)
    st.pyplot(fig_tfidf)

    # ================== SVM ==================
    st.subheader("Model SVM (Label Hybrid)")
    if st.button("Latih Model"):
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
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Akurasi Model: {acc:.2%}")
        st.dataframe(
            pd.DataFrame(
                classification_report(y_test, y_pred, output_dict=True)
            ).transpose()
        )

else:
    st.info("Upload dataset CSV untuk memulai.")
