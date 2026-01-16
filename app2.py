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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== KAMUS & LEXICON (SESUAI METODOLOGI) ==================
# Dictionary Normalisasi: mengubah kata gaul/slang menjadi kata baku (dictionary-based)
NORMALIZATION_DICT = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "engga": "tidak", "nggak": "tidak",
    "brg": "barang", "sdh": "sudah", "dgn": "dengan", "thx": "terima kasih",
    "tks": "terima kasih", "makasih": "terima kasih", "mksh": "terima kasih", "bgt": "banget",
    "kalo": "kalau", "kl": "kalau", "tp": "tapi", "dr": "dari",
    "bs": "bisa", "sy": "saya", "ak": "aku", "aq": "aku",
    "mantul": "mantap", "mantap": "bagus", "jos": "bagus", 
    "ok": "oke", "oke": "bagus", "good": "bagus", "best": "bagus",
    "jelek": "buruk", "parah": "buruk", "ancur": "buruk", "rusak": "buruk",
    "dtg": "datang", "sampe": "sampai", "nyampe": "sampai", "cepet": "cepat",
    "kirim": "pengiriman", "kurir": "pengiriman", "packing": "kemasan",
    "seller": "penjual", "respon": "tanggapan", "bintang": "rating",
    "gan": "juragan", "sis": "kakak", "kak": "kakak", "om": "paman",
    "admin": "penjual", "olshop": "toko online", "bhn": "bahan",
    "adem": "sejuk", "mlar": "melar", "size": "ukuran", "pas": "sesuai",
    "trmksh": "terima kasih", "thankyou": "terima kasih", "recommended": "rekomendasi",
    "recomended": "rekomendasi", "rekomend": "rekomendasi", "mantab": "mantap",
    "bgus": "bagus", "bgs": "bagus", "krn": "karena", "utk": "untuk",
    "buat": "untuk", "krm": "kirim", "pngrmn": "pengiriman", "cpt": "cepat",
    "cpet": "cepat", "lma": "lama", "telat": "terlambat", "mrh": "murah",
    "mhl": "mahal", "ori": "asli", "kw": "palsu", "fake": "palsu",
    "rsk": "rusak", "ssuai": "sesuai", "krg": "kurang", "byk": "banyak",
    "sgt": "sangat", "bener": "benar", "krenn": "keren", "gokil": "keren",
    "bngt": "banget", "bngd": "banget", "puass": "puas", "puasss": "puas",
    "kcwa": "kecewa", "jlek": "jelek", "jelk": "jelek", "bruk": "buruk"
}

# Lexicon Positif (untuk pelabelan berbasis leksikon)
POSITIVE_WORDS = {
    "bagus", "baik", "cepat", "rapi", "aman", "sesuai", "mantap", "puas", 
    "oke", "keren", "nyaman", "suka", "awet", "murah", "ramah", "lengkap",
    "lucu", "halus", "lembut", "tebal", "asli", "original", "rekomendasi",
    "top", "memuaskan", "berfungsi", "pas", "cocok", "sejuk",
    "adem", "dingin", "menyerap", "keringat", "elegan", "mewah", "premium",
    "rapih", "kilat", "gesit", "responsif", "sopan", "jujur", "amanah",
    "bonus", "hadiah", "terjangkau", "hemat", "diskon", "promo", "bersih",
    "wangi", "harum", "cantik", "ganteng", "kece", "modis", "trendy",
    "terima kasih", "thanks", "love", "senang", "sukses", "berkualitas",
    "recommended", "sempurna", "istimewa", "terbaik", "favorit", "worth",
    "gercep", "excellent", "seneng", "super", "juara", "aman", "terpercaya"
}

# Lexicon Negatif (untuk pelabelan berbasis leksikon)
NEGATIVE_WORDS = {
    "jelek", "buruk", "lambat", "lama", "rusak", "cacat", "pecah", "penyok",
    "kecewa", "salah", "beda", "tipis", "kasar", "kotor", "mahal", "bohong",
    "palsu", "kw", "robek", "bolong", "batal", "retur", "komplain", "parah",
    "nyesel", "kurang", "tidak", "panas", "gerah", "gatal", "sempit", 
    "longgar", "kebesaran", "kekecilan", "luntur", "pudar", "kusam",
    "bau", "ape", "lecek", "kusut", "benang", "jahitan", "lepas", "copot",
    "penipuan", "penipu", "lamban", "lelet", "jutek", "galak", "sombong",
    "ribet", "susah", "baret", "gores", "bekas", "mengecewakan",
    "hancur", "ampun", "zonk", "terrible", "awful", "worst", "bad",
    "mengecewakan", "terlambat", "slow", "najis", "sampah", "php",
    "berbeda", "error", "pecah", "sobek", "busuk", "horrible"
}

# Stopwords: kata umum yang tidak memiliki makna sentimen
STOPWORDS = {
    "yang", "di", "dan", "itu", "ini", "dari", "ke", "untuk", "dengan", "nya",
    "saya", "aku", "kami", "kita", "bisa", "ada", "adalah", "juga", "karena",
    "tapi", "namun", "atau", "jadi", "jika", "kalau", "sudah", "lagi", "akan",
    "pada", "masih", "saja", "yg", "dg", "rt", "dgn", "ny", "d", "k",
    "kalo", "biar", "bikin", "bilang", "gak", "ga", "krn", "nih",
    "sih", "si", "tau", "tdk", "tuh", "utk", "ya", "jd", "jgn", "sdh", 
    "aja", "n", "t", "nyg", "hehe", "pen", "u", "nan", "loh", "rt",
    "oleh", "se", "te", "an", "kan", "dia", "mereka", "ia", "telah",
    "sedang", "pernah", "belum", "bukan", "jangan", "bila", "maka",
    "walaupun", "meskipun", "agar", "supaya", "semoga", "dalam", "kepada",
    "terhadap", "antara", "tentang", "hingga", "sambil", "demi", "sebelum",
    "sesudah", "saat", "ketika", "sewaktu", "begitu", "seperti", "bagai",
    "ibarat", "umpama", "laksana", "seolah", "serupa", "macam", "secara",
    "setiap", "seluruh", "semua", "para", "sang", "si", "sri", "banget",
    "sekali", "sangat", "very", "nih", "deh", "dong", "kok",
    "sih", "lho", "kah", "pun"
}

# ================== FUNGSI PREPROCESSING BERTAHAP (SESUAI METODOLOGI) ==================
def cleaning(text):
    """
    Tahap 1 - Cleaning: menghapus tanda baca, angka, dan karakter non-alfanumerik
    Input: "Barang bagus!!! cepat sampai... thx 123"
    Output: "Barang bagus cepat sampai thx"
    """
    if not isinstance(text, str):
        return ""
    # Menghapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def case_folding(text):
    """
    Tahap 2 - Case Folding: mengubah seluruh teks menjadi huruf kecil
    Input: "Barang BAGUS Cepat Sampai"
    Output: "barang bagus cepat sampai"
    """
    return text.lower()

def normalization(text):
    """
    Tahap 3 - Normalization: mengubah kata gaul/slang menjadi kata baku menggunakan kamus dictionary-based
    Input: "brg bgus bgt, cpt sampe, thx"
    Output: "barang bagus banget cepat sampai terima kasih"
    """
    words = text.split()
    normalized = [NORMALIZATION_DICT.get(word, word) for word in words]
    return ' '.join(normalized)

def stopword_removal(text):
    """
    Tahap 4 - Stopword Removal: menghapus kata umum yang tidak memiliki makna sentimen
    Input: "barang yang bagus banget dan cepat sampai"
    Output: "barang bagus banget cepat sampai"
    """
    words = text.split()
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return ' '.join(filtered)

def preprocess_text_step_by_step(text):
    """
    Pipeline preprocessing BERTAHAP dengan menyimpan hasil setiap tahap
    Return: dictionary berisi hasil setiap tahapan
    """
    original = str(text)
    
    # Tahap 1: Cleaning
    step1_cleaning = cleaning(original)
    
    # Tahap 2: Case Folding
    step2_case_folding = case_folding(step1_cleaning)
    
    # Tahap 3: Normalization
    step3_normalization = normalization(step2_case_folding)
    
    # Tahap 4: Stopword Removal
    step4_stopword_removal = stopword_removal(step3_normalization)
    
    return {
        'original': original,
        'step1_cleaning': step1_cleaning,
        'step2_case_folding': step2_case_folding,
        'step3_normalization': step3_normalization,
        'step4_stopword_removal': step4_stopword_removal,
        'final': step4_stopword_removal
    }

def preprocess_text(text):
    """Pipeline preprocessing lengkap (untuk pemrosesan cepat)"""
    text = cleaning(text)
    text = case_folding(text)
    text = normalization(text)
    text = stopword_removal(text)
    return text

# ================== PELABELAN HYBRID APPROACH (SESUAI METODOLOGI) ==================
def lexicon_sentiment(text):
    """Penentuan label awal menggunakan pendekatan lexicon-based sentiment"""
    words = set(text.split())
    pos_count = len(words.intersection(POSITIVE_WORDS))
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    
    if pos_count > neg_count:
        return 'Positif'
    elif neg_count > pos_count:
        return 'Negatif'
    else:
        return 'Netral'

def hybrid_labeling(row, rating_col):
    """Validasi silang dengan rating untuk koreksi label (Hybrid Approach)"""
    lexicon_label = lexicon_sentiment(row['text_preprocessed'])
    
    try:
        rating = float(row[rating_col])
    except:
        return lexicon_label
    
    # Aturan koreksi berdasarkan rating
    # Jika sentimen leksikon = Positif dan rating ≤ 3 → label dikoreksi menjadi Netral
    if lexicon_label == 'Positif' and rating <= 3:
        return 'Netral'
    # Jika sentimen leksikon = Negatif dan rating ≥ 4 → label dikoreksi menjadi Netral
    elif lexicon_label == 'Negatif' and rating >= 4:
        return 'Netral'
    else:
        return lexicon_label

# ================== UI UTAMA ==================
st.title("
