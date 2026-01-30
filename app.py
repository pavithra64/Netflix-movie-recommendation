import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from recommender import FlixHub

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# ---------------------------------------------------
# Load & prepare data
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df["description"] = df["description"].fillna("")
    return df

@st.cache_resource
def build_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["description"])
    return tfidf_matrix

df = load_data()
tfidf_matrix = build_tfidf(df)
flixhub = FlixHub(df, tfidf_matrix)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🎬 Netflix Movie & TV Show Recommendation System")

st.write(
    "Get personalized movie and TV show recommendations "
    "based on content similarity using Machine Learning."
)

st.markdown("### 🎥 How it works")
st.info(
    "This system uses **TF-IDF vectorization** and "
    "**cosine similarity** to recommend similar content."
)

st.markdown("### 🔍 Search")
title = st.text_input(
    "Enter a Movie or TV Show title",
    placeholder="e.g. Blood & Water"
)

if st.button("🎯 Get Recommendations"):
    if title.strip() == "":
        st.warning("Please enter a title.")
    else:
        movies, tv_shows = flixhub.recommendation(title)

        if not movies and not tv_shows:
            st.error("No recommendations found.")
        else:
            if movies:
                st.subheader("🎬 Recommended Movies")
                for i, m in enumerate(movies, 1):
                    st.write(f"{i}. {m}")

            if tv_shows:
                st.subheader("📺 Recommended TV Shows")
                for i, t in enumerate(tv_shows, 1):
                    st.write(f"{i}. {t}")
