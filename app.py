import streamlit as st
import pickle
from recommender import FlixHub

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# ---------------------------------------------------
# Load data (cached for performance)
# ---------------------------------------------------
@st.cache_data
def load_artifacts():
    with open("final_data.pkl", "rb") as f:
        df = pickle.load(f)

    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)

    return df, cosine_sim


df, cosine_sim = load_artifacts()

# Initialize recommender
flixhub = FlixHub(df, cosine_sim)

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
    "This recommendation system uses **TF-IDF vectorization** and "
    "**cosine similarity** to recommend movies and TV shows based on content."
)

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
st.markdown("### 🔍 Search")
movie_name = st.text_input(
    "Enter a Movie or TV Show title",
    placeholder="e.g. Chappie"
)

# ---------------------------------------------------
# Recommendation Button
# ---------------------------------------------------
if st.button("🎯 Get Recommendations"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie or TV show name.")
    else:
        try:
            movies, tv_shows = flixhub.recommendation(
                movie_name, total_result=10
            )

            if not movies and not tv_shows:
                st.error("No recommendations found. Try another title.")
            else:
                if movies:
                    st.subheader("🎬 Recommended Movies")
                    for i, movie in enumerate(movies, 1):
                        st.write(f"{i}. {movie}")

                if tv_shows:
                    st.subheader("📺 Recommended TV Shows")
                    for i, show in enumerate(tv_shows, 1):
                        st.write(f"{i}. {show}")

        except Exception as e:
            st.error("Something went wrong while generating recommendations.")
