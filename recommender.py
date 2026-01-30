import re
from sklearn.metrics.pairwise import cosine_similarity

class FlixHub:
    def __init__(self, df, tfidf_matrix):
        self.df = df.reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix

    def find_id(self, title):
        for idx, name in enumerate(self.df["title"]):
            if re.search(title.lower(), name.lower()):
                return idx
        return None

    def recommendation(self, title, total_result=10):
        idx = self.find_id(title)
        if idx is None:
            return [], []

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        sim_indices = similarity_scores.argsort()[::-1][1:total_result+1]
        result_df = self.df.iloc[sim_indices]

        movies = result_df[result_df["type"] == "Movie"]["title"].tolist()
        tv_shows = result_df[result_df["type"] == "TV Show"]["title"].tolist()

        return movies, tv_shows
