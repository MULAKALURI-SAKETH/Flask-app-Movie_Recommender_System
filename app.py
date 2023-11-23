# Import necessary libraries 
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from wtforms import SelectField, StringField, SubmitField
from wtforms.validators import InputRequired

# Data Collection
movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
credits_df = pd.read_csv("data/tmdb_5000_credits.csv")

# Data Preprocessing
movies_df["title"] = movies_df["title"].drop_duplicates()
credits_df = credits_df.rename(columns={"movie_id": "id"})
merged_df = movies_df.merge(credits_df, on="id").drop("title_y", axis=1)
merged_df = merged_df.rename(columns={"title_x": "title"})
def string_to_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []
features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    merged_df[feature] = merged_df[feature].apply(literal_eval)
features = ["cast", "keywords", "genres"]
for feature in features:
    merged_df[feature] = merged_df[feature].apply(string_to_list)
# Removing unwanted characters from the attribute values
def data_cleaning(feature):
    if isinstance(feature, list):
        new_feature = [str.lower(v.replace(" ", "")) for v in feature]
        return new_feature
    else:
        if isinstance(feature, str):
            new_str = str.lower(feature.replace(" ", ""))
            return new_str
        else: return ""

# Sentiment Analysis
sa = SentimentIntensityAnalyzer()
merged_df["reviews"] = merged_df['keywords'].apply(lambda x: " ".join(x))
merged_df["sentiment"] = merged_df["reviews"].apply(sa.polarity_scores)
merged_df["sentiment"] = merged_df["sentiment"].apply(lambda x: "Yes" if x["compound"] >= 0.05 else "No" if x["compound"] <= -0.05 else "Neutral")
features = ["cast", "keywords"]
for feature in features:
    merged_df[feature] = merged_df[feature].apply(data_cleaning)
merged_df["genres"] = merged_df["genres"].apply(lambda x: " ".join(x))

merged_df = merged_df.reset_index()
movie_credits = merged_df[["title", "vote_average", "genres", "reviews", "sentiment"]]
movie_credits = movie_credits[movie_credits["genres"] != ""]
movie_credits.loc[:, 'genres'] = movie_credits.loc[:, 'genres'].apply(str.lower)

movie_credits1 = movie_credits[movie_credits["sentiment"] == "No"]
print(movie_credits1)
# Exploratory Data Analysis
Famous movies
famous_movies = merged_df.sort_values("popularity", ascending=False)
plt.figure(figsize=(12, 8))
plt.barh(famous_movies['title'].head(10), famous_movies["popularity"].head(10), color="red")
plt.gca().invert_yaxis()
plt.title("Famous movies")
plt.xlabel("Popularity")
plt.ylabel("Title")
plt.show()


# High budget movies
high_budget_movies = merged_df.sort_values("budget", ascending=False)
plt.figure(figsize=(12, 8))
plt.barh(high_budget_movies["title"].head(10), high_budget_movies["budget"].head(10), color="blue")
plt.gca().invert_yaxis()
plt.title("High Budget Movies")
plt.xlabel("Budget")
plt.ylabel("Title")
plt.show()

# Feature Extraction
tv = TfidfVectorizer()
genre_matrix = tv.fit_transform(movie_credits["genres"])
genre_sim = cosine_similarity(genre_matrix)

# Content based filtering method to recommend similar movies
def recommendations(movie_name, genre):
    # Find the index of the given movie in the merged dataset
    movie_index = movie_credits[(movie_credits['title'] == movie_name) & (movie_credits['genres'].str.contains(str(genre).lower()))].index[0]

    # Similarity scores for the given movie
    sim_scores = genre_sim[movie_index]

    # Sort the movies based on similarity scores
    sorted_indices = sim_scores.argsort()[::-1]

    # Retrieve top 10 similar movies (excluding the input movie itself)
    top_movies = sorted_indices[1:11]

    # Return the top 10 recommended movie titles
    recommended_movies = movie_credits.loc[top_movies, 'title']
    return recommended_movies

# K Nearest Neighbors algorithm for recommending movies
features = movie_credits[movie_credits["genres"] != ""].dropna()

pca = PCA(n_components=22)
reduced_features = pca.fit_transform(csr_matrix(genre_matrix).toarray())

knn_model = NearestNeighbors(n_neighbors=11, algorithm='brute', p=2)
knn_model.fit(reduced_features)


def get_recommendations(movie_title, genre):
    movie = movie_credits[(movie_credits['title'] == movie_title) & (movie_credits['genres'].str.contains(str(genre).lower()))]
    if movie["sentiment"].values == "No":
        return None
    movie = movie.index[0]
    movie_feature_pca = pca.transform(csr_matrix(genre_matrix).toarray()[movie].reshape(1, -1))
    _, indices = knn_model.kneighbors(movie_feature_pca.reshape(1, -1))
    similar_movies = features.iloc[indices[0]][0:][["title", "sentiment"]]
    similar_movies = similar_movies[similar_movies['title'] != movie_title]
    return similar_movies

app = Flask(__name__)
API_KEY = '36c656bed0e8f7ed54c5098b76fe866e'
app.config['SECRET_KEY'] = "Ms^&21Lxc()854"

class MovieForm(FlaskForm):
    movie_title = StringField("Movie Name", validators=[InputRequired()])
    movie_genre = SelectField("Genre", choices=[("Action", "Action"), ("Adventure", "Adventure"), ("Animation", "Animation"), ("Comedy", "Comedy"), ("Drama", "Drama"), ("Family", "Family"), ("Fantasy", "Fantasy"), ("Horror", "Horror"), ("Science Fiction", "Science Fiction")], validators=[InputRequired()])
    submit = SubmitField('Recommend')

def get_poster_path(movie_title):
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': API_KEY,
        'query': movie_title,
    }
    response = requests.get(base_url, params=params)
    data = dict(response.json())
    if (data != dict()) and 'results' in data.keys():
        movie_id = data["results"][0]['id']
        return get_poster_path_by_id(movie_id)
    else:
        print(f"No results found")
        return

def get_poster_path_by_id(movie_id):
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        'api_key': API_KEY,
    }

    response = requests.get(base_url, params=params)
    data = dict(response.json())
    if (data != dict()) and ('poster_path' in data.keys()):
        return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
    else:
        return ""


@app.route("/", methods=["GET", "POST"])
def home_page():
    movie_form = MovieForm()
    if request.method == "POST":
        if movie_form.validate_on_submit():
            movie_title = movie_form.movie_title.data
            genre = movie_form.movie_genre.data
            return redirect(url_for("movies_page", movie_title=movie_title, movie_genre=genre))
    return render_template("home.html", form=movie_form)

@app.route("/recommendations/<string:movie_title>?<string:movie_genre>")
def movies_page(movie_title, movie_genre):
    try:
        p_movie_posters, n_movie_posters = {}, {}
        similar_movies = get_recommendations(movie_title, movie_genre)        
        if similar_movies is None:
            return render_template("not_recommended.html", title=movie_title)
        p_movies = similar_movies[similar_movies["sentiment"] != "No"]["title"].values.tolist()
        n_movies = similar_movies[similar_movies["sentiment"] == "No"]["title"].values.tolist()
        for movie in p_movies:
            p_movie_posters[movie] = get_poster_path(movie)
        for movie in n_movies:
            n_movie_posters[movie] = get_poster_path(movie)
        return render_template("result.html", title="Result page", recommended_movies=p_movie_posters, negative_movies=n_movie_posters)
    except IndexError as e:
        return "The movie with given genre not found."

@app.errorhandler(500)
def servererror(error):
    return "<b style='font-size:40px;'>The movie with given genre not found</b>"

if __name__ == '__main__':
    app.run(debug=True)
