# Import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import InputRequired
import requests
from ast import literal_eval
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# Data Collection
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Data Preprocessing
movies["title"] = movies["title"].drop_duplicates()
credits = credits.rename(columns={"movie_id": "id"})

merged_df = movies.merge(credits, on="id").drop("title_y", axis=1)
merged_df = merged_df.rename(columns={"title_x": "title"})


def fetch_director(x):
    '''This function fetches the director names from
        the crew attribute from the dataset'''
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return ""


def string_to_list(x):
    '''Create a list consisting of names of the actors
        of the particular movies'''
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    merged_df[feature] = merged_df[feature].apply(literal_eval)
merged_df["director"] = merged_df["crew"].apply(fetch_director)
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
        else:
            return ""


features = ["cast", "keywords", "director"]
for feature in features:
    merged_df[feature] = merged_df[feature].apply(data_cleaning)
merged_df["genres"] = merged_df["genres"].apply(lambda x: " ".join(x))

merged_df = merged_df.reset_index()
indices = pd.Series(merged_df.index, index=merged_df["title"])

movie_credits = merged_df[["title", "vote_average", "genres"]]
movie_credits = movie_credits[movie_credits["genres"] != ""]
movie_credits.loc[:, 'genres'] = movie_credits.loc[:, 'genres'].apply(str.lower)

# Exploratory Data Analysis
# Famous movies
mostly_watched = merged_df.sort_values("popularity", ascending=False).head(10)["title"].values
print(mostly_watched)

# High budget movies
high_b = merged_df.sort_values("budget", ascending=False).head(10)['title'].values
print(high_b)

# Feature Extraction
tv = TfidfVectorizer()
genre_matrix = tv.fit_transform(movie_credits["genres"])
genre_sim = cosine_similarity(genre_matrix)
# print(genre_sim)


# Content based filtering method to recommend similar movies
def recommendations(movie_name, genre):
    # Find the index of the given movie in the merged dataset
    movie_index = movie_credits[
        (movie_credits['title'] == movie_name) & (movie_credits['genres'].str.contains(str(genre).lower()))].index[0]

    # Obtain similarity scores for the given movie
    sim_scores = genre_sim[movie_index]

    # Sort the movies based on similarity scores
    sorted_indices = sim_scores.argsort()[::-1]

    # Retrieve top 10 similar movies (excluding the input movie itself)
    top_movies = sorted_indices[1:11]

    # Return the top 10 recommended movie titles
    recommended_movies = movie_credits.loc[top_movies, 'title']
    return recommended_movies


recommended_movies = recommendations("The Martian", "Science Fiction")
print(recommended_movies)

# # K Nearest Neighbors algorithm
features = movie_credits[movie_credits["genres"] != ""]
features = features.dropna()
tv = TfidfVectorizer()
feature_matrix = tv.fit_transform(features['genres'])

pca = PCA(n_components=22)
reduced_features = pca.fit_transform(csr_matrix(feature_matrix).toarray())

knn_model = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine')
knn_model.fit(reduced_features)



def get_recommendations(movie_title, genre):
    movie = movie_credits[(movie_credits['title'] == movie_title) & (movie_credits['genres'].str.contains(str(genre).lower()))].index[0]
    movie_feature_pca = pca.transform(csr_matrix(feature_matrix).toarray()[movie].reshape(1, -1))
    _, indices = knn_model.kneighbors(movie_feature_pca.reshape(1, -1))
    recommendations = features.iloc[indices[0]][:10]["title"].values
    return recommendations


app = Flask(__name__)
app.config['SECRET_KEY'] = 'cba0e071eb22673dbd11de86c743e064'

class MovieForm(FlaskForm):
    movie_title = StringField("Movie Name", validators=[InputRequired()])
    movie_genre = SelectField("Genre", choices=[("selected", "Choose..."), ("1", "Action"), ("2", "Adventure"), ("3", "Comedy"), ("4", "Drama"), ("5", "Fantasy"), ("6", "Science Fiction"), ("7", "Horror")], validators=[InputRequired()])
    submit = SubmitField('Recommend')

def get_poster_path(movie_title):
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': app.secret_key,
        'query': movie_title,
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if 'results' in data and data["results"]:
        movie_id = data["results"][0]["id"]
        return get_poster_path_by_id(movie_id)
    else:
        print(f"No results found")
        return
    
def get_poster_path_by_id(movie_id):
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        'api_key': app.secret_key,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["poster_path"]:
        return f"https://image.tmdb.org/t/b/w500{data['poster_path']}"
    else:
        return


@app.route("/", methods=["GET", "POST"])
def home_page():
    movie_form = MovieForm()
    movie_posters = {}
    
    if movie_form.validate_on_submit():
        movie_title = movie_form.movie_title.data
        genre = movie_form.movie_genre.data
        recommendations = get_recommendations(movie_title, genre).values
        for movie in recommendations:
            movie_posters[movie] = get_poster_path(movie)

        return redirect(url_for("recommendations", movies=movie_posters))

    return render_template("home.html", title="Movie Magic", form=movie_form)


@app.route("/recommendations",  methods=["POST"])
def movies_page():
    movies = request.args.getlist("movies")
    if not movies:
        return redirect(url_for("home_page"))

    return render_template("result.html", title="Movie Magic", movie_recommendations=movies)


if __name__ == '__main__':
    app.run(debug=True)
