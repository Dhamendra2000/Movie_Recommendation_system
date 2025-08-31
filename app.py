import streamlit as st
import pickle
import requests

def fetch_poster(movie_title):
    api_key = "b65b7446"
    url = f"https://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    data = requests.get(url).json()
    return data.get('Poster', "https://via.placeholder.com/200x300?text=No+Image")

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_titles = []
    recommended_posters = []

    for i in movie_list:
        movie_title = new_df.iloc[i[0]].title
        recommended_titles.append(movie_title)
        recommended_posters.append(fetch_poster(movie_title))

    return recommended_titles, recommended_posters

# Load data
new_df = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies_list = new_df['title'].values

# Streamlit UI
st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select a movie to get recommendations:',
    movies_list
)

if st.button('Recommend'):
    titles, posters = recommend(selected_movie_name)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"<img src='{posters[i]}' style='height:200px; display:block; margin:auto;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:14px'>{titles[i]}</p>", unsafe_allow_html=True)
