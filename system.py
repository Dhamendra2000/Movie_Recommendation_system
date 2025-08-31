import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
# import these two data set
movies=pd.read_csv('tmdb_5000_movies.csv')
credit=pd.read_csv('tmdb_5000_credits .csv')
# print(movies.head(1))
# print(credit.head(1))
# i will make one dataset merging them these two dataset  because it is difficult to use these dataset separately
merged_data = movies.merge(credit, on='title')
# print(merged_data.head())
# i will remove some feature in the dataset which is not usable for me
# this is the content based recommender system
'''
i will keep these feature for furthure use 
otherwise i will remove other feature
1.genres
2.id
3.keywords
4.title
5.overview
6.cast
7.crew
'''
# below i have done like select all the usable feature and made efficient dataset for this project and store this in movie and this is the final datset
movies=merged_data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# print(merges_data.columns)  this will show about all the features in the merged dataset
# print(movies.columns)  this will show about all the selecete features in the dataset
# ahead i will work on this dataset
# i have to make a new dataframe  which contains three things 'movie_id','movie_title', and 'tag'
# tag is made by merging the overview genres keywords cast and crew
# we need some data preprocessing it measn remove missing datapoint and als remove duplicates datapoint clean the dataseet for better model
# print(movies.isnull().sum()) this is for checking the missing data
#  i have to that movies which does not have overview information
movies.dropna(inplace=True)  
# print(movies.isnull().sum())this is to check the missing information is removed or not

# now i will check for duplicate row 
movies.duplicated().sum
# print(movies.iloc[0].genres)
# genres is so  beared format so i have to change into another fromat like { action,adventure,fantasy,science scinecefiction}
# for this i will create one helper function which name is convert
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L 
movies['genres'] = movies['genres'].apply(convert)
# print(movies.head())
movies['keywords'] = movies['keywords'].apply(convert)
# print(movies.head())
# in cast so many things are there so  i need only three things and i will remove all the things
# we need actual name of actor
# belowe function is used to append the required information from the movie dataset otherwise i will remove the remainig information
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L 
movies['cast']=movies['cast'].apply(convert3)
#  we need only that dictionary wher job is equal to dictionary after that i will remove all other dictionary
# for this i will use again one helper function

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew']=movies['crew'].apply(fetch_director)
# print(movies.head())
# overview column is string and its need to convert into string so that we can concatenate with otherr string
movies['overview']=movies['overview'].apply(lambda x:x.split())
# pd.set_option('display.max_columns', None)
# print(movies.head()) this is used to print all the column of tje dataset
# now i need some transformation because  if you keep these words separately then it will create some problem 
# for this i will write this code
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
# print(movies['genres']) this is for printing the column genres of movie
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x]) 
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# pd.set_option('display.max_columns', None)
# print(movies.head())
# now i will create one new column and after cooncating all the column put into that tag column
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
# pd.set_option('display.max_columns', None)
# print(movies.head())
# now we dont need of remainning column we can use only one column that is tag
# we will make new dataframe 
new_df=movies[['movie_id','title','tags']]
# print(new_df)
# now i will convert tag list into a string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
# print(new_df.head())
# now i will convert all the string into lowercase letterr
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
#  now i will apply saming here saming does suppose we have loving ,loved and lover then saming these all words into love
ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags']=new_df['tags'].apply(stem)
# print(new_df.head())
#now i have to text vectorization 
#  i will use bagoff words to convert this 
# print(new_df['tags'][0])
# below code is used to remvoe the stopword in tag string
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
# print(vectors.shape) 
print(cv.get_feature_names_out()[:100])
#  we have to calculate distance of the movire with one another and distance is maximum will show less similarities
# we can calculate cosine distance/
# if angel is small then  we can say it is very close
similarity=cosine_similarity(vectors)
print(similarity)
#  i will sort the data based on the movie index
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
# now i will create one recommend function in which i will give movie name then i will return 5 similar movie to me
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity(movie_index)
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# now i have to send the movies list to website file 
#  for this i will use pickle
pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))




    
