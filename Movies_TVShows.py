import requests
import pandas as pd
from fuzzywuzzy import process
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

base_url = "https://api.themoviedb.org/3/trending/all/day"
api_key = "f6******************************"

def fetch_data(page=1):
    params = {
        'api_key': api_key,
        'page': page
    }
    response = requests.get(base_url, params=params)
    try:
        movie_data = response.json()
        return movie_data
    except ValueError:
        return None

def fetch_all_data():
    all_data = []
    page = 1
    while True:
        movie_data = fetch_data(page)
        if movie_data is None:
            break
        results = movie_data.get("results", [])
        if not results:
            break
        all_data.extend(results)
        if page >= movie_data.get("total_pages", 0):
            break
        page += 1
    return pd.DataFrame(all_data)

def clean_data(all_movie_data):
    film_television = all_movie_data[['id', 'original_language', 'original_title', 'overview', 'poster_path', 'media_type', 'popularity', 'genre_ids', 'release_date']]
    film_television['original_title'].fillna(all_movie_data['original_name'], inplace=True)
    film_television['release_date'].fillna(all_movie_data['first_air_date'], inplace=True)
    cols_to_check = ['original_language', 'overview', 'poster_path', 'genre_ids']
    film_television.dropna(subset=cols_to_check, inplace=True)
    film_television.rename(columns={'original_title': 'Title'}, inplace=True)
    film_television.rename(columns={'original_language': 'Language'}, inplace=True)
    film_television.rename(columns={'poster_path': 'Image'}, inplace=True)
    language_codes={'en': 'English', 'zh': 'Chinese', 'ko': 'Korean', 'ja': 'Japanese', 'es': 'Spanish', 'fr': 'French', 'it': 'Italian', 'de': 'German', 'pl': 'Polish', 'fi': 'Finnish', 'is': 'Icelandic', 'hi': 'Hindi', 'pt': 'Portuguese', 'tr': 'Turkish', 'hr': 'Croatian', 'vi': 'Vietnamese', 'da': 'Danish', 'ml': 'Malayalam', 'th': 'Thai', 'uk': 'Ukrainian', 'cn': 'Cantonese', 'tl': 'Tagalog', 'te': 'Telugu', 'id': 'Indonesian', 'ru': 'Russian', 'ar': 'Arabic', 'nl': 'Dutch', 'fa': 'Farsi', 'ms': 'Malay', 'sv': 'Swedish', 'kn': 'Kannada', 'no': 'Norwegian', 'hu': 'Hungarian', 'sr': 'Serbian', 'ta': 'Tamil', 'cs': 'Czech', 'bs': 'Bosnian', 'gl': 'Galician', 'et': 'Estonian', 'he': 'Hebrew', 'kk': 'Kazakh', 'lt': 'Lithuanian', 'ca': 'Catalan', 'ku': 'Kurdish', 'pa': 'Punjabi', 'mk': 'Macedonian', 'bn': 'Bengali', 'el': 'Greek', 'af': 'Afrikaans', 'gu': 'Gujarati', 'ro': 'Romanian', 'eu': 'Basque', 'la': 'Latin', 'sk': 'Slovak'}
    film_television['Language'] = film_television['Language'].map(language_codes)
    film_television['genre_ids'] = film_television['genre_ids'].apply(lambda x: str(x[0]) if x else '')
    film_television['popularity'] = film_television['popularity'].astype(str)
    film_television['tags'] = film_television['overview'] + film_television['media_type'] + film_television['popularity'] + film_television['genre_ids']
    film_television['tags'] = film_television['tags'].apply(lambda x:x.split())
    film_television['tags'] = film_television['tags'].apply(lambda x: " ".join(x))
    film_television['tags'] = film_television['tags'].apply(lambda x:x.lower())
    
    ps = PorterStemmer()
    def stems(text):
        T = []
        for i in text.split():
            T.append(ps.stem(i))
        return " ".join(T)

    film_television['tags'] = film_television['tags'].apply(stems)
    
    return film_television

def get_recommendations(movie, film_television):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(film_television['tags']).toarray()
    similarity = cosine_similarity(vector)

    similar_titles = process.extract(movie, film_television['Title'], limit=5)
    closest_title = similar_titles[0][0]
    index = film_television[film_television['Title'] == closest_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = [film_television.iloc[i[0]] for i in distances[1:11]]
    
    return recommendations
