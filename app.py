from flask import Flask, render_template, request
from Movies_TVShows import fetch_all_data, clean_data, get_recommendations
from googletrans import Translator

app = Flask(__name__)

all_movie_data = fetch_all_data()
film_television = clean_data(all_movie_data)

def translate_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text).text
    return translated_text

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = get_recommendations(movie, film_television)
    return render_template("index.html", recommendations=recommendations, translate_to_english = translate_to_english)

if __name__ == "__main__":
    app.run(debug=True)
