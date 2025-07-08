import os
from flask import Flask, render_template, request
import numpy as np
import re
from gensim.models import KeyedVectors
import gensim.downloader as api
from itertools import combinations
import time
import gzip
import json
from collections import defaultdict

app = Flask(__name__)

def load_models():
    global models 
    
    if not os.path.exists("numberbatch.bin"):
        print("Extracting numberbatch.bin at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " ...")
        gzipped_file_path = 'numberbatch-en-19.08.txt.gz'
        txt_output_path = 'numberbatch-en-19.08.txt'

        with gzip.open(gzipped_file_path, 'rt', encoding='utf-8') as f_in:
            with open(txt_output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())

        model = KeyedVectors.load_word2vec_format(txt_output_path, binary=False)
        model.save("numberbatch.bin")
        os.remove(txt_output_path)
        print("numberbatch.bin extracted and saved.")
        
    print("Loading models at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " ...")

    model_google = api.load('word2vec-google-news-300')
    model_glove = api.load('glove-wiki-gigaword-300')
    model_wiki = api.load('fasttext-wiki-news-subwords-300')
    model_numberbatch = KeyedVectors.load("numberbatch.bin")

    models = {
        "google": model_google,
        "glove": model_glove,
        "wiki": model_wiki,
        "numberbatch": model_numberbatch
    }
    
    print("Models loaded at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



weights = {
    "google": 0.8366 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
    "glove": 0.6980 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
    "wiki": 1.2405 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
    "numberbatch": 1.8192 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
}

# Preprocess words
def preprocess_word(word, model):
    """
    Preprocess multi-word expressions (MWE) for accomodation by word2vec models.

    Args:
        word (str): The word to preprocess.
        model (gensim.models.word2vec): The word2vec model to check for MWE.

    Returns:
        str: The preprocessed word.
    """
    mwe = re.sub(r'[-\s]', '_', word.lower())
    
    if mwe not in model:
        mwe = re.sub(r'_', '', mwe)
    
    return mwe

def compute_similarity_matrix(model, words):
    words = [preprocess_word(word, model) for word in words]
    words = [word for word in words if word in model]
    
    similarity_matrix = {}
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i < j:  # Avoid redundant computations
                similarity_matrix[(word1, word2)] = model.similarity(word1, word2)
    return similarity_matrix

def guess_best_combination(model, words, similarity_matrix=None, lives=4):
    if len(words) == 4:
        return [list(words) * lives]
    words = [preprocess_word(word, model) for word in words]
    words = [word for word in words if word in model]

    if len(words) < 4 or lives < 1:
        return None

    if similarity_matrix is None:
        similarity_matrix = compute_similarity_matrix(model, words)

    all_combinations = list(combinations(words, 4))
    scored_combinations = []

    for combination in all_combinations:
        similarities = []
        for i, word1 in enumerate(combination):
            for j, word2 in enumerate(combination):
                if i < j:
                    similarities.append(similarity_matrix.get((word1, word2), similarity_matrix.get((word2, word1), 0)))

        average_similarity = np.mean(similarities)
        scored_combinations.append((combination, average_similarity))

    # Sort combinations by average similarity in descending order
    scored_combinations.sort(key=lambda x: x[1], reverse=True)

    # Return up to four attempts in descending order of similarity
    top_guesses = [list(comb[0]) for comb in scored_combinations[:lives]]
    return top_guesses

def aggregate_rankings(words, lives=4):
    ranking_scores = defaultdict(float)

    for name, model in models.items():
        model_guesses = guess_best_combination(model, words, similarity_matrix=None, lives=lives)
        if model_guesses:
            for rank, guess in enumerate(model_guesses):
                ranking_scores[tuple(guess)] += weights[name] / (rank + 1)

    sorted_guesses = sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
    return [list(guess[0]) for guess in sorted_guesses[:lives]]

def read_game_stats(filename):
    with open("stats/" + filename, "r") as file:
        data = json.load(file)

    games = []
    for game in data["games"]:
        game_str = f'<div class="game-title"> {game["name"].capitalize()} </div>'
        game_str += ", ".join(game["words"]) + '<br><br>'
        game_str += '<div class="game-title">Guesses</div>'
        for guess in game["guesses"]:
            group_words = ", ".join(guess["words"])
            emoji = "Invalid" if guess["invalid"] else ("✅" if guess["correct"] else "❌")
            
            formatted_line = (
                '<div style="display: flex; justify-content: space-between; margin: 5px 0;">'
                f'<span style="text-align: left;">{group_words}</span>'
                f'<span style="text-align: right;">{emoji}</span>'
                '</div>'
            )

            game_str += formatted_line
            
        game_str += '<br><div class="game-title">Answers</div>'
        for answer in game["answers"]:
            group_words = ", ".join(answer["words"])
            difficulty = "🟨" 
            if answer["difficulty"] == 1:
                difficulty = "🟩"
            elif answer["difficulty"] == 2:
                difficulty = "🟦"
            elif answer["difficulty"] == 3:
                difficulty = "🟪"
            category = answer["category"].capitalize()
            
            formatted_line = (
                '<div style="display: flex; justify-content: space-between; margin: 5px 0;">'
                f'<span style="text-align: left;">{group_words}</span>'
                f'<span style="text-align: right;">{difficulty} {category}</span>'
                '</div>'
            )

            game_str += formatted_line
            
        games.append(game_str)
    return games

@app.route("/", methods=["GET", "POST"])
def home():
    suggestions = None
    if request.method == "POST":
        words = request.form["words"].split(",")
        exclusions = request.form["exclusions"].split(",")
        words = [word.strip() for word in words]
        exclusions = [word.strip() for word in exclusions]
        suggestions = aggregate_rankings(words, lives=1000)
        suggestions = [list(filter(lambda x: x not in exclusions, suggestion)) for suggestion in suggestions]
        suggestions = [suggestion for suggestion in suggestions if len(suggestion) == 4]
    return render_template("home.html", suggestions=suggestions)

@app.route("/model1", methods=["GET", "POST"])
def model1():
    games = read_game_stats("gpt4o.json")
    return render_template("statistics.html", games=games, model_name="GPT-4o")

@app.route("/model2", methods=["GET", "POST"])
def model2():
    games = read_game_stats("numberbatch.json")
    return render_template("statistics.html", games=games, model_name="Numberbatch")

@app.route("/model3", methods=["GET", "POST"])
def model3():
    games = read_game_stats("method3.json")
    return render_template("statistics.html", games=games, model_name="model3")

@app.route("/model4", methods=["GET", "POST"])
def model4():
    games = read_game_stats("method4.json")
    return render_template("statistics.html", games=games, model_name="model4")

@app.route("/model5", methods=["GET", "POST"])
def model5():
    games = read_game_stats("method5.json")
    return render_template("statistics.html", games=games, model_name="model5")

if __name__ == "__main__":
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        load_models()
    app.run(debug=True)