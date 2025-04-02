from flask import Flask, render_template, request
import numpy as np
import gzip
import re
from gensim.models import KeyedVectors
import gensim.downloader as api
from itertools import combinations
import tempfile
from collections import defaultdict

app = Flask(__name__)

# Load models
model_google = api.load('word2vec-google-news-300')
model_glove = api.load('glove-wiki-gigaword-300')
model_wiki = api.load('fasttext-wiki-news-subwords-300')

gzipped_file_path = 'numberbatch-en-19.08.txt.gz'
with gzip.open(gzipped_file_path, 'rt', encoding='utf-8') as f_in:
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write(f_in.read())
        temp_file_path = temp_file.name
model_numberbatch = KeyedVectors.load_word2vec_format(temp_file_path, binary=False)

weights = {
    "google": 0.8366 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
    "glove": 0.6980 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
    "wiki": 1.2405 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
    "numberbatch": 1.8192 / (0.8366 + 0.6980 + 1.2405 + 1.8192),
}

models = {
    "google": model_google,
    "glove": model_glove,
    "wiki": model_wiki,
    "numberbatch": model_numberbatch
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

@app.route("/", methods=["GET", "POST"])
def home():
    suggestions = None
    if request.method == "POST":
        words = request.form["words"].split(",")
        words = [word.strip() for word in words]
        suggestions = aggregate_rankings(words, lives=100)
    return render_template("home.html", suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)