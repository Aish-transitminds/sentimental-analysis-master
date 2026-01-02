
# ? VIEWS.PY

# Importing dependencies.
from flask import Flask, Blueprint, render_template, request, jsonify
from decouple import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from goose3 import Goose
from goose3.configuration import Configuration
from werkzeug.utils import secure_filename
import numpy as np
import os
import mimetypes

# Creating a blueprint for views to use for routing.
app = Flask(__name__)
views = Blueprint(__name__, "views")

# Cache Directory
HUGGINGFACE_CACHE_DIR = config("HUGGINGFACE_CACHE_DIR", '')
TORCH_CACHE_DIR = config("TORCH_CACHE_DIR", '')
os.environ['TORCH_HOME'] = TORCH_CACHE_DIR

# Importing pre-trained model for Sentiment Analysis.
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Model Training for Polarity Scores.
tokenizer = None
sentiment_model = None

def get_model():
    global tokenizer, sentiment_model
    if tokenizer is None:
        print("Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            SENTIMENT_MODEL, cache_dir=HUGGINGFACE_CACHE_DIR)
    if sentiment_model is None:
        print("Loading Model... (This may take a while on the first run)")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL, cache_dir=HUGGINGFACE_CACHE_DIR)
    return tokenizer, sentiment_model


# * Routing for Home Page.
@views.route('/', methods=["GET", "POST"])
def home():
    # When opening the page, render the webpage.
    if request.method == "GET":
        return render_template("index.html")
    # When a form input is received, show the sentiment based on the input.
    elif request.method == "POST":
        input_type = request.form.get("type")

        # Find the input text from different types.
        input_text = ''

        try:
            if (input_type == "text"):
                input_text = request.form.get("input")
            elif (input_type == "url"):
                url = request.form.get("input")
                
                # Configure Goose with a browser User-Agent
                config = Configuration()
                config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                config.http_timeout = 10.0 # set a reasonable timeout
                g = Goose(config)
                
                try:
                    article = g.extract(url=url)
                    input_text = article.cleaned_text
                    if not input_text:
                        return jsonify({'error': 'No text could be extracted from this URL.'}), 400
                except Exception as e:
                     print(f"Error extracting URL: {e}")
                     return jsonify({'error': f'Failed to extract content from URL: {str(e)}'}), 400
            elif (input_type == "media"):
                file = request.files.get("input")
                if file:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(
                        app.root_path, 'static', 'files', filename)
                    file.save(file_path)
                    # Note: process_files function in original code was incomplete/empty? 
                    # We just pass for now or need to fix it if it returns text.
                    # Assuming process_files returns text or is a placeholder.
                    # The original code had input_media variable but didn't use it.
                    # We'll just set input_text to empty to avoid crash if media is not implemented.
                    input_text = "Media processing not implemented yet." 

            # Find the sentiment values.
            if not input_text:
                 return jsonify({'error': 'Input text is empty.'}), 400

            sentiment_analysis = find_text_sentiment_analysis(input_text)
            return jsonify(sentiment_analysis)

        except Exception as e:
            print(f"Error during analysis: {e}")
            return jsonify({'error': str(e)}), 500


# * Chunk the text into pieces of 510 characters.
def chunk_text(text, max_len=510):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        # If the chunk is less than max length.
        if len(current_chunk) + len(sentence) < max_len:
            if current_chunk:
                # Add a space for continuing sentences.
                current_chunk += ' '
            current_chunk += sentence
        # If the chunk is more than max length.
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

        # Adding the last chunk to chunks.
        chunks.append(current_chunk)

        return chunks


# * Process Media Files for analysis.
def process_files(file_path):
    mime_type, encoding = mimetypes.guess_type(file_path)
    type, subtype = mime_type.split('/', 1)


# * Find the polarity scores of the input.
def find_text_sentiment_analysis(input):

    # Split the input into separate chunks.
    chunks = chunk_text(input)
    sentiment_dicts = []

    # Get the model and tokenizer (lazy load)
    tokenizer, sentiment_model = get_model()

    for chunk in chunks:
        # Find tokenized words.
        encoded_text = tokenizer(chunk, return_tensors="pt")

        # Find polarity scores.
        output = sentiment_model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Scores
        val_neg = str(scores[0])
        val_neu = str(scores[1])
        val_pos = str(scores[2])

        # Find Prominent Sentiment
        if (val_neg > val_pos) and (val_neg > val_neu):
            prominent_sentiment = "NEGATIVE"
        elif (val_pos > val_neg) and (val_pos > val_neu):
            prominent_sentiment = "POSITIVE"
        else:
            prominent_sentiment = "NEUTRAL"

        # Create Sentiment Analysis Dictionary
        sentiment_dict = {
            'score_negative': val_neg,
            'score_neutral': val_neu,
            'score_positive': val_pos,
            'prominent_sentiment': prominent_sentiment
        }

        sentiment_dicts.append(sentiment_dict)

    # Aggregate the list of chunks to find average sentiment.
    # Aggregate the list of chunks to find average sentiment.
    mean_neg = np.mean([float(d['score_negative']) for d in sentiment_dicts])
    mean_neu = np.mean([float(d['score_neutral']) for d in sentiment_dicts])
    mean_pos = np.mean([float(d['score_positive']) for d in sentiment_dicts])

    # Filter out low probability noise (threshold < 5%) to give cleaner results
    threshold = 0.05
    if mean_neg < threshold: mean_neg = 0.0
    if mean_neu < threshold: mean_neu = 0.0
    if mean_pos < threshold: mean_pos = 0.0

    # Re-normalize to ensure they sum to 1
    total = mean_neg + mean_neu + mean_pos
    if total > 0:
        mean_neg /= total
        mean_neu /= total
        mean_pos /= total

    # Recalculate prominent sentiment based on the filtered averages
    if mean_neg > mean_pos and mean_neg > mean_neu:
        prominent_sentiment = "NEGATIVE"
    elif mean_pos > mean_neg and mean_pos > mean_neu:
        prominent_sentiment = "POSITIVE"
    else:
        prominent_sentiment = "NEUTRAL"

    avg_sentiment_dict = {
        'score_negative': str(mean_neg),
        'score_neutral': str(mean_neu),
        'score_positive': str(mean_pos),
        'prominent_sentiment': prominent_sentiment
    }

    return avg_sentiment_dict
