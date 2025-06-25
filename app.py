from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load IMDB word index for encoding input text
word_index = imdb.get_word_index()

# Function to preprocess and encode input text
def encode_text(text):
    tokens = text.lower().split()  # Split input into words
    encoded = [word_index.get(word, 2) for word in tokens]  # 2 = <UNK> for unknown words
    padded = pad_sequences([encoded], maxlen=256)  # Pad to match training input length
    return padded

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')  # Loads the input form

# Route to handle prediction POST request
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']  # Get input from form
    processed = encode_text(review)  # Preprocess input
    prediction = model.predict(processed)[0][0]  # Predict sentiment

    # Interpret prediction
    sentiment = "Positive 😀" if prediction > 0.5 else "Negative 😞"
    confidence = f"{prediction*100:.2f}%" if prediction > 0.5 else f"{(1-prediction)*100:.2f}%"

    return render_template('index.html', prediction_text=f"Sentiment: {sentiment} (Confidence: {confidence})")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


