import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import contractions
import re


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import joblib
import json



from streamlit_lottie import st_lottie
st.set_page_config(layout='wide')
#Animation files load funcion
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

amazon = load_lottiefile('Animation - 1696469171082.json')
positive = load_lottiefile('Animation - 1696469273776.json')
negative = load_lottiefile('Animation - 1696469331232.json')
# Load your trained model and vectorizer
model = joblib.load('svm_model.pkl')  # Load the SVM model from the file

# Load your TF-IDF vectorizer here
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the TF-IDF vectorizer from the file

# Streamlit app title
st.title('Amazon Reviews Sentiment Analysis')
st_lottie(amazon, speed=1, reverse=False, quality="low",loop=True, height=250)
# Text input for user to enter a review
# Add a centering column
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    user_input = st.text_area('Enter a review:')

    if st.button('Analyze Sentiment'):
        if user_input:
        # Preprocess the input text
            user_input = re.sub(r'[\W_]+', ' ', contractions.fix(re.sub(r'\d+', '', user_input.replace(' s ', ' ')))).lower()
            tokens = nltk.word_tokenize(user_input)
            user_input = " ".join([token for token in tokens if token not in stopwords.words('english')])
            user_input = ' '.join([SnowballStemmer("english").stem(word) for word in user_input.split()])

        # Vectorize the input text using the loaded TF-IDF vectorizer
            user_input_vec = vectorizer.transform([user_input])

        # Make a prediction
            prediction = model.predict(user_input_vec)

        # Display the result
            if prediction == 0:
                st.write('Sentiment: Negative')
                st_lottie(negative, speed=1, reverse=False, quality="low",loop=True, height=250)

            else:
                st.write('Sentiment: Positive')
                st_lottie(positive, speed=1, reverse=False, quality="low",loop=True, height=250)
        else:
            st.write('Please enter a review for analysis.')
