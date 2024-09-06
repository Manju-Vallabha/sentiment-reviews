import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import contractions

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set up Streamlit page configuration
st.set_page_config(layout='wide')

# Load trained model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app title
st.title('Amazon Reviews Sentiment Analysis')

# Create a centered column for text input
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    user_input = st.text_area('Enter a review:')

    if st.button('Analyze Sentiment'):
        if user_input:
            # Preprocess the input text
            user_input = re.sub(r'[\W_]+', ' ', contractions.fix(re.sub(r'\d+', '', user_input))).lower()
            tokens = word_tokenize(user_input)
            user_input = ' '.join([SnowballStemmer('english').stem(token) for token in tokens if token not in stopwords.words('english')])

            # Vectorize the input text
            user_input_vec = vectorizer.transform([user_input])

            # Make a prediction
            prediction = model.predict(user_input_vec)

            # Display the result
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            st.info(f'Sentiment: {sentiment}')
        else:
            st.write('Please enter a review for analysis.')
