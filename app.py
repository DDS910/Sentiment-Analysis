import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
	text = text.lower()
	tokens = nltk.word_tokenize(text)
	tokens = [word for word in tokens if word not in stop_words]
	tokens = [lemmatizer.lemmatize(word) for word in tokens]
	clean_text = ' '.join(tokens)
	return clean_text

st.title("Sentiment Analysist App")
input_text = st.text_area("Masukkan kalimat: ")

if st.button("Sentiment Predict"):
	cleaned = preprocess(input_text)
	vectorized = vectorizer.transform([cleaned])
	prediction = model.predict(vectorized)
	st.write(f"Sentiment Prediction: **{prediction[0]}**")
