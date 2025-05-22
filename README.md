# Sentiment-Analysis
This project is the development of a Sentiment Analysis system for English tweet texts using supervised machine learning methods. The system classifies tweets into three sentiment categories: positive, neutral, and negative. The model is trained using a labeled dataset to recognize language patterns and emotions in the text.

# Dataset
The dataset used in this project is taken from Kaggle. Here is the link: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset.

# Features
- Sentiment classification into 3 classes: positive, negative, and neutral
- Text pre-processing including lowercasing, tokenization, stopwords removal, and lemmatization

# File Explanation
- app.py : displays the web UI of the Sentiment Analysis model where users can input comments to be predicted as negative, positive, or neutral
- sentiment_model_english.py : script for the Sentiment Analysis model
- sentiment_model.pkl : trained sentiment analysis model saved using the Joblib module
- tfidf_vectorizer.pk : TF-IDF object fitted on training data and saved using the Joblib module
- Tweets.csv : dataset used to train the Sentiment Analysis model

# How to run
1. Install all the libraries (streamlit, pandas, scikit-learn, nltk, joblib, seaborn)
2. Open your terminal or command prompt and change the directory to where the project is located
3. Start the Streamlit application with: streamlit run app.py


