import pandas as pd
import nltk
nltk.download()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#Dataset Exploration
df = pd.read_csv('Tweets.csv')
df.head(5)

#Data Preprocessing
#Convert text to lowercase
df['text'] = df['text'].str.lower()
#Converting text column value type to string
df['text'] = df['text'].astype(str)
#Tokenization
df['tokens'] = df['text'].apply(nltk.word_tokenize)
#Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['tokens'] = df['tokens'].apply(lambda x : [word for word in x if word not in stopwords])
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['clean_text'] = df['tokens'].apply(lambda x: ' '.join(x))

#Split the dataset
x = df['clean_text']
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Feature Extraction
vectorizer = TfidfVectorizer(
    ngram_range=(1,3) ,
    max_df=0.95,
    min_df=5,
    stop_words='english'
)
x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)


#Build and train sentiment analysis model
model = SVC(kernel='linear', C=1.0, class_weight='balanced')
model.fit(x_train_vectors, y_train)

#Evaluate the model
y_pred = model.predict(x_test_vectors)

print("Classification Report: ")
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("Accuracy score: ", accuracy_score(y_test, y_pred))
print(y.value_counts())

import joblib

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')