import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



import pandas as pd

# Load the dataset with proper column names
columns = ["id", "entity", "sentiment", "tweet"]
train_df = pd.read_csv("twitter_training.csv", names=columns, header=None)
val_df = pd.read_csv("twitter_validation.csv", names=columns, header=None)

# Display the first few rows
print(train_df.head())
print(val_df.head())

# Check dataset structure
print(train_df.info())


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Function for text preprocessing
def clean_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers

    tokens = word_tokenize(text)  # Tokenization
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization

    return " ".join(filtered_tokens)

# Apply text cleaning to the dataset
train_df["clean_tweet"] = train_df["tweet"].apply(clean_text)
val_df["clean_tweet"] = val_df["tweet"].apply(clean_text)

# Display cleaned text
print(train_df[["tweet", "clean_tweet"]].head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Convert text to numeric features
X_train = vectorizer.fit_transform(train_df["clean_tweet"]).toarray()
X_val = vectorizer.transform(val_df["clean_tweet"]).toarray()

# Convert sentiment labels into numeric format
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["sentiment"])
y_val = label_encoder.transform(val_df["sentiment"])

# Display feature shapes
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)

# Evaluate performance
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

def predict_sentiment(tweet):
    cleaned_tweet = clean_text(tweet)
    vectorized_tweet = vectorizer.transform([cleaned_tweet]).toarray()
    prediction = model.predict(vectorized_tweet)
    return label_encoder.inverse_transform(prediction)[0]

# Test example
test_tweet = "I love this product! It's amazing."
print("Predicted Sentiment:", predict_sentiment(test_tweet))


while True:
    user_input = input("\nEnter a tweet (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting sentiment analysis. Goodbye!")
        break

    predicted_sentiment = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {predicted_sentiment}")

