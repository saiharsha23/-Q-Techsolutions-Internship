import pandas as pd 
data=pd.read_csv('./data/spam_ham_dataset.csv')
print(data.head())
print(data.isnull().sum())
print(data['label'].value_counts())


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


import nltk
nltk.download('stopwords')


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    
    text = text.lower()
   
    text = re.sub(r'[^a-z\s]', '', text)
    
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    
    return ' '.join(words)


data['processed_text'] = data['text'].apply(preprocess_text)

print(data[['text', 'processed_text']].head())


from sklearn.model_selection import train_test_split
X = data['processed_text']
y = data['label_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training size: {len(X_train)}")
print(f"Testing size: {len(X_test)}")


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(max_features=5000) 

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF Vectorized shape (training): {X_train_tfidf.shape}")
print(f"TF-IDF Vectorized shape (testing): {X_test_tfidf.shape}")



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_tfidf, y_train)
y_pred_knn = knn_model.predict(X_test_tfidf)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"\nKNN Accuracy: {knn_accuracy:.2f}")
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.2f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of naive bayes: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred)) 
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



output_df = pd.DataFrame({
    'Email': X_test,  
    'Actual Label': y_test,  
    'Predicted Label': y_pred,  
})


output_df['Actual Label'] = output_df['Actual Label'].map({0: 'ham', 1: 'spam'})
output_df['Predicted Label'] = output_df['Predicted Label'].map({0: 'ham', 1: 'spam'})
output_file = 'spam_predictions.csv'
output_df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")











