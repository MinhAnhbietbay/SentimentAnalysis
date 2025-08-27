import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')  # Tải lexicon cho VADER
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Tải và Khám phá Dữ liệu (EDA)
data = pd.read_csv('datasets/amazon_reviews_sample.csv')

# Xem dữ liệu đầu tiên
print("Dữ liệu đầu tiên:")
print(data.head())

# Số lượng positive và negative reviews (sử dụng cột 'score')
print("\nSố lượng positive và negative reviews:")
print(data['score'].value_counts())  # Cột 'score' thay vì 'label'

# Phần trăm positive và negative
print("\nPhần trăm positive và negative:")
print(data['score'].value_counts() / len(data) * 100)

# Độ dài dài nhất và ngắn nhất của reviews
length_reviews = data['review'].str.len()
print("\nĐộ dài dài nhất của review:", max(length_reviews))
print("Độ dài ngắn nhất của review:", min(length_reviews))

# Word Cloud cho toàn bộ reviews
all_text = ' '.join(data['review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud của Toàn bộ Reviews')
plt.show()

# Word Cloud cho positive và negative riêng biệt
positive_text = ' '.join(data[data['score'] == 1]['review'])
negative_text = ' '.join(data[data['score'] == 0]['review'])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud cho Positive Reviews')
plt.show()

wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud cho Negative Reviews')
plt.show()

# 2. Lexicon-based (Rule-based)
# 2.1 TextBlob (Polarity và Subjectivity)
def textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Polarity: -1 (negative) đến 1 (positive)

data['textblob_polarity'] = data['review'].apply(textblob_sentiment)
data['textblob_pred'] = data['textblob_polarity'].apply(lambda x: 1 if x >= 0 else 0)  # Ngưỡng 0 cho positive/negative

print("\nKết quả TextBlob:")
print(data[['review', 'score', 'textblob_polarity', 'textblob_pred']].head())

textblob_accuracy = accuracy_score(data['score'], data['textblob_pred'])
print("Độ chính xác TextBlob:", textblob_accuracy)

# 2.2 VADER (Valence Aware Dictionary and sEntiment Reasoner)
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    scores = sia.polarity_scores(text)
    return scores['compound']  # Compound score: -1 (negative) đến 1 (positive)

data['vader_compound'] = data['review'].apply(vader_sentiment)
data['vader_pred'] = data['vader_compound'].apply(lambda x: 1 if x >= 0 else 0)

print("\nKết quả VADER:")
print(data[['review', 'score', 'vader_compound', 'vader_pred']].head())

vader_accuracy = accuracy_score(data['score'], data['vader_pred'])
print("Độ chính xác VADER:", vader_accuracy)

# 3. Chuẩn bị dữ liệu cho Machine Learning
X = data['review']
y = data['score'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Bag-of-Words (BoW) với các Classifier
vectorizer_bow = CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))  # Thêm n-grams

X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

# 4.1 Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)
y_pred_nb = nb_model.predict(X_test_bow)

print("\nKết quả Naive Bayes (BoW):")
print("Độ chính xác:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred_nb))

# 4.2 Logistic Regression
lr_model = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
lr_model.fit(X_train_bow, y_train)
y_pred_lr_bow = lr_model.predict(X_test_bow)

print("\nKết quả Logistic Regression (BoW):")
print("Độ chính xác:", accuracy_score(y_test, y_pred_lr_bow))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_bow))
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred_lr_bow))

# 5. TF-IDF với các Classifier
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))  # Thêm n-grams

X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# 5.1 SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

print("\nKết quả SVM (TF-IDF):")
print("Độ chính xác:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred_svm))

# 5.2 Logistic Regression với TF-IDF
lr_model_tfidf = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
lr_model_tfidf.fit(X_train_tfidf, y_train)
y_pred_lr_tfidf = lr_model_tfidf.predict(X_test_tfidf)

print("\nKết quả Logistic Regression (TF-IDF):")
print("Độ chính xác:", accuracy_score(y_test, y_pred_lr_tfidf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_tfidf))
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred_lr_tfidf))

# 6. So sánh các Phương pháp
results = {
    'TextBlob': textblob_accuracy,
    'VADER': vader_accuracy,
    'Naive Bayes (BoW)': accuracy_score(y_test, y_pred_nb),
    'Logistic Regression (BoW)': accuracy_score(y_test, y_pred_lr_bow),
    'SVM (TF-IDF)': accuracy_score(y_test, y_pred_svm),
    'Logistic Regression (TF-IDF)': accuracy_score(y_test, y_pred_lr_tfidf)
}

print("\nSo sánh độ chính xác các phương pháp:")
for method, acc in results.items():
    print(f"{method}: {acc:.4f}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('So sánh Độ chính xác các Phương pháp Sentiment Analysis')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()
