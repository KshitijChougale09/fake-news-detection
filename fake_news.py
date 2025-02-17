import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

st.title("📰 Fake News Detection using Machine Learning")

# File Upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.write(df.head())

    # Data Preprocessing
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['text'], inplace=True)
    
    # Tokenization & Stopword Removal
    stop_words = set(stopwords.words('english'))
    df['processed_text'] = df['text'].apply(lambda x: [word.lower() for word in word_tokenize(str(x)) if word.isalpha() and word.lower() not in stop_words])
    
    # Word Frequency Analysis
    word_freq = {}
    for text in df['processed_text']:
        for word in text:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Visualization - WordCloud
    all_text = ' '.join([' '.join(words) for words in df['processed_text']])
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_text)
    
    st.subheader("Word Cloud of News Articles")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
    # Sentiment Analysis
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['processed_text'].apply(lambda x: sid.polarity_scores(' '.join(x))['compound'])
    df['sentiment_category'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    
    st.subheader("Sentiment Analysis Distribution")
    st.bar_chart(df['sentiment_category'].value_counts())
    
    # Fake News Classification
    if 'label' in df.columns:
        X = df['processed_text'].apply(lambda x: ' '.join(x))
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy:.2f}")
        
        # Display Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Dataset must contain a 'label' column with 'true' and 'fake' values.")
