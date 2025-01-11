import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis")

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()

st.title("Social Media Sentiment Analyzer with Summarization")

# File uploader for bulk analysis
uploaded_file = st.file_uploader("Upload a file (CSV or TXT)", type=["csv", "txt"])
posts = []

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        posts = df["Post"].tolist()
    elif uploaded_file.name.endswith(".txt"):
        posts = uploaded_file.read().decode("utf-8").splitlines()

if posts:
    st.write(f"Loaded {len(posts)} posts for analysis.")

# Text input for single post
post = st.text_area("Enter a post for analysis:")
if post:
    posts.append(post)

if st.button("Analyze"):
    if posts:
        summaries = [summarizer(post, max_length=50, min_length=25, do_sample=False)[0]["summary_text"] for post in posts]
        sentiment_results = [sentiment_analyzer(post)[0]["label"] for post in posts]

        # Display results
        for i, post in enumerate(posts):
            st.write(f"### Post {i+1}: {post}")
            st.write(f"**Summary:** {summaries[i]}")
            st.write(f"**Sentiment:** {sentiment_results[i]}")
        
        # Visualize sentiment distribution
        visualize_sentiment(sentiment_results)

        # Generate word cloud
        if st.button("Generate Word Cloud"):
            generate_wordcloud(posts)
    else:
        st.warning("No posts to analyze. Please upload a file or enter text.")
