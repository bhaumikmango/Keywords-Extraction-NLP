Keyword Extraction using NLP 🔍
This repository contains a web application built with Streamlit that performs keyword extraction from text using a pre-trained NLP model. The application leverages a TF-IDF vectorizer to identify the most important keywords from a given text, making it a valuable tool for summarizing content, analyzing documents, and understanding key themes.

Project Overview
The project is a machine learning-based application for extracting keywords from a block of text. The core components include:

A pre-trained model based on a TF-IDF vectorizer.

A Streamlit-powered user interface for seamless interaction.

Visualizations, including a word cloud and a bar chart, to display the extracted keywords and their importance scores.

The model was trained on a dataset of academic papers to learn the statistical importance of words in a corpus.

Features
Keyword Extraction: Takes any text input and returns a list of the most relevant keywords.

Interactive UI: A user-friendly interface built with Streamlit allows for easy text input and result display.

Text Preprocessing: The application automatically cleans and processes the input text by performing tasks such as lowercasing, removing special characters, stop word removal, stemming, and lemmatization before keyword extraction.

Visualizations: Displays the top keywords in a dynamic word cloud and a bar chart, providing a clear visual representation of their importance scores.

Use Cases
This project can be used for a variety of tasks, including:

Content Summarization: Quickly get the gist of a long article, research paper, or blog post without reading the entire text.

SEO (Search Engine Optimization): Identify relevant keywords for a new blog post or web page to improve its ranking in search results.

Document Analysis: Analyze a large corpus of documents to discover the main topics, themes, and trends.

Customer Feedback Analysis: Extract keywords from customer reviews or survey responses to understand common sentiments, pain points, or popular features.

News Trend Analysis: Automatically monitor news headlines and articles to identify emerging trends and popular topics in real time.

Technologies Used
Python

Streamlit: For building the interactive web application.

NLTK: For natural language processing tasks like tokenization, stop word removal, and stemming/lemmatization.

Scikit-learn: For creating the CountVectorizer and TfidfTransformer to generate keyword importance scores.

Pandas: For data manipulation and handling the training dataset.

Plotly & Matplotlib: For generating interactive and static data visualizations.

WordCloud: For creating the keyword word cloud visualization.

Installation
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/bhaumikmango/Keywords-Extraction-NLP.git
cd Keywords-Extraction-NLP
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the dependencies:

pip install -r requirements.txt
Download NLTK data: The application requires NLTK data to be downloaded. The app.py script handles this automatically upon first run. Alternatively, you can download all NLTK packages manually:

python -c "import nltk; nltk.download('all')"
Train the model: You must train the model and save the necessary .pkl files before running the application. Use the train.py script for this purpose.

python train.py
Usage
Once the dependencies are installed and the model is trained, you can launch the Streamlit application with the following command:

streamlit run app.py
This will open the application in your default web browser, where you can enter text and extract keywords.

Project Structure
Keywords-Extraction-NLP/
├── app.py               # Main Streamlit application
├── train.py             # Script to train the TF-IDF model
├── requirements.txt     # List of project dependencies
├── main.ipynb           # Jupyter Notebook for data exploration and model training
├── papers.csv           # Sample dataset used for training
├── Count_Vector.pkl     # Saved CountVectorizer model
├── TFIDF_Transformer.pkl# Saved TfidfTransformer model
└── Feature_Names.pkl    # Saved feature names from the CountVectorizer
Deployed Application
You can access the live Streamlit application here: https://bhaumikmango-keywords-extraction-nlp-app-u397r.streamlit.app/
