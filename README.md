# 🧠 Keywords Extraction using NLP

A full-stack NLP application that extracts the **most relevant keywords** from uploaded research papers using **TF-IDF vectorization** and displays them via a user-friendly web interface built with Flask.

> 🚀 Live demo and deployment-ready with trained model files and search functionality.

---

## 📌 Project Overview

This project implements an **end-to-end pipeline** for automatic keyword extraction from scientific documents (PDF/text). It preprocesses textual data, transforms it into vector form using **CountVectorizer** and **TF-IDF**, and extracts the top-scoring keywords from each document. The app allows file upload and keyword searching via a web frontend.

---

## 🔍 Core Features

- 📂 Upload a `.txt` document and get top keywords based on TF-IDF scores
- 🔍 Search for keywords within the pre-trained vocabulary
- 📈 Uses `CountVectorizer` and `TfidfTransformer` from `sklearn`
- 🧪 Pre-trained vectorizer and transformer for consistent inference
- ⚙️ Clean and modular Flask backend
- 🧼 NLTK-powered advanced preprocessing pipeline

---

## 🛠️ Tech Stack

| Layer       | Tools Used                                     |
|-------------|------------------------------------------------|
| Language    | Python 3                                       |
| NLP         | `nltk`, `sklearn`                              |
| Vectorizer  | `CountVectorizer`, `TfidfTransformer`          |
| Backend     | Flask                                          |
| Frontend    | HTML, Jinja2 templates                         |
| Deployment  | Compatible with Render, Heroku, or Vercel      |

---

## 🧹 NLP Preprocessing Steps

- Lowercasing
- Removal of HTML tags
- Tokenization with NLTK
- Custom stopword filtering (including common scientific words)
- Lemmatization using WordNetLemmatizer

---

## 🧪 Model Files

These serialized model files are used during inference:

- `Count_Vector.pkl` – Vectorizer vocabulary
- `TFIDF_Transformer.pkl` – Trained transformer
- `Feature_Names.pkl` – Feature list for keyword matching

> All files are loaded in `app.py` for real-time predictions.

---

## 🖥️ Web Interface Features

### `/` – Upload Page

- Upload a text file.
- View extracted keywords.

### `/extract_keywords` – Keyword Generation

- Shows top 20 keywords using TF-IDF from the uploaded document.

### `/search_keywords` – Keyword Search

- Query to search keywords from trained vocabulary.

---

## 📁 Project Structure

```bash
Keywords-Extraction-NLP/
│
├── app.py                   # Flask backend logic
├── templates/
│   ├── index.html           # Upload & search interface
│   ├── keywords.html        # Extracted keywords view
│   └── keywordslist.html    # Keyword search results
├── model_files/
│   ├── Count_Vector.pkl
│   ├── TFIDF_Transformer.pkl
│   └── Feature_Names.pkl
├── papers.csv               # Sample dataset (optional)
├── requirements.txt         # Python dependencies
├── README.md                # You're here!
```

## 🚀 How to Run Locally
### 1. Clone the Repository
git clone https://github.com/bhaumikmango/Keywords-Extraction-NLP.git
cd Keywords-Extraction-NLP
### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
### 3. Install Dependencies
pip install -r requirements.txt
### 4. Run Flask App
python app.py
Access the app at: http://127.0.0.1:5000/

## 📦 Requirements
Python 3.8+

pip (Python package manager)

See requirements.txt for complete list

## 🙌 Acknowledgments
💼 Dataset inspired by real-world academic repositories

🔍 NLP pipeline built using NLTK

💡 TF-IDF logic inspired by Scikit-learn documentation

🌐 Hosted via Flask

## 👨‍💻 Author
Bhaumik Mango

🔗 GitHub Profile

🌍 Passionate about NLP, AI, and building impactful tools

## ⭐ Contribute & Support
If you like this project, please ⭐ the repository.
Feel free to open issues or pull requests to enhance its capabilities!
