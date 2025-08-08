import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle 
df=pd.read_csv('papers.csv')

stop_words = set(stopwords.words('english')) 
new_words = ['fig', 'figure', 'image', 'sample', 'using',
             'show', 'result', 'large', 'also', 'one',
             'two', 'three', 'four', 'five', 'six', 'seven',
             'eight', 'nine']
stop_words = list(stop_words.union(new_words))
nltk.download('all')

def preprocessing_text(txt):
    txt = txt.lower()
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub(r'[^a-zA-Z]', ' ', txt)
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word not in stop_words]
    txt = [word for word in txt if len(word)>=3]
    stemming = PorterStemmer()
    txt = [stemming.stem(word) for word in txt]
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]
    return ' '.join(txt)

docs = df['paper_text'].apply(lambda x:preprocessing_text(x))
cv = CountVectorizer(max_df=0.95, max_features=5000, ngram_range=(1,2))
words_count_vectors = cv.fit_transform(docs)
tfidf_trans = TfidfTransformer(smooth_idf=True, use_idf=True) 
tfidf_trans = tfidf_trans.fit(words_count_vectors)
feature_names = cv.get_feature_names_out()

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score,3))
        feature_vals.append(feature_names[idx])
    
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

feature_names=cv.get_feature_names_out()

def get_keywords(idx, docs):
    tf_idf_vector=tfidf_trans.transform(cv.transform([docs[idx]]))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    return keywords

def print_results(idx,keywords, df):
    print("\n=====Title=====")
    print(df['title'][idx])
    print("\n=====Abstract=====")
    print(df['abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])

pickle.dump(cv, open('Count_Vector.pkl', 'wb'))
pickle.dump(tfidf_trans, open('TFIDF_Transformer.pkl', 'wb'))
pickle.dump(feature_names, open('Feature_Names.pkl', 'wb'))