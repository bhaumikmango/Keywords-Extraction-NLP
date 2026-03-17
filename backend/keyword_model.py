import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import mlflow.pyfunc

class KeywordExtractorModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        with open(context.artifacts["count_vector"], "rb") as f:
            self.cv = pickle.load(f)
        with open(context.artifacts["tfidf_transformer"], "rb") as f:
            self.tfidf_trans = pickle.load(f)
        with open(context.artifacts["feature_names"], "rb") as f:
            self.feature_names = pickle.load(f)

        nltk.download("stopwords", quiet=True)
        nltk.download("punkt",     quiet=True)
        nltk.download("wordnet",   quiet=True)

        base_sw = set(stopwords.words("english"))
        extra   = {"fig","figure","image","sample","using","show","result",
                   "large","also","one","two","three","four","five","six",
                   "seven","eight","nine"}
        self.stop_words = list(base_sw | extra)
        self.stemmer    = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        tokens = nltk.word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words and len(w) >= 3]
        tokens = [self.stemmer.stem(w) for w in tokens]
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    def _sort_coo(self, coo):
        return sorted(zip(coo.col, coo.data), key=lambda x: (x[1], x[0]), reverse=True)

    def predict(self, context, model_input):
        import pandas as pd
        results = []
        for _, row in model_input.iterrows():
            text  = str(row.get("text", ""))
            top_n = int(row.get("top_n", 10))
            proc  = self._preprocess(text)
            if not proc.strip():
                results.append({})
                continue
            vec   = self.tfidf_trans.transform(self.cv.transform([proc]))
            items = self._sort_coo(vec.tocoo())[:top_n]
            kw    = {self.feature_names[i]: round(float(s), 4)
                     for i, s in items if i < len(self.feature_names)}
            results.append(kw)
        return results