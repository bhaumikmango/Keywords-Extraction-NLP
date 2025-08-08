from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables to store loaded models
cv = None
tfidf_trans = None
feature_names = None
stop_words = None
stemming = None
lmtr = None

def download_nltk_data():
    """Download required NLTK data if not present"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        logger.info("NLTK data already available")
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK data downloaded successfully")

def load_models():
    """Load pre-trained models and initialize components"""
    global cv, tfidf_trans, feature_names, stop_words, stemming, lmtr
    
    try:
        # Download NLTK data
        download_nltk_data()
        
        # Load pre-trained models
        logger.info("Loading pre-trained models...")
        
        with open('Count_Vector.pkl', 'rb') as f:
            cv = pickle.load(f)
            
        with open('TFIDF_Transformer.pkl', 'rb') as f:
            tfidf_trans = pickle.load(f)
            
        with open('Feature_Names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        # Initialize preprocessing components
        base_stop_words = set(stopwords.words('english'))
        new_words = ['fig', 'figure', 'image', 'sample', 'using',
                     'show', 'result', 'large', 'also', 'one',
                     'two', 'three', 'four', 'five', 'six', 'seven',
                     'eight', 'nine']
        stop_words = list(base_stop_words.union(new_words))
        
        stemming = PorterStemmer()
        lmtr = WordNetLemmatizer()
        
        logger.info("Models loaded successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def preprocessing_text(txt):
    """Preprocess text for keyword extraction"""
    try:
        txt = txt.lower()
        txt = re.sub(r'<.*?>', ' ', txt)
        txt = re.sub(r'[^a-zA-Z]', ' ', txt)
        txt = nltk.word_tokenize(txt)
        txt = [word for word in txt if word not in stop_words]
        txt = [word for word in txt if len(word) >= 3]
        txt = [stemming.stem(word) for word in txt]
        txt = [lmtr.lemmatize(word) for word in txt]
        return ' '.join(txt)
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise

def sort_coo(coo_matrix):
    """Sort COO matrix by values"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """Extract top N features from sorted items"""
    sorted_items = sorted_items[:topn]
    
    results = {}
    for idx, score in sorted_items:
        if idx < len(feature_names):  # Safety check
            keyword = feature_names[idx]
            results[keyword] = round(float(score), 4)  # Ensure it's a regular float
    
    return results

def get_keywords(text, topn=10):
    """Extract keywords from text using loaded models"""
    try:
        # Preprocess text
        processed_text = preprocessing_text(text)
        
        if not processed_text.strip():
            return {}
        
        # Transform text using loaded models
        tf_idf_vector = tfidf_trans.transform(cv.transform([processed_text]))
        
        # Extract keywords
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, topn)
        
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise

# Initialize models when the module loads
models_loaded = load_models()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

@app.route('/api/extract', methods=['POST'])
def extract_keywords_endpoint():
    """Extract keywords from provided text"""
    try:
        # Check if models are loaded
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded properly'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        text = data.get('text', '').strip()
        num_keywords = data.get('numKeywords', 10)
        
        # Validate input
        if not text:
            return jsonify({
                'error': 'No text provided'
            }), 400
        
        if len(text) < 10:
            return jsonify({
                'error': 'Text too short for meaningful keyword extraction'
            }), 400
        
        # Validate num_keywords
        try:
            num_keywords = int(num_keywords)
            if num_keywords < 1 or num_keywords > 50:
                num_keywords = 10
        except (ValueError, TypeError):
            num_keywords = 10
        
        # Extract keywords
        logger.info(f"Extracting {num_keywords} keywords from text of length {len(text)}")
        keywords = get_keywords(text, num_keywords)
        
        if not keywords:
            return jsonify({
                'error': 'No keywords could be extracted from the provided text'
            }), 400
        
        logger.info(f"Successfully extracted {len(keywords)} keywords")
        
        # Return response
        response = make_response(jsonify(keywords))
        response.headers['Content-Type'] = 'application/json'
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error in extract endpoint: {e}")
        return jsonify({
            'error': 'Internal server error occurred during keyword extraction'
        }), 500

@app.route('/api/extract', methods=['OPTIONS'])
def extract_keywords_options():
    """Handle OPTIONS request for CORS"""
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if models_loaded:
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Cannot start server - models failed to load")
        print("Error: Models could not be loaded. Please check that all .pkl files are present.")