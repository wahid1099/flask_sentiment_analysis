
from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk


nltk.download('punkt')  
nltk.download('stopwords')  
nltk.download('punkt_tab')


model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


app = Flask(__name__)


def preprocess_text(text):
    try:
       
        text = re.sub(r'<.*?>', '', text)
       
        text = re.sub(r'http\S+|www\S+', '', text)
      
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        text = text.lower()
        
        tokens = word_tokenize(text)
       
        tokens = [word for word in tokens if word not in stopwords.words('english')]
       
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
        return ' '.join(tokens)
    except LookupError as e:
        print(f"NLTK resource error: {e}")
        raise ValueError("Failed to tokenize text. Ensure NLTK resources are downloaded.")


@app.route("/")
def home():
    return render_template("index.html")  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"Received data: {data}")
    if not data:
        return jsonify({"error": "Invalid JSON format"}), 400
    review = data.get("review", "")


    processed_review = preprocess_text(review)
    
   
    vectorized_review = vectorizer.transform([processed_review]).toarray()
    
   
    prediction = model.predict(vectorized_review)[0]
    sentiment = "positive" if prediction == 1 else "negative"

    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
