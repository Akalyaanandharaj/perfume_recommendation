from flask import Flask, jsonify
from flask import request
from perfume_recom import get_recommendations,df
import pickle

app = Flask(__name__)


@app.route('/perfume', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        occasion = request.args.get('occasion')
        age_range = request.args.get('age_range')
        description = request.args.get('description')
        print(occasion)
        print(age_range)
        print(description)

        # Load TF-IDF Vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Load Cosine Similarity Matrix
        with open('cosine_sim.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)

        results = get_recommendations(occasion, age_range, description, cosine_sim, df, tfidf_vectorizer)
        return jsonify(results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
