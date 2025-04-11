from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load your simulated dataset
users_df = pd.read_csv('large_users.csv')  # Use relative path if CSV is in same folder

# Preprocess for similarity calculation
users_df['features'] = users_df['skills'].fillna('') + ',' + users_df['interests'].fillna('')
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
feature_matrix = vectorizer.fit_transform(users_df['features'])

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API route
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    input_skills = data.get('skills', '')
    input_interests = data.get('interests', '')
    team_size = int(data.get('team_size', 4))

    input_text = input_skills + ',' + input_interests
    input_vector = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vector, feature_matrix).flatten()

    top_indices = similarities.argsort()[-team_size:][::-1]
    recommendations = users_df.iloc[top_indices][['user_id', 'skills', 'interests']].to_dict(orient='records')

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
