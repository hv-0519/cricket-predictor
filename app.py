from flask import Flask, render_template, request, jsonify, session
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)
app.secret_key = 'cricket_prediction_secret_key_2024'

# Database initialization
def init_db():
    conn = sqlite3.connect('cricket_predictions.db')
    c = conn.cursor()
    
    # Predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  team1 TEXT,
                  team2 TEXT,
                  venue TEXT,
                  toss_winner TEXT,
                  toss_decision TEXT,
                  match_type TEXT,
                  team1_win_prob REAL,
                  team2_win_prob REAL,
                  key_factors TEXT,
                  prediction_time TIMESTAMP)''')
    
    # Match history table
    c.execute('''CREATE TABLE IF NOT EXISTS match_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  team1 TEXT,
                  team2 TEXT,
                  venue TEXT,
                  toss_winner TEXT,
                  toss_decision TEXT,
                  match_type TEXT,
                  winner TEXT,
                  team1_score INTEGER,
                  team2_score INTEGER,
                  match_date DATE)''')
    
    conn.commit()
    conn.close()

# Generate synthetic training data
def generate_training_data():
    np.random.seed(42)
    teams = ['India', 'Australia', 'England', 'Pakistan', 'South Africa', 
             'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan']
    venues = ['Mumbai', 'Melbourne', 'Lords', 'Dubai', 'Cape Town', 
              'Auckland', 'Kingston', 'Colombo', 'Dhaka', 'Sydney']
    
    data = []
    for _ in range(1000):
        team1, team2 = np.random.choice(teams, 2, replace=False)
        venue = np.random.choice(venues)
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(['bat', 'bowl'])
        match_type = np.random.choice(['ODI', 'T20', 'Test'])
        
        # Simulate realistic features
        team1_rating = np.random.uniform(70, 95)
        team2_rating = np.random.uniform(70, 95)
        venue_advantage = 1 if np.random.random() > 0.5 else 0
        toss_advantage = 1 if toss_winner == team1 else 0
        
        # Winner based on ratings with some randomness
        win_prob = 1 / (1 + np.exp(-(team1_rating - team2_rating) / 10))
        winner = team1 if np.random.random() < win_prob else team2
        
        data.append({
            'team1': team1, 'team2': team2, 'venue': venue,
            'toss_winner': toss_winner, 'toss_decision': toss_decision,
            'match_type': match_type, 'team1_rating': team1_rating,
            'team2_rating': team2_rating, 'venue_advantage': venue_advantage,
            'toss_advantage': toss_advantage, 'winner': winner
        })
    
    return pd.DataFrame(data)

# Train model
def train_model():
    df = generate_training_data()
    
    # Encode categorical variables
    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    le_toss_decision = LabelEncoder()
    le_match_type = LabelEncoder()
    
    all_teams = list(set(df['team1'].tolist() + df['team2'].tolist()))
    le_team.fit(all_teams)
    le_venue.fit(df['venue'])
    le_toss_decision.fit(df['toss_decision'])
    le_match_type.fit(df['match_type'])
    
    # Prepare features
    X = pd.DataFrame({
        'team1': le_team.transform(df['team1']),
        'team2': le_team.transform(df['team2']),
        'venue': le_venue.transform(df['venue']),
        'toss_winner_is_team1': (df['toss_winner'] == df['team1']).astype(int),
        'toss_decision': le_toss_decision.transform(df['toss_decision']),
        'match_type': le_match_type.transform(df['match_type']),
        'team1_rating': df['team1_rating'],
        'team2_rating': df['team2_rating'],
        'venue_advantage': df['venue_advantage'],
        'toss_advantage': df['toss_advantage']
    })
    
    y = (df['winner'] == df['team1']).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    joblib.dump(model, 'model.pkl')
    joblib.dump(le_team, 'le_team.pkl')
    joblib.dump(le_venue, 'le_venue.pkl')
    joblib.dump(le_toss_decision, 'le_toss_decision.pkl')
    joblib.dump(le_match_type, 'le_match_type.pkl')
    
    return model, le_team, le_venue, le_toss_decision, le_match_type

# Load or train model
def load_model():
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        le_team = joblib.load('le_team.pkl')
        le_venue = joblib.load('le_venue.pkl')
        le_toss_decision = joblib.load('le_toss_decision.pkl')
        le_match_type = joblib.load('le_match_type.pkl')
    else:
        model, le_team, le_venue, le_toss_decision, le_match_type = train_model()
    return model, le_team, le_venue, le_toss_decision, le_match_type

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model, le_team, le_venue, le_toss_decision, le_match_type = load_model()
        
        team1 = data['team1']
        team2 = data['team2']
        venue = data['venue']
        toss_winner = data['toss_winner']
        toss_decision = data['toss_decision']
        match_type = data['match_type']
        
        # Prepare features
        X_pred = pd.DataFrame({
            'team1': [le_team.transform([team1])[0]],
            'team2': [le_team.transform([team2])[0]],
            'venue': [le_venue.transform([venue])[0]],
            'toss_winner_is_team1': [1 if toss_winner == team1 else 0],
            'toss_decision': [le_toss_decision.transform([toss_decision])[0]],
            'match_type': [le_match_type.transform([match_type])[0]],
            'team1_rating': [np.random.uniform(75, 90)],
            'team2_rating': [np.random.uniform(75, 90)],
            'venue_advantage': [1 if np.random.random() > 0.5 else 0],
            'toss_advantage': [1 if toss_winner == team1 else 0]
        })
        
        # Get probability
        prob = model.predict_proba(X_pred)[0]
        team1_prob = round(prob[1] * 100, 2)
        team2_prob = round(prob[0] * 100, 2)
        
        # Feature importance
        feature_importance = model.feature_importances_
        features = ['Team Strength', 'Opposition', 'Venue', 'Toss Winner', 
                   'Toss Decision', 'Match Type', 'Team1 Rating', 'Team2 Rating',
                   'Venue Advantage', 'Toss Advantage']
        
        key_factors = []
        for feat, imp in sorted(zip(features, feature_importance), 
                               key=lambda x: x[1], reverse=True)[:5]:
            key_factors.append({'factor': feat, 'importance': round(imp * 100, 2)})
        
        # Save prediction
        conn = sqlite3.connect('cricket_predictions.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (team1, team2, venue, toss_winner, toss_decision, match_type,
                      team1_win_prob, team2_win_prob, key_factors, prediction_time)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (team1, team2, venue, toss_winner, toss_decision, match_type,
                   team1_prob, team2_prob, json.dumps(key_factors), datetime.now()))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'team1_prob': team1_prob,
            'team2_prob': team2_prob,
            'key_factors': key_factors
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def history():
    conn = sqlite3.connect('cricket_predictions.db')
    predictions = pd.read_sql_query('SELECT * FROM predictions ORDER BY id DESC LIMIT 20', conn)
    conn.close()
    return render_template('history.html', predictions=predictions.to_dict('records'))

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/api/analytics-data')
def analytics_data():
    conn = sqlite3.connect('cricket_predictions.db')
    predictions = pd.read_sql_query('SELECT * FROM predictions', conn)
    conn.close()
    
    if len(predictions) == 0:
        return jsonify({'team_stats': [], 'venue_stats': []})
    
    team_stats = []
    all_teams = set(predictions['team1'].tolist() + predictions['team2'].tolist())
    for team in all_teams:
        team1_matches = predictions[predictions['team1'] == team]
        team2_matches = predictions[predictions['team2'] == team]
        avg_prob = (team1_matches['team1_win_prob'].mean() + 
                   (100 - team2_matches['team2_win_prob'].mean())) / 2
        if not np.isnan(avg_prob):
            team_stats.append({'team': team, 'avg_win_prob': round(avg_prob, 2)})
    
    venue_stats = predictions.groupby('venue').size().to_dict()
    venue_stats = [{'venue': k, 'count': v} for k, v in venue_stats.items()]
    
    return jsonify({'team_stats': team_stats, 'venue_stats': venue_stats})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)