from flask import Flask, render_template, request
import pandas as pd
import json
import plotly
import plotly.express as px
import joblib

app = Flask(__name__)

# Load the AI model and vectorizer once when the app starts
try:
    model = joblib.load('medical_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    print(f"Model files not found: {e}")

def get_clean_data():
    try:
        # Read the file
        df = pd.read_csv('drugsComTrain_raw.csv')

        # Basic Cleaning
        df = df.dropna(subset=['condition', 'review'])
        df['review'] = df['review'].str.replace('&#039;', "'", regex=False)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # Split for balancing
        dep = df[df['condition'] == 'Depression']
        hbp = df[df['condition'] == 'High Blood Pressure']
        dia = df[df['condition'] == 'Type 2 Diabetes']

        # Ensure we have data for all three
        if dep.empty or hbp.empty or dia.empty:
            return df # Return unbalanced if one is missing to avoid errors

        min_size = min(len(dep), len(hbp), len(dia))

        # Undersampling
        dep_bal = dep.sample(n=min_size, random_state=42)
        hbp_bal = hbp.sample(n=min_size, random_state=42)
        dia_bal = dia.sample(n=min_size, random_state=42)

        balanced_df = pd.concat([dep_bal, hbp_bal, dia_bal])

        # KEY FIX: reset_index(drop=True) ensures 'condition' stays as a COLUMN
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return balanced_df
    except Exception as e:
        print(f"Data Error: {e}")
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        user_text = request.form.get('symptoms')

        # 1. Get raw confidence scores from the AI
        input_vector = vectorizer.transform([user_text])
        scores = model.decision_function(input_vector)[0]

        # 2. THE HONESTY CHECK
        import numpy as np
        max_score = np.max(scores)

        if max_score < 0.8:
            result = {
                'condition': "Unrecognized Symptom",
                'drug': "N/A",
                'rating': "N/A",
                'is_low_confidence': True,
                'review_snippet': "The system is unsure. Please describe medical symptoms clearly or consult a doctor."
            }
        else:
            # 3. Proceed only if the AI is confident
            predicted_condition = model.predict(input_vector)[0]
            df = get_clean_data()
            temp_df = df.copy()
            matches = temp_df[temp_df['condition'] == predicted_condition]

            if not matches.empty:
                #Sort by rating AND usefulCount to get the most reliable choice
                recommendation = matches.sort_values(by=['rating', 'usefulCount'], ascending=False).iloc[0]

                result = {
                    'condition': predicted_condition,
                    'drug': recommendation['drugName'],
                    'rating': recommendation['rating'],
                    'useful_count': int(recommendation['usefulCount']), # Added for Task 2 analysis
                    'is_low_confidence': False,
                    'review_snippet': recommendation['review'][:200] + "..."
                }
            else:
                result = {
                    'condition': "Error",
                    'drug': "No specific drug found",
                    'rating': "N/A",
                    'is_low_confidence': True,
                    'review_snippet': "Condition identified but no recommendation available."
                }

    return render_template('predict.html', result=result)

@app.route('/insights')
def insights():
    df = get_clean_data()
    if df.empty: return "Error loading data."

    fig1 = px.pie(df, names='condition', title='Reviews per Condition')
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    avg_rating = df.groupby('condition')['rating'].mean().reset_index()
    fig2 = px.bar(avg_rating, x='condition', y='rating', color='condition', title='Avg Patient Rating')
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    fig3 = px.histogram(df, x='rating_category', color='condition', barmode='group', title='Sentiment by Condition')
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    fig4 = px.scatter(df.sample(min(len(df), 1000)), x='review_length', y='rating', color='condition', title='Review Length vs Rating')
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    stats = {
        'total': f"{len(df):,}",
        'avg_rating': round(df['rating'].mean(), 1),
        'top_condition': df['condition'].value_counts().idxmax()
    }

    return render_template('insights.html', graph1=graph1JSON, graph2=graph2JSON, graph3=graph3JSON, graph4=graph4JSON, stats=stats)

if __name__ == '__main__':
    app.run(debug=True)