# Med AI: Patient Symptom & Drug Recommendation System

##  Project Overview
This project is an AI-driven medical dashboard designed to predict health conditions based on patient symptoms and recommend the most "useful" medications. Built for the Limkokwing University AI & Data Science assignment.

## AI Model Details
- **Algorithm:** Passive Aggressive Classifier (PAC)
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Tuning:** Hyperparameter optimization via GridSearchCV (C-parameter tuning)
- **Safety Feature:** Implemented a **0.8 Confidence Threshold** to prevent "hallucinations" or false predictions on unrecognized input.

##  Tech Stack
- **Backend:** Python / Flask
- **Machine Learning:** Scikit-Learn, Pandas, Joblib
- **Frontend:** HTML5, CSS (Bootstrap), Plotly for Data Insights

##  How to Run
1. Install requirements: `pip install flask pandas scikit-learn joblib plotly`
2. Run the training script: `python train.py`
3. Start the app: `python app.py`
4. Visit `http://127.0.0.1:5000` in your browser.

### Dashboard Preview
![App Screenshot](https://github.com/Puseletso24/Med-AI-Predictor/blob/main/Screenshot%20(258).png)
![App Screenshot](https://github.com/Puseletso24/Med-AI-Predictor/blob/main/Screenshot%20(259).png)
![App Screenshot](https://github.com/Puseletso24/Med-AI-Predictor/blob/main/Screenshot%20(260).png)

