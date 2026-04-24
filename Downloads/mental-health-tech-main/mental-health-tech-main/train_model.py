import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    print("Cleaning data...")
    # Drop irrelevant columns
    df = df.drop(['Timestamp', 'state', 'comments'], axis=1)

    # Clean Age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df.loc[(df['Age'] < 18) | (df['Age'] > 75), 'Age'] = np.nan
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Clean Gender
    df['Gender'] = df['Gender'].str.lower().str.strip()
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "cis male"]
    female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]
    trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]

    for (row, col) in df.iterrows():
        if str.lower(col.Gender) in male_str:
            df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
        if str.lower(col.Gender) in female_str:
            df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
        if str.lower(col.Gender) in trans_str:
            df['Gender'].replace(to_replace=col.Gender, value='other', inplace=True)

    df['Gender'] = df['Gender'].replace(to_replace=['a little about you', 'p'], value='other')

    # Fill NA values
    df['self_employed'] = df['self_employed'].fillna('No')
    df['work_interfere'] = df['work_interfere'].fillna('Don\'t know')

    return df

def train_model(df):
    print("Training model...")
    # Select features
    features = ['Age', 'Gender', 'family_history', 'work_interfere', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'leave']
    target = 'treatment'
    
    X = df[features].copy()
    y = df[target].copy()

    # Encode categorical variables
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    encoders['target'] = le_target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': rf, 'encoders': encoders, 'features': features}, 'models/mental_health_model.joblib')
    print("Model saved to models/mental_health_model.joblib")

if __name__ == "__main__":
    file_path = "data/survey.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please place the dataset in the data/ directory.")
    else:
        df = load_data(file_path)
        df_cleaned = clean_data(df)
        train_model(df_cleaned)
