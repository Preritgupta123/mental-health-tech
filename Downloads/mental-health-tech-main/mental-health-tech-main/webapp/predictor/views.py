import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings

# Load model globally
MODEL_PATH = os.path.join(settings.BASE_DIR.parent, 'models', 'mental_health_model.joblib')
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    encoders = model_data['encoders']
    features = model_data['features']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def home(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST' and model:
        try:
            # Extract inputs
            input_data = {}
            for feature in features:
                val = request.POST.get(feature, '')
                input_data[feature] = [val]
            
            df = pd.DataFrame(input_data)

            # Preprocess Age
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            df['Age'] = df['Age'].fillna(30) # Default age if invalid
            
            # Preprocess Gender (basic mapping if they manually typed)
            df['Gender'] = df['Gender'].str.lower().str.strip()

            # Encode features
            for feature in features:
                if feature in encoders and feature != 'target':
                    le = encoders[feature]
                    # Handle unseen labels by assigning a default or handling gracefully
                    # For simplicity, we assume inputs match the classes
                    # In a real app, you'd map unseen to a default class
                    try:
                        df[feature] = le.transform(df[feature].astype(str))
                    except ValueError:
                        # Fallback if value wasn't in training data
                        # We use the first class as a fallback
                        df[feature] = le.transform([le.classes_[0]])
            
            # Predict
            pred = model.predict(df)[0]
            
            # Decode prediction
            result = encoders['target'].inverse_transform([pred])[0]
            
            return render(request, 'index.html', {'result': result, 'inputs': request.POST})
        except Exception as e:
            return render(request, 'index.html', {'error': str(e)})

    return render(request, 'index.html')
