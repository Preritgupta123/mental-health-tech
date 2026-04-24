# Mental Health in Tech Survey Analysis & Prediction

This project is an end-to-end Machine Learning pipeline that predicts whether an individual in the tech industry might seek treatment for mental health conditions, based on various workplace and demographic factors. It consists of:
1. **Data Pipeline & ML Model**: Data cleaning, feature engineering, and a Random Forest Classifier trained on the Kaggle OSMI Mental Health in Tech Survey dataset.
2. **Web Application**: A modern, full-stack Django web app with a sleek glassmorphic UI to serve the model predictions.
3. **Colab Notebook**: An interactive Jupyter Notebook for exploring the data and training the model.

## 📂 Project Structure
- `data/`: Contains the raw dataset `survey.csv`.
- `notebooks/`: Contains the Jupyter notebook `mental_health_model.ipynb` where data analysis and modeling happen.
- `models/`: Where the exported `.joblib` ML model resides.
- `webapp/`: The complete Django project that loads the model and provides a user interface.
- `train_model.py`: Script to train and export the Random Forest model.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Instructions for git to properly ignore environment specifics.

## 🚀 Quick Start

### 1. Setup Environment
**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
Run the provided script to clean data, train the Random Forest model, and save it to the `models/` directory:
```bash
python train_model.py
```

### 3. Start the Web App
Navigate into the `webapp` folder, make sure your virtual environment is active, apply migrations, and start the development server:
```bash
cd webapp
python manage.py migrate
python manage.py runserver
```

Open your browser to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to use the application!

## 📓 Google Colab / Jupyter Notebook
If you want to view the step-by-step data exploration and model training process, open `notebooks/mental_health_model.ipynb` in Jupyter Notebook, JupyterLab, or upload it to Google Colab.

---
*Built as a comprehensive AI & Web Development showcase.*
