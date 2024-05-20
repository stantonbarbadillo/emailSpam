import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize the vectorizer and model
vectorizer = TfidfVectorizer(stop_words='english')
model = LogisticRegression()

def preprocess_text(df):
    X = vectorizer.fit_transform(df['Message'])
    y = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    logging.info("Text data has been vectorized and labels encoded")
    return X, y

def train_model(df):
    try:
        X, y = preprocess_text(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        global metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        logging.info("Model trained successfully with metrics: %s", metrics)
    except Exception as e:
        logging.error("Error during model training: %s", str(e))
        raise e

def classify_email(email_content):
    try:
        if not hasattr(vectorizer, 'vocabulary_'):
            raise ValueError("Vectorizer is not fitted. Please train the model first.")
        X = vectorizer.transform([email_content])
        prediction = model.predict(X)
        logging.info("Email classified as: %s", 'spam' if prediction[0] else 'not spam')
        return 'spam' if prediction[0] else 'not spam'
    except Exception as e:
        logging.error("Error during email classification: %s", str(e))
        raise e

def get_metrics():
    try:
        logging.info("Fetched model metrics")
        return metrics
    except Exception as e:
        logging.error("Error fetching metrics: %s", str(e))
        raise e

def get_visualizations():
    try:
        visualizations = {}
        df = pd.read_csv('email.csv')  # Ensure to load the dataset for visualization

        # Visualization 1: Data Distribution
        fig, ax = plt.subplots()
        sns.countplot(x='Category', data=df, ax=ax)
        ax.set_title('Data Distribution')
        visualizations['data_distribution'] = fig_to_base64(fig)
        logging.info("Data distribution visualization created")

        # Visualization 2: Model Performance
        fig, ax = plt.subplots()
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax)
        ax.set_title('Model Performance Metrics')
        visualizations['model_performance'] = fig_to_base64(fig)
        logging.info("Model performance visualization created")

        # Visualization 3: Feature Importance (Top 10)
        fig, ax = plt.subplots()
        feature_importance = pd.DataFrame({
            'feature': vectorizer.get_feature_names_out(),
            'importance': model.coef_[0]
        }).sort_values(by='importance', ascending=False)[:10]
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        ax.set_title('Top 10 Features')
        visualizations['feature_importance'] = fig_to_base64(fig)
        logging.info("Feature importance visualization created")

        return visualizations
    except Exception as e:
        logging.error("Error creating visualizations: %s", str(e))
        raise e

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')
