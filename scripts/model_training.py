from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import datetime

def build_model_pipeline():
    return Pipeline([
        ('model', RandomForestRegressor(n_estimators=100))
    ])

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(model):
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    joblib.dump(model, f'model_{timestamp}.pkl')