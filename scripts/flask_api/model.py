import joblib

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)  # Load your trained model
    scaler = joblib.load(scaler_path)  # Load your scaler
    return model, scaler