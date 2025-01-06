import pandas as pd

def preprocess_input(data, scaler):
    # Convert input JSON to DataFrame
    df = pd.DataFrame(data)
    
    # Extract features (ensure these match the training features)
    features = df[["Weekday", "IsWeekend", "DaysToHoliday", "DaysAfterHoliday", 
                   "BeginningOfMonth", "MidMonth", "EndOfMonth", "Season", "Open", "Promo"]]
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    return features_scaled