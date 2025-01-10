from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
import sys
sys.path.append('../')
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler



def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to get the latest model filename
def get_latest_model(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError("No model files found in the directory.")
    latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, latest_model)



model_directory = "../notebooks/models/"
model_path = get_latest_model(model_directory)
model = load_model(model_path)

scaler_path = "../notebooks/scaler.pkl"
scaler = load_scaler(scaler_path)


app = FastAPI()

app.mount("/static", StaticFiles(directory="../static"), name="static")


# Define input data model
class PredictionInput(BaseModel):
    Id: float
    Store: int
    DayOfWeek: float
    Open: float
    Promo: float
    StateHoliday: float
    SchoolHoliday: float
    StoreType: float
    Assortment: float
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: float
    Promo2SinceWeek: float
    Promo2SinceYear: float
    PromoInterval: float
    Year	: float
    Month	: float
    Day	: float
    WeekOfYear: float	
    CompetitionOpen	: float
    Promo2Open	: float
    IsPromo2Month	: float
    day_of_week	: float
    is_weekend	: float
    days_to_holiday	: float
    days_after_holiday	: float
    beginning_of_month	: float
    mid_of_month	: float
    end_of_month	: float
    is_month_end	: float
    is_month_start	: float
    quarter: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    data = np.array([[input_data.Store,input_data.DayOfWeek,input_data.Open,input_data.Promo,
                      input_data.StateHoliday,input_data.SchoolHoliday,input_data.StoreType,
                      input_data.Assortment,input_data.CompetitionDistance,
                      input_data.CompetitionOpenSinceMonth,input_data.CompetitionOpenSinceYear,
                      input_data.Promo2,input_data.Promo2SinceWeek,input_data.Promo2SinceYear,
                      input_data.PromoInterval,input_data.Year	,input_data.Month	,input_data.Day	,
                      input_data.WeekOfYear	,input_data.CompetitionOpen	,input_data.Promo2Open	,
                      input_data.IsPromo2Month	,input_data.day_of_week	,input_data.is_weekend	,
                      input_data.days_to_holiday	,input_data.days_after_holiday	,
                      input_data.beginning_of_month	,input_data.mid_of_month	,input_data.end_of_month	,
                      input_data.is_month_end	,input_data.is_month_start	,input_data.quarter]]) 
    scaled_data = scaler.transform(data)
    scaled_prediction = model.predict(scaled_data)

    return {"prediction": scaled_prediction[0]}
