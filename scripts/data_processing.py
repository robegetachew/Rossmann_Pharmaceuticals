import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data

def extract_features(data):
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        raise ValueError("The 'Date' column must be in datetime format.")
    
    data['Weekday'] = data['Date'].dt.weekday
    data['IsWeekend'] = data['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Assuming a predefined list of holidays
    holidays = pd.to_datetime(['2022-01-01', '2022-12-25'])  # Add all relevant holidays
    data['DaysToHoliday'] = data['Date'].apply(
        lambda x: min((holiday - x).days for holiday in holidays if (holiday - x).days > 0)
        if any((holiday - x).days > 0 for holiday in holidays) else 0
    )
    data['DaysAfterHoliday'] = data['Date'].apply(
        lambda x: min((x - holiday).days for holiday in holidays if (x - holiday).days > 0)
        if any((x - holiday).days > 0 for holiday in holidays) else 0
    )

    data['BeginningOfMonth'] = data['Date'].dt.day <= 10
    data['MidMonth'] = data['Date'].dt.day.between(11, 20)
    data['EndOfMonth'] = data['Date'].dt.day > 20
    data['Season'] = data['Date'].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, etc.

    return data

def handle_missing_values(data):
    if data['Date'].isnull().any():
        print("Warning: There are invalid date entries in the dataset.")
    return data.fillna(method='ffill')

def convert_categorical(data):
    # Handle StateHoliday mapping
    data['StateHoliday'] = data['StateHoliday'].astype(str)
    state_holiday_mapping = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    data['StateHoliday'] = data['StateHoliday'].map(state_holiday_mapping)
    data['StateHoliday'].fillna(0, inplace=True)

    # Convert other categorical variables to numeric using one-hot encoding
    categorical_columns = ['SchoolHoliday']  # Add other categorical columns if needed
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    return data

def scale_features(data, feature_columns):
    scaler = StandardScaler()
    # Ensure all feature columns are numeric
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

def split_data(data, feature_columns, target_column):
    from sklearn.model_selection import train_test_split
    X = data[feature_columns]
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
