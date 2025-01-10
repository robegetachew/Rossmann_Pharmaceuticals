import pandas as pd
from scripts.logger_config import *
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set up the logger
logger = setup_logger()


def remove_outliers(df,exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
    logger.info("Removing outliers for numerical columns")
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        if column in exclude_columns:
            logger.info(f"Skipping outlier removal for column: {column}")
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df[column] = df[column].clip(lower_bound, upper_bound)
        
    return df



def remove_missing_values(df, threshold=0.5):
    logger.info(f"Removing missing values with threshold {threshold}")
    # Calculate the threshold number of non-NA values required
    threshold_count = int(threshold * df.shape[1])
    
    df_cleaned = df.dropna(thresh=threshold_count)
    
    return df_cleaned


def remove_categorical_outliers(df, threshold=0.01):
    logger.info(f"Removing categorical outliers with threshold {threshold}")
    for column in df.select_dtypes(include=['object']).columns:
        counts = df[column].value_counts(normalize=True)
        
        rare_categories = counts[counts < threshold].index
        
        df[column] = df[column].replace(rare_categories, 'Other')
        
    return df


def remove_missing_values_categorical(df, fill_value='Unknown'):
    logger.info(f"Removing missing values in categorical columns with fill value {fill_value}")
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(fill_value, inplace=True)
        
    return df

def preprocess_data(df):
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract features from the date
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] > 5).astype(int)

    # Combine holiday columns to minimize forward fills
    df['days_to_holiday'] = np.where(df['StateHoliday'] != 'none', 0, np.nan)
    df['days_after_holiday'] = np.where(df['SchoolHoliday'] == 1, 0, np.nan)
    df[['days_to_holiday', 'days_after_holiday']] = df[['days_to_holiday', 'days_after_holiday']].fillna(0)

    # Beginning, mid, and end of the month
    df['beginning_of_month'] = (df['Date'].dt.day <= 10).astype(int)
    df['mid_of_month'] = ((df['Date'].dt.day > 10) & (df['Date'].dt.day <= 20)).astype(int)
    df['end_of_month'] = (df['Date'].dt.day > 20).astype(int)

    # Additional features
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['quarter'] = df['Date'].dt.quarter

    df.drop(columns=['Date'], inplace=True)

    # Handle missing values (customize this if needed)
    df.fillna(0, inplace=True)
    
    # Label encode Store to avoid memory issues with get_dummies
    if 'StoreType' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['StoreType'] = le.fit_transform(df['StoreType'])

    # Encode stateHoliday column
    df['StateHoliday'] = df['StateHoliday'].map({
        'a': 1,
        'b': 2,
        'c': 3,
        '0': 0
    }).fillna(0).astype(int)

    df['SchoolHoliday'] = df['SchoolHoliday'].astype(int)

    df['Assortment'] = df['Assortment'].map({
        'a': 1,
        'b': 2,
        'c': 3,
        'none': 0
    }).fillna(0).astype(int)

    # Ensure schoolHoliday is in the right format (it should already be 1 or 0)
    df['Assortment'] = df['Assortment'].astype(int)

    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(int)  # Assuming it's binary (1/0)

    if 'SalesPeriod' in df.columns:
        df['SalesPeriod'] = le.fit_transform(df['SalesPeriod'])  # Label encode SalesPeriod

    if 'SeasonalHoliday' in df.columns:
        df['SeasonalHoliday'] = le.fit_transform(df['SeasonalHoliday'])  # Label encode SeasonalHoliday

    if 'PromoInterval' in df.columns:
        df['PromoInterval'] = df['PromoInterval'].astype(str)
        df['PromoInterval'] = le.fit_transform(df['PromoInterval'])  # Label encode PromoInterval

    # Scale only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop('Store', errors='ignore')  # Exclude 'Store' from scaling

    scaler = StandardScaler()
    df[numeric_cols] = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

    return df
