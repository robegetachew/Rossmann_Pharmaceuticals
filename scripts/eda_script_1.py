import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path=None):
    """Load a specified dataset based on the provided path."""
    logging.info("Loading data...")
    
    if data_path:
        try:
            dataset = pd.read_csv(data_path)
            logging.info("Data loaded successfully from %s", data_path)
            return dataset
        except Exception as e:
            logging.error("Error loading data from %s: %s", data_path, e)
            return None
    else:
        logging.warning("No data path provided.")
        return None


def clean_data(df):
    """Clean missing values in the DataFrame."""
    logging.info("Cleaning data...")
    
    for column in df.columns:
        if df[column].dtype == 'object':  # Fill categorical columns with mode
            df[column] = df[column].fillna(df[column].mode()[0])
        elif pd.api.types.is_numeric_dtype(df[column]):  # Fill numerical columns with median
            df[column] = df[column].fillna(df[column].median())
        else:  # Fill other types with a placeholder
            df[column] = df[column].fillna('Unknown')
    
    logging.info("Data cleaning completed.")
    return df

def plot_promotion_distribution(train, test):
    """Plot the distribution of promotions in training and test sets."""
    logging.info("Plotting promotion distribution...")
    plt.figure(figsize=(12, 6))
    
    # Count distribution in training set
    train_promo_counts = train['Promo'].value_counts(normalize=True)
    test_promo_counts = test['Promo'].value_counts(normalize=True)

    # Create a DataFrame for easier plotting
    promo_distribution = pd.DataFrame({
        'Train': train_promo_counts,
        'Test': test_promo_counts
    }).fillna(0)  # Fill NaN with 0 for missing values in either dataset

    # Plotting
    promo_distribution.plot(kind='bar', figsize=(12, 6))
    plt.title('Promotion Distribution: Train vs Test')
    plt.xlabel('Promotion Status')
    plt.ylabel('Proportion of Stores')
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal
    plt.legend(title='Dataset')
    plt.grid(axis='y')  # Add grid for better readability
    plt.show()

    logging.info("Promotion distribution plotted.")


def analyze_promotions(df):
    """Analyze the effect of promotions on sales and customer behavior."""
    logging.info("Analyzing promotions...")

    # Grouping by Promo to calculate average sales and average number of customers
    promo_analysis = df.groupby('Promo').agg({
        'Sales': 'mean',
        'Customers': 'mean'
    }).reset_index()

    # Visualizing average sales with and without promo
    plt.figure(figsize=(12, 6))

    # Bar plot for average sales and average customers
    plt.subplot(1, 2, 1)
    sns.barplot(x='Promo', y='Sales', data=promo_analysis)
    plt.title('Average Sales with and without Promo')
    plt.xlabel('Promo')
    plt.ylabel('Average Sales')

    plt.subplot(1, 2, 2)
    sns.barplot(x='Promo', y='Customers', data=promo_analysis)
    plt.title('Average Customers with and without Promo')
    plt.xlabel('Promo')
    plt.ylabel('Average Customers')

    plt.tight_layout()
    plt.show()

    # Analyzing the impact of promotions on existing customer behavior
    existing_sales = df[df['Promo'] == 0]['Sales'].sum()
    promo_sales = df[df['Promo'] == 1]['Sales'].sum()

    logging.info("Total Sales without Promo: %s", existing_sales)
    logging.info("Total Sales with Promo: %s", promo_sales)
    

def seasonal_analysis_with_holidays(df):
    """Analyze seasonal effects on sales, highlighting specific holidays."""
    logging.info("Analyzing seasonal effects with holidays...")
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create a column for month
    df['Month'] = df['Date'].dt.month
    
    # Create a column for holiday type
    df['Holiday_Type'] = df['StateHoliday'].replace({'0': 'None', 'a': 'Public Holiday', 
                                                      'b': 'Easter Holiday', 'c': 'Christmas'})
    
    # Calculate average sales by month and holiday type
    month_holiday_sales = df.groupby(['Month', 'Holiday_Type'])['Sales'].mean().unstack(fill_value=0)

    # Plotting
    plt.figure(figsize=(12, 6))
    month_holiday_sales.plot(kind='bar')
    plt.title('Average Monthly Sales by Holiday Type')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.legend(title='Holiday Type')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    logging.info("Seasonal analysis with holidays completed.")

def customer_behavior_analysis(df):
    """Analyze customer behavior in relation to sales."""
    logging.info("Analyzing customer behavior...")
    
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='Customers', y='Sales', data=df)
    plt.title('Sales vs Number of Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()
    
    logging.info("Customer behavior analysis completed.")

def store_opening_impact(df):
    """Analyze the impact of store openings on sales."""
    logging.info("Analyzing impact of store openings on sales...")
    
    opening_sales = df[df['Open'] == 1].groupby('Store')['Sales'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store', y='Sales', data=opening_sales)
    plt.title('Average Sales by Store (Open)')
    plt.xlabel('Store')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=90)
    plt.show()
    
    logging.info("Store opening impact analysis completed.")
    
    
def holiday_analysis(df):
    """Analyze sales behavior before, during, and after holidays."""
    logging.info("Analyzing holiday effects on sales...")
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Identify holidays based on the 'StateHoliday' column
    holiday_dates = df[df['StateHoliday'] != '0']['Date'].unique()

    # Create a new column for holiday status
    df['Holiday_Status'] = 'Before'
    
    # Define time periods relative to holidays
    for holiday in holiday_dates:
        holiday = pd.to_datetime(holiday)
        # Set 'During' for the holiday date
        df.loc[df['Date'] == holiday, 'Holiday_Status'] = 'During'
        # Set 'After' for the day after the holiday
        df.loc[df['Date'] == holiday + pd.Timedelta(days=1), 'Holiday_Status'] = 'After'
        # Set 'Before' for the day before the holiday
        df.loc[df['Date'] == holiday - pd.Timedelta(days=1), 'Holiday_Status'] = 'Before'
    
    # Group by holiday status and calculate mean sales
    sales_summary = df.groupby('Holiday_Status')['Sales'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Holiday_Status', y='Sales', data=sales_summary, hue='Holiday_Status', legend=False)
    
    plt.title('Average Sales Behavior Before, During, and After Holidays')
    plt.xlabel('Holiday Status')
    plt.ylabel('Average Sales')
    plt.grid(axis='y')  # Add grid for better readability
    plt.show()

    logging.info("Holiday analysis completed.")

def promo_effectiveness_analysis(sales_df, store_df):
    """Analyze the effectiveness of promotions by store type."""
    logging.info("Analyzing promo effectiveness by store type...")
    
    # Merge sales data with store data
    merged_df = sales_df.merge(store_df, on='Store', how='left')

    # Grouping by Promo and StoreType to calculate average sales
    promo_sales = merged_df.groupby(['Promo', 'StoreType'])['Sales'].mean().reset_index()

    # Visualizing the average sales with and without promo for each store type
    plt.figure(figsize=(12, 6))
    sns.barplot(x='StoreType', y='Sales', hue='Promo', data=promo_sales)
    plt.title('Promo Effectiveness by Store Type')
    plt.xlabel('Store Type')
    plt.ylabel('Average Sales')
    plt.legend(title='Promo')
    plt.tight_layout()
    plt.show()
    
    logging.info("Promo effectiveness analysis completed.")
    
def analyze_weekday_open_stores(train_data):
    """Identify stores open on all weekdays and analyze their weekend sales."""
    logging.info("Analyzing stores open on all weekdays and their sales...")
    # Convert 'Date' to datetime if not already
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    
    # Extract the day of the week (0=Monday, 6=Sunday)
    train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
    
    # Group by Store and check open status for weekdays (0-4)
    weekday_open_stores = train_data[train_data['DayOfWeek'] < 5]  # Monday to Friday
    open_stores = weekday_open_stores.groupby('Store')['Open'].agg(all_open='all').reset_index()
    
    # Filter stores that are open all weekdays
    open_all_weekdays = open_stores[open_stores['all_open'] == True]
    
    # Graph 1: Specific stores open on all weekdays
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Store', data=open_all_weekdays, hue='Store', palette='pastel', legend=False)
    plt.title('Stores Open on All Weekdays', fontsize=14)
    plt.xlabel('Number of Stores', fontsize=12)
    plt.ylabel('Store', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Identify sales for these stores
    sales_data = train_data[train_data['Store'].isin(open_all_weekdays['Store'])]
    
    # Analyze weekday and weekend sales
    sales_data.loc[:, 'Period'] = sales_data['DayOfWeek'].apply(lambda x: 'Weekday' if x < 5 else 'Weekend')
    
    # Group by Period and Store to calculate average sales
    avg_sales = sales_data.groupby(['Store', 'Period'])['Sales'].mean().reset_index()

    # Graph 2: Average sales comparison for stores open all weekdays
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Store', y='Sales', hue='Period', data=avg_sales, palette='pastel')
    plt.title('Average Sales for Stores Open on All Weekdays', fontsize=14)
    plt.xlabel('Store', fontsize=12)
    plt.ylabel('Average Sales ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Sales Period')
    plt.tight_layout()
    plt.show()
    
    logging.info("Analysis on weekday and sales completed.")
    return open_all_weekdays, avg_sales


def store_hours_analysis(df):
    """Analyze trends during store opening and closing times."""
    logging.info("Analyzing trends during store opening and closing times...")
    open_sales = df.groupby('Open')['Sales'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Open', y='Sales', data=open_sales)
    plt.title('Average Sales Based on Store Open Status')
    plt.xlabel('Store Open')
    plt.ylabel('Average Sales')
    plt.show()

def analyze_assortment_effect_on_sales(store_data, train_data):
    """Check how the assortment type affects sales."""
    
    # Merge store data with sales data on 'Store'
    merged_data = pd.merge(train_data, store_data, on='Store', how='left')

    # Group by Assortment and calculate average sales
    assortment_sales = merged_data.groupby('Assortment')['Sales'].mean().reset_index()
    
    # Visualization of average sales by assortment type
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Assortment', y='Sales', data=assortment_sales)
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    logging.info("Analyzing trends during store opening and closing times completed.")
    return assortment_sales

def analyze_competitor_distance_effect(store_data, train_data):
    """Analyze how the distance to the next competitor affects sales, focusing on city center stores."""
    logging.info("Analyzing the distance to the next competitor affects sales...")
    # Merge store data with sales data
    merged_data = pd.merge(train_data, store_data, on='Store', how='left')

    # Convert 'Date' to datetime if not already
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    # Filter for city center stores (assuming 'a' corresponds to city center stores)
    city_center_data = merged_data[merged_data['StoreType'] == 'a']

    # Remove rows with NA values in CompetitionDistance or Sales
    city_center_data = city_center_data.dropna(subset=['CompetitionDistance', 'Sales'])

    # Plotting the relationship for city center stores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=city_center_data)
    plt.title('Sales vs. Distance to Next Competitor (City Center Stores)')
    plt.xlabel('Distance to Next Competitor (meters)')
    plt.ylabel('Sales')
    plt.axhline(y=city_center_data['Sales'].mean(), color='r', linestyle='--', label='Mean Sales')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Analyze for all stores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=merged_data.dropna(subset=['CompetitionDistance', 'Sales']))
    plt.title('Sales vs. Distance to Next Competitor (All Stores)')
    plt.xlabel('Distance to Next Competitor (meters)')
    plt.ylabel('Sales')
    plt.axhline(y=merged_data['Sales'].mean(), color='r', linestyle='--', label='Mean Sales')
    plt.legend()
    plt.tight_layout()
    plt.show()
    logging.info("Analyzing the distance to the next competitor affects sales completed.")
    return city_center_data

def analyze_new_competitors_effect(store_data, train_data):
    """Check how the opening or reopening of new competitors affects stores."""
    logging.info("Analyzing the opening or reopening of new competitors...")
    # Merge store data with sales data
    merged_data = pd.merge(train_data, store_data, on='Store', how='left')

    # Check for stores with NA CompetitionDistance initially
    stores_with_na = merged_data[merged_data['CompetitionDistance'].isna()]

    # Get unique stores that initially had NA values
    unique_stores_with_na = stores_with_na['Store'].unique()

    # Initialize a list to hold sales data for analysis
    before_after_data = []

    for store in unique_stores_with_na:
        store_data = merged_data[merged_data['Store'] == store]
        
        # Get the first date when the CompetitionDistance is updated
        updated_entry = store_data[store_data['CompetitionDistance'].notna()]
        
        if not updated_entry.empty:
            update_date = updated_entry.iloc[0]['Date']

            # Separate sales data before and after the update
            before_sales = store_data[store_data['Date'] < update_date]['Sales']
            after_sales = store_data[store_data['Date'] >= update_date]['Sales']

            # Append before and after sales data
            before_after_data.append({
                'Store': store,
                'Before_After': 'Before Update',
                'Sales': before_sales.mean() if not before_sales.empty else 0
            })
            before_after_data.append({
                'Store': store,
                'Before_After': 'After Update',
                'Sales': after_sales.mean() if not after_sales.empty else 0
            })

    # Convert the list to a DataFrame
    sales_comparison = pd.DataFrame(before_after_data)

    # Check if sales_comparison is empty
    if sales_comparison.empty:
        print("No data available for sales comparison.")
        return None
    
    # Ensure Store and Before_After are treated as categorical
    sales_comparison['Store'] = sales_comparison['Store'].astype('category')
    sales_comparison['Before_After'] = sales_comparison['Before_After'].astype('category')

    # Visualization of sales before and after the distance update
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Store', y='Sales', hue='Before_After', data=sales_comparison)
    plt.title('Sales Before and After Competitor Distance Update')
    plt.xlabel('Store')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    logging.info("Analyzing the opening or reopening of new competitors completed.")
    return sales_comparison