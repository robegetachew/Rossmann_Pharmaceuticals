import matplotlib.pyplot as plt
import seaborn as sns
from scripts.logger_config import setup_logger

logger = setup_logger()

def missing_values(data):
    logger.info("Checking for missing values...")
    result = {}
    for key in data:
        error_count= data[key].isna().sum()
        result[key] = error_count
    return result

def check_outliers(data,columns_to_plot):
    logger.info("Checking for outliers...")
    df_melted = data.melt(value_vars=columns_to_plot, var_name='Variable', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x='Variable', y='Value')
    plt.title('Box Plot of Selected Columns')
    plt.xlabel('Variables')
    plt.ylabel('Values')
    plt.show()