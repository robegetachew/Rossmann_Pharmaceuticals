from sklearn.metrics import mean_absolute_error

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae, y_pred

def feature_importance(pipeline, feature_columns):
    import pandas as pd
    
    importances = pipeline.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
    return importance_df.sort_values(by='Importance', ascending=False)