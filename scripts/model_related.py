def predict_sales(input_data):
    # Scale the input data using the stored scaler
    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    
    # Make prediction
    predicted_scaled = pipeline.predict(input_scaled)

    # Return the predicted sales
    return predicted_scaled[0]