from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from datetime import datetime, timedelta

app = Flask(__name__)

# Load models
btc_model = load_model('btc_lstm.h5')
bnb_model = load_model('bnb_lstm.h5')

# Load scalers
btc_scaler = MinMaxScaler()
bnb_scaler = MinMaxScaler()

# Load BTC training data (replace with your actual BTC training data)
btc_data = pd.read_csv('btc_training_data.csv')  # Change to your actual BTC training data file
btc_scaler.fit(btc_data[['Close']].values.reshape(-1, 1))

# Load BNB training data (replace with your actual BNB training data)
bnb_data = pd.read_csv('bnb_training_data.csv')  # Change to your actual BNB training data file
bnb_scaler.fit(bnb_data[['Close']].values.reshape(-1, 1))

def get_prediction_data(model, scaler, training_data, days_to_predict):
    # Calculate prediction dates
    last_training_date = datetime.strptime(training_data['Date'].max(), '%Y-%m-%d')
    prediction_dates = [(last_training_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_to_predict + 1)]

    # Prepare input data for the model
    last_sequence = training_data['Close'].tail(360).values.reshape(-1, 1)
    input_sequence = scaler.transform(last_sequence).reshape(1, 360, 1)

    # Predict values
    predicted_values_scaled = []
    for _ in range(days_to_predict):
        prediction = model.predict(input_sequence, verbose=0)
        predicted_values_scaled.append(prediction[0, 0])
        input_sequence = np.roll(input_sequence, -1)
        input_sequence[0, -1, 0] = prediction[0, 0]

    # Convert predicted values back to the original scale
    predicted_values = scaler.inverse_transform(np.array(predicted_values_scaled).reshape(-1, 1)).reshape(-1)

    return prediction_dates, predicted_values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    crypto_choice = request.form['crypto']
    days_to_predict = int(request.form['days'])

    # Check crypto choice and load corresponding model and data
    if crypto_choice == 'BTC-USD':
        model = btc_model
        scaler = btc_scaler
        training_data = btc_data
    elif crypto_choice == 'BNB-USD':
        model = bnb_model
        scaler = bnb_scaler
        training_data = bnb_data
    else:
        error_message = 'Invalid crypto choice. Please choose BTC-USD or BNB-USD.'
        return render_template('error.html', message=error_message)

    # Get prediction data
    prediction_dates, predicted_values = get_prediction_data(model, scaler, training_data, days_to_predict)

    # Placeholder for accuracy calculation (replace with your actual calculation)
    accuracy = 0.0

    # Create a line chart using Plotly Express
    fig = px.line(x=prediction_dates, y=predicted_values, labels={'y': 'Predicted Price', 'x': 'Date'},
                  title=f'Predicted Prices for {crypto_choice} in the Next {days_to_predict} Days')
    
    # Convert the chart to HTML
    chart_html = fig.to_html(full_html=False)

    # Render the result template with the obtained data
    return render_template('result.html', crypto=crypto_choice, days=days_to_predict, accuracy=accuracy, chart=chart_html)

if __name__ == '__main__':
    app.run(debug=True)
