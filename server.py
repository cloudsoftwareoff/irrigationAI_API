from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask import Flask, render_template
# Load the model from file
loaded_model = joblib.load("model.sav")

# Create a Flask application
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
#Define a route to accept input data and return predictions
@app.route('/predict', methods=['POST'])
def home():
    # Get the request data as a JSON object
    data = request.get_json()

    # Extract the data from the JSON object
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    soil_moisture = float(data['soil_moisture'])
    rainfall = float(data['rainfall'])

    # create a dataframe with the input data
    new_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Soil Moisture': [soil_moisture],
        'Rainfall': [rainfall]
    })

    # make the prediction 
    y_pred = loaded_model.predict(new_data)

    # Convert the NumPy array to a Python list
    y_pred_list = y_pred.tolist()

    # return the prediction as a JSON response
    return jsonify({'prediction': y_pred_list})


if __name__ == '__main__':
    app.run(debug=True)
