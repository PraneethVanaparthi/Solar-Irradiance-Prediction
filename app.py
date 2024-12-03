import os
from flask import Flask, render_template, request
import numpy as np
import joblib
app = Flask(__name__)
ensemble_model = joblib.load('solar_irradiance_model.pkl')
@app.route('/')
def home():
    return render_template('mini.html')
@app.route('/predict', methods=['POST'])
def predict():
    latitudes = float(request.form['latitudes'])
    longitudes = float(request.form['longitudes'])
    month = int(request.form['month'])
    temperature = float(request.form['temperature'])
    dhi = float(request.form['dhi'])
    dni = float(request.form['dni'])
    ghi = float(request.form['ghi'])
    solar_zenith_angle = float(request.form['solar_zenith_angle'])
    features = np.array([[latitudes, longitudes, month, temperature, dhi, dni, ghi, solar_zenith_angle]])
    xgb_model = ensemble_model['xgb_model']
    rf_model = ensemble_model['rf_model']
    meta_model = ensemble_model['meta_model']
    xgb_pred = xgb_model.predict(features)
    rf_pred = rf_model.predict(features)
    stacked_features = np.column_stack((xgb_pred, rf_pred))
    final_prediction = meta_model.predict(stacked_features)
    final_prediction_with_unit = f"{final_prediction[0]} W/m²"
    return render_template('mini.html', prediction=final_prediction_with_unit,
                           latitudes=latitudes, longitudes=longitudes, month=month,
                           temperature=temperature, dhi=dhi, dni=dni,
                           ghi=ghi, solar_zenith_angle=solar_zenith_angle)

if __name__ == '__main__':
    app.run(debug=True)
