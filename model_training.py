import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
data = pd.read_csv('updated.csv')
print(data.columns.tolist())
X = data[['Latitudes', 'Longitudes', 'Month', 'Temperature(°C)', 'DHI(w/m2)', 'DNI(w/m2)', 'GHI(w/m2)', 'Solar Zenith Angle(Degrees)']]
y = data['Solar Irradiance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_pred_train = xgb_model.predict(X_train)
rf_pred_train = rf_model.predict(X_train)
xgb_pred_test = xgb_model.predict(X_test)
rf_pred_test = rf_model.predict(X_test)
stacked_train = np.column_stack((xgb_pred_train, rf_pred_train))
stacked_test = np.column_stack((xgb_pred_test, rf_pred_test))
meta_model = LinearRegression()
meta_model.fit(stacked_train, y_train)
final_predictions = meta_model.predict(stacked_test)
mae = mean_absolute_error(y_test, final_predictions)
print(f"Mean Absolute Error of Stacking Ensemble: {mae}")
ensemble_model = {
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'meta_model': meta_model
}
joblib.dump(ensemble_model, 'solar_irradiance_model.pkl')
print("Ensemble model saved successfully.")
