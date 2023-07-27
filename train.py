import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\keyke\Documents\botcam\Proyecto\scr\Data\Processed\datafims.csv")
X = df.drop(["director", "writer", "country", "company", "gross", "released_month", "released_country", "star"], axis=1)
y = df["gross"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = XGBRegressor(n_jobs=-1)
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]}
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='neg_mean_squared_error')
xgb_grid_search.fit(X_train, y_train)

xgb_best_params = xgb_grid_search.best_params_
xgb_best_model = xgb_grid_search.best_estimator_
xgb_mse_scores = -cross_val_score(xgb_best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
xgb_mae_scores = -cross_val_score(xgb_best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
xgb_r2_scores = cross_val_score(xgb_best_model, X_train, y_train, cv=5, scoring='r2')

print("XGBoost Regression scaled (Best Model) Cross-Validated Metrics:")
print("MSE:", xgb_mse_scores.mean())
print("MAE:", xgb_mae_scores.mean())
print("R^2:", xgb_r2_scores.mean())
model_filename = "my_model.pkl"
joblib.dump(xgb_best_model, model_filename)
