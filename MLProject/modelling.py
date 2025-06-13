import argparse
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

# --- Argument Parser for MLflow CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# --- MLflow Tracking ---
if os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME'):
    # Use remote MLflow tracking (DagsHub)
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN')
    mlflow.set_tracking_uri("https://dagshub.com/karindaamelia/air-quality-model.mlflow")
    mlflow.set_experiment("air-quality-basic")
    print("Using remote MLflow tracking on DagsHub")
    use_remote = True
else:
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("air-quality-basic-local")
    print("DagsHub credentials not found. Using local MLflow tracking.")
    use_remote = False

# --- Load Data ---
data = pd.read_csv(args.data_path)
X = data.drop(columns=["AH"])
y = data["AH"]

# --- Preprocessing ---
X = pd.get_dummies(X, drop_first=True)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# --- MLflow Autologging ---
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestRegressor(random_state=args.random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)

    # Manual log
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("evs", evs)
    
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features", X_train.shape[1])

    # Save model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    mlflow.log_artifact(args.model_output)

    # Register model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    try:
        mlflow.register_model(model_uri=model_uri, name="air-quality-prediction")
        print("✅ Model registered.")
    except Exception as e:
        print("⚠️ Model registration skipped or failed:", e)

    # Info: Tracking location
    if use_remote:
        print("📍 Tracking URL: https://dagshub.com/karindaamelia/air-quality-model.mlflow")
    else:
        print("📍 Tracking dir: ./mlruns")

    # Instructions for serving the model
    registered_model_name = "air-quality-prediction"
    if use_remote:
        print("\n📦 To serve the model remotely from model registry, run:")
        print(f"🔗 mlflow models serve -m 'models:/{registered_model_name}/latest' --port 5000")
    else:
        print("\n📦 To serve the model locally from this run, run:")
        print(f"🔗 mlflow models serve -m '{model_uri}' --port 5000")

