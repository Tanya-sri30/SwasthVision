import pandas as pd
import os
import joblib

def predict_risk(input_csv):
    # Paths
    models_path = "models"
    scaler_file = os.path.join(models_path, "scaler.pkl")
    model_file = os.path.join(models_path, "rf_model.pkl")

    # --- STEP 1: Load scaler and model ---
    scaler = joblib.load(scaler_file)
    model = joblib.load(model_file)
    print("✅ Loaded scaler and model.")

    # --- STEP 2: Load input data ---
    df = pd.read_csv(input_csv)
    print(f"✅ Loaded input data: {df.shape}")

    # --- STEP 3: Handle missing values ---
    df = df.dropna()
    print(f"✅ Data after dropping NaNs: {df.shape}")

    # --- STEP 4: Derived features (same as training) ---
    df['ph_range'] = df['ph_max'] - df['ph_min']
    df['do_range'] = df['do_max'] - df['do_min']
    df['bod_range'] = df['bod_max'] - df['bod_min']
    df['conductivity_range'] = df['conductivity_max'] - df['conductivity_min']

    # --- STEP 5: Select features ---
    feature_cols = [
        'ph_min', 'ph_max', 'ph_range',
        'do_min', 'do_max', 'do_range',
        'bod_min', 'bod_max', 'bod_range',
        'conductivity_min', 'conductivity_max', 'conductivity_range',
        'coliform_max', 'rainfall', 'temperature', 'population_density'
    ]
    X = df[feature_cols]

    # --- STEP 6: Scale features ---
    X_scaled = scaler.transform(X)

    # --- STEP 7: Predict ---
    predictions = model.predict(X_scaled)
    df['predicted_risk'] = predictions

    print("✅ Predictions done.")
    print(df[['predicted_risk']].value_counts())

    # --- STEP 8: Save predictions ---
    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")

    return df

if __name__ == "__main__":
    input_file = os.path.join("data/processed/features_scaled.csv")  # example input
    predict_risk(input_file)
