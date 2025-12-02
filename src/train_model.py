import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    # Paths
    scaled_data_path = os.path.join("data/processed/features_scaled.csv")
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)

    # --- STEP 0: Load scaled dataset ---
    df = pd.read_csv(scaled_data_path)
    print("✅ Loaded scaled dataset:", df.shape)

    # --- STEP 0.5: Check & drop NaNs ---
    print("⚠️ Checking for NaNs:")
    print(df.isnull().sum())
    df = df.dropna()
    print(f"✅ Dataset after dropping NaNs: {df.shape}")

    # --- STEP 1: Separate features and target ---
    X = df.drop(columns=['risk_level'])
    y = df['risk_level']

    # --- STEP 1.5: Check target classes ---
    print("Risk level value counts:\n", y.value_counts())

    # --- STEP 2: Split dataset ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- STEP 3: Train Random Forest with Grid Search ---
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid,
        cv=5, n_jobs=-1, verbose=1, scoring='accuracy'
    )

    print("🚀 Training Random Forest...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"✅ Best parameters: {grid_search.best_params_}")

    # --- STEP 4: Evaluate ---
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # --- STEP 5: Save model ---
    model_file = os.path.join(models_path, "rf_model.pkl")
    joblib.dump(best_model, model_file)
    print(f"✅ Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
