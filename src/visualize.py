import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

def visualize_data():
    # Paths
    predictions_path = os.path.join("predictions.csv")
    processed_data_path = os.path.join("data/processed/features_scaled.csv")
    model_path = os.path.join("models", "rf_model.pkl")
    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)

    # --- STEP 1: Load data ---
    df_pred = pd.read_csv(predictions_path)
    df_data = pd.read_csv(processed_data_path)
    print(f"✅ Loaded predictions: {df_pred.shape}")
    print(f"✅ Loaded processed features: {df_data.shape}")

    # --- STEP 2: Drop NaNs ---
    df_pred = df_pred.dropna()
    df_data = df_data.dropna()
    print(f"✅ Data after dropping NaNs: {df_pred.shape}, {df_data.shape}")

    # --- STEP 3: Merge predictions for analysis ---
    if 'predicted_risk' in df_pred.columns:
        df_data['predicted_risk'] = df_pred['predicted_risk']

    # --- STEP 4: Distribution of predicted risk ---
    plt.figure(figsize=(6,4))
    sns.countplot(
        x='predicted_risk',
        hue='predicted_risk',
        data=df_data,
        palette='Set2',
        dodge=False,
        legend=False
    )
    plt.title("Distribution of Predicted Risk Levels")
    plt.xlabel("Predicted Risk")
    plt.ylabel("Count")
    plt.tight_layout()
    risk_plot_file = os.path.join(output_path, "risk_distribution.png")
    plt.savefig(risk_plot_file)
    plt.close()
    print(f"✅ Saved risk distribution plot as '{risk_plot_file}'.")

    # --- STEP 5: Feature Importance ---
    try:
        model = joblib.load(model_path)
        if hasattr(model, "feature_importances_"):
            feature_cols = df_data.drop(columns=['risk_level', 'predicted_risk'], errors='ignore').columns
            importances = model.feature_importances_
            
            if len(importances) == len(feature_cols):
                plt.figure(figsize=(10,6))
                sns.barplot(x=importances, y=feature_cols, palette="viridis")
                plt.title("Feature Importance")
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                plt.tight_layout()
                fi_plot_file = os.path.join(output_path, "feature_importance.png")
                plt.savefig(fi_plot_file)
                plt.close()
                print(f"✅ Saved feature importance plot as '{fi_plot_file}'.")
            else:
                print("⚠️ Could not plot feature importance: Length mismatch between features and importances.")
        else:
            print("⚠️ Model has no attribute 'feature_importances_'.")
    except Exception as e:
        print("⚠️ Could not plot feature importance:", e)

    # --- STEP 6: Boxplots of features by predicted risk ---
    feature_cols = df_data.drop(columns=['risk_level', 'predicted_risk'], errors='ignore').columns
    for col in feature_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(
            x='predicted_risk',
            y=col,
            data=df_data,
            palette='Set3'
        )
        plt.title(f"{col} by Predicted Risk Level")
        plt.xlabel("Predicted Risk")
        plt.ylabel(col)
        plt.tight_layout()
        boxplot_file = os.path.join(output_path, f"{col}_boxplot.png")
        plt.savefig(boxplot_file)
        plt.close()
    print(f"✅ Saved boxplots for all features in '{output_path}'.")

if __name__ == "__main__":
    visualize_data()
