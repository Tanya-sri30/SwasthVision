import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Paths ---
outputs_path = "outputs"
os.makedirs(outputs_path, exist_ok=True)

scaled_data_path = "data/processed/features_scaled.csv"
predictions_path = "predictions.csv"

# --- Load predictions ---
df_pred = pd.read_csv(predictions_path)
print(f"✅ Loaded predictions: {df_pred.shape}")

# --- Load original features for plotting ---
df_features = pd.read_csv(scaled_data_path)
print(f"✅ Loaded processed features: {df_features.shape}")

# --- Merge predictions with features ---
df = pd.merge(df_features, df_pred, left_index=True, right_index=True)
df = df.dropna()
print(f"✅ Data after dropping NaNs: {df.shape}, {df_features.shape}")

# --- 1️⃣ Risk distribution plot ---
plt.figure(figsize=(6,5))
sns.countplot(
    x='predicted_risk',
    data=df,
    palette='Set2',
    hue=None
)
plt.title("Risk Distribution")
plt.savefig(os.path.join(outputs_path, "risk_distribution.png"))
plt.close()
print(f"✅ Saved risk distribution plot as '{outputs_path}/risk_distribution.png'")

# --- 2️⃣ Feature importance plot (if model exists) ---
try:
    model = joblib.load("models/rf_model.pkl")
    feature_cols = df_features.columns.tolist()
    importances = model.feature_importances_

    plt.figure(figsize=(8,6))
    sns.barplot(
        x=importances,
        y=feature_cols,
        palette="viridis",
        hue=None
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_path, "feature_importance.png"))
    plt.close()
    print(f"✅ Saved feature importance plot as '{outputs_path}/feature_importance.png'")
except Exception as e:
    print(f"⚠️ Could not plot feature importance: {e}")

# --- 3️⃣ Boxplots for all features in a single grid ---
feature_cols = df_features.columns.tolist()
num_features = len(feature_cols)
cols = 4
rows = (num_features + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    sns.boxplot(
        x='predicted_risk',
        y=col,
        data=df,
        palette='Set3',
        hue=None,
        ax=axes[i]
    )
    axes[i].set_title(col)

# Remove empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(outputs_path, "all_boxplots_grid.png"))
plt.close()
print(f"✅ Saved boxplots for all features in '{outputs_path}/all_boxplots_grid.png'")
