import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_final_dataset():
    # Load both datasets
    water_quality = pd.read_csv("data/raw/Water_pond_tanks_2021.csv", encoding='latin1')
    pollution_disease = pd.read_csv("data/raw/water_pollution_disease.csv ", encoding='latin1')

    print("✅ Loaded datasets:")
    
    print(water_quality.columns.tolist())
    print(pollution_disease.columns.tolist())
    print("Water Quality shape:", water_quality.shape)
    print("Pollution & Disease shape:", pollution_disease.shape)

    # --- STEP 1: Select useful columns ---
    water_quality = water_quality[[
        'State Name',
        'pH (Min)', 'pH (Max)',
        'Dissolved Oxygen (mg/L) (Min)', 'Dissolved Oxygen (mg/L) (Max)',
        'BOD (mg/L) (Min)', 'BOD (mg/L) (Max)',
        'Conductivity (?mhos/cm) (Min)', 'Conductivity (?mhos/cm) (Max)',
        'Total Coliform (MPN/100ml) (Max)'
    ]]

    pollution_disease = pollution_disease[[
    'Country', 'Region', 'Rainfall (mm per year)', 'Temperature (Â°C)',
    'Population Density (people per kmÂ²)', 'Diarrheal Cases per 100,000 people',
    'Cholera Cases per 100,000 people', 'Typhoid Cases per 100,000 people'
]]


    # --- STEP 2: Clean column names ---
    water_quality.columns = [
        'state', 'ph_min', 'ph_max', 'do_min', 'do_max',
        'bod_min', 'bod_max', 'conductivity_min', 'conductivity_max', 'coliform_max'
    ]
    pollution_disease.columns = [
        'country', 'region', 'rainfall', 'temperature',
        'population_density', 'diarrhea_cases', 'cholera_cases', 'typhoid_cases'
    ]

    # --- STEP 3: Merge datasets ---
    # Since states and regions don’t match exactly, use a simple concatenation
    merged = pd.concat([water_quality, pollution_disease], axis=1).dropna()

    # --- STEP 4: Create target variable (risk level) ---
    merged["total_cases"] = (
        merged["diarrhea_cases"] +
        merged["cholera_cases"] +
        merged["typhoid_cases"]
    )

    def classify_risk(cases):
        if cases < 500:
            return "Low"
        elif cases < 2000:
            return "Medium"
        else:
            return "High"

    merged["risk_level"] = merged["total_cases"].apply(classify_risk)

    # --- STEP 5: Save final dataset ---
    merged.to_csv("data/processed/final_dataset.csv", index=False)
    print("✅ Final dataset saved to data/processed/final_dataset.csv")
    print("Final shape:", merged.shape)
    print("Columns:", merged.columns.tolist())

if __name__ == "__main__":
    create_final_dataset()
