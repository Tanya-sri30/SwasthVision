import pandas as pd 
df = pd.read_csv("data/processed/final_dataset.csv")
po = pd.read_csv("data/raw/water_pollution.diseases.csv")
print(df.columns.tolist())
print(df.shape)