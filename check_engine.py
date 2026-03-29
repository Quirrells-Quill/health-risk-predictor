import pandas as pd

df = pd.read_csv("data/air_quality_health.csv")
print(df.columns.tolist())