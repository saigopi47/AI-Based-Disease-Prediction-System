import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
df = pd.read_csv("Training.csv")

# split
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# train
model = RandomForestClassifier()
model.fit(X, y)

# save model
joblib.dump(model, "disease_model.joblib")

print("Model trained and saved!")