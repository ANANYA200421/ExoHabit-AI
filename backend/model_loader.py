import joblib, os

BASE=os.path.dirname(os.path.abspath(__file__))
MODEL=os.path.join(BASE,"../models/final_habitability_model.pkl")

model=joblib.load(MODEL)
print("MODEL LOADED")
