import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "..",
        "models",
        "habitability_pipeline.pkl"   # <-- make sure this filename matches
    )
