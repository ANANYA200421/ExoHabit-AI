import numpy as np


def prepare_features(data):
    """
    Convert JSON input to model format
    """

    required_fields = [
        "pl_rade",
        "pl_eqt",
        "pl_orbper",
        "st_teff",
        "stellar_compatibility"
    ]

    # Check missing fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    try:
        features = [float(data[f]) for f in required_fields]
        return np.array(features).reshape(1, -1)
    except Exception:
        raise ValueError("Invalid data type. All inputs must be numeric.")


def generate_response(pred, prob):
    """
    Format output response
    """

    habitability_class = "Habitable" if pred == 1 else "Non-Habitable"

    return {
        "habitability_class": habitability_class,
        "probability": round(float(prob), 4),
        "habitability_index": round(float(prob), 4),
        "confidence_percent": round(float(prob) * 100, 2)
    }
