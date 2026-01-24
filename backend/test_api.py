import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:5000"

print("=" * 80)
print("🧪 TESTING EXOHABITAI BACKEND API")
print("=" * 80)

# ============================================================================
# TEST 1: Health Check
# ============================================================================
print("\n[TEST 1] Health Check - GET /")
print("-" * 80)

try:
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 2: Single Planet Prediction
# ============================================================================
print("\n[TEST 2] Single Planet Prediction - POST /predict")
print("-" * 80)

# Example planet data (scaled features)
sample_planet = {
    "pl_orbper_scaled": -0.39,
    "pl_orbsmax_scaled": 2.51,
    "pl_rade_scaled": -0.06,
    "pl_bmasse_scaled": -0.18,
    "pl_orbeccen_scaled": 6.47,
    "pl_insol_scaled": -0.25,
    "st_teff_scaled": -6.98,
    "st_rad_scaled": -1.33,
    "st_mass_scaled": -3.18,
    "st_met_scaled": 1.34
}

try:
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_planet,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 3: Batch Prediction
# ============================================================================
print("\n[TEST 3] Batch Prediction - POST /predict_batch")
print("-" * 80)

batch_data = {
    "planets": [
        {
            "pl_orbper_scaled": -0.39,
            "pl_orbsmax_scaled": 2.51,
            "pl_rade_scaled": -0.06,
            "pl_bmasse_scaled": -0.18,
            "pl_orbeccen_scaled": 6.47,
            "pl_insol_scaled": -0.25,
            "st_teff_scaled": -6.98,
            "st_rad_scaled": -1.33,
            "st_mass_scaled": -3.18,
            "st_met_scaled": 1.34
        },
        {
            "pl_orbper_scaled": 1.2,
            "pl_orbsmax_scaled": 0.8,
            "pl_rade_scaled": 2.5,
            "pl_bmasse_scaled": 3.1,
            "pl_orbeccen_scaled": 0.1,
            "pl_insol_scaled": 1.5,
            "st_teff_scaled": 0.5,
            "st_rad_scaled": 1.2,
            "st_mass_scaled": 0.9,
            "st_met_scaled": -0.3
        }
    ]
}

try:
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=batch_data,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 4: Top Habitable Planets
# ============================================================================
print("\n[TEST 4] Top Habitable Planets - GET /top_habitable")
print("-" * 80)

try:
    response = requests.get(
        f"{BASE_URL}/top_habitable",
        params={'limit': 5, 'min_confidence': 0.7}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 5: Model Information
# ============================================================================
print("\n[TEST 5] Model Information - GET /model_info")
print("-" * 80)

try:
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    # Print key information
    print(f"Model Type: {result.get('model_type')}")
    print(f"Feature Count: {result.get('feature_count')}")
    
    if 'feature_importance' in result:
        print("\nTop 5 Important Features:")
        importance = result['feature_importance']
        for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
            print(f"  {i}. {feature}: {score:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 6: Dataset Statistics
# ============================================================================
print("\n[TEST 6] Dataset Statistics - GET /stats")
print("-" * 80)

try:
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 7: Error Handling - Invalid Input
# ============================================================================
print("\n[TEST 7] Error Handling - Invalid Input")
print("-" * 80)

invalid_data = {
    "pl_orbper_scaled": "invalid",  # Invalid type
    "pl_orbsmax_scaled": 2.51
}

try:
    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_data,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ API TESTING COMPLETE")
print("=" * 80)