## 🚀 ExoHabitAI: The Intelligent Exoplanet Discovery Engine

ExoHabitAI is a full‑stack AI analytics platform that identifies potentially habitable exoplanets from NASA’s
confirmed exoplanet archive. Using a **hybrid tree‑based ML pipeline** and engineered habitability indices, it turns
raw astrophysical parameters into **ranked candidates, interactive dashboards, and decision‑ready insights**.

Live stack (local version in this repo):

- **Backend**: Flask + scikit‑learn + XGBoost
- **ML**: Random Forest / XGBoost pipeline with engineered indices
- **Frontend**: Bootstrap 5, Chart.js, Plotly
- **Data**: NASA Exoplanet Archive CSV (processed via pandas)

---

## 📊 Quick Stats *(example numbers – adapt after training)*

- **1,000+** exoplanets analyzed per run  
- **Top‑K** habitable candidates ranked by score  
- **Multiple models compared** (Random Forest, XGBoost, Logistic Regression)  
- **ROC, precision, recall, F1** tracked and stored in `models/model_metrics.json`  
- **Top‑10 habitable planets** exported to `models/top_habitable_planets.json`

---

## ✨ Key Features

### 🎨 Cinematic User Experience
- **Glassmorphism Design**: Frosted dark UI with purple–cyan gradients  
- **Smooth Animations**: Scroll‑based fade‑ins and KPI counter animations  
- **Space‑themed Background**: Particle field rendered behind the dashboard  
- **Responsive Layout**: Works on desktop, tablet, and mobile

### 🤖 Intelligent ML Engine
- **Engineered Habitability Indices**:  
  - `habitability_score_index` – combines temperature, radius, and orbital distance  
  - `stellar_compatibility_index` – incorporates stellar temperature and luminosity  
- **Robust Preprocessing**: Median/mode imputation, outlier clipping, one‑hot encoding  
- **Model Zoo**: Random Forest (with GridSearch), XGBoost, Logistic Regression  
- **Best Model Persistence**: Saved as `models/best_model.pkl` for reuse in the API

### 📈 Interactive Analytics Dashboard
- **4 KPI Cards**: Total planets, % predicted habitable, average score, model accuracy  
- **Feature Intelligence Panel**: Horizontal gradient bar chart (top 5 features)  
- **Habitability Score Distribution**: Modern donut chart (Chart.js)  
- **Star–Planet Relationship Explorer**: Plotly scatter with zoom, hover, and distance slider  
- **Top‑5 Recommended Planets**: Ranked list with score + confidence badges

### 🛰 Ranking & Discovery
- **Habitability Ranking**: Automatic ordering by model habitability score  
- **Top Candidates**: JSON export of high‑scoring planets for follow‑up analysis  
- **Readable Scores**: “High / Medium / Low” confidence labels for quick triage

---

## 🧠 Dataset & Feature Engineering

**Source dataset (local path expected by the pipeline):**

- `c:\\Users\\tumar\\Desktop\\infy internship\\exoplanet_raw.csv`

The loader in `models/preprocessing.py` uses `pandas.read_csv(..., comment='#')` to skip NASA metadata rows.

### Core Features

NASA column names are mapped into a compact feature set:

- `planet_radius`  
- `planet_mass`  
- `orbital_period`  
- `equilibrium_temp`  
- `distance_from_star`  
- `stellar_temp`  
- `luminosity`  
- `metallicity`  
- `star_type` (categorical → one‑hot encoded)

### Cleaning & Target

The preprocessing pipeline:

- Fills missing **numeric** values with the **median**  
- Fills missing **categorical** values with the **most frequent** value  
- Removes exact duplicates and clips extreme numeric outliers (IQR rule)  
- Creates:
  - `habitability_score_index`  
  - `stellar_compatibility_index`  
- Ensures a binary target `habitability_label` using rule‑based criteria if absent:
  - Temperatures between 240–320 K  
  - Radius between 0.5–2.5 Earth radii  
  - Distance in a habitable‑zone‑like band

---

## 🧪 Models & Evaluation

The training script in `models/train_model.py` compares:

- **Random Forest Classifier** – tuned via `GridSearchCV` (accuracy‑based)  
- **XGBoost Classifier** – gradient boosting baseline  
- **Logistic Regression** – linear baseline

For each model, the pipeline can compute:

- Accuracy  
- Precision  
- Recall  
- F1 score  
- ROC‑AUC (where valid)  
- Confusion matrix and ROC curves  
- Feature importance plots

Results and plots are written to:

- `models/model_metrics.json` – metrics for all models + chosen best  
- `models/plots/` – ROC curve, habitability distribution, star–planet scatter  
- `models/top_habitable_planets.json` – top‑K ranked planets + scores

---

## 🧱 Project Structure

```text
ExoHabitAI/
├─ api.py                 # Flask app + routes + dashboard APIs
├─ templates/             # Multi‑page HTML frontend
│  ├─ home.html           # Landing page
│  ├─ predict.html        # Prediction form + result view
│  ├─ dashboard.html      # Habitability Intelligence Dashboard
│  └─ about.html          # About page + exoplanet facts carousel
├─ models/
│  ├─ preprocessing.py    # Data loading, cleaning, feature engineering, preprocessing
│  ├─ train_model.py      # Model training, selection, top‑planets export
│  ├─ evaluate.py         # Metrics + plots
│  ├─ __init__.py
│  ├─ best_model.pkl      # Saved best model (generated)
│  ├─ model_metrics.json  # Evaluation metrics (generated)
│  ├─ top_habitable_planets.json
│  └─ plots/              # PNG plots (generated)
├─ requirements.txt
└─ README.md
```

If you adopt a `backend/` + `frontend/` split, move `api.py` and `templates/` under `backend/` and update the
`Flask(..., template_folder=...)` configuration accordingly.

---

## ⚙️ Tech Stack

| Layer           | Technology                        |
|----------------|-----------------------------------|
| Frontend       | HTML, Bootstrap 5, vanilla JS     |
| Charts         | Chart.js, Plotly.js               |
| Backend        | Flask                             |
| ML Models      | scikit‑learn, XGBoost             |
| Data           | pandas, NumPy                     |
| Plots          | Matplotlib                        |
| Persistence    | joblib, JSON files                |

---

## 🚀 Quick Start (Local Development)

### 1. Backend & ML environment

```bash
cd c:\Users\tumar\xyz
python -m venv venv
venv\Scripts\activate        # On Windows
python -m pip install -r requirements.txt
```

Train the ML pipeline (optional if `best_model.pkl` already exists):

```bash
python -m models.train_model
```

Run the Flask app:

```bash
python api.py
```

Then open `http://localhost:5000` in your browser.

### 2. Frontend pages

The frontend is rendered directly by Flask using Jinja templates:

- `GET /` → `home.html`  
- `GET /predict` → prediction form (rule‑based scorer for stability)  
- `GET /dashboard` → analytics dashboard (uses `/api/dashboard/*` JSON endpoints)  
- `GET /about` → about + interesting facts carousel

---

## 🔌 API Endpoints (Summary)

### Core

- `GET /health` – health check (`{"status": "ok", "model_loaded": ...}`)  
- `POST /api/predict` – JSON prediction (currently uses rule‑based fallback for robustness)

### Dashboard Data

- `GET /model-metrics` – full metrics JSON  
- `GET /top-planets` – top‑K habitable planets (from training pipeline)  
- `GET /plots/<filename>` – static plots from `models/plots/`  
- `GET /api/dashboard/summary` – KPI numbers (total, % habitable, avg score, accuracy)  
- `GET /api/dashboard/features` – top‑5 feature importances  
- `GET /api/dashboard/distribution` – counts + percentages for Low/Medium/High buckets  
- `GET /api/dashboard/scatter` – star–planet scatter data  
- `GET /api/dashboard/recommendations` – top‑5 ranked planets

---

## ✅ Dashboard Features Checklist

- 4 key metrics: **Total**, **% Habitable**, **Average Score**, **Model Accuracy**  
-  Feature Intelligence Panel with horizontal bar chart  
-  Habitability score donut distribution  
-  Interactive star–planet scatter (zoom, hover, filter)  
-  Top‑5 recommended exoplanets table with confidence badges  
-  Glassmorphic dark‑space UI with subtle animations

---

## 🎯 Project Milestones

- Data preprocessing & feature engineering  
- ML model development & optimization  
- Backend API implementation (Flask)  
- Analytics dashboard UI/UX  
- Model evaluation, metrics export, and plots  
- Top‑planets ranking + recommendation view

