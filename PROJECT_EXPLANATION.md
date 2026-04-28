# MindCheck v3 — Project Explanation

## 1. Project overview
This project is a mental health risk assessment web app built with `Flask` and a machine learning model (`RandomForestClassifier`). It reads two datasets, transforms survey-based data into numerical features, trains a model on the combined dataset, and then uses a web form to collect user responses and generate a risk report.

The main files are:
- `app.py` — main Flask application, data preparation, model training, prediction logic
- `requirements.txt` — required Python packages
- `templates/index.html` — form page where the user enters responses
- `templates/result.html` — report page showing the prediction
- `survey.csv` — first survey dataset
- `Mental_Health_Dataset.csv` — second, larger dataset

---

## 2. What the major imports do
```python
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings, datetime
```

- `Flask` creates the web application.
- `render_template` loads HTML templates from the `templates/` folder.
- `request` reads user data submitted through the web form.
- `numpy` (`np`) handles numeric arrays and lightweight math.
- `pandas` (`pd`) loads CSV files and manipulates dataset tables.
- `RandomForestClassifier` is the machine learning model used for prediction.
- `train_test_split` divides data into training and testing sets.
- `accuracy_score` computes how well the trained model performs.
- `warnings.filterwarnings('ignore')` hides non-critical warnings.
- `datetime` is used to include the current date on the report.

---

## 3. Flask basics and how deployment works
In `app.py`, `app = Flask(__name__)` creates the application.

The two Flask routes are:
- `@app.route('/')` — displays the main assessment form using `index.html`
- `@app.route('/predict', methods=['POST'])` — receives form input and returns `result.html`

The app is started by running `python app.py` and listens on port `5000` because of:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

That means the app runs locally and can be opened at `http://localhost:5000`.

---

## 4. Where training starts
Training begins after the feature-engineering section in `app.py` under the comment:
```python
# ── Train model at startup ─────────────────────────────────────────────────────
```

The actual steps are:
1. `combined = load_and_merge()` loads and prepares the data.
2. `X = combined[ALL_COLS].values` selects the feature columns.
3. `y = combined['risk_level'].values` selects the target labels.
4. `train_test_split(...)` splits the data into train and test sets.
5. `MODEL = RandomForestClassifier(...)` creates the classifier.
6. `MODEL.fit(X_train, y_train)` trains the model.
7. `MODEL_ACCURACY = round(accuracy_score(...), 2)` computes accuracy.

So training is performed once when the server starts.

---

## 5. Where data cleaning and feature engineering happens
The function `load_and_merge()` is the heart of cleaning, engineering, and merging data.

It does this in three major sections:
- DS1: load and prepare `survey.csv`
- DS2: load and prepare `Mental_Health_Dataset.csv`
- Combine: merge the two datasets into one training dataset

### 5.1 DS1 cleaning and feature mapping
Inside `load_and_merge()`, the project reads `survey.csv` with:
```python
df1 = pd.read_csv('survey.csv')
```

Then it converts raw text answers into numeric features, for example:
- `gender_enc` from `Gender`
- `age_clean` and `age_bucket` from `Age`
- `country_enc` from `Country`
- `growing_stress` from `work_interfere`
- `changes_habits` from `mental_health_consequence`
- `coping_struggles`, `work_interest`, `social_weakness`, `mh_interview`, `care_options`
- `family_risk` and `has_treatment`

The code also builds a synthetic risk label, `risk_level`, from multiple survey fields. That label becomes the target the model learns.

### 5.2 DS2 cleaning and feature mapping
The second dataset is loaded via:
```python
df2 = pd.read_csv('Mental_Health_Dataset.csv')
```

It maps similar fields into the same feature names used for training, for example:
- `Growing_Stress` → `growing_stress`
- `Changes_Habits` → `changes_habits`
- `Mental_Health_History` → `mh_history`
- `Coping_Struggles` → `coping_struggles`
- `Work_Interest` → `work_interest`
- `Social_Weakness` → `social_weakness`
- `family_history`, `treatment`, `Mood_Swings`, `Days_Indoors`
- `Occupation`, `Gender`, `Country`

It also computes `risk_raw` and bins that into `risk_level`.

### 5.3 Interaction features
After both datasets have their base features, an `interactions()` helper creates additional features:
- `stress_coping = growing_stress * coping_struggles`
- `mood_social = mood_swings * social_weakness`
- `indoor_stress = days_indoors * growing_stress`
- `care_history = care_options * mh_history`

These capture combined effects of multiple factors.

---

## 6. Where dataset merging happens
The merge occurs at the end of `load_and_merge()`:
```python
COLS = BASE_COLS + INTERACTION_COLS

ds2_sample = df2.sample(30000, random_state=42)
ds1_rep    = pd.concat([df1]*8, ignore_index=True)
combined   = pd.concat(
    [ds1_rep[COLS+['risk_level']], ds2_sample[COLS+['risk_level']]],
    ignore_index=True).dropna()
```

Key points:
- `ds2_sample` randomly selects 30,000 rows from `df2` to keep training size manageable.
- `ds1_rep` repeats `df1` eight times to give the smaller dataset more weight.
- `pd.concat([...])` vertically joins both datasets into one combined table.
- `.dropna()` removes any rows with missing values.

The final dataset is used to train the model.

---

## 7. What the model learns
The model uses these feature columns:
```python
BASE_COLS = [
    'growing_stress','changes_habits','mh_history','coping_struggles',
    'work_interest','social_weakness','mh_interview','care_options',
    'family_risk','has_treatment','mood_swings','days_indoors',
    'gender_enc','country_enc','occupation','age_bucket'
]
INTERACTION_COLS = ['stress_coping','mood_social','indoor_stress','care_history']
ALL_COLS = BASE_COLS + INTERACTION_COLS
```

The target column is:
- `risk_level` — an integer label from 0 to 3 representing Normal / Mild / Moderate / Severe risk.

The model is a random forest set up as:
```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

This means:
- 400 decision trees are trained.
- each tree is limited to depth 20 to avoid overfitting too much.
- the model uses square-root-of-feature random sampling per tree split.
- training uses all CPU cores (`n_jobs=-1`).

---

## 8. How the prediction path works
The prediction path is handled by the `predict()` function.

### Step-by-step:
1. Read demographic form fields from `request.form`.
2. Map form fields to numeric encodings:
   - `gender` → `gender_enc`
   - `age_group` → `age_bucket`
   - `country` → `country_enc`
3. Call `build_feature_vector(form, gender_enc, age_bucket, country_enc)`.
4. Inside `build_feature_vector()`, the form values are converted into the same features used by training.
5. The feature vector is passed to `MODEL.predict(X_input)`.
6. `MODEL.predict_proba(X_input)` returns class probabilities.
7. The app also computes display metrics such as:
   - emotional, social, stress, outlook scores
   - overall report score
   - recommendations and notes
8. The result dictionary is passed to `render_template('result.html', result=result)`.

### What the feature builder does
`build_feature_vector()` uses form values like:
- `depression`, `anxiety`, `isolation`, `social_relationships`
- `academic_pressure`, `financial_concerns`, `study_satisfaction`
- `help_seeking`, `stigma_concern`, `mood_swings_q`, `days_indoors_q`
- `occupation`, `family_history`

It converts them into the training feature space using formulas such as:
- `growing_stress = (dep/5 + anx/5 + pres/5) / 3`
- `coping_struggles = (dep/5 * 0.5 + (1 - sat/5) * 0.3 + fin/5 * 0.2)`
- `work_interest = (1 - sat/5) * 0.6 + fin/5 * 0.2 + pres/5 * 0.2`
- `social_weakness = iso/5 * 0.6 + (1 - soc/5) * 0.4`
- plus the same interaction features used for training.

---

## 9. What the result page shows
The result page uses the returned `result` dictionary to display:
- `patient_name`
- `report_date`
- overall risk category (`Normal`, `Mild`, `Moderate`, `Severe`)
- confidence percentage
- four domain summaries with risk colors and recommendations
- demographic context and country-specific resource notes
- a professional-looking report layout built in `templates/result.html`

This templated report is generated dynamically by Flask and shown in the browser.

---

## 10. How the form page works
The form page `templates/index.html` contains:
- patient profile questions (name, gender, age, country, occupation)
- emotional health questions
- social wellbeing questions
- stress and functional capacity questions
- future outlook and lifestyle questions
- submit button that sends a POST request to `/predict`

The page uses HTML form controls and simple JavaScript to show progress as questions are answered.

---

## 11. Beginner-level summary
If you are new to this kind of project, the simplest flow is:
1. Start the app: `python app.py`
2. Open `http://localhost:5000`
3. Fill the questionnaire and submit.
4. The app uses a trained machine learning model to predict a risk level.
5. The app returns a report page with advice and a risk classification.

It is a standard Flask app with a machine learning model embedded in the server.

---

## 12. Intermediate-level understanding
At the intermediate level, learn these key ideas:
- `pandas` is used to load and clean data from CSV files.
- `numpy` is used for numeric arrays and final feature vectors.
- Categorical text fields are converted to numbers before training.
- `RandomForestClassifier` is the algorithm that learns from the data.
- The app creates both base features and interaction features.
- A `Flask` route can render HTML templates and accept form submissions.

One important concept is that the app trains the model once at startup, and then uses that fixed model for every user request.

---

## 13. Advanced-level details
For advanced readers, here are deeper details:

### Data balancing and sampling
- `df1` is repeated eight times using `pd.concat([df1]*8)` to increase its influence.
- `df2` is sampled down to 30,000 rows using `df2.sample(30000, random_state=42)`.
- This balancing strategy reduces the dominance of the larger dataset.

### Risk target engineering
- In DS1, a `risk_raw` score is constructed from many survey fields and then cut into bins.
- In DS2, another `risk_raw` formula is created and cut into the same four bins.
- This makes both datasets share the same `risk_level` label definitions.

### Feature design
- `BASE_COLS` contains direct feature values.
- `INTERACTION_COLS` contains product features that capture relationships between variables.
- These interaction terms can help the model learn non-linear effects.

### Model setup
- `n_estimators=400` means many trees, which improves stability.
- `max_depth=20` caps tree growth to prevent extreme overfitting.
- `max_features='sqrt'` means each split sees only a subset of features, increasing diversity.

### Prediction formatting
- `predict()` also computes domain risk scores for emotional, social, stress, and outlook.
- It converts those to 0–3 risk categories with `to_risk()` and `score_to_10()`.
- It builds recommendations from static dictionaries such as `OVERALL_REC`, `DOMAIN_REC`, and `COUNTRY_RESOURCES`.

### Template rendering
- `render_template('index.html', ...)` sends values into Jinja2 templates.
- `render_template('result.html', result=result)` renders the prediction results inside HTML.
- The user sees a styled report without writing raw HTML in Python.

---

## 14. How to run the project
1. Install packages:
```bash
pip install -r requirements.txt
```
2. Make sure both CSV files are in the same folder as `app.py`.
3. Run:
```bash
python app.py
```
4. Open the browser at:
```text
http://localhost:5000
```

---

## 15. Useful notes for improvement
- The model trains when the app starts, which means startup is slower but prediction is fast.
- The dataset balancing is simple; a more advanced approach would use stratified sampling.
- The `result.html` template is static HTML with Jinja placeholders; it can be extended with richer visualizations.
- If you want to deploy publicly, use a production server like `gunicorn` or `Waitress` instead of `app.run(debug=True)`.

---

## 16. Fast reference for key code locations
- `app.py` import section: basic dependencies and libraries
- `load_and_merge()`: data cleaning, mapping, interaction features, dataset merge
- `BASE_COLS` / `INTERACTION_COLS`: feature list definition
- training section: `MODEL.fit(...)`
- `build_feature_vector()`: how user input becomes model input
- `predict()`: prediction logic and report packaging
- Flask routes: `/` and `/predict`
- HTML templates: `templates/index.html`, `templates/result.html`

---

## 17. Conclusion
This project combines web development and machine learning.
- `Flask` creates the user interface and handles form submission.
- `pandas` and `numpy` preprocess data.
- `RandomForestClassifier` learns from labeled survey responses.
- The app returns a human-readable mental health risk report.

It is a good example of a small end-to-end ML web app with both beginner and advanced elements.
