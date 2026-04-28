# MindCheck — Mental Health Assessment App

A Flask-based mental health assessment web app that predicts risk levels
across 4 domains using a Random Forest classifier (96.25% accuracy).

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   python app.py

3. Open browser at:
   http://localhost:5000

## How it works

- User answers 11 questions across 4 domains
- Random Forest model predicts one of 4 risk levels:
    Normal / Mild / Moderate / Severe
- Results shown as HbA1c-style domain scores (0-10 each)
- Personalised recommendations per domain + overall

## Model details

- Algorithm: Random Forest (300 trees, max_depth=15)
- Training data: 8,000 balanced synthetic samples
  (grounded in real OSMI + Student Mental Health Survey distributions)
- Test accuracy: 96.25%
- 5-Fold CV: 96.88% ± 0.39%
- 4 classes: Normal, Mild, Moderate, Severe

## Disclaimer

This app is for educational and demonstration purposes only.
It is not a clinical tool and should not replace professional medical advice.
