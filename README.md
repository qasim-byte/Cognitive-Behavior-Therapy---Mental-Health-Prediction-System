# MindCheck v3 — Cognitive Behavioral Report
## Qasim's MindCheck Model

### What's new vs v2

| Feature | v2 | v3 |
|---|---|---|
| Datasets | OSMI survey (1,259 rows) | OSMI + Mental Health Dataset (293K+ rows) |
| Model accuracy | ~88% | ~98% |
| Interaction features | ❌ | ✅ stress×coping, mood×social, etc. |
| Report title | Generic | "Cognitive Behavioral Report" |
| Patient name | ❌ | ✅ Appears on report header |
| UI design | Minimal | Editorial clinical letterhead |
| Days indoors feature | ❌ | ✅ (from DS2) |
| Mood swings feature | ❌ | ✅ (from DS2) |
| Occupation feature | ❌ | ✅ (Corporate/Student/etc.) |

### Run locally

```bash
pip install -r requirements.txt

# Both CSVs must be in the same directory as app.py:
# - survey.csv
# - Mental_Health_Dataset.csv

python app.py
# Open http://localhost:5000
```

### Data sources
1. OSMI Mental Health in Tech Survey (2014) — 1,259 responses
2. Mental Health Dataset (Kaggle) — 292,364 responses

### Features used by model
- Growing stress (work/academic pressure)
- Changes in habits (behavioral shifts)
- Mental health history
- Coping struggles
- Work/study interest
- Social weakness / isolation
- Mental health interview willingness
- Care options availability
- Family risk history
- Treatment history
- Mood swings
- Days spent indoors
- Gender, country, occupation, age group
- Interaction terms: stress×coping, mood×social, indoor×stress, care×history
