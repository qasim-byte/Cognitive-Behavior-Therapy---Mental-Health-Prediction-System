from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Build and train model on startup ─────────────────────────────────────────
def generate_samples(n, risk_level):
    np.random.seed(42 + risk_level)
    ranges = {
        0: dict(dep=(1,2), anx=(1,2), iso=(1,2), fut=(1,2), pres=(1,3), fin=(1,3), soc=(3,5), sat=(3,5), slp=(2,3), spt=(2,4)),
        1: dict(dep=(2,3), anx=(2,3), iso=(2,3), fut=(2,3), pres=(2,4), fin=(2,4), soc=(2,4), sat=(2,4), slp=(1,3), spt=(1,3)),
        2: dict(dep=(3,4), anx=(3,4), iso=(3,4), fut=(3,4), pres=(3,5), fin=(3,5), soc=(1,3), sat=(1,3), slp=(1,2), spt=(1,2)),
        3: dict(dep=(4,5), anx=(4,5), iso=(4,5), fut=(4,5), pres=(4,5), fin=(4,5), soc=(1,2), sat=(1,2), slp=(1,2), spt=(1,1)),
    }
    r = ranges[risk_level]
    def ri(lo, hi): return np.random.randint(lo, hi+1, n).clip(1,5)
    data = {
        'depression':         ri(*r['dep']),
        'anxiety':            ri(*r['anx']),
        'isolation':          ri(*r['iso']),
        'future_insecurity':  ri(*r['fut']),
        'academic_pressure':  ri(*r['pres']),
        'financial_concerns': ri(*r['fin']),
        'social_relationships': ri(*r['soc']),
        'study_satisfaction': ri(*r['sat']),
        'average_sleep':      np.random.choice([1,2,3], n, p=[0.3,0.5,0.2] if risk_level>=2 else [0.1,0.4,0.5]),
        'sports_engagement':  np.random.choice([1,2,3,4], n, p=[0.5,0.3,0.15,0.05] if risk_level>=2 else [0.2,0.3,0.3,0.2]),
        'campus_discrimination': np.random.choice([0,1], n, p=[0.6,0.4] if risk_level>=2 else [0.8,0.2]),
        'risk_level': risk_level
    }
    df = pd.DataFrame(data)
    noise_cols = ['depression','anxiety','isolation','future_insecurity',
                  'academic_pressure','financial_concerns','social_relationships','study_satisfaction']
    for col in noise_cols:
        df[col] = (df[col] + np.random.normal(0, 0.3, n)).round().clip(1,5).astype(int)
    return df

def engineer_features(d):
    d = d.copy()
    d['emotional_score'] = (d['depression'] + d['anxiety']) / 2
    d['social_score']    = (d['isolation'] + (6 - d['social_relationships'])) / 2
    d['stress_score']    = (d['academic_pressure'] + d['financial_concerns'] + (6 - d['study_satisfaction'])) / 3
    d['outlook_score']   = d['future_insecurity']
    d['composite']       = (d['emotional_score'] + d['social_score'] + d['stress_score'] + d['outlook_score']) / 4
    d['high_dep_anx']    = ((d['depression'] >= 4) & (d['anxiety'] >= 4)).astype(int)
    d['sleep_stress']    = d['average_sleep'] * d['academic_pressure']
    d['sport_social']    = d['sports_engagement'] * d['social_relationships']
    return d

FEATURE_COLS = ['depression','anxiety','isolation','future_insecurity',
                'academic_pressure','financial_concerns','social_relationships',
                'study_satisfaction','average_sleep','sports_engagement',
                'campus_discrimination','emotional_score','social_score',
                'stress_score','outlook_score','composite','high_dep_anx',
                'sleep_stress','sport_social']

frames = [generate_samples(2000, l) for l in range(4)]
df_all = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
df_feat = engineer_features(df_all)

X = df_feat[FEATURE_COLS]
y = df_feat['risk_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

MODEL = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=2,
                                max_features='sqrt', random_state=42, n_jobs=-1)
MODEL.fit(X_train, y_train)
MODEL_ACCURACY = round(accuracy_score(y_test, MODEL.predict(X_test)) * 100, 2)
print(f"[Model ready] Accuracy: {MODEL_ACCURACY}%")

# ── Prediction logic ──────────────────────────────────────────────────────────
RISK_LABELS = ['Normal', 'Mild', 'Moderate', 'Severe']
RISK_COLORS = ['#1D9E75', '#EF9F27', '#D85A30', '#A32D2D']

DOMAIN_RECOMMENDATIONS = {
    'emotional': {
        0: "Your emotional health looks great. Keep doing what you're doing — regular reflection and self-care go a long way.",
        1: "You may be experiencing occasional low moods or worry. Consider journaling or mindfulness exercises.",
        2: "Noticeable signs of anxiety or low mood. Speaking to a counsellor or trusted person can help significantly.",
        3: "Strong indicators of emotional distress. Please reach out to a mental health professional — you deserve support.",
    },
    'social': {
        0: "Your social connections are healthy. Nurturing these relationships will continue to support your wellbeing.",
        1: "Some social withdrawal detected. Try to schedule regular time with friends or family, even briefly.",
        2: "Significant isolation risk. Join a group, club, or community activity to rebuild social bonds.",
        3: "High isolation. Social connection is critical right now — please reach out to someone you trust today.",
    },
    'stress': {
        0: "Your stress and workload feel manageable. Maintain healthy boundaries to keep it that way.",
        1: "Moderate stress levels. Prioritise tasks, take short breaks, and don't skip sleep.",
        2: "High stress affecting daily function. Consider speaking to a counsellor about workload management.",
        3: "Overwhelming stress levels. Seek academic or workplace support and reduce non-essential commitments urgently.",
    },
    'outlook': {
        0: "You feel positive about your future. Stay curious and keep setting small, achievable goals.",
        1: "Mild uncertainty about the future. Try breaking big goals into smaller steps to build confidence.",
        2: "Future feels uncertain or threatening. Career counselling or goal-setting workshops can help.",
        3: "Severe future insecurity. Talk to a mentor, counsellor, or advisor — a clearer path forward is possible.",
    },
}

OVERALL_RECOMMENDATIONS = {
    0: [
        "Maintain your current sleep and exercise routines.",
        "Continue social activities and hobbies you enjoy.",
        "Periodic self-check-ins (monthly) to stay aware of changes.",
    ],
    1: [
        "Introduce a 10-minute daily mindfulness or breathing practice.",
        "Aim for 7-8 hours of sleep consistently.",
        "Talk to a friend or family member about how you're feeling.",
        "Reduce screen time before bed.",
    ],
    2: [
        "Book an appointment with a counsellor or therapist.",
        "Reduce academic or work commitments where possible.",
        "Engage in at least 3 sessions of physical activity per week.",
        "Avoid caffeine and alcohol as coping mechanisms.",
        "Use campus or workplace mental health resources.",
    ],
    3: [
        "Seek professional mental health support immediately.",
        "Inform a trusted person (family, friend, supervisor) about how you feel.",
        "Contact a mental health helpline if in crisis.",
        "Take medical leave if necessary — your health comes first.",
        "Avoid isolation — stay connected with your support network.",
    ],
}

def score_to_10(raw_score, invert=False):
    """Convert 1-5 scale score to 0-10. If invert=True, higher raw = better."""
    normalized = ((raw_score - 1) / 4) * 10
    return round(10 - normalized if invert else normalized, 1)

def predict(form):
    raw = {
        'depression':           int(form['depression']),
        'anxiety':              int(form['anxiety']),
        'isolation':            int(form['isolation']),
        'future_insecurity':    int(form['future_insecurity']),
        'academic_pressure':    int(form['academic_pressure']),
        'financial_concerns':   int(form['financial_concerns']),
        'social_relationships': int(form['social_relationships']),
        'study_satisfaction':   int(form['study_satisfaction']),
        'average_sleep':        int(form['average_sleep']),
        'sports_engagement':    int(form['sports_engagement']),
        'campus_discrimination':int(form['campus_discrimination']),
    }
    df_input = pd.DataFrame([raw])
    df_input = engineer_features(df_input)
    X_input  = df_input[FEATURE_COLS]

    pred_class = int(MODEL.predict(X_input)[0])
    proba      = MODEL.predict_proba(X_input)[0].tolist()

    # Domain scores (0-10, higher = worse)
    emotional_raw = (raw['depression'] + raw['anxiety']) / 2
    social_raw    = (raw['isolation'] + (6 - raw['social_relationships'])) / 2
    stress_raw    = (raw['academic_pressure'] + raw['financial_concerns'] + (6 - raw['study_satisfaction'])) / 3
    outlook_raw   = raw['future_insecurity']
    composite_raw = (emotional_raw + social_raw + stress_raw + outlook_raw) / 4

    def to_risk(score):
        if score <= 2.0: return 0
        elif score <= 3.0: return 1
        elif score <= 4.0: return 2
        else: return 3

    domains = {
        'emotional': {
            'label': 'Emotional Health',
            'score': round(score_to_10(emotional_raw), 1),
            'risk': to_risk(emotional_raw),
            'risk_label': RISK_LABELS[to_risk(emotional_raw)],
            'color': RISK_COLORS[to_risk(emotional_raw)],
            'recommendation': DOMAIN_RECOMMENDATIONS['emotional'][to_risk(emotional_raw)],
        },
        'social': {
            'label': 'Social Wellbeing',
            'score': round(score_to_10(social_raw), 1),
            'risk': to_risk(social_raw),
            'risk_label': RISK_LABELS[to_risk(social_raw)],
            'color': RISK_COLORS[to_risk(social_raw)],
            'recommendation': DOMAIN_RECOMMENDATIONS['social'][to_risk(social_raw)],
        },
        'stress': {
            'label': 'Stress Resilience',
            'score': round(score_to_10(stress_raw), 1),
            'risk': to_risk(stress_raw),
            'risk_label': RISK_LABELS[to_risk(stress_raw)],
            'color': RISK_COLORS[to_risk(stress_raw)],
            'recommendation': DOMAIN_RECOMMENDATIONS['stress'][to_risk(stress_raw)],
        },
        'outlook': {
            'label': 'Future Outlook',
            'score': round(score_to_10(outlook_raw), 1),
            'risk': to_risk(outlook_raw),
            'risk_label': RISK_LABELS[to_risk(outlook_raw)],
            'color': RISK_COLORS[to_risk(outlook_raw)],
            'recommendation': DOMAIN_RECOMMENDATIONS['outlook'][to_risk(outlook_raw)],
        },
    }

    return {
        'overall_risk': pred_class,
        'overall_label': RISK_LABELS[pred_class],
        'overall_color': RISK_COLORS[pred_class],
        'overall_score': round(score_to_10(composite_raw), 1),
        'confidence': round(max(proba) * 100, 1),
        'proba': {RISK_LABELS[i]: round(proba[i]*100, 1) for i in range(4)},
        'domains': domains,
        'recommendations': OVERALL_RECOMMENDATIONS[pred_class],
        'model_accuracy': MODEL_ACCURACY,
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    result = predict(request.form)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
