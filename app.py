from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings, datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)

CLINIC_NAME   = "Qasim's MindCheck Model"
REPORT_TITLE  = "Cognitive Behavioral Report"
COUNTRY_LIST  = ['United States','United Kingdom','Canada','Germany',
                 'Netherlands','Ireland','Australia','France','India','Other']

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge():
    """Load both datasets, harmonise features, return combined DataFrame."""

    # ── OSMI Survey (DS1) ────────────────────────────────────────────────────
    df1 = pd.read_csv("./survey.csv")

    def clean_gender(g):
        g = str(g).strip().lower()
        if g in ['male','m','man','cis male','male (cis)','mal','maile',
                 'male-ish','malr','msle','something kinda male?']: return 0
        elif g in ['female','f','woman','cis female','trans-female','femake',
                   'femail','female (trans)','queer/she/they','female ']: return 1
        return 2

    df1['gender_enc']  = df1['Gender'].apply(clean_gender)
    df1['age_clean']   = df1['Age'].apply(lambda x: x if 15<=x<=75 else np.nan).fillna(30)
    df1['age_bucket']  = df1['age_clean'].apply(
        lambda a: 0 if a<25 else (1 if a<35 else (2 if a<45 else 3)))
    df1['country_enc'] = df1['Country'].apply(
        lambda x: COUNTRY_LIST.index(x) if x in COUNTRY_LIST else 9)

    wi_f = {'Never':0.0,'Rarely':0.25,'Sometimes':0.6,'Often':1.0}
    df1['growing_stress']   = df1['work_interfere'].map(wi_f).fillna(0.5)
    df1['changes_habits']   = df1['mental_health_consequence'].map(
        {'No':0,'Maybe':0.5,'Yes':1}).fillna(0.5)
    df1['mh_history']       = 0.5
    df1['coping_struggles'] = df1['work_interfere'].map(
        {'Often':1,'Sometimes':0.6,'Rarely':0.2,'Never':0}).fillna(0.4)
    df1['work_interest']    = 1 - df1['work_interfere'].map(
        {'Often':0,'Sometimes':0.3,'Rarely':0.7,'Never':1}).fillna(0.5)
    df1['social_weakness']  = df1['coworkers'].map(
        {'Yes':0,'Some of them':0.5,'No':1}).fillna(0.5)
    df1['mh_interview']     = df1['mental_health_interview'].map(
        {'No':1,'Maybe':0.5,'Yes':0}).fillna(1)
    df1['care_options']     = df1['care_options'].map(
        {'No':2,'Not sure':1,'Yes':0}).fillna(1)
    df1['family_risk']      = (df1['family_history']=='Yes').astype(float)
    df1['has_treatment']    = (df1['treatment']=='Yes').astype(float)
    df1['mood_swings']      = df1['work_interfere'].map(
        {'Often':2,'Sometimes':1,'Rarely':0.5,'Never':0}).fillna(1)
    df1['days_indoors']     = df1['remote_work'].map({'Yes':3,'No':1}).fillna(1)
    df1['occupation']       = 0  # mostly corporate/tech

    # DS1 risk target
    wi_r = {'Never':1,'Rarely':2,'Sometimes':3,'Often':4}
    mhc_r= {'No':0,'Maybe':1,'Yes':2}
    lv_r = {'Very easy':0,'Somewhat easy':1,"Don't know":2,
             'Somewhat difficult':3,'Very difficult':4}
    sh_r = {'Yes':0,"Don't know":1,'No':2}
    bn_r = {'Yes':0,"Don't know":1,'No':2}
    df1['_ws'] = df1['work_interfere'].map(wi_r).fillna(2)
    df1['_sf'] = df1['mental_health_consequence'].map(mhc_r).fillna(1)
    df1['_ob'] = (df1['obs_consequence']=='Yes').astype(int)
    df1['_nb'] = df1['benefits'].map(bn_r).fillna(1)
    df1['_ld'] = df1['leave'].map(lv_r).fillna(2)
    df1['_sh'] = df1['seek_help'].map(sh_r).fillna(1)
    df1['_re'] = (df1['remote_work']=='Yes').astype(int)
    df1['risk_raw'] = (df1['_ws'] + df1['family_risk'] + df1['has_treatment'] +
                       df1['_sf'] + df1['_ob']*0.5 + df1['_nb']*0.5 +
                       df1['_ld']*0.4 + df1['_sh']*0.5 + df1['_re']*0.3)
    df1['risk_level'] = pd.cut(df1['risk_raw'],
        bins=[0,3.5,5.5,7.5,100], labels=[0,1,2,3]).astype(int)

    # ── Mental Health Dataset (DS2) ──────────────────────────────────────────
    df2 = pd.read_csv("./Mental_Health_Dataset.csv")

    yn = {'Yes':1,'No':0,'Maybe':0.5}
    df2['growing_stress']   = df2['Growing_Stress'].map(yn).fillna(0.5)
    df2['changes_habits']   = df2['Changes_Habits'].map(yn).fillna(0.5)
    df2['mh_history']       = df2['Mental_Health_History'].map(yn).fillna(0.5)
    df2['coping_struggles'] = df2['Coping_Struggles'].map({'Yes':1,'No':0}).fillna(0)
    df2['work_interest']    = df2['Work_Interest'].map(
        {'Yes':0,'Maybe':0.5,'No':1}).fillna(0.5)
    df2['social_weakness']  = df2['Social_Weakness'].map(yn).fillna(0.5)
    df2['mh_interview']     = df2['mental_health_interview'].map(
        {'Yes':0,'Maybe':0.5,'No':1}).fillna(1)
    df2['care_options']     = df2['care_options'].map(
        {'Yes':0,'Not sure':1,'No':2}).fillna(1)
    df2['family_risk']      = (df2['family_history']=='Yes').astype(float)
    df2['has_treatment']    = (df2['treatment']=='Yes').astype(float)
    df2['mood_swings']      = df2['Mood_Swings'].map(
        {'Low':0,'Medium':1,'High':2}).fillna(1)
    df2['days_indoors']     = df2['Days_Indoors'].map(
        {'Go out Every day':0,'1-14 days':1,'15-30 days':2,
         '31-60 days':3,'More than 2 months':4}).fillna(2)
    df2['occupation']       = df2['Occupation'].map(
        {'Corporate':0,'Business':1,'Student':2,'Housewife':3,'Others':4}).fillna(4)
    df2['gender_enc']       = df2['Gender'].map({'Male':0,'Female':1}).fillna(0).astype(int)
    df2['country_enc']      = df2['Country'].apply(
        lambda x: COUNTRY_LIST.index(x) if x in COUNTRY_LIST else 9)
    df2['age_bucket']       = 1  # not available

    def r2(row):
        s  = row['growing_stress']*2 + row['coping_struggles']*2
        s += row['mood_swings'] + row['social_weakness'] + row['work_interest']
        s += row['changes_habits'] + row['mh_history']
        s += row['days_indoors']*0.5 + row['care_options']*0.3 + row['family_risk']*0.5
        return s
    df2['risk_raw']   = df2.apply(r2, axis=1)
    df2['risk_level'] = pd.cut(df2['risk_raw'],
        bins=[0,2.5,5.0,7.5,100], labels=[0,1,2,3]).astype(int)

    # ── Interaction features ─────────────────────────────────────────────────
    def interactions(df):
        df = df.copy()
        df['stress_coping']  = df['growing_stress'] * df['coping_struggles']
        df['mood_social']    = df['mood_swings']    * df['social_weakness']
        df['indoor_stress']  = df['days_indoors']   * df['growing_stress']
        df['care_history']   = df['care_options']   * df['mh_history']
        return df

    df1 = interactions(df1)
    df2 = interactions(df2)

    # ── Combine ──────────────────────────────────────────────────────────────
    COLS = BASE_COLS + INTERACTION_COLS
    ds2_sample = df2.sample(30000, random_state=42)
    ds1_rep    = pd.concat([df1]*8, ignore_index=True)   # balance DS1
    combined   = pd.concat(
        [ds1_rep[COLS+['risk_level']], ds2_sample[COLS+['risk_level']]],
        ignore_index=True).dropna()
    return combined

BASE_COLS = [
    'growing_stress','changes_habits','mh_history','coping_struggles',
    'work_interest','social_weakness','mh_interview','care_options',
    'family_risk','has_treatment','mood_swings','days_indoors',
    'gender_enc','country_enc','occupation','age_bucket'
]
INTERACTION_COLS = ['stress_coping','mood_social','indoor_stress','care_history']
ALL_COLS = BASE_COLS + INTERACTION_COLS

# ── Train model at startup ─────────────────────────────────────────────────────
print("[MindCheck] Loading datasets and training model...")
combined = load_and_merge()
X = combined[ALL_COLS].values
y = combined['risk_level'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

MODEL = RandomForestClassifier(
    n_estimators=400, max_depth=20, min_samples_leaf=1,
    max_features='sqrt', random_state=42, n_jobs=-1)
MODEL.fit(X_train, y_train)
MODEL_ACCURACY = round(accuracy_score(y_test, MODEL.predict(X_test)) * 100, 2)
print(f"[MindCheck] Model ready — Accuracy: {MODEL_ACCURACY}%")

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

RISK_LABELS = ['Normal', 'Mild', 'Moderate', 'Severe']
RISK_COLORS = ['#2A9E6E', '#E8A020', '#D05020', '#9E1C1C']
RISK_BG     = ['#E8F8F0', '#FEF3E2', '#FCEAE4', '#FAE8E8']

DOMAIN_REC = {
    'emotional': {
        0: "Emotional equilibrium is strong. Sustain it through regular reflection, quality sleep, and social engagement.",
        1: "Occasional low moods or anxious thoughts are present. Brief daily mindfulness or journalling can help stabilise.",
        2: "Noticeable emotional strain detected. A conversation with a counsellor or trusted person is strongly advisable.",
        3: "Significant emotional distress indicated. Please reach out to a mental health professional — support is available.",
    },
    'social': {
        0: "Social connections are healthy and supportive. Keep nurturing these bonds — they are a protective factor.",
        1: "Mild social withdrawal present. Scheduling regular low-pressure contact with others can help.",
        2: "Moderate isolation risk detected. Joining a group, class, or community activity can rebuild connection.",
        3: "High isolation risk. Human connection is critical right now — please reach out to someone you trust today.",
    },
    'stress': {
        0: "Stress and workload feel manageable. Maintain current boundaries and healthy coping routines.",
        1: "Moderate stress present. Prioritise tasks, take structured breaks, and protect sleep hours.",
        2: "High stress affecting functioning. Consider speaking to a counsellor about workload and pressure management.",
        3: "Overwhelming stress levels. Seek academic or workplace support and reduce non-essential commitments urgently.",
    },
    'outlook': {
        0: "A positive and grounded outlook on the future. Continue setting small, meaningful goals.",
        1: "Mild uncertainty about the future. Break larger goals into achievable milestones for clarity.",
        2: "Future feels uncertain or threatening. Career or goal-setting guidance can help create a clearer path.",
        3: "Severe future insecurity detected. A mentor, counsellor, or advisor can help you find direction and hope.",
    },
}

OVERALL_REC = {
    0: [
        "Maintain consistent sleep, exercise, and social routines.",
        "Continue hobbies and activities that bring you meaning.",
        "Perform brief monthly self-check-ins to track any changes.",
    ],
    1: [
        "Introduce 10 minutes of daily mindfulness or breathing exercises.",
        "Target 7–8 hours of sleep each night consistently.",
        "Talk openly with a trusted friend or family member.",
        "Reduce passive screen time, especially before bed.",
    ],
    2: [
        "Schedule a consultation with a counsellor or therapist.",
        "Reduce academic or work commitments where feasible.",
        "Engage in physical activity at least 3 times per week.",
        "Avoid relying on caffeine, alcohol, or substances to cope.",
        "Explore mental health resources available in your institution or workplace.",
    ],
    3: [
        "Seek professional mental health support as a matter of priority.",
        "Inform a trusted person — family, friend, or manager — about how you are feeling.",
        "Contact a mental health helpline if you feel in crisis.",
        "Take medical leave if required; your health must come first.",
        "Actively resist isolation — stay connected with your support network.",
    ],
}

COUNTRY_RESOURCES = {
    'United States':  "SAMHSA helpline: 1-800-662-4357. Employee Assistance Programmes (EAPs) are available at most workplaces.",
    'United Kingdom': "NHS Talking Therapies and Mind UK (mind.org.uk) offer free mental health support.",
    'Canada':         "Crisis Services Canada: 1-833-456-4566. Provincial health plans cover counselling.",
    'Germany':        "German statutory health insurance (GKV) covers psychotherapy. Ask your GP for a referral.",
    'Australia':      "Beyond Blue (beyondblue.org.au) and Lifeline (13 11 14) offer free national support.",
    'Netherlands':    "GGZ care is available through Dutch health insurance. Contact your huisarts (GP) for referral.",
    'Ireland':        "Pieta House (116 123) and HSE mental health services are available nationally.",
    'France':         "Numéro National de Prévention du Suicide: 3114. CMPs (mental health centres) are free.",
    'India':       "Visit openwho.org or contact a local GP. The WHO Mental Health Atlas lists services globally.",
    'Other':          "For immediate mental health emergencies, call the Umang Pakistan 24/7 Helpline at 0311-7786264. For general medical emergencies, dial 1122 (Rescue 1122) nationwide.",
}

AGE_INSIGHTS = {
    'Under 25': "Young adults experience peak onset of many mental health conditions. Early support leads to significantly better outcomes.",
    '25–34':    "Early-career pressure, financial stress, and identity transitions are common drivers in this age group.",
    '35–44':    "Mid-life stress often stems from compounded responsibilities. Work-life balance is a key intervention point.",
    '45+':      "Greater life experience offers resilience, but social isolation risk can increase. Regular connection is protective.",
}

GENDER_NOTES = {
    'Male':   "Men statistically under-report distress and are less likely to seek help — yet early intervention has strong positive outcomes. Reaching out is a sign of strength.",
    'Female': "Women tend to seek support more readily, which is strongly associated with better recovery. Your self-awareness is a genuine asset.",
    'Other':  "Non-binary and gender-diverse individuals often face compounding stressors. Affirming, inclusive mental health care is available and effective.",
}

def score_to_10(raw, invert=False):
    norm = ((raw - 1) / 4) * 10
    return round(10 - norm if invert else norm, 1)

def to_risk(score, lo=(1.5, 2.5, 3.5)):
    if score <= lo[0]: return 0
    elif score <= lo[1]: return 1
    elif score <= lo[2]: return 2
    return 3

def build_feature_vector(form, gender_enc, age_bucket, country_enc):
    """Map user's form inputs → model feature vector."""
    dep  = int(form['depression'])
    anx  = int(form['anxiety'])
    iso  = int(form['isolation'])
    soc  = int(form['social_relationships'])
    pres = int(form['academic_pressure'])
    fin  = int(form['financial_concerns'])
    sat  = int(form['study_satisfaction'])
    slp  = int(form['average_sleep'])
    spt  = int(form['sports_engagement'])
    disc = int(form['campus_discrimination'])
    fam  = int(form['family_history'])
    hs   = int(form['help_seeking'])        # 1=likely,2=maybe,3=unlikely
    stg  = int(form['stigma_concern'])      # 1-5
    ms   = int(form['mood_swings_q'])       # 1=low,2=med,3=high
    di   = int(form['days_indoors_q'])      # 1-5
    occ  = int(form['occupation'])          # 0-4

    # Map to feature space
    growing_stress   = (dep/5 + anx/5 + pres/5) / 3
    changes_habits   = stg / 5
    mh_history       = fam * 0.8
    coping_struggles = (dep/5 * 0.5 + (1 - sat/5)*0.3 + fin/5*0.2)
    work_interest    = (1 - sat/5)*0.6 + fin/5*0.2 + pres/5*0.2
    social_weakness  = iso/5 * 0.6 + (1 - soc/5)*0.4
    mh_interview     = (hs - 1) / 2          # 0=seeks,1=maybe,2=avoids
    care_options_f   = (hs - 1)               # 0,1,2
    family_risk      = float(fam)
    has_treatment    = 1.0 if hs == 1 else 0.0
    mood_swings      = float(ms - 1)          # 0,1,2
    days_indoors     = float(di - 1)          # 0-4

    # Interaction features
    stress_coping  = growing_stress * coping_struggles
    mood_social    = mood_swings    * social_weakness
    indoor_stress  = days_indoors   * growing_stress
    care_history   = care_options_f * mh_history

    return np.array([[
        growing_stress, changes_habits, mh_history, coping_struggles,
        work_interest, social_weakness, mh_interview, care_options_f,
        family_risk, has_treatment, mood_swings, days_indoors,
        gender_enc, country_enc, occ, age_bucket,
        stress_coping, mood_social, indoor_stress, care_history
    ]], dtype=float)


def predict(form):
    # Demographic inputs
    gender_map  = {'Male':0,'Female':1,'Other':2}
    age_map     = {'Under 25':0,'25–34':1,'35–44':2,'45+':3}

    gender_str  = form.get('gender','Male')
    age_str     = form.get('age_group','25–34')
    country_str = form.get('country','Other')
    patient_name= form.get('patient_name','Anonymous').strip() or 'Anonymous'

    gender_enc  = gender_map.get(gender_str, 0)
    age_bucket  = age_map.get(age_str, 1)
    country_enc = COUNTRY_LIST.index(country_str) if country_str in COUNTRY_LIST else 9

    X_input = build_feature_vector(form, gender_enc, age_bucket, country_enc)

    pred_class = int(MODEL.predict(X_input)[0])
    proba      = MODEL.predict_proba(X_input)[0].tolist()

    # Domain raw scores for display (1-5 scale)
    dep = int(form['depression'])
    anx = int(form['anxiety'])
    iso = int(form['isolation'])
    soc = int(form['social_relationships'])
    pres= int(form['academic_pressure'])
    fin = int(form['financial_concerns'])
    sat = int(form['study_satisfaction'])

    emo_raw  = (dep + anx) / 2
    soc_raw  = (iso + (6 - soc)) / 2
    str_raw  = (pres + fin + (6 - sat)) / 3
    out_raw  = int(form['future_insecurity'])
    comp_raw = (emo_raw + soc_raw + str_raw + out_raw) / 4

    domains = {}
    for key, label, val in [
        ('emotional', 'Emotional Health',   emo_raw),
        ('social',    'Social Wellbeing',    soc_raw),
        ('stress',    'Stress Resilience',   str_raw),
        ('outlook',   'Future Outlook',      out_raw),
    ]:
        r = to_risk(val, lo=(2.0, 3.0, 4.0))
        domains[key] = {
            'label': label,
            'score': round(score_to_10(val), 1),
            'risk': r,
            'risk_label': RISK_LABELS[r],
            'color': RISK_COLORS[r],
            'bg':    RISK_BG[r],
            'recommendation': DOMAIN_REC[key][r],
        }

    country_resource = COUNTRY_RESOURCES.get(country_str, COUNTRY_RESOURCES['Other'])
    family_note = (
        "Family history of mental illness is a documented risk factor. "
        "Regular check-ins with a healthcare professional are advisable."
        if int(form.get('family_history', 0)) == 1 else ""
    )

    recs = list(OVERALL_REC[pred_class])
    recs.append(f"📍 {country_str} resource: {country_resource}")

    now = datetime.datetime.now()
    report_date = f"{now.day} {now.strftime('%B %Y')}"

    return {
        'patient_name':   patient_name,
        'report_date':    report_date,
        'clinic_name':    CLINIC_NAME,
        'report_title':   REPORT_TITLE,
        'overall_risk':   pred_class,
        'overall_label':  RISK_LABELS[pred_class],
        'overall_color':  RISK_COLORS[pred_class],
        'overall_bg':     RISK_BG[pred_class],
        'overall_score':  round(score_to_10(comp_raw), 1),
        'confidence':     round(max(proba) * 100, 1),
        'proba':          {RISK_LABELS[i]: round(proba[i]*100,1) for i in range(4)},
        'domains':        domains,
        'recommendations': recs,
        'model_accuracy':  MODEL_ACCURACY,
        'demo': {
            'gender':       gender_str,
            'age_group':    age_str,
            'country':      country_str,
            'gender_note':  GENDER_NOTES.get(gender_str, ''),
            'age_insight':  AGE_INSIGHTS.get(age_str, ''),
            'family_note':  family_note,
            'country_note': country_resource,
        }
    }

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html',
                           clinic_name=CLINIC_NAME,
                           report_title=REPORT_TITLE,
                           model_accuracy=MODEL_ACCURACY)

@app.route('/predict', methods=['POST'])
def predict_route():
    result = predict(request.form)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
