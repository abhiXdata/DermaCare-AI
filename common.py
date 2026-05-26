import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== GLOSSARY DATA ====================
GLOSSARY = {
    "Skin Redness": "🩸 Skin looks red because blood vessels expand - like a mild sunburn",
    "Skin Scaling": "❄️ Dry skin that flakes off like dandruff",
    "Definite Borders": "🗺️ You can clearly see where the rash starts and ends",
    "Itching": "🤚 Feeling that makes you want to scratch your skin",
    "Koebner Phenomenon": "✨ New rash appears exactly where skin was injured",
    "Polygonal Papules": "📐 Small raised bumps with straight edges",
    "Follicular Papules": "⚫ Small bumps around hair roots",
    "Oral Mucosal Involvement": "👄 Rash or sores inside the mouth",
    "Knee/Elbow Involvement": "🦵 Rash specifically on the knees or elbows",
    "Scalp Involvement": "💇 Rash on the head where hair grows",
    "Family History": "👨‍👩‍👧 Blood relatives had similar skin problems",
    "Melanin incontinence": "🎨 Skin pigment leaks out from damaged cells",
    "Eosinophils in infiltrate": "🦠 Allergy cells gather in the skin",
    "PNL infiltrate": "⚔️ Germ-fighting cells rush into the skin",
    "Fibrosis of papillary dermis": "🔧 Small scars form in top skin layer",
    "Exocytosis": "🏃 Fighting cells move into outer skin layer",
    "Acanthosis": "📏 Skin's middle layer gets thicker",
    "Hyperkeratosis": "🛡️ Outer skin layer gets too thick like a callus",
    "Parakeratosis": "🔬 Skin cells don't mature properly",
    "Clubbing of rete ridges": "👊 Finger-like projections become rounded",
    "Elongation of rete ridges": "📈 Skin projections grow longer",
    "Thinning of suprapapillary epidermis": "🥚 Skin becomes very thin above bumps",
    "Spongiform pustule": "💧 Pimple-like pockets of pus form",
    "Munro microabscess": "🔬 Tiny collections of fighting cells",
    "Focal hypergranulosis": "📍 Some spots of grain layer get thicker",
    "Disappearance of granular layer": "👻 Grain layer vanishes in some areas",
    "Vacuolisation of basal layer": "🫧 Bottom cells form empty bubbles",
    "Spongiosis": "💦 Skin cells swell with extra fluid",
    "Saw-tooth appearance": "🪚 Bottom of skin looks jagged like a saw",
    "Follicular horn plug": "🚫 Hair hole gets blocked with dead cells",
    "Perifollicular parakeratosis": "🌀 Immature cells surround hair holes",
    "Inflammatory mononuclear infiltrate": "⏰ Long-term fighting cells gather",
    "Band-like infiltrate": "🎗️ Fighting cells line up like a ribbon"
}

# ==================== DATASET ====================
CLINICAL_FEATURES = {
    "Skin Redness": "erythema",
    "Skin Scaling": "scaling",
    "Definite Borders": "definite_borders",
    "Itching": "itching",
    "Koebner Phenomenon": "koebner_phenomenon",
    "Polygonal Papules": "polygonal_papules",
    "Follicular Papules": "follicular_papules",
    "Oral Mucosal Involvement": "oral_mucosal_involvement",
    "Knee/Elbow Involvement": "knee_and_elbow_involvement",
    "Scalp Involvement": "scalp_involvement",
    "Family History": "family_history"
}

HISTOPATHOLOGY_FEATURES = {
    "Melanin incontinence": "melanin_incontinence",
    "Eosinophils in infiltrate": "eosinophils_in_the_infiltrate",
    "PNL infiltrate": "pnl_infiltrate",
    "Fibrosis of papillary dermis": "fibrosis_of_the_papillary_dermis",
    "Exocytosis": "exocytosis",
    "Acanthosis": "acanthosis",
    "Hyperkeratosis": "hyperkeratosis",
    "Parakeratosis": "parakeratosis",
    "Clubbing of rete ridges": "clubbing_of_the_rete_ridges",
    "Elongation of rete ridges": "elongation_of_the_rete_ridges",
    "Thinning of suprapapillary epidermis": "thinning_of_the_suprapapillary_epidermis",
    "Spongiform pustule": "spongiform_pustule",
    "Munro microabscess": "munro_microabcess",
    "Focal hypergranulosis": "focal_hypergranulosis",
    "Disappearance of granular layer": "disappearance_of_the_granular_layer",
    "Vacuolisation of basal layer": "vacuolisation_and_damage_of_basal_layer",
    "Spongiosis": "spongiosis",
    "Saw-tooth appearance": "saw_tooth_appearance_of_retes",
    "Follicular horn plug": "follicular_horn_plug",
    "Perifollicular parakeratosis": "perifollicular_parakeratosis",
    "Inflammatory mononuclear infiltrate": "inflammatory_monoluclear_inflitrate",
    "Band-like infiltrate": "band_like_infiltrate"
}

YES_NO = {"No": 0, "Yes": 1}

DISEASES = {
    1: "Psoriasis", 2: "Seborrheic Dermatitis", 3: "Lichen Planus",
    4: "Pityriasis Rosea", 5: "Chronic Dermatitis", 6: "Pityriasis Rubra Pilaris"
}

DISEASE_ICONS = {
    "Psoriasis": "🟡", "Seborrheic Dermatitis": "🟠", "Lichen Planus": "🟣",
    "Pityriasis Rosea": "🔴", "Chronic Dermatitis": "🟢", "Pityriasis Rubra Pilaris": "🔵"
}

# ==================== LOAD DATASET ====================
@st.cache_resource
def load_dataset():
    try:
        df = pd.read_csv('dermatology.csv')
        df = df.replace('?', np.nan)
        
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            df['Age'].fillna(df['Age'].median(), inplace=True)
        else:
            df['Age'] = np.random.randint(1, 90, len(df))
        
        for col in df.columns:
            if col != 'Age' and col != 'class':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
        
        clinical_cols = []
        for col in CLINICAL_FEATURES.values():
            if col in df.columns:
                clinical_cols.append(col)
        
        histo_cols = []
        for col in HISTOPATHOLOGY_FEATURES.values():
            if col in df.columns:
                histo_cols.append(col)
        
        clinical_cols.append('Age')
        
        if len(clinical_cols) <= 1:
            return create_demo_data()
        
        return df, clinical_cols, histo_cols
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return create_demo_data()
        
def create_demo_data():
    np.random.seed(42)
    n = 366
    data = {}
    
    for col in CLINICAL_FEATURES.values():
        if col == 'family_history':
            data[col] = np.random.randint(0, 2, n)
        else:
            data[col] = np.random.randint(0, 4, n)
    
    for col in HISTOPATHOLOGY_FEATURES.values():
        data[col] = np.random.randint(0, 4, n)
    
    data['Age'] = np.random.randint(1, 90, n)
    data['class'] = np.random.randint(1, 7, n)
    
    df = pd.DataFrame(data)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
    
    clinical_cols = list(CLINICAL_FEATURES.values()) + ['Age']
    histo_cols = list(HISTOPATHOLOGY_FEATURES.values())
    
    return df, clinical_cols, histo_cols
    
@st.cache_resource
def train_model():
    df, clinical_cols, histo_cols = load_dataset()
    y = df['class']
    all_cols = clinical_cols + histo_cols
    
    X = df[all_cols].copy()
    X = X.fillna(0)
    y = y.fillna(1).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_scaled, y)
    return model, scaler, all_cols
    
@st.cache_data
def get_model_metrics():
    df, clinical_cols, histo_cols = load_dataset()
    y = df['class']
    all_cols = clinical_cols + histo_cols
    
    X = df[all_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler_test = StandardScaler()
    X_train_scaled = scaler_test.fit_transform(X_train)
    X_test_scaled = scaler_test.transform(X_test)
    
    test_model = RandomForestClassifier(random_state=42, n_estimators=100)
    test_model.fit(X_train_scaled, y_train)
    
    y_pred = test_model.predict(X_test_scaled)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='weighted') * 100,
        'recall': recall_score(y_test, y_pred, average='weighted') * 100,
        'f1': f1_score(y_test, y_pred, average='weighted') * 100
    }
