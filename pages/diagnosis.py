import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

st.set_page_config(
    page_title="Diagnosis Report - DermaCare AI",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
        <style>
        @media (max-width: 768px) {
            h1 { font-size: 1.8rem !important; }
            h2 { font-size: 1.4rem !important; }
        }
        
        .stApp { background: #f0f2f6; }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        [data-testid="stSidebar"] * { color: white !important; }
        
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
            border: none !important;
            border-radius: 8px !important;
        }
        
        .diagnosis-card {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def diagnosis_page():
    st.title("📋 Diagnosis Report")
    
    data = st.session_state.patient_data
    
    if not data or 'clinical' not in data:
        st.error("No patient data found. Please start a new assessment.")
        if st.button("Start New Assessment", use_container_width=True):
            st.switch_page("app.py")
        return

    with st.spinner("🧠 Analyzing patient data..."):
        try:
            model, scaler, features = train_model()
            
            inputs = []
            
            for display, col_name in CLINICAL_FEATURES.items():
                val = data['clinical'].get(display, "None (0)")
                if display == "Family History":
                    inputs.append(YES_NO.get(val, 0))
                else:
                    severity_map = {"None (0)": 0, "Mild (1)": 1, "Moderate (2)": 2, "Severe (3)": 3}
                    inputs.append(severity_map.get(val, 0))
            
            if 'histopathology' in data:
                for display, col_name in HISTOPATHOLOGY_FEATURES.items():
                    val = data['histopathology'].get(display, "None (0)")
                    severity_map = {"None (0)": 0, "Mild (1)": 1, "Moderate (2)": 2, "Severe (3)": 3}
                    inputs.append(severity_map.get(val, 0))
            else:
                inputs.extend([0] * len(HISTOPATHOLOGY_FEATURES))
            
            inputs.append(data['age'])
            
            expected_features = len(features)
            if len(inputs) != expected_features:
                if len(inputs) < expected_features:
                    inputs.extend([0] * (expected_features - len(inputs)))
                else:
                    inputs = inputs[:expected_features]
            
            X = np.array(inputs).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]
            
            disease = DISEASES.get(pred, "Unknown")
            confidence = probs[pred - 1] * 100
            
            st.markdown(f"""
            <div class="diagnosis-card">
                <div style="font-size: 4rem;">{DISEASE_ICONS.get(disease, '🏥')}</div>
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">PRIMARY DIAGNOSIS</div>
                <div style="color: white; font-size: 2rem; font-weight: bold;">{disease}</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 0.5rem; display: inline-block;">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Confidence Level")
                st.progress(int(confidence))
                st.markdown(f"""
                <div style="background: #e8f0fe; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <b>📛 Name:</b> {data['name']}<br>
                    <b>🎂 Age:</b> {data['age']} years<br>
                    <b>⏰ Duration:</b> {data['duration']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Differential Diagnoses")
                top = np.argsort(probs)[-4:][::-1]
                for idx in top:
                    other = DISEASES.get(idx + 1, "Unknown")
                    prob = probs[idx] * 100
                    st.markdown(f"**{DISEASE_ICONS.get(other, '')} {other}**")
                    st.progress(int(prob))
            
            st.markdown("---")
            st.caption("⚕️ **Disclaimer:** AI-assisted prediction - Please confirm with a qualified dermatologist")
            
            if st.button("🔄 New Patient Assessment", use_container_width=True):
                st.session_state.patient_data = {}
                st.switch_page("app.py")
            
        except Exception as e:
            st.error(f"❌ Diagnosis Error: {str(e)}")

def main():
    load_css()
    
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 3rem;">🩺</div>
                <h3 style="color: white;">DermaCare AI</h3>
                <p style="color: rgba(255,255,255,0.9);">Version 2.0</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
        if st.button("🩺 Clinical Assessment", use_container_width=True):
            st.switch_page("app.py")
        if st.button("🔬 Histopathology", use_container_width=True):
            st.switch_page("pages/histopathology.py")
        if st.button("📋 Diagnosis Report", use_container_width=True):
            st.switch_page("pages/diagnosis.py")
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.markdown("📧 lalzareabhishek@gmail.com")
    
    diagnosis_page()

if __name__ == "__main__":
    main()
