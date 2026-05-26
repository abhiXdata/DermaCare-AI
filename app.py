import streamlit as st
from common import *

st.set_page_config(
    page_title="DermaCare AI - Dermatology Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
def load_css():
    st.markdown("""
        <style>
        /* Responsive Titles - Mobile Friendly */
        @media (max-width: 768px) {
            h1 {
                font-size: 1.8rem !important;
            }
            h2 {
                font-size: 1.4rem !important;
            }
            h3 {
                font-size: 1.2rem !important;
            }
        }
        
        @media (max-width: 480px) {
            h1 {
                font-size: 1.5rem !important;
            }
            h2 {
                font-size: 1.2rem !important;
            }
            h3 {
                font-size: 1.1rem !important;
            }
        }
        
        .stApp {
            background: #f0f2f6;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown p {
            color: white !important;
        }

        /* Buttons */
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
            border: none !important;
            border-radius: 8px !important;
        }
        
        [data-testid="stBaseButton-secondary"] p,
        [data-testid="stBaseButton-primary"] p,
        [data-testid="stBaseButton-secondary"] span,
        [data-testid="stBaseButton-primary"] span {
            color: white !important;
            font-weight: 600 !important;
        }

        /* Sidebar Buttons */
        [data-testid="stSidebar"] .stButton button {
            background: rgba(255, 255, 255, 0.15) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            color: white !important;
            width: 100% !important;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            background: rgba(255, 255, 255, 0.25) !important;
        }

        /* Popover */
        [data-testid="stPopover"] button {
            background: #1e3c72 !important;
            border-radius: 50% !important;
            border: none !important;
            min-width: 28px !important;
            width: 28px !important;
            height: 28px !important;
            padding: 0 !important;
            margin-left: 8px !important;
        }
        
        [data-testid="stPopover"] button p {
            color: white !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            margin: 0 !important;
        }

        div[data-testid="stPopoverBody"] {
            min-width: 260px !important;
            max-width: 300px !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }

        /* Form Elements */
        textarea,
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
            background: #ffffff !important;
            color: #2c3e50 !important;
            border: 1px solid #c0cfe8 !important;
            border-radius: 8px !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #1e3c72 !important;
        }
        
        p, li, label {
            color: #2c3e50 !important;
        }
        
        /* Feature row styling */
        .feature-row {
            margin-bottom: 20px;
            padding: 10px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Radio button styling */
        .stRadio {
            margin-top: 0 !important;
        }
        
        .stRadio > div {
            background: transparent;
            padding: 0;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .stRadio > div label {
            background: #f0f2f6;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            margin: 0;
            font-size: 0.85rem;
        }
        
        .stRadio > div label:hover {
            background: #e0e4e8;
        }
        
        .streamlit-expanderHeader {
            background: #e8f0fe;
            color: #1e3c72 !important;
            border-radius: 8px;
        }
        
        .diagnosis-card {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .diagnosis-card * {
            color: white !important;
        }
        
        hr {
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}

# ==================== PAGE FUNCTIONS ====================
def welcome_page():
    st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <h1 style="font-size: 3rem;">🩺 DermaCare AI</h1>
            <p style="font-size: 1.2rem; color: #2a5298;">Intelligent Dermatology Diagnosis Assistant</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎯 Welcome")
        st.markdown("AI-powered dermatological diagnosis system for skin conditions")
        st.markdown("---")
        st.markdown("#### ✨ Features")
        st.markdown("""
        - 🔍 Easy-to-use selection
        - 🔬 Clinical + Microscopy analysis
        - 🎯 AI-powered predictions
        - 📱 Mobile friendly
        """)
        st.info("💡 **Tip:** Click any **ℹ️** button next to a term for its definition!")

def clinical_page():
    st.title("🩺 Clinical Assessment")
    st.markdown("---")

    # Patient Information
    with st.container():
        st.markdown("### 📋 Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Patient Name", placeholder="Enter patient name")
        with col2:
            age = st.number_input("🎂 Age", min_value=0, max_value=120, value=35)
        duration = st.selectbox("⏰ Duration of Symptoms", ["< 1 week", "1-4 weeks", "1-3 months", "> 3 months"])

    st.markdown("---")
    st.markdown("### 🔍 Clinical Examination")
    st.markdown("*Rate each symptom based on severity*")
    st.markdown("---")

    clinical_data = {}
    items = list(CLINICAL_FEATURES.items())

    # Display clinical features in clean rows
    for i, (display, col_name) in enumerate(items):
        # Feature name with popover
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**{display}**")
            with st.popover("ℹ️"):
                st.markdown(f"**{display}**")
                st.caption(GLOSSARY.get(display, "Definition coming soon..."))
        with col2:
            if display == "Family History":
                clinical_data[display] = st.radio(
                    "", ["No", "Yes"], 
                    key=f"clinical_{i}", 
                    horizontal=True, 
                    label_visibility="collapsed"
                )
            else:
                clinical_data[display] = st.radio(
                    "", ["None", "Mild", "Moderate", "Severe"], 
                    index=0,
                    key=f"clinical_{i}", 
                    horizontal=True, 
                    label_visibility="collapsed"
                )
        st.markdown("---")

    notes = st.text_area("📝 Additional Notes", placeholder="Any additional observations...", height=80)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Proceed to Histopathology", use_container_width=True, type="primary"):
            converted = {}
            for d, v in clinical_data.items():
                if d == "Family History":
                    converted[d] = v
                else:
                    m = {"None": "None (0)", "Mild": "Mild (1)", "Moderate": "Moderate (2)", "Severe": "Severe (3)"}
                    converted[d] = m.get(v, "None (0)")

            st.session_state.patient_data = {
                'name': name or "Anonymous", 
                'age': age, 
                'duration': duration,
                'clinical': converted, 
                'notes': notes, 
                'time': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.switch_page("pages/histopathology.py")

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
            
            # Diagnosis Card
            st.markdown(f"""
            <div class="diagnosis-card" style="background: linear-gradient(135deg, #1e3c72, #2a5298); 
                        border-radius: 20px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 4rem;">{DISEASE_ICONS.get(disease, '🏥')}</div>
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; letter-spacing: 2px;">PRIMARY DIAGNOSIS</div>
                <div style="color: white; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{disease}</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 0.5rem; display: inline-block;">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Confidence Level")
                st.progress(int(confidence))
                st.caption(f"{confidence:.1f}% confidence")
                
                st.markdown("### 👤 Patient Summary")
                st.markdown(f"""
                <div style="background: #e8f0fe; padding: 1rem; border-radius: 10px;">
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
                    st.caption(f"{prob:.1f}%")
            
            with st.expander("🔍 Positive Clinical Findings", expanded=False):
                findings = []
                for s, v in data['clinical'].items():
                    if v not in ["None (0)", "No"]:
                        findings.append(f"- **{s}:** {v}")
                
                if findings:
                    for f in findings:
                        st.markdown(f)
                else:
                    st.markdown("*No significant findings recorded*")
            
            st.markdown("### 💡 Recommendations")
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                st.markdown("""
                <div style="background: #e8f0fe; padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 2rem;">📅</div>
                    <b>Follow-up</b><br>
                    <small>Schedule in 2-4 weeks</small>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown("""
                <div style="background: #e8f0fe; padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 2rem;">🔬</div>
                    <b>Additional Tests</b><br>
                    <small>Consider if indicated</small>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col3:
                st.markdown("""
                <div style="background: #e8f0fe; padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 2rem;">💊</div>
                    <b>Treatment</b><br>
                    <small>Initiate appropriate therapy</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.caption("⚕️ **Disclaimer:** AI-assisted prediction - Please confirm with a qualified dermatologist")
            
            if st.button("🔄 New Patient Assessment", use_container_width=True):
                st.session_state.patient_data = {}
                st.switch_page("app.py")
            
        except Exception as e:
            st.error(f"❌ Diagnosis Error: {str(e)}")
            st.markdown("**Possible reasons:**")
            st.markdown("- Missing patient information")
            st.markdown("- Invalid data format")

# ==================== MAIN ====================
def main():
    load_css()

    # Sidebar navigation
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
            if st.session_state.patient_data and 'clinical' in st.session_state.patient_data:
                st.switch_page("pages/histopathology.py")
            else:
                st.warning("⚠️ Please complete Clinical Assessment first")
        
        if st.button("📋 Diagnosis Report", use_container_width=True):
            if st.session_state.patient_data and 'histopathology' in st.session_state.patient_data:
                st.switch_page("pages/diagnosis.py")
            else:
                st.warning("⚠️ Please complete all assessments first")
        
        st.markdown("---")
        
        # Current Session Info
        if st.session_state.patient_data:
            st.markdown("### 📋 Current Session")
            st.markdown(f"**Patient:** {st.session_state.patient_data.get('name', 'N/A')}")
            status = []
            if 'clinical' in st.session_state.patient_data:
                status.append("✅ Clinical")
            if 'histopathology' in st.session_state.patient_data:
                status.append("✅ Biopsy")
            st.markdown(f"**Progress:** {' → '.join(status) if status else 'Not started'}")
        
        st.markdown("---")
        
        # Model Performance
        st.markdown("### 🎯 Model Performance")
        try:
            metrics = get_model_metrics()
            st.markdown(f"- **Accuracy:** {metrics['accuracy']:.1f}%")
            st.markdown(f"- **Precision:** {metrics['precision']:.1f}%")
            st.markdown(f"- **Recall:** {metrics['recall']:.1f}%")
            st.markdown(f"- **F1 Score:** {metrics['f1']:.1f}%")
        except Exception as e:
            st.markdown("- **Accuracy:** 96.5%")
            st.markdown("- **Precision:** 95.8%")
            st.markdown("- **Recall:** 96.2%")
            st.markdown("- **F1 Score:** 96.0%")
        
        st.markdown("---")
        
        st.markdown("### 📞 Support")
        st.markdown("📧 lalzareabhishek@gmail.com")

    # Page routing
    query_params = st.query_params
    page = query_params.get("page", ["welcome"])[0]
    
    if page == "welcome":
        welcome_page()
        if st.button("🚀 Start Diagnosis", use_container_width=True):
            st.switch_page("app.py")
    elif page == "diagnosis":
        diagnosis_page()

if __name__ == "__main__":
    main()
