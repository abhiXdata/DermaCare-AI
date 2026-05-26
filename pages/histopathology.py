import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

st.set_page_config(
    page_title="Histopathology - DermaCare AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
        <style>
        @media (max-width: 768px) {
            h1 { font-size: 1.8rem !important; }
            h2 { font-size: 1.4rem !important; }
            h3 { font-size: 1.2rem !important; }
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
            font-size: 0.85rem;
        }
        
        .streamlit-expanderHeader {
            background: #e8f0fe;
            color: #1e3c72 !important;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def histopathology_page():
    st.title("🔬 Histopathology Analysis")
    st.markdown("---")

    if not st.session_state.patient_data or 'clinical' not in st.session_state.patient_data:
        st.warning("⚠️ Please complete the clinical assessment first.")
        if st.button("← Go to Clinical Assessment", use_container_width=True):
            st.switch_page("app.py")
        return

    st.info(f"**Patient:** {st.session_state.patient_data.get('name', 'Unknown')}")

    histo_data = {}
    items = list(HISTOPATHOLOGY_FEATURES.items())

    groups = {
        "📊 Epidermal Changes": items[:8],
        "🔥 Inflammatory Features": items[8:16],
        "🏥 Dermal Changes": items[16:]
    }

    for group_name, group_items in groups.items():
        with st.expander(group_name, expanded=True):
            for i, (display, col_name) in enumerate(group_items):
                # Feature name with popover
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{display}**")
                    with st.popover("ℹ️"):
                        st.markdown(f"**{display}**")
                        st.caption(GLOSSARY.get(display, "Definition coming soon..."))
                with col2:
                    histo_data[display] = st.radio(
                        "", ["None", "Mild", "Moderate", "Severe"], 
                        index=0,
                        key=f"histo_{group_name}_{i}", 
                        horizontal=True, 
                        label_visibility="collapsed"
                    )
                st.markdown("---")

    path_notes = st.text_area("📝 Pathologist's Notes", placeholder="Any microscopic observations...", height=80)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Clinical", use_container_width=True):
            st.switch_page("app.py")
    with col2:
        if st.button("💊 Generate Diagnosis", use_container_width=True, type="primary"):
            converted = {}
            for d, v in histo_data.items():
                m = {"None": "None (0)", "Mild": "Mild (1)", "Moderate": "Moderate (2)", "Severe": "Severe (3)"}
                converted[d] = m.get(v, "None (0)")
            st.session_state.patient_data['histopathology'] = converted
            st.session_state.patient_data['path_notes'] = path_notes
            st.switch_page("pages/diagnosis.py")

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
            st.switch_page("pages/histopathology.py")
        if st.button("📋 Diagnosis Report", use_container_width=True):
            if st.session_state.patient_data and 'histopathology' in st.session_state.patient_data:
                st.switch_page("pages/diagnosis.py")
            else:
                st.warning("⚠️ Please complete all assessments first")
        
        st.markdown("---")
        if st.session_state.patient_data:
            st.markdown("### 📋 Current Session")
            st.markdown(f"**Patient:** {st.session_state.patient_data.get('name', 'N/A')}")
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.markdown("📧 lalzareabhishek@gmail.com")
    
    histopathology_page()

if __name__ == "__main__":
    main()
