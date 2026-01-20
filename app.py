"""
Ship Value Prediction - Streamlit Interface (Demo Mode Compatible)
====================================================================

Main application file for Ship Insured Value Prediction using Streamlit.

Author: Samir halila
Date: 2026-01-20
Version: 1.0.2 (Demo Mode Added)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Ship Insured Value Prediction",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp { background-color: #05020f; }
    [data-testid="stSidebar"] { background-color: #002147 !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #0b1d33 !important;
        font-weight: 700;
    }
    .prediction-box {
        background: linear-gradient(135deg, #003366 0%, #00509d 100%);
        color: white; padding: 30px; border-radius: 15px;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 2px solid #ffcc00; margin: 20px 0;
    }
    .prediction-box h1 { color: #ffcc00 !important; font-size: 3.5rem !important; margin: 10px 0; }

    div.stButton > button:first-child {
        background-color: #ffcc00; color: #002147; font-weight: bold;
        border: none; border-radius: 5px; width: 100%;
    }
    div.stButton > button:first-child:hover { background-color: #e6b800; }

    .stMetric { background-color: #516e91; border: 1px solid #516e91; border-radius: 10px; padding: 10px; }
    
    .info-box {
        background-color: #3d403a; padding: 20px; border-radius: 10px;
        border-left: 5px solid #3d403a; margin-top: 20px;
    }
    
    /* Warning for Demo Mode */
    .demo-warning {
        background-color: #fff3cd; color: #856404; padding: 15px;
        border-radius: 5px; border: 1px solid #ffeeba; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# --- MOCK MODEL FOR DEMO PURPOSES ---
class DummyModel:
    """A fake model to use if real files are missing."""
    def predict(self, X):
        # Simple linear regression logic for demo: Value = (DWT*500) + (GRT*200) + constant
        # This just ensures the UI changes when you move sliders.
        vals = []
        for _, row in X.iterrows():
            val = (row['DWT'] * 500) + (row['GRT'] * 200) + 5_000_000
            vals.append(val)
        return np.array(vals)

# --- PREDICTOR CLASS ---
class ShipValuePredictor:
    """Handles model loading and inference for ship value predictions"""

    def __init__(self, model_path: str = 'models/xgb_model_v1.pkl', config_path: str = 'models/xgb_model_v1.json'):
        """Initialize the predictor"""
        base_dir = Path(__file__).parent
        self.model_path = base_dir / model_path
        self.config_path = base_dir / config_path
        
        self.model = None
        self.config = None
        self.feature_names = None
        self.is_demo_mode = False
        
        try:
            self.load_model()
            self.load_config()
        except Exception as e:
            logger.warning(f"Switching to Demo Mode due to error: {e}")
            self.enable_demo_mode()

    def enable_demo_mode(self):
        """Activates demo mode when files are missing"""
        self.is_demo_mode = True
        self.model = DummyModel()
        # Default config matching the input data keys
        self.config = {
            'features': {
                'feature_names': ['AGE', 'DWT', 'GRT', 'Puissance_Moteur', 'PAYS_ENC', 'is_IACS', 'TYPE_ENCODED']
            },
            'preprocessing': {'target_log_transform': False},
            'performance_metrics': {
                'r2_score': 'N/A (Demo)', 'mae': 0, 'rmse': 0, 'reliability_percent': 0,
                'training_samples': 0, 'test_samples': 0
            }
        }
        self.feature_names = self.config['features']['feature_names']
        logger.info("Demo Mode Activated.")

    def load_model(self) -> None:
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully")
    
    def load_config(self) -> None:
        """Load model configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        self.feature_names = self.config.get('features', {}).get('feature_names', [])
        logger.info("Configuration loaded successfully")
    
    def predict(self, input_data: Dict) -> Tuple[float, Dict]:
        """Make prediction for ship value"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        X = pd.DataFrame([input_data])
        
        # Ensure all columns exist
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_names]
        prediction = self.model.predict(X)[0]
        
        # Check if inverse log transform is needed
        if self.config and self.config.get('preprocessing', {}).get('target_log_transform', False):
            prediction = np.exp(prediction)
        
        return prediction, {
            'features_used': self.feature_names,
            'feature_values': input_data,
            'model_info': self.config.get('model_info', {}) if self.config else {}
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from config"""
        if self.config:
            return self.config.get('features', {}).get('feature_importance', {})
        return {}
    
    def get_model_metrics(self) -> Dict:
        """Get model performance metrics"""
        if self.config:
            return self.config.get('performance_metrics', {})
        return {}


@st.cache_resource
def load_predictor():
    """Load predictor (cached)"""
    try:
        predictor = ShipValuePredictor()
        if predictor.is_demo_mode:
            st.warning("⚠️ **Demo Mode Active**: Real model files not found. Running with mock data.")
        return predictor
    except Exception as e:
        st.error(f"Critical error loading model: {e}")
        return None


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def set_form_session_state(example: Dict):
    """Helper to set form values from sidebar buttons"""
    for k, v in example.items():
        st.session_state[k] = v
    st.rerun()


def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    defaults = {
        'age': 10, 'dwt': 10000.0, 'grt': 8000.0, 'power': 5000.0,
        'type': 'General Cargo', 'iacs': 1, 'country': 'China'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Header
    col1, col2 = st.columns([1, 6])
    with col1: st.image("https://d1nhio0ox7pgb.cloudfront.net/_img/v_collection_png/512x512/shadow/containership.png", width=50)
    with col2: st.title("Ship Insured Value Prediction")
    
    # SIDEBAR
    st.sidebar.markdown("<h2 style='text-align: center; color: white;'>⚓ NAV-IA</h2>", unsafe_allow_html=True)
    st.sidebar.image("https://knowhow.distrelec.com/wp-content/uploads/2023/07/iStock-1427031637-1024x576.jpg", use_container_width=True)
    st.sidebar.markdown("<p style='text-align: center; color: #d1dce5;'>Développement de solutions d’aide à la décision</p>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("<h3 style='color: #ffcc00;'>📖 Guide d'Utilisation</h3>", unsafe_allow_html=True)
    st.sidebar.info("\n1. Remplir les paramètres\n2. Cliquer sur **CALCULER**")
    st.sidebar.markdown("---")
    
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color: #ffcc00;'>📊 À Propos</h3>", unsafe_allow_html=True)
    predictor_check = load_predictor()
    if predictor_check and predictor_check.is_demo_mode:
        st.sidebar.error("⚠️ Mode Démo Activé\n(Fichiers manquants)")
    else:
        st.sidebar.success("**Modèle:** XGBoost Regressor\n**Performance:** R² = 0.75")

    st.markdown("---")
    st.markdown("<h4>Single Prediction - Prédiction pour 1 Navire</h4>", unsafe_allow_html=True)
    
    predictor = load_predictor()
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📈 Historique", "ℹ️ Infos Modèle"])
    
    # TAB 1
    with tab1:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("📋 Ship Information")
            with st.form(key='ship_form'):
                age = st.slider("Ship Age (Years)", 0, 50, int(st.session_state['age']), 1, key='age')
                dwt = st.number_input("DWT (Deadweight Tonnage)", 100.0, 500000.0, float(st.session_state['dwt']), 1000.0, key='dwt')
                grt = st.number_input("GRT (Gross Register Tonnage)", 100.0, 500000.0, float(st.session_state['grt']), 1000.0, key='grt')
                power = st.number_input("Engine Power (kW)", 100.0, 100000.0, float(st.session_state['power']), 500.0, key='power')
                # IACS Member
                st.markdown("**Classification Society**")
                is_iacs = st.selectbox("Classification Society (IACS Member)", [('Yes - IACS Member', 1), ('No - Non-IACS', 0)], format_func=lambda x: x[0], index=0 if st.session_state['iacs'] == 1 else 1, key='iacs')[1]
                # Country / Builder
                st.markdown("**Builder/Flag Country**")
                countries = {'China': 0, 'Japan': 1, 'South Korea': 2, 'Philippines': 3, 'Germany': 4, 'Singapore': 5, 'USA': 6, 'Europe (Other)': 7, 'India': 8, 'Others': 9}
                country_name = st.selectbox("Select Country/Builder", list(countries.keys()), index=list(countries.keys()).index(st.session_state['country']), key='country')
                country_encoded = countries[country_name]
                # Ship Type
                st.markdown("**Ship Type**")
                ship_types = {'General Cargo': 0, 'Container': 1, 'Oil Tanker': 2, 'Bulk Carrier': 3, 'RoRo': 4, 'Multipurpose': 5, 'LPG Tanker': 6, 'Chemical Tanker': 7, 'Dredger': 8, 'Offshore Supply': 9}
                type_name = st.selectbox("Select Ship Type", list(ship_types.keys()), index=list(ship_types.keys()).index(st.session_state['type']), key='type')
                type_encoded = ship_types[type_name]
                
                submit_button = st.form_submit_button("🔮 CALCULER LA VALEUR", use_container_width=True, type="primary")
        
        with col2:
            if submit_button and predictor:
                errors = []
                warnings = []
                if age > 40: warnings.append(f"⚠️ Navire très ancien ({age} ans)")
                if grt < 100: errors.append("GRT doit être au moins 100")
                if power < 100: errors.append("Puissance moteur > 100 kW")
                
                if errors:
                    for e in errors: st.error(e)
                else:
                    if warnings: [st.warning(w) for w in warnings]
                    
                    input_data = {'AGE': age, 'DWT': dwt, 'GRT': grt, 'Puissance_Moteur': power, 'TYPE_ENCODED': type_encoded, 'is_IACS': is_iacs, 'PAYS_ENC': country_encoded}
                    
                    try:
                        predicted_value, info = predictor.predict(input_data)
                        
                        st.session_state.prediction_history.append({
                            'timestamp': pd.Timestamp.now(), 'age': age, 'dwt': dwt, 'grt': grt, 'power': power, 'type': type_name, 'iacs': 'Yes' if is_iacs else 'No', 'country': country_name, 'predicted_value': predicted_value
                        })
                        
                        st.subheader("🎯 Prediction Result")
                        st.markdown(f"""
                            <div class="prediction-box">
                                <h3>Estimated Insured Value</h3>
                                <h1>{format_currency(predicted_value)}</h1>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.subheader("📊 Input Summary")
                        m1, m2 = st.columns(2)
                        with m1: st.metric("Ship Age", f"{age} years"); st.metric("DWT", f"{dwt:,.0f} tons"); st.metric("GRT", f"{grt:,.0f} tons")
                        with m2: st.metric("Engine Power", f"{power:,.0f} kW"); st.metric("Ship Type", type_name); st.metric("IACS Member", "Yes" if is_iacs else "No")
                        
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
            elif submit_button and not predictor:
                st.error("Model not available.")

    # TAB 2
    with tab2:
        st.subheader("📈 Historique des Prédictions")
        if st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['predicted_value'] = df['predicted_value'].apply(format_currency)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---"); st.subheader("📊 Statistiques")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total Prédictions", len(st.session_state.prediction_history))
            with c2: st.metric("Dernière Valeur", format_currency(st.session_state.prediction_history[-1]['predicted_value']))
            with c3: st.metric("Valeur Max", format_currency(max(p['predicted_value'] for p in st.session_state.prediction_history)))
            
            if st.button("🗑️ Effacer l'Historique"): st.session_state.prediction_history = []; st.rerun()
        else:
            st.info("Aucune prédiction encore.")

    # TAB 3
    with tab3:
        st.subheader("ℹ️ Informations du Modèle")
        if predictor:
            metrics = predictor.get_model_metrics()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("R² Score", f"{metrics.get('r2_score', 'N/A')}"); st.metric("Training Samples", f"{metrics.get('training_samples', 'N/A')}")
            with c2: st.metric("MAE", f"${metrics.get('mae', 0):,.0f}"); st.metric("Test Samples", f"{metrics.get('test_samples', 'N/A')}")
            with c3: st.metric("RMSE", f"${metrics.get('rmse', 0):,.0f}"); st.metric("Reliability", f"{metrics.get('reliability_percent', 'N/A')}%")
            
            st.markdown("---"); st.subheader("Feature Importance")
            imp = predictor.get_feature_importance()
            if imp:
                imp_df = pd.DataFrame(list(imp.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='viridis')
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance data available in config.")
        else:
            st.warning("Model not loaded.")

        st.markdown("---")
        st.markdown("""
            <div class="info-box">
                <strong>ℹ️ Model Information:</strong>
                <ul>
                    <li>Model trained on historical ship valuation data</li>
                    <li>Considers technical characteristics and ship type</li>
                    <li>Predictions based on XGBoost regression</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()