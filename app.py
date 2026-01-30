import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
from datetime import datetime

def inject_custom_css():
    st.markdown("""
        <style>
            /* Fond général */
            .main {
                background-color: #78c4e6;#coleur de fond principal
                color: #78c4e6;
                font-family: 'Segoe UI', sans-serif;
            }

            /* Titres */
            h1, h2, h3, h4 {
                color: #00b4d8;
            }

            /* Encadrés */
            .stForm, .stTable {
                background-color: #1c1f26;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,180,216,0.2);
            }

            /* Bouton */
            button[kind="primary"] {
                background-color:#4d99bf;
                color: #ffffff;
                border-radius: 8px;
                padding: 0.5em 1em;
                font-weight: bold;
            }

            /* Sidebar */
            .css-1d391kg {
                background-color: #12517a;
            }

            /* Tableau historique */
            .stTable tbody tr {
                background-color: #1c1f26;
                border-bottom: 1px solid #2c2f36;
            }

            /* Input fields */
            input, select {
                background-color: #2c2f36;
                color: white;
                border: 1px solid #00b4d8;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION ---
st.set_page_config(page_title="Expert Ship Valuator", page_icon="⚓", layout="wide")

class ShipValuePredictor:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.model_path = self.base_path / "models" / "xgb_model_v1.pkl"
        self.config_path = self.base_path / "models" / "xgb_model_v1.json"
        
        # Ordre STRICT du training.py
        self.feature_names = ['AGE', 'DWT', 'GRT', 'Puissance_Moteur', 'TYPE_ENCODED', 'is_IACS']
        
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None
        
        # Import des valeurs réelles du JSON
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None

    def predict(self, input_dict):
        X = pd.DataFrame([input_dict])
        X = X[self.feature_names] # RÉALIGNEMENT FORCÉ
        log_pred = self.model.predict(X)[0]
        return np.expm1(log_pred)

# --- INITIALISATION HISTORIQUE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

def main():
    #apply custom CSS
    inject_custom_css()
    #--- APPLICATION ---
    predictor = ShipValuePredictor()
    st.title("⚓ Ship Valuation Expert System")
    # 
    st.markdown("### Estimation de la valeur des navires basée sur des modèles ML avancés")
    # --- SIDEBAR ---
    st.sidebar.image("logo.png", width=200)
    #centrer le texte
    st.sidebar.markdown("<div style='text-align: center;'> IA appliquée à la valorisation des navires</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔹 Saisie des données")
    st.sidebar.markdown("### 🔹 Résultat de la Prédiction")
    st.sidebar.markdown("### 🔹 Historique des simulations")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Développé par **Samir Hallina**")
    st.sidebar.markdown("[GitHub Repository](https://github.com/samirhalila/Ship-Insured-Value-Prediction-Model)")


    if not predictor.model:
        st.error("Modèle introuvable dans /models")
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📥 Saisie des données")
        with st.form("valuation_form"):
            year = st.number_input("Année de construction", 1980, 2026, 2015)
            age = 2026 - year
            dwt = st.number_input("DWT", 100, 500000, 30000)
            grt = st.number_input("GRT", 100, 500000, 20000)
            pwr = st.number_input("Puissance (kW)", 100, 100000, 10000)
            
            t_map = {'General Cargo':0, 'Container':1, 'Oil Tanker':2, 'Bulk Carrier':3, 'RoRo':4, 'LPG Tanker':6}
            ship_type = st.selectbox("Type de Navire", list(t_map.keys()))
            
            iacs = st.selectbox("IACS", ["Oui", "Non"])
            
            c_map = {"Chine": 0, "Japon": 1, "Corée": 2, "Europe": 7, "Autre": 9}
            country = st.selectbox("Pays", list(c_map.keys()))
            
            submit = st.form_submit_button("Calculer la Valeur Réelle", type="primary")

    with col2:
        if submit:
            data = {
                'AGE': age, 'DWT': dwt, 'GRT': grt, 'Puissance_Moteur': pwr,
                'TYPE_ENCODED': t_map[ship_type],
                'is_IACS': 1 if iacs == "Oui" else 0,
            }
            
            valeur = predictor.predict(data)
            
            # Sauvegarde dans l'historique
            st.session_state['history'].append({
                'Date': datetime.now().strftime("%H:%M:%S"),
                'Type': ship_type,
                'Pays': country,
                'Âge': age,
                'Prix Estimé': f"${valeur:,.0f}"
            })
            st.markdown("## 📈 Résultat de la Prédiction")
            st.markdown(f""" <div style="background-color:#1c1f26; padding:20px; border-radius:10px; box-shadow:0 0 10px rgba(0,180,216,0.3);"> <h3 style="color:#00b4d8;">Valeur Estimée</h3> <p style="font-size:24px; font-weight:bold; color:#f0f2f6;">${valeur:,.0f}</p> </div> """, unsafe_allow_html=True)

            st.success(f"### Valeur Estimée : ${valeur:,.0f}")
            
            # --- COMPARAISON AVEC LES POIDS RÉELS DU JSON ---
            if predictor.config:
                st.info("📊 **Analyse de l'importance (Source JSON):**")
                imp_df = pd.DataFrame({
                    'Feature': predictor.config['XGBoost']['features'],
                    'Poids Réel': predictor.config['XGBoost']['feature_importance']
                }).sort_values('Poids Réel', ascending=False)
                st.table(imp_df)

    # --- AFFICHAGE HISTORIQUE ---

    if st.session_state['history']: 
        df_hist = pd.DataFrame(st.session_state['history']) 
        st.markdown("---") 
        st.subheader("📜 Historique des simulations") 
        st.table(df_hist.tail(5)) 
        st.markdown("## 📊 Statistiques") 
        st.metric("Total Prédictions", len(df_hist)) 
        max_val = df_hist['Prix Estimé'].str.replace("$", "").str.replace(",", "").astype(float).max() 
        st.metric("Valeur Max", f"${max_val:,.0f}")
if __name__ == "__main__":
    main()
