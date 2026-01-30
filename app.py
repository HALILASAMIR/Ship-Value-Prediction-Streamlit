import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
from datetime import datetime
# --- FONCTION CSS PERSONNALISÉE ---
def inject_custom_css():
    st.markdown("""
        <style>
            /* 1. PALETTE DE COULEURS UNIFIÉE */
            :root {
                --bleu-fond: #78c4e6;
                --bleu-marine: #12517a;
                --bleu-fonce: #0d3b66;
                --bleu-bouton: #4d99bf;
                --blanc-pur: #ffffff;
            }

            /* Fond de l'application */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(160deg, var(--bleu-fond) 0%, #a2d9f2 100%) !important;
            }
            
            /* Sidebar : Marine profond */
            [data-testid="stSidebar"] {
                background-color: var(--bleu-marine) !important;
            }

            /* 2. TYPOGRAPHIE ET COULEURS DE TEXTE */
            h1, h2, h3, p, span, label {
                color: var(--bleu-fonce) !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            /* Texte blanc UNIQUEMENT dans la sidebar */
            [data-testid="stSidebar"] .stMarkdown p, 
            [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] label {
                color: var(--blanc-pur) !important;
            }

            /* 3. CARTES ET FORMULAIRES (Le secret du design pro) */
            div[data-testid="stForm"], .stMetric, .stTable {
                background-color: rgba(255, 255, 255, 0.85) !important;
                border-radius: 12px !important;
                padding: 25px !important;
                box-shadow: 0 8px 20px rgba(13, 59, 102, 0.1) !important;
                border: 1px solid rgba(255, 255, 255, 0.5) !important;
            }

            /* 4. BOUTON (Harmonisé avec la sidebar) */
            button[kind="primary"] {
                background-color: var(--bleu-marine) !important;
                color: white !important;
                border-radius: 8px !important;
                border: none !important;
                transition: all 0.3s ease;
                height: 3em;
            }

            button[kind="primary"]:hover {
                background-color: var(--bleu-fonce) !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }

            /* 5. CHAMPS DE SAISIE (Inputs) */
            input, select, div[data-baseweb="select"] {
                background-color: white !important;
                border: 1px solid var(--bleu-fond) !important;
                border-radius: 6px !important;
                color: var(--bleu-fonce) !important;
            }

            /* Résultat de la prédiction (Le bandeau final) */
            .prediction-box {
                background-color: var(--bleu-marine) !important;
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border-left: 5px solid var(--bleu-fond);
            }
                button[kind="primary"] {
                background: linear-gradient(135deg, #d4af37 0%, #f1d592 100%) !important;
                color: #0d3b66 !important;
                border: none !important;
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
    st.sidebar.markdown("<h3 style='text-align: center;'> IA appliquée à la valorisation des navires</h3>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔹 Saisie des données")
    st.sidebar.markdown("### 🔹 Résultat de la Prédiction")
    st.sidebar.markdown("### 🔹 Historique des simulations")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Développé par **Samir Hallina**")
    st.sidebar.markdown("[GitHub Repository](https://github.com/HALILASAMIR)")


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
            
            submit = st.form_submit_button("Calculer la Valeur ", type="primary")

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
            st.markdown(f"""
                <div style="
                    background: #5da2cf;
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    border: 2px solid #d4af37;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
                    margin: 20px 0;
                ">
                    <h3 style="
                        color: white !important; 
                        font-family: 'Georgia', serif; 
                        text-transform: uppercase; 
                        letter-spacing: 2px;
                        margin-bottom: 10px;
                        font-size: 18px;
                    ">
                        ✨ Valeur Estimée du Navire
                    </h3>
                    <p style="
                        font-size: 45px; 
                        font-weight: 900; 
                        color: white; 
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                        margin: 0;
                        font-family: 'Verdana', sans-serif;
                    ">
                        ${valeur:,.0f}
                    </p>
                    <div style="
                        width: 50px; 
                        height: 3px; 
                        background: #d4af37; 
                        margin: 15px auto 0;
                        border-radius: 2px;
                    "></div>
                </div>
            """, unsafe_allow_html=True)
    
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


