# Streamlit Interface - Ship Value Prediction

## Structure

```
streamlit_interface/
├── app.py              # Application principale
├── run.py              # Launcher
├── config.json         # Configuration
├── README.md           # Documentation
└── requirements.txt    # Dépendances
```

## Démarrage Rapide

### Option 1: Python (depuis ce dossier)
```bash
cd streamlit_interface
python run.py
```

### Option 2: Streamlit Direct
```bash
streamlit run app.py
```

### Option 3: Depuis le dossier parent
```bash
python -m streamlit run streamlit_interface/app.py
```

## Accès

URL: `http://localhost:8501`

## Dépendances

- streamlit >= 1.0.0
- plotly >= 5.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- xgboost >= 1.5.0
- joblib >= 1.0.0

## Utilisation

### Formulaire
1. Remplir les 7 paramètres du navire
2. Sélectionner le type et le pays
3. Cliquer "Predict Value"

### Résultats
- Valeur assurée estimée en USD
- Résumé des paramètres
- Métriques du modèle

## Fichiers Modèle

Les fichiers modèle doivent être dans le dossier parent:
- `../xgb_model_v1.pkl`
- `../xgb_model_v1.json`

## Documentation Complète

Voir `README.md` pour plus de détails.

## Support

Pour les problèmes:
1. Vérifier que `xgb_model_v1.pkl` existe
2. Vérifier que `xgb_model_v1.json` existe
3. Réinstaller les dépendances: `pip install -r requirements.txt`

---

**Version:** 1.0.0  
**Date:** 2026-01-20  
**Auteur:** Samir
