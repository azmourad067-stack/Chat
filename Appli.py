import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression
import shap
import itertools
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏇 Analyseur Hippique IA Pro",
    page_icon="🏇",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        border-left: 5px solid #f59e0b;
        padding: 1rem 1rem 1rem 1.5rem;
        background: linear-gradient(90deg, #fffbeb 0%, #ffffff 100%);
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .confidence-high { color: #10b981; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

CONFIGS = {
    "PLAT": {
        "description": "🏃 Course de galop - Handicap poids + avantage corde intérieure",
        "optimal_draws": [1, 2, 3, 4],
        "weight_importance": 0.25
    },
    "ATTELE_AUTOSTART": {
        "description": "🚗 Trot attelé autostart - Numéros 4-6 optimaux", 
        "optimal_draws": [4, 5, 6],
        "weight_importance": 0.05
    },
    "ATTELE_VOLTE": {
        "description": "🔄 Trot attelé volté - Numéro sans importance",
        "optimal_draws": [],
        "weight_importance": 0.05
    }
}

@st.cache_resource
class AdvancedHorseRacingDL:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_correlations = {}
        self.is_trained = False
    
    def extract_music_features(self, music_str):
        """Extraction avancée des performances passées"""
        if pd.isna(music_str) or music_str == '':
            return {
                'wins': 0, 'places': 0, 'total_races': 0,
                'win_rate': 0, 'place_rate': 0, 'consistency': 0,
                'recent_form': 0, 'best_position': 10,
                'avg_position': 8, 'position_variance': 5
            }
        
        music = str(music_str)
        positions = [int(c) for c in music if c.isdigit() and int(c) > 0]
        
        if not positions:
            return {
                'wins': 0, 'places': 0, 'total_races': 0,
                'win_rate': 0, 'place_rate': 0, 'consistency': 0,
                'recent_form': 0, 'best_position': 10,
                'avg_position': 8, 'position_variance': 5
            }
        
        total = len(positions)
        wins = positions.count(1)
        places = sum(1 for p in positions if p <= 3)
        
        # Forme récente (3 dernières courses)
        recent = positions[:3]
        recent_form = sum(1/p for p in recent) / len(recent) if recent else 0
        
        # Régularité
        consistency = 1 / (np.std(positions) + 1) if len(positions) > 1 else 0
        
        return {
            'wins': wins,
            'places': places,
            'total_races': total,
            'win_rate': wins / total if total > 0 else 0,
            'place_rate': places / total if total > 0 else 0,
            'consistency': consistency,
            'recent_form': recent_form,
            'best_position': min(positions),
            'avg_position': np.mean(positions),
            'position_variance': np.var(positions)
        }
    
    def prepare_enhanced_features(self, df, race_type="PLAT", historical_data=None):
        """Création de features étendues avec corrélations"""
        features = pd.DataFrame()
        
        # === FEATURES DE BASE ===
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.1)
        features['log_odds'] = np.log1p(df['odds_numeric'])
        features['sqrt_odds'] = np.sqrt(df['odds_numeric'])
        features['odds_squared'] = df['odds_numeric'] ** 2
        
        # === FEATURES DE POSITION ===
        features['draw'] = df['draw_numeric']
        features['draw_normalized'] = df['draw_numeric'] / df['draw_numeric'].max()
        
        # Avantage position selon type de course
        optimal_draws = CONFIGS[race_type]['optimal_draws']
        features['optimal_draw'] = df['draw_numeric'].apply(
            lambda x: 1 if x in optimal_draws else 0
        )
        features['draw_distance_optimal'] = df['draw_numeric'].apply(
            lambda x: min([abs(x - opt) for opt in optimal_draws]) if optimal_draws else 0
        )
        
        # === FEATURES DE POIDS ===
        features['weight'] = df['weight_kg']
        features['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-6)
        features['weight_rank'] = df['weight_kg'].rank()
        weight_importance = CONFIGS[race_type]['weight_importance']
        features['weight_advantage'] = (df['weight_kg'].max() - df['weight_kg']) * weight_importance
        
        # === FEATURES D'ÂGE ET SEXE ===
        if 'Âge/Sexe' in df.columns:
            features['age'] = df['Âge/Sexe'].str.extract('(\d+)').astype(float).fillna(4)
            features['is_mare'] = df['Âge/Sexe'].str.contains('F', na=False).astype(int)
            features['is_stallion'] = df['Âge/Sexe'].str.contains('H', na=False).astype(int)
            features['age_squared'] = features['age'] ** 2
            features['age_optimal'] = features['age'].apply(lambda x: 1 if 4 <= x <= 6 else 0)
        else:
            features['age'] = 4.5
            features['is_mare'] = 0
            features['is_stallion'] = 0
            features['age_squared'] = 20.25
            features['age_optimal'] = 1
        
        # === FEATURES DE MUSIQUE (HISTORIQUE) ===
        if 'Musique' in df.columns:
            music_features = df['Musique'].apply(self.extract_music_features)
            for key in music_features.iloc[0].keys():
                features[f'music_{key}'] = [m[key] for m in music_features]
        else:
            for key in ['wins', 'places', 'total_races', 'win_rate', 'place_rate', 
                       'consistency', 'recent_form', 'best_position', 'avg_position', 'position_variance']:
                features[f'music_{key}'] = 0
        
        # === FEATURES D'INTERACTION ===
        features['odds_draw_interaction'] = features['odds_inv'] * features['draw_normalized']
        features['odds_weight_interaction'] = features['log_odds'] * features['weight_normalized']
        features['age_weight_interaction'] = features['age'] * features['weight']
        features['form_odds_interaction'] = features['music_recent_form'] * features['odds_inv']
        features['consistency_weight'] = features['music_consistency'] * features['weight_advantage']
        
        # === FEATURES DE CLASSEMENT RELATIF ===
        features['odds_rank'] = df['odds_numeric'].rank()
        features['odds_percentile'] = df['odds_numeric'].rank(pct=True)
        features['weight_percentile'] = df['weight_kg'].rank(pct=True)
        
        # === FEATURES STATISTIQUES ===
        features['odds_z_score'] = (df['odds_numeric'] - df['odds_numeric'].mean()) / (df['odds_numeric'].std() + 1e-6)
        features['is_favorite'] = (df['odds_numeric'] == df['odds_numeric'].min()).astype(int)
        features['is_outsider'] = (df['odds_numeric'] > df['odds_numeric'].quantile(0.75)).astype(int)
        
        # === FEATURES DE CONTEXTE ===
        features['field_size'] = len(df)
        features['competitive_index'] = df['odds_numeric'].std() / (df['odds_numeric'].mean() + 1e-6)
        
        # === NOUVELLES FEATURES ===
        features['race_type_plat'] = 1 if race_type == "PLAT" else 0
        features['race_type_attele'] = 1 if "ATTELE" in race_type else 0
        features['race_type_obstacle'] = 1 if race_type == "OBSTACLE" else 0
        
        # Stats jockey/driver (si disponibles)
        if 'jockey_win_rate' in df.columns:
            features['jockey_win_rate'] = df['jockey_win_rate']
        else:
            features['jockey_win_rate'] = 0.15  # Valeur par défaut
        
        # Hippodrome (si données historiques)
        if historical_data:
            features['hippodrome_win_rate'] = df['hippodrome'].map(historical_data.get('win_rates', {})).fillna(0.1)
        else:
            features['hippodrome_win_rate'] = 0.1
        
        # Interactions supplémentaires
        features['odds_music_interaction'] = features['odds_inv'] * features['music_recent_form']
        features['weight_jockey_interaction'] = features['weight_kg'] * features['jockey_win_rate']
        
        return features.fillna(0)
    
    def calculate_feature_weights(self, X, y):
        """Calcul des poids basés sur les corrélations"""
        correlations = {}
        for col in X.columns:
            if X[col].std() > 0:
                corr = np.corrcoef(X[col], y)[0, 1]
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
        self.feature_correlations = correlations
        return correlations
    
    def build_deep_model(self, input_dim):
        """Construction du réseau de neurones"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear')  # Régression pour le score
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_and_predict(self, X, y, cv_folds=5):
        """Entraînement avec DL et validation"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Calcul des poids des features
        weights = self.calculate_feature_weights(pd.DataFrame(X_scaled, columns=X.columns), y)
        
        # Sélection des meilleures features
        selector = SelectKBest(score_func=f_regression, k=30)
        X_selected = selector.fit_transform(X_scaled, y)
        
        # Cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_selected):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.build_deep_model(X_selected.shape[1])
            model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_data=(X_val, y_val))
            
            pred = model.predict(X_val).flatten()
            cv_scores.append(r2_score(y_val, pred))
        
        # Modèle final
        self.model = self.build_deep_model(X_selected.shape[1])
        self.model.fit(X_selected, y, epochs=100, batch_size=16, verbose=0)
        
        # Prédictions finales
        predictions = self.model.predict(X_selected).flatten()
        
        # Confiance basée sur la variance des prédictions
        confidence = 1 / (1 + np.std(predictions))
        
        self.is_trained = True
        return predictions, np.mean(cv_scores), confidence

@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        
        table = soup.find('table')
        if not table:
            return None, "Aucun tableau trouvé"
            
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:
                horses_data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),
                    "Poids": cols[-2].get_text(strip=True) if len(cols) > 4 else "60.0",
                    "Musique": cols[2].get_text(strip=True) if len(cols) > 5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols) > 6 else "",
                })

        if not horses_data:
            return None, "Aucune donnée extraite"
            
        return pd.DataFrame(horses_data), "Succès"
        
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def safe_convert(value, convert_func, default=0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def prepare_data(df):
    df = df.copy()
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    
    def extract_weight(poids_str):
        if pd.isna(poids_str):
            return 60.0
        match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
        return float(match.group(1).replace(',', '.')) if match else 60.0
    
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    df = df[df['odds_numeric'] > 0]
    df = df.reset_index(drop=True)
    return df

def auto_detect_race_type(df):
    weight_std = df['weight_kg'].std()
    weight_mean = df['weight_kg'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💪 Écart-type poids", f"{weight_std:.1f} kg")
    with col2:
        st.metric("⚖️ Poids moyen", f"{weight_mean:.1f} kg")
    with col3:
        st.metric("🏇 Nb chevaux", len(df))
    
    if weight_std > 2.5:
        detected = "PLAT"
        reason = "Grande variation de poids (handicap)"
    elif weight_mean > 65 and weight_std < 1.5:
        detected = "ATTELE_AUTOSTART"
        reason = "Poids uniformes élevés (attelé)"
    else:
        detected = "PLAT"
        reason = "Configuration par défaut"
    
    st.info(f"🤖 **Type détecté**: {detected} | **Raison**: {reason}")
    return detected

def create_advanced_visualization(df_ranked, dl_model=None):
    """Visualisations avancées avec métriques DL"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '🏆 Scores de Confiance', 
            '📊 Distribution Cotes', 
            '🧠 Importance Features',
            '⚖️ Poids vs Performance', 
            '📈 Validation Croisée',
            '🎯 Corrélation Cotes-Scores'
        ),
        specs=[
            [{"secondary_y": False}, {"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    colors = px.colors.qualitative.Set3
    
    # 1. Scores avec confiance
    if 'score_final' in df_ranked.columns and 'confidence' in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['rang'],
                y=df_ranked['score_final'],
                mode='markers+lines',
                marker=dict(
                    size=df_ranked['confidence'] * 20,
                    color=df_ranked['confidence'],
                    colorscale='Viridis
