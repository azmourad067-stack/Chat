# -*- coding: utf-8 -*-
"""
Streamlit app amélioré — Analyseur Hippique IA (avec Deep Learning, export modèle, et génération de combinaisons e-trio)
Fichier unique prêt à pousser sur GitHub et déployer sur Streamlit Cloud.
"""

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
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# SKLearn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="🏇 Analyseur Hippique IA (DL)",
    page_icon="🏇",
    layout="wide"
)

# --------------------- Styling ---------------------
st.markdown("""
<style>
    .main-header { font-size: 2.6rem; color: #1e3a8a; text-align: center; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.7rem; border-radius: 10px; color: white; text-align: center; margin: 0.4rem 0; }
    .prediction-box { border-left: 4px solid #f59e0b; padding-left: 1rem; background-color: #fffbeb; margin: 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

# --------------------- Configs & Defaults ---------------------
CONFIGS = {
    "PLAT": {"description": "🏃 Course de galop - Handicap poids + avantage corde intérieure", "optimal_draws": [1,2,3,4]},
    "ATTELE_AUTOSTART": {"description": "🚗 Trot attelé autostart - Numéros 4-6 optimaux", "optimal_draws": [4,5,6]},
    "ATTELE_VOLTE": {"description": "🔄 Trot attelé volté - Numéro sans importance", "optimal_draws": []}
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------- Utilities ---------------------
def safe_convert(value, convert_func, default=0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str):
        return 60.0
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(match.group(1).replace(',', '.')) if match else 60.0

# --------------------- Scraper (amélioré, mais simple) ---------------------
@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        if not table:
            return None, 'Aucun tableau trouvé'
        rows = table.find_all('tr')[1:]
        horses_data = []
        for row in rows:
            cols = row.find_all(['td','th'])
            if len(cols) >= 4:
                horses_data.append({
                    'Numéro de corde': cols[0].get_text(strip=True),
                    'Nom': cols[1].get_text(strip=True),
                    'Musique': cols[2].get_text(strip=True) if len(cols) > 2 else '',
                    'Âge/Sexe': cols[3].get_text(strip=True) if len(cols) > 3 else '',
                    'Poids': cols[-2].get_text(strip=True) if len(cols) > 3 else '60',
                    'Cote': cols[-1].get_text(strip=True)
                })
        if not horses_data:
            return None, 'Aucune donnée extraite'
        return pd.DataFrame(horses_data), 'Succès'
    except Exception as e:
        return None, f'Erreur: {e}'

# --------------------- Feature Engineering ---------------------

def music_to_features(music_str):
    """Transforme la musique en caractéristiques :
       - recent_wins: nombre de '1' dans la musique
       - recent_top3: nombre de places <=3
       - weighted_score: score pondéré par récence (dernier résultat pèse +)
    """
    s = str(music_str)
    digits = [int(ch) for ch in re.findall(r'\d+', s)]
    if not digits:
        return 0, 0, 0.0
    recent_wins = sum(1 for d in digits if d == 1)
    recent_top3 = sum(1 for d in digits if d <= 3)
    weights = np.linspace(1, 0.3, num=len(digits))
    weighted_score = sum((4 - d) * w for d,w in zip(digits, weights)) / (len(digits) + 1e-6)
    return recent_wins, recent_top3, weighted_score


def prepare_data(df):
    df = df.copy()
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    df['weight_kg'] = df['Poids'].apply(extract_weight)

    ages = []
    is_female = []
    r_wins=[]; r_top3=[]; r_weighted=[]
    for val in df.get('Âge/Sexe', ['']*len(df)):
        m = re.search(r'(\d+)', str(val))
        ages.append(float(m.group(1)) if m else 4.0)
        is_female.append(1 if str(val).upper().find('F')!=-1 or str(val).upper().find('M')!=-1 and str(val).upper().find('H')==-1 else 0)
    for mus in df.get('Musique', ['']*len(df)):
        a,b,c = music_to_features(mus)
        r_wins.append(a); r_top3.append(b); r_weighted.append(c)

    df['age'] = ages
    df['is_female'] = is_female
    df['recent_wins'] = r_wins
    df['recent_top3'] = r_top3
    df['recent_weighted'] = r_weighted

    df = df[df['odds_numeric'] > 0]
    df = df.reset_index(drop=True)
    return df

# --------------------- Model Manager (DL + fallback ML) ---------------------
@st.cache_resource
class ModelManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.feature_cols = ['odds_numeric','draw_numeric','weight_kg','age','is_female','recent_wins','recent_top3','recent_weighted']
        self.model_path = os.path.join(MODEL_DIR, 'dl_model.keras')
        self.scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')

    def build_model(self, input_dim):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

    def train(self, X, y, epochs=80, batch_size=8, val_split=0.15):
        Xs = self.scaler.fit_transform(X)
        model = self.build_model(Xs.shape[1])
        cb = [callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
        history = model.fit(Xs, y, validation_split=val_split, epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)
        self.model = model
        self.history = history.history
        model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        return history.history

    def predict(self, X):
        if self.model is None:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                try:
                    self.model = models.load_model(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                except Exception:
                    return np.zeros(len(X))
            else:
                return np.zeros(len(X))
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs).flatten()
        return preds

model_manager = ModelManager()

# --------------------- Combinations generator (e-trio / e-super) ---------------------
from itertools import combinations, permutations

def generate_e_trio(df, n_combinations=35, mix_favorites_outsiders=True):
    """Génère combinaisons e-trio (3 chevaux). Stratégie: prioriser favoris + mélanger outsiders."""
    df = df.copy().reset_index(drop=True)
    df['fav_score'] = 1 / (df['odds_numeric'] + 0.1)
    df = df.sort_values('fav_score', ascending=False).reset_index()
    favorites = df.head(max(3, int(len(df)*0.3)))['Nom'].tolist()
    outsiders = df.tail(max(3, int(len(df)*0.3)))['Nom'].tolist()
    pool = list(dict.fromkeys(favorites + outsiders + df['Nom'].tolist()))

    combos = []
    # mix: 1-2 favorites + outsider, 2 favorites + 1 outsider, etc.
    for a in pool:
        for b in pool:
            for c in pool:
                if a!=b and b!=c and a!=c:
                    combos.append(tuple([a,b,c]))
    # remove duplicates irrespective of order
    combos_unique = []
    seen = set()
    for comb in combos:
        key = tuple(sorted(comb))
        if key not in seen:
            seen.add(key)
            combos_unique.append(comb)
    # score combos by sum of fav_score
    name_to_score = dict(zip(df['Nom'], df['fav_score']))
    def combo_score(comb):
        return sum(name_to_score.get(n, 0) for n in comb)
    combos_unique = sorted(combos_unique, key=combo_score, reverse=True)
    return combos_unique[:min(n_combinations, len(combos_unique))]

# --------------------- Visualizations ---------------------

def create_visualization(df_ranked, feature_importance=None):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('🏆 Scores par Position','📊 Distribution Cotes','⚖️ Poids vs Score','🧠 Features'))
    colors = px.colors.qualitative.Pastel
    score_col = 'score_final' if 'score_final' in df_ranked.columns else 'ml_score'
    if score_col in df_ranked.columns:
        fig.add_trace(go.Scatter(x=df_ranked['rang'], y=df_ranked[score_col], mode='markers+lines', text=df_ranked['Nom'], name='Score'), row=1,col=1)
    fig.add_trace(go.Histogram(x=df_ranked['odds_numeric'], nbinsx=8, name='Cotes'), row=1, col=2)
    if score_col in df_ranked.columns:
        fig.add_trace(go.Scatter(x=df_ranked['weight_kg'], y=df_ranked[score_col], mode='markers', text=df_ranked['Nom'], name='Poids vs Score'), row=2, col=1)
    if feature_importance:
        keys, vals = zip(*sorted(feature_importance.items(), key=lambda x:x[1], reverse=True))
        fig.add_trace(go.Bar(x=list(vals)[:6], y=list(keys)[:6], orientation='h', name='Importance'), row=2, col=2)
    fig.update_layout(height=650, showlegend=True, title_text='📊 Analyse Complète', title_x=0.5)
    return fig

# --------------------- App UI ---------------------

def main():
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA (Deep Learning)</h1>', unsafe_allow_html=True)
    st.markdown('*Amélioration du script: DL (Keras), meilleur feature engineering, génération e-trio, sauvegarde modèle, README pour GitHub/Streamlit.*')

    with st.sidebar:
        st.header('⚙️ Configuration')
        race_type = st.selectbox('🏁 Type de course', ['AUTO','PLAT','ATTELE_AUTOSTART','ATTELE_VOLTE'])
        enable_dl = st.checkbox('✅ Activer Deep Learning (Keras)', value=True)
        dl_epochs = st.number_input('🏋️‍♂️ Epochs', min_value=5, max_value=500, value=80, step=5)
        dl_val = st.slider('📐 Split Validation', 0.05, 0.4, 0.15, 0.05)
        ml_confidence = st.slider('🎯 Poids ML dans score final', 0.0, 1.0, 0.6, 0.05)
        num_combos = st.number_input('🔢 Nb combinaisons e-trio', min_value=5, max_value=200, value=35, step=1)
        st.markdown('---')
        st.info('🔎 Le modèle DL est entraîné si vous fournissez un target historique (col: "placement" ou "rank"). Sinon, entraînement auto sur pseudo-target.')

    tab1, tab2, tab3 = st.tabs(['🌐 URL Analysis','📁 Upload CSV','🧪 Test Data'])
    df_final = None

    with tab1:
        st.subheader('🔍 Analyse d\'URL de Course')
        col1, col2 = st.columns([3,1])
        with col1:
            url = st.text_input('🌐 URL de la course:', placeholder='https://...')
        with col2:
            analyze_button = st.button('🔍 Analyser')
        if analyze_button and url:
            with st.spinner('🔄 Extraction...'):
                df, msg = scrape_race_data(url)
                if df is not None:
                    st.success(f'✅ {len(df)} chevaux extraits')
                    st.dataframe(df.head())
                    df_final = df
                else:
                    st.error(f'❌ {msg}')

    with tab2:
        st.subheader('📤 Upload CSV (historique possible)')
        uploaded_file = st.file_uploader('Fichier CSV', type='csv')
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f'✅ {len(df_final)} chevaux chargés')
                st.dataframe(df_final.head())
            except Exception as e:
                st.error(f'❌ Erreur: {e}')

    with tab3:
        st.subheader('🧪 Données de Test')
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button('🏃 Test Plat'):
                df_final = pd.DataFrame({'Nom':['A','B','C','D','E'],'Numéro de corde':['1','2','3','4','5'],'Cote':['3.2','4.8','7.5','6.2','9.1'],'Poids':['56.5','57.0','58.5','59.0','57.5'],'Musique':['1a2a3a','2a1a4a','3a3a1a','1a4a2a','4a2a5a'],'Âge/Sexe':['4H','5M','3F','6H','4M']})
                st.success('✅ Données PLAT chargées')
        with c2:
            if st.button('🚗 Test Attelé'):
                df_final = pd.DataFrame({'Nom':['R1','R2','R3','R4','R5'],'Numéro de corde':['1','2','3','4','5'],'Cote':['4.2','8.5','15.0','3.8','6.8'],'Poids':['68.0']*5,'Musique':['2a1a4a','4a3a2a','6a5a8a','1a2a1a','3a4a5a'],'Âge/Sexe':['5H','6M','4F','7H','5M']})
                st.success('✅ Données ATTELÉ chargées')
        with c3:
            if st.button('⭐ Test Premium'):
                df_final = pd.DataFrame({'Nom':['P1','P2','P3','P4','P5'],'Numéro de corde':['1','2','3','4','5'],'Cote':['3.2','4.8','7.5','6.2','9.1'],'Poids':['59.5']*5,'Musique':['1a1a2a','1a3a1a','2a1a4a','1a2a1a','3a1a2a'],'Âge/Sexe':['4H','5H','4H','5F','5F']})
                st.success('✅ Données PREMIUM chargées')
        if df_final is not None:
            st.dataframe(df_final)

    # ----- Core processing -----
    if df_final is not None and len(df_final)>0:
        st.markdown('---')
        st.header('🎯 Analyse et Résultats (DL + heuristiques)')
        df_prep = prepare_data(df_final)
        if len(df_prep)==0:
            st.error('❌ Aucune donnée valide')
            return

        if race_type=='AUTO':
            # heuristic detection
            weight_std = df_prep['weight_kg'].std()
            weight_mean = df_prep['weight_kg'].mean()
            if weight_std>2.5:
                detected='PLAT'
            elif weight_mean>65 and weight_std<1.5:
                detected='ATTELE_AUTOSTART'
            else:
                detected='PLAT'
            st.info(f'🤖 Type détecté: {detected}')
        else:
            detected = race_type
            st.info(f'📋 {CONFIGS[detected]["description"]}')

        # FEATURES
        X = df_prep[model_manager.feature_cols].fillna(0)

        # TARGET: if user provided historical placement/rank use it else pseudo-target from odds+music
        if 'placement' in df_prep.columns or 'rank' in df_prep.columns:
            if 'placement' in df_prep.columns:
                y = 1.0 / (df_prep['placement'].astype(float) + 0.1)
            else:
                y = 1.0 / (df_prep['rank'].astype(float) + 0.1)
            y_source = 'historical'
        else:
            # pseudo target: combine inverse odds and recent_weighted
            y = 0.7*(1.0/(df_prep['odds_numeric']+0.1)) + 0.3*(df_prep['recent_weighted'] / (df_prep['recent_weighted'].max()+1e-6))
            # add slight noise to avoid degenerate solutions
            y = y + np.random.normal(0, 0.02, size=len(y))
            y_source = 'pseudo'

        # Train DL model if enabled
        dl_history = None
        dl_preds = np.zeros(len(X))
        if enable_dl and len(X) >= 4:
            with st.spinner('🏋️‍♂️ Entraînement modèle DL...'):
                try:
                    dl_history = model_manager.train(X.values, y.values, epochs=int(dl_epochs), val_split=float(dl_val))
                    dl_preds = model_manager.predict(X.values)
                    st.success('✅ Modèle DL entraîné')
                except Exception as e:
                    st.warning(f'⚠️ Erreur entraînement DL: {e}')
                    dl_preds = np.zeros(len(X))
        else:
            st.info('ℹ️ DL non activé ou pas assez de données. Utilisation du score traditionnel.')

        # normalize preds
        if dl_preds.max() != dl_preds.min():
            dl_norm = (dl_preds - dl_preds.min())/(dl_preds.max()-dl_preds.min())
        else:
            dl_norm = np.zeros_like(dl_preds)

        # classical score
        trad = 1.0/(df_prep['odds_numeric']+0.1)
        if trad.max()!=trad.min():
            trad = (trad - trad.min())/(trad.max()-trad.min())

        # final score
        final_score = (1-ml_confidence)*trad + ml_confidence*dl_norm
        df_prep['ml_score'] = dl_norm
        df_prep['score_final'] = final_score
        df_ranked = df_prep.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked)+1)

        # ----- Display -----
        c1,c2 = st.columns([2,1])
        with c1:
            st.subheader('🏆 Classement Final')
            display_cols = ['rang','Nom','Cote','Numéro de corde','Poids','score_final']
            disp = df_ranked[display_cols].copy()
            disp['Score'] = disp['score_final'].round(3)
            disp = disp.drop('score_final', axis=1)
            st.dataframe(disp, use_container_width=True)

        with c2:
            st.subheader('📊 Métriques')
            st.markdown(f'<div class="metric-card">🧠 Source target<br><strong>{y_source}</strong></div>', unsafe_allow_html=True)
            favoris = len(df_ranked[df_ranked['odds_numeric']<5])
            outsiders = len(df_ranked[df_ranked['odds_numeric']>15])
            st.markdown(f'<div class="metric-card">⭐ Favoris<br><strong>{favoris}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎲 Outsiders<br><strong>{outsiders}</strong></div>', unsafe_allow_html=True)
            st.subheader('🥇 Top 3')
            for i in range(min(3, len(df_ranked))):
                h = df_ranked.iloc[i]
                st.markdown(f'<div class="prediction-box"><strong>{i+1}. {h["Nom"]}</strong><br>Cote: {h["Cote"]} | Score: {h["score_final"]:.3f}</div>', unsafe_allow_html=True)

        # Visuals
        st.subheader('📊 Visualisations')
        fig = create_visualization(df_ranked)
        st.plotly_chart(fig, use_container_width=True)

        # Generate e-trio combinations
        st.subheader('🎲 Générateur e-trio')
        combos = generate_e_trio(df_ranked, n_combinations=int(num_combos))
        for idx, c in enumerate(combos):
            st.markdown(f'{idx+1}. {c[0]} — {c[1]} — {c[2]}')

        # Export
        st.subheader('💾 Export')
        colx, coly = st.columns(2)
        with colx:
            csv_data = df_ranked.to_csv(index=False)
            st.download_button('📄 CSV', csv_data, f'pronostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        with coly:
            json_data = df_ranked.to_json(orient='records', indent=2)
            st.download_button('📋 JSON', json_data, f'pronostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        # Save model snapshot and show training curve if trained
        if dl_history:
            st.subheader('📈 Courbe d\'entraînement')
            hist = pd.DataFrame(dl_history)
            st.line_chart(hist[['loss','val_loss']])
            st.success(f'✅ Modèle sauvegardé: {model_manager.model_path}')

    st.markdown('---')
    st.markdown('**Instructions GitHub / Déploiement**: voir README généré dans repo.')

if __name__ == '__main__':
    main()
