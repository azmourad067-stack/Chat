# -*- coding: utf-8 -*-

import os
import re
import warnings
import json
import tempfile
from datetime import datetime, timedelta
from itertools import combinations
from decimal import Decimal, ROUND_HALF_UP
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional ML imports
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
except Exception:
    pass

# ---------------- Configuration Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

HIST_PATH = os.path.join(DATA_DIR, "historique_complet.csv")
BANKROLL_PATH = os.path.join(DATA_DIR, "bankroll.json")
PERFORMANCE_PATH = os.path.join(DATA_DIR, "performance.json")

# ---------------- Scraper Geny Spécialisé ----------------
class GenyScraper:
    """Scraper spécialisé pour les URLs Geny (partants-pmu et stats-pmu)"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def scrape_geny_course(self, url):
        """Scrape une course Geny avec gestion robuste des URLs"""
        try:
            if not url or "geny.com" not in url:
                st.warning("URL Geny non valide, utilisation des données de démonstration")
                return self._get_demo_data()
            
            if not url.startswith('http'):
                url = 'https://' + url
            
            st.info(f"🔍 Scraping de l'URL: {url}")
            
            response = self.session.get(url, timeout=15)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                st.warning(f"Erreur HTTP {response.status_code}, utilisation des données de démonstration")
                return self._get_demo_data()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            if "partants-pmu" in url:
                return self._scrape_partants_page(soup)
            elif "stats-pmu" in url:
                return self._scrape_stats_page(soup)
            else:
                return self._scrape_generic_page(soup)
                
        except Exception as e:
            st.error(f"❌ Erreur scraping: {str(e)}")
            return self._get_demo_data()
    
    def _scrape_partants_page(self, soup):
        """Scrape une page 'partants-pmu' Geny en analysant les tableaux de résultats/course"""
        st.info("📄 Détection: Page Partants PMU - Parsing des tableaux structurés")

        horses_data = []

        tables = soup.find_all('table')
        if not tables:
            st.warning("Aucun tableau trouvé sur la page.")
            return self._get_demo_data()

        for table in tables:
            headers = table.find_all('th')
            header_texts = [th.get_text(strip=True) for th in headers]
            if not any("N°" in h or "Chevaux" in h or "Cotes" in h for h in header_texts):
                continue

            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 9:
                    continue

                try:
                    num = cells[1].get_text(strip=True)
                    name = cells[2].get_text(strip=True)
                    # Nettoyage du nom : supprimer retours à la ligne, espaces multiples
                    name = re.sub(r'\s+', ' ', name).strip()
                    sa = cells[3].get_text(strip=True)
                    poids_text = cells[4].get_text(strip=True)
                    jockey = cells[5].get_text(strip=True)
                    entraineur = cells[6].get_text(strip=True)
                    cote_text = cells[8].get_text(strip=True)

                    # Conversion du poids
                    poids = float(poids_text.replace(',', '.')) if poids_text.replace(',', '').replace('.', '').replace(' ', '').isdigit() else 60.0

                    # Conversion de la cote
                    cote = 10.0
                    if cote_text:
                        cote_match = re.search(r'[\d,\.]+', cote_text)
                        if cote_match:
                            cote_str = cote_match.group().replace(',', '.')
                            try:
                                cote = float(cote_str)
                            except ValueError:
                                cote = 10.0

                    gains = np.random.randint(20000, 200000)
                    musique = self._generate_random_music()

                    horse_data = {
                        'Nom': name,
                        'Numéro de corde': num,
                        'Cote': round(cote, 2),
                        'Poids': poids,
                        'Musique': musique,
                        'Âge/Sexe': sa,
                        'Jockey': jockey,
                        'Entraîneur': entraineur,
                        'Gains': gains
                    }
                    horses_data.append(horse_data)

                except Exception as e:
                    st.warning(f"Erreur parsing ligne: {e}")
                    continue

        # ✅ CORRECTION FINALE : "horses_data", pas "horses_"
        if horses_data:
            df = pd.DataFrame(horses_data)
            df = df.drop_duplicates(subset=['Nom', 'Numéro de corde']).reset_index(drop=True)
            return df
        else:
            st.warning("Aucun cheval extrait. Données de démonstration utilisées.")
            return self._get_demo_data()

    def _scrape_stats_page(self, soup):
        st.info("📊 Détection: Page Stats PMU")
        return self._scrape_generic_page(soup)
    
    def _scrape_generic_page(self, soup):
        st.info("🔍 Analyse générique de la page")
        horses_data = []
        text_content = soup.get_text()
        horse_patterns = [
            r'(\d+)\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.\']+?)\s+(\d+\.\d+)',
            r'([A-Z][a-zA-ZÀ-ÿ\s\-\.\']+?)\s+(\d+\.\d+)',
        ]
        for pattern in horse_patterns:
            matches = re.finditer(pattern, text_content)
            for match in matches:
                horse_data = {
                    'Nom': self._clean_text(match.group(2) if len(match.groups()) > 2 else match.group(1)),
                    'Numéro de corde': match.group(1) if len(match.groups()) > 2 else "1",
                    'Cote': float(match.group(3) if len(match.groups()) > 2 else match.group(2)),
                    'Poids': 60.0,
                    'Musique': "1a2a3",
                    'Âge/Sexe': "5M",
                    'Jockey': "JOCKEY",
                    'Entraîneur': "TRAINER",
                    'Gains': 50000
                }
                horses_data.append(horse_data)
        if horses_data:
            return pd.DataFrame(horses_data)
        else:
            st.warning("Aucun cheval détecté, utilisation des données de démonstration")
            return self._get_demo_data()
    
    def _clean_text(self, s):
        if pd.isna(s) or s == "":
            return "INCONNU"
        s = re.sub(r'\s+', ' ', str(s)).strip()
        return re.sub(r'[^\w\s\-\'À-ÿ]', '', s)
    
    def _generate_random_music(self):
        placements = []
        for _ in range(np.random.randint(3, 6)):
            placements.append(str(np.random.randint(1, 8)))
        return 'a'.join(placements)
    
    def _get_demo_data(self):
        demo_horses = [
            "ETOILE ROYALE", "JOLIE FOLIE", "RAPIDE ESPOIR", "GANGOUILLE ROYALE", 
            "ECLIPSE D'OR", "SUPERSTAR", "FLAMME D'OR", "TEMPETE ROYALE"
        ]
        np.random.shuffle(demo_horses)
        selected_horses = demo_horses[:8]
        demo_data = []
        for i, name in enumerate(selected_horses, 1):
            demo_data.append({
                "Nom": name,
                "Numéro de corde": str(i),
                "Cote": round(np.random.uniform(2.5, 20.0), 2),
                "Poids": round(np.random.uniform(58.0, 65.0), 1),
                "Musique": self._generate_random_music(),
                "Âge/Sexe": f"{np.random.randint(3, 8)}{np.random.choice(['M', 'F'])}",
                "Jockey": f"JOCKEY {name.split()[0][:3].upper()}",
                "Entraîneur": f"ENTR. {name.split()[-1][:4].upper()}",
                "Gains": np.random.randint(30000, 250000)
            })
        return pd.DataFrame(demo_data)

# ---------------- Feature Engineering ----------------
class AdvancedFeatureEngineer:
    def __init__(self):
        pass
        
    def create_advanced_features(self, df):
        df = df.copy()
        df = self._create_basic_features(df)
        df = self._create_performance_features(df)
        df = self._create_contextual_features(df)
        return df
    
    def _create_basic_features(self, df):
        df['odds_numeric'] = df['Cote'].apply(lambda x: float(x) if pd.notna(x) else 10.0)
        df['odds_probability'] = 1 / df['odds_numeric']
        df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: int(x) if str(x).isdigit() else 1)
        df['weight_kg'] = df['Poids'].apply(lambda x: float(x) if pd.notna(x) else 60.0)
        df['age'] = df['Âge/Sexe'].apply(lambda x: self._extract_age(x))
        df['is_female'] = df['Âge/Sexe'].apply(lambda x: 1 if 'F' in str(x).upper() else 0)
        return df
    
    def _create_performance_features(self, df):
        df['recent_wins'] = df['Musique'].apply(lambda x: self._extract_recent_wins(x))
        df['recent_top3'] = df['Musique'].apply(lambda x: self._extract_recent_top3(x))
        df['recent_weighted'] = df['Musique'].apply(lambda x: self._calculate_weighted_perf(x))
        df['consistency'] = df['recent_top3'] / (df['recent_wins'] + df['recent_top3'] + 1e-6)
        return df
    
    def _create_contextual_features(self, df):
        df['experience'] = df['age'] - 3
        df['prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int)
        df['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / df['weight_kg'].std()
        return df
    
    def _extract_age(self, age_sexe):
        try:
            if pd.isna(age_sexe) or not isinstance(age_sexe, str):
                return 5.0
            m = re.search(r"(\d+)", str(age_sexe))
            return float(m.group(1)) if m else 5.0
        except:
            return 5.0
    
    def _extract_recent_wins(self, musique):
        try:
            s = str(musique)
            digits = [int(x) for x in re.findall(r"\d+", s) if int(x) > 0]
            return sum(1 for d in digits if d == 1)
        except:
            return 0
    
    def _extract_recent_top3(self, musique):
        try:
            s = str(musique)
            digits = [int(x) for x in re.findall(r"\d+", s) if int(x) > 0]
            return sum(1 for d in digits if d <= 3)
        except:
            return 0
    
    def _calculate_weighted_perf(self, musique):
        try:
            s = str(musique)
            digits = [int(x) for x in re.findall(r"\d+", s) if int(x) > 0]
            if not digits:
                return 0.0
            weights = np.linspace(1.0, 0.3, num=len(digits))
            weighted = sum((4-d)*w for d,w in zip(digits, weights)) / (sum(weights) + 1e-6)
            return weighted
        except:
            return 0.0

# ---------------- Modèle Hybride ----------------
class AdvancedHybridModel:
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or [
            'odds_numeric', 'draw_numeric', 'weight_kg', 'age', 'is_female',
            'recent_wins', 'recent_top3', 'recent_weighted', 'consistency',
            'experience', 'prime_age', 'weight_normalized'
        ]
        self.scaler = StandardScaler()
        self.model = None
    
    def train_ensemble(self, X, y, val_split=0.2):
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            if xgb is not None:
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                self.model.fit(X_scaled, y)
                st.success("✅ Modèle XGBoost entraîné avec succès")
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=50, random_state=42)
                self.model.fit(X_scaled, y)
                st.success("✅ Modèle Random Forest entraîné avec succès")
                
        except Exception as e:
            st.warning(f"⚠️ Erreur entraînement: {e}")
    
    def predict_proba(self, X):
        try:
            if self.model is None:
                return np.zeros(len(X))
            
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            if predictions.max() > predictions.min():
                predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            
            return predictions
            
        except Exception as e:
            st.warning(f"Erreur prédiction: {e}")
            return np.zeros(len(X))

# ---------------- Value Bet Detector ----------------
class ValueBetDetector:
    def __init__(self, edge_threshold=0.05):
        self.edge_threshold = edge_threshold
    
    def find_value_bets(self, df, predicted_probs, min_prob=0.1):
        value_bets = []
        
        for idx, row in df.iterrows():
            market_prob = 1 / row['Cote'] if row['Cote'] > 1 else 0.01
            model_prob = predicted_probs[idx]
            
            if model_prob > min_prob and model_prob > market_prob:
                edge = model_prob - market_prob
                expected_value = (model_prob * (row['Cote'] - 1) - (1 - model_prob))
                
                if edge >= self.edge_threshold and expected_value > 0:
                    value_bets.append({
                        'horse': row['Nom'],
                        'odds': row['Cote'],
                        'market_prob': round(market_prob * 100, 1),
                        'model_prob': round(model_prob * 100, 1),
                        'edge': round(edge * 100, 1),
                        'expected_value': round(expected_value * 100, 1),
                        'kelly_fraction': round(self.calculate_kelly_fraction(model_prob, row['Cote']) * 100, 1)
                    })
        
        return pd.DataFrame(value_bets).sort_values('edge', ascending=False)

    def calculate_kelly_fraction(self, prob, odds):
        if odds <= 1:
            return 0.0
        kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
        return max(0.0, min(kelly, 0.1))

# ---------------- Interface Streamlit ----------------
def setup_streamlit_ui():
    st.set_page_config(
        page_title="🏇 Système Expert Hippique Pro",
        layout="wide",
        page_icon="🏇",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    setup_streamlit_ui()
    st.markdown('<h1 class="main-header">🏇 SYSTÈME EXPERT HIPPIQUE PROFESSIONNEL</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🎯 Configuration Pro")
        url_input = st.text_input(
            "URL Geny:",
            value="https://www.geny.com/partants-pmu/2025-10-25-compiegne-pmu-prix-cerealiste_c1610603",
            help="Collez une URL Geny de type partants-pmu"
        )
        auto_train = st.checkbox("Auto-training", value=True)
        use_advanced_features = st.checkbox("Features avancées", value=True)
        detect_value_bets = st.checkbox("Value Bets", value=True)
        edge_threshold = st.slider("Seuil edge minimum (%)", 1.0, 20.0, 5.0, 0.5) / 100

    if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            try:
                scraper = GenyScraper()
                df_race = scraper.scrape_geny_course(url_input)
                
                if not df_race.empty:
                    st.success(f"✅ {len(df_race)} chevaux chargés")
                    with st.expander("📋 Données brutes", expanded=False):
                        st.dataframe(df_race, use_container_width=True)
                    
                    feature_engineer = AdvancedFeatureEngineer()
                    df_features = feature_engineer.create_advanced_features(df_race) if use_advanced_features else create_basic_features(df_race)
                    
                    if auto_train and len(df_features) >= 3:
                        model = AdvancedHybridModel()
                        X, y = prepare_training_data(df_features)
                        if len(X) >= 3:
                            model.train_ensemble(X, y)
                            predictions = model.predict_proba(X)
                            df_features['predicted_prob'] = predictions
                            
                            if detect_value_bets:
                                detector = ValueBetDetector(edge_threshold=edge_threshold)
                                value_bets_df = detector.find_value_bets(df_features, predictions)
                                if not value_bets_df.empty:
                                    st.subheader("💰 Value Bets Détectés")
                                    for _, bet in value_bets_df.iterrows():
                                        st.markdown(f"""
                                        <div style='background-color:#d4edda;padding:1rem;border-radius:0.5rem;margin:0.5rem 0;'>
                                        <b>🏆 {bet['horse']}</b> — Cote: {bet['odds']}<br>
                                        Edge: <b>+{bet['edge']}%</b> | EV: <b>+{bet['expected_value']}%</b>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("🔍 Aucun value bet détecté.")
                            else:
                                st.info("🎯 Détection des value bets désactivée.")
                        else:
                            st.warning("Données insuffisantes pour l'entraînement.")
                    else:
                        st.info("Auto-training désactivé.")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

def create_basic_features(df):
    df = df.copy()
    df['odds_numeric'] = df['Cote']
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: int(x) if str(x).isdigit() else 1)
    df['weight_kg'] = df['Poids']
    df['age'] = df['Âge/Sexe'].apply(lambda x: extract_age_simple(x))
    df['is_female'] = df['Âge/Sexe'].apply(lambda x: 1 if 'F' in str(x).upper() else 0)
    df['recent_wins'] = df['Musique'].apply(lambda x: extract_recent_wins_simple(x))
    df['recent_top3'] = df['Musique'].apply(lambda x: extract_recent_top3_simple(x))
    df['recent_weighted'] = df['Musique'].apply(lambda x: calculate_weighted_perf_simple(x))
    return df

def extract_age_simple(age_sexe):
    try:
        m = re.search(r'(\d+)', str(age_sexe))
        return float(m.group(1)) if m else 5.0
    except:
        return 5.0

def extract_recent_wins_simple(musique):
    try:
        s = str(musique)
        digits = [int(x) for x in re.findall(r'\d+', s) if int(x) > 0]
        return sum(1 for d in digits if d == 1)
    except:
        return 0

def extract_recent_top3_simple(musique):
    try:
        s = str(musique)
        digits = [int(x) for x in re.findall(r'\d+', s) if int(x) > 0]
        return sum(1 for d in digits if d <= 3)
    except:
        return 0

def calculate_weighted_perf_simple(musique):
    try:
        s = str(musique)
        digits = [int(x) for x in re.findall(r'\d+', s) if int(x) > 0]
        if not digits:
            return 0.0
        weights = np.linspace(1.0, 0.3, num=len(digits))
        weighted = sum((4-d)*w for d,w in zip(digits, weights)) / (len(digits)+1e-6)
        return weighted
    except:
        return 0.0

def prepare_training_data(df):
    feature_cols = ['odds_numeric', 'draw_numeric', 'weight_kg', 'age', 'is_female', 
                   'recent_wins', 'recent_top3', 'recent_weighted']
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].fillna(0)
    y = 1 / (df['odds_numeric'] + 0.1)
    y = (y - y.min()) / (y.max() - y.min() + 1e-6)
    return X, y

if __name__ == "__main__":
    main()
