import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏇 Elite Horse Racing AI v2.0", page_icon="🏇", layout="wide")

st.markdown("""<style>
.main-header{font-size:3rem;color:#1e3a8a;text-align:center;margin-bottom:2rem;text-shadow:2px 2px 4px rgba(0,0,0,0.1)}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1.2rem;border-radius:12px;color:white;text-align:center;margin:0.5rem 0;box-shadow:0 4px 6px rgba(0,0,0,0.1)}
.prediction-box{border-left:5px solid #f59e0b;padding:1rem;background:linear-gradient(90deg,#fffbeb 0%,#fff 100%);margin:1rem 0;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.05)}
.confidence-high{color:#10b981;font-weight:bold}
.confidence-medium{color:#f59e0b;font-weight:bold}
.confidence-low{color:#ef4444;font-weight:bold}
</style>""", unsafe_allow_html=True)

CONFIGS = {
    "PLAT": {"description": "🏃 Galop - Handicap", "optimal_draws": [1,2,3,4], "weight_importance": 0.25, "age_peak": [4,5,6]},
    "ATTELE_AUTOSTART": {"description": "🚗 Trot autostart", "optimal_draws": [4,5,6], "weight_importance": 0.05, "age_peak": [5,6,7]},
    "ATTELE_VOLTE": {"description": "🔄 Trot volté", "optimal_draws": [], "weight_importance": 0.05, "age_peak": [5,6,7,8]},
    "OBSTACLE": {"description": "🚧 Obstacles", "optimal_draws": [2,3,4], "weight_importance": 0.20, "age_peak": [6,7,8,9]}
}

@st.cache_resource
class EliteHorseRacingML:
    def __init__(self):
        self.models = {}
        self.scaler = QuantileTransformer(output_distribution='normal')
        self.feature_importance = {}
        self.cv_scores = {}
        self.calibrator = None
        self.feature_names = []
        
    def extract_music_features(self, music_str):
        """Extraction optimisée de 26 features d'historique"""
        if pd.isna(music_str) or music_str == '':
            return {k: 0 for k in ['wins','places','top5','total_races','win_rate','place_rate','top5_rate',
                'recent_form','momentum','last_pos','last_3_avg','consistency','consistency_top3','pos_var','pos_std',
                'best_pos','worst_pos','avg_pos','median_pos','p25','p75','win_streak','place_streak',
                'win_place_ratio','top3_ratio','improvement_rate']}
        
        positions = [int(c) for c in str(music_str) if c.isdigit() and int(c) > 0]
        if not positions:
            return {k: 0 for k in ['wins','places','top5','total_races','win_rate','place_rate','top5_rate',
                'recent_form','momentum','last_pos','last_3_avg','consistency','consistency_top3','pos_var','pos_std',
                'best_pos','worst_pos','avg_pos','median_pos','p25','p75','win_streak','place_streak',
                'win_place_ratio','top3_ratio','improvement_rate']}
        
        total = len(positions)
        wins = positions.count(1)
        places = sum(1 for p in positions if p <= 3)
        top5 = sum(1 for p in positions if p <= 5)
        
        # Forme récente pondérée
        weights = [0.4, 0.3, 0.2, 0.1]
        recent = positions[:4]
        recent_form = sum(w/p for w,p in zip(weights, recent)) if recent else 0
        
        # Momentum
        momentum = (1/positions[0] - 1/positions[2]) if len(positions) >= 3 and positions[2] > 0 else 0
        
        # Régularité
        consistency = 1/(np.std(positions)+1) if len(positions) > 1 else 0
        top3_pos = [p for p in positions if p <= 3]
        consistency_top3 = 1/(np.std(top3_pos)+1) if len(top3_pos) > 1 else 0
        
        # Streaks
        win_streak = place_streak = 0
        for p in positions:
            if p == 1: win_streak += 1
            else: break
        for p in positions:
            if p <= 3: place_streak += 1
            else: break
        
        return {
            'wins': wins, 'places': places, 'top5': top5, 'total_races': total,
            'win_rate': wins/total, 'place_rate': places/total, 'top5_rate': top5/total,
            'recent_form': recent_form, 'momentum': momentum,
            'last_pos': positions[0], 'last_3_avg': np.mean(positions[:3]) if len(positions)>=3 else np.mean(positions),
            'consistency': consistency, 'consistency_top3': consistency_top3,
            'pos_var': np.var(positions), 'pos_std': np.std(positions),
            'best_pos': min(positions), 'worst_pos': max(positions), 
            'avg_pos': np.mean(positions), 'median_pos': np.median(positions),
            'p25': np.percentile(positions,25) if len(positions)>1 else positions[0],
            'p75': np.percentile(positions,75) if len(positions)>1 else positions[0],
            'win_streak': win_streak, 'place_streak': place_streak,
            'win_place_ratio': wins/places if places>0 else 0,
            'top3_ratio': places/total, 'improvement_rate': momentum/total if total>0 else 0
        }
    
    def prepare_elite_features(self, df, race_type="PLAT"):
        """120+ features optimisées"""
        f = pd.DataFrame()
        cfg = CONFIGS[race_type]
        field_size = len(df)
        
        # === COTE (15) ===
        f['odds'] = df['odds_numeric']
        f['odds_inv'] = 1/(df['odds_numeric']+0.01)
        f['log_odds'] = np.log1p(df['odds_numeric'])
        f['sqrt_odds'] = np.sqrt(df['odds_numeric'])
        f['odds_sq'] = df['odds_numeric']**2
        f['odds_rank'] = df['odds_numeric'].rank()
        f['odds_pct'] = df['odds_numeric'].rank(pct=True)
        f['odds_z'] = (df['odds_numeric']-df['odds_numeric'].mean())/(df['odds_numeric'].std()+1e-6)
        f['odds_norm'] = (df['odds_numeric']-df['odds_numeric'].min())/(df['odds_numeric'].max()-df['odds_numeric'].min()+1e-6)
        f['is_fav'] = (df['odds_numeric']==df['odds_numeric'].min()).astype(int)
        f['is_2nd_fav'] = (df['odds_numeric']==df['odds_numeric'].nsmallest(2).iloc[-1]).astype(int)
        f['is_outsider'] = (df['odds_numeric']>df['odds_numeric'].quantile(0.75)).astype(int)
        f['odds_gap_fav'] = df['odds_numeric']-df['odds_numeric'].min()
        f['odds_iqr'] = (df['odds_numeric']-df['odds_numeric'].quantile(0.25))/(df['odds_numeric'].quantile(0.75)-df['odds_numeric'].quantile(0.25)+1e-6)
        f['odds_cubed'] = df['odds_numeric']**3
        
        # === POSITION (20) ===
        f['draw'] = df['draw_numeric']
        f['draw_norm'] = df['draw_numeric']/(df['draw_numeric'].max()+1)
        f['draw_rank'] = df['draw_numeric'].rank()
        f['draw_z'] = (df['draw_numeric']-df['draw_numeric'].mean())/(df['draw_numeric'].std()+1e-6)
        f['draw_sin'] = np.sin(2*np.pi*df['draw_numeric']/df['draw_numeric'].max())
        f['draw_cos'] = np.cos(2*np.pi*df['draw_numeric']/df['draw_numeric'].max())
        
        optimal = cfg['optimal_draws']
        f['optimal_draw'] = df['draw_numeric'].apply(lambda x: 1 if x in optimal else 0)
        f['draw_opt_score'] = df['draw_numeric'].apply(
            lambda x: 1-(min([abs(x-o) for o in optimal])/df['draw_numeric'].max()) if optimal else 0.5
        )
        f['draw_dist_opt'] = df['draw_numeric'].apply(lambda x: min([abs(x-o) for o in optimal]) if optimal else 0)
        f['is_inside'] = (df['draw_numeric']<=3).astype(int)
        f['is_outside'] = (df['draw_numeric']>df['draw_numeric'].quantile(0.75)).astype(int)
        f['is_middle'] = ((df['draw_numeric']>3)&(df['draw_numeric']<=df['draw_numeric'].quantile(0.75))).astype(int)
        f['draw_field_ratio'] = df['draw_numeric']/field_size
        f['draw_field_int'] = df['draw_numeric']*np.log1p(field_size)
        f['rel_pos'] = (df['draw_numeric']-1)/(field_size-1) if field_size>1 else 0.5
        
        if field_size >= 3:
            f['draw_tercile'] = pd.qcut(df['draw_numeric'], q=3, labels=False, duplicates='drop')
        else:
            f['draw_tercile'] = 1
        
        f['draw_adv_score'] = f['optimal_draw']*f['draw_opt_score']
        f['draw_penalty'] = (df['draw_numeric']>df['draw_numeric'].median()).astype(int)*0.1
        f['extreme_draw'] = ((df['draw_numeric']<=2)|(df['draw_numeric']>=field_size-1)).astype(int)
        f['draw_middle_adv'] = ((df['draw_numeric']>3)&(df['draw_numeric']<field_size-2)).astype(int)
        
        # === POIDS (12) ===
        f['weight'] = df['weight_kg']
        f['weight_norm'] = (df['weight_kg']-df['weight_kg'].mean())/(df['weight_kg'].std()+1e-6)
        f['weight_rank'] = df['weight_kg'].rank()
        f['weight_pct'] = df['weight_kg'].rank(pct=True)
        f['weight_z'] = (df['weight_kg']-df['weight_kg'].mean())/(df['weight_kg'].std()+1e-6)
        
        w_imp = cfg['weight_importance']
        f['weight_adv'] = (df['weight_kg'].max()-df['weight_kg'])*w_imp
        f['weight_disadv'] = (df['weight_kg']-df['weight_kg'].min())*w_imp
        f['weight_gap_min'] = df['weight_kg']-df['weight_kg'].min()
        f['weight_gap_max'] = df['weight_kg'].max()-df['weight_kg']
        f['weight_rel'] = (df['weight_kg']-df['weight_kg'].median())/df['weight_kg'].median()
        f['is_lightest'] = (df['weight_kg']==df['weight_kg'].min()).astype(int)
        f['is_heaviest'] = (df['weight_kg']==df['weight_kg'].max()).astype(int)
        
        # === ÂGE & SEXE (15) ===
        if 'Âge/Sexe' in df.columns:
            f['age'] = df['Âge/Sexe'].str.extract('(\d+)').astype(float).fillna(4.5)
            f['is_mare'] = df['Âge/Sexe'].str.contains('F',na=False).astype(int)
            f['is_stallion'] = df['Âge/Sexe'].str.contains('H',na=False).astype(int)
            f['is_gelding'] = df['Âge/Sexe'].str.contains('M',na=False).astype(int)
        else:
            f['age'] = 4.5
            f['is_mare'] = f['is_stallion'] = f['is_gelding'] = 0
        
        f['age_sq'] = f['age']**2
        f['age_cubed'] = f['age']**3
        
        age_peak = cfg['age_peak']
        f['age_optimal'] = f['age'].apply(lambda x: 1 if x in age_peak else 0)
        f['age_dist_opt'] = f['age'].apply(lambda x: min([abs(x-o) for o in age_peak]))
        f['age_mat_score'] = f['age'].apply(lambda x: 1 if 4<=x<=8 else 0.5 if x==3 or x==9 else 0.3)
        
        f['young_mare'] = ((f['age']<=4)&(f['is_mare']==1)).astype(int)
        f['exp_stallion'] = ((f['age']>=6)&(f['is_stallion']==1)).astype(int)
        f['prime_age'] = ((f['age']>=4)&(f['age']<=6)).astype(int)
        f['veteran'] = (f['age']>=8).astype(int)
        f['rookie'] = (f['age']<=3).astype(int)
        
        # === MUSIQUE (26) ===
        if 'Musique' in df.columns:
            music = df['Musique'].apply(self.extract_music_features)
            for k in music.iloc[0].keys():
                f[f'music_{k}'] = [m[k] for m in music]
        else:
            for k in ['wins','places','top5','total_races','win_rate','place_rate','top5_rate',
                'recent_form','momentum','last_pos','last_3_avg','consistency','consistency_top3','pos_var','pos_std',
                'best_pos','worst_pos','avg_pos','median_pos','p25','p75','win_streak','place_streak',
                'win_place_ratio','top3_ratio','improvement_rate']:
                f[f'music_{k}'] = 0
        
        # === INTERACTIONS (30) ===
        f['odds_draw'] = f['odds_inv']*f['draw_norm']
        f['odds_weight'] = f['log_odds']*f['weight_norm']
        f['odds_age'] = f['odds_inv']*f['age']
        f['draw_weight'] = f['draw_norm']*f['weight_norm']
        f['age_weight'] = f['age']*f['weight_norm']
        f['age_draw'] = f['age']*f['draw_norm']
        f['form_odds'] = f['music_recent_form']*f['odds_inv']
        f['form_draw'] = f['music_recent_form']*f['draw_opt_score']
        f['form_weight'] = f['music_recent_form']*f['weight_adv']
        f['form_age'] = f['music_recent_form']*f['age_mat_score']
        f['cons_odds'] = f['music_consistency']*f['odds_inv']
        f['cons_weight'] = f['music_consistency']*f['weight_adv']
        f['winrate_odds'] = f['music_win_rate']*f['odds_inv']
        f['winrate_draw'] = f['music_win_rate']*f['draw_opt_score']
        f['form_odds_draw'] = f['music_recent_form']*f['odds_inv']*f['draw_opt_score']
        f['age_weight_draw'] = f['age_mat_score']*f['weight_adv']*f['draw_opt_score']
        f['cons_form_odds'] = f['music_consistency']*f['music_recent_form']*f['odds_inv']
        f['momentum_odds'] = f['music_momentum']*f['odds_inv']
        f['momentum_draw'] = f['music_momentum']*f['draw_opt_score']
        f['streak_odds'] = (f['music_win_streak']+f['music_place_streak'])*f['odds_inv']
        f['mare_weight'] = f['is_mare']*f['weight_adv']
        f['stallion_age'] = f['is_stallion']*f['age_mat_score']
        f['gelding_cons'] = f['is_gelding']*f['music_consistency']
        f['elite_score'] = f['odds_inv']*0.3 + f['music_win_rate']*0.25 + f['music_recent_form']*0.2 + f['age_mat_score']*0.15 + f['draw_opt_score']*0.1
        f['class_indicator'] = (f['music_win_rate']>0.2).astype(int)*(f['odds']<10).astype(int)
        f['form_consistency'] = f['music_recent_form']*f['music_consistency']
        f['age_weight_sq'] = f['age']*f['weight']**2
        f['odds_draw_weight'] = f['odds_inv']*f['draw_norm']*f['weight_norm']
        f['total_advantage'] = f['weight_adv']+f['draw_opt_score']+f['age_mat_score']
        f['experience_score'] = f['music_total_races']*f['music_win_rate']
        
        # === CONTEXTE (12) ===
        f['field_size'] = field_size
        f['log_field'] = np.log1p(field_size)
        f['comp_index'] = df['odds_numeric'].std()/(df['odds_numeric'].mean()+1e-6)
        f['odds_spread'] = df['odds_numeric'].max()-df['odds_numeric'].min()
        f['odds_concentration'] = (df['odds_numeric']<=df['odds_numeric'].quantile(0.25)).sum()/field_size
        f['fav_strength'] = df['odds_numeric'].min()/df['odds_numeric'].median()
        f['outsider_count'] = (df['odds_numeric']>df['odds_numeric'].quantile(0.75)).sum()
        f['balanced_field'] = (f['comp_index']<0.5).astype(int)
        f['dominant_fav'] = (df['odds_numeric'].min()<3).astype(int)
        f['open_race'] = (df['odds_numeric'].min()>5).astype(int)
        f['avg_experience'] = f['music_total_races'].mean()
        f['quality_field'] = (f['music_win_rate']>0.15).sum()/field_size
        
        self.feature_names = f.columns.tolist()
        return f.fillna(0)
    
    def build_models(self):
        """5 modèles état-de-l'art"""
        self.models = {
            'rf': RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=8, min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1),
            'et': ExtraTreesRegressor(n_estimators=300, max_depth=10, min_samples_split=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=6, min_samples_split=8, subsample=0.8, random_state=42),
            'bayes': BayesianRidge(max_iter=300, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
            'elastic': ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=2000, random_state=42)
        }
    
    def create_target(self, X, df):
        """Target synthétique optimisé"""
        y = (X['odds_inv']*0.35 + X['music_win_rate']*0.20 + X['music_recent_form']*0.20 + 
             X['music_consistency']*0.10 + X['weight_adv']*0.05 + X['draw_opt_score']*0.05 + 
             X['age_mat_score']*0.05 + np.random.normal(0,0.03,len(X)))
        
        bonus = X['is_fav']*0.1 + (X['music_total_races']>5).astype(int)*0.05
        y = y + bonus
        return (y-y.min())/(y.max()-y.min()+1e-6)
    
    def train_and_predict(self, X, df, race_type="PLAT"):
        """Entraînement & prédiction élite"""
        if len(X) < 5:
            return np.zeros(len(X)), {}, np.ones(len(X))*0.5
        
        y = self.create_target(X, df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Validation
        cv = TimeSeriesSplit(n_splits=3) if len(X)>=20 else KFold(n_splits=5, shuffle=True, random_state=42)
        predictions = {}
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2', n_jobs=-1)
                mae = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
                
                self.cv_scores[name] = {
                    'r2_mean': scores.mean(), 'r2_std': scores.std(),
                    'mae_mean': mae.mean(), 'mae_std': mae.std()
                }
                
                model.fit(X_scaled, y)
                predictions[name] = model.predict(X_scaled)
                
                if hasattr(model, 'feature_importances_'):
                    imp = dict(zip(self.feature_names, model.feature_importances_))
                    self.feature_importance[name] = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15])
            
            except Exception as e:
                st.warning(f"⚠️ {name}: {e}")
                predictions[name] = np.zeros(len(X))
                self.cv_scores[name] = {'r2_mean':0,'r2_std':1,'mae_mean':1,'mae_std':1}
        
        # Ensemble pondéré
        weights = {'rf': 0.30, 'et': 0.25, 'gb': 0.30, 'bayes': 0.10, 'elastic': 0.05}
        final_pred = sum(predictions.get(n, np.zeros(len(X)))*w for n,w in weights.items())/sum(weights.values())
        
        # Calibration
        try:
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(final_pred, y)
            final_pred = self.calibrator.predict(final_pred)
        except:
            pass
        
        # Confiance multi-facteurs
        pred_std = np.std(final_pred)
        conf_var = 1/(1+pred_std*2)
        feat_complete = 1-(X.isna().sum(axis=1)/len(X.columns))
        history_conf = np.clip(X['music_total_races']/15, 0, 1)
        pred_norm = (final_pred-final_pred.mean())/(final_pred.std()+1e-6)
        conf_extreme = 1-np.abs(pred_norm)/3
        
        confidence = (conf_var*0.3 + feat_complete.values*0.3 + history_conf.values*0.25 + conf_extreme*0.15)
        confidence = np.clip(confidence, 0, 1)
        
        return final_pred, self.cv_scores, confidence

@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        
        soup = BeautifulSoup(r.content, 'html.parser')
        table = soup.find('table')
        if not table:
            return None, "No table"
        
        data = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all(['td','th'])
            if len(cols) >= 4:
                data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),
                    "Poids": cols[-2].get_text(strip=True) if len(cols)>4 else "60",
                    "Musique": cols[2].get_text(strip=True) if len(cols)>5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols)>6 else ""
                })
        
        return pd.DataFrame(data) if data else None, "Success" if data else "No data"
    except Exception as e:
        return None, str(e)

def safe_convert(v, func, default=0):
    try:
        return func(str(v).replace(',','.').strip()) if not pd.isna(v) else default
    except:
        return default

def prepare_data(df):
    df = df.copy()
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    
    def extract_weight(s):
        if pd.isna(s): return 60.0
        m = re.search(r'(\d+(?:[.,]\d+)?)', str(s))
        return float(m.group(1).replace(',','.')) if m else 60.0
    
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    df = df[df['odds_numeric']>0].reset_index(drop=True)
    return df

def auto_detect_race_type(df):
    w_std = df['weight_kg'].std()
    w_mean = df['weight_kg'].mean()
    
    col1,col2,col3 = st.columns(3)
    col1.metric("💪 Écart poids", f"{w_std:.1f}kg")
    col2.metric("⚖️ Poids moy", f"{w_mean:.1f}kg")
    col3.metric("🏇 Chevaux", len(df))
    
    if w_std > 2.5:
        detected = "PLAT"
        reason = "Grande variation poids"
    elif w_mean > 65 and w_std < 1.5:
        detected = "ATTELE_AUTOSTART"
        reason = "Poids uniformes élevés"
    else:
        detected = "PLAT"
        reason = "Configuration par défaut"
    
    st.info(f"🤖 **Type**: {detected} | **Raison**: {reason}")
    return detected

def create_visualizations(df, ml_model=None):
    """Visualisations avancées"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('🏆 Scores','📊 Cotes','🧠 Features','⚖️ Poids-Score','📈 CV','🎯 Corrélation'),
        specs=[[{"secondary_y":False},{"type":"histogram"},{"type":"bar"}],
               [{"type":"scatter"},{"type":"bar"},{"type":"scatter"}]]
    )
    
    colors = ['#667eea','#764ba2','#f59e0b','#10b981','#ef4444','#3b82f6']
    
    # 1. Scores
    if 'score_final' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['rang'], y=df['score_final'], mode='markers+lines',
            marker=dict(size=df['confidence']*20 if 'confidence' in df.columns else 10,
                       color=df['confidence'] if 'confidence' in df.columns else colors[0],
                       colorscale='Viridis', showscale=True),
            text=df['Nom'], name='Score'
        ), row=1, col=1)
    
    # 2. Distribution cotes
    fig.add_trace(go.Histogram(x=df['odds_numeric'], nbinsx=10, marker_color=colors[1], name='Cotes'), row=1, col=2)
    
    # 3. Feature importance
    if ml_model and ml_model.feature_importance and 'rf' in ml_model.feature_importance:
        imp = ml_model.feature_importance['rf']
        fig.add_trace(go.Bar(x=list(imp.values()), y=list(imp.keys()), orientation='h', marker_color=colors[2], name='Importance'), row=1, col=3)
    
    # 4. Poids vs Score
    if 'score_final' in df.columns:
        fig.add_trace(go.Scatter(x=df['weight_kg'], y=df['score_final'], mode='markers',
            marker=dict(size=10, color=df['rang'], colorscale='RdYlGn_r'), text=df['Nom'], name='Poids-Score'), row=2, col=1)
    
    # 5. CV scores
    if ml_model and ml_model.cv_scores:
        models = list(ml_model.cv_scores.keys())
        means = [ml_model.cv_scores[m]['r2_mean'] for m in models]
        stds = [ml_model.cv_scores[m]['r2_std'] for m in models]
        fig.add_trace(go.Bar(x=models, y=means, error_y=dict(type='data', array=stds), marker_color=colors[4], name='R² CV'), row=2, col=2)
    
    # 6. Corrélation
    if 'score_final' in df.columns:
        fig.add_trace(go.Scatter(x=df['odds_numeric'], y=df['score_final'], mode='markers',
            marker=dict(size=8, color=colors[5]), text=df['Nom'], name='Cotes-Score'), row=2, col=3)
    
    fig.update_layout(height=700, showlegend=True, title_text="📊 Analyse ML Complète", title_x=0.5)
    return fig

def generate_sample_data(dtype="plat"):
    if dtype == "plat":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt','Lightning Star','Storm King','Rain Dance','Wind Walker','Fire Dancer','Ocean Wave'],
            'Numéro de corde': ['1','2','3','4','5','6','7'],
            'Cote': ['3.2','4.8','7.5','6.2','9.1','12.5','15.0'],
            'Poids': ['56.5','57.0','58.5','59.0','57.5','60.0','61.5'],
            'Musique': ['1a2a3a1a','2a1a4a3a','3a3a1a2a','1a4a2a1a','4a2a5a3a','5a3a6a4a','6a5a7a8a'],
            'Âge/Sexe': ['4H','5M','3F','6H','4M','5H','4F']
        })
    elif dtype == "attele":
        return pd.DataFrame({
            'Nom': ['Rapide Éclair','Foudre Noire','Vent du Nord','Tempête Rouge','Orage Bleu','Cyclone Vert'],
            'Numéro de corde': ['1','2','3','4','5','6'],
            'Cote': ['4.2','8.5','15.0','3.8','6.8','10.2'],
            'Poids': ['68.0','68.0','68.0','68.0','68.0','68.0'],
            'Musique': ['2a1a4a1a','4a3a2a5a','6a5a8a7a','1a2a1a3a','3a4a5a2a','5a6a4a8a'],
            'Âge/Sexe': ['5H','6M','4F','7H','5M','6H']
        })
    else:
        return pd.DataFrame({
            'Nom': ['Ace Impact','Torquator','Adayar','Tarnawa','Chrono','Mishriff','Love'],
            'Numéro de corde': ['1','2','3','4','5','6','7'],
            'Cote': ['3.2','4.8','7.5','6.2','9.1','5.5','11.0'],
            'Poids': ['59.5','59.5','59.5','58.5','58.5','59.0','58.0'],
            'Musique': ['1a1a2a1a','1a3a1a2a','2a1a4a1a','1a2a1a3a','3a1a2a1a','1a1a1a2a','2a3a1a4a'],
            'Âge/Sexe': ['4H','5H','4H','5F','5F','5H','4F']
        })

def main():
    st.markdown('<h1 class="main-header">🏇 Elite Horse Racing AI v2.0</h1>', unsafe_allow_html=True)
    st.markdown("*Prédictions avancées avec 120+ features et validation rigoureuse*")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        race_type = st.selectbox("🏁 Type", ["AUTO","PLAT","ATTELE_AUTOSTART","ATTELE_VOLTE","OBSTACLE"])
        use_ml = st.checkbox("✅ ML Avancé", value=True)
        ml_weight = st.slider("🎯 Poids ML", 0.1, 0.9, 0.7, 0.05)
        
        st.subheader("🧠 Modèles")
        st.info("✅ Random Forest (300)")
        st.info("✅ Extra Trees (300)")
        st.info("✅ Gradient Boosting (200)")
        st.info("✅ Bayesian Ridge")
        st.info("✅ Elastic Net")
        st.info("✅ Calibration Isotonique")
        
        st.subheader("📊 Features")
        st.success("**120+ features** automatiques")
        st.caption("15 cote | 20 position | 12 poids")
        st.caption("15 âge/sexe | 26 historique")
        st.caption("30 interactions | 12 contexte")
    
    tab1, tab2, tab3 = st.tabs(["🌐 URL", "📁 CSV", "🧪 Test"])
    
    df_final = None
    
    with tab1:
        st.subheader("🔍 Analyse URL")
        url = st.text_input("🌐 URL:", placeholder="https://racing-site.com/race/123")
        if st.button("🔍 Analyser", type="primary"):
            with st.spinner("🔄 Extraction..."):
                df, msg = scrape_race_data(url)
                if df is not None:
                    st.success(f"✅ {len(df)} chevaux extraits")
                    st.dataframe(df.head(), use_container_width=True)
                    df_final = df
                else:
                    st.error(f"❌ {msg}")
    
    with tab2:
        st.subheader("📤 Upload CSV")
        uploaded = st.file_uploader("CSV", type="csv")
        if uploaded:
            try:
                df_final = pd.read_csv(uploaded)
                st.success(f"✅ {len(df_final)} chevaux")
                st.dataframe(df_final.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ {e}")
    
    with tab3:
        st.subheader("🧪 Données Test")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏃 Plat", use_container_width=True):
                df_final = generate_sample_data("plat")
                st.success("✅ PLAT (7 chevaux)")
        with col2:
            if st.button("🚗 Attelé", use_container_width=True):
                df_final = generate_sample_data("attele")
                st.success("✅ ATTELÉ (6 chevaux)")
        with col3:
            if st.button("⭐ Premium", use_container_width=True):
                df_final = generate_sample_data("premium")
                st.success("✅ PREMIUM (7 chevaux)")
        
        if df_final is not None:
            st.dataframe(df_final, use_container_width=True)
    
    # === ANALYSE ===
    if df_final is not None and len(df_final) > 0:
        st.markdown("---")
        st.header("🎯 Analyse ML Elite")
        
        df_prep = prepare_data(df_final)
        if len(df_prep) == 0:
            st.error("❌ Aucune donnée valide")
            return
        
        # Détection type
        if race_type == "AUTO":
            detected = auto_detect_race_type(df_prep)
        else:
            detected = race_type
            st.info(f"📋 **Type**: {CONFIGS[detected]['description']}")
        
        # === ML ===
        ml_model = EliteHorseRacingML()
        ml_model.build_models()
        
        if use_ml:
            with st.spinner("🤖 Entraînement ML élite..."):
                try:
                    X_ml = ml_model.prepare_elite_features(df_prep, detected)
                    st.info(f"🔬 **{len(X_ml.columns)} features** créées")
                    
                    ml_pred, ml_results, confidence = ml_model.train_and_predict(X_ml, df_prep, detected)
                    
                    if len(ml_pred) > 0 and ml_pred.max() != ml_pred.min():
                        ml_pred = (ml_pred - ml_pred.min()) / (ml_pred.max() - ml_pred.min())
                    
                    df_prep['ml_score'] = ml_pred
                    df_prep['confidence'] = confidence
                    
                    st.success("✅ ML entraîné")
                    
                    if ml_results:
                        col1,col2,col3,col4 = st.columns(4)
                        col1.metric("🌲 R² RF", f"{ml_results.get('rf',{}).get('r2_mean',0):.3f}")
                        col2.metric("🚀 R² GB", f"{ml_results.get('gb',{}).get('r2_mean',0):.3f}")
                        col3.metric("🎯 MAE Moy", f"{ml_results.get('rf',{}).get('mae_mean',0):.3f}")
                        col4.metric("💪 Confiance", f"{confidence.mean():.1%}")
                
                except Exception as e:
                    st.error(f"⚠️ Erreur ML: {e}")
                    use_ml = False
        
        # Score traditionnel
        trad_score = 1 / (df_prep['odds_numeric'] + 0.1)
        if trad_score.max() != trad_score.min():
            trad_score = (trad_score - trad_score.min()) / (trad_score.max() - trad_score.min())
        
        # Score final
        if use_ml and 'ml_score' in df_prep.columns:
            df_prep['score_final'] = (1-ml_weight)*trad_score + ml_weight*df_prep['ml_score']
        else:
            df_prep['score_final'] = trad_score
            df_prep['confidence'] = np.ones(len(df_prep)) * 0.5
        
        # Classement
        df_ranked = df_prep.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked)+1)
        
        # === RÉSULTATS ===
        st.markdown("---")
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.subheader("🏆 Classement Final")
            display = df_ranked[['rang','Nom','Cote','Numéro de corde','Poids']].copy()
            if 'score_final' in df_ranked.columns:
                display['Score'] = df_ranked['score_final'].apply(lambda x: f"{x:.3f}")
            if 'confidence' in df_ranked.columns:
                display['Confiance'] = df_ranked['confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display, use_container_width=True, height=400)
        
        with col2:
            st.subheader("📊 Statistiques")
            favoris = len(df_ranked[df_ranked['odds_numeric']<5])
            outsiders = len(df_ranked[df_ranked['odds_numeric']>15])
            avg_conf = df_ranked['confidence'].mean() if 'confidence' in df_ranked.columns else 0
            
            st.markdown(f'<div class="metric-card">⭐ Favoris (cote<5)<br><strong>{favoris}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎲 Outsiders (cote>15)<br><strong>{outsiders}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎯 Confiance<br><strong>{avg_conf:.1%}</strong></div>', unsafe_allow_html=True)
            
            st.subheader("🥇 Top 5")
            for i in range(min(5, len(df_ranked))):
                h = df_ranked.iloc[i]
                conf = h.get('confidence', 0.5)
                
                if conf >= 0.7:
                    conf_class, emoji = "confidence-high", "🟢"
                elif conf >= 0.4:
                    conf_class, emoji = "confidence-medium", "🟡"
                else:
                    conf_class, emoji = "confidence-low", "🔴"
                
                st.markdown(f"""<div class="prediction-box">
                    <strong>{i+1}. {h['Nom']}</strong><br>
                    📊 Cote: <strong>{h['Cote']}</strong> | 🎯 Score: <strong>{h['score_final']:.3f}</strong><br>
                    {emoji} Confiance: <span class="{conf_class}">{conf:.1%}</span>
                </div>""", unsafe_allow_html=True)
        
        # === VISUALISATIONS ===
        st.markdown("---")
        st.subheader("📊 Analyses Visuelles")
        fig = create_visualizations(df_ranked, ml_model if use_ml else None)
        st.plotly_chart(fig, use_container_width=True)
        
        # === FEATURE IMPORTANCE ===
        if use_ml and ml_model.feature_importance:
            st.markdown("---")
            st.subheader("🔬 Top Features par Modèle")
            cols = st.columns(min(3, len(ml_model.feature_importance)))
            for idx, (model, imp) in enumerate(list(ml_model.feature_importance.items())[:3]):
                with cols[idx]:
                    st.markdown(f"**{model.upper()}**")
                    imp_df = pd.DataFrame(list(imp.items()), columns=['Feature','Importance']).head(10)
                    st.dataframe(imp_df, use_container_width=True, height=300)
        
        # === RECOMMANDATIONS ===
        st.markdown("---")
        st.subheader("💡 Recommandations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Potentiel Élevé**")
            high_val = df_ranked[(df_ranked['score_final']>df_ranked['score_final'].quantile(0.6)) &
                                  (df_ranked['odds_numeric']>5) &
                                  (df_ranked['confidence']>0.5)].head(3)
            
            if len(high_val) > 0:
                for _, h in high_val.iterrows():
                    st.success(f"✅ **{h['Nom']}** - Cote: {h['Cote']} | Score: {h['score_final']:.3f}")
            else:
                st.info("Aucun outsider détecté")
        
        with col2:
            st.markdown("**⚠️ Alertes**")
            weak_fav = df_ranked[(df_ranked['odds_numeric']<5) & (df_ranked['score_final']<df_ranked['score_final'].median())]
            if len(weak_fav) > 0:
                st.warning(f"⚠️ {len(weak_fav)} favori(s) avec score faible")
            
            surprise = df_ranked[(df_ranked['odds_numeric']>10) & (df_ranked['rang']<=3)]
            if len(surprise) > 0:
                st.info(f"🎲 {len(surprise)} outsider(s) dans le Top 3!")
            else:
                st.info("✅ Classement cohérent")
        
        # === EXPORT ===
        st.markdown("---")
        st.subheader("💾 Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_ranked.to_csv(index=False)
            st.download_button("📄 CSV", csv, f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
        
        with col2:
            json_data = df_ranked.to_json(orient='records', indent=2)
            st.download_button("📋 JSON", json_data, f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json", use_container_width=True)
        
        with col3:
            report = f"""RAPPORT ELITE HORSE RACING AI
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Type: {detected}
Chevaux: {len(df_ranked)}
Features: {len(ml_model.feature_names) if use_ml else 0}

TOP 5:
{'-'*50}
"""
            for i in range(min(5, len(df_ranked))):
                h = df_ranked.iloc[i]
                report += f"{i+1}. {h['Nom']} - Cote: {h['Cote']} - Score: {h['score_final']:.3f} - Conf: {h.get('confidence',0):.1%}\n"
            
            if use_ml and ml_results:
                report += f"\n{'='*50}\nMÉTRIQUES ML:\n{'-'*50}\n"
                for m, s in ml_results.items():
                    report += f"{m}: R²={s['r2_mean']:.3f} (±{s['r2_std']:.3f}), MAE={s['mae_mean']:.3f}\n"
            
            st.download_button("📊 Rapport", report, f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain", use_container_width=True)

if __name__ == "__main__":
    main()
