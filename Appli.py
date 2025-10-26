import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import re
import warnings
import sys
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class AnalyseurHippique:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.features_names = []
        
    def extraire_donnees_geny(self, url):
        """Extrait les données d'une course depuis une URL Geny.com"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraction des informations de course
            course_info = self._extraire_info_course(soup)
            
            # Extraction des partants
            partants = self._extraire_partants(soup)
            
            if not partants:
                raise ValueError("Aucun partant trouvé")
                
            print(f"✅ {len(partants)} chevaux extraits avec succès!")
            
            return {
                'course_info': course_info,
                'partants': partants
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction: {str(e)}")
            return None
    
    def _extraire_info_course(self, soup):
        """Extrait les informations générales de la course"""
        info = {}
        
        try:
            # Titre de la course
            title = soup.find('h2')
            if title:
                info['titre'] = title.get_text(strip=True)
            
            # Distance et type
            course_details = soup.find_all('text')
            for detail in course_details:
                text = detail.get_text()
                if 'm -' in text:
                    distance_match = re.search(r'(\d+)m', text)
                    if distance_match:
                        info['distance'] = int(distance_match.group(1))
                
                if 'Plat' in text:
                    info['type'] = 'PLAT'
                elif 'Obstacle' in text or 'Haies' in text:
                    info['type'] = 'OBSTACLE'
                else:
                    info['type'] = 'PLAT'  # par défaut
                    
            # Terrain
            terrain_elem = soup.find(string=re.compile(r'Terrain\s*:\s*'))
            if terrain_elem:
                terrain = terrain_elem.strip().split(':')[-1].strip()
                info['terrain'] = terrain
                
        except Exception as e:
            print(f"Erreur extraction info course: {e}")
            
        return info
    
    def _extraire_partants(self, soup):
        """Extrait les données des partants"""
        partants = []
        
        try:
            # Recherche du tableau des partants
            tables = soup.find_all('table')
            partants_table = None
            
            for table in tables:
                headers = table.find_all('th') if table.find('thead') else table.find_all('td')
                header_text = ' '.join([h.get_text().strip() for h in headers[:5]])
                
                if any(keyword in header_text.lower() for keyword in ['cheval', 'poids', 'jockey', 'n°']):
                    partants_table = table
                    break
            
            if not partants_table:
                # Essai avec une structure différente
                rows = soup.find_all('tr')
                for i, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 8:  # Nombre minimum de colonnes attendu
                        try:
                            # Tentative d'extraction des données
                            numero = cells[0].get_text(strip=True)
                            if numero.isdigit():
                                partant = self._extraire_partant_from_row(cells)
                                if partant:
                                    partants.append(partant)
                        except:
                            continue
            else:
                # Extraction depuis le tableau trouvé
                rows = partants_table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 6:
                        partant = self._extraire_partant_from_row(cells)
                        if partant:
                            partants.append(partant)
                            
        except Exception as e:
            print(f"Erreur extraction partants: {e}")
            
        return partants
    
    def _extraire_partant_from_row(self, cells):
        """Extrait les données d'un partant depuis une ligne de tableau"""
        try:
            partant = {}
            
            # Numéro
            partant['numero'] = int(cells[0].get_text(strip=True))
            
            # Nom du cheval
            cheval_cell = cells[1]
            partant['cheval'] = cheval_cell.get_text(strip=True)
            
            # Corde (si disponible)
            if len(cells) > 2:
                corde_text = cells[2].get_text(strip=True)
                partant['corde'] = int(corde_text) if corde_text.isdigit() else partant['numero']
            
            # Sexe/Age
            if len(cells) > 3:
                sa_text = cells[3].get_text(strip=True)
                partant['sexe_age'] = sa_text
                
                # Parsing sexe et âge
                match = re.match(r'([HFM])(\d+)', sa_text)
                if match:
                    partant['sexe'] = match.group(1)
                    partant['age'] = int(match.group(2))
                else:
                    partant['sexe'] = 'H'
                    partant['age'] = 5
            
            # Poids
            if len(cells) > 4:
                poids_text = cells[4].get_text(strip=True)
                poids_match = re.search(r'(\d+)', poids_text)
                if poids_match:
                    partant['poids'] = float(poids_match.group(1))
                else:
                    partant['poids'] = 60.0
            
            # Décharge
            if len(cells) > 5:
                decharge_text = cells[5].get_text(strip=True)
                if decharge_text and decharge_text != '-':
                    try:
                        partant['decharge'] = float(decharge_text.replace(',', '.'))
                    except:
                        partant['decharge'] = 0.0
                else:
                    partant['decharge'] = 0.0
            
            # Jockey
            if len(cells) > 6:
                partant['jockey'] = cells[6].get_text(strip=True)
            
            # Entraîneur
            if len(cells) > 7:
                partant['entraineur'] = cells[7].get_text(strip=True)
            
            # Musique
            if len(cells) > 8:
                partant['musique'] = cells[8].get_text(strip=True)
            
            # Valeur
            if len(cells) > 9:
                valeur_text = cells[9].get_text(strip=True)
                try:
                    partant['valeur'] = float(valeur_text) if valeur_text else 30.0
                except:
                    partant['valeur'] = 30.0
            
            # Cotes
            if len(cells) > 10:
                cote_text = cells[10].get_text(strip=True)
                try:
                    partant['cote_ref'] = float(cote_text.replace(',', '.'))
                except:
                    partant['cote_ref'] = 10.0
            
            if len(cells) > 11:
                cote_text = cells[11].get_text(strip=True)
                try:
                    partant['cote_actuelle'] = float(cote_text.replace(',', '.'))
                except:
                    partant['cote_actuelle'] = partant.get('cote_ref', 10.0)
            
            return partant
            
        except Exception as e:
            print(f"Erreur extraction partant: {e}")
            return None
    
    def creer_features(self, donnees):
        """Crée les features pour l'analyse ML"""
        partants = donnees['partants']
        df = pd.DataFrame(partants)
        
        print("🔧 Création des features avancées...")
        
        # Features de base
        features = ['numero', 'corde', 'age', 'poids', 'decharge', 'valeur', 'cote_ref', 'cote_actuelle']
        
        for feature in features:
            if feature not in df.columns:
                if feature == 'decharge':
                    df[feature] = 0.0
                elif feature == 'valeur':
                    df[feature] = 30.0
                elif feature == 'cote_ref':
                    df[feature] = 10.0
                elif feature == 'cote_actuelle':
                    df[feature] = df.get('cote_ref', 10.0)
                else:
                    df[feature] = 0
        
        # Nettoyage des données
        df = df.fillna({
            'poids': 60.0,
            'decharge': 0.0,
            'age': 5,
            'valeur': 30.0,
            'cote_ref': 10.0,
            'cote_actuelle': 10.0
        })
        
        # Features calculées
        df['poids_net'] = df['poids'] - df['decharge']
        df['rapport_poids_age'] = df['poids_net'] / df['age']
        df['evolution_cote'] = df['cote_actuelle'] - df['cote_ref']
        df['favorabilite'] = 1 / df['cote_actuelle']
        
        # Features de musique (si disponible)
        if 'musique' in df.columns:
            df['nb_victoires_5_courses'] = df['musique'].apply(self._compter_victoires)
            df['nb_places_5_courses'] = df['musique'].apply(self._compter_places)
            df['regularite'] = df['musique'].apply(self._calculer_regularite)
        else:
            df['nb_victoires_5_courses'] = 0
            df['nb_places_5_courses'] = 0
            df['regularite'] = 0.5
        
        # Encodage des variables catégorielles
        categorical_features = ['sexe', 'jockey', 'entraineur']
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    try:
                        df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature].astype(str))
                    except:
                        df[f'{feature}_encoded'] = 0
        
        # Features statistiques
        df['poids_rank'] = df['poids_net'].rank(ascending=False)
        df['valeur_rank'] = df['valeur'].rank(ascending=False)
        df['cote_rank'] = df['cote_actuelle'].rank(ascending=True)
        
        # Normalisation des rangs
        n_partants = len(df)
        df['poids_rank_norm'] = df['poids_rank'] / n_partants
        df['valeur_rank_norm'] = df['valeur_rank'] / n_partants
        df['cote_rank_norm'] = df['cote_rank'] / n_partants
        
        # Features combinées
        df['score_composite'] = (
            df['favorabilite'] * 0.3 +
            (1 - df['cote_rank_norm']) * 0.2 +
            df['valeur_rank_norm'] * 0.15 +
            df['regularite'] * 0.15 +
            (1 - df['poids_rank_norm']) * 0.1 +
            df['nb_victoires_5_courses'] * 0.1
        )
        
        # Sélection des features finales
        self.features_names = [
            'numero', 'corde', 'age', 'poids_net', 'valeur', 'cote_actuelle',
            'favorabilite', 'evolution_cote', 'nb_victoires_5_courses', 
            'nb_places_5_courses', 'regularite', 'poids_rank_norm', 
            'valeur_rank_norm', 'cote_rank_norm', 'score_composite'
        ]
        
        # Ajout des features encodées si disponibles
        for feature in categorical_features:
            if f'{feature}_encoded' in df.columns:
                self.features_names.append(f'{feature}_encoded')
        
        print(f"✅ {len(self.features_names)} features créées automatiquement!")
        
        return df
    
    def _compter_victoires(self, musique):
        """Compte les victoires dans la musique"""
        if pd.isna(musique) or not isinstance(musique, str):
            return 0
        return musique.count('1')
    
    def _compter_places(self, musique):
        """Compte les places (1, 2, 3) dans la musique"""
        if pd.isna(musique) or not isinstance(musique, str):
            return 0
        return sum(musique.count(str(i)) for i in [1, 2, 3])
    
    def _calculer_regularite(self, musique):
        """Calcule un score de régularité"""
        if pd.isna(musique) or not isinstance(musique, str):
            return 0.5
        
        # Supprime les caractères non numériques
        resultats = re.findall(r'\d+', musique)
        if not resultats:
            return 0.5
        
        # Calcule la régularité (inverse de l'écart-type normalisé)
        positions = [int(r) for r in resultats[:5]]  # 5 dernières courses max
        if len(positions) < 2:
            return 0.5
        
        try:
            regularite = 1 / (1 + np.std(positions) / np.mean(positions))
            return min(regularite, 1.0)
        except:
            return 0.5
    
    def generer_target_ml(self, df):
        """Génère le target pour l'entraînement ML"""
        print("🎯 Génération du target ML...")
        
        # Simulation d'un target basé sur les features
        # Dans un vrai système, ce serait les résultats historiques
        
        # Score de performance basé sur plusieurs critères
        score = (
            df['score_composite'] * 0.4 +
            (1 / df['cote_actuelle']) * 0.3 +
            df['regularite'] * 0.2 +
            (df['nb_victoires_5_courses'] / 5) * 0.1
        )
        
        # Normalisation
        score = (score - score.min()) / (score.max() - score.min())
        
        # Création des classes (top 3 = 1, autres = 0)
        target = np.zeros(len(df))
        top_indices = score.nlargest(3).index
        target[top_indices] = 1
        
        return target
    
    def entrainer_modeles(self, X, y):
        """Entraîne plusieurs modèles ML"""
        print("🚀 Entraînement des modèles...")
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Définition des modèles
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
        }
        
        # Entraînement et évaluation
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            try:
                # Cross-validation
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                
                # Entraînement complet
                model.fit(X_scaled, y)
                self.models[name] = model
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    
                print(f"✅ {name}: Score CV = {mean_score:.3f}")
                
            except Exception as e:
                print(f"❌ Erreur {name}: {e}")
        
        # Création d'un modèle ensemble
        try:
            ensemble_models = [(name, model) for name, model in self.models.items()]
            if len(ensemble_models) >= 2:
                ensemble = VotingClassifier(
                    estimators=ensemble_models,
                    voting='soft'
                )
                ensemble.fit(X_scaled, y)
                self.models['ENSEMBLE'] = ensemble
                self.best_model = ensemble
                best_model_name = 'ENSEMBLE'
                print("✅ Modèle ENSEMBLE créé")
        except Exception as e:
            print(f"❌ Erreur ENSEMBLE: {e}")
            if best_model_name:
                self.best_model = self.models[best_model_name]
        
        print(f"✅ Système IA entraîné avec succès!")
        print(f"🏆 Meilleur modèle: {best_model_name}")
        
        return {
            'best_model': best_model_name,
            'nb_models': len(self.models),
            'best_score': best_score
        }
    
    def predire_classement(self, df):
        """Prédit le classement final"""
        if not self.best_model:
            print("❌ Aucun modèle entraîné")
            return df
        
        try:
            # Préparation des features
            X = df[self.features_names].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Prédictions
            if hasattr(self.best_model, 'predict_proba'):
                probas = self.best_model.predict_proba(X_scaled)
                if probas.shape[1] > 1:
                    df['proba_victoire'] = probas[:, 1]
                else:
                    df['proba_victoire'] = probas[:, 0]
            else:
                df['proba_victoire'] = self.best_model.decision_function(X_scaled)
            
            # Classement IA
            df['rang_ia'] = df['proba_victoire'].rank(ascending=False, method='first')
            
            # Score final (0-100)
            df['score_ia'] = (df['proba_victoire'] * 100).round(1)
            
            return df.sort_values('rang_ia')
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            # Fallback sur score composite
            df['rang_ia'] = df['score_composite'].rank(ascending=False, method='first')
            df['score_ia'] = (df['score_composite'] * 100).round(1)
            return df.sort_values('rang_ia')
    
    def analyser_course(self, url):
        """Analyse complète d'une course"""
        print("🏇 Analyseur Hippique IA Pro Max")
        print("=" * 50)
        
        # 1. Extraction des données
        print(f"🔍 Analyse de l'URL: {url}")
        donnees = self.extraire_donnees_geny(url)
        
        if not donnees:
            return None
        
        # 2. Création des features
        df = self.creer_features(donnees)
        
        # 3. Analyse statistique
        print("\n📊 Analyse Statistique:")
        print(f"🏇 Nombre de chevaux: {len(df)}")
        print(f"⚖️ Poids moyen: {df['poids'].mean():.1f} kg")
        print(f"💪 Écart-type poids: {df['poids'].std():.1f} kg")
        
        # 4. Entraînement ML
        y = self.generer_target_ml(df)
        X = df[self.features_names].fillna(0)
        
        rapport_ml = self.entrainer_modeles(X, y)
        
        # 5. Prédictions
        df_final = self.predire_classement(df)
        
        # 6. Affichage des résultats
        self.afficher_resultats(df_final, donnees['course_info'], rapport_ml)
        
        return df_final
    
    def afficher_resultats(self, df, course_info, rapport_ml):
        """Affiche les résultats de l'analyse"""
        print("\n" + "=" * 50)
        print("🏆 CLASSEMENT FINAL IA")
        print("=" * 50)
        
        # Informations de course
        if course_info:
            print(f"📍 Course: {course_info.get('titre', 'N/A')}")
            print(f"🏁 Distance: {course_info.get('distance', 'N/A')}m")
            print(f"🏇 Type: {course_info.get('type', 'PLAT')}")
            print(f"🌱 Terrain: {course_info.get('terrain', 'N/A')}")
        
        print(f"\n🤖 Rapport ML:")
        print(f"🏆 Meilleur Modèle: {rapport_ml['best_model']}")
        print(f"🔧 Modèles Entraînés: {rapport_ml['nb_models']}")
        
        print(f"\n{'Rang':<4} {'N°':<3} {'Cheval':<20} {'Score':<6} {'Cote':<6} {'Jockey':<15}")
        print("-" * 65)
        
        for _, row in df.head(10).iterrows():
            print(f"{int(row['rang_ia']):<4} "
                  f"{int(row['numero']):<3} "
                  f"{str(row['cheval'])[:19]:<20} "
                  f"{row['score_ia']:<6.1f} "
                  f"{row['cote_actuelle']:<6.1f} "
                  f"{str(row.get('jockey', 'N/A'))[:14]:<15}")
        
        print("\n🎯 PRONOSTICS RECOMMANDÉS:")
        top3 = df.head(3)
        print(f"🥇 BASE: {top3.iloc[0]['cheval']} (N°{int(top3.iloc[0]['numero'])})")
        print(f"🥈 COMPLEMENT: {top3.iloc[1]['cheval']} (N°{int(top3.iloc[1]['numero'])})")
        print(f"🥉 OUTSIDER: {top3.iloc[2]['cheval']} (N°{int(top3.iloc[2]['numero'])})")
        
        # Combinaisons
        nums = [int(row['numero']) for _, row in top3.iterrows()]
        print(f"\n💰 COMBINAISONS:")
        print(f"🎯 Couplé: {nums[0]}-{nums[1]}")
        print(f"🎯 Trio: {nums[0]}-{nums[1]}-{nums[2]}")

def main():
    """Fonction principale"""
    if len(sys.argv) < 2:
        print("Usage: python analyseur_hippique.py <URL_GENY>")
        print("\nExemple:")
        print("python analyseur_hippique.py https://www.geny.com/partants-pmu/2025-10-24-lyon-la-soie-pmu-prix-de-confluence_c1610417")
        return
    
    url = sys.argv[1]
    
    # Vérification de l'URL
    if 'geny.com' not in url:
        print("❌ Veuillez fournir une URL valide de Geny.com")
        return
    
    # Analyse
    analyseur = AnalyseurHippique()
    resultat = analyseur.analyser_course(url)
    
    if resultat is not None:
        print("\n✅ Analyse terminée avec succès!")
        
        # Sauvegarde optionnelle
        try:
            filename = f"analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            resultat.to_csv(filename, index=False, encoding='utf-8')
            print(f"💾 Résultats sauvegardés dans: {filename}")
        except Exception as e:
            print(f"⚠️ Impossible de sauvegarder: {e}")
    else:
        print("❌ Échec de l'analyse")

if __name__ == "__main__":
    main()
