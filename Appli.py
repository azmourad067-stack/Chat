def _scrape_partants_page(self, soup):
        """Scrape une page 'partants-pmu' Geny"""
        st.info("📄 Détection: Page Partants PMU")
        
        horses_data = []
        
        # Méthode 1: Chercher le tableau principal
        table = soup.find('table') or soup.find('tbody')
        
        if table:
            st.info("✅ Tableau détecté, extraction en cours...")
            rows = table.find_all('tr')
            
            for row in rows:
                # Chercher les cellules
                cells = row.find_all(['td', 'th'])
                if len(cells) < 3:  # Ignorer les lignes d'en-tête
                    continue
                
                horse_data = self._extract_from_table_row(row, cells)
                if horse_data and horse_data.get('Nom'):
                    horses_data.append(horse_data)
                    st.success(f"✓ Cheval extrait: {horse_data['Nom']}")
        
        # Méthode 2: Fallback vers divs/spans
        if not horses_data:
            st.info("📦 Tentative extraction via divs...")
            horse_elements = soup.find_all('tr') or soup.find_all('div', recursive=True)
            
            for element in horse_elements[:20]:
                horse_data = self._extract_horse_data_from_element(element)
                if horse_data and horse_data['Nom'] != "CHEVAL INCONNU":
                    horses_data.append(horse_data)
        
        if horses_data:
            st.success(f"🎯 {len(horses_data)} chevaux extraits avec succès")
            return pd.DataFrame(horses_data)
        else:
            st.warning("⚠️ Extraction standard échouée, passage en mode analyse générique")
            return self._scrape_generic_page(soup)
