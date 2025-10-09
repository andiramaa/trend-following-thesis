"""
Data Processor per Tesi Magistrale - Trend Following Strategies
Modulo per processare file Excel Refinitiv dei futures
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RefinitivDataProcessor:
    """
    Processore per dati futures Refinitiv in formato Excel
    """
    
    def __init__(self, raw_data_path: str = "data/raw", 
                 processed_data_path: str = "data/processed"):
        self.raw_path = raw_data_path
        self.processed_path = processed_data_path
        self.asset_data = {}
        
        os.makedirs(self.processed_path, exist_ok=True)
        
        print(f"Data Processor inizializzato")
        print(f"Path dati raw: {self.raw_path}")
        print(f"Path dati processati: {self.processed_path}")
    
    def load_refinitiv_excel(self, file_path: str, asset_name: str) -> Optional[pd.DataFrame]:
        """
        Carica singolo file Excel Refinitiv - versione adattiva
        """
        try:
            print(f"Caricamento {asset_name} da {file_path}...")
            
            # Prima leggiamo il file per capire la struttura
            df_raw = pd.read_excel(file_path, skiprows=28, header=None)
            
            # Trova la prima riga con dati numerici (date)
            date_row_idx = 0
            for i in range(min(10, len(df_raw))):
                first_cell = df_raw.iloc[i, 0]
                if pd.notna(first_cell):
                    try:
                        pd.to_datetime(first_cell)
                        date_row_idx = i
                        break
                    except:
                        continue
            
            # Rileggi dal punto giusto
            df_raw = pd.read_excel(file_path, skiprows=28+date_row_idx)
            
            print(f"  File ha {len(df_raw.columns)} colonne")
            
            # Identifica colonne per posizione (più robusto)
            col_mapping = {}
            
            # Prima colonna dovrebbe essere la data
            if len(df_raw.columns) >= 1:
                col_mapping['Exchange Date'] = df_raw.columns[0]
            
            # Le successive dovrebbero essere prezzi
            price_cols = ['Close', 'Net', 'Pct_Change', 'Open', 'Low', 'High', 'Volume', 'OI']
            for i, col_name in enumerate(price_cols):
                if i + 1 < len(df_raw.columns):
                    col_mapping[col_name] = df_raw.columns[i + 1]
            
            # Mantieni solo le colonne che ci servono
            required_cols = ['Exchange Date', 'Open', 'High', 'Low', 'Close']
            available_mapping = {k: v for k, v in col_mapping.items() if k in required_cols}
            
            if len(available_mapping) < 3:
                print(f"Errore: {asset_name} non ha abbastanza colonne valide")
                return None
            
            # Crea DataFrame con solo le colonne necessarie
            df = df_raw[[col_mapping[k] for k in available_mapping.keys()]].copy()
            df.columns = list(available_mapping.keys())
            
            # Se manca High o Low, usa Close come proxy
            if 'High' not in df.columns:
                df['High'] = df['Close']
            if 'Low' not in df.columns:
                df['Low'] = df['Close']
            if 'Open' not in df.columns:
                df['Open'] = df['Close']
            
            # Converti data
            df['Exchange Date'] = pd.to_datetime(df['Exchange Date'], errors='coerce')
            df = df.dropna(subset=['Exchange Date'])
            df.set_index('Exchange Date', inplace=True)
            
            # Converti prezzi in numerico
            price_columns = [col for col in df.columns if col != 'Exchange Date']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rimuovi righe senza prezzi
            df = df.dropna(subset=['Close'])
            df = df.sort_index()
            
            # Rimuovi righe con prezzi zero o negativi
            df = df[df['Close'] > 0]
            
            if len(df) > 100:
                print(f"Successo {asset_name}: {len(df)} osservazioni dal {df.index[0].date()} al {df.index[-1].date()}")
                return df
            else:
                print(f"Troppo poche osservazioni per {asset_name}")
                return None
                
        except Exception as e:
            print(f"Errore caricamento {asset_name}: {str(e)}")
            return None
    
    def load_all_assets(self) -> Dict[str, pd.DataFrame]:
        """
        Carica tutti i file Excel dalla cartella raw
        """
        print("Inizio caricamento tutti gli asset...")
        
        # Mappa nomi file -> nomi asset standard
        file_mapping = {
            'sp500': 'SP500',
            'nasdaq': 'NASDAQ', 
            'eurostoxx': 'EUROSTOXX',
            'cac40': 'CAC40',
            'dax': 'DAX',
            'ftse100': 'FTSE100',
            'hangseng': 'HANGSENG',
            'hsi': 'HANGSENG',
            'omx': 'OMX',
            'ust10y': 'UST10Y',
            'gold': 'GOLD',
            'crudeoil': 'CRUDE',
            'crude': 'CRUDE',
            'btc_usd': 'BTC',
            'btc': 'BTC'
        }
        
        loaded_assets = {}
        
        # Cerca file Excel nella cartella raw
        if not os.path.exists(self.raw_path):
            print(f"Errore: cartella {self.raw_path} non trovata")
            return loaded_assets
        
        excel_files = [f for f in os.listdir(self.raw_path) if f.endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print("Nessun file Excel trovato nella cartella raw")
            return loaded_assets
        
        print(f"Trovati {len(excel_files)} file Excel")
        
        # Carica ogni file
        for file_name in excel_files:
            file_path = os.path.join(self.raw_path, file_name)
            
            # Identifica asset dal nome file
            asset_name = None
            file_lower = file_name.lower().replace('.xlsx', '').replace('.xls', '')
            
            for key, value in file_mapping.items():
                if key in file_lower:
                    asset_name = value
                    break
            
            if asset_name is None:
                print(f"Nome asset non riconosciuto per file: {file_name}")
                continue
            
            # Carica il file
            df = self.load_refinitiv_excel(file_path, asset_name)
            
            if df is not None:
                loaded_assets[asset_name] = df
        
        print(f"Caricati {len(loaded_assets)} asset con successo")
        self.asset_data = loaded_assets
        return loaded_assets
    
    def clean_and_validate(self) -> None:
        """
        Pulizia e validazione dati
        """
        print("Pulizia e validazione dati...")
        
        for asset_name, df in self.asset_data.items():
            print(f"Pulizia {asset_name}...")
            
            # Calcola rendimenti per identificare outliers
            df['Returns'] = df['Close'].pct_change()
            
            # Identifica outliers estremi (>10 deviazioni standard)
            returns_std = df['Returns'].std()
            outlier_threshold = 10 * returns_std
            
            outliers = abs(df['Returns']) > outlier_threshold
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                print(f"  Trovati {n_outliers} outliers estremi in {asset_name}")
                df.loc[outliers, ['Open', 'High', 'Low', 'Close']] = np.nan
            
            # Forward fill per gap singoli
            df = df.fillna(method='ffill', limit=5)
            
            # Rimuovi righe ancora con NaN
            initial_length = len(df)
            df = df.dropna(subset=['Close'])
            final_length = len(df)
            
            if initial_length != final_length:
                print(f"  Rimosse {initial_length - final_length} righe con dati mancanti")
            
            # Verifica consistenza OHLC
            inconsistent = (df['High'] < df['Low']) | (df['Close'] < df['Low']) | (df['Close'] > df['High'])
            if inconsistent.any():
                print(f"  Trovate {inconsistent.sum()} inconsistenze OHLC in {asset_name}")
                df.loc[inconsistent, 'High'] = df.loc[inconsistent, ['High', 'Close']].max(axis=1)
                df.loc[inconsistent, 'Low'] = df.loc[inconsistent, ['Low', 'Close']].min(axis=1)
            
            self.asset_data[asset_name] = df
        
        print("Pulizia completata")
    
    def save_processed_data(self) -> None:
        """
        Salva dati processati come CSV
        """
        print("Salvataggio dati processati...")
        
        for asset_name, df in self.asset_data.items():
            file_path = os.path.join(self.processed_path, f"{asset_name.lower()}_processed.csv")
            df.to_csv(file_path, date_format='%Y-%m-%d')
            print(f"Salvato {asset_name} -> {file_path}")
        
        print("Salvataggio completato")
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Crea summary statistico dei dati caricati
        """
        if not self.asset_data:
            print("Nessun dato caricato")
            return pd.DataFrame()
        
        summary_data = []
        
        for asset_name, df in self.asset_data.items():
            summary = {
                'Asset': asset_name,
                'Start_Date': df.index[0].date(),
                'End_Date': df.index[-1].date(),
                'Observations': len(df),
                'Missing_Days': df['Close'].isna().sum(),
                'Avg_Price': df['Close'].mean(),
                'Min_Price': df['Close'].min(),
                'Max_Price': df['Close'].max(),
                'Volatility_Daily': df['Returns'].std(),
                'Volatility_Annual': df['Returns'].std() * np.sqrt(252)
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)


def main():
    """
    Funzione principale per testare il processore
    """
    print("=== DATA PROCESSOR TEST ===")
    
    # Inizializza processore
    processor = RefinitivDataProcessor()
    
    # Carica tutti gli asset
    data = processor.load_all_assets()
    
    if data:
        # Pulizia dati
        processor.clean_and_validate()
        
        # Summary
        summary = processor.get_data_summary()
        print("\nSUMMARY DATI:")
        print(summary.to_string(index=False))
        
        # Salva dati processati
        processor.save_processed_data()
        
        print(f"\nProcessing completato per {len(data)} asset")
    else:
        print("Nessun dato caricato. Verifica i file nella cartella data/raw/")


if __name__ == "__main__":
    main()