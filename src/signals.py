"""
Signals Module per Tesi Magistrale - Trend Following Strategies
Calcola segnali Moving Average Crossover e Breakout Rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class TrendFollowingSignals:
    """
    Calcola segnali trend-following per strategie sistematiche
    Implementa MA crossover, breakout rules e combinazioni
    """
    
    def __init__(self, data_path: str = "data/processed"):
        """
        Inizializza il generatore di segnali
        
        Args:
            data_path: Percorso dati processati CSV
        """
        self.data_path = data_path
        self.asset_data = {}
        self.signals_data = {}
        
        print(f"Signals Module inizializzato")
        print(f"Data path: {self.data_path}")
    
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Carica i dati processati dal data processor
        
        Returns:
            Dict con asset_name: DataFrame
        """
        print("Caricamento dati processati...")
        
        if not os.path.exists(self.data_path):
            print(f"Errore: cartella {self.data_path} non trovata")
            return {}
        
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('_processed.csv')]
        
        if not csv_files:
            print("Nessun file processato trovato")
            return {}
        
        loaded_data = {}
        
        for file_name in csv_files:
            asset_name = file_name.replace('_processed.csv', '').upper()
            file_path = os.path.join(self.data_path, file_name)
            
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                loaded_data[asset_name] = df
                print(f"Caricato {asset_name}: {len(df)} osservazioni")
            except Exception as e:
                print(f"Errore caricamento {asset_name}: {e}")
        
        self.asset_data = loaded_data
        print(f"Caricati {len(loaded_data)} asset")
        return loaded_data
    
    def calculate_moving_average_signals(self, 
                                       short_window: int = 20,
                                       long_window: int = 60) -> None:
        """
        Calcola segnali Moving Average Crossover
        
        Args:
            short_window: Finestra MA breve (default 20)
            long_window: Finestra MA lunga (default 60)
        """
        print(f"Calcolo segnali MA ({short_window} vs {long_window})...")
        
        for asset_name, df in self.asset_data.items():
            if asset_name not in self.signals_data:
                self.signals_data[asset_name] = pd.DataFrame(index=df.index)
            
            signals_df = self.signals_data[asset_name]
            
            # Calcola Moving Averages
            signals_df[f'MA_{short_window}'] = df['Close'].rolling(short_window).mean()
            signals_df[f'MA_{long_window}'] = df['Close'].rolling(long_window).mean()
            
            # Segnale MA: +1 quando MA_short > MA_long, -1 altrimenti
            ma_signal = np.where(
                signals_df[f'MA_{short_window}'] > signals_df[f'MA_{long_window}'], 
                1, -1
            )
            
            signals_df[f'Signal_MA_{short_window}_{long_window}'] = ma_signal
            
            # Calcola strength del trend (distanza tra MA)
            ma_spread = ((signals_df[f'MA_{short_window}'] - signals_df[f'MA_{long_window}']) / 
                        signals_df[f'MA_{long_window}'])
            signals_df[f'MA_Spread_{short_window}_{long_window}'] = ma_spread
            
            self.signals_data[asset_name] = signals_df
        
        print("Segnali MA calcolati")
    
    def calculate_breakout_signals(self, lookback_window: int = 20) -> None:
        """
        Calcola segnali Breakout Rules
        
        Args:
            lookback_window: Finestra per max/min (default 20)
        """
        print(f"Calcolo segnali Breakout ({lookback_window} giorni)...")
        
        for asset_name, df in self.asset_data.items():
            if asset_name not in self.signals_data:
                self.signals_data[asset_name] = pd.DataFrame(index=df.index)
            
            signals_df = self.signals_data[asset_name]
            
            # Calcola rolling max e min
            signals_df[f'High_Max_{lookback_window}'] = df['High'].rolling(lookback_window).max()
            signals_df[f'Low_Min_{lookback_window}'] = df['Low'].rolling(lookback_window).min()
            
            # Segnali breakout
            breakout_signal = np.zeros(len(df))
            
            # Long quando prezzo supera massimo del periodo
            long_breakout = df['Close'] > signals_df[f'High_Max_{lookback_window}'].shift(1)
            
            # Short quando prezzo va sotto minimo del periodo  
            short_breakout = df['Close'] < signals_df[f'Low_Min_{lookback_window}'].shift(1)
            
            breakout_signal[long_breakout] = 1
            breakout_signal[short_breakout] = -1
            
            # Forward fill per mantenere segnale fino a nuovo breakout
            breakout_signal = pd.Series(breakout_signal, index=df.index)
            breakout_signal = breakout_signal.replace(0, np.nan).fillna(method='ffill')
            breakout_signal = breakout_signal.fillna(0)  # Primi valori
            
            signals_df[f'Signal_Breakout_{lookback_window}'] = breakout_signal
            
            # Calcola momentum (ROC)
            roc = df['Close'].pct_change(lookback_window)
            signals_df[f'ROC_{lookback_window}'] = roc
            
            self.signals_data[asset_name] = signals_df
        
        print("Segnali Breakout calcolati")
    
    def calculate_multiple_timeframes(self,
                                    ma_pairs: List[Tuple[int, int]] = None,
                                    breakout_windows: List[int] = None) -> None:
        """
        Calcola segnali su multipli timeframes
        
        Args:
            ma_pairs: Liste di tuple (short, long) per MA
            breakout_windows: Liste di finestre per breakout
        """
        if ma_pairs is None:
            ma_pairs = [(20, 60), (50, 120), (100, 200)]
        
        if breakout_windows is None:
            breakout_windows = [20, 60, 120, 250]
        
        print("Calcolo segnali multi-timeframe...")
        
        # Calcola tutti i MA crossover
        for short, long in ma_pairs:
            self.calculate_moving_average_signals(short, long)
        
        # Calcola tutti i breakout
        for window in breakout_windows:
            self.calculate_breakout_signals(window)
        
        print(f"Calcolati {len(ma_pairs)} MA signals e {len(breakout_windows)} Breakout signals")
    
    def create_composite_signals(self) -> None:
        """
        Crea segnali compositi combinando diversi timeframes
        """
        print("Creazione segnali compositi...")
        
        for asset_name, signals_df in self.signals_data.items():
            # Trova tutte le colonne di segnali
            ma_signals = [col for col in signals_df.columns if col.startswith('Signal_MA_')]
            breakout_signals = [col for col in signals_df.columns if col.startswith('Signal_Breakout_')]
            
            if ma_signals:
                # Segnale MA medio
                ma_composite = signals_df[ma_signals].mean(axis=1)
                signals_df['Signal_MA_Composite'] = np.sign(ma_composite)
                signals_df['Signal_MA_Strength'] = abs(ma_composite)
            
            if breakout_signals:
                # Segnale Breakout medio
                breakout_composite = signals_df[breakout_signals].mean(axis=1)
                signals_df['Signal_Breakout_Composite'] = np.sign(breakout_composite)
                signals_df['Signal_Breakout_Strength'] = abs(breakout_composite)
            
            # Segnale finale combinato (MA + Breakout)
            if ma_signals and breakout_signals:
                combined_signal = (signals_df['Signal_MA_Composite'] + 
                                 signals_df['Signal_Breakout_Composite']) / 2
                signals_df['Signal_Combined'] = np.sign(combined_signal)
                signals_df['Signal_Combined_Strength'] = abs(combined_signal)
            
            elif ma_signals:
                signals_df['Signal_Combined'] = signals_df['Signal_MA_Composite']
                signals_df['Signal_Combined_Strength'] = signals_df['Signal_MA_Strength']
            
            elif breakout_signals:
                signals_df['Signal_Combined'] = signals_df['Signal_Breakout_Composite']
                signals_df['Signal_Combined_Strength'] = signals_df['Signal_Breakout_Strength']
            
            self.signals_data[asset_name] = signals_df
        
        print("Segnali compositi creati")
    
    def calculate_signal_quality_metrics(self) -> pd.DataFrame:
        """
        Calcola metriche di qualità dei segnali
        
        Returns:
            DataFrame con statistiche per asset
        """
        print("Calcolo metriche qualità segnali...")
        
        metrics_data = []
        
        for asset_name, signals_df in self.signals_data.items():
            if 'Signal_Combined' not in signals_df.columns:
                continue
            
            df = self.asset_data[asset_name]
            signal = signals_df['Signal_Combined']
            returns = df['Returns']
            
            # Calcola rendimenti della strategia
            strategy_returns = signal.shift(1) * returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0:
                continue
            
            # Metriche base
            total_return = (1 + strategy_returns).prod() - 1
            ann_return = (1 + strategy_returns).prod() ** (252/len(strategy_returns)) - 1
            ann_vol = strategy_returns.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Hit rate (% di trade vincenti)
            hit_rate = (strategy_returns > 0).mean()
            
            # Numero cambi di segnale (turnover)
            signal_changes = (signal != signal.shift(1)).sum()
            turnover = signal_changes / len(signal)
            
            # Maximum Drawdown
            equity = (1 + strategy_returns).cumprod()
            drawdown = equity / equity.cummax() - 1
            max_dd = drawdown.min()
            
            metrics = {
                'Asset': asset_name,
                'Total_Return': total_return,
                'Annualized_Return': ann_return,
                'Volatility': ann_vol,
                'Sharpe_Ratio': sharpe,
                'Hit_Rate': hit_rate,
                'Max_Drawdown': max_dd,
                'Turnover': turnover,
                'Signal_Changes': signal_changes,
                'Observations': len(strategy_returns)
            }
            
            metrics_data.append(metrics)
        
        return pd.DataFrame(metrics_data)
    
    def save_signals(self, output_path: str = "data/signals") -> None:
        """
        Salva segnali calcolati
        
        Args:
            output_path: Percorso per salvare i segnali
        """
        os.makedirs(output_path, exist_ok=True)
        print(f"Salvataggio segnali in {output_path}...")
        
        for asset_name, signals_df in self.signals_data.items():
            file_path = os.path.join(output_path, f"{asset_name.lower()}_signals.csv")
            signals_df.to_csv(file_path, date_format='%Y-%m-%d')
            print(f"Salvato {asset_name} -> {file_path}")
        
        print("Salvataggio segnali completato")
    
    def get_signal_summary(self, asset_name: str) -> None:
        """
        Mostra summary dei segnali per un asset
        
        Args:
            asset_name: Nome dell'asset da analizzare
        """
        if asset_name not in self.signals_data:
            print(f"Asset {asset_name} non trovato")
            return
        
        signals_df = self.signals_data[asset_name]
        
        print(f"\n=== SUMMARY SEGNALI {asset_name} ===")
        
        # Conta segnali per tipo
        if 'Signal_Combined' in signals_df.columns:
            signal = signals_df['Signal_Combined']
            long_periods = (signal == 1).sum()
            short_periods = (signal == -1).sum()
            neutral_periods = (signal == 0).sum()
            
            print(f"Periodi Long: {long_periods} ({long_periods/len(signal)*100:.1f}%)")
            print(f"Periodi Short: {short_periods} ({short_periods/len(signal)*100:.1f}%)")
            print(f"Periodi Neutral: {neutral_periods} ({neutral_periods/len(signal)*100:.1f}%)")
            
            # Cambi di segnale
            signal_changes = (signal != signal.shift(1)).sum()
            print(f"Cambi di segnale: {signal_changes}")
            print(f"Turnover medio: {signal_changes/len(signal)*100:.2f}%")
        
        # Mostra prime e ultime osservazioni
        print(f"\nPrime 5 osservazioni:")
        cols_to_show = [col for col in signals_df.columns if 'Signal' in col][:3]
        if cols_to_show:
            print(signals_df[cols_to_show].head())
        
        print(f"\nUltime 5 osservazioni:")
        if cols_to_show:
            print(signals_df[cols_to_show].tail())


def main():
    """
    Funzione principale per testare il modulo segnali
    """
    print("=== SIGNALS MODULE TEST ===")
    
    # Inizializza generatore segnali
    signal_generator = TrendFollowingSignals()
    
    # Carica dati processati
    data = signal_generator.load_processed_data()
    
    if not data:
        print("Nessun dato trovato. Esegui prima data_processor.py")
        return
    
    # Calcola segnali multi-timeframe
    signal_generator.calculate_multiple_timeframes()
    
    # Crea segnali compositi
    signal_generator.create_composite_signals()
    
    # Calcola metriche qualità
    quality_metrics = signal_generator.calculate_signal_quality_metrics()
    
    if not quality_metrics.empty:
        print("\nMETRICHE QUALITA SEGNALI:")
        print(quality_metrics.round(4).to_string(index=False))
    
    # Salva segnali
    signal_generator.save_signals()
    
    # Mostra summary per alcuni asset
    test_assets = ['SP500', 'GOLD', 'BTC']
    for asset in test_assets:
        if asset in signal_generator.signals_data:
            signal_generator.get_signal_summary(asset)
    
    print(f"\nProcessing segnali completato per {len(data)} asset")


if __name__ == "__main__":
    main()