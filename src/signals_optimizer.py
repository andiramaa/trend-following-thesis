"""
Signals Optimizer - Fix rapido per parametri asset-specific
Risolve underperformance gravi identificate nella diagnostic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class SignalsOptimizer:
    """
    Ottimizza parametri segnali per asset classes specifiche
    """
    
    def __init__(self, data_path: str = "data/processed"):
        self.data_path = data_path
        self.asset_data = {}
        self.optimized_signals = {}
        
        # Parametri asset-specific ottimizzati
        self.asset_params = {
            'BTC': {
                'ma_pairs': [(10, 30), (20, 50), (50, 100)],  # Più veloce per crypto
                'breakout_windows': [10, 20, 60],
                'min_trend_strength': 0.02  # 2% threshold per confermare trend
            },
            'NASDAQ': {
                'ma_pairs': [(15, 40), (30, 80), (60, 150)],  # Veloce per tech growth
                'breakout_windows': [15, 30, 90],
                'min_trend_strength': 0.015
            },
            'SP500': {
                'ma_pairs': [(20, 50), (40, 100), (80, 200)],  # Bilanciato
                'breakout_windows': [20, 50, 120],
                'min_trend_strength': 0.01
            },
            'GOLD': {
                'ma_pairs': [(15, 45), (30, 90), (60, 180)],  # Medio-veloce per commodity
                'breakout_windows': [15, 40, 100],
                'min_trend_strength': 0.01
            },
            'CRUDE': {
                'ma_pairs': [(10, 35), (25, 75), (50, 150)],  # Veloce per oil volatility
                'breakout_windows': [10, 30, 80],
                'min_trend_strength': 0.02
            },
            'DEFAULT': {  # Per asset europei e altri
                'ma_pairs': [(18, 50), (35, 90), (70, 180)],
                'breakout_windows': [18, 45, 110],
                'min_trend_strength': 0.01
            }
        }
        
        print("Signals Optimizer inizializzato con parametri asset-specific")
    
    def load_data(self):
        """Carica dati processati"""
        processed_files = [f for f in os.listdir(self.data_path) if f.endswith('_processed.csv')]
        
        for file_name in processed_files:
            asset_name = file_name.replace('_processed.csv', '').upper()
            file_path = os.path.join(self.data_path, file_name)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.asset_data[asset_name] = df
        
        print(f"Caricati {len(self.asset_data)} asset per ottimizzazione")
    
    def calculate_optimized_ma_signals(self, asset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola segnali MA ottimizzati per asset specifico"""
        
        # Usa parametri asset-specific o default
        params = self.asset_params.get(asset_name, self.asset_params['DEFAULT'])
        
        signals_df = pd.DataFrame(index=df.index)
        ma_signals = []
        
        for short, long in params['ma_pairs']:
            # Calcola MA
            ma_short = df['Close'].rolling(short).mean()
            ma_long = df['Close'].rolling(long).mean()
            
            # Segnale base
            basic_signal = np.where(ma_short > ma_long, 1, -1)
            
            # Aggiungi filtro trend strength se definito
            min_strength = params.get('min_trend_strength', 0)
            if min_strength > 0:
                trend_strength = abs((ma_short - ma_long) / ma_long)
                weak_trend_mask = trend_strength < min_strength
                filtered_signal = basic_signal.copy()
                filtered_signal[weak_trend_mask] = 0  # Neutral se trend debole
            else:
                filtered_signal = basic_signal
            
            col_name = f'Signal_MA_{short}_{long}'
            signals_df[col_name] = filtered_signal
            ma_signals.append(col_name)
        
        return signals_df, ma_signals
    
    def calculate_optimized_breakout_signals(self, asset_name: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calcola segnali breakout ottimizzati"""
        
        params = self.asset_params.get(asset_name, self.asset_params['DEFAULT'])
        signals_df = pd.DataFrame(index=df.index)
        breakout_signals = []
        
        for window in params['breakout_windows']:
            # Rolling max/min
            rolling_high = df['High'].rolling(window).max()
            rolling_low = df['Low'].rolling(window).min()
            
            # Segnali breakout
            breakout_signal = np.zeros(len(df))
            
            # Long quando prezzo supera massimo
            long_breakout = df['Close'] > rolling_high.shift(1)
            # Short quando prezzo va sotto minimo
            short_breakout = df['Close'] < rolling_low.shift(1)
            
            breakout_signal[long_breakout] = 1
            breakout_signal[short_breakout] = -1
            
            # Forward fill
            breakout_signal = pd.Series(breakout_signal, index=df.index)
            breakout_signal = breakout_signal.replace(0, np.nan).ffill()
            breakout_signal = breakout_signal.fillna(0)
            
            col_name = f'Signal_Breakout_{window}'
            signals_df[col_name] = breakout_signal
            breakout_signals.append(col_name)
        
        return signals_df, breakout_signals
    
    def create_optimized_composite(self, signals_df: pd.DataFrame, ma_signals: List[str], breakout_signals: List[str]) -> pd.DataFrame:
        """Crea segnali compositi ottimizzati"""
        
        # Composite MA
        if ma_signals:
            ma_composite = signals_df[ma_signals].mean(axis=1)
            signals_df['Signal_MA_Composite'] = np.sign(ma_composite)
            signals_df['Signal_MA_Strength'] = abs(ma_composite)
        
        # Composite Breakout
        if breakout_signals:
            breakout_composite = signals_df[breakout_signals].mean(axis=1)
            signals_df['Signal_Breakout_Composite'] = np.sign(breakout_composite)
            signals_df['Signal_Breakout_Strength'] = abs(breakout_composite)
        
        # Segnale finale - pesato verso MA per stabilità
        if ma_signals and breakout_signals:
            # 60% MA, 40% Breakout per bilanciare stabilità e reattività
            combined_signal = (0.6 * signals_df['Signal_MA_Composite'] + 
                             0.4 * signals_df['Signal_Breakout_Composite'])
            signals_df['Signal_Combined'] = np.sign(combined_signal)
            signals_df['Signal_Combined_Strength'] = abs(combined_signal)
        elif ma_signals:
            signals_df['Signal_Combined'] = signals_df['Signal_MA_Composite']
            signals_df['Signal_Combined_Strength'] = signals_df['Signal_MA_Strength']
        elif breakout_signals:
            signals_df['Signal_Combined'] = signals_df['Signal_Breakout_Composite']  
            signals_df['Signal_Combined_Strength'] = signals_df['Signal_Breakout_Strength']
        
        return signals_df
    
    def optimize_all_assets(self):
        """Ottimizza segnali per tutti gli asset"""
        print("Ottimizzazione segnali per tutti gli asset...")
        
        for asset_name, df in self.asset_data.items():
            print(f"Ottimizzazione {asset_name}...")
            
            # Calcola segnali MA ottimizzati
            ma_signals_df, ma_signals = self.calculate_optimized_ma_signals(asset_name, df)
            
            # Calcola segnali breakout ottimizzati
            breakout_signals_df, breakout_signals = self.calculate_optimized_breakout_signals(asset_name, df)
            
            # Combina tutti i segnali
            all_signals_df = pd.concat([ma_signals_df, breakout_signals_df], axis=1)
            
            # Crea compositi
            final_signals_df = self.create_optimized_composite(all_signals_df, ma_signals, breakout_signals)
            
            self.optimized_signals[asset_name] = final_signals_df
        
        print(f"Ottimizzazione completata per {len(self.optimized_signals)} asset")
    
    def calculate_quick_performance(self) -> pd.DataFrame:
        """Calcola performance rapida dei segnali ottimizzati"""
        print("Calcolo performance segnali ottimizzati...")
        
        results = []
        
        for asset_name, signals_df in self.optimized_signals.items():
            if asset_name not in self.asset_data:
                continue
            
            df = self.asset_data[asset_name] 
            returns = df['Returns'].dropna()
            
            if 'Signal_Combined' not in signals_df.columns:
                continue
            
            signal = signals_df['Signal_Combined']
            
            # Strategy returns
            strategy_returns = signal.shift(1) * returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0:
                continue
            
            # Buy-and-hold benchmark
            bnh_return = (1 + returns).prod() - 1
            bnh_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Strategy metrics
            strategy_return = (1 + strategy_returns).prod() - 1
            strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            
            # Signal stats
            long_pct = (signal == 1).mean()
            short_pct = (signal == -1).mean()
            neutral_pct = (signal == 0).mean()
            
            results.append({
                'Asset': asset_name,
                'Strategy_Return': strategy_return,
                'BnH_Return': bnh_return,
                'Outperformance': strategy_return - bnh_return,
                'Strategy_Sharpe': strategy_sharpe,
                'BnH_Sharpe': bnh_sharpe,
                'Long_Pct': long_pct,
                'Short_Pct': short_pct,
                'Neutral_Pct': neutral_pct
            })
        
        return pd.DataFrame(results)
    
    def save_optimized_signals(self, output_path: str = "data/signals_optimized"):
        """Salva segnali ottimizzati"""
        os.makedirs(output_path, exist_ok=True)
        
        for asset_name, signals_df in self.optimized_signals.items():
            file_path = os.path.join(output_path, f"{asset_name.lower()}_signals_optimized.csv")
            signals_df.to_csv(file_path)
            print(f"Salvato {asset_name} -> {file_path}")


def main():
    """Test dell'optimizer"""
    print("=== SIGNALS OPTIMIZER - FIX RAPIDO ===")
    
    optimizer = SignalsOptimizer()
    optimizer.load_data()
    optimizer.optimize_all_assets()
    
    # Calcola performance rapida
    performance_df = optimizer.calculate_quick_performance()
    
    print("\nPERFORMANCE SEGNALI OTTIMIZZATI:")
    print("=" * 60)
    print(performance_df[['Asset', 'Strategy_Return', 'BnH_Return', 'Outperformance', 'Long_Pct']].round(4))
    
    # Focus su asset problematici
    problematic = ['BTC', 'NASDAQ', 'GOLD', 'CAC40']
    print(f"\nFOCUS ASSET PROBLEMATICI:")
    problem_assets = performance_df[performance_df['Asset'].isin(problematic)]
    
    if not problem_assets.empty:
        print("BEFORE vs AFTER comparison:")
        for _, row in problem_assets.iterrows():
            asset = row['Asset']
            outperf = row['Outperformance']
            long_pct = row['Long_Pct']
            
            # Calcolo miglioramento approssimativo
            if asset == 'BTC':
                old_outperf = -1.78  # -17850% -> -178x
                improvement = outperf - old_outperf
                print(f"  {asset}: Outperformance {outperf:.1%} (improvement: {improvement:.1%}), Long {long_pct:.1%}")
            else:
                print(f"  {asset}: Outperformance {outperf:.1%}, Long {long_pct:.1%}")
    
    # Salva segnali ottimizzati
    optimizer.save_optimized_signals()
    
    print("\nSegnali ottimizzati salvati in data/signals_optimized/")
    print("Fix rapido completato. Ora possiamo procedere con risk management su basi solide.")


if __name__ == "__main__":
    main()