"""
Risk Management Module per Tesi Magistrale - Trend Following Strategies
Implementa stop-loss, volatility targeting e position sizing dinamico
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class TrendFollowingRiskManager:
    """
    Sistema completo di risk management per strategie trend-following
    Integra stop-loss, volatility targeting e position sizing
    """
    
    def __init__(self, 
                 processed_data_path: str = "data/processed",
                 signals_path: str = "data/signals_optimized",
                 target_vol: float = 0.10,
                 max_leverage: float = 3.0):
        """
        Inizializza il risk manager
        
        Args:
            processed_data_path: Percorso dati asset processati
            signals_path: Percorso segnali ottimizzati
            target_vol: Volatilità target annualizzata (default 10%)
            max_leverage: Leverage massimo per posizione (default 3x)
        """
        self.processed_path = processed_data_path
        self.signals_path = signals_path
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        
        self.asset_data = {}
        self.signals_data = {}
        self.risk_managed_strategies = {}
        
        print(f"Risk Manager inizializzato")
        print(f"Target volatility: {target_vol:.1%}")
        print(f"Max leverage: {max_leverage}x")
    
    def load_data(self) -> None:
        """Carica dati processati e segnali ottimizzati"""
        print("Caricamento dati per risk management...")
        
        # Carica dati processati
        processed_files = [f for f in os.listdir(self.processed_path) if f.endswith('_processed.csv')]
        for file_name in processed_files:
            asset_name = file_name.replace('_processed.csv', '').upper()
            file_path = os.path.join(self.processed_path, file_name)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.asset_data[asset_name] = df
        
        # Carica segnali ottimizzati
        signal_files = [f for f in os.listdir(self.signals_path) if f.endswith('_signals_optimized.csv')]
        for file_name in signal_files:
            asset_name = file_name.replace('_signals_optimized.csv', '').upper()
            file_path = os.path.join(self.signals_path, file_name)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.signals_data[asset_name] = df
        
        print(f"Caricati {len(self.asset_data)} asset data e {len(self.signals_data)} signals")
    
    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calcola Average True Range per stop-loss dinamici
        
        Args:
            df: DataFrame con OHLC data
            window: Finestra per media mobile ATR
            
        Returns:
            Serie ATR
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window).mean()
        
        return atr
    
    def apply_stop_loss(self, 
                       asset_name: str,
                       stop_type: str = 'atr',
                       fixed_threshold: float = 0.05,
                       atr_multiplier: float = 2.0,
                       atr_window: int = 14) -> pd.DataFrame:
        """
        Applica regole stop-loss ai segnali
        
        Args:
            asset_name: Nome asset
            stop_type: 'fixed' o 'atr'
            fixed_threshold: Soglia stop fisso (es. 0.05 = 5%)
            atr_multiplier: Moltiplicatore ATR per stop dinamico
            atr_window: Finestra calcolo ATR
            
        Returns:
            DataFrame con posizioni e stop-loss applicati
        """
        if asset_name not in self.asset_data or asset_name not in self.signals_data:
            return pd.DataFrame()
        
        df = self.asset_data[asset_name].copy()
        signals_df = self.signals_data[asset_name].copy()
        
        if 'Signal_Combined' not in signals_df.columns:
            print(f"Segnale combinato non trovato per {asset_name}")
            return pd.DataFrame()
        
        # Calcola ATR se necessario
        if stop_type == 'atr':
            df['ATR'] = self.calculate_atr(df, atr_window)
        
        # DataFrame risultati
        results_df = pd.DataFrame(index=df.index)
        results_df['Price'] = df['Close']
        results_df['Returns'] = df['Returns']
        results_df['Signal_Raw'] = signals_df['Signal_Combined']
        
        # Variabili per tracking stop-loss
        results_df['Position'] = 0.0
        results_df['Entry_Price'] = np.nan
        results_df['Stop_Price'] = np.nan
        results_df['Stop_Triggered'] = False
        results_df['Returns_Strategy'] = 0.0
        
        current_position = 0
        entry_price = 0
        stop_price = 0
        
        for i in range(1, len(results_df)):
            date = results_df.index[i]
            prev_date = results_df.index[i-1]
            
            price = results_df.loc[date, 'Price']
            signal = results_df.loc[date, 'Signal_Raw']
            prev_position = results_df.loc[prev_date, 'Position']
            
            stop_triggered = False
            
            # Check stop-loss se in posizione
            if current_position != 0 and not np.isnan(stop_price):
                if current_position > 0 and price <= stop_price:  # Long stop
                    stop_triggered = True
                elif current_position < 0 and price >= stop_price:  # Short stop
                    stop_triggered = True
            
            # Se stop triggered, chiudi posizione
            if stop_triggered:
                current_position = 0
                entry_price = 0
                stop_price = np.nan
                results_df.loc[date, 'Stop_Triggered'] = True
            
            # Nuovo segnale di entry (solo se non in posizione o cambio direzione)
            elif signal != 0 and (current_position == 0 or np.sign(signal) != np.sign(current_position)):
                current_position = signal
                entry_price = price
                
                # Calcola nuovo stop price
                if stop_type == 'fixed':
                    if current_position > 0:  # Long
                        stop_price = entry_price * (1 - fixed_threshold)
                    else:  # Short
                        stop_price = entry_price * (1 + fixed_threshold)
                
                elif stop_type == 'atr':
                    atr_value = df.loc[date, 'ATR'] if not pd.isna(df.loc[date, 'ATR']) else 0
                    if current_position > 0:  # Long
                        stop_price = entry_price - (atr_multiplier * atr_value)
                    else:  # Short
                        stop_price = entry_price + (atr_multiplier * atr_value)
            
            # Aggiorna risultati
            results_df.loc[date, 'Position'] = current_position
            results_df.loc[date, 'Entry_Price'] = entry_price if current_position != 0 else np.nan
            results_df.loc[date, 'Stop_Price'] = stop_price if current_position != 0 else np.nan
            
            # Calcola rendimenti strategia
            if i > 0:
                results_df.loc[date, 'Returns_Strategy'] = (
                    prev_position * results_df.loc[date, 'Returns']
                )
        
        return results_df
    
    def apply_volatility_targeting(self, 
                                 strategy_df: pd.DataFrame,
                                 vol_window: int = 60,
                                 vol_method: str = 'rolling') -> pd.DataFrame:
        """
        Applica volatility targeting alle posizioni
        
        Args:
            strategy_df: DataFrame con posizioni da scalare
            vol_window: Finestra per stima volatilità
            vol_method: 'rolling' o 'ewma'
            
        Returns:
            DataFrame con posizioni scalate per volatilità
        """
        if 'Returns_Strategy' not in strategy_df.columns:
            return strategy_df
        
        strategy_df = strategy_df.copy()
        
        # Calcola volatilità rolling dei rendimenti underlying
        if vol_method == 'rolling':
            rolling_vol = strategy_df['Returns'].rolling(vol_window).std() * np.sqrt(252)
        else:  # EWMA
            rolling_vol = strategy_df['Returns'].ewm(span=vol_window).std() * np.sqrt(252)
        
        # Calcola scaling factor
        vol_scaling = self.target_vol / rolling_vol
        vol_scaling = vol_scaling.clip(0, self.max_leverage)  # Applica leverage cap
        
        # Applica scaling alle posizioni
        strategy_df['Vol_Scaling'] = vol_scaling
        strategy_df['Position_Scaled'] = strategy_df['Position'] * vol_scaling
        strategy_df['Returns_Vol_Targeted'] = (
            strategy_df['Position_Scaled'].shift(1) * strategy_df['Returns']
        )
        
        return strategy_df
    
    def create_comprehensive_strategy(self,
                                    asset_name: str,
                                    stop_type: str = 'atr',
                                    fixed_threshold: float = 0.05,
                                    atr_multiplier: float = 2.0,
                                    atr_window: int = 14,
                                    vol_window: int = 60,
                                    vol_method: str = 'rolling') -> Dict:
        """
        Crea strategia completa con stop-loss e volatility targeting
        
        Returns:
            Dict con risultati completi della strategia
        """
        print(f"Creazione strategia completa per {asset_name}...")
        
        # Step 1: Applica stop-loss
        strategy_df = self.apply_stop_loss(
            asset_name, stop_type, fixed_threshold, atr_multiplier, atr_window
        )
        
        if strategy_df.empty:
            return {}
        
        # Step 2: Applica volatility targeting
        strategy_df = self.apply_volatility_targeting(
            strategy_df, vol_window, vol_method
        )
        
        # Step 3: Calcola metriche performance
        metrics = self.calculate_strategy_metrics(strategy_df)
        
        # Step 4: Analisi stop-loss
        stop_analysis = self.analyze_stop_loss_performance(strategy_df)
        
        return {
            'data': strategy_df,
            'metrics': metrics,
            'stop_analysis': stop_analysis,
            'config': {
                'stop_type': stop_type,
                'fixed_threshold': fixed_threshold,
                'atr_multiplier': atr_multiplier,
                'atr_window': atr_window,
                'vol_window': vol_window,
                'vol_method': vol_method
            }
        }
    
    def calculate_strategy_metrics(self, strategy_df: pd.DataFrame) -> Dict:
        """Calcola metriche complete della strategia"""
        metrics = {}
        
        # Baseline (senza risk management)
        baseline_returns = strategy_df['Returns_Strategy'].dropna()
        if len(baseline_returns) > 0:
            metrics['baseline_total_return'] = (1 + baseline_returns).prod() - 1
            metrics['baseline_sharpe'] = baseline_returns.mean() / baseline_returns.std() * np.sqrt(252) if baseline_returns.std() > 0 else 0
            metrics['baseline_vol'] = baseline_returns.std() * np.sqrt(252)
            
            baseline_equity = (1 + baseline_returns).cumprod()
            baseline_dd = baseline_equity / baseline_equity.cummax() - 1
            metrics['baseline_max_dd'] = baseline_dd.min()
        
        # Vol-targeted
        vol_returns = strategy_df['Returns_Vol_Targeted'].dropna()
        if len(vol_returns) > 0:
            metrics['vol_targeted_total_return'] = (1 + vol_returns).prod() - 1
            metrics['vol_targeted_sharpe'] = vol_returns.mean() / vol_returns.std() * np.sqrt(252) if vol_returns.std() > 0 else 0
            metrics['vol_targeted_vol'] = vol_returns.std() * np.sqrt(252)
            
            vol_equity = (1 + vol_returns).cumprod()
            vol_dd = vol_equity / vol_equity.cummax() - 1
            metrics['vol_targeted_max_dd'] = vol_dd.min()
        
        # Buy-and-hold benchmark
        bnh_returns = strategy_df['Returns'].dropna()
        if len(bnh_returns) > 0:
            metrics['bnh_total_return'] = (1 + bnh_returns).prod() - 1
            metrics['bnh_sharpe'] = bnh_returns.mean() / bnh_returns.std() * np.sqrt(252) if bnh_returns.std() > 0 else 0
            metrics['bnh_vol'] = bnh_returns.std() * np.sqrt(252)
        
        return metrics
    
    def analyze_stop_loss_performance(self, strategy_df: pd.DataFrame) -> Dict:
        """Analizza efficacia stop-loss"""
        analysis = {}
        
        # Conta stop triggers
        stop_triggers = strategy_df['Stop_Triggered'].sum()
        total_days = len(strategy_df)
        
        analysis['stop_triggers_count'] = stop_triggers
        analysis['stop_trigger_frequency'] = stop_triggers / total_days * 100
        
        # Performance nei giorni successivi ai trigger
        trigger_dates = strategy_df[strategy_df['Stop_Triggered']].index
        
        if len(trigger_dates) > 0:
            post_trigger_returns = []
            for trigger_date in trigger_dates:
                # Prendi rendimenti 5 giorni dopo trigger
                try:
                    trigger_idx = strategy_df.index.get_loc(trigger_date)
                    if trigger_idx + 5 < len(strategy_df):
                        future_returns = strategy_df['Returns'].iloc[trigger_idx+1:trigger_idx+6]
                        post_trigger_returns.extend(future_returns.tolist())
                except:
                    continue
            
            if post_trigger_returns:
                analysis['avg_return_post_trigger'] = np.mean(post_trigger_returns)
                analysis['positive_post_trigger_pct'] = np.mean([r > 0 for r in post_trigger_returns])
        
        return analysis
    
    def run_comprehensive_analysis(self,
                                 asset_list: List[str] = None,
                                 stop_configs: List[Dict] = None) -> Dict:
        """
        Esegue analisi completa su multipli asset e configurazioni
        
        Args:
            asset_list: Lista asset da analizzare (None = tutti)
            stop_configs: Lista configurazioni stop-loss da testare
            
        Returns:
            Dict con risultati completi
        """
        if asset_list is None:
            asset_list = list(self.asset_data.keys())
        
        if stop_configs is None:
            stop_configs = [
                {'stop_type': 'fixed', 'fixed_threshold': 0.02},  # 2% fixed
                {'stop_type': 'fixed', 'fixed_threshold': 0.05},  # 5% fixed
                {'stop_type': 'atr', 'atr_multiplier': 1.5},      # 1.5x ATR
                {'stop_type': 'atr', 'atr_multiplier': 2.0},      # 2.0x ATR
                {'stop_type': 'atr', 'atr_multiplier': 3.0},      # 3.0x ATR
            ]
        
        print(f"Analisi completa su {len(asset_list)} asset e {len(stop_configs)} configurazioni...")
        
        results = {}
        summary_data = []
        
        for asset_name in asset_list:
            if asset_name not in self.asset_data or asset_name not in self.signals_data:
                continue
                
            results[asset_name] = {}
            
            for i, config in enumerate(stop_configs):
                config_name = f"Config_{i+1}_{config['stop_type']}"
                
                strategy_result = self.create_comprehensive_strategy(
                    asset_name, **config
                )
                
                if strategy_result:
                    results[asset_name][config_name] = strategy_result
                    
                    # Aggiungi a summary
                    metrics = strategy_result['metrics']
                    summary_data.append({
                        'Asset': asset_name,
                        'Config': config_name,
                        'Stop_Type': config['stop_type'],
                        'Baseline_Return': metrics.get('baseline_total_return', 0),
                        'Vol_Targeted_Return': metrics.get('vol_targeted_total_return', 0),
                        'Baseline_Sharpe': metrics.get('baseline_sharpe', 0),
                        'Vol_Targeted_Sharpe': metrics.get('vol_targeted_sharpe', 0),
                        'Baseline_MaxDD': metrics.get('baseline_max_dd', 0),
                        'Vol_Targeted_MaxDD': metrics.get('vol_targeted_max_dd', 0),
                        'BnH_Return': metrics.get('bnh_total_return', 0)
                    })
        
        results['summary'] = pd.DataFrame(summary_data)
        
        return results
    
    def save_results(self, results: Dict, output_path: str = "data/risk_managed") -> None:
        """Salva risultati risk management"""
        os.makedirs(output_path, exist_ok=True)
        
        # Salva summary
        if 'summary' in results:
            summary_path = os.path.join(output_path, 'risk_management_summary.csv')
            results['summary'].to_csv(summary_path, index=False)
            print(f"Summary salvato in {summary_path}")
        
        # Salva dati dettagliati per ogni asset
        for asset_name, asset_results in results.items():
            if asset_name == 'summary':
                continue
                
            asset_dir = os.path.join(output_path, asset_name.lower())
            os.makedirs(asset_dir, exist_ok=True)
            
            for config_name, config_results in asset_results.items():
                if 'data' in config_results:
                    file_path = os.path.join(asset_dir, f"{config_name.lower()}.csv")
                    config_results['data'].to_csv(file_path)


def main():
    """Test completo del risk management system"""
    print("=== RISK MANAGEMENT SYSTEM TEST ===")
    
    # Inizializza risk manager
    risk_manager = TrendFollowingRiskManager(target_vol=0.10)
    
    # Carica dati
    risk_manager.load_data()
    
    # Test su asset selezionati (includendo problematici)
    test_assets = ['SP500', 'BTC', 'GOLD', 'NASDAQ', 'CAC40']
    
    # Esegui analisi completa
    results = risk_manager.run_comprehensive_analysis(
        asset_list=test_assets
    )
    
    # Mostra summary risultati
    if 'summary' in results and not results['summary'].empty:
        print("\nSUMMARY RISK MANAGEMENT:")
        print("=" * 80)
        
        summary_df = results['summary']
        
        # Mostra miglior configurazione per asset
        for asset in test_assets:
            asset_data = summary_df[summary_df['Asset'] == asset]
            if not asset_data.empty:
                best_config = asset_data.loc[asset_data['Vol_Targeted_Sharpe'].idxmax()]
                
                print(f"\n{asset} - Best Config: {best_config['Config']}")
                print(f"  Vol-Targeted Return: {best_config['Vol_Targeted_Return']:.2%}")
                print(f"  Vol-Targeted Sharpe: {best_config['Vol_Targeted_Sharpe']:.3f}")
                print(f"  Max Drawdown: {best_config['Vol_Targeted_MaxDD']:.2%}")
                print(f"  vs Buy-Hold: {best_config['BnH_Return']:.2%}")
    
    # Salva risultati
    risk_manager.save_results(results)
    
    print(f"\nRisk management analysis completata.")
    print("Risultati salvati in data/risk_managed/")


if __name__ == "__main__":
    main()