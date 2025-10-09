"""
Diagnostics Module per Tesi Magistrale - Trend Following Strategies
Analisi diagnostica delle performance baseline per identificare problemi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

class TrendFollowingDiagnostics:
    """
    Modulo diagnostico per analizzare performance anomale e problemi nei segnali
    """
    
    def __init__(self, 
                 processed_data_path: str = "data/processed",
                 signals_data_path: str = "data/signals",
                 results_path: str = "results"):
        """
        Inizializza il modulo diagnostico
        """
        self.processed_path = processed_data_path
        self.signals_path = signals_data_path
        self.results_path = results_path
        
        self.asset_data = {}
        self.signals_data = {}
        self.diagnostics_results = {}
        
        os.makedirs(self.results_path, exist_ok=True)
        
        print("Diagnostics Module inizializzato")
    
    def load_all_data(self) -> None:
        """
        Carica dati processati e segnali
        """
        print("Caricamento dati per diagnostics...")
        
        # Carica dati processati
        processed_files = [f for f in os.listdir(self.processed_path) if f.endswith('_processed.csv')]
        for file_name in processed_files:
            asset_name = file_name.replace('_processed.csv', '').upper()
            file_path = os.path.join(self.processed_path, file_name)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.asset_data[asset_name] = df
        
        # Carica segnali
        signal_files = [f for f in os.listdir(self.signals_path) if f.endswith('_signals.csv')]
        for file_name in signal_files:
            asset_name = file_name.replace('_signals.csv', '').upper()
            file_path = os.path.join(self.signals_path, file_name)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.signals_data[asset_name] = df
        
        print(f"Caricati {len(self.asset_data)} asset data e {len(self.signals_data)} signals")
    
    def calculate_buy_and_hold_benchmark(self) -> Dict[str, Dict]:
        """
        Calcola performance buy-and-hold per confronto
        """
        print("Calcolo benchmark buy-and-hold...")
        
        bnh_results = {}
        
        for asset_name, df in self.asset_data.items():
            returns = df['Returns'].dropna()
            
            if len(returns) == 0:
                continue
            
            # Buy and hold metrics
            total_return = (1 + returns).prod() - 1
            ann_return = (1 + returns).prod() ** (252/len(returns)) - 1
            ann_vol = returns.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Drawdown
            equity = (1 + returns).cumprod()
            drawdown = equity / equity.cummax() - 1
            max_dd = drawdown.min()
            
            bnh_results[asset_name] = {
                'Total_Return': total_return,
                'Ann_Return': ann_return,
                'Volatility': ann_vol,
                'Sharpe_Ratio': sharpe,
                'Max_Drawdown': max_dd,
                'Hit_Rate': (returns > 0).mean()
            }
        
        return bnh_results
    
    def analyze_trend_strategy_vs_benchmark(self) -> pd.DataFrame:
        """
        Confronta performance trend-following vs buy-and-hold
        """
        print("Analisi trend-following vs buy-and-hold...")
        
        bnh_results = self.calculate_buy_and_hold_benchmark()
        comparison_data = []
        
        for asset_name, signals_df in self.signals_data.items():
            if asset_name not in self.asset_data:
                continue
            
            df = self.asset_data[asset_name]
            returns = df['Returns']
            
            if 'Signal_Combined' not in signals_df.columns:
                continue
            
            signal = signals_df['Signal_Combined']
            
            # Strategy returns
            strategy_returns = signal.shift(1) * returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0:
                continue
            
            # Strategy metrics
            total_return_strategy = (1 + strategy_returns).prod() - 1
            ann_return_strategy = (1 + strategy_returns).prod() ** (252/len(strategy_returns)) - 1
            ann_vol_strategy = strategy_returns.std() * np.sqrt(252)
            sharpe_strategy = ann_return_strategy / ann_vol_strategy if ann_vol_strategy > 0 else 0
            
            # Benchmark metrics
            bnh_metrics = bnh_results.get(asset_name, {})
            
            comparison_data.append({
                'Asset': asset_name,
                'Strategy_Return': total_return_strategy,
                'BnH_Return': bnh_metrics.get('Total_Return', 0),
                'Strategy_Sharpe': sharpe_strategy,
                'BnH_Sharpe': bnh_metrics.get('Sharpe_Ratio', 0),
                'Strategy_Vol': ann_vol_strategy,
                'BnH_Vol': bnh_metrics.get('Volatility', 0),
                'Outperformance': total_return_strategy - bnh_metrics.get('Total_Return', 0),
                'Sharpe_Diff': sharpe_strategy - bnh_metrics.get('Sharpe_Ratio', 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    def analyze_performance_by_periods(self, asset_name: str) -> Dict:
        """
        Analizza performance per sotto-periodi specifici
        """
        if asset_name not in self.asset_data or asset_name not in self.signals_data:
            return {}
        
        df = self.asset_data[asset_name]
        signals_df = self.signals_data[asset_name]
        
        if 'Signal_Combined' not in signals_df.columns:
            return {}
        
        returns = df['Returns']
        signal = signals_df['Signal_Combined']
        strategy_returns = signal.shift(1) * returns
        
        # Definisci periodi di interesse
        periods = {
            'Pre_Crisis_2005_2007': ('2005-01-01', '2007-12-31'),
            'Financial_Crisis_2008': ('2008-01-01', '2008-12-31'),
            'Recovery_2009_2011': ('2009-01-01', '2011-12-31'),
            'QE_Era_2012_2015': ('2012-01-01', '2015-12-31'),
            'Post_Brexit_2016_2019': ('2016-01-01', '2019-12-31'),
            'COVID_Era_2020_2021': ('2020-01-01', '2021-12-31'),
            'Recent_2022_2025': ('2022-01-01', '2025-12-31')
        }
        
        period_results = {}
        
        for period_name, (start_date, end_date) in periods.items():
            try:
                # Filtra dati per periodo
                mask = (strategy_returns.index >= start_date) & (strategy_returns.index <= end_date)
                period_strategy_returns = strategy_returns[mask].dropna()
                period_bnh_returns = returns[mask].dropna()
                
                if len(period_strategy_returns) < 10:  # Troppo poche osservazioni
                    continue
                
                # Calcola metriche per il periodo
                strategy_total = (1 + period_strategy_returns).prod() - 1
                bnh_total = (1 + period_bnh_returns).prod() - 1
                
                strategy_sharpe = (period_strategy_returns.mean() * np.sqrt(252)) / (period_strategy_returns.std() * np.sqrt(252)) if period_strategy_returns.std() > 0 else 0
                bnh_sharpe = (period_bnh_returns.mean() * np.sqrt(252)) / (period_bnh_returns.std() * np.sqrt(252)) if period_bnh_returns.std() > 0 else 0
                
                period_results[period_name] = {
                    'Strategy_Return': strategy_total,
                    'BnH_Return': bnh_total,
                    'Strategy_Sharpe': strategy_sharpe,
                    'BnH_Sharpe': bnh_sharpe,
                    'Outperformance': strategy_total - bnh_total,
                    'Observations': len(period_strategy_returns)
                }
                
            except Exception as e:
                print(f"Errore nel periodo {period_name} per {asset_name}: {e}")
                continue
        
        return period_results
    
    def investigate_btc_anomaly(self) -> Dict:
        """
        Investigazione specifica per performance negativa di BTC
        """
        print("Investigazione anomalia BTC...")
        
        if 'BTC' not in self.asset_data or 'BTC' not in self.signals_data:
            print("Dati BTC non trovati")
            return {}
        
        df = self.asset_data['BTC']
        signals_df = self.signals_data['BTC']
        
        results = {}
        
        # 1. Verifica range temporale
        results['date_range'] = f"{df.index[0].date()} to {df.index[-1].date()}"
        results['total_days'] = len(df)
        
        # 2. Performance buy-and-hold BTC
        btc_returns = df['Returns'].dropna()
        btc_bnh_return = (1 + btc_returns).prod() - 1
        results['btc_bnh_total_return'] = btc_bnh_return
        results['btc_bnh_ann_return'] = (1 + btc_returns).prod() ** (252/len(btc_returns)) - 1
        
        # 3. Analisi segnali
        if 'Signal_Combined' in signals_df.columns:
            signal = signals_df['Signal_Combined']
            
            results['signal_distribution'] = {
                'long_periods': (signal == 1).sum(),
                'short_periods': (signal == -1).sum(),
                'neutral_periods': (signal == 0).sum()
            }
            
            # 4. Performance strategia
            strategy_returns = signal.shift(1) * btc_returns
            strategy_returns = strategy_returns.dropna()
            strategy_total = (1 + strategy_returns).prod() - 1
            results['strategy_total_return'] = strategy_total
            
            # 5. Analisi per anno - trova anni problematici
            yearly_performance = {}
            for year in range(2015, 2026):  # BTC data starts 2014
                try:
                    year_mask = strategy_returns.index.year == year
                    year_returns = strategy_returns[year_mask]
                    if len(year_returns) > 10:
                        yearly_return = (1 + year_returns).prod() - 1
                        yearly_performance[year] = yearly_return
                except:
                    continue
            
            results['yearly_performance'] = yearly_performance
            
            # 6. Identifica worst performing years
            if yearly_performance:
                worst_years = sorted(yearly_performance.items(), key=lambda x: x[1])[:3]
                results['worst_years'] = worst_years
        
        return results
    
    def create_equity_curves(self, assets: List[str] = None) -> None:
        """
        Crea equity curves per asset selezionati
        """
        if assets is None:
            assets = ['SP500', 'BTC', 'GOLD', 'CAC40']  # Asset rappresentativi
        
        print(f"Creazione equity curves per {assets}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, asset_name in enumerate(assets):
            if i >= 4 or asset_name not in self.asset_data or asset_name not in self.signals_data:
                continue
            
            df = self.asset_data[asset_name]
            signals_df = self.signals_data[asset_name]
            
            if 'Signal_Combined' not in signals_df.columns:
                continue
            
            returns = df['Returns']
            signal = signals_df['Signal_Combined']
            
            # Buy and hold equity curve
            bnh_equity = (1 + returns).cumprod()
            
            # Strategy equity curve
            strategy_returns = signal.shift(1) * returns
            strategy_equity = (1 + strategy_returns).cumprod()
            
            # Plot
            ax = axes[i]
            ax.plot(bnh_equity.index, bnh_equity.values, label='Buy & Hold', alpha=0.7, linewidth=2)
            ax.plot(strategy_equity.index, strategy_equity.values, label='Trend Following', alpha=0.7, linewidth=2)
            ax.set_title(f'{asset_name} - Equity Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Cumulative Return')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'equity_curves_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Chiude il grafico invece di mostrarlo
        print("Grafici salvati (non visualizzati per evitare blocchi)")
        
        print(f"Equity curves salvate in {self.results_path}/equity_curves_comparison.png")
    
    def generate_diagnostic_report(self) -> None:
        """
        Genera report diagnostico completo
        """
        print("Generazione report diagnostico...")
        
        # 1. Confronto generale
        comparison_df = self.analyze_trend_strategy_vs_benchmark()
        
        # 2. Investigazione BTC
        btc_analysis = self.investigate_btc_anomaly()
        
        # 3. Analisi per periodi - asset problematici
        problematic_assets = ['BTC', 'CAC40', 'EUROSTOXX', 'CRUDE']
        period_analysis = {}
        
        for asset in problematic_assets:
            if asset in self.asset_data:
                period_analysis[asset] = self.analyze_performance_by_periods(asset)
        
        # Salva risultati
        report_path = os.path.join(self.results_path, 'diagnostic_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== DIAGNOSTIC REPORT - TREND FOLLOWING STRATEGIES ===\n\n")
            
            f.write("1. CONFRONTO GENERALE TREND-FOLLOWING vs BUY-AND-HOLD\n")
            f.write("=" * 60 + "\n")
            f.write(comparison_df.round(4).to_string(index=False))
            f.write("\n\n")
            
            f.write("2. INVESTIGAZIONE ANOMALIA BTC\n")
            f.write("=" * 30 + "\n")
            for key, value in btc_analysis.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\n")
            
            f.write("3. ANALISI PER SOTTO-PERIODI\n")
            f.write("=" * 30 + "\n")
            for asset, periods in period_analysis.items():
                f.write(f"\n{asset}:\n")
                for period, metrics in periods.items():
                    f.write(f"  {period}: Strategy={metrics['Strategy_Return']:.2%}, BnH={metrics['BnH_Return']:.2%}, Outperf={metrics['Outperformance']:.2%}\n")
        
        print(f"Report diagnostico salvato in {report_path}")
        
        # Stampa summary chiave
        print("\nKEY FINDINGS:")
        print("=" * 50)
        
        # Asset con worst outperformance
        worst_performers = comparison_df.nsmallest(3, 'Outperformance')
        print("Worst performing assets vs Buy-and-Hold:")
        for _, row in worst_performers.iterrows():
            print(f"  {row['Asset']}: {row['Outperformance']:.1%} underperformance")
        
        # BTC summary
        if btc_analysis:
            print(f"\nBTC Analysis:")
            print(f"  Buy-and-Hold return: {btc_analysis.get('btc_bnh_total_return', 0):.1%}")
            print(f"  Strategy return: {btc_analysis.get('strategy_total_return', 0):.1%}")
            print(f"  Long periods: {btc_analysis.get('signal_distribution', {}).get('long_periods', 0)}")


def main():
    """
    Esegue analisi diagnostica completa
    """
    print("=== DIAGNOSTICS MODULE - ANALISI BASELINE ===")
    
    diagnostics = TrendFollowingDiagnostics()
    
    # Carica tutti i dati
    diagnostics.load_all_data()
    
    # Genera equity curves
    diagnostics.create_equity_curves(['SP500', 'BTC', 'GOLD', 'CAC40'])
    
    # Genera report completo
    diagnostics.generate_diagnostic_report()
    
    print("\nAnalisi diagnostica completata.")
    print("Controlla i file in results/ per dettagli completi.")


if __name__ == "__main__":
    main()