"""
Overlay Integration Module per Tesi Magistrale - Trend Following Strategies
Implementa trend-following come overlay su portafogli long-only tradizionali
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class TrendFollowingOverlay:
    """
    Sistema di overlay trend-following per portafogli long-only
    Combina allocazioni strategiche con protezione dinamica trend-following
    """
    
    def __init__(self, 
                 portfolios_path: str = "data/portfolios",
                 processed_data_path: str = "data/processed"):
        """
        Inizializza il sistema overlay
        
        Args:
            portfolios_path: Percorso portafogli trend-following
            processed_data_path: Percorso dati asset per benchmark long-only
        """
        self.portfolios_path = portfolios_path
        self.processed_data_path = processed_data_path
        
        self.trend_portfolios = {}
        self.long_only_benchmarks = {}
        self.overlay_results = {}
        
        print("Trend-Following Overlay System inizializzato")
    
    def load_trend_portfolios(self) -> None:
        """Carica portafogli trend-following costruiti"""
        print("Caricamento portafogli trend-following...")
        
        portfolio_files = [
            'equal_weighted_portfolio.csv',
            'risk_parity_portfolio.csv'
        ]
        
        for file_name in portfolio_files:
            file_path = os.path.join(self.portfolios_path, file_name)
            
            if os.path.exists(file_path):
                portfolio_name = file_name.replace('_portfolio.csv', '')
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Verifica colonne necessarie
                required_cols = ['Returns_Vol_Targeted', 'Equity_Vol_Targeted']
                if all(col in df.columns for col in required_cols):
                    self.trend_portfolios[portfolio_name] = df
                    print(f"Caricato {portfolio_name}: {len(df)} osservazioni")
                else:
                    print(f"Colonne mancanti in {portfolio_name}")
        
        print(f"Caricati {len(self.trend_portfolios)} portafogli trend-following")
    
    def create_long_only_benchmarks(self) -> None:
        """Crea benchmark long-only tradizionali"""
        print("Creazione benchmark long-only...")
        
        # Carica SP500 come proxy equity principale
        sp500_path = os.path.join(self.processed_data_path, 'sp500_processed.csv')
        
        if not os.path.exists(sp500_path):
            print("SP500 data non trovato per benchmark")
            return
        
        sp500_df = pd.read_csv(sp500_path, index_col=0, parse_dates=True)
        
        if 'Returns' not in sp500_df.columns:
            print("Colonna Returns mancante in SP500")
            return
        
        sp500_returns = sp500_df['Returns'].dropna()
        
        # 1. Pure Equity (100% SP500)
        pure_equity_df = pd.DataFrame(index=sp500_returns.index)
        pure_equity_df['Returns'] = sp500_returns
        pure_equity_df['Equity'] = (1 + sp500_returns).cumprod()
        pure_equity_df['Drawdown'] = (
            pure_equity_df['Equity'] / pure_equity_df['Equity'].cummax() - 1
        )
        
        self.long_only_benchmarks['pure_equity'] = pure_equity_df
        
        # 2. Balanced 60/40 Portfolio
        # Simula bond component con volatilità ridotta e correlazione parziale
        np.random.seed(42)  # Riproducibilità
        bond_vol = sp500_returns.std() * 0.4  # Bond meno volatili
        
        # Crea bond returns con correlazione ~0.1 con equity
        bond_correlation = 0.1
        bond_independent = pd.Series(
            np.random.normal(0, bond_vol * np.sqrt(1 - bond_correlation**2), len(sp500_returns)),
            index=sp500_returns.index
        )
        bond_correlated = sp500_returns * bond_correlation * (bond_vol / sp500_returns.std())
        bond_returns = bond_correlated + bond_independent
        
        # Portfolio 60/40
        portfolio_6040_returns = 0.6 * sp500_returns + 0.4 * bond_returns
        
        portfolio_6040_df = pd.DataFrame(index=sp500_returns.index)
        portfolio_6040_df['Returns'] = portfolio_6040_returns
        portfolio_6040_df['Equity'] = (1 + portfolio_6040_returns).cumprod()
        portfolio_6040_df['Drawdown'] = (
            portfolio_6040_df['Equity'] / portfolio_6040_df['Equity'].cummax() - 1
        )
        portfolio_6040_df['Equity_Returns'] = sp500_returns
        portfolio_6040_df['Bond_Returns'] = bond_returns
        
        self.long_only_benchmarks['portfolio_6040'] = portfolio_6040_df
        
        print("Benchmark long-only creati: Pure Equity, 60/40 Portfolio")
    
    def create_overlay_combinations(self, 
                                   overlay_allocations: List[float] = None) -> None:
        """
        Crea combinazioni overlay con diverse allocazioni
        
        Args:
            overlay_allocations: Lista percentuali allocazione overlay (es. [0.1, 0.2, 0.3])
        """
        if overlay_allocations is None:
            overlay_allocations = [0.10, 0.15, 0.20, 0.25, 0.30]
        
        print(f"Creazione overlay con allocazioni: {[f'{x:.0%}' for x in overlay_allocations]}")
        
        for trend_name, trend_portfolio in self.trend_portfolios.items():
            for benchmark_name, benchmark_portfolio in self.long_only_benchmarks.items():
                
                # Allinea date (intersezione)
                common_dates = trend_portfolio.index.intersection(benchmark_portfolio.index)
                
                if len(common_dates) < 100:  # Minimo dati richiesti
                    continue
                
                trend_aligned = trend_portfolio.loc[common_dates]
                benchmark_aligned = benchmark_portfolio.loc[common_dates]
                
                # Testa diverse allocazioni overlay
                for overlay_pct in overlay_allocations:
                    
                    overlay_results = self.calculate_overlay_performance(
                        trend_aligned, benchmark_aligned, overlay_pct, 
                        f"{trend_name}_on_{benchmark_name}_{overlay_pct:.0%}"
                    )
                    
                    if overlay_results:
                        combo_key = f"{trend_name}_on_{benchmark_name}"
                        if combo_key not in self.overlay_results:
                            self.overlay_results[combo_key] = {}
                        
                        self.overlay_results[combo_key][f"{overlay_pct:.0%}"] = overlay_results
        
        print(f"Creati overlay per {len(self.overlay_results)} combinazioni")
    
    def calculate_overlay_performance(self,
                                    trend_df: pd.DataFrame,
                                    benchmark_df: pd.DataFrame,
                                    overlay_pct: float,
                                    combo_name: str) -> Dict:
        """
        Calcola performance overlay combination
        
        Args:
            trend_df: DataFrame portafoglio trend-following
            benchmark_df: DataFrame benchmark long-only  
            overlay_pct: Percentuale allocazione overlay (es. 0.2 = 20%)
            combo_name: Nome combinazione
            
        Returns:
            Dict con risultati performance
        """
        
        # Rendimenti componenti
        trend_returns = trend_df['Returns_Vol_Targeted']
        benchmark_returns = benchmark_df['Returns']
        
        # Combinazione overlay
        # Core allocation: (1 - overlay_pct) al benchmark
        # Overlay allocation: overlay_pct al trend-following
        combined_returns = (1 - overlay_pct) * benchmark_returns + overlay_pct * trend_returns
        
        # Crea DataFrame risultati
        overlay_df = pd.DataFrame(index=trend_returns.index)
        overlay_df['Returns_Benchmark'] = benchmark_returns
        overlay_df['Returns_Trend'] = trend_returns
        overlay_df['Returns_Combined'] = combined_returns
        overlay_df['Overlay_Allocation'] = overlay_pct
        
        # Equity curves
        overlay_df['Equity_Benchmark'] = (1 + benchmark_returns).cumprod()
        overlay_df['Equity_Trend'] = (1 + trend_returns).cumprod()
        overlay_df['Equity_Combined'] = (1 + combined_returns).cumprod()
        
        # Drawdowns
        overlay_df['Drawdown_Benchmark'] = (
            overlay_df['Equity_Benchmark'] / overlay_df['Equity_Benchmark'].cummax() - 1
        )
        overlay_df['Drawdown_Combined'] = (
            overlay_df['Equity_Combined'] / overlay_df['Equity_Combined'].cummax() - 1
        )
        
        # Calcola metriche
        metrics = self.calculate_overlay_metrics(overlay_df)
        
        return {
            'data': overlay_df,
            'metrics': metrics,
            'overlay_pct': overlay_pct,
            'combo_name': combo_name
        }
    
    def calculate_overlay_metrics(self, overlay_df: pd.DataFrame) -> Dict:
        """Calcola metriche complete overlay"""
        metrics = {}
        
        # Per ogni serie di rendimenti
        return_series = {
            'benchmark': overlay_df['Returns_Benchmark'].dropna(),
            'trend': overlay_df['Returns_Trend'].dropna(),
            'combined': overlay_df['Returns_Combined'].dropna()
        }
        
        for series_name, returns in return_series.items():
            if len(returns) == 0:
                continue
            
            # Metriche base
            total_return = (1 + returns).prod() - 1
            ann_return = (1 + returns).prod() ** (252/len(returns)) - 1
            ann_vol = returns.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino = ann_return / downside_vol if downside_vol > 0 else 0
            
            # Drawdown
            equity = (1 + returns).cumprod()
            drawdown = equity / equity.cummax() - 1
            max_drawdown = drawdown.min()
            
            # Calmar
            calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Higher moments
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            metrics[series_name] = {
                'total_return': total_return,
                'ann_return': ann_return,
                'volatility': ann_vol,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_drawdown,
                'calmar': calmar,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'hit_rate': (returns > 0).mean()
            }
        
        # Metriche overlay-specific
        if 'benchmark' in metrics and 'combined' in metrics:
            bench = metrics['benchmark']
            comb = metrics['combined']
            
            metrics['overlay_benefits'] = {
                'return_improvement': comb['ann_return'] - bench['ann_return'],
                'sharpe_improvement': comb['sharpe'] - bench['sharpe'],
                'drawdown_improvement': comb['max_drawdown'] - bench['max_drawdown'], # Negativo = miglioramento
                'volatility_change': comb['volatility'] - bench['volatility'],
                'calmar_improvement': comb['calmar'] - bench['calmar']
            }
        
        return metrics
    
    def analyze_crisis_performance(self) -> Dict:
        """Analizza performance overlay durante periodi di crisi"""
        print("Analisi performance durante crisi...")
        
        crisis_periods = {
            'Financial_Crisis_2008': ('2007-10-01', '2009-03-31'),
            'COVID_Crash_2020': ('2020-02-01', '2020-05-31'),
            'Recent_Volatility_2022': ('2022-01-01', '2022-12-31')
        }
        
        crisis_analysis = {}
        
        for combo_key, overlay_data in self.overlay_results.items():
            crisis_analysis[combo_key] = {}
            
            # Analizza performance per ogni allocazione
            for allocation, result_data in overlay_data.items():
                overlay_df = result_data['data']
                crisis_analysis[combo_key][allocation] = {}
                
                # Analizza ogni periodo di crisi
                for crisis_name, (start_date, end_date) in crisis_periods.items():
                    try:
                        mask = (overlay_df.index >= start_date) & (overlay_df.index <= end_date)
                        crisis_data = overlay_df[mask]
                        
                        if len(crisis_data) < 10:  # Dati insufficienti
                            continue
                        
                        # Performance durante crisi
                        bench_return = (1 + crisis_data['Returns_Benchmark']).prod() - 1
                        combined_return = (1 + crisis_data['Returns_Combined']).prod() - 1
                        
                        # Max drawdown durante crisi
                        bench_dd = crisis_data['Drawdown_Benchmark'].min()
                        combined_dd = crisis_data['Drawdown_Combined'].min()
                        
                        crisis_analysis[combo_key][allocation][crisis_name] = {
                            'benchmark_return': bench_return,
                            'combined_return': combined_return,
                            'outperformance': combined_return - bench_return,
                            'benchmark_max_dd': bench_dd,
                            'combined_max_dd': combined_dd,
                            'dd_protection': combined_dd - bench_dd,  # Negativo = protezione
                            'observations': len(crisis_data)
                        }
                        
                    except Exception as e:
                        continue
        
        return crisis_analysis
    
    def find_optimal_allocations(self) -> pd.DataFrame:
        """Trova allocazioni overlay ottimali per ogni combinazione"""
        print("Ricerca allocazioni ottimali...")
        
        optimal_results = []
        
        for combo_key, overlay_data in self.overlay_results.items():
            best_sharpe = -np.inf
            best_calmar = -np.inf
            best_allocation_sharpe = None
            best_allocation_calmar = None
            
            for allocation, result_data in overlay_data.items():
                metrics = result_data['metrics']
                
                if 'combined' in metrics:
                    combined_metrics = metrics['combined']
                    sharpe = combined_metrics['sharpe']
                    calmar = combined_metrics['calmar']
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_allocation_sharpe = allocation
                    
                    if calmar > best_calmar:
                        best_calmar = calmar
                        best_allocation_calmar = allocation
            
            if best_allocation_sharpe:
                best_sharpe_data = overlay_data[best_allocation_sharpe]
                best_metrics = best_sharpe_data['metrics']['combined']
                bench_metrics = best_sharpe_data['metrics']['benchmark']
                
                optimal_results.append({
                    'Combination': combo_key,
                    'Optimal_Allocation_Sharpe': best_allocation_sharpe,
                    'Optimal_Sharpe': best_sharpe,
                    'Combined_Return': best_metrics['ann_return'],
                    'Benchmark_Return': bench_metrics['ann_return'],
                    'Return_Improvement': best_metrics['ann_return'] - bench_metrics['ann_return'],
                    'Combined_MaxDD': best_metrics['max_drawdown'],
                    'Benchmark_MaxDD': bench_metrics['max_drawdown'],
                    'DD_Improvement': best_metrics['max_drawdown'] - bench_metrics['max_drawdown'],
                    'Combined_Volatility': best_metrics['volatility'],
                    'Benchmark_Volatility': bench_metrics['volatility']
                })
        
        return pd.DataFrame(optimal_results)
    
    def save_overlay_results(self, output_path: str = "data/overlay") -> None:
        """Salva risultati overlay analysis"""
        os.makedirs(output_path, exist_ok=True)
        
        # Salva dati dettagliati per ogni combinazione e allocazione
        for combo_key, overlay_data in self.overlay_results.items():
            combo_dir = os.path.join(output_path, combo_key.replace('_on_', '_'))
            os.makedirs(combo_dir, exist_ok=True)
            
            for allocation, result_data in overlay_data.items():
                file_name = f"overlay_{allocation}.csv"
                file_path = os.path.join(combo_dir, file_name)
                result_data['data'].to_csv(file_path)
        
        # Salva allocazioni ottimali
        optimal_df = self.find_optimal_allocations()
        if not optimal_df.empty:
            optimal_path = os.path.join(output_path, 'optimal_allocations.csv')
            optimal_df.to_csv(optimal_path, index=False)
            print(f"Allocazioni ottimali salvate in {optimal_path}")
        
        # Salva crisis analysis
        crisis_results = self.analyze_crisis_performance()
        if crisis_results:
            crisis_path = os.path.join(output_path, 'crisis_analysis.csv')
            
            # Flatten crisis results per CSV
            crisis_flat = []
            for combo, allocations in crisis_results.items():
                for allocation, crises in allocations.items():
                    for crisis, metrics in crises.items():
                        crisis_flat.append({
                            'Combination': combo,
                            'Allocation': allocation,
                            'Crisis_Period': crisis,
                            **metrics
                        })
            
            if crisis_flat:
                crisis_df = pd.DataFrame(crisis_flat)
                crisis_df.to_csv(crisis_path, index=False)
                print(f"Crisis analysis salvata in {crisis_path}")


def main():
    """Test completo overlay integration"""
    print("=== OVERLAY INTEGRATION TEST ===")
    
    # Inizializza overlay system
    overlay_system = TrendFollowingOverlay()
    
    # Carica dati
    overlay_system.load_trend_portfolios()
    overlay_system.create_long_only_benchmarks()
    
    # Crea overlay combinations
    overlay_system.create_overlay_combinations()
    
    # Trova allocazioni ottimali
    optimal_df = overlay_system.find_optimal_allocations()
    
    if not optimal_df.empty:
        print("\nALLOCAZIONI OVERLAY OTTIMALI:")
        print("=" * 80)
        
        for _, row in optimal_df.iterrows():
            print(f"\n{row['Combination']}:")
            print(f"  Allocazione ottimale: {row['Optimal_Allocation_Sharpe']}")
            print(f"  Sharpe combinato: {row['Optimal_Sharpe']:.3f}")
            print(f"  Return improvement: {row['Return_Improvement']:.2%}")
            print(f"  Drawdown improvement: {row['DD_Improvement']:.2%}")
            print(f"  Combined return: {row['Combined_Return']:.2%} vs Benchmark: {row['Benchmark_Return']:.2%}")
    
    # Salva risultati
    overlay_system.save_overlay_results()
    
    print(f"\nOverlay integration completata.")
    print("Risultati salvati in data/overlay/")


if __name__ == "__main__":
    main()