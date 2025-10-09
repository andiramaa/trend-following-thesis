"""
Robustness Analysis Module per Tesi Magistrale - Trend Following Strategies
Walk-forward testing, sensitivity analysis e validazione out-of-sample
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from itertools import product

class TrendFollowingRobustness:
    """
    Sistema completo di robustness testing per strategie trend-following
    Implementa walk-forward, sensitivity analysis e stress testing
    """
    
    def __init__(self, 
                 processed_data_path: str = "data/processed",
                 overlay_path: str = "data/overlay"):
        """
        Inizializza il sistema robustness
        
        Args:
            processed_data_path: Percorso dati asset processati
            overlay_path: Percorso risultati overlay
        """
        self.processed_path = processed_data_path
        self.overlay_path = overlay_path
        
        self.asset_data = {}
        self.robustness_results = {}
        
        print("Robustness Analysis System inizializzato")
    
    def load_data(self) -> None:
        """Carica dati per robustness testing"""
        print("Caricamento dati per robustness analysis...")
        
        # Carica dati asset processati
        processed_files = [f for f in os.listdir(self.processed_path) if f.endswith('_processed.csv')]
        
        for file_name in processed_files:
            asset_name = file_name.replace('_processed.csv', '').upper()
            file_path = os.path.join(self.processed_path, file_name)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.asset_data[asset_name] = df
        
        print(f"Caricati {len(self.asset_data)} asset per robustness testing")
    
    def walk_forward_analysis(self,
                             asset_name: str,
                             training_window: int = 1260,  # 5 anni
                             testing_window: int = 252,    # 1 anno
                             step_size: int = 126) -> Dict: # 6 mesi
        """
        Implementa walk-forward analysis per validazione out-of-sample
        
        Args:
            asset_name: Nome asset da testare
            training_window: Giorni per training window
            testing_window: Giorni per testing window  
            step_size: Giorni step per rolling window
            
        Returns:
            Dict con risultati walk-forward
        """
        print(f"Walk-forward analysis per {asset_name}...")
        
        if asset_name not in self.asset_data:
            return {}
        
        df = self.asset_data[asset_name]
        returns = df['Returns'].dropna()
        
        if len(returns) < training_window + testing_window:
            print(f"Dati insufficienti per {asset_name}")
            return {}
        
        # Parametri da testare durante optimization
        param_grid = {
            'ma_short': [10, 15, 20, 25],
            'ma_long': [40, 50, 60, 70],
            'atr_multiplier': [1.5, 2.0, 2.5, 3.0],
            'vol_target': [0.08, 0.10, 0.12]
        }
        
        walk_forward_results = []
        
        # Rolling windows
        start_idx = training_window
        while start_idx + testing_window <= len(returns):
            
            # Training period
            train_start = start_idx - training_window
            train_end = start_idx
            train_data = returns.iloc[train_start:train_end]
            
            # Testing period
            test_start = start_idx
            test_end = start_idx + testing_window
            test_data = returns.iloc[test_start:test_end]
            
            # Optimize parameters on training data
            best_params, best_sharpe = self.optimize_parameters_on_window(
                train_data, param_grid
            )
            
            # Test on out-of-sample data
            test_performance = self.backtest_with_parameters(
                test_data, best_params
            )
            
            walk_forward_results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                'train_sharpe': best_sharpe,
                'test_sharpe': test_performance.get('sharpe', 0),
                'test_return': test_performance.get('total_return', 0),
                'test_max_dd': test_performance.get('max_drawdown', 0)
            })
            
            start_idx += step_size
        
        return {
            'asset': asset_name,
            'results': walk_forward_results,
            'summary': self.summarize_walk_forward(walk_forward_results)
        }
    
    def optimize_parameters_on_window(self, returns: pd.Series, param_grid: Dict) -> Tuple[Dict, float]:
        """Ottimizza parametri su finestra di training"""
        
        best_sharpe = -np.inf
        best_params = {}
        
        # Grid search su parametri
        param_combinations = list(product(*param_grid.values()))
        
        for combination in param_combinations[:50]:  # Limita per performance
            params = dict(zip(param_grid.keys(), combination))
            
            # Skip invalid combinations
            if params['ma_short'] >= params['ma_long']:
                continue
            
            # Backtest con questi parametri
            performance = self.backtest_with_parameters(returns, params)
            sharpe = performance.get('sharpe', -np.inf)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()
        
        return best_params, best_sharpe
    
    def backtest_with_parameters(self, returns: pd.Series, params: Dict) -> Dict:
        """Backtest semplificato con parametri specificati"""
        
        if len(returns) < max(params['ma_short'], params['ma_long']) + 20:
            return {'sharpe': -np.inf, 'total_return': 0, 'max_drawdown': 0}
        
        # Calcola moving averages
        prices = (1 + returns).cumprod()
        ma_short = prices.rolling(params['ma_short']).mean()
        ma_long = prices.rolling(params['ma_long']).mean()
        
        # Segnali trend
        signal = np.where(ma_short > ma_long, 1, -1)
        signal = pd.Series(signal, index=returns.index)
        
        # Volatility targeting
        rolling_vol = returns.rolling(60).std() * np.sqrt(252)
        vol_scaling = params['vol_target'] / rolling_vol
        vol_scaling = vol_scaling.clip(0, 3.0).fillna(1.0)
        
        # Strategy returns
        strategy_returns = signal.shift(1) * returns * vol_scaling.shift(1)
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {'sharpe': -np.inf, 'total_return': 0, 'max_drawdown': 0}
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        equity = (1 + strategy_returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        max_drawdown = drawdown.min()
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown
        }
    
    def summarize_walk_forward(self, walk_forward_results: List[Dict]) -> Dict:
        """Riassume risultati walk-forward"""
        if not walk_forward_results:
            return {}
        
        test_sharpes = [r['test_sharpe'] for r in walk_forward_results if not np.isinf(r['test_sharpe'])]
        test_returns = [r['test_return'] for r in walk_forward_results]
        test_dds = [r['test_max_dd'] for r in walk_forward_results]
        
        return {
            'num_periods': len(walk_forward_results),
            'avg_test_sharpe': np.mean(test_sharpes) if test_sharpes else 0,
            'std_test_sharpe': np.std(test_sharpes) if test_sharpes else 0,
            'positive_periods_pct': sum(1 for r in test_returns if r > 0) / len(test_returns) if test_returns else 0,
            'avg_test_return': np.mean(test_returns),
            'avg_max_drawdown': np.mean(test_dds),
            'sharpe_consistency': len([s for s in test_sharpes if s > 0]) / len(test_sharpes) if test_sharpes else 0
        }
    
    def parameter_sensitivity_analysis(self) -> Dict:
        """Analisi sensitivity per parametri chiave"""
        print("Parameter sensitivity analysis...")
        
        # Test su asset rappresentativi
        test_assets = ['SP500', 'BTC', 'GOLD']
        available_assets = [asset for asset in test_assets if asset in self.asset_data]
        
        if not available_assets:
            available_assets = list(self.asset_data.keys())[:3]
        
        sensitivity_results = {}
        
        # Parametri da testare
        parameter_ranges = {
            'ma_ratio': [(10,30), (15,45), (20,60), (25,75), (30,90)],  # (short, long)
            'atr_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'vol_target': [0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
            'stop_threshold': [0.02, 0.03, 0.05, 0.07, 0.10]
        }
        
        for asset_name in available_assets:
            print(f"Sensitivity analysis per {asset_name}...")
            
            asset_sensitivity = {}
            returns = self.asset_data[asset_name]['Returns'].dropna()
            
            if len(returns) < 1000:  # Dati insufficienti
                continue
            
            # Test sensitivity per ogni parametro
            for param_name, param_values in parameter_ranges.items():
                param_results = []
                
                for param_value in param_values:
                    # Crea configurazione base
                    base_config = {
                        'ma_short': 20,
                        'ma_long': 60,
                        'atr_multiplier': 2.0,
                        'vol_target': 0.10,
                        'stop_threshold': 0.05
                    }
                    
                    # Modifica parametro testato
                    if param_name == 'ma_ratio':
                        base_config['ma_short'] = param_value[0]
                        base_config['ma_long'] = param_value[1]
                    else:
                        base_config[param_name] = param_value
                    
                    # Backtest
                    performance = self.backtest_with_parameters(returns, base_config)
                    
                    param_results.append({
                        'param_value': param_value,
                        'sharpe': performance['sharpe'],
                        'return': performance['total_return'],
                        'max_dd': performance['max_drawdown']
                    })
                
                asset_sensitivity[param_name] = param_results
            
            sensitivity_results[asset_name] = asset_sensitivity
        
        return sensitivity_results
    
    def subperiod_analysis(self) -> Dict:
        """Analisi performance per sotto-periodi"""
        print("Subperiod analysis...")
        
        periods = {
            'Full_Sample': ('2005-01-01', '2025-12-31'),
            'Pre_Crisis': ('2005-01-01', '2007-12-31'),
            'Crisis_Period': ('2008-01-01', '2009-12-31'),
            'Recovery': ('2010-01-01', '2015-12-31'),
            'Low_Vol_Era': ('2016-01-01', '2019-12-31'),
            'COVID_Era': ('2020-01-01', '2022-12-31'),
            'Recent': ('2023-01-01', '2025-12-31')
        }
        
        subperiod_results = {}
        
        # Test overlay performance per periodo
        overlay_file = os.path.join(self.overlay_path, 'optimal_allocations.csv')
        
        if not os.path.exists(overlay_file):
            print("File overlay optimal allocations non trovato")
            return {}
        
        optimal_df = pd.read_csv(overlay_file)
        
        # Prendi migliore combinazione
        if optimal_df.empty:
            return {}
        
        best_combo = optimal_df.loc[optimal_df['Optimal_Sharpe'].idxmax()]
        combo_name = best_combo['Combination']
        allocation = best_combo['Optimal_Allocation_Sharpe']
        
        # Carica dati overlay
        combo_dir = combo_name.replace('_on_', '_')
        overlay_data_path = os.path.join(self.overlay_path, combo_dir, f"overlay_{allocation}.csv")
        
        if not os.path.exists(overlay_data_path):
            print(f"Overlay data file non trovato: {overlay_data_path}")
            return {}
        
        overlay_df = pd.read_csv(overlay_data_path, index_col=0, parse_dates=True)
        
        # Analizza ogni sotto-periodo
        for period_name, (start_date, end_date) in periods.items():
            try:
                mask = (overlay_df.index >= start_date) & (overlay_df.index <= end_date)
                period_data = overlay_df[mask]
                
                if len(period_data) < 50:  # Dati insufficienti
                    continue
                
                # Calcola metriche per periodo
                bench_returns = period_data['Returns_Benchmark'].dropna()
                combined_returns = period_data['Returns_Combined'].dropna()
                
                period_metrics = {}
                
                for returns_name, returns in [('benchmark', bench_returns), ('combined', combined_returns)]:
                    if len(returns) == 0:
                        continue
                    
                    total_return = (1 + returns).prod() - 1
                    ann_return = (1 + returns).prod() ** (252/len(returns)) - 1
                    ann_vol = returns.std() * np.sqrt(252)
                    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                    
                    equity = (1 + returns).cumprod()
                    drawdown = equity / equity.cummax() - 1
                    max_drawdown = drawdown.min()
                    
                    period_metrics[returns_name] = {
                        'total_return': total_return,
                        'ann_return': ann_return,
                        'volatility': ann_vol,
                        'sharpe': sharpe,
                        'max_drawdown': max_drawdown,
                        'observations': len(returns)
                    }
                
                subperiod_results[period_name] = period_metrics
                
            except Exception as e:
                print(f"Errore nel periodo {period_name}: {e}")
                continue
        
        return subperiod_results
    
    def stress_testing(self) -> Dict:
        """Stress testing su scenari estremi"""
        print("Stress testing...")
        
        # Carica migliore overlay combination
        overlay_file = os.path.join(self.overlay_path, 'optimal_allocations.csv')
        
        if not os.path.exists(overlay_file):
            return {}
        
        optimal_df = pd.read_csv(overlay_file)
        if optimal_df.empty:
            return {}
        
        best_combo = optimal_df.loc[optimal_df['Optimal_Sharpe'].idxmax()]
        combo_name = best_combo['Combination']
        allocation = best_combo['Optimal_Allocation_Sharpe']
        
        # Carica dati overlay
        combo_dir = combo_name.replace('_on_', '_')
        overlay_data_path = os.path.join(self.overlay_path, combo_dir, f"overlay_{allocation}.csv")
        
        if not os.path.exists(overlay_data_path):
            return {}
        
        overlay_df = pd.read_csv(overlay_data_path, index_col=0, parse_dates=True)
        
        stress_results = {}
        
        # 1. Worst drawdown periods analysis
        combined_returns = overlay_df['Returns_Combined'].dropna()
        equity = (1 + combined_returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        
        # Trova worst drawdown periods
        worst_dd = drawdown.min()
        worst_dd_date = drawdown.idxmin()
        
        # Performance durante worst drawdown
        dd_start = equity[:worst_dd_date].idxmax()  # Peak prima del drawdown
        
        stress_results['worst_drawdown'] = {
            'start_date': dd_start,
            'bottom_date': worst_dd_date,
            'max_drawdown': worst_dd,
            'duration_days': (worst_dd_date - dd_start).days,
            'period_return': combined_returns[dd_start:worst_dd_date].sum()
        }
        
        # 2. High volatility periods
        rolling_vol = combined_returns.rolling(20).std() * np.sqrt(252)
        high_vol_threshold = rolling_vol.quantile(0.9)
        high_vol_periods = rolling_vol > high_vol_threshold
        
        if high_vol_periods.any():
            high_vol_returns = combined_returns[high_vol_periods]
            stress_results['high_volatility'] = {
                'threshold': high_vol_threshold,
                'num_periods': high_vol_periods.sum(),
                'avg_return': high_vol_returns.mean(),
                'avg_vol': high_vol_returns.std() * np.sqrt(252),
                'sharpe': high_vol_returns.mean() / high_vol_returns.std() * np.sqrt(252) if high_vol_returns.std() > 0 else 0
            }
        
        # 3. Tail risk analysis
        left_tail = combined_returns.quantile(0.05)
        right_tail = combined_returns.quantile(0.95)
        
        stress_results['tail_risk'] = {
            'var_5pct': left_tail,
            'var_95pct': right_tail,
            'expected_shortfall': combined_returns[combined_returns <= left_tail].mean(),
            'tail_ratio': abs(right_tail / left_tail) if left_tail < 0 else np.inf
        }
        
        return stress_results
    
    def save_robustness_results(self, output_path: str = "data/robustness") -> None:
        """Salva tutti i risultati robustness"""
        os.makedirs(output_path, exist_ok=True)
        
        # Salva walk-forward results
        if 'walk_forward' in self.robustness_results:
            wf_path = os.path.join(output_path, 'walk_forward_results.csv')
            
            wf_flat = []
            for asset, wf_data in self.robustness_results['walk_forward'].items():
                for result in wf_data.get('results', []):
                    wf_flat.append({
                        'Asset': asset,
                        **result,
                        'best_params': str(result['best_params'])
                    })
            
            if wf_flat:
                wf_df = pd.DataFrame(wf_flat)
                wf_df.to_csv(wf_path, index=False)
                print(f"Walk-forward results salvati in {wf_path}")
        
        # Salva sensitivity results
        if 'sensitivity' in self.robustness_results:
            sens_path = os.path.join(output_path, 'sensitivity_results.csv')
            
            sens_flat = []
            for asset, asset_sens in self.robustness_results['sensitivity'].items():
                for param, param_results in asset_sens.items():
                    for result in param_results:
                        sens_flat.append({
                            'Asset': asset,
                            'Parameter': param,
                            'Value': str(result['param_value']),
                            'Sharpe': result['sharpe'],
                            'Return': result['return'],
                            'MaxDD': result['max_dd']
                        })
            
            if sens_flat:
                sens_df = pd.DataFrame(sens_flat)
                sens_df.to_csv(sens_path, index=False)
                print(f"Sensitivity results salvati in {sens_path}")
        
        # Salva subperiod results
        if 'subperiod' in self.robustness_results:
            sub_path = os.path.join(output_path, 'subperiod_results.csv')
            
            sub_flat = []
            for period, period_data in self.robustness_results['subperiod'].items():
                for strategy, metrics in period_data.items():
                    sub_flat.append({
                        'Period': period,
                        'Strategy': strategy,
                        **metrics
                    })
            
            if sub_flat:
                sub_df = pd.DataFrame(sub_flat)
                sub_df.to_csv(sub_path, index=False)
                print(f"Subperiod results salvati in {sub_path}")


def main():
    """Test completo robustness analysis"""
    print("=== ROBUSTNESS ANALYSIS TEST ===")
    
    # Inizializza robustness system
    robustness = TrendFollowingRobustness()
    robustness.load_data()
    
    # 1. Walk-forward analysis (su asset selezionati per velocità)
    test_assets = ['SP500', 'BTC']
    walk_forward_results = {}
    
    for asset in test_assets:
        if asset in robustness.asset_data:
            wf_result = robustness.walk_forward_analysis(asset)
            if wf_result:
                walk_forward_results[asset] = wf_result
    
    robustness.robustness_results['walk_forward'] = walk_forward_results
    
    # 2. Parameter sensitivity
    sensitivity_results = robustness.parameter_sensitivity_analysis()
    robustness.robustness_results['sensitivity'] = sensitivity_results
    
    # 3. Subperiod analysis
    subperiod_results = robustness.subperiod_analysis()
    robustness.robustness_results['subperiod'] = subperiod_results
    
    # 4. Stress testing
    stress_results = robustness.stress_testing()
    robustness.robustness_results['stress'] = stress_results
    
    # Mostra summary risultati
    print("\nROBUSTNESS ANALYSIS SUMMARY:")
    print("=" * 80)
    
    # Walk-forward summary
    if walk_forward_results:
        print("\nWALK-FORWARD RESULTS:")
        for asset, wf_data in walk_forward_results.items():
            summary = wf_data.get('summary', {})
            print(f"  {asset}:")
            print(f"    Avg Test Sharpe: {summary.get('avg_test_sharpe', 0):.3f}")
            print(f"    Positive Periods: {summary.get('positive_periods_pct', 0):.1%}")
            print(f"    Consistency: {summary.get('sharpe_consistency', 0):.1%}")
    
    # Subperiod summary
    if subperiod_results:
        print("\nSUBPERIOD PERFORMANCE:")
        for period, data in subperiod_results.items():
            if 'combined' in data and 'benchmark' in data:
                comb_sharpe = data['combined'].get('sharpe', 0)
                bench_sharpe = data['benchmark'].get('sharpe', 0)
                print(f"  {period}: Combined {comb_sharpe:.3f} vs Benchmark {bench_sharpe:.3f}")
    
    # Stress testing summary
    if stress_results:
        print("\nSTRESS TESTING:")
        if 'worst_drawdown' in stress_results:
            wd = stress_results['worst_drawdown']
            print(f"  Worst Drawdown: {wd['max_drawdown']:.2%} ({wd['duration_days']} days)")
        
        if 'tail_risk' in stress_results:
            tr = stress_results['tail_risk']
            print(f"  VaR 5%: {tr['var_5pct']:.2%}")
            print(f"  Expected Shortfall: {tr['expected_shortfall']:.2%}")
    
    # Salva tutti i risultati
    robustness.save_robustness_results()
    
    print(f"\nRobustness analysis completata.")
    print("Risultati salvati in data/robustness/")


if __name__ == "__main__":
    main()