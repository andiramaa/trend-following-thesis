"""
Portfolio Construction Module per Tesi Magistrale - Trend Following Strategies
Aggregazione multi-asset con risk-parity e equal-weighting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class TrendFollowingPortfolio:
    """
    Costruttore di portafoglio multi-asset per strategie trend-following
    Implementa equal-weighting, risk-parity e analisi performance
    """
    
    def __init__(self, 
                 risk_managed_path: str = "data/risk_managed",
                 target_vol: float = 0.10):
        """
        Inizializza il portfolio constructor
        
        Args:
            risk_managed_path: Percorso dati risk-managed
            target_vol: Volatilità target del portafoglio
        """
        self.risk_managed_path = risk_managed_path
        self.target_vol = target_vol
        
        self.asset_strategies = {}
        self.portfolios = {}
        
        print(f"Portfolio Constructor inizializzato")
        print(f"Target portfolio volatility: {target_vol:.1%}")
    
    def load_risk_managed_data(self) -> None:
        """Carica i migliori risultati risk-managed per ogni asset"""
        print("Caricamento dati risk-managed...")
        
        # Carica summary per identificare migliori configurazioni
        summary_path = os.path.join(self.risk_managed_path, 'risk_management_summary.csv')
        
        if not os.path.exists(summary_path):
            print(f"Summary file non trovato: {summary_path}")
            return
        
        summary_df = pd.read_csv(summary_path)
        
        # Trova migliore configurazione per ogni asset (per Sharpe ratio)
        best_configs = {}
        for asset in summary_df['Asset'].unique():
            asset_data = summary_df[summary_df['Asset'] == asset]
            best_idx = asset_data['Vol_Targeted_Sharpe'].idxmax()
            best_config = asset_data.loc[best_idx]
            best_configs[asset] = {
                'config_name': best_config['Config'],
                'sharpe': best_config['Vol_Targeted_Sharpe'],
                'return': best_config['Vol_Targeted_Return'],
                'max_dd': best_config['Vol_Targeted_MaxDD']
            }
        
        # Carica dati dettagliati per ogni asset
        for asset, config_info in best_configs.items():
            asset_dir = os.path.join(self.risk_managed_path, asset.lower())
            config_file = os.path.join(asset_dir, f"{config_info['config_name'].lower()}.csv")
            
            if os.path.exists(config_file):
                df = pd.read_csv(config_file, index_col=0, parse_dates=True)
                
                # Verifica che abbia le colonne necessarie
                if 'Returns_Vol_Targeted' in df.columns:
                    self.asset_strategies[asset] = {
                        'data': df,
                        'config': config_info,
                        'returns': df['Returns_Vol_Targeted'].dropna()
                    }
                    print(f"Caricato {asset}: Sharpe {config_info['sharpe']:.3f}")
                else:
                    print(f"Colonne mancanti per {asset}")
            else:
                print(f"File non trovato: {config_file}")
        
        print(f"Caricati {len(self.asset_strategies)} asset strategies")
    
    def create_equal_weighted_portfolio(self) -> pd.DataFrame:
        """Crea portafoglio equal-weighted"""
        print("Creazione portafoglio equal-weighted...")
        
        if not self.asset_strategies:
            return pd.DataFrame()
        
        # Raccogli tutti i rendimenti
        all_returns = {}
        for asset, strategy in self.asset_strategies.items():
            all_returns[asset] = strategy['returns']
        
        # Crea DataFrame con tutti i rendimenti
        returns_df = pd.DataFrame(all_returns)
        
        # Rimuovi date con troppi NaN (meno del 50% degli asset)
        min_assets = len(self.asset_strategies) // 2
        returns_df = returns_df.dropna(thresh=min_assets)
        
        # Equal-weighted portfolio (media semplice)
        portfolio_returns = returns_df.mean(axis=1, skipna=True)
        
        # Crea DataFrame risultati
        portfolio_df = pd.DataFrame(index=returns_df.index)
        portfolio_df['Returns'] = portfolio_returns
        portfolio_df['Equity'] = (1 + portfolio_returns).cumprod()
        portfolio_df['Drawdown'] = (
            portfolio_df['Equity'] / portfolio_df['Equity'].cummax() - 1
        )
        
        # Conta asset attivi per giorno
        portfolio_df['Active_Assets'] = returns_df.count(axis=1)
        
        return portfolio_df
    
    def create_risk_parity_portfolio(self, lookback_window: int = 60) -> pd.DataFrame:
        """
        Crea portafoglio risk-parity basato su volatilità inversa
        
        Args:
            lookback_window: Finestra per calcolo volatilità
            
        Returns:
            DataFrame con portafoglio risk-parity
        """
        print("Creazione portafoglio risk-parity...")
        
        if not self.asset_strategies:
            return pd.DataFrame()
        
        # Raccogli rendimenti
        all_returns = {}
        for asset, strategy in self.asset_strategies.items():
            all_returns[asset] = strategy['returns']
        
        returns_df = pd.DataFrame(all_returns)
        min_assets = len(self.asset_strategies) // 2
        returns_df = returns_df.dropna(thresh=min_assets)
        
        # Calcola pesi risk-parity dinamici
        portfolio_returns = []
        weights_history = []
        
        for i in range(lookback_window, len(returns_df)):
            # Finestra per calcolo volatilità
            window_returns = returns_df.iloc[i-lookback_window:i]
            
            # Calcola volatilità per ogni asset (ignora NaN)
            vols = {}
            for asset in returns_df.columns:
                asset_returns = window_returns[asset].dropna()
                if len(asset_returns) > 10:  # Min 10 osservazioni
                    vols[asset] = asset_returns.std()
            
            if len(vols) < 2:  # Serve almeno 2 asset
                portfolio_returns.append(0)
                weights_history.append({})
                continue
            
            # Calcola pesi inversely proportional alla volatilità
            inv_vols = {asset: 1/vol if vol > 0 else 0 for asset, vol in vols.items()}
            total_inv_vol = sum(inv_vols.values())
            
            if total_inv_vol > 0:
                weights = {asset: inv_vol/total_inv_vol for asset, inv_vol in inv_vols.items()}
            else:
                # Fallback equal-weight se problemi
                weights = {asset: 1/len(vols) for asset in vols.keys()}
            
            # Calcola rendimento portafoglio per questa data
            current_returns = returns_df.iloc[i]
            portfolio_return = sum(
                weights.get(asset, 0) * current_returns[asset] 
                for asset in current_returns.index 
                if not pd.isna(current_returns[asset]) and asset in weights
            )
            
            portfolio_returns.append(portfolio_return)
            weights_history.append(weights.copy())
        
        # Crea DataFrame risultati
        portfolio_dates = returns_df.index[lookback_window:]
        portfolio_df = pd.DataFrame(index=portfolio_dates)
        portfolio_df['Returns'] = portfolio_returns
        portfolio_df['Equity'] = (1 + pd.Series(portfolio_returns, index=portfolio_dates)).cumprod()
        portfolio_df['Drawdown'] = (
            portfolio_df['Equity'] / portfolio_df['Equity'].cummax() - 1
        )
        
        # Aggiungi storia pesi (per analisi)
        self.risk_parity_weights = weights_history
        
        return portfolio_df
    
    def create_volatility_targeted_portfolio(self, base_portfolio: pd.DataFrame) -> pd.DataFrame:
        """
        Applica volatility targeting al portafoglio
        
        Args:
            base_portfolio: DataFrame portafoglio base
            
        Returns:
            DataFrame portafoglio vol-targeted
        """
        if base_portfolio.empty or 'Returns' not in base_portfolio.columns:
            return base_portfolio
        
        portfolio_df = base_portfolio.copy()
        
        # Calcola volatilità rolling del portafoglio
        rolling_vol = portfolio_df['Returns'].rolling(60).std() * np.sqrt(252)
        
        # Calcola scaling factor
        vol_scaling = self.target_vol / rolling_vol
        vol_scaling = vol_scaling.clip(0, 2.0)  # Cap a 2x leverage
        
        # Applica scaling
        portfolio_df['Vol_Scaling'] = vol_scaling
        portfolio_df['Returns_Vol_Targeted'] = portfolio_df['Returns'] * vol_scaling.shift(1)
        portfolio_df['Equity_Vol_Targeted'] = (1 + portfolio_df['Returns_Vol_Targeted']).cumprod()
        portfolio_df['Drawdown_Vol_Targeted'] = (
            portfolio_df['Equity_Vol_Targeted'] / portfolio_df['Equity_Vol_Targeted'].cummax() - 1
        )
        
        return portfolio_df
    
    def calculate_portfolio_metrics(self, portfolio_df: pd.DataFrame, 
                                  returns_col: str = 'Returns') -> Dict:
        """Calcola metriche complete del portafoglio"""
        if portfolio_df.empty or returns_col not in portfolio_df.columns:
            return {}
        
        returns = portfolio_df[returns_col].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Metriche base
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + returns).prod() ** (252/len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown metrics
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': ann_return,
            'Volatility': ann_vol,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Calmar_Ratio': calmar,
            'Max_Drawdown': max_drawdown,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Hit_Rate': hit_rate,
            'Observations': len(returns)
        }
    
    def create_benchmark_comparison(self) -> pd.DataFrame:
        """Crea confronto con benchmark tradizionali"""
        print("Creazione benchmark comparison...")
        
        # Usa SP500 come proxy equity market
        if 'SP500' not in self.asset_strategies:
            return pd.DataFrame()
        
        sp500_data = self.asset_strategies['SP500']['data']
        
        if 'Returns' not in sp500_data.columns:
            return pd.DataFrame()
        
        # Benchmark 60/40 portfolio simulation
        equity_returns = sp500_data['Returns']
        
        # Simula bond returns (più conservativo)
        # Usa metà volatilità dell'equity con correlazione 0.2
        bond_vol = equity_returns.std() * 0.5
        bond_returns = pd.Series(
            np.random.normal(0, bond_vol, len(equity_returns)),
            index=equity_returns.index
        )
        
        # 60/40 portfolio
        portfolio_6040 = 0.6 * equity_returns + 0.4 * bond_returns
        
        # Crea DataFrame confronto
        comparison_df = pd.DataFrame(index=equity_returns.index)
        comparison_df['SP500'] = equity_returns
        comparison_df['Portfolio_6040'] = portfolio_6040
        comparison_df['Equity_SP500'] = (1 + equity_returns).cumprod()
        comparison_df['Equity_6040'] = (1 + portfolio_6040).cumprod()
        
        return comparison_df
    
    def run_complete_portfolio_analysis(self) -> Dict:
        """Esegue analisi completa portafoglio"""
        print("=== PORTFOLIO CONSTRUCTION ANALYSIS ===")
        
        results = {}
        
        # 1. Equal-weighted portfolio
        eq_weighted = self.create_equal_weighted_portfolio()
        if not eq_weighted.empty:
            eq_weighted_vol_targeted = self.create_volatility_targeted_portfolio(eq_weighted)
            
            results['equal_weighted'] = {
                'data': eq_weighted_vol_targeted,
                'base_metrics': self.calculate_portfolio_metrics(eq_weighted, 'Returns'),
                'vol_targeted_metrics': self.calculate_portfolio_metrics(
                    eq_weighted_vol_targeted, 'Returns_Vol_Targeted'
                )
            }
        
        # 2. Risk-parity portfolio
        risk_parity = self.create_risk_parity_portfolio()
        if not risk_parity.empty:
            risk_parity_vol_targeted = self.create_volatility_targeted_portfolio(risk_parity)
            
            results['risk_parity'] = {
                'data': risk_parity_vol_targeted,
                'base_metrics': self.calculate_portfolio_metrics(risk_parity, 'Returns'),
                'vol_targeted_metrics': self.calculate_portfolio_metrics(
                    risk_parity_vol_targeted, 'Returns_Vol_Targeted'
                )
            }
        
        # 3. Benchmark comparison
        benchmark_data = self.create_benchmark_comparison()
        if not benchmark_data.empty:
            results['benchmarks'] = {
                'data': benchmark_data,
                'sp500_metrics': self.calculate_portfolio_metrics(benchmark_data, 'SP500'),
                'portfolio_6040_metrics': self.calculate_portfolio_metrics(
                    benchmark_data, 'Portfolio_6040'
                )
            }
        
        return results
    
    def save_portfolio_results(self, results: Dict, output_path: str = "data/portfolios") -> None:
        """Salva risultati portfolio"""
        os.makedirs(output_path, exist_ok=True)
        
        # Salva dati dettagliati
        for portfolio_type, portfolio_data in results.items():
            if 'data' in portfolio_data:
                file_path = os.path.join(output_path, f"{portfolio_type}_portfolio.csv")
                portfolio_data['data'].to_csv(file_path)
                print(f"Salvato {portfolio_type} portfolio -> {file_path}")
        
        # Crea summary metriche
        summary_data = []
        
        for portfolio_type, portfolio_data in results.items():
            if 'base_metrics' in portfolio_data:
                base_metrics = portfolio_data['base_metrics']
                summary_data.append({
                    'Portfolio': f"{portfolio_type}_base",
                    **base_metrics
                })
            
            if 'vol_targeted_metrics' in portfolio_data:
                vol_metrics = portfolio_data['vol_targeted_metrics']
                summary_data.append({
                    'Portfolio': f"{portfolio_type}_vol_targeted",
                    **vol_metrics
                })
            
            # Aggiungi benchmark se presenti
            if portfolio_type == 'benchmarks':
                if 'sp500_metrics' in portfolio_data:
                    summary_data.append({
                        'Portfolio': 'SP500_benchmark',
                        **portfolio_data['sp500_metrics']
                    })
                if 'portfolio_6040_metrics' in portfolio_data:
                    summary_data.append({
                        'Portfolio': '60_40_benchmark',
                        **portfolio_data['portfolio_6040_metrics']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_path, 'portfolio_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary salvato in {summary_path}")


def main():
    """Test portfolio construction"""
    print("=== PORTFOLIO CONSTRUCTION TEST ===")
    
    # Inizializza portfolio constructor
    portfolio_builder = TrendFollowingPortfolio(target_vol=0.10)
    
    # Carica dati risk-managed
    portfolio_builder.load_risk_managed_data()
    
    # Esegui analisi completa
    results = portfolio_builder.run_complete_portfolio_analysis()
    
    # Mostra risultati
    print("\nPORTFOLIO PERFORMANCE SUMMARY:")
    print("=" * 80)
    
    for portfolio_type, data in results.items():
        if portfolio_type == 'benchmarks':
            continue
            
        print(f"\n{portfolio_type.upper()} PORTFOLIO:")
        
        if 'base_metrics' in data:
            base = data['base_metrics']
            print(f"  Base - Return: {base.get('Annualized_Return', 0):.2%}, "
                  f"Sharpe: {base.get('Sharpe_Ratio', 0):.3f}, "
                  f"MaxDD: {base.get('Max_Drawdown', 0):.2%}")
        
        if 'vol_targeted_metrics' in data:
            vol = data['vol_targeted_metrics']
            print(f"  Vol-Targeted - Return: {vol.get('Annualized_Return', 0):.2%}, "
                  f"Sharpe: {vol.get('Sharpe_Ratio', 0):.3f}, "
                  f"MaxDD: {vol.get('Max_Drawdown', 0):.2%}")
    
    # Confronto con benchmark
    if 'benchmarks' in results:
        print(f"\nBENCHMARK COMPARISON:")
        benchmarks = results['benchmarks']
        
        if 'sp500_metrics' in benchmarks:
            sp500 = benchmarks['sp500_metrics']
            print(f"  SP500 - Return: {sp500.get('Annualized_Return', 0):.2%}, "
                  f"Sharpe: {sp500.get('Sharpe_Ratio', 0):.3f}, "
                  f"MaxDD: {sp500.get('Max_Drawdown', 0):.2%}")
        
        if 'portfolio_6040_metrics' in benchmarks:
            p6040 = benchmarks['portfolio_6040_metrics']
            print(f"  60/40 - Return: {p6040.get('Annualized_Return', 0):.2%}, "
                  f"Sharpe: {p6040.get('Sharpe_Ratio', 0):.3f}, "
                  f"MaxDD: {p6040.get('Max_Drawdown', 0):.2%}")
    
    # Salva risultati
    portfolio_builder.save_portfolio_results(results)
    
    print(f"\nPortfolio construction completata.")
    print("Risultati salvati in data/portfolios/")


if __name__ == "__main__":
    main()