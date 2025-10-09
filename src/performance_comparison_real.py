"""
Performance Comparison: Baseline vs Optimized (Dati Reali)
Legge i risultati ottimizzati dai CSV e confronta con baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_optimized_results():
    """Legge i risultati ottimizzati dai CSV"""
    optimized_dir = "data/signals_optimized"
    processed_dir = "data/processed"
    
    results = {}
    
    # Trova tutti i file ottimizzati
    if not os.path.exists(optimized_dir):
        print(f"Cartella {optimized_dir} non trovata!")
        return results
    
    optimized_files = [f for f in os.listdir(optimized_dir) if f.endswith('_signals_optimized.csv')]
    print(f"Trovati {len(optimized_files)} file ottimizzati")
    
    for file_name in optimized_files:
        try:
            # Estrai nome asset
            asset_name = file_name.replace('_signals_optimized.csv', '').upper()
            
            # Carica segnali ottimizzati
            signals_path = os.path.join(optimized_dir, file_name)
            signals_df = pd.read_csv(signals_path, index_col=0, parse_dates=True)
            
            # Carica dati originali
            processed_file = f"{asset_name.lower()}_processed.csv"
            processed_path = os.path.join(processed_dir, processed_file)
            
            if not os.path.exists(processed_path):
                print(f"File processato non trovato: {processed_path}")
                continue
                
            data_df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
            
            # Calcola performance ottimizzata
            if 'Signal_Combined' in signals_df.columns and 'Returns' in data_df.columns:
                returns = data_df['Returns'].dropna()
                signal = signals_df['Signal_Combined']
                
                # Allinea segnali e returns
                common_index = returns.index.intersection(signal.index)
                if len(common_index) == 0:
                    continue
                    
                returns_aligned = returns.loc[common_index]
                signal_aligned = signal.loc[common_index]
                
                # Strategy returns (shift signal by 1 per evitare look-ahead bias)
                strategy_returns = signal_aligned.shift(1) * returns_aligned
                strategy_returns = strategy_returns.dropna()
                
                if len(strategy_returns) > 0:
                    # Calcola metriche
                    strategy_total_return = (1 + strategy_returns).prod() - 1
                    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
                    
                    # Buy-and-hold per confronto
                    bnh_total_return = (1 + returns_aligned).prod() - 1
                    bnh_sharpe = returns_aligned.mean() / returns_aligned.std() * np.sqrt(252) if returns_aligned.std() > 0 else 0
                    
                    results[asset_name] = {
                        'strategy_return': strategy_total_return * 100,  # Convert to percentage
                        'bnh_return': bnh_total_return * 100,
                        'strategy_sharpe': strategy_sharpe,
                        'bnh_sharpe': bnh_sharpe
                    }
                    
                    print(f"✓ {asset_name}: Strategy {strategy_total_return:.1%}, BnH {bnh_total_return:.1%}")
                else:
                    print(f"✗ {asset_name}: Nessun return calcolabile")
            else:
                print(f"✗ {asset_name}: Colonne mancanti")
                
        except Exception as e:
            print(f"✗ Errore con {file_name}: {e}")
            continue
    
    return results

def create_performance_comparison():
    """Crea il grafico di confronto performance"""
    
    # Dati baseline dalla tua tabella (converti in percentuali)
    baseline_data = {
        'NASDAQ-100': 180.8, 'GOLD': 93.3, 'SP500': 21.2, 'DAX': 9.3, 
        'HANGSENG': -2.8, 'USTDY': -7.3, 'OMX': -13.3, 'FTSE100': -65.9, 
        'CAC40': -73.6, 'EUROSTOXX': -78.2, 'CRUDE': -82.7, 'BTC': -66.7
    }
    
    # Carica risultati ottimizzati
    optimized_data = load_optimized_results()
    
    if not optimized_data:
        print("Nessun dato ottimizzato caricato!")
        return
    
    # Allinea i dati - usa solo asset presenti in entrambi
    assets = []
    baseline_returns = []
    optimized_returns = []
    
    for asset in baseline_data.keys():
        # Cerca corrispondenze possibili (gestisci variazioni nei nomi)
        asset_variants = [asset, asset.replace('-', ''), asset.split('-')[0]]
        
        found_asset = None
        for variant in asset_variants:
            if variant in optimized_data:
                found_asset = variant
                break
        
        if found_asset:
            assets.append(asset)  # Usa nome originale per display
            baseline_returns.append(baseline_data[asset])
            optimized_returns.append(optimized_data[found_asset]['strategy_return'])
            print(f"Confronto {asset}: Baseline {baseline_data[asset]:.1f}% vs Optimized {optimized_data[found_asset]['strategy_return']:.1f}%")
    
    if not assets:
        print("Nessuna corrispondenza trovata tra baseline e optimized!")
        return
    
    # Crea il grafico
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Posizioni delle barre
    x_pos = np.arange(len(assets))
    bar_width = 0.35
    
    # Crea le barre
    bars1 = ax.bar(x_pos - bar_width/2, baseline_returns, bar_width, 
                   label='Baseline Strategy', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x_pos + bar_width/2, optimized_returns, bar_width, 
                   label='Optimized Strategy', color='#2ca02c', alpha=0.8)
    
    # Personalizza il grafico
    ax.set_xlabel('Asset Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Returns (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Optimized Strategy\n(Real Data)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(assets, rotation=45, ha='right')
    
    # Linea a zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Aggiungi etichette sui valori
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Legenda e griglia
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Statistiche riassuntive
    improvements = [opt - base for base, opt in zip(baseline_returns, optimized_returns)]
    avg_improvement = np.mean(improvements)
    improved_count = sum(1 for x in improvements if x > 0)
    
    textstr = f'Average Improvement: {avg_improvement:.1f} percentage points\n'
    textstr += f'Assets Improved: {improved_count}/{len(assets)}\n'
    textstr += f'Best Improvement: {max(improvements):.1f}pp ({assets[improvements.index(max(improvements))]})'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Salva il grafico
    output_path = "results/performance_comparison_real.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Grafico salvato: {output_path}")
    
    plt.show()
    
    # Stampa tabella riassuntiva
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Asset':<12} {'Baseline':<10} {'Optimized':<10} {'Improvement':<12} {'Status'}")
    print("-"*80)
    
    for i, asset in enumerate(assets):
        baseline = baseline_returns[i]
        optimized = optimized_returns[i] 
        improvement = improvements[i]
        status = "✓ IMPROVED" if improvement > 0 else "✗ WORSE" if improvement < -1 else "≈ SIMILAR"
        
        print(f"{asset:<12} {baseline:>8.1f}% {optimized:>9.1f}% {improvement:>+10.1f}pp {status}")

if __name__ == "__main__":
    print("=== PERFORMANCE COMPARISON - DATI REALI ===")
    create_performance_comparison()