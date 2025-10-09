"""
Robustness Analysis Charts per Sezione 4.4
Crea subperiod performance chart e walk-forward analysis table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_robustness_data():
    """Carica i risultati dell'analisi di robustezza"""
    try:
        # Carica subperiod results
        subperiod_df = pd.read_csv('data/robustness/subperiod_results.csv')
        
        # Carica walk-forward results
        walkforward_df = pd.read_csv('data/robustness/walk_forward_results.csv')
        
        return subperiod_df, walkforward_df
    except FileNotFoundError as e:
        print(f"File non trovato: {e}")
        return None, None

def create_subperiod_performance_chart():
    """Crea grafico performance per sub-periodi"""
    subperiod_df, _ = load_robustness_data()
    
    if subperiod_df is None:
        print("Dati subperiod non disponibili")
        return
    
    # Prepara i dati per il grafico
    periods = subperiod_df['Period'].unique()
    strategies = subperiod_df['Strategy'].unique()
    
    # Filtra solo i periodi principali (evita Full_Sample per chiarezza)
    main_periods = ['Pre_Crisis', 'Crisis_Period', 'Recovery', 'Low_Vol_Era', 'COVID_Era', 'Recent']
    available_periods = [p for p in main_periods if p in periods]
    
    if len(available_periods) < 3:
        available_periods = list(periods)[:6]  # Usa i primi 6 se non ci sono quelli attesi
    
    # Prepara dati per plotting
    combined_sharpe = []
    benchmark_sharpe = []
    period_labels = []
    
    for period in available_periods:
        period_data = subperiod_df[subperiod_df['Period'] == period]
        
        combined_row = period_data[period_data['Strategy'] == 'combined']
        benchmark_row = period_data[period_data['Strategy'] == 'benchmark']
        
        if not combined_row.empty and not benchmark_row.empty:
            combined_sharpe.append(combined_row.iloc[0]['sharpe'])
            benchmark_sharpe.append(benchmark_row.iloc[0]['sharpe'])
            
            # Clean period labels
            clean_label = period.replace('_', ' ').title()
            if clean_label == 'Low Vol Era':
                clean_label = 'Low Volatility'
            elif clean_label == 'Covid Era':
                clean_label = 'COVID Era'
            
            period_labels.append(clean_label)
    
    if not combined_sharpe:
        print("Nessun dato valido per il grafico subperiod")
        return
    
    # Crea il grafico
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(period_labels))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, benchmark_sharpe, width, 
                   label='Benchmark Strategy', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, combined_sharpe, width, 
                   label='Optimized Strategy', color='lightgreen', alpha=0.8)
    
    # Add value labels
    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9, fontweight='bold')
    
    add_value_labels(bars1, benchmark_sharpe)
    add_value_labels(bars2, combined_sharpe)
    
    # Customize chart
    ax.set_xlabel('Market Regimes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Robustness Across Market Regimes: Subperiod Performance Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(period_labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add summary statistics
    avg_improvement = np.mean([c - b for c, b in zip(combined_sharpe, benchmark_sharpe)])
    outperform_count = sum(1 for c, b in zip(combined_sharpe, benchmark_sharpe) if c > b)
    
    textstr = f'Average Sharpe Improvement: +{avg_improvement:.3f}\n'
    textstr += f'Periods Outperformed: {outperform_count}/{len(period_labels)}'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('results/robustness_subperiod_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return period_labels, combined_sharpe, benchmark_sharpe

def create_walkforward_table():
    """Crea tabella walk-forward analysis per LaTeX"""
    _, walkforward_df = load_robustness_data()
    
    if walkforward_df is None:
        print("Dati walk-forward non disponibili")
        return
    
    # Aggrega risultati per asset
    assets = walkforward_df['Asset'].unique()
    
    table_data = []
    
    for asset in assets:
        asset_data = walkforward_df[walkforward_df['Asset'] == asset]
        
        if asset_data.empty:
            continue
        
        # Calcola statistiche aggregate
        test_sharpes = asset_data['test_sharpe']
        test_returns = asset_data['test_return']
        
        # Filtra valori infiniti/NaN
        valid_sharpes = test_sharpes[np.isfinite(test_sharpes)]
        valid_returns = test_returns[np.isfinite(test_returns)]
        
        if len(valid_sharpes) == 0:
            continue
        
        avg_test_sharpe = valid_sharpes.mean()
        std_test_sharpe = valid_sharpes.std()
        positive_periods = (valid_returns > 0).sum()
        total_periods = len(valid_returns)
        consistency = (valid_sharpes > 0).sum() / len(valid_sharpes) * 100
        
        table_data.append({
            'Asset': asset,
            'Periods': total_periods,
            'Avg_Test_Sharpe': avg_test_sharpe,
            'Std_Test_Sharpe': std_test_sharpe,
            'Positive_Periods_Pct': positive_periods / total_periods * 100 if total_periods > 0 else 0,
            'Consistency_Pct': consistency
        })
    
    if not table_data:
        print("Nessun dato valido per la tabella walk-forward")
        return
    
    # Crea DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Stampa tabella LaTeX
    print("=== WALK-FORWARD ANALYSIS TABLE FOR LATEX ===")
    print("Asset & Periods & Avg Test & Std Test & Positive & Consistency \\\\")
    print("& Tested & Sharpe & Sharpe & Periods & (\\%) \\\\")
    print("\\hline")
    
    for _, row in table_df.iterrows():
        print(f"{row['Asset']} & {row['Periods']} & {row['Avg_Test_Sharpe']:.3f} & "
              f"{row['Std_Test_Sharpe']:.3f} & {row['Positive_Periods_Pct']:.1f}\\% & "
              f"{row['Consistency_Pct']:.1f}\\% \\\\")
    
    # Calcola e stampa summary row
    avg_sharpe = table_df['Avg_Test_Sharpe'].mean()
    avg_consistency = table_df['Consistency_Pct'].mean()
    avg_positive = table_df['Positive_Periods_Pct'].mean()
    total_periods = table_df['Periods'].sum()
    
    print("\\hline")
    print(f"\\textbf{{Average}} & {total_periods} & \\textbf{{{avg_sharpe:.3f}}} & "
          f"- & \\textbf{{{avg_positive:.1f}\\%}} & \\textbf{{{avg_consistency:.1f}\\%}} \\\\")
    
    # Salva anche come CSV per riferimento
    table_df.to_csv('results/walkforward_summary_table.csv', index=False)
    print(f"\nTabella salvata anche in results/walkforward_summary_table.csv")
    
    return table_df

def print_robustness_insights():
    """Stampa insights chiave per la sezione"""
    subperiod_df, walkforward_df = load_robustness_data()
    
    if subperiod_df is None or walkforward_df is None:
        print("Dati insufficienti per insights")
        return
    
    print("\n=== KEY ROBUSTNESS INSIGHTS ===")
    
    # Subperiod insights
    periods = subperiod_df['Period'].unique()
    improvements = []
    
    for period in periods:
        period_data = subperiod_df[subperiod_df['Period'] == period]
        combined_row = period_data[period_data['Strategy'] == 'combined']
        benchmark_row = period_data[period_data['Strategy'] == 'benchmark']
        
        if not combined_row.empty and not benchmark_row.empty:
            improvement = combined_row.iloc[0]['sharpe'] - benchmark_row.iloc[0]['sharpe']
            improvements.append(improvement)
    
    if improvements:
        avg_improvement = np.mean(improvements)
        consistent_outperform = sum(1 for imp in improvements if imp > 0) / len(improvements)
        
        print(f"Subperiod Analysis:")
        print(f"  Average Sharpe improvement: {avg_improvement:.3f}")
        print(f"  Consistent outperformance: {consistent_outperform:.1%}")
    
    # Walk-forward insights
    if not walkforward_df.empty:
        total_periods = len(walkforward_df)
        avg_test_sharpe = walkforward_df['test_sharpe'][np.isfinite(walkforward_df['test_sharpe'])].mean()
        
        print(f"Walk-Forward Analysis:")
        print(f"  Total out-of-sample periods tested: {total_periods}")
        print(f"  Average out-of-sample Sharpe: {avg_test_sharpe:.3f}")

if __name__ == "__main__":
    print("=== ROBUSTNESS ANALYSIS CHARTS ===")
    
    try:
        # 1. Crea grafico subperiod
        print("1. Creating subperiod performance chart...")
        period_data = create_subperiod_performance_chart()
        
        # 2. Crea tabella walk-forward
        print("\n2. Creating walk-forward analysis table...")
        table_data = create_walkforward_table()
        
        # 3. Stampa insights chiave
        print_robustness_insights()
        
        print(f"\nFiles generated:")
        print("- results/robustness_subperiod_analysis.png")
        print("- results/walkforward_summary_table.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run robustness_analysis.py first to generate the data files.")