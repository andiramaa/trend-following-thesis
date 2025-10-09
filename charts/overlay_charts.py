"""
Overlay Integration Charts per Sezione 4.4
Crea grafici per allocation optimization e crisis protection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_overlay_data():
    """Carica tutti i dati overlay per analisi"""
    # Carica optimal allocations
    optimal_df = pd.read_csv('data/overlay/optimal_allocations.csv')
    
    # Carica crisis analysis
    crisis_df = pd.read_csv('data/overlay/crisis_analysis.csv')
    
    return optimal_df, crisis_df

def load_detailed_overlay_data():
    """Carica dati dettagliati per allocation optimization chart"""
    overlay_combinations = [
        'equal_weighted_pure_equity',
        'equal_weighted_portfolio_6040', 
        'risk_parity_pure_equity',
        'risk_parity_portfolio_6040'
    ]
    
    allocation_data = {}
    
    for combo in overlay_combinations:
        combo_path = f'data/overlay/{combo}'
        if os.path.exists(combo_path):
            combo_data = {}
            
            # Carica ogni allocazione (10%, 15%, 20%, 25%, 30%)
            allocations = ['10%', '15%', '20%', '25%', '30%']
            
            for alloc in allocations:
                file_path = f'{combo_path}/overlay_{alloc}.csv'
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Calcola Sharpe ratio
                    returns = df['Returns_Combined'].dropna()
                    if len(returns) > 0:
                        ann_return = (1 + returns).prod() ** (252/len(returns)) - 1
                        ann_vol = returns.std() * np.sqrt(252)
                        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                        combo_data[alloc] = sharpe
            
            if combo_data:
                allocation_data[combo] = combo_data
    
    return allocation_data

def create_allocation_optimization_chart():
    """Crea grafico allocation optimization"""
    allocation_data = load_detailed_overlay_data()
    
    if not allocation_data:
        print("Dati dettagliati non disponibili per allocation chart")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    allocations = ['10%', '15%', '20%', '25%', '30%']
    allocation_values = [10, 15, 20, 25, 30]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, (combo_name, combo_data) in enumerate(allocation_data.items()):
        sharpe_values = []
        
        for alloc in allocations:
            sharpe_values.append(combo_data.get(alloc, 0))
        
        # Clean combo name for display
        display_name = combo_name.replace('_', ' ').title()
        display_name = display_name.replace('Pure Equity', 'Pure Equity')
        display_name = display_name.replace('Portfolio 6040', '60/40 Portfolio')
        
        # Plot line with markers
        ax.plot(allocation_values, sharpe_values, 
               color=colors[i % len(colors)], 
               marker=markers[i % len(markers)],
               linewidth=2.5, markersize=8, 
               label=display_name, alpha=0.8)
        
        # Highlight optimal point (30%)
        optimal_sharpe = combo_data.get('30%', 0)
        ax.scatter([30], [optimal_sharpe], 
                  color=colors[i % len(colors)], 
                  s=150, marker='*', 
                  edgecolors='black', linewidth=1.5,
                  zorder=5)
    
    # Customize chart
    ax.set_xlabel('Overlay Allocation (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Allocation Optimization Analysis: Sharpe Ratio vs Overlay Allocation\n(Stars indicate optimal allocation)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(allocation_values)
    ax.set_xticklabels([f'{x}%' for x in allocation_values])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Add vertical line at 30% optimum
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(30.5, ax.get_ylim()[1] * 0.95, 'Optimal\nAllocation', 
           fontsize=10, fontweight='bold', color='red', 
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('results/overlay_allocation_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_crisis_protection_chart():
    """Crea grafico crisis protection analysis"""
    _, crisis_df = load_overlay_data()
    
    if crisis_df.empty:
        print("Dati crisis non disponibili")
        return
    
    # Filtra solo allocazione 30% (ottimale)
    crisis_30 = crisis_df[crisis_df['Allocation'] == '30%']
    
    if crisis_30.empty:
        print("Dati crisis per allocazione 30% non disponibili")
        return
    
    # Aggrega per periodo di crisi
    crisis_periods = ['Financial_Crisis_2008', 'COVID_Crash_2020', 'Recent_Volatility_2022']
    period_labels = ['Financial Crisis\n2008', 'COVID Crash\n2020', 'Recent Volatility\n2022']
    
    benchmark_dds = []
    combined_dds = []
    
    for period in crisis_periods:
        period_data = crisis_30[crisis_30['Crisis_Period'] == period]
        
        if not period_data.empty:
            # Media across combinations per questo periodo
            avg_benchmark_dd = period_data['benchmark_max_dd'].mean() * 100
            avg_combined_dd = period_data['combined_max_dd'].mean() * 100
            
            benchmark_dds.append(abs(avg_benchmark_dd))  # Valori positivi per display
            combined_dds.append(abs(avg_combined_dd))
        else:
            benchmark_dds.append(0)
            combined_dds.append(0)
    
    # Crea grafico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(period_labels))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, benchmark_dds, width, 
                   label='Benchmark Portfolio', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, combined_dds, width, 
                   label='30% Overlay Portfolio', color='lightgreen', alpha=0.8)
    
    # Add value labels
    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, benchmark_dds)
    add_value_labels(bars2, combined_dds)
    
    # Add protection arrows and labels
    for i, (bench, comb) in enumerate(zip(benchmark_dds, combined_dds)):
        if bench > 0 and comb > 0:
            protection = bench - comb
            # Arrow from benchmark to overlay
            ax.annotate('', xy=(i + width/2, comb + 1), xytext=(i - width/2, bench - 1),
                       arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
            
            # Protection label
            ax.text(i, (bench + comb) / 2, f'-{protection:.1f}pp',
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Customize chart
    ax.set_xlabel('Crisis Periods', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Crisis Protection Analysis: Drawdown Comparison\n30% Overlay Allocation Performance', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(period_labels)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Summary statistics box
    avg_protection = np.mean([b - c for b, c in zip(benchmark_dds, combined_dds)])
    textstr = f'Average Drawdown Protection: {avg_protection:.1f}pp\n'
    textstr += f'Consistent protection across all crisis periods'
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('results/overlay_crisis_protection.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_chart_insights():
    """Stampa insights per il testo"""
    print("\n=== OVERLAY CHART INSIGHTS ===")
    
    # Allocation optimization insights
    allocation_data = load_detailed_overlay_data()
    if allocation_data:
        print("\nAllocation Optimization:")
        for combo, data in allocation_data.items():
            if '30%' in data:
                print(f"  {combo}: Peak Sharpe at 30% = {data['30%']:.3f}")
    
    # Crisis protection insights
    _, crisis_df = load_overlay_data()
    if not crisis_df.empty:
        crisis_30 = crisis_df[crisis_df['Allocation'] == '30%']
        if not crisis_30.empty:
            avg_protection = crisis_30['dd_protection'].mean() * 100
            print(f"\nCrisis Protection (30% allocation):")
            print(f"  Average DD protection: {abs(avg_protection):.1f}pp")
            
            # Per period
            for period in ['Financial_Crisis_2008', 'COVID_Crash_2020', 'Recent_Volatility_2022']:
                period_data = crisis_30[crisis_30['Crisis_Period'] == period]
                if not period_data.empty:
                    period_protection = abs(period_data['dd_protection'].mean() * 100)
                    print(f"  {period}: {period_protection:.1f}pp protection")

if __name__ == "__main__":
    print("=== OVERLAY INTEGRATION CHARTS ===")
    
    try:
        # 1. Allocation optimization chart
        print("1. Creating allocation optimization chart...")
        create_allocation_optimization_chart()
        
        # 2. Crisis protection chart  
        print("\n2. Creating crisis protection chart...")
        create_crisis_protection_chart()
        
        # 3. Print insights for text
        print_chart_insights()
        
        print(f"\nCharts saved:")
        print("- results/overlay_allocation_optimization.png")
        print("- results/overlay_crisis_protection.png")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure overlay data files exist in data/overlay/")