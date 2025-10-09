"""
Risk Management Analysis - Charts per Sezione 4.2
Mostra efficacia del risk management attraverso grafici mirati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_risk_data():
    """Carica i risultati del risk management"""
    df = pd.read_csv('data/risk_managed/risk_management_summary.csv')
    return df

def create_risk_return_scatter():
    """Crea scatter plot Risk-Return prima e dopo risk management"""
    df = load_risk_data()
    
    # Prepara i dati per il scatter plot
    assets = df['Asset'].unique()
    
    # Calcola medie per asset (across all configurations)
    asset_summary = []
    
    for asset in assets:
        asset_data = df[df['Asset'] == asset]
        
        # Media delle performance baseline vs vol-targeted
        baseline_sharpe = asset_data['Baseline_Sharpe'].mean()
        vol_targeted_sharpe = asset_data['Vol_Targeted_Sharpe'].mean()
        baseline_maxdd = asset_data['Baseline_MaxDD'].mean() * 100  # Convert to percentage
        vol_targeted_maxdd = asset_data['Vol_Targeted_MaxDD'].mean() * 100
        
        asset_summary.append({
            'Asset': asset,
            'Baseline_Sharpe': baseline_sharpe,
            'Vol_Targeted_Sharpe': vol_targeted_sharpe,
            'Baseline_MaxDD': abs(baseline_maxdd),  # Use absolute value for plotting
            'Vol_Targeted_MaxDD': abs(vol_targeted_maxdd)
        })
    
    asset_df = pd.DataFrame(asset_summary)
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot baseline (before risk management)
    baseline_scatter = ax.scatter(asset_df['Baseline_MaxDD'], asset_df['Baseline_Sharpe'], 
                                 s=120, alpha=0.7, color='red', label='Baseline Strategy', 
                                 edgecolors='darkred', linewidth=1)
    
    # Plot vol-targeted (after risk management)
    vol_targeted_scatter = ax.scatter(asset_df['Vol_Targeted_MaxDD'], asset_df['Vol_Targeted_Sharpe'], 
                                     s=120, alpha=0.7, color='green', label='Risk-Managed Strategy',
                                     edgecolors='darkgreen', linewidth=1)
    
    # Add asset labels
    for i, row in asset_df.iterrows():
        # Label for baseline
        ax.annotate(row['Asset'], 
                   (row['Baseline_MaxDD'], row['Baseline_Sharpe']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='darkred', fontweight='bold')
        
        # Label for vol-targeted
        ax.annotate(row['Asset'], 
                   (row['Vol_Targeted_MaxDD'], row['Vol_Targeted_Sharpe']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='darkgreen', fontweight='bold')
        
        # Draw arrows showing improvement
        ax.annotate('', xy=(row['Vol_Targeted_MaxDD'], row['Vol_Targeted_Sharpe']),
                   xytext=(row['Baseline_MaxDD'], row['Baseline_Sharpe']),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5))
    
    # Customize the plot
    ax.set_xlabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return Profile: Before vs After Risk Management\n(Lower MaxDD and Higher Sharpe = Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Add summary statistics
    avg_sharpe_improvement = (asset_df['Vol_Targeted_Sharpe'] - asset_df['Baseline_Sharpe']).mean()
    avg_maxdd_improvement = (asset_df['Baseline_MaxDD'] - asset_df['Vol_Targeted_MaxDD']).mean()
    
    textstr = f'Average Improvements:\n'
    textstr += f'Sharpe Ratio: +{avg_sharpe_improvement:.3f}\n'
    textstr += f'Max Drawdown: -{avg_maxdd_improvement:.1f}pp'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('results/risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return asset_df

def create_configuration_comparison():
    """Crea grafico di confronto configurazioni per asset selezionati"""
    df = load_risk_data()
    
    # Focus su BTC e SP500 (casi più interessanti)
    focus_assets = ['BTC', 'SP500', 'NASDAQ']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Stop-Loss Configuration Effectiveness by Asset', fontsize=16, fontweight='bold')
    
    for idx, asset in enumerate(focus_assets):
        ax = axes[idx]
        asset_data = df[df['Asset'] == asset].copy()
        
        if asset_data.empty:
            continue
        
        # Prepare data for bar chart
        configs = asset_data['Config'].values
        config_labels = [f"{row['Stop_Type'].upper()}\n{row['Config'].split('_')[1]}" 
                        for _, row in asset_data.iterrows()]
        
        baseline_sharpe = asset_data['Baseline_Sharpe'].values
        vol_targeted_sharpe = asset_data['Vol_Targeted_Sharpe'].values
        
        x_pos = np.arange(len(configs))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x_pos - width/2, baseline_sharpe, width, 
                      label='Baseline', color='lightcoral', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, vol_targeted_sharpe, width, 
                      label='Risk-Managed', color='lightgreen', alpha=0.8)
        
        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        add_value_labels(bars1, baseline_sharpe)
        add_value_labels(bars2, vol_targeted_sharpe)
        
        # Customize subplot
        ax.set_xlabel('Configuration', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio', fontsize=10, fontweight='bold')
        ax.set_title(f'{asset}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(config_labels, rotation=0, ha='center', fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best configuration
        best_idx = np.argmax(vol_targeted_sharpe)
        ax.axvline(x=best_idx, color='gold', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(best_idx, max(vol_targeted_sharpe) * 1.1, 'BEST', 
               ha='center', fontsize=9, fontweight='bold', color='orange')
    
    plt.tight_layout()
    plt.savefig('results/configuration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table():
    """Crea tabella riassuntiva per LaTeX"""
    df = load_risk_data()
    
    # Best configuration per asset
    assets = df['Asset'].unique()
    summary_data = []
    
    for asset in assets:
        asset_data = df[df['Asset'] == asset]
        
        # Find best configuration (highest Vol_Targeted_Sharpe)
        best_config = asset_data.loc[asset_data['Vol_Targeted_Sharpe'].idxmax()]
        
        # Calculate averages for comparison
        avg_baseline_sharpe = asset_data['Baseline_Sharpe'].mean()
        avg_vol_targeted_sharpe = asset_data['Vol_Targeted_Sharpe'].mean()
        avg_baseline_maxdd = asset_data['Baseline_MaxDD'].mean()
        avg_vol_targeted_maxdd = asset_data['Vol_Targeted_MaxDD'].mean()
        
        summary_data.append({
            'Asset': asset,
            'Best_Config': best_config['Config'],
            'Best_Stop_Type': best_config['Stop_Type'],
            'Best_Vol_Return': best_config['Vol_Targeted_Return'],
            'Best_Vol_Sharpe': best_config['Vol_Targeted_Sharpe'],
            'Best_Vol_MaxDD': best_config['Vol_Targeted_MaxDD'],
            'Avg_Baseline_Sharpe': avg_baseline_sharpe,
            'Avg_Vol_Sharpe': avg_vol_targeted_sharpe,
            'Sharpe_Improvement': avg_vol_targeted_sharpe - avg_baseline_sharpe,
            'MaxDD_Improvement': (avg_baseline_maxdd - avg_vol_targeted_maxdd) * 100  # pp improvement
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("=== RISK MANAGEMENT SUMMARY TABLE ===")
    print("\nLatex Table Data:")
    print("Asset & Best Config & Stop Type & Vol Return & Vol Sharpe & Sharpe Δ & MaxDD Improve \\\\")
    print("\\hline")
    
    for _, row in summary_df.iterrows():
        print(f"{row['Asset']} & {row['Best_Config']} & {row['Best_Stop_Type']} & "
              f"{row['Best_Vol_Return']:.1%} & {row['Best_Vol_Sharpe']:.3f} & "
              f"+{row['Sharpe_Improvement']:.3f} & +{row['MaxDD_Improvement']:.1f}pp \\\\")
    
    return summary_df

if __name__ == "__main__":
    print("=== RISK MANAGEMENT ANALYSIS CHARTS ===")
    
    # Create charts
    print("1. Creating Risk-Return Scatter Plot...")
    asset_summary = create_risk_return_scatter()
    
    print("\n2. Creating Configuration Comparison...")
    create_configuration_comparison()
    
    print("\n3. Creating Summary Table...")
    summary_df = create_summary_table()
    
    print(f"\nCharts saved to results/")
    print("- risk_return_scatter.png")
    print("- configuration_comparison.png")