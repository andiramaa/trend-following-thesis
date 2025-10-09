"""
Portfolio Analysis Charts per Sezione 4.3
Visualizza performance dei portafogli vs benchmark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_portfolio_data():
    """Carica tutti i dati dei portafogli"""
    # Load summary
    summary_df = pd.read_csv('data/portfolios/portfolio_summary.csv')
    
    # Load detailed data
    eq_weighted = pd.read_csv('data/portfolios/equal_weighted_portfolio.csv', 
                             index_col=0, parse_dates=True)
    risk_parity = pd.read_csv('data/portfolios/risk_parity_portfolio.csv', 
                             index_col=0, parse_dates=True)
    benchmarks = pd.read_csv('data/portfolios/benchmarks_portfolio.csv', 
                           index_col=0, parse_dates=True)
    
    return summary_df, eq_weighted, risk_parity, benchmarks

def create_performance_comparison_chart():
    """Crea grafico comparativo delle performance"""
    summary_df, _, _, _ = load_portfolio_data()
    
    # Prepare data for plotting
    portfolios = []
    returns = []
    volatilities = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for _, row in summary_df.iterrows():
        portfolios.append(row['Portfolio'])
        returns.append(row['Annualized_Return'] * 100)
        volatilities.append(row['Volatility'] * 100)  
        sharpe_ratios.append(row['Sharpe_Ratio'])
        max_drawdowns.append(abs(row['Max_Drawdown'] * 100))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Portfolio Construction Analysis: Performance Comparison', fontsize=16, fontweight='bold')
    
    # Clean portfolio names for display
    clean_names = [name.replace('_', ' ').title() for name in portfolios]
    colors = ['skyblue', 'lightgreen', 'orange', 'red', 'purple', 'brown']
    
    # 1. Annualized Returns
    bars1 = ax1.bar(range(len(portfolios)), returns, color=colors[:len(portfolios)], alpha=0.8)
    ax1.set_title('Annualized Returns (%)', fontweight='bold')
    ax1.set_xticks(range(len(portfolios)))
    ax1.set_xticklabels(clean_names, rotation=45, ha='right')
    ax1.set_ylabel('Return (%)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, returns):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # 2. Sharpe Ratios
    bars2 = ax2.bar(range(len(portfolios)), sharpe_ratios, color=colors[:len(portfolios)], alpha=0.8)
    ax2.set_title('Sharpe Ratios', fontweight='bold')
    ax2.set_xticks(range(len(portfolios)))
    ax2.set_xticklabels(clean_names, rotation=45, ha='right')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, sharpe_ratios):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # 3. Volatility
    bars3 = ax3.bar(range(len(portfolios)), volatilities, color=colors[:len(portfolios)], alpha=0.8)
    ax3.set_title('Volatility (%)', fontweight='bold')
    ax3.set_xticks(range(len(portfolios)))
    ax3.set_xticklabels(clean_names, rotation=45, ha='right')
    ax3.set_ylabel('Volatility (%)')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, volatilities):
        ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # 4. Maximum Drawdown
    bars4 = ax4.bar(range(len(portfolios)), max_drawdowns, color=colors[:len(portfolios)], alpha=0.8)
    ax4.set_title('Maximum Drawdown (%)', fontweight='bold')
    ax4.set_xticks(range(len(portfolios)))
    ax4.set_xticklabels(clean_names, rotation=45, ha='right')
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars4, max_drawdowns):
        ax4.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/portfolio_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_risk_return_scatter_portfolio():
    """Crea scatter plot risk-return dei portafogli"""
    summary_df, _, _, _ = load_portfolio_data()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    returns = []
    volatilities = []
    labels = []
    colors = []
    
    color_map = {
        'equal_weighted': 'blue',
        'risk_parity': 'green', 
        'SP500': 'red',
        '60_40': 'orange'
    }
    
    for _, row in summary_df.iterrows():
        returns.append(row['Annualized_Return'] * 100)
        volatilities.append(row['Volatility'] * 100)
        
        # Clean label and determine color
        if 'equal_weighted' in row['Portfolio']:
            label = 'Equal Weight' + (' (Vol-Targeted)' if 'vol_targeted' in row['Portfolio'] else ' (Base)')
            color = 'lightblue' if 'vol_targeted' in row['Portfolio'] else 'blue'
        elif 'risk_parity' in row['Portfolio']:
            label = 'Risk Parity' + (' (Vol-Targeted)' if 'vol_targeted' in row['Portfolio'] else ' (Base)')
            color = 'lightgreen' if 'vol_targeted' in row['Portfolio'] else 'green'
        elif 'SP500' in row['Portfolio']:
            label = 'S&P 500'
            color = 'red'
        elif '60_40' in row['Portfolio']:
            label = '60/40 Portfolio'
            color = 'orange'
        else:
            label = row['Portfolio']
            color = 'gray'
            
        labels.append(label)
        colors.append(color)
    
    # Create scatter plot
    scatter = ax.scatter(volatilities, returns, c=colors, s=120, alpha=0.8, edgecolors='black')
    
    # Add labels for each point
    for i, (vol, ret, label) in enumerate(zip(volatilities, returns, labels)):
        ax.annotate(label, (vol, ret), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Risk-Return Profile\n(Higher Return, Lower Risk = Better)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add efficient frontier line (conceptual)
    x_line = np.linspace(min(volatilities), max(volatilities), 100)
    y_line = np.sqrt(x_line) * 2  # Simple curve for illustration
    ax.plot(x_line, y_line, 'k--', alpha=0.3, label='Efficient Frontier (Conceptual)')
    
    plt.tight_layout()
    plt.savefig('results/portfolio_risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_equity_curves_comparison():
    """Crea confronto delle equity curves"""
    summary_df, eq_weighted, risk_parity, benchmarks = load_portfolio_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot equity curves
    if 'Equity_Vol_Targeted' in eq_weighted.columns:
        ax.plot(eq_weighted.index, eq_weighted['Equity_Vol_Targeted'], 
               label='Equal-Weighted Portfolio', color='blue', linewidth=2)
    
    if 'Equity_Vol_Targeted' in risk_parity.columns:
        ax.plot(risk_parity.index, risk_parity['Equity_Vol_Targeted'], 
               label='Risk-Parity Portfolio', color='green', linewidth=2)
    
    if 'Equity_SP500' in benchmarks.columns:
        ax.plot(benchmarks.index, benchmarks['Equity_SP500'], 
               label='S&P 500', color='red', linewidth=2, alpha=0.7)
    
    if 'Equity_6040' in benchmarks.columns:
        ax.plot(benchmarks.index, benchmarks['Equity_6040'], 
               label='60/40 Portfolio', color='orange', linewidth=2, alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (Normalized to 1)', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Performance Comparison: Equity Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to better see relative performance
    
    plt.tight_layout()
    plt.savefig('results/portfolio_equity_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_latex_table():
    """Stampa tabella LaTeX formatted"""
    summary_df, _, _, _ = load_portfolio_data()
    
    print("=== PORTFOLIO PERFORMANCE TABLE FOR LATEX ===")
    print("Portfolio & Ann.Return & Volatility & Sharpe & Sortino & Max DD \\\\")
    print("\\hline")
    
    for _, row in summary_df.iterrows():
        name = row['Portfolio'].replace('_', '\\_')
        ann_ret = row['Annualized_Return'] * 100
        vol = row['Volatility'] * 100
        sharpe = row['Sharpe_Ratio']
        sortino = row['Sortino_Ratio'] 
        max_dd = row['Max_Drawdown'] * 100
        
        print(f"{name} & {ann_ret:.1f}\\% & {vol:.1f}\\% & {sharpe:.3f} & "
              f"{sortino:.3f} & {max_dd:.1f}\\% \\\\")

if __name__ == "__main__":
    print("=== PORTFOLIO ANALYSIS CHARTS ===")
    
    try:
        # Create all charts
        print("1. Creating performance comparison chart...")
        create_performance_comparison_chart()
        
        print("2. Creating risk-return scatter...")
        create_risk_return_scatter_portfolio()
        
        print("3. Creating equity curves comparison...")
        create_equity_curves_comparison()
        
        print("4. Generating LaTeX table...")
        print_latex_table()
        
        print(f"\nCharts saved to results/:")
        print("- portfolio_performance_comparison.png")
        print("- portfolio_risk_return_scatter.png") 
        print("- portfolio_equity_curves.png")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure portfolio data files exist in data/portfolios/")