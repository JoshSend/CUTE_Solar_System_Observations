"""
Unusual dependency: Seaborn
I just like Seaborn, it's super easy to make nice quality plots handling pandas dataframes.
And the 'set_context' feature makes adjusting the same plot for paper, poster,
and presentation super easy.

I got a little carried away and made a few plots already.
Of course feel free to do whatever you wish.

Created: September 10th, 2026
@author: jose5987
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Base Directory
HERE = Path(__file__).parent
OUT = HERE # leaving unchanged so outputs are sent to this folder. Change if needed

# CSV Files, assuming they're in the same folder as HERE
errors_csv = HERE / 'CUTE_mars_attitude_error.csv'
summary_csv = HERE / 'CUTE_mars_attitude_error_summary.csv'

# 'errors' or 'summary'
MODE = "errors"
SAVE = None

# ==================================
sns.set_context('paper')

if SAVE:
    save_path = OUT
else:
    save_path = None

def plot_errors(csv_path: str, save_path: str = None):
    df = pd.read_csv(csv_path)
    
    # Ensure visit is treated as a distinct categorical variable for discrete line styling
    df['visit_label'] = df['visit'].astype(str)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # Panel 1: Intra-visit Offset Drift over Time
    sns.lineplot(
        data=df, 
        x='t_min', 
        y='offset_arcsec',
        estimator='mean', 
        hue='visit_label', 
        style='visit_label',
        markers=True, 
        dashes=False, 
        linewidth=1.5,
        ax=ax1, 
        alpha=0.85
    )
    ax1.set_title('Offset Evolution Over Time per Visit', fontsize=12, pad=10)
    ax1.set_xlabel('Elapsed Time in Visit (min)')
    ax1.set_ylabel('Total Offset (arcsec)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(title='Visit', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Panel 2: 2D Slit Plane Projection (Across vs. Along Slit)
    sns.scatterplot(
        data=df, 
        x='offset_across_slit_arcsec', 
        y='offset_along_slit_arcsec', 
        hue='visit_label', 
        style='visit_label',
        ax=ax2, 
        alpha=0.8, 
        s=50
    )
    # Target center marker (0, 0)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax2.set_title('2D Pointing Alignment Relative to Slit Center (0,0)', fontsize=12, pad=10)
    ax2.set_xlabel('Across-Slit Offset (arcsec)')
    ax2.set_ylabel('Along-Slit Offset (arcsec)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(title='Visit', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()

def plot_summary(csv_path: str, save_path: str = None):
    df = pd.read_csv(csv_path)
    x = df['visit'].astype(str)
    
    plt.figure(figsize=(10, 5))
    
    # Line plot connecting mean offsets per visit
    plt.plot(x, df['mean_offset_arcsec'], marker='o', color='tab:blue', label='Mean Offset', linewidth=2)
    
    # Error bars for standard deviation
    plt.errorbar(x, df['mean_offset_arcsec'], 
                 yerr=df['std_offset_arcsec'], fmt='none', ecolor='tab:blue',
                 capsize=4, label='± Std Dev')
    
    # Shaded region showing full min-to-max range within each visit
    plt.fill_between(x, df['min_offset_arcsec'], df['max_offset_arcsec'], color='tab:red', alpha=0.15, label='Min-Max Range')

    plt.yscale('symlog')
    
    plt.ylabel('Offset (arcsec)')
    plt.title('Average CUTE Mars Offset per Visit')
    plt.xticks(ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

# -------------
def main():
    if MODE == 'errors':
        plot_errors(csv_path=errors_csv, save_path=save_path)
    else:
        plot_summary(csv_path=summary_csv, save_path=save_path)

if __name__ == '__main__':
    main()