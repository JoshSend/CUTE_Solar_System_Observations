"""
Helper for comparing flux data in CSV files using Pandas

Created Sept. 24th, 2026
@author: jose5987
"""

import re
from pathlib import Path
from glob import glob
import pandas as pd

# Helpers
def _get_spectra_dir()->Path: # gets directory for Mars spectra
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    spectra_dir = base_dir.parent / 'Mars_analysis_2026' / 'codes' / 'output' / 'Spectra'

    return spectra_dir

def load_csv(spectra_dir: Path)-> pd.DataFrame:
    all_dfs = []

    csv_paths = list(spectra_dir.glob('Visit*/csv/*.csv'))

    if not csv_paths:
        print(f'No CSV files under {spectra_dir}')
        return pd.DataFrame()

    for path in csv_paths:
        df = pd.read_csv(path)
        df['visit'] = path.parents[1].name
        df['source_file'] = path.name

        match = re.search(r'_frmid_(\d+)_', path.name)
        df['frmid'] = int(match.group(1)) if match else None

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def max_flux_per_visit(df)-> pd.DataFrame:
    """
    Determines frmid with maximum flux in a visit
    """
    idx_max_flux = df.groupby('visit')['flux'].idxmax()

    summary = (
        df.loc[idx_max_flux, ['visit', 'frmid', 'flux']]
        .sort_values(by='visit')
        .reset_index(drop=True)
    )
    return summary

def main():
    spectra_dir = _get_spectra_dir()
    print(f'Reading CSVs from {spectra_dir}')

    df = load_csv(spectra_dir)
    if df.empty:
        return

    max_flux_df = max_flux_per_visit(df)
    print(max_flux_df.to_string(index=False))

if __name__ == '__main__':
    main()