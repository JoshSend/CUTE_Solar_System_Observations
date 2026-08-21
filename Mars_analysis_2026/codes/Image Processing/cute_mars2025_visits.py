'''
Driver for the NASA CUTE (Colorado Ultraviolet Transit Experiment)
data reduction of 2025 Mars Observations in Near Ultraviolet
utilizing image processing logic from cute_mars2025.py
@Author: jose5987
Date Created: 8/20/2026
'''

import os
from cute_mars2025 import CuteReference, CuteVisit, CuteMovie

'''
Fits files live in ...\CUTE_observations\<VisitN>.
edit OBS_DIR if your observation folders are somewhere else
'''
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OBS_DIR = os.path.join(CODE_DIR, '..', '..', 'CUTE_observations')

# --------------------
'''
Mode options:
    MODE = 'still' or 'gif'
        'gif'       : produces a movie of all fits files in frame-id order
        'still'     : produces plots for single fits files

View options:
    VIEW = 'summary' or 'spectrum' or 'regions'
        'summary'   : region-check image (top) + 1D spectrum (bottom)
        'spectrum'  : only 1D spectrum
        'regions'   : only region-check image
'''

MODE = 'gif'

VIEW = 'summary'

# --------------------

'''
Used when MODE == 'still'
label -> file_path (relative to OBS_DIR)
Add one line per frame to inspect
'''
STILLS = {
    'Visit1 frmid 5029':
        os.path.join('Visit1',
                     'cute_TRIM2D_scan_targetID340_2024_12_22_03_49_frmid_3726_V2' \
                     '_nimgpkts_444_L1id20927_botrows_24_midrows_56.fits'),
}

VISITS = ['Visit2']  # ex. ['Visit1', 'Visit2', 'Visit3']

def run_stills(reference):
    '''
    Show a still figure for each individual file in STILLS.
    '''
    for label, relpath in STILLS.items():
        path = os.path.join(OBS_DIR, relpath)
        print(f'Showing {label} ({VIEW}): {path}')
        visit = CuteVisit(path, reference)
        visit.process()
 
        if VIEW == 'summary':
            visit.show_summary(title=label)
        elif VIEW == 'spectrum':
            visit.show_spectrum(title=f'{label} NUV Spectra with CUTE')
        elif VIEW == 'regions':
            visit.plot_regions(title=label)
        else:
            raise ValueError(
                f"VIEW must be 'summary', 'spectrum', or 'regions', "
                f"not {VIEW!r}")
 
 
def run_movies(reference):
    '''
    Build one animated GIF per visit folder in VISITS.
    '''
    for label in VISITS:
        folder = os.path.join(OBS_DIR, label)
        print(f'{label}: building movie from {folder}')
        movie = CuteMovie(folder, reference)
        movie.make_gif(os.path.join(CODE_DIR, f'{label}.gif'),
                       fps=5, view=VIEW)

# ====================================================

def main():
    # load wavelength + effective area 
    reference = CuteReference()

    if MODE == 'still':
        run_stills(reference)
    elif MODE == 'gif':
        run_movies(reference)
    else:
        raise ValueError(f"MODE must be 'still' or 'gif' not {MODE!r}")

if __name__ == '__main__':
    main()   