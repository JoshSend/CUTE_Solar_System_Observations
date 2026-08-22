"""
Driver for image processing of NASA CUTE (Colorado Ultraviolet Transit Experiment)
2025 Mars Observations in Near Ultraviolet.
@Author: jose5987
Date Created: 8/21/2026

Uses processing logic from cute_mars2025.py
"""

import matplotlib.pyplot as plt
from cute_mars2025 import CuteReference, CuteObservation, load_observation

#==============================================
# USER INPUTS
# Two options for output controlled by STATIC boolean:
#   STATIC = True  : One trace region spectra and 1D spectrum 
#                    will be plotted. Requires FILENAME input.
#   STATIC = False : Movie of trace region and 1D spectrum displayed
#                    that goes in order of frameid of all fits files
#                    in a visit folder.

STATIC = False

VISIT = "Visit7"

# if STATIC = True:
#   Input file name str OR specific frame id as an int
FILENAME = 5157
#FILENAME = 'cute_TRIM2D_scan_targetID340_2025_01_12_06_17_frmid_4874_V2_nimgpkts_445_L1id21580_botrows_24_midrows_55.fits'

# WORK ORDER: files can have same frameid but differed midrows.
#==============================================

def main():
    ref = CuteReference()

    if STATIC:
        if FILENAME is None:
            raise ValueError("STATIC mode needs a FILENAME (or set STATIC = False to animate)")
        obs = load_observation(visit=VISIT, filename=FILENAME, reference=ref)
        fig1, ax1 = obs.plot_trace()
        fig2, ax2 = obs.plot_spectrum(box_pts=5, ylim=None)
        plt.show()
    else:
        CuteObservation.animate_visit(visit=VISIT, reference=ref, kind='both', fps=5)

if __name__ == '__main__':
    main()