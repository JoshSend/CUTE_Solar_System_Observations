"""
Driver for image processing of NASA CUTE (Colorado Ultraviolet Transit Experiment)
2025 Mars Observations in Near Ultraviolet.
@Author: jose5987
Date Created: 8/21/2026

Uses processing logic from cute_mars2025.py
"""

import os
import matplotlib.pyplot as plt
from cute_mars2025 import CuteReference, CuteObservation, load_observation, _get_output_dir

#==============================================
# USER INPUTS
# Two options for output controlled by STATIC boolean:
#   STATIC = True  : One trace region spectra and 1D spectrum 
#                    will be plotted. Requires FILENAME input.
#   STATIC = False : Movie of trace region and 1D spectrum displayed
#                    that goes in order of frameid of all fits files
#                    in a visit folder.

# Two options for SAVE boolean:
#   SAVE = True  : saves the files to the 'output' folder
#                  already present in the base directory,
#                  then displays.
#   SAVE = False : only displays the output.

STATIC = True

SAVE = False
output_dir = 'output'

VISIT = "Visit2"
# e.g "Visit2" or "Visit3" or ...

# if STATIC = True:
#   Input file name str OR specific frame id as an int
FILENAME = None
# e.g 4874 or 'cute_TRIM2D_scan_..._frmid_4874_..._midrows_55.fits' 

# WORK ORDER: files can have same frameid but differed midrows.
# WORK ORDER: tracking needs to be able to adjust width and height (currently only height)

#==============================================

def main():
    ref = CuteReference()
    out_path = _get_output_dir(output_dir)

    if STATIC:
        if FILENAME is None:
            raise ValueError("STATIC mode needs a FILENAME (or set STATIC = False to animate)")
        obs = load_observation(visit=VISIT, filename=FILENAME, reference=ref)

        fig1, ax1 = obs.plot_trace()
        fig2, ax2 = obs.plot_spectrum(box_pts=5, ylim=None)

        if SAVE:
            # Files are named based on FITS file
            stem = os.path.splitext(os.path.basename(obs.fits_fname))[0]
            trace_png = os.path.join(out_path, f"{stem}_trace.png")
            spec_png  = os.path.join(out_path, f"{stem}_spectrum.png")
            fig1.savefig(trace_png, dpi=200, bbox_inches='tight')
            fig2.savefig(spec_png,  dpi=200, bbox_inches='tight')
            print(f"Saved:\n  {trace_png}\n  {spec_png}")

        plt.show()
        
    else:
        CuteObservation.animate_visit(
            visit=VISIT, reference=ref, kind='both', fps=5,
            save=SAVE, output_dir=out_path
        )

if __name__ == '__main__':
    main()