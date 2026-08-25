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

# ==============================================
# USER INPUTS
#
# MODE selects what to produce (pick one):
#   'static'   : one frame trace plot + 1D spectrum. Needs FILENAME
#   'visit'    : movie of one visit (trace + spectrum), all frames in frmid order.
#   'grid'     : grid movie, one 1D-spectrum panel per visit in GRID_VISITS.
#   'sequence' : one 1D-spectrum panel that plays every frame of each visit
#                in turn, Visit1 -> ... -> Visit9.

MODE = "grid"

# used by 'static' and 'visit':
#   input visit folder name str
VISIT = "Visit9"
# e.g "Visit2" or "Visit3" or ...

# used by 'static':
#   input file name str OR specific frame id as an int
FILENAME = 4872
# e.g 4874 or 'cute_TRIM2D_scan_..._frmid_4874_..._midrows_55.fits' 

# used by 'grid' and 'sequence' (Visit6 skipped: planet not in slit)
GRID_VISITS = ['Visit1', 'Visit2', 'Visit3', 'Visit4',
               'Visit5', 'Visit7', 'Visit8', 'Visit9']

# used by 'visit', 'grid', and 'sequence':
#   skips specified frameids in integer list input
SKIP_FRMID = [4929]
# e.g [4861, 4874, ...]
# List of frames where Mars is not in slit:
#       4929, 

# save figures/GIFs to the output folder
SAVE = False
output_dir = 'output'

# WORK ORDER: 'static' mode opening animated grid (RESOLVED in isolated use case, TESTING NEEDED)
# WORK ORDER: implement and adjust effective area multiplication factor (from 2024 to 2025)
# ==============================================

def main():
    ref = CuteReference()

    if MODE == "static":
        if FILENAME is None:
            raise ValueError("MODE 'static' needs a FILENAME")
        out_path = _get_output_dir(output_dir, VISIT)     # output/<VISIT>/
        obs = load_observation(visit=VISIT, filename=FILENAME, reference=ref)

        fig1, ax1 = obs.plot_trace()
        fig2, ax2 = obs.plot_spectrum(box_pts=5, ylim=None)

        if SAVE:
            stem = os.path.splitext(os.path.basename(obs.fits_fname))[0]
            trace_png = os.path.join(out_path, f"{stem}_trace.png")
            spec_png  = os.path.join(out_path, f"{stem}_spectrum.png")
            fig1.savefig(trace_png, dpi=200, bbox_inches='tight')
            fig2.savefig(spec_png,  dpi=200, bbox_inches='tight')
            print(f"Saved:\n  {trace_png}\n  {spec_png}")

        plt.show()

    elif MODE == "visit":
        out_path = _get_output_dir(output_dir, VISIT)
        CuteObservation.animate_visit(
            visit=VISIT, reference=ref, kind='both', fps=5,
            save=SAVE, output_dir=out_path, skip_frmid=SKIP_FRMID
        )

    elif MODE == "grid":
        out_path = _get_output_dir(output_dir)
        CuteObservation.animate_grid(
            GRID_VISITS, reference=ref, fps=5,
            save=SAVE, output_dir=out_path,
            skip_frmid=SKIP_FRMID
        )

    elif MODE == "sequence":
        out_path = _get_output_dir(output_dir)
        CuteObservation.animate_sequence(
            GRID_VISITS, reference=ref, fps=5,
            save=SAVE, output_dir=out_path,
            skip_frmid=SKIP_FRMID
        )

    else:
        raise ValueError(
            f"MODE must be 'static', 'visit', 'grid', or 'sequence', not {MODE!r}"
        )


if __name__ == '__main__':
    main()