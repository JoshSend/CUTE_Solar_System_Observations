"""
Driver for image processing of NASA CUTE (Colorado Ultraviolet Transit Experiment)
2025 Mars Observations in Near Ultraviolet.
@Author: jose5987
Date Created: 8/21/2026

Uses processing logic from cute_mars2025.py
"""

import os
import sys
import matplotlib.pyplot as plt

# cute_mars2025 exists in the "Image Processing" subfolder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Image Processing'))
from cute_mars2025 import CuteReference, CuteObservation, load_observation, _get_output_dir

# ==============================================
# USER INPUTS
#
# MODE selects what to produce (pick one):
#   'static'     : one frame trace plot + 1D spectrum. Needs FILENAME
#   'static_all' : trace + 1D spectrum PNG for EVERY frame of EVERY visit in
#   'overlay'    : every frame of VISIT drawn over each other on one axes,
#                  colored first -> last
#   'overlay_all': one overlay panel per visit in GRID_VISITS, single figure.
#   'visit'      : movie of one visit (trace + spectrum), all frames in frmid order.
#   'visit_all'  : movie of all visits (trace + spectrum)
#   'grid'       : grid movie, one 1D-spectrum panel per visit in GRID_VISITS.
#   'sequence'   : one 1D-spectrum panel that plays every frame of each visit
#                  in turn, Visit1 -> ... -> Visit9.

MODE = "visit_all"

# save figures/GIFs to the output folder
SAVE = True 

# used by 'static' and 'visit':
#   input visit folder name str
VISIT = 'Visit2' # e.g. "Visit2" or "Visit3" or ...

# used by 'static':
#   input file name str OR specific frame id as an int
FILENAME = 4874
# e.g 4874 or 'cute_TRIM2D_scan_..._frmid_4874_..._midrows_55.fits' 

# -----------------------------
# OPTIONAL (more specific) USER INPUTS

# used by 'grid' and 'sequence' (Visit6 skipped: planet not in slit)
GRID_VISITS = ['Visit1', 'Visit2', 'Visit3', 'Visit4',
               'Visit5', 'Visit7', 'Visit8', 'Visit9']

SKIP_FRMID = [] # 'visit', 'grid', 'sequence': skips specified frameids in integer list input
# Mars not in slit in frame 4929

# --- 'static_all' options ---
# 'both' or 'trace' or 'spectrum': what to write per frame
STATIC_ALL_KIND = 'both'

STATIC_ALL_OVERWRITE = True # overwrite pre-existing files

STATIC_ALL_YLIM = (0.0, 3.0) # fixed flux range
STATIC_ALL_VMIN = None # Flux range min
STATIC_ALL_VMAX = None # Flux range max

 
# --- 'overlay' and 'overlay_all' options ---
OVERLAY_PALETTE = None # colormap
OVERLAY_YLIM = None # True = autoscale y-lim

# Boxcar smoothing for 1D spectra
box_pts = 15

# --- output directory ---
output_dir = 'output'
VISIT_SUBDIR = 'Spectra'
ALL_VISITS_SUBDIR = os.path.join('codes', 'Spectra')

# ==============================================
 
def _visit_out(visit):
    """output/Spectra/<Visit>/ -- products belonging to one visit."""
    return _get_output_dir(output_dir, os.path.join(VISIT_SUBDIR, visit))
 
 
def _all_visits_out():
    """output/codes/Spectra/ -- products spanning every visit."""
    return _get_output_dir(output_dir, ALL_VISITS_SUBDIR)
 
 
def main():
    ref = CuteReference()
 
    if MODE == "static":
        if FILENAME is None:
            raise ValueError("MODE 'static' needs a FILENAME")
        out_path = _visit_out(VISIT)
        obs = load_observation(visit=VISIT, filename=FILENAME, reference=ref)
 
        # no vmin/vmax -> plot_trace stretches this frame between its own
        # 5th and 99th percentile, so the trace reads against a dark field
        fig1, ax1 = obs.plot_trace()
        fig2, ax2 = obs.plot_spectrum(box_pts=box_pts,
                                      ylim=None)
 
        if SAVE:
            stem = os.path.splitext(os.path.basename(obs.fits_fname))[0]
            trace_png = os.path.join(out_path, f"{stem}_trace.png")
            spec_png  = os.path.join(out_path, f"{stem}_spectrum.png")
            fig1.savefig(trace_png, dpi=200, bbox_inches='tight')
            fig2.savefig(spec_png,  dpi=200, bbox_inches='tight')
            print(f"Saved:\n  {trace_png}\n  {spec_png}")
 
        plt.show()
 
    elif MODE == "static_all":
        # every frame of every visit, written straight to disk -- nothing to
        # type in, nothing shown on screen. save_all_frames makes the
        # per-visit subfolders under output/Spectra/ itself.
        out_root = _get_output_dir(output_dir, VISIT_SUBDIR)   # output/Spectra/
        CuteObservation.save_all_frames(
            GRID_VISITS, reference=ref, output_root=out_root,
            kind=STATIC_ALL_KIND, box_pts=box_pts,
            ylim=STATIC_ALL_YLIM,
            vmin=STATIC_ALL_VMIN, vmax=STATIC_ALL_VMAX,
            skip_frmid=SKIP_FRMID, overwrite=STATIC_ALL_OVERWRITE
        )
 
    elif MODE == "overlay":
        out_path = _visit_out(VISIT)
        CuteObservation.plot_visit_overlay(
            VISIT, reference=ref, box_pts=box_pts,
            ylim=OVERLAY_YLIM, palette=OVERLAY_PALETTE,
            skip_frmid=SKIP_FRMID,
            save=SAVE, output_dir=out_path, show=True
        )
 
    elif MODE == "overlay_all":
        out_path = _all_visits_out()
        CuteObservation.plot_overlay_grid(
            GRID_VISITS, reference=ref, box_pts=box_pts,
            ylim=OVERLAY_YLIM, palette=OVERLAY_PALETTE,
            skip_frmid=SKIP_FRMID,
            save=SAVE, output_dir=out_path
        )
 
    elif MODE == "visit":
        out_path = _visit_out(VISIT)
        CuteObservation.animate_visit(
            visit=VISIT, reference=ref, kind='both', fps=5,
            save=SAVE, output_dir=out_path, skip_frmid=SKIP_FRMID,
            box_pts=box_pts
        )
 
    elif MODE == "visit_all":
        # both GIFs for every visit, straight to disk -- animate_all_visits
        # makes the per-visit subfolders under output/Spectra/ itself.
        out_root = _get_output_dir(output_dir, VISIT_SUBDIR)   # output/Spectra/
        CuteObservation.animate_all_visits(
            GRID_VISITS, reference=ref, output_root=out_root,
            kind='both', fps=5, box_pts=box_pts, skip_frmid=SKIP_FRMID
        )
 
    elif MODE == "grid":
        out_path = _all_visits_out()
        CuteObservation.animate_grid(
            GRID_VISITS, reference=ref, fps=5,
            save=SAVE, output_dir=out_path,
            skip_frmid=SKIP_FRMID
        )
 
    elif MODE == "sequence":
        out_path = _all_visits_out()
        CuteObservation.animate_sequence(
            GRID_VISITS, reference=ref, fps=5,
            save=SAVE, output_dir=out_path,
            skip_frmid=SKIP_FRMID
        )
 
    else:
        raise ValueError(
            "MODE must be 'static', 'static_all', 'overlay', 'overlay_all', "
            f"'visit', 'visit_all', 'grid', or 'sequence', not {MODE!r}"
        )
 
 
if __name__ == '__main__':
    main()