"""
Driver for image processing of NASA CUTE (Colorado Ultraviolet Transit Experiment)
2025 Mars Observations in Near Ultraviolet.
@Author: jose5987
Date Created: 8/21/2026

Uses processing logic from cute_mars2025.py
"""

import os
import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd

# cute_mars2025 exists in the "Image Processing" subfolder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Image Processing'))
from cute_mars2025 import CuteReference, CuteObservation, load_observation, _get_output_dir, slit_fill_fraction

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
#   'bright'     : Brightest frame plotting
#   'bright_all' : 
#   
#   'solar'      : Solar overlay plotting
#   'solar_all'  : 
#   'solar_grid' :
#   'reflect_all': Cross-visit solar reflectance
#   'solar_diff' :
#   'cutevsolar' :

MODE = "cutevsolar"

# save figures/GIFs to the output folder
SAVE = False

FIT = True # Fit spectra to the brightest frame in each visit
FIT_METHOD = 'linear' # 'scale' | 'linear' | 'ratio' | 'median'
FIT_WEIGHTED = False

# used by 'static' and 'visit':
#   input visit folder name str
VISIT = 'Visit3' # e.g. "Visit2" or "Visit3" or ...

# used by 'static':
#   input file name str OR specific frame id as an int
FILENAME = 4923
# e.g 4874 or 'cute_TRIM2D_scan_..._frmid_4874_..._midrows_55.fits'
 
# -----------------------------
# OPTIONAL (more specific) USER INPUTS

# used by 'grid' and 'sequence' (Visit6 skipped: planet not in slit)
GRID_VISITS = ['Visit1', 'Visit2', 'Visit3', 'Visit4',
               'Visit5', 'Visit7', 'Visit8', 'Visit9']

SKIP_FRMID = [4929] # 'visit', 'grid', 'sequence': skips specified frameids in integer list input
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

# --- 'solar' and 'solar_all' options ---
SOLAR_SCALE = 'auto'        # 'auto' or a float
SOLAR_MODE  = 'reflectance'
# common      : convert to same units on log axis
# twin        : each in their own units
# scaled      : visual, scaled so that their magnitudes are the same. Just to see shapes, not physically relevant.
# overlay     : both in same units, with reflected solar spectrum
# reflectance : F_CUTE / F_reflected solar (solar-removed product)

# --- random ---
box_pts = 15 # Boxcar smoothing for 1D spectra
fps = 2 # Frames per second for gifs.
CuteObservation.SCI_HALF = None # Half of science trace region. 
                                # int for specified, none for adaptive.
                                # recommended 8, but x<14

# --- output directory ---
output_dir = 'output'
VISIT_SUBDIR = 'Spectra'
ALL_VISITS_SUBDIR = 'Spectra'
CORR_SUBDIR = 'Correlation'

if FIT:
    VISIT_SUBDIR = ALL_VISITS_SUBDIR = 'Spectra_fit'

# --- solar overlay directories ---
# Solar_spectra lives one level up from codes:  Mars_analysis_2026\Solar_spectra
_DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))        # ...\codes
SOLAR_DIR   = os.path.abspath(os.path.join(_DRIVER_DIR, '..', 'Solar_spectra'))

# ==============================================

# brightest frame per visit
BRIGHTEST = {
    'Visit1': 3739, 'Visit2': 4860, 'Visit3': 4926, 'Visit4': 4975,
    'Visit5': 5030, 'Visit7': 5145, 'Visit8': 5209, 'Visit9': 5260,
}

# Mars-Earth distance in AU
MARS_EARTH_AU = {
    'Visit1': 0.69, 'Visit2': 0.64, 'Visit3': 0.65, 'Visit4': 0.67,
    'Visit5': 0.69, 'Visit7': 0.77, 'Visit8': 0.87, 'Visit9': 1.12,
}

PHASE_ANGLE = {
    'Visit1': 20.05, 'Visit2': 4.22, 'Visit3': 3.72, 'Visit4': 9.56,
    'Visit5': 14.75, 'Visit7': 23.45, 'Visit8': 29.26, 'Visit9': 35.72,
}

ATTITUDE_CSV = os.path.join(_DRIVER_DIR, 'output', 'Slit_Overlay',
                            'CUTE_mars_attitude_error.csv')

def _f_slit_for(visit, frmid):
    """Slit-filling fraction for one visit's brightest frame, from the attitude CSV."""
    df = pd.read_csv(ATTITUDE_CSV)
    row = df[(df['visit'] == visit) & (df['frmid'] == int(frmid))]
    if row.empty:
        print(f"  {visit}: frmid {frmid} not in attitude CSV -- f_slit=1")
        return 1.0
    across = float(row['offset_across_slit_arcsec'].iloc[0])
    along  = float(row['offset_along_slit_arcsec'].iloc[0])
    return slit_fill_fraction(across, along, MARS_EARTH_AU[visit])

def _brightness_csv(visit):
    return _get_output_dir(output_dir,
                           os.path.join('Spectra', visit, 'csv'))
 
def _visit_out(visit):
    """output/Spectra/<Visit>/ -- products belonging to one visit."""
    return _get_output_dir(output_dir, os.path.join(VISIT_SUBDIR, visit))
 
 
def _all_visits_out():
    """output/codes/Spectra/ -- products spanning every visit."""
    return _get_output_dir(output_dir, ALL_VISITS_SUBDIR)

def _correlation_out(visit):
    return _get_output_dir(os.path.join(output_dir, 'Correlation'), visit)
 
 
def main():
    CuteObservation.FIT = FIT
    CuteObservation.FIT_TO = BRIGHTEST if FIT else None
    CuteObservation.FIT_METHOD = FIT_METHOD
    CuteObservation.FIT_WEIGHTED = FIT_WEIGHTED

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
            tag = CuteObservation.fit_tag()
            trace_png = os.path.join(out_path, f"{stem}_trace{tag}.png")
            spec_png  = os.path.join(out_path, f"{stem}_spectrum{tag}.png")
            fig1.savefig(trace_png, dpi=200, bbox_inches='tight')
            fig2.savefig(spec_png,  dpi=200, bbox_inches='tight')
            csv_path = obs.save_spectrum_csv(os.path.join(out_path, 'csv'))
            print(f"  {csv_path}")
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
            visit=VISIT, reference=ref, kind='both', fps=fps,
            save=SAVE, output_dir=out_path, skip_frmid=SKIP_FRMID,
            box_pts=box_pts
        )
 
    elif MODE == "visit_all":
        # both GIFs for every visit, straight to disk -- animate_all_visits
        # makes the per-visit subfolders under output/Spectra/ itself.
        out_root = _get_output_dir(output_dir, VISIT_SUBDIR)   # output/Spectra/
        CuteObservation.animate_all_visits(
            GRID_VISITS, reference=ref, output_root=out_root,
            kind='both', fps=fps, box_pts=box_pts, skip_frmid=SKIP_FRMID
        )
 
    elif MODE == "grid":
        out_path = _all_visits_out()
        CuteObservation.animate_grid(
            GRID_VISITS, reference=ref, fps=fps,
            save=SAVE, output_dir=out_path,
            skip_frmid=SKIP_FRMID
        )
 
    elif MODE == "sequence":
        out_path = _all_visits_out()
        CuteObservation.animate_sequence(
            GRID_VISITS, reference=ref, fps=fps,
            save=SAVE, output_dir=out_path,
            skip_frmid=SKIP_FRMID
        )

    elif MODE == 'bright':
        csv_dir = _brightness_csv(VISIT)
        out = _correlation_out(VISIT) if SAVE else None
        CuteObservation.plot_brightness_correlation(
            VISIT, BRIGHTEST[VISIT], csv_dir,
            save=SAVE, output_dir=out, show=not SAVE or True)

    elif MODE == 'bright_all':
        for v, fid in BRIGHTEST.items():
            csv_dir = _brightness_csv(v)
            out = _correlation_out(v) if SAVE else None
            CuteObservation.plot_brightness_correlation(
                v, fid, csv_dir, save=SAVE, output_dir=out, show=not SAVE)

    elif MODE == 'solar':
        csv_dir = _brightness_csv(VISIT)
        hits = glob.glob(os.path.join(csv_dir, f"*frmid_{BRIGHTEST[VISIT]}*.csv"))
        if not hits:
            raise FileNotFoundError(f"no CUTE CSV for frmid {BRIGHTEST[VISIT]} in {csv_dir}")
        solar_file = os.path.join(SOLAR_DIR, f"{VISIT.lower()}_solar_spectrum.csv")
        if not os.path.exists(solar_file):
            raise FileNotFoundError(f"no solar file: {solar_file}")
        out = _get_output_dir(os.path.join(output_dir, 'solar_overlay')) if SAVE else None
        CuteObservation.plot_mars_solar_overlay(
            hits[0], solar_file, mars_earth_au=MARS_EARTH_AU[VISIT], mode=SOLAR_MODE,
            title=f"{VISIT} frmid {BRIGHTEST[VISIT]}  -  Mars vs Solar Spectrum (TSIS)",
            save=SAVE, output_dir=out, phase_angle=PHASE_ANGLE[VISIT],
            fname=f"{VISIT.lower()}_solar_overlay.png", show=not SAVE)

    elif MODE == 'solar_all':
        for v, fid in BRIGHTEST.items():
            csv_dir = _brightness_csv(v)
            hits = glob.glob(os.path.join(csv_dir, f"*frmid_{fid}*.csv"))
            solar_file = os.path.join(SOLAR_DIR, f"{v.lower()}_solar_spectrum.csv")
            if not hits or not os.path.exists(solar_file):
                print(f"  {v}: missing CUTE CSV or solar file, skipping")
                continue
            out = _get_output_dir(os.path.join(output_dir, 'solar_overlay')) if SAVE else None
            CuteObservation.plot_mars_solar_overlay(
                hits[0], solar_file, mars_earth_au=MARS_EARTH_AU[v], mode=SOLAR_MODE,
                title=f"{v} frmid {fid}  -  Mars vs Solar Spectrum (TSIS)",
                phase_angle=PHASE_ANGLE[v],
                save=SAVE, output_dir=out,
                fname=f"{v.lower()}_solar_overlay.png", show=not SAVE)

    elif MODE == 'reflect_all':
        items = []
        for v, fid in BRIGHTEST.items():
            hits = glob.glob(os.path.join(_brightness_csv(v), f"*frmid_{fid}*.csv"))
            sf = os.path.join(SOLAR_DIR, f"{v.lower()}_solar_spectrum.csv")
            if hits and os.path.exists(sf):
                items.append((v, hits[0], sf, MARS_EARTH_AU[v]))
            else:
                print(f"  {v}: missing CUTE CSV or solar file, skipping")
        out = _get_output_dir(os.path.join(output_dir, 'solar_overlay')) if SAVE else None
        CuteObservation.plot_reflectance_all(items, save=SAVE, output_dir=out, show=not SAVE)

    elif MODE == 'solar_grid':
        items = []
        for v, fid in BRIGHTEST.items():
            hits = glob.glob(os.path.join(_brightness_csv(v), f"*frmid_{fid}*.csv"))
            sf = os.path.join(SOLAR_DIR, f"{v.lower()}_solar_spectrum.csv")
            if hits and os.path.exists(sf):
                fsl = _f_slit_for(v, fid)
                print(f"  {v}: f_slit = {fsl:.3f}")
                items.append((v, hits[0], sf, MARS_EARTH_AU[v], PHASE_ANGLE[v], fsl))
            else:
                print(f"  {v}: missing CUTE CSV or solar file, skipping")
        out = _get_output_dir(os.path.join(output_dir, 'solar_overlay')) if SAVE else None
        CuteObservation.plot_reflectance_grid(items, save=SAVE, output_dir=out, show=not SAVE, ymax=0.3)

    elif MODE == 'solar_diff':
        items = []
        for v, fid in BRIGHTEST.items():
            hits = glob.glob(os.path.join(_brightness_csv(v), f"*frmid_{fid}*.csv"))
            sf = os.path.join(SOLAR_DIR, f"{v.lower()}_solar_spectrum.csv")
            if hits and os.path.exists(sf):
                fsl = _f_slit_for(v, fid)
                items.append((v, hits[0], sf, MARS_EARTH_AU[v], PHASE_ANGLE[v], fsl))
            else:
                print(f"  {v}: missing CUTE CSV or solar file, skipping")
        out = _get_output_dir(os.path.join(output_dir, 'solar_percent_diff')) if SAVE else None
        CuteObservation.plot_solar_percent_diff(
            items, ref_label='Visit1',
            save=SAVE, output_dir=out, show=not SAVE)

    elif MODE == 'cutevsolar':
        items = []
        for v, fid in BRIGHTEST.items():
            hits = glob.glob(os.path.join(_brightness_csv(v), f"*frmid_{fid}*.csv"))
            sf = os.path.join(SOLAR_DIR, f"{v.lower()}_solar_spectrum.csv")
            if hits and os.path.exists(sf):
                items.append((v, hits[0], sf, MARS_EARTH_AU[v], PHASE_ANGLE[v],
                              _f_slit_for(v, fid)))
            else:
                print(f"  {v}: missing CUTE CSV or solar file, skipping")
        out = _get_output_dir(os.path.join(output_dir, 'cute_v_solar')) if SAVE else None
        CuteObservation.plot_cute_v_solar(items, save=SAVE, output_dir=out, show=not SAVE)

    else:
        raise ValueError(
            "MODE must be 'static', 'static_all', 'overlay', 'overlay_all', "
            f"'visit', 'visit_all', 'grid', or 'sequence', not {MODE!r}"
        )
 
 
if __name__ == '__main__':
    main()