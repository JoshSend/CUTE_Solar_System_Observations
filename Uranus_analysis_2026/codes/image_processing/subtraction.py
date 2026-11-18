"""
Background subtraction logic and all base processing for Uranus exposures.

author: @jose5987
"""

import sys
from pathlib import Path

import numpy as np

# --- import Helper class from utils ---
codes_dir = str(Path(__file__).resolve().parents[1])
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)
from utils.helpers import Directories, Observation_info

# ------ class initializations ------
dirs   = Directories()      # directory helper
info   = Observation_info() # Grabs relevant observation info that's shared

class CuteObservation():

    # ------ constants ------
    GAIN = 1.5                  # electrons per DN
    H = 6.626075540e-27         # erg s
    C = 2.99792458e+10          # cm / s
    N_SCI_PIX = 2048            # science pixels (excludes overscan)
    APERTURE = 1.4              # science half-height = APERATURE *(measured FWHM / 2)
    SCI_HALF = None             # rows: set a number for fixed half-height. None = ad

    # ---- fiting ----
    FIT = False              # scale each frame's flux to its visit's brightest
    FIT_TO = None            # {visit: brightest_frmid}
    FIT_BAND = (2490, 3250)  # wavelength band the fit is computed over
    FIT_METHOD = 'scale'     # 'scale' | 'linear' | 'ratio' | 'median'
    FIT_WEIGHTED = True      # weight by the brightest frame's 1/sigma (chi^2 fit)
    _bright_cache = {}       # visit -> (wave, flux, err), reduced once
    _fit_scale_printed = set()

    # ---- hot-pixel rejection ----
    HOT_REJECT   = True        # drop hot pixels from the science & dark regions
    HOT_WIN      = 7           # median-filter window along dispersion (odd, >=3)
    HOT_SIGMA    = 5.0         # reject pixels > this many robust sigma above local median
    APERTURE_SUM = True        # True: store n_sc*(mean_sci - mean_dk) (aperture sum, keeps
                               #       the flux scale). False: store the raw per-pixel mean
                               #       difference (rescales flux/albedo by ~1/n_sc).

    def __init__(self, visit=None, base_dir=None)
        self.output_dir = dirs.out_dir()

        self.visit = visit
        self.frame_id = self._extract_frame_id()

    def _build_regions(self, row_shift=0.0, sci_grow=0.0):
        """
        Define upper/lower edges of the science and dark traces as y-values for
        each science column.
        'row_shift' moves every edge up or down... auto-tracking.
        """
        nx = self.nx
        s = row_shift
        g = sci_grow

        # science trace edges
        y1_sc, y2_sc = 37 - 1 + s - g, 69 - 1 + s - g   # lower edge
        y3_sc, y4_sc = 59 - 1 + s + g, 88 - 1 + s + g   # upper edge
        # dark background (strip) edges, below trace
        y1_dk, y2_dk = 7 - 1 + s - g, 39 - 1 + s - g
        y3_dk, y4_dk = 29 - 1 + s + g, 58 - 1 + s + g

        x_left = nx - 1 - (52 - 1)        # flipped col of raw pixel 52
        x_right = nx - 1 - (2099 - 1)     # flipped col of raw pixel 2099
        span = x_right - x_left

        m1_sc = (y2_sc - y1_sc) / span
        m2_sc = (y4_sc - y3_sc) / span
        m1_dk = (y2_dk - y1_dk) / span
        m2_dk = (y4_dk - y3_dk) / span

        # flipped columns 100..2147, short->long wavelength (2048 values)
        self.xval = np.arange(nx - 2100, nx - 52, 1, dtype=int)

        # point-slope form from the left edge
        self.yval1_sc = y1_sc + m1_sc * (self.xval - x_left)   # trace lower
        self.yval2_sc = y3_sc + m2_sc * (self.xval - x_left)   # trace upper
        self.yval1_dk = y1_dk + m1_dk * (self.xval - x_left)   # dark  lower
        self.yval2_dk = y3_dk + m2_dk * (self.xval - x_left)   # dark  upper

    def extract_spectrum(self):
        dk_arr = np.zeros(2048, dtype=float)
        sc_arr = np.zeros(2048, dtype=float)
        err_arr = np.zeros(2048, dtype=float)          # 1-sigma error, in DN
        nrows = self.img.shape[0]
        g = self.GAIN
        rn2 = self.READ_NOISE ** 2
        clamp = lambda v: min(max(int(round(v)), 0), nrows)

        hot = self._build_hot_mask() if self.HOT_REJECT else np.zeros(self.img.shape, bool)
        self.n_hot_sci = self.n_hot_dk = 0
        self.n_hot_sci_col = np.zeros(2048, dtype=int)
        self.n_hot_dk_col  = np.zeros(2048, dtype=int)
        self.wdth_sci_col  = np.zeros(2048, dtype=int)
        self.wdth_dk_col   = np.zeros(2048, dtype=int)

        for i in range(2048):
            col = self.xval[i]

            # ---- dark strip: mean of hot-pixel-free pixels ----
            yy1, yy2 = clamp(self.yval1_dk[i]), clamp(self.yval2_dk[i])
            dk_pix = self.img[yy1:yy2, col].astype(float)
            dk_hot = hot[yy1:yy2, col]
            self.n_hot_dk_col[i] = int(dk_hot.sum())
            self.wdth_dk_col[i]  = yy2 - yy1  
            self.n_hot_dk += self.n_hot_dk_col[i]
            dk_good = dk_pix[~dk_hot]
            if dk_good.size == 0:
                dk_good = dk_pix                         # all flagged -> keep all
            n_dk = dk_good.size
            m_dk = float(dk_good.mean()) if n_dk else 0.0
            dk_arr[i] = m_dk

            # ---- science box: mean of hot-pixel-free pixels ----
            yy3, yy4 = clamp(self.yval1_sc[i]), clamp(self.yval2_sc[i])
            n_sc = yy4 - yy3                             # full aperture height (rows)
            if n_sc <= 0:
                continue
            sc_pix = self.img[yy3:yy4, col].astype(float)
            sc_hot = hot[yy3:yy4, col]
            self.n_hot_sci_col[i] = int(sc_hot.sum())
            self.wdth_sci_col[i]  = n_sc
            self.n_hot_sci += self.n_hot_sci_col[i]
            sc_good = sc_pix[~sc_hot]
            if sc_good.size == 0:
                sc_good = sc_pix
            n_sc_good = sc_good.size
            m_sc = float(sc_good.mean())

            sig = m_sc - m_dk                           # per-pixel dark-subtracted signal
            sc_arr[i] = (n_sc * sig) if self.APERTURE_SUM else sig

            # error on the mean-difference signal (e- -> DN), scaled to the aperture
            var_sc = (g * max(m_sc, 0.0) + rn2) / g ** 2 / max(n_sc_good, 1)
            var_dk = (g * max(m_dk, 0.0) + rn2) / g ** 2 / max(n_dk, 1)
            scale = n_sc if self.APERTURE_SUM else 1.0
            err_arr[i] = scale * np.sqrt(max(var_sc + var_dk, 0.0))

        self.sc_err_dn = err_arr
        if self.HOT_REJECT:
            key = (self.visit, self.frame_id)
            if key not in type(self)._hot_printed:
                type(self)._hot_printed.add(key)
                print(f"  {self.visit} frmid {self.frame_id}: hot pixels removed "
                      f"(science {self.n_hot_sci}, dark {self.n_hot_dk})")
        return sc_arr

    def _compute_flux(self):
        """
        Converts DN spectrum to photon flux density in units of
        10^-9 erg/s/cm^2/A, propages per-pixel error the same way.
        """

        wave_sol = self.wv_soln
        eff_area = self.eff_area
        wave_cm = wave_sol * 1.0e-8

        # flux = spectra * conv, so the error scales by the same conv
        conv = (self.GAIN / self.exptime) * (self.H * self.C) \
            / (wave_cm * eff_area) / 1.0e-9

        self.flux_err = self.sc_err_dn * conv
        return self.spectra * conv
