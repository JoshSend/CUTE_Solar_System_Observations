"""
Image Processing for the NASA CUTE (Colorado Ultraviolet Transit Experiment)
2025 Mars Observations in Near Ultraviolet
@Author: jose5987
Date Created: 8/21/2026

Based off previous processing logic written by dobh6980
"""

# --------- Relevant Imports ---------

import os                            # file handling
import glob                          # glob
import numpy as np                   # computation
from astropy.io import fits          # fits handling
from astropy.modeling import models, fitting # FITTING
import matplotlib.pyplot as plt      # plotting
import re
from scipy.ndimage import maximum_filter1d, gaussian_filter1d

# Animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Turn off matplotlib's auto interpolation for all images
plt.rcParams['image.interpolation'] = 'none'

# --------- Helper Functions ---------

def _get_default_dir():
    """
    All input files are read from the folder this script lives in,
    so the whole folder is portable.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    return base_dir

def _get_observations_dir():
    """
    Locates the CUTE_observations root folder
    relative to script location.
    """
    base = _get_default_dir()
    return os.path.abspath(os.path.join(base, '..', '..', 'CUTE_observations'))

def _get_output_dir(output_dir, visit=None):
    """
    Locates (and creates) the output folder
    for figures to be saved to.
    """
    base = _get_default_dir()
    path = os.path.abspath(os.path.join(base, '..', output_dir))
    if visit: # store output in its own visit subfolder
        path = os.path.join(path, visit)
    os.makedirs(path, exist_ok=True)   # create it if it doesn't exist yet
    return path

def _resolve_frame(folder, file_or_frmid):
    '''
    Turn `file_or_frmid` into a full FITS path inside `folder`.
    Accepts either a full filename ('...frmid_4923_...fits')
    or the frame id (the int 4923, or the string '4923'), which is
    matched against the `frmid_<n>` field of the filenames in `folder`.
    '''
    s = str(file_or_frmid)
    if s.endswith('.fits'):                      # a full filename
        path = os.path.join(folder, s)
        if not os.path.exists(path):
            raise FileNotFoundError(f'FITS file not found at: {path}')
        return path

    frmid = int(file_or_frmid)                   # otherwise, look it up by id
    # match the frmid_<number> field exactly, ignoring targetID340, dates, etc.
    pat = re.compile(r'frmid[_-]?0*' + str(frmid) + r'(?!\d)')
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith('.fits') and pat.search(fn):
            return os.path.join(folder, fn)

    # not found: report what's actually available
    avail = sorted(
        int(m.group(1))
        for fn in os.listdir(folder) if fn.lower().endswith('.fits')
        for m in [re.search(r'frmid[_-]?(\d+)', fn)] if m
    )
    raise FileNotFoundError(
        f'No FITS file with frmid {frmid} in {folder}. '
        f'Available frmids: {avail}')

def load_observation(visit, filename, reference=None):
    """
    Loads a single CuteObservation. `filename` may be a full FITS filename
    OR a frame id (e.g. 4868), which is looked up in the visit folder.
    """
    if reference is None:
        reference = CuteReference()

    folder = os.path.join(_get_observations_dir(), visit)
    full_path = _resolve_frame(folder, filename)
    return CuteObservation(full_path, reference=reference, visit=visit)

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth

def _gsmooth(y, fwhm_pix):
    """Gaussian-smooth y by a kernel of the given FWHM in pixels."""
    sigma = max(fwhm_pix / 2.3548, 1e-6)
    half = int(np.ceil(4 * sigma))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
    return np.convolve(y, k, mode='same')

def _degrade_to_R(wave, flux, R_from, R_to):
    """
    Convolve a spectrum from resolving power R_from down to R_to (R_to < R_from),
    with a wavelength-dependent kernel (constant R = constant width in ln-lambda).
    Returns flux unchanged if R_to >= R_from.
    """
    flux = np.asarray(flux, float)
    if R_to >= R_from:
        return flux.copy()
    lnw = np.log(wave)
    lnu = np.linspace(lnw[0], lnw[-1], len(wave))      # uniform ln-lambda grid
    dln = lnu[1] - lnu[0]
    fu = np.interp(lnu, lnw, flux)
    fwhm_lnw = np.sqrt((1.0 / R_to) ** 2 - (1.0 / R_from) ** 2)   # added width
    fs = _gsmooth(fu, fwhm_lnw / dln)
    return np.interp(lnw, lnu, fs)

# --- Mars disk / slit filling fraction ---
R_MARS_KM = 3389.5
AU_KM     = 1.495978707e8
ARCSEC    = 206264.806

# CUTE slit outline in the slit frame (arcsec): px = across-slit, py = along-slit
SLIT_PX = [-60., 60., 60., 30., 30., 15., 15., -15., -15., -30., -30., -60., -60.]
SLIT_PY = [-591.807, -591.807, -351.805, -351.805, 351.805, 351.805, 680.817,
           680.817, 351.805, 351.805, -351.805, -351.805, -591.807]

def mars_radius_arcsec(mars_earth_au):
    """Apparent radius of Mars (arcsec) at Earth-Mars distance Delta (AU)."""
    return R_MARS_KM / (float(mars_earth_au) * AU_KM) * ARCSEC

def slit_fill_fraction(across_arcsec, along_arcsec, mars_earth_au,
                       n=40000, seed=0):
    """
    Geometric fraction of Mars' disk that falls inside the CUTE slit.
    Disk radius = mars_radius_arcsec(Delta), centred at (across, along) in the
    slit frame (from the attitude analysis). Uniform-brightness full disk;
    phase illumination is handled separately by the Lambert phase function.
    """
    from matplotlib.path import Path as _Path
    rho = mars_radius_arcsec(mars_earth_au)
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    r = rho * np.sqrt(rng.uniform(0, 1, n))          # uniform over the disk
    pts = np.column_stack([across_arcsec + r * np.cos(t),
                           along_arcsec + r * np.sin(t)])
    poly = _Path(np.column_stack([SLIT_PX, SLIT_PY]))
    return float(poly.contains_points(pts).mean())

def _xcorr_shift(wave, a, b, max_lag_pix=40, cont_fwhm_pix=120.0):
    """
    Wavelength shift (Angstrom) that best aligns template `b` to spectrum `a`,
    from a high-passed, integer-lag cross-correlation with parabolic refinement.
    Apply as:  b_aligned = np.interp(wave - shift, wave, b)
    """
    hp = lambda y: np.nan_to_num(y - _gsmooth(y, cont_fwhm_pix))
    A, B = hp(a), hp(b)
    lags = np.arange(-max_lag_pix, max_lag_pix + 1)
    cc = np.array([np.sum(A * np.roll(B, k)) for k in lags], float)
    i = int(np.argmax(cc)); dpix = float(lags[i])
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i-1], cc[i], cc[i+1]; den = y0 - 2*y1 + y2
        if den != 0:
            dpix = lags[i] + 0.5 * (y0 - y2) / den
    return dpix * float(np.median(np.diff(wave)))

# Hue order for per-frame overlays. Checked for colorblind separation against a
# white page: worst adjacent pair dE 9.1 under protanopia, 19.6 normal vision.
# Don't reorder or extend it -- past ~8 hues the added colors stop being
# distinguishable, which is what makes a 15-line overlay unreadable.
OVERLAY_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                  '#e87ba4', '#008300', '#4a3aa7', '#e34948']
# A visit with more frames than hues reuses the hues with the next linestyle,
# so 8 hues x 3 styles = 24 combinations, each frame still unique. Visits top
# out at 15 frames, i.e. 8 solid + 7 dashed.
OVERLAY_DASHES = ['-', '--', ':']

def _visit_files(visit, pattern='*.fits', skip_frmid=None, required=True):
    """
    Every FITS file in a visit folder, in frame-id order, with `skip_frmid`
    frames dropped. Shared by the batch-save and overlay helpers so they all
    see exactly the same frame list as the animators.
    """
    skip = {int(s) for s in (skip_frmid or [])}
    folder = os.path.join(_get_observations_dir(), visit)
    files = sorted(glob.glob(os.path.join(folder, pattern)),
                   key=CuteObservation._frmid_of)
    if not files:
        if required:
            raise FileNotFoundError(f'No {pattern!r} files found in {folder}')
        return []
    files = [f for f in files if CuteObservation._frmid_of(f) not in skip]
    if not files and required:
        raise ValueError(f'No frames left in {visit} after applying skip_frmid')
    return files

def _lambert_phase(alpha_deg):
    """Integral phase function of a Lambert sphere, normalized to 1 at alpha=0:
       Phi(a) = [sin a + (pi - a) cos a] / pi.   (first-order; Mars isn't exactly
       Lambertian, so swap in a measured Mars phase law for rigorous work.)"""
    a = np.radians(float(alpha_deg))
    return (np.sin(a) + (np.pi - a) * np.cos(a)) / np.pi

def continuum_bins(wave, refl, solar, bin_A=100.0, env_win_A=25.0,
                   clean_frac=0.85, min_pts=5):
    """
    Continuum albedo: bin `refl` only at wavelengths where the SOLAR spectrum is
    near its local continuum (line-free), so no line cores and no interpolation.
    Returns (bin_center, albedo_median, albedo_err) for bins with enough points.
    """
    refl = np.asarray(refl, float); solar = np.asarray(solar, float)
    dl = float(np.median(np.diff(wave)))
    win = max(int(round(env_win_A / dl)), 3)
    envelope = gaussian_filter1d(maximum_filter1d(solar, win), win / 2.0)
    clean = np.isfinite(refl) & np.isfinite(solar) & (solar >= clean_frac * envelope)

    edges = np.arange(wave[0], wave[-1] + bin_A, bin_A)
    wb, pb, eb = [], [], []
    for i in range(len(edges) - 1):
        sel = clean & (wave >= edges[i]) & (wave < edges[i + 1])
        k = int(sel.sum())
        if k >= min_pts:
            v = refl[sel]
            wb.append(0.5 * (edges[i] + edges[i + 1]))
            pb.append(float(np.median(v)))
            eb.append(float(np.std(v) / np.sqrt(k)))
    return np.array(wb), np.array(pb), np.array(eb)

# ------------------------------------

class CuteReference:
    """
    Grabs all static directories/files shared by 2025 Mars CUTE Observations

    Directories/Files to grab:
        base directory
        effective area
        wavelength solution
    """

    eff_area_fname = 'cute_recalculated_effa_2025.txt'
    wv_soln_fname  = 'flight_quad_wavelength_solution_final.dat'

    def __init__(self, base_dir=None):
        self.base_dir       = base_dir or _get_default_dir()
        self.wv_soln  = self._get_wv_soln()
        self.eff_area = self._get_eff_area()

    def _get_wv_soln(self):
        path = os.path.join(self.base_dir, CuteReference.wv_soln_fname)
        wave_sol = []
        data = open(path, 'r')
        lines = data.readlines()

        count = 0
        for j in lines:
            if count > 0:
                li = j.strip()
                st = j.split()
                ww = float(st[0])
                wave_sol.append(ww)
            count = count + 1

        return np.array(wave_sol)

    def _get_eff_area(self):
        path = os.path.join(self.base_dir, CuteReference.eff_area_fname)

        eff_area = []
        wave_val = []

        data = open(path, 'r')
        lines = data.readlines()

        count = 0
        for j in lines:
            if count > 1:
                li = j.strip()
                st = j.split()
                ww = float(st[1])
                ee = float(st[0])
                eff_area.append(ee)
                wave_val.append(ww)
            count = count + 1

        eff_area = np.array(eff_area)
        wave_val = np.array(wave_val)

        return np.interp(self.wv_soln, wave_val, eff_area)

# ------------------------------------

class CuteObservation:
    """
    One CUTE exposure ("visit")
    """

    # --------- constants -> class attributes ---------
    GAIN = 1.5                  # electrons per DN
    H = 6.626075540e-27         # erg s
    C = 2.99792458e+10          # cm / s
    N_SCI_PIX = 2048            # science pixels (excludes overscan)
    APERTURE = 1.4              # science half-height = APERATURE *(measured FWHM / 2)
    SCI_HALF = None             # rows: set a number for fixed half-height. None = adaptive
    READ_NOISE = 3.6            # e- per pixel # PLACEHOLDER PLACEHOLDER PLACEHOLDER needs cute actual e- per pixel

    # ---- fiting ----
    FIT = False              # scale each frame's flux to its visit's brightest
    FIT_TO = None            # {visit: brightest_frmid}
    FIT_BAND = (2490, 3250)  # wavelength band the fit is computed over
    FIT_METHOD = 'scale'     # 'scale' | 'linear' | 'ratio' | 'median'
    FIT_WEIGHTED = True      # weight by the brightest frame's 1/sigma (chi^2 fit)
    _bright_cache = {}       # visit -> (wave, flux, err), reduced once
    _fit_scale_printed = set()

    def __init__(
        self, fits_fname, reference: CuteReference, visit=None, base_dir=None,
        track=True, widen=True,
    ):
        self.fits_fname = fits_fname
        self.reference = reference
        self.base_dir = base_dir or reference.base_dir

        self.visit = visit
        self.frame_id = self._extract_frame_id()

        self.img, self.exptime = self._load_image()
        self.nx = self.img.shape[1]

        self.row_offset = 0.0
        self.trace_fwhm = 0.0
        self.sci_grow = 0.0
        self._build_regions()                       # nominal box
        if track:
            self.row_offset, self.trace_fwhm = self._measure_trace_shape()

        # Pick the science half-height:
        #   SCI_HALF set -> FIXED width (2*SCI_HALF rows), ignores FWHM
        #   elif widen   -> adaptive, APERTURE * FWHM
        #   else         -> nominal box height
        i_mid = len(self.xval) // 2
        h0 = 0.5 * (self.yval2_sc[i_mid] - self.yval1_sc[i_mid])   # nominal half-height
        if self.SCI_HALF is not None:
            target_half = self.SCI_HALF
        elif widen and self.trace_fwhm > 0:
            target_half = self.APERTURE * (self.trace_fwhm / 2.0)
        else:
            target_half = h0
        self.sci_grow = max(target_half - h0, 1 - h0)      # never below ~1px
        self._build_regions(row_shift=self.row_offset, sci_grow=self.sci_grow)

        self.spectra = self.extract_spectrum()
        self.flux = self._compute_flux()

        self.fit_scale, self.fit_label = 1.0, ''
        if self.FIT and self.FIT_TO and self.visit in self.FIT_TO:
            self.flux, self.fit_scale, self.fit_label = self._fit_to_brightest()
            self.flux_err = self.flux_err * self.fit_scale
            key = (self.visit, self.frame_id)
            if key not in type(self)._fit_scale_printed:
                type(self)._fit_scale_printed.add(key)
                print(f"  {self.visit} frmid {self.frame_id}: {self.fit_label}")

    @classmethod
    def fit_tag(cls):
        """'_FIT-<method>' when fitting is on, else ''."""
        return f'_FIT-{cls.FIT_METHOD}' if cls.FIT else ''

    def _load_image(self):
        with fits.open(self.fits_fname) as fits_file:
            hdr = fits_file[0].header
            img = np.fliplr(fits_file[0].data)

            # Exposure time is per-frame in the header (MILLISECONDS). Prefer the
            # measured EXPTIME, then EXPOSURE, then commanded EXPCMD -- no silent
            # universal default, so a missing value is caught instead of guessed.
            for key in ('EXPTIME', 'EXPOSURE', 'EXPCMD'):
                if key in hdr:
                    exptime_ms = float(hdr[key])
                    break
            else:
                raise KeyError(
                    f"no exposure keyword (EXPTIME/EXPOSURE/EXPCMD) in "
                    f"{os.path.basename(self.fits_fname)}")
            if exptime_ms <= 0:
                raise ValueError(
                    f"non-positive exposure {exptime_ms} ms in "
                    f"{os.path.basename(self.fits_fname)}")

            exptime = exptime_ms / 1000.0   # seconds
        return img, exptime

    def _build_regions(self, row_shift=0.0, sci_grow=0.0):
        """
        Define upper/lower edges of the science trace and of the background
        strip as a y-value for each science column. `row_shift` moves every
        edge up(+)/down(-) by that many rows (used by auto-tracking).
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

    def _measure_trace_shape(self, band=100, frac=0.5):
        """
        Estimate how far the trace has drifted (in rows).
        Takes a band of columns and collapses it to a
        single vertical profile, one brightness per
        detector row.
        Takes every bright row and compares it to where the
        expected nominal center is, then outputs as offset.
        """
        i_mid = len(self.xval) // 2
        c_mid = int(self.xval[i_mid])
        c_lo = max(c_mid - band, 0)
        c_hi = min(c_mid + band, self.nx)

        prof = np.median(self.img[:, c_lo:c_hi], axis=1).astype(float)
        prof -= np.median(prof)             # remove background level
        prof = np.clip(prof, 0, None)       # keep only positive signal
        if prof.max() <= 0:                 # nothing bright
            return 0.0, 0.0

        mask = prof >= frac * prof.max()
        rows = np.arange(prof.size)
        weights = np.where(mask, prof, 0.0)
        center = float((rows * weights).sum() / weights.sum())
        fwhm = float(mask.sum())            # rows above half-max ~ FWHM

        nominal = 0.5 * (self.yval1_sc[i_mid] + self.yval2_sc[i_mid])
        return center - nominal, fwhm

    # --------- reduction ---------
    def extract_spectrum(self):
        dk_arr = np.zeros(2048, dtype=float)
        sc_arr = np.zeros(2048, dtype=float)
        err_arr = np.zeros(2048, dtype=float)          # 1-sigma error, in DN
        nrows = self.img.shape[0]
        g = self.GAIN
        rn2 = self.READ_NOISE ** 2
        clamp = lambda v: min(max(int(round(v)), 0), nrows)

        for i in range(2048):
            col = self.xval[i]

            yy1, yy2 = clamp(self.yval1_dk[i]), clamp(self.yval2_dk[i])
            n_dk = yy2 - yy1
            dk_arr[i] = np.median(self.img[yy1:yy2, col]) if n_dk > 0 else 0.0

            yy3, yy4 = clamp(self.yval1_sc[i]), clamp(self.yval2_sc[i])
            n_sc = yy4 - yy3
            if n_sc > 0:
                box_sum = float(np.sum(self.img[yy3:yy4, col]))   # DN, source+bkg
                sc_arr[i] = box_sum - n_sc * dk_arr[i]            # dark-subtracted

                # CCD aperture-sum error: Poisson of all electrons in the box,
                # read noise per science pixel, and the uncertainty in the
                # background level estimated from n_dk dark pixels.
                total_e = g * box_sum
                bkg_e = g * dk_arr[i]
                var_e = total_e + n_sc * rn2
                if n_dk > 0:
                    var_e += (n_sc ** 2) * (bkg_e + rn2) / n_dk
                err_arr[i] = np.sqrt(max(var_e, 0.0)) / g         # back to DN

        self.sc_err_dn = err_arr
        return sc_arr

    def _compute_flux(self):
        """
        Converts DN spectrum to photon flux density in units of
        10^-9 erg/s/cm^2/A, and propagates the per-pixel error the same way.
        """
        wave_sol = self.reference.wv_soln
        eff_area = self.reference.eff_area
        wave_cm = wave_sol * 1.0e-8

        # flux = spectra * conv, so the error scales by the same conv
        conv = (self.GAIN / self.exptime) * (self.H * self.C) \
            / (wave_cm * eff_area) / 1.0e-9

        self.flux_err = self.sc_err_dn * conv
        return self.spectra * conv

    def _fit_to_brightest(self):
        """
        Fit so that brightest ~ f(other) over FIT_BAND, then apply f to this
        frame's full flux. Returns (fitted_flux, slope, label). Method is
        chosen by FIT_METHOD, so swapping estimators is a one-word change.
            'scale'  : a*x            (Linear1D, intercept fixed at 0)
            'linear' : a*x + b        (Linear1D, free intercept -- absorbs bkg offset)
            'ratio'  : a = sum/sum    (integrated-flux ratio)
            'median' : a = median(x_b/x_o)  (robust per-pixel ratio)
        """
        try:
            fid = int(self.frame_id)
        except (TypeError, ValueError):
            return self.flux, 1.0, 'skip'
        if fid == int(self.FIT_TO[self.visit]):
            return self.flux, 1.0, 'reference'

        wv_b, fx_b, er_b = self._get_bright_flux()
        wave = self.reference.wv_soln
        lo, hi = self.FIT_BAND
        band = (wave >= lo) & (wave <= hi)
        xo, yb = self.flux[band], fx_b[band]

        # astropy convention: weight = 1/sigma  ->  minimizes chi^2
        w = None
        if self.FIT_WEIGHTED:
            w = 1.0 / np.clip(er_b[band], 1e-30, None)

        method = self.FIT_METHOD
        if method in ('scale', 'linear'):
            fixed = {'intercept': True} if method == 'scale' else {}
            m0 = models.Linear1D(slope=1.0, intercept=0.0, fixed=fixed)
            m = fitting.LinearLSQFitter()(m0, xo, yb, weights=w)
            a, b = float(m.slope.value), float(m.intercept.value)
        elif method == 'ratio':
            a, b = float(np.sum(yb) / np.sum(xo)), 0.0
        elif method == 'median':
            good = xo > np.nanmax(xo) * 0.05          # ignore the noise floor
            a, b = float(np.nanmedian(yb[good] / xo[good])), 0.0
        else:
            raise ValueError(f"unknown FIT_METHOD {method!r}")

        fitted = a * self.flux + b
        label = f"{method}: a={a:.4f}" + (f", b={b:+.4g}" if b else "")
        return fitted, a, label

    def _get_bright_flux(self):
        """(wave, flux, err) of this visit's brightest frame -- reduced once, cached."""
        if self.visit not in self._bright_cache:
            folder = os.path.join(_get_observations_dir(), self.visit)
            path = _resolve_frame(folder, int(self.FIT_TO[self.visit]))
            saved = type(self).FIT
            type(self).FIT = False            # reduce reference WITHOUT fitting -> no recursion
            try:
                b = type(self)(path, self.reference, visit=self.visit)
            finally:
                type(self).FIT = saved
            self._bright_cache[self.visit] = (b.reference.wv_soln, b.flux, b.flux_err)
        return self._bright_cache[self.visit]

    # --------- outputs ---------
    def _extract_frame_id(self):
        """
        Parses the 'frmid_####' token out of
        filename strings
        """
        bare_name = os.path.basename(self.fits_fname)
        parts = bare_name.split('_')

        if "frmid" in parts:
            idx = parts.index('frmid')
            return parts[idx + 1]

        return "Unknown"

    def plot_trace(self, title=None, vmin=None, vmax=None, ax=None):
        if title is None:
            visit_str = f"{self.visit} " if self.visit else ""
            title = f"Mars - {visit_str}frmid {self.frame_id}"
        if vmin is None:
            vmin = np.percentile(self.img, 5)
        if vmax is None:
            vmax = np.percentile(self.img, 99)

        own = ax is None
        fig, ax = plt.subplots() if own else (ax.figure, ax)

        ax.set_title(title)
        im = ax.imshow(self.img, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        ax.plot(self.xval, self.yval1_sc, color='r', lw=2)
        ax.plot(self.xval, self.yval2_sc, color='r', lw=2)
        ax.plot(self.xval, self.yval1_dk, color='y', lw=2)
        ax.plot(self.xval, self.yval2_dk, color='y', lw=2)
        if own:                       # only add a colorbar for a standalone figure
            fig.colorbar(im, ax=ax)
        return fig, ax


    def plot_spectrum(self, box_pts=15, title=None, xlim=(2490, 3250),
                    ylim=(0.0, 2.0), ax=None, color='maroon'):
        if title is None:
            visit_str = f"{self.visit}" if self.visit else ""
            title = f"Mars NUV Spectra with CUTE - {visit_str} frmid {self.frame_id}"

        own = ax is None
        fig, ax = plt.subplots(figsize=(10, 4)) if own else (ax.figure, ax)

        ax.plot(self.reference.wv_soln, smooth(self.flux, box_pts),
                color=color, lw=1.5)
        ax.set_title(title)
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel(
            r"Flux ($10^{-9}\ \mathrm{erg}\ \mathrm{s}^{-1}\ \mathrm{cm}^{-2}\ \mathrm{\AA}^{-1}$)"
        )
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        if own:
            fig.tight_layout()
        return fig, ax

    @staticmethod
    def _frmid_of(fname):
        """Pull the frame id out of a filename via its frmid_<n> field."""
        m = re.search(r'frmid[_-]?(\d+)', os.path.basename(fname))
        return int(m.group(1)) if m else None

    # --------- batch static saves ---------
    @classmethod
    def save_visit_frames(cls, visit, reference=None, output_dir=None,
                          kind='both', box_pts=15, xlim=(2490, 3250),
                          ylim=(0.0, 2.0), vmin=None, vmax=None,
                          pattern='*.fits', skip_frmid=None, dpi=200,
                          overwrite=True, verbose=True, write_csv=True):
        '''
        Save a static PNG for EVERY frame in one visit -- no manual FILENAME.

            kind='spectrum' : one 1D-spectrum PNG per frame
            kind='trace'    : one 2D trace PNG per frame
            kind='both'     : both (default)

        Files land in `output_dir` as "<fits stem>_trace.png" /
        "<fits stem>_spectrum.png", i.e. exactly the names MODE 'static'
        already writes, so a batch run and a one-off run agree.

        `vmin`/`vmax` default to None, which lets plot_trace stretch each
        frame between its own 5th and 99th percentile. A hard ceiling well
        below the trace (the old vmax=6000) saturates the whole science box
        into flat yellow and hides the spectrum -- only set vmin/vmax by hand
        when every frame really is on the same brightness scale.

        `ylim` is fixed by default so every saved spectrum is on the same
        scale and frames stay comparable; pass ylim=None to autoscale each.
        `overwrite=False` skips frames whose PNGs already exist, which makes
        a re-run cheap after adding new data.

        Returns the list of paths written.
        '''
        if reference is None:
            reference = CuteReference()
        if output_dir is None:
            raise ValueError("save_visit_frames needs an output_dir")
        if kind == 'both':
            kinds = ['trace', 'spectrum']
        elif kind in ('trace', 'spectrum'):
            kinds = [kind]
        else:
            raise ValueError(
                f"kind must be 'spectrum', 'trace', or 'both', not {kind!r}")

        files = _visit_files(visit, pattern=pattern, skip_frmid=skip_frmid)
        os.makedirs(output_dir, exist_ok=True)

        saved, skipped = [], 0
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            targets = {k: os.path.join(output_dir, f"{stem}_{k}{cls.fit_tag()}.png")
                       for k in kinds}

            if not overwrite and all(os.path.exists(p) for p in targets.values()):
                skipped += 1
                continue

            obs = cls(f, reference, visit=visit)          # obs created HERE
            for k, path in targets.items():
                if k == 'trace':
                    fig, _ = obs.plot_trace(vmin=vmin, vmax=vmax)
                else:
                    fig, _ = obs.plot_spectrum(box_pts=box_pts,
                                               xlim=xlim, ylim=ylim)
                fig.savefig(path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved.append(path)

            if write_csv:
                obs.save_spectrum_csv(os.path.join(output_dir, 'csv'))

        if verbose:
            msg = f"{visit}: saved {len(saved)} PNG(s) for {len(files)} frame(s)"
            if skipped:
                msg += f" ({skipped} frame(s) already on disk, left alone)"
            print(f"  {msg} -> {output_dir}")
        return saved

    def save_spectrum_csv(self, output_dir):
        """Write this frame's 1D spectrum as wave,flux,error (one row per pixel)."""
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.fits_fname))[0]
        path = os.path.join(output_dir, f"{stem}{self.fit_tag()}.csv")
        arr = np.column_stack([self.reference.wv_soln, self.flux, self.flux_err])
        np.savetxt(path, arr, delimiter=',', header='wave,flux,err',
                   comments='', fmt='%.6e')
        return path

    @classmethod
    def save_all_frames(cls, visits, reference=None, output_root=None,
                        **kwargs):
        '''
        Run save_visit_frames over a list of visits, one subfolder per visit
        (<output_root>/<visit>/). Visits with no usable frames are reported
        and stepped over rather than killing the whole batch.

        Returns {visit: [paths]}.
        '''
        if reference is None:
            reference = CuteReference()
        if output_root is None:
            raise ValueError("save_all_frames needs an output_root")

        results = {}
        for v in visits:
            out_path = os.path.join(output_root, v)
            try:
                results[v] = cls.save_visit_frames(
                    v, reference=reference, output_dir=out_path, **kwargs)
            except (FileNotFoundError, ValueError) as err:
                print(f"  {v}: skipped ({err})")
                results[v] = []

        total = sum(len(p) for p in results.values())
        print(f"Done: {total} PNG(s) across {len(results)} visit(s) "
              f"-> {output_root}")
        return results

    @classmethod
    def reflectance_curve(cls, cute_csv, solar_file, mars_earth_au,
                          phase_angle=None, mars_sun_au=None, solar_wave_unit='nm',
                          solar_delimiter=',', match_resolution=True, f_slit=None,
                          R_cute=700.0, R_hsrs=28000.0, R_work=500.0,
                            align=True,
                          max_lag_pix=40, xlim=(2490, 3250),
                          bin_A=30.0, clean_frac=0.85):
        """
        Returns (wave, refl, solar_shift_A, bin_wave, bin_albedo, bin_err).
        `refl` is the full F_CUTE/F_pred ratio (geometric albedo p if phase_angle
        given); the bin_* arrays are the continuum albedo -- medians of refl at
        solar-line-free wavelengths only, no interpolation.
        """
        R_MARS_AU, W_TO_CGS, CUTE_UNIT = 2.2659e-5, 100.0, 1e-9
        d = np.genfromtxt(cute_csv, delimiter=',', names=True)
        nmn = d.dtype.names
        cw = np.asarray(d['wave'] if 'wave' in nmn else d[nmn[0]], float)
        cf = np.asarray(d['flux'] if 'flux' in nmn else d[nmn[1]], float)
        s = np.atleast_2d(np.genfromtxt(solar_file, delimiter=solar_delimiter,
                                        comments='#'))
        sw = s[:, 0].astype(float)
        if mars_sun_au is not None:
            col = 2 if s.shape[1] > 2 else 1
            solar_at_mars = s[:, col].astype(float) / float(mars_sun_au) ** 2
        else:
            solar_at_mars = s[:, 1].astype(float)
        if str(solar_wave_unit).lower().startswith('nm'):
            sw = sw * 10.0
        o = np.argsort(sw); sw, solar_at_mars = sw[o], solar_at_mars[o]
        solar_pred = (solar_at_mars * (R_MARS_AU / float(mars_earth_au)) ** 2
                      * W_TO_CGS / CUTE_UNIT)

        # Work at a common resolution R_work (<= R_cute): degrade CUTE down from
        # its native R_cute AND HSRS down from R_hsrs to the SAME R_work. HSRS has
        # the true line shape, so it matches CUTE at R_work and the lines cancel;
        # R_work is chosen low enough to average down CUTE's per-pixel noise.
        cute_m  = (_degrade_to_R(cw, cf, R_cute, R_work)
                   if match_resolution else smooth(cf, 15))
        solar_r = _degrade_to_R(sw, solar_pred, R_hsrs, R_work)
        solar_on_cw = np.interp(cw, sw, solar_r)
        sh = 0.0
        if align:
            sh = _xcorr_shift(cw, cute_m, solar_on_cw, max_lag_pix)
            solar_on_cw = np.interp(cw - sh, cw, solar_on_cw)

        with np.errstate(divide='ignore', invalid='ignore'):
            refl = cute_m / solar_on_cw
        if phase_angle is not None:
            refl = refl / _lambert_phase(phase_angle)
        if f_slit is not None and f_slit > 0:
            refl = refl / f_slit                     # slit-filling correction -> full-disk p

        # continuum albedo at solar-line-free wavelengths (no interpolation)
        wb, pb, eb = continuum_bins(cw, refl, solar_on_cw,
                                    bin_A=bin_A, clean_frac=clean_frac)
        return cw, refl, sh, wb, pb, eb

    @classmethod
    def plot_reflectance_all(cls, items, xlim=(2490, 3250), ymax=0.12,
                             save=False, output_dir=None,
                             fname='reflectance_all.png', dpi=200, show=True):
        """items: list of (label, cute_csv, solar_file, mars_earth_au)."""
        cmap = plt.colormaps.get_cmap('tab10')
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, (label, cc, sf, dae) in enumerate(items):
            cw, refl, sh, wb, pb, eb = cls.reflectance_curve(cc, sf, dae, xlim=xlim)
            ax.plot(cw, refl, color=cmap(i % 10), lw=1.1, alpha=0.9,
                    label=f"{label}  (shift {sh:+.1f} A)")
        ax.set_xlim(xlim); ax.set_ylim(0, ymax)
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel(r"Reflectance  $p\,\Phi$")
        ax.set_title("Mars reflectance -- all visits")
        ax.grid(True, ls='--', lw=0.6, alpha=0.6)
        ax.legend(fontsize=8, ncol=2, loc='upper right')
        fig.tight_layout()
        if save:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, fname), dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        return fig, ax

    @classmethod
    def plot_reflectance_grid(cls, items, ncols=4, xlim=(2490, 3250), ymax=0.1,
                              share_ylim=True,
                              suptitle="2025 Mars NUV Geometric Albedo (continuum-binned)",
                              save=False, output_dir=None,
                              fname='albedo_grid.png', dpi=200, show=True):
        """items: list of (label, cute_csv, solar_file, mars_earth_au, phase_angle_deg)."""
        n = len(items)
        if n == 0:
            raise ValueError("no items to plot")
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False, sharex=True, constrained_layout=True)
        axes_flat = axes.ravel()
        for ax in axes_flat[n:]:
            ax.axis('off')
        fig.suptitle(suptitle, fontsize=16)

        for ax, (label, cc, sf, dae, pa, fsl) in zip(axes_flat, items):
            cw, refl, sh, wb, pb, eb = cls.reflectance_curve(
                cc, sf, dae, phase_angle=pa, f_slit=fsl, xlim=xlim)
            ax.plot(cw, refl, color='maroon', lw=0.7, alpha=0.50)        # full ratio
            ax.errorbar(wb, pb, yerr=eb, fmt='o', ms=3.5, color='k',
                        capsize=2, lw=1, zorder=5)                       # binned points
            ax.plot(wb, pb, color='royalblue', lw=1.4, zorder=4)         # <-- line through bins                     # continuum albedo
            ax.set_title(f"{label}  (a={pa:.1f} deg, shift {sh:+.1f} A)", fontsize=9)
            ax.set_xlim(xlim)
            ax.grid(True, ls='--', lw=0.5, alpha=0.5)
            if share_ylim:
                ax.set_ylim(0, ymax)

        for i, ax in enumerate(axes_flat[:n]):
            if i + ncols >= n:
                ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=9)
        for r in range(nrows):
            axes[r, 0].set_ylabel(r"Geometric albedo $p$", fontsize=9)

        if save:
            if output_dir is None:
                raise ValueError("save=True needs an output_dir")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f"Saved {path}")
        if show:
            plt.show()
        return fig, axes

    @classmethod
    def plot_mars_solar_overlay(cls, cute_csv, solar_file, mars_earth_au=None,
                                mars_sun_au=None, solar_wave_unit='nm', phase_angle=None,
                                solar_delimiter=',', mode='overlay', scale='auto',
                                match_resolution=True, R_cute=700.0, R_tsis=300.0,
                                align=True, max_lag_pix=40, box_pts=15,
                                xlim=(2490, 3250), title=None, save=False,
                                output_dir=None, fname="mars_solar_overlay.png",
                                dpi=200, show=True):
        """
        Overplot a CUTE Mars spectrum and the reflected solar spectrum, with the
        full reflected-light geometry applied so the two are directly comparable.

        mode 'reflectance' returns F_obs/F_pred = p*Phi, after:
          match_resolution : convolve CUTE down to the solar FWHM (solar_fwhm_A, A)
                             so sharp solar lines don't ring in the ratio.
          align            : cross-correlate and shift the solar onto CUTE's
                             wavelength scale (removes derivative spikes at lines).
        See earlier docstring for mars_earth_au / mars_sun_au geometry.
        """
        R_MARS_AU = 2.2659e-5     # Mars radius in AU
        W_TO_CGS  = 100.0         # 1 W/m^2/nm = 100 erg/s/cm^2/A
        CUTE_UNIT = 1e-9          # CUTE flux stored in units of 1e-9 erg/s/cm^2/A
        solar_c   = 'royalblue'

        # --- CUTE spectrum (1e-9 erg/s/cm^2/A) ---
        d = np.genfromtxt(cute_csv, delimiter=',', names=True)
        nm = d.dtype.names
        cw = np.asarray(d['wave'] if 'wave' in nm else d[nm[0]], float)
        cf = np.asarray(d['flux'] if 'flux' in nm else d[nm[1]], float)
        cf_s = smooth(cf, box_pts)

        # --- solar -> "at Mars" (W/m^2/nm) ---
        s = np.atleast_2d(np.genfromtxt(solar_file, delimiter=solar_delimiter,
                                        comments='#'))
        sw = s[:, 0].astype(float)
        if mars_sun_au is not None:
            col = 2 if s.shape[1] > 2 else 1
            solar_at_mars = s[:, col].astype(float) / float(mars_sun_au) ** 2
        else:
            solar_at_mars = s[:, 1].astype(float)
        if str(solar_wave_unit).lower().startswith('nm'):
            sw = sw * 10.0
        o = np.argsort(sw); sw, solar_at_mars = sw[o], solar_at_mars[o]

        # --- predicted reflected flux at Earth (albedo = 1), in 1e-9 cgs units ---
        solar_pred_disp = None
        if mars_earth_au is not None:
            dilution = (R_MARS_AU / float(mars_earth_au)) ** 2
            solar_pred_disp = solar_at_mars * dilution * W_TO_CGS / CUTE_UNIT

        lo, hi = xlim
        cb = (cw >= lo) & (cw <= hi)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_xlim(xlim); ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.grid(True, ls='--', lw=0.6, alpha=0.6)

        if mode == 'overlay':
            if solar_pred_disp is None:
                raise ValueError("mode='overlay' needs mars_earth_au")
            l1, = ax.plot(cw, cf_s, color='maroon', lw=1.6,
                          label='CUTE Mars (observed)', zorder=3)
            l2, = ax.plot(sw, solar_pred_disp, color=solar_c, lw=1.4, alpha=0.9,
                          label='Solar reflected (albedo = 1)', zorder=2)
            ax.set_ylabel(r"Flux ($10^{-9}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
            ax.legend(handles=[l1, l2], loc='upper right', fontsize=9)

        elif mode == 'reflectance':
            if solar_pred_disp is None:
                raise ValueError("mode='reflectance' needs mars_earth_au")
            plt.close(fig)                      # discard shared axis; use 2 panels
            cute_m = _degrade_to_R(cw, cf, R_cute, R_tsis) if match_resolution else cf_s
            solar_on_cw = np.interp(cw, sw, solar_pred_disp)
            sh = 0.0
            if align:
                sh = _xcorr_shift(cw, cute_m, solar_on_cw, max_lag_pix)
                solar_on_cw = np.interp(cw - sh, cw, solar_on_cw)

            with np.errstate(divide='ignore', invalid='ignore'):
                refl = cute_m / solar_on_cw                      # p * Phi(alpha)
            if phase_angle is not None:
                phi = _lambert_phase(phase_angle)
                refl = refl / phi                                # -> geometric albedo p
            band = cb & np.isfinite(solar_on_cw) & np.isfinite(cute_m)
            a = float(np.sum(cute_m[band] * solar_on_cw[band]) /
                      np.sum(solar_on_cw[band] ** 2))
            resid = cute_m - a * solar_on_cw

            fig, (axr, axs) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            axr.plot(cw, refl, color='maroon', lw=1.4,
                     label=('Geometric albedo p' if phase_angle is not None
                            else 'Reflectance (I/F)'))
            axr.set_ylabel(r"Geometric albedo $p$" if phase_angle is not None
                           else r"Reflectance  $p\,\Phi$")
            axr.set_title(r"Division:  $F_{\rm CUTE}/F_{\rm pred}$")
            good = cb & np.isfinite(refl)
            if good.any():
                med = float(np.nanmedian(refl[good])); axr.set_ylim(0, 3 * med)
                print(f"  reflectance median = {med:.4f}, albedo scale a = {a:.4f}"
                      + (f", Phi({phase_angle:.1f} deg) = {_lambert_phase(phase_angle):.3f}"
                         if phase_angle is not None else "")
                      + f", solar shift = {sh:+.2f} A")
            axr.grid(True, ls='--', lw=0.6, alpha=0.6)
            axr.legend(loc='upper right', fontsize=9)

            axs.plot(cw, resid, color='teal', lw=1.2, label=f'CUTE - {a:.3g}×solar')
            axs.axhline(0, color='k', lw=0.8, alpha=0.6)
            axs.set_ylabel(r"Residual ($10^{-9}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
            axs.set_xlabel(r"Wavelength ($\AA$)")
            axs.set_title(r"Subtraction:  $F_{\rm CUTE} - a\,F_{\rm pred}$")
            axs.set_xlim(xlim)
            axs.grid(True, ls='--', lw=0.6, alpha=0.6)
            axs.legend(loc='upper right', fontsize=9)

            fig.suptitle(title or "CUTE Mars reflectance", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            if save:
                if output_dir is None:
                    raise ValueError("save=True needs an output_dir")
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, fname)
                fig.savefig(path, dpi=dpi, bbox_inches='tight')
                print(f"Saved {path}")
            if show:
                plt.show()
            return fig, (axr, axs)

        elif mode == 'twin':
            l1, = ax.plot(cw, cf_s, color='maroon', lw=1.6, label='CUTE Mars', zorder=3)
            ax.set_ylabel(r"CUTE flux ($10^{-9}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
            ax2 = ax.twinx()
            l2, = ax2.plot(sw, solar_at_mars, color=solar_c, lw=1.4, alpha=0.9,
                           label='Solar (at Mars)', zorder=2)
            ax2.set_ylabel('Solar irradiance at Mars (W m$^{-2}$ nm$^{-1}$)')
            ax.legend(handles=[l1, l2], loc='upper right', fontsize=9)

        elif mode == 'common':
            # CUTE and the Sun->Mars solar, both in absolute erg/s/cm^2/A, on a
            # log axis (they differ by ~1e10). No scaling factor.
            cute_abs  = cf_s * CUTE_UNIT
            solar_abs = solar_at_mars * W_TO_CGS
            l1, = ax.plot(cw, cute_abs, color='maroon', lw=1.6,
                          label='CUTE Mars (observed)', zorder=3)
            l2, = ax.plot(sw, solar_abs, color=solar_c, lw=1.4, alpha=0.9,
                          label='Solar (at Mars)', zorder=2)
            ax.set_yscale('log')
            ax.set_ylabel(r"Flux / irradiance (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
            ax.legend(handles=[l1, l2], loc='center right', fontsize=9)

        elif mode == 'scaled':
            l1, = ax.plot(cw, cf_s, color='maroon', lw=1.6, label='CUTE Mars', zorder=3)
            ax.set_ylabel(r"CUTE flux ($10^{-9}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
            sb = (sw >= lo) & (sw <= hi)
            a = (float(np.nanmedian(cf_s[cb]) / np.nanmedian(solar_at_mars[sb]))
                 if scale == 'auto' else float(scale))
            l2, = ax.plot(sw, solar_at_mars * a, color=solar_c, lw=1.4, alpha=0.9,
                          label=f'Solar (at Mars)  x {a:.3g}', zorder=2)
            print(f"  solar scale factor = {a:.4g}")
            ax.legend(handles=[l1, l2], loc='upper right', fontsize=9)

        else:
            raise ValueError(
                f"unknown mode {mode!r} -- use 'overlay', 'scaled', 'twin', "
                "'common', or 'reflectance'")

        ax.set_title(title or "CUTE Mars vs solar spectrum")
        fig.tight_layout()

        if save:
            if output_dir is None:
                raise ValueError("save=True needs an output_dir")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f"Saved {path}")
        if show:
            plt.show()
        return fig, ax

    @classmethod
    def plot_solar_percent_diff(cls, items, ref_label='visit1',
                                wmin=2900.0, wmax=3300.0, grid_A=1.0, smooth_A=10,
                                band_nm=(248.0, 330.6), figsize=(9, 11),
                                save=False, output_dir=None, dpi=200, show=True,
                                **refl_kw):
        """
        Three stacked panels (no titles):
          1. integrated solar flux at Mars per visit  (scatter + faint best-fit)
          2. geometric-albedo (I/F) % difference from `ref_label`, 2900-3300 A
          3. solar-flux-at-Mars  % difference from `ref_label`, 2900-3300 A

        `items`: the same 6-tuples used by the grid mode --
                 (label, cute_csv, solar_file, mars_earth_au, phase_angle, f_slit)
        Panels 2 & 3 interpolate every visit onto a common grid and reference V1.
        Comparing them is the diagnostic: if the albedo %diff tracks the solar
        %diff, the visit-to-visit change is residual solar; if the albedo is flat
        while the solar varies, the reduction removed the solar cleanly.
        """
        import re
        import numpy as np
        import matplotlib.pyplot as plt
        trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

        def vnum(lbl):
            m = re.search(r'(\d+)', str(lbl))
            return int(m.group(1)) if m else -1

        def gsmooth(y, sig):
            if sig <= 0:
                return y
            n = int(np.ceil(4 * sig))
            k = np.exp(-0.5 * (np.arange(-n, n + 1) / sig) ** 2); k /= k.sum()
            return np.convolve(y, k, mode='same')

        items = sorted(items, key=lambda it: vnum(it[0]))
        labels = [it[0] for it in items]
        vnums = [vnum(l) for l in labels]
        ref_i = labels.index(ref_label) if ref_label in labels else 0
        #colors = plt.cm.viridis(np.linspace(0, 0.92, len(items)))
        _PALETTE = ['#4477AA', '#EE6677', '#228833', '#AA3377', '#EE7733',
                    '#009988', '#CCBB44', '#000000', '#999999', '#882255']
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(items))]
        cw_g = np.arange(wmin, wmax + grid_A, grid_A)
        sig = (smooth_A / 2.3548) / grid_A

        # panel 1: integrated flux at Mars
        integ = []
        for (_lab, _cc, solar_file, _mea, _pa, _fs) in items:
            d = np.genfromtxt(solar_file, delimiter=',', comments='#')
            wl, irr = d[:, 0], d[:, 1]                      # nm, W/m^2/nm at Mars
            m = (wl >= band_nm[0]) & (wl <= band_nm[1]) & np.isfinite(irr)
            integ.append(trap(irr[m], wl[m]))
        integ = np.array(integ)

        # panel 2: % diff of BAND-INTEGRATED geometric albedo (one value per visit)
        alb_int = []
        for (label, cute_csv, solar_file, mea, pa, fs) in items:
            cw, refl, *_ = cls.reflectance_curve(cute_csv, solar_file, mea,
                                                 phase_angle=pa, f_slit=fs, **refl_kw)
            m = (cw >= wmin) & (cw <= wmax) & np.isfinite(refl)
            alb_int.append(trap(refl[m], cw[m]))        # integral over 2900-3300 A
        alb_int = np.array(alb_int)
        alb_int_pd = (alb_int - alb_int[ref_i]) / alb_int[ref_i] * 100.0

        # panel 3: % diff of BAND-INTEGRATED solar flux at Mars (one value per visit)
        sol_int = []
        for (_lab, _cc, solar_file, _mea, _pa, _fs) in items:
            d = np.genfromtxt(solar_file, delimiter=',', comments='#')
            wA, irr = d[:, 0] * 10.0, d[:, 1]                      # nm->A, at Mars
            m = (wA >= wmin) & (wA <= wmax) & np.isfinite(irr)
            sol_int.append(trap(irr[m], wA[m]))                    # integral over 2900-3300 A
        sol_int = np.array(sol_int)
        sol_int_pd = (sol_int - sol_int[ref_i]) / sol_int[ref_i] * 100.0

        fig, ax = plt.subplots(3, 1, figsize=figsize)

        ax[0].scatter(vnums, integ, c=colors, s=55, zorder=3, edgecolor='k', lw=0.4)
        cfit = np.polyfit(vnums, integ, 1)
        xx = np.linspace(min(vnums), max(vnums), 50)
        ax[0].plot(xx, np.polyval(cfit, xx), color='gray', lw=1.0, alpha=0.35, zorder=1)
        ax[0].set_xlabel('visit number')
        ax[0].set_ylabel(r'integrated flux at Mars (W m$^{-2}$)')
        ax[0].set_xticks(vnums)
        ax[0].grid(True, ls='--', lw=0.5, alpha=0.5)

        ax[1].scatter(vnums, alb_int_pd, c=colors, s=55, zorder=3, edgecolor='k', lw=0.4)
        c1 = np.polyfit(vnums, alb_int_pd, 1)
        ax[1].plot(xx, np.polyval(c1, xx), color='gray', lw=1.0, alpha=0.35, zorder=1)
        ax[1].axhline(0, color='k', lw=0.8, alpha=0.4)
        ax[1].set_xlabel('visit number')
        ax[1].set_ylabel(f'integrated albedo % diff from {ref_label}')
        ax[1].set_xticks(vnums)
        ax[1].grid(True, ls='--', lw=0.5, alpha=0.5)

        ax[2].scatter(vnums, sol_int_pd, c=colors, s=55, zorder=3, edgecolor='k', lw=0.4)
        c2 = np.polyfit(vnums, sol_int_pd, 1)
        ax[2].plot(xx, np.polyval(c2, xx), color='gray', lw=1.0, alpha=0.35, zorder=1)
        ax[2].axhline(0, color='k', lw=0.8, alpha=0.4)
        ax[2].set_xlabel('visit number')
        ax[2].set_ylabel(f'integrated solar-at-Mars % diff from {ref_label}')
        ax[2].set_xticks(vnums)
        ax[2].grid(True, ls='--', lw=0.5, alpha=0.5)
        fig.tight_layout()
        if save:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, 'solar_percent_diff.png')
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f"Saved {path}")
        if show:
            plt.show()
        return fig, ax

    # --------- frame-to-frame comparison ---------
    @classmethod
    def plot_visit_overlay(cls, visit, reference=None, ax=None, box_pts=15,
                           xlim=(2490, 3250), ylim=None, palette=None,
                           dashes=None, lw=1.3, alpha=0.9, title=None,
                           legend=True, legend_loc=None, legend_fontsize=8,
                           legend_ncol=1, pattern='*.fits', skip_frmid=None,
                           save=False, output_dir=None, dpi=200, show=False):
        '''
        Every frame of ONE visit drawn over each other on a single axes, so
        the change in the spectrum across the visit is visible at a glance.

        Each frame gets its own color from `palette` (default OVERLAY_COLORS,
        8 hues picked to stay separable, including for colorblind readers) and
        is named in a legend by frmid. A visit with more frames than hues
        reuses the hues with the next linestyle from `dashes`, so at the 15
        frames a visit tops out at you get 8 solid + 7 dashed -- every frame
        still uniquely identifiable, which a continuous colormap could not do.

        `ylim=None` autoscales to the frames inside `xlim`.
        `legend_loc=None` parks the legend outside the axes on the right for a
        standalone figure; pass e.g. 'upper left' to keep it inside (what the
        grid does, where there is no room beside each panel).

        Returns (fig, ax).
        '''
        if reference is None:
            reference = CuteReference()
        colors = list(palette) if palette else list(OVERLAY_COLORS)
        styles = list(dashes) if dashes else list(OVERLAY_DASHES)

        files = _visit_files(visit, pattern=pattern, skip_frmid=skip_frmid)
        n = len(files)
        if n > len(colors) * len(styles):
            print(f"  {visit}: {n} frames exceeds "
                  f"{len(colors) * len(styles)} color/linestyle combinations "
                  f"-- some frames will look identical")

        own = ax is None
        fig, ax = plt.subplots(figsize=(11, 5)) if own else (ax.figure, ax)

        wave = reference.wv_soln
        in_band = ((wave >= xlim[0]) & (wave <= xlim[1])) if xlim \
            else np.ones_like(wave, dtype=bool)

        peak = 0.0
        for i, f in enumerate(files):
            obs = cls(f, reference, visit=visit)
            y = smooth(obs.flux, box_pts)
            ax.plot(wave, y,
                    color=colors[i % len(colors)],
                    linestyle=styles[(i // len(colors)) % len(styles)],
                    lw=lw, alpha=alpha, label=str(obs.frame_id))
            if np.any(in_band):
                peak = max(peak, float(np.nanmax(y[in_band])))

        ax.set_title(title if title is not None else
                     f"Mars NUV Spectra with CUTE - {visit} overlap")
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel(
            r"Flux ($10^{-9}\ \mathrm{erg}\ \mathrm{s}^{-1}\ \mathrm{cm}^{-2}\ \mathrm{\AA}^{-1}$)"
        )
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylim(ylim if ylim else (0.0, 1.15 * peak if peak > 0 else 1.0))
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

        if legend and n:
            if legend_loc is None:      # outside, in the old colorbar's slot
                leg = ax.legend(title="frmid", loc='center left',
                                bbox_to_anchor=(1.01, 0.5),
                                fontsize=legend_fontsize, ncol=legend_ncol,
                                frameon=False, handlelength=2.4,
                                labelspacing=0.35)
            else:
                leg = ax.legend(title="frmid", loc=legend_loc,
                                fontsize=legend_fontsize, ncol=legend_ncol,
                                framealpha=0.85, handlelength=1.8,
                                labelspacing=0.25, columnspacing=0.9)
            leg.get_title().set_fontsize(legend_fontsize)

        if own:
            # leave room for the legend column instead of tight_layout, which
            # doesn't account for artists parked outside the axes
            fig.subplots_adjust(left=0.08, right=0.86, top=0.92, bottom=0.12)

        if save:
            if output_dir is None:
                raise ValueError("save=True needs an output_dir")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"{visit}_overlay{cls.fit_tag()}.png")
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f"Saved {path}  ({n} frames)")

        if show:
            plt.show()
        return fig, ax

    @classmethod
    def plot_overlay_grid(cls, visits, reference=None, ncols=4, box_pts=15,
                          xlim=(2490, 3250), ylim=None, palette=None,
                          dashes=None, share_ylim=True,
                          suptitle="2025 Mars NUV Spectra with CUTE - all frames per visit",
                          pattern='*.fits', skip_frmid=None,
                          save=False, output_dir=None, dpi=200):
        '''
        One overlay panel per visit in a single static figure -- the still
        counterpart to animate_grid. Each panel carries its own compact frmid
        legend (frame ids differ between visits); `share_ylim` puts every
        panel on one flux scale so visits are comparable too.
        '''
        if reference is None:
            reference = CuteReference()

        usable = []
        for v in visits:
            files = _visit_files(v, pattern=pattern, skip_frmid=skip_frmid,
                                 required=False)
            if files:
                usable.append(v)
            else:
                print(f"  {v}: no frames left, leaving it out of the grid")
        if not usable:
            raise ValueError("No frames left after applying skip_frmid")

        nrows = int(np.ceil(len(usable) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows),
                                 squeeze=False, constrained_layout=True)
        axes_flat = axes.ravel()
        for ax in axes_flat[len(usable):]:
            ax.axis('off')
        fig.suptitle(suptitle, fontsize=16)

        for ax, v in zip(axes_flat, usable):
            cls.plot_visit_overlay(v, reference=reference, ax=ax,
                                   box_pts=box_pts, xlim=xlim, ylim=ylim,
                                   palette=palette, dashes=dashes, lw=1.0,
                                   title=v, legend=True,
                                   legend_loc='upper left',
                                   legend_fontsize=5.5, legend_ncol=3,
                                   pattern=pattern, skip_frmid=skip_frmid)

        if share_ylim and ylim is None:
            top = max(ax.get_ylim()[1] for ax in axes_flat[:len(usable)])
            for ax in axes_flat[:len(usable)]:
                ax.set_ylim(0.0, top)

        if save:
            if output_dir is None:
                raise ValueError("save=True needs an output_dir")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"grid_overlay{cls.fit_tag()}.png")
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f"Saved {path}")
        plt.show()
        return fig, axes

    # ---- palette reused from the overlay plots -------------------------------
    OVERLAY_COLORS = plt.cm.viridis(np.linspace(0, 0.85, 12))

    @classmethod
    def plot_brightness_correlation(cls, visit, brightest_frmid, csv_dir,
                                    xlim=(2490, 3250), box_pts=15,
                                    dim_alpha=0.50, bright_color='gold',
                                    palette=None, save=False, output_dir=None,
                                    dpi=200, show=True):
        """
        Two-panel figure built from the per-frame CSVs in
        Spectra/Trace_Tracking/<visit>/csv/.

        Top:    every frame's flux(wavelength) overlaid; the brightest frame
                drawn in `bright_color` at full alpha / high zorder, all others
                dimmed to `dim_alpha`.
        Bottom: for each non-brightest frame, a scatter of its flux vs the
                brightest frame's flux (over the xlim band) with a 1:1 dashed
                reference line; Pearson r is shown per frame in the legend.
        """
        # ---- gather CSVs ------------------------------------------------------
        paths = sorted(glob.glob(os.path.join(csv_dir, '*.csv')),
                       key=lambda p: (cls._frmid_of(p) or 0))
        if not paths:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        frames = {}          # frmid -> (wave, flux)
        for p in paths:
            fid = cls._frmid_of(p)
            if fid is None:
                continue
            d = np.genfromtxt(p, delimiter=',', names=True)
            # tolerate iSLAT-style header renames
            wname = 'wave' if 'wave' in d.dtype.names else d.dtype.names[0]
            fname = 'flux' if 'flux' in d.dtype.names else d.dtype.names[1]
            frames[fid] = (np.asarray(d[wname]), np.asarray(d[fname]))

        if brightest_frmid not in frames:
            raise KeyError(
                f"Brightest frame {brightest_frmid} not among CSVs in {csv_dir} "
                f"(found {sorted(frames)})")

        wv_b, fx_b = frames[brightest_frmid]
        fx_b_s = smooth(fx_b, box_pts)

        others = [fid for fid in frames if fid != brightest_frmid]
        colors = palette if palette is not None else cls.OVERLAY_COLORS

        # ---- figure -----------------------------------------------------------
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 9),
            gridspec_kw={'height_ratios': [1.4, 1.0]})

        # ---- top: overlaid spectra -------------------------------------------
        for i, fid in enumerate(others):
            wv, fx = frames[fid]
            ax_top.plot(wv, smooth(fx, box_pts),
                        color=colors[i % len(colors)], lw=1.0,
                        alpha=dim_alpha, zorder=2, label=f"{fid}")
        ax_top.plot(wv_b, fx_b_s, color=bright_color, lw=2.5, alpha=1.0,
                    zorder=5, label=f"{brightest_frmid}")

        ax_top.set_xlim(*xlim)
        ax_top.set_xlabel(r'Wavelength [$\AA$]')
        ax_top.set_ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
        ax_top.set_title(f'{visit}  -  all frames')
        ax_top.grid(True, alpha=0.5)
        ax_top.legend(fontsize=7, ncol=2, loc='upper right')

        # ---- bottom: correlation vs brightest --------------------------------
        band = (wv_b >= xlim[0]) & (wv_b <= xlim[1])
        xb = fx_b_s[band]
        for i, fid in enumerate(others):
            wv, fx = frames[fid]
            fx_s = smooth(fx, box_pts)
            # align to the brightest wavelength grid if lengths differ
            if fx_s.shape != fx_b_s.shape:
                fx_s = np.interp(wv_b, wv, fx_s)
            y = fx_s[band]
            r = np.corrcoef(xb, y)[0, 1]
            ax_bot.scatter(xb, y, s=6, color=colors[i % len(colors)],
                           alpha=0.5, label=f"{fid}  (r={r:.3f})")

        lo = min(xb.min(), 0)
        hi = xb.max()
        ax_bot.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=1,
                    label='1:1')
        ax_bot.set_xlabel(f'frmid {brightest_frmid} flux')
        ax_bot.set_ylabel('Other-frame flux')
        ax_bot.set_title('Correlation with brightest frame')
        ax_bot.legend(fontsize=7, ncol=2, loc='upper left')

        fig.suptitle(f'2025 Mars NUV Spectra with CUTE  -  {visit}',
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f'{visit}_brightness_correlation.png')
            fig.savefig(path, dpi=dpi)
            print(f"Saved {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # --------- animations ---------
    @classmethod
    def animate_visit(cls, visit, reference=None, kind='both', fps=5,
                    box_pts=15, xlim=(2490, 3250), ylim=None,
                    vmin=None, vmax=None, pattern='*.fits',
                    save=False, output_dir=None, skip_frmid=None,
                    show=True):
        '''
        Animate every FITS frame in a visit folder, in frame-id order, and DISPLAY
        it (nothing is saved). Two separate windows by default:
            kind='spectrum' : 1D spectrum movie
            kind='trace'    : 2D trace + boundaries movie
            kind='both'     : both windows at once (default)
            save=False      : DISPLAY on screen only
            save=True       : write on GIF per kind into output_dir
            show=False      : write the GIFs without opening a window, which is
                              what animate_all_visits needs to run unattended
        '''
        if reference is None:
            reference = CuteReference()
        skip = {int(s) for s in (skip_frmid or [])}       # frame ids to drop

        folder = os.path.join(_get_observations_dir(), visit)
        files = sorted(glob.glob(os.path.join(folder, pattern)), key=cls._frmid_of)
        if not files:
            raise FileNotFoundError(f'No {pattern!r} files in {folder}')
        files = [f for f in files if cls._frmid_of(f) not in skip]   # drop skipped
        if not files:
            raise ValueError(f"No frames left in {visit} after applying skip_frmid")
        if save and output_dir is None:
            raise ValueError("save=True needs an output_dir")

        kinds = ['spectrum', 'trace'] if kind == 'both' else [kind]
        # keep the FuncAnimation objects referenced until they're shown/saved
        anims = []
        for k in kinds:
            anim = cls._animate_one(files, reference, visit, k, fps,
                                    box_pts, xlim, ylim, vmin, vmax)
            anims.append(anim)
            if save:
                gif_path = os.path.join(output_dir, f"{visit}_{k}{cls.fit_tag()}.gif")
                anim.save(gif_path, writer=PillowWriter(fps=fps))
                print(f"Saved {gif_path}")

        if show:
            plt.show()      # display, whether or not we saved
        return anims

    @classmethod
    def animate_all_visits(cls, visits, reference=None, output_root=None,
                           kind='both', fps=5, box_pts=15, **kwargs):
        '''
        Write the GIFs for EVERY visit -- the movie counterpart of
        save_all_frames. Each visit's GIFs land in <output_root>/<visit>/, the
        same place MODE 'visit' puts them, so a batch run and a one-off run
        agree.

        Nothing is displayed: each visit's figures are closed as soon as its
        GIFs are written, so a long batch doesn't accumulate open windows.
        A visit with no usable frames is reported and stepped over instead of
        killing the run. This re-reduces every frame twice per kind, so expect
        it to take about as long as running MODE 'visit' on each visit in turn.

        Returns {visit: [gif paths]}.
        '''
        if reference is None:
            reference = CuteReference()
        if output_root is None:
            raise ValueError("animate_all_visits needs an output_root")
        kinds = ['spectrum', 'trace'] if kind == 'both' else [kind]

        results = {}
        for v in visits:
            out_path = os.path.join(output_root, v)
            os.makedirs(out_path, exist_ok=True)
            try:
                cls.animate_visit(visit=v, reference=reference, kind=kind,
                                  fps=fps, box_pts=box_pts, save=True,
                                  output_dir=out_path, show=False, **kwargs)
                results[v] = [os.path.join(out_path, f"{v}_{k}.gif")
                              for k in kinds]
            except (FileNotFoundError, ValueError) as err:
                print(f"  {v}: skipped ({err})")
                results[v] = []
            finally:
                plt.close('all')    # a batch otherwise holds every figure open

        total = sum(len(p) for p in results.values())
        print(f"Done: {total} GIF(s) across {len(results)} visit(s) "
              f"-> {output_root}")
        return results

    @classmethod
    def _animate_one(cls, files, reference, visit, kind, fps,
                    box_pts, xlim, ylim, vmin, vmax):
        n = len(files)
        first = cls(files[0], reference, visit=visit)   # probe for stable scaling

        if kind == 'spectrum':
            fig, ax = plt.subplots(figsize=(10, 4))
            if ylim is None:                            # fixed y-range = no flicker
                ylim = (0.0, 1.15 * float(np.nanmax(first.flux)))

            def draw(i):
                ax.clear()
                cls(files[i], reference, visit=visit).plot_spectrum(
                    ax=ax, box_pts=15, xlim=xlim, ylim=ylim)

        elif kind == 'trace':
            fig, ax = plt.subplots()
            if vmin is None:
                vmin = float(np.percentile(first.img, 5))
            if vmax is None:
                vmax = float(np.percentile(first.img, 99))
            fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax)), ax=ax)

            def draw(i):
                ax.clear()
                cls(files[i], reference, visit=visit).plot_trace(
                    ax=ax, vmin=vmin, vmax=vmax)

        else:
            raise ValueError(
                f"kind must be 'spectrum', 'trace', or 'both', not {kind!r}")

        return FuncAnimation(fig, draw, frames=n, interval=1000 / fps)

    # --------- crazy plots ---------
    @classmethod
    def animate_grid(
        cls, visits, reference=None, fps=5, box_pts=15,
        xlim=(2490, 3250), ylim=None, ncols=4,
        suptitle="2025 Mars NUV Spectra with CUTE",
        save=False, output_dir=None, pattern='*fits',
        skip_frmid=None,
    ):
        """
        Grid movie of all visits 1D spectra.
        One subplot per visit
        """

        if reference is None:
            reference = CuteReference()
        skip = {int(s) for s in (skip_frmid or [])}   # frame ids to drop

        # Assigning each visit an individual color
        cmap = plt.colormaps.get_cmap('tab10')
        visit_colors = {v: cmap(i % 10) for i, v in enumerate(visits)}

        visit_files = {}
        for v in visits:
            folder = os.path.join(_get_observations_dir(), v)
            files = sorted(glob.glob(os.path.join(folder, pattern)), key=cls._frmid_of)
            if not files:
                raise FileNotFoundError(f'No {pattern!r} files found in {folder}')
            files = [f for f in files if cls._frmid_of(f) not in skip]   # drop skipped
            if not files:
                print(f"  {v}: all frames skipped, leaving it out of the grid")
                continue
            visit_files[v] = files
        if not visit_files:
            raise ValueError("No frames left after applying skip_frmid")
        n_anim = max(len(f) for f in visit_files.values())

        n_panels = len(visit_files)                   # use the FILTERED count
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False, constrained_layout=True)
        axes_flat = axes.ravel()
        for ax in axes_flat[n_panels:]:               # hide leftover cells
            ax.axis('off')
        fig.suptitle(suptitle, fontsize=16)

        # one shared y-range across all panels (comparable + no flicker)
        if ylim is None:
            peaks = [float(np.nanmax(cls(f[0], reference, visit=v).flux))
                    for v, f in visit_files.items()]
            ylim = (0.0, 1.15 * max(peaks))

        def draw(i):
            for ax, (v, files) in zip(axes_flat, visit_files.items()):
                ax.clear()
                idx = min(i, len(files) - 1)     # hold last frame for shorter visits
                obs = cls(files[idx], reference, visit=v)
                obs.plot_spectrum(ax=ax, box_pts=15, xlim=xlim, ylim=ylim,
                                  color=visit_colors[v],
                                  title=f"{v} frmid {obs.frame_id}")

        anim = FuncAnimation(fig, draw, frames=n_anim, interval=1000 / fps)
        if save:
            if output_dir is None:
                raise ValueError("save=True needs an output_dir")
            path = os.path.join(output_dir, f"grid_spectra{cls.fit_tag()}.gif")
            anim.save(path, writer=PillowWriter(fps=fps))
            print(f"Saved {path}")
        plt.show()
        return anim

    @classmethod
    def animate_sequence(cls, visits, reference=None, fps=5, box_pts=15,
                        xlim=(2490, 3250), ylim=None,
                        suptitle="2025 Mars NUV Spectra with CUTE",
                        save=False, output_dir=None, pattern='*.fits',
                        skip_frmid=None):
        '''
        One 1D-spectrum panel that plays through EVERY frame of each visit in
        turn: all of visits[0], then all of visits[1], ... Frame title is
        "Visit# frmid ####"; the figure carries `suptitle`.
        save=False -> display; save=True -> write sequence_spectra.gif.
        '''
        if reference is None:
            reference = CuteReference()
        skip = {int(s) for s in (skip_frmid or [])}   # frame ids to drop

        # Assigning each visit an individual color
        cmap = plt.colormaps.get_cmap('tab10')
        visit_colors = {v: cmap(i % 10) for i, v in enumerate(visits)}

        seq = []
        for v in visits:
            folder = os.path.join(_get_observations_dir(), v)
            files = sorted(glob.glob(os.path.join(folder, pattern)), key=cls._frmid_of)
            if not files:
                raise FileNotFoundError(f'No {pattern!r} files found in {folder}')
            seq.extend((v, f) for f in files if cls._frmid_of(f) not in skip)
        if not seq:
            raise ValueError("No frames left after applying skip_frmid")




        # one shared y-range (sample the first frame of each visit) -> no flicker
        if ylim is None:
            peaks, seen = [], set()
            for v, f in seq:
                if v not in seen:
                    seen.add(v)
                    peaks.append(float(np.nanmax(cls(f, reference, visit=v).flux)))
            ylim = (0.0, 1.15 * max(peaks))

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(suptitle, fontsize=14)

        def draw(i):
            ax.clear()
            v, f = seq[i]
            obs = cls(f, reference, visit=v)
            obs.plot_spectrum(ax=ax, box_pts=box_pts, xlim=xlim, ylim=ylim,
                            color=visit_colors[v],
                            title=f"{v} frmid {obs.frame_id}")

        anim = FuncAnimation(fig, draw, frames=len(seq), interval=1000 / fps)
        if save:
            if output_dir is None:
                raise ValueError("save=True needs an output_dir")
            path = os.path.join(output_dir, f"sequence_spectra{cls.fit_tag()}.gif")
            anim.save(path, writer=PillowWriter(fps=fps))
            print(f"Saved {path}  ({len(seq)} frames)")
        plt.show()
        return anim