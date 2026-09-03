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
from astropy.io import fits          # fit handling
import matplotlib.pyplot as plt      # plotting

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
    Accepts either a full filename ('...frmid_4868_...fits') or a frame id
    (the int 4868, or the string '4868'), which is looked up in the folder.
    '''
    s = str(file_or_frmid)
    if s.endswith('.fits'):                       # a full filename was given
        path = os.path.join(folder, s)
        if not os.path.exists(path):
            raise FileNotFoundError(f'FITS file not found at: {path}')
        return path

    frmid = int(file_or_frmid)                    # otherwise treat as a frame id
    hits = [p for p in glob.glob(os.path.join(folder, '*.fits'))
            if CuteObservation._frmid_of(p) == frmid]
    if not hits:
        raise FileNotFoundError(f'No FITS file with frmid {frmid} in {folder}')
    if len(hits) > 1:
        raise ValueError(f'Multiple files with frmid {frmid} in {folder}: '
                         f'{[os.path.basename(h) for h in hits]}')
    return hits[0]

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
        self._build_regions()                       # nominal fixed box
        if track:
            self.row_offset, self.trace_fwhm = self._measure_trace_shape()
            if widen and self.trace_fwhm > 0:       # widening now OFF by default
                i_mid = len(self.xval) // 2
                h0 = 0.5 * (self.yval2_sc[i_mid] - self.yval1_sc[i_mid])
                half_target = self.APERTURE * (self.trace_fwhm / 2.0)
                self.sci_grow = max(half_target - h0, 1 - h0)
            self._build_regions(row_shift=self.row_offset, sci_grow=self.sci_grow)

        self.spectra = self.extract_spectrum()
        self.flux = self._compute_flux()

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
        y1_dk, y2_dk = 7 - 1 + s, 39 - 1 + s
        y3_dk, y4_dk = 29 - 1 + s, 58 - 1 + s

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
        nrows = self.img.shape[0]
        clamp = lambda v: min(max(int(round(v)), 0), nrows)

        for i in range(2048):
            col = self.xval[i]

            yy1, yy2 = clamp(self.yval1_dk[i]), clamp(self.yval2_dk[i])
            dk_arr[i] = np.median(self.img[yy1:yy2, col]) if yy2 > yy1 else 0.0

            yy3, yy4 = clamp(self.yval1_sc[i]), clamp(self.yval2_sc[i])
            if yy4 > yy3:
                sc_arr[i] = np.sum(self.img[yy3:yy4, col] - dk_arr[i])

        return sc_arr

    def _compute_flux(self):
        """
        Converts DN spectrum to photon flux density
        in units of 10^-9 erg/s/cm^2/A
        """
        wave_sol = self.reference.wv_soln
        eff_area = self.reference.eff_area

        mars_spec_e = self.spectra * (self.GAIN / self.exptime)      # DN -> e-/s
        wave_cm = wave_sol * 1.0e-8                                  # Angstrom -> cm
        flux_cgs = (mars_spec_e * self.H * self.C) / (wave_cm * eff_area)

        return flux_cgs / 1.0e-9   

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
    def _frmid_of(path):
        '''
        Frame-id number from a filename, for sorting frames in order.
        '''
        parts = os.path.basename(path).split('_')
        if 'frmid' in parts:
            tok = parts[parts.index('frmid') + 1]
            return int(tok) if tok.isdigit() else tok
        return os.path.basename(path)

    @classmethod
    def animate_visit(cls, visit, reference=None, kind='both', fps=5,
                    box_pts=15, xlim=(2490, 3250), ylim=None,
                    vmin=None, vmax=None, pattern='*.fits',
                    save=False, output_dir=None, skip_frmid=None):
        '''
        Animate every FITS frame in a visit folder, in frame-id order, and DISPLAY
        it (nothing is saved). Two separate windows by default:
            kind='spectrum' : 1D spectrum movie
            kind='trace'    : 2D trace + boundaries movie
            kind='both'     : both windows at once (default)
            save=False      : DISPLAY on screen only
            save=True       : write on GIF per kind into output_dir
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
                gif_path = os.path.join(output_dir, f"{visit}_{k}.gif")
                anim.save(gif_path, writer=PillowWriter(fps=fps))
                print(f"Saved {gif_path}")

        plt.show()          # always display, whether or not we saved
        return anims

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
            path = os.path.join(output_dir, "grid_spectra.gif")
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
            path = os.path.join(output_dir, "sequence_spectra.gif")
            anim.save(path, writer=PillowWriter(fps=fps))
            print(f"Saved {path}  ({len(seq)} frames)")
        plt.show()
        return anim