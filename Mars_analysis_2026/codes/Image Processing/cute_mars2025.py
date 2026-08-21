'''
Image Processing for the NASA CUTE (Colorado Ultraviolet Transit Experiment)
2025 Mars Observations in Near Ultraviolet
@Author: jose5987
Date Created: 8/20/2026

Based off previous processing logic written by dobh6980
'''

# Relevant Imports
import os                            # file handling
import numpy as np                   # computation
from astropy.io import fits          # fit handling
from numpy.typing import NDArray     # type hinting
import matplotlib.pyplot as plt      # plotting
# Animation (gif)
from matplotlib.animation import FuncAnimation, PillowWriter 
import glob
import re

# ====================================================

def _default_base_dir() -> str:
    '''
    All input files are read from the folder this script lives in,
    so the whole folder is portable.
    '''
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def smooth(y, box_pts):
    '''
    Smoothing 1D spectrum
    '''
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

# ------------------------------

class CuteReference:
    '''
    Shared calibration data used by every visit.

    wave_sol : wavelength for each of the 2048 science pixels
    eff_area : effective area, resampled onto wave_sol
    '''

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or _default_base_dir()
        self.wave_sol = self._load_wavelength_solution()
        self.eff_area = self._load_effective_area()

    def _load_wavelength_solution(self):
        path = os.path.join(self.base_dir,
                            'flight_quad_wavelength_solution_final.dat')
        # column 0 is wavelength; 1-line header
        return np.loadtxt(path, skiprows=1, usecols=0)

    def _load_effective_area(self):
        path = os.path.join(self.base_dir,
                            'cute_recalculated_effa_2025.txt')
        # 2-line header; col 0 = eff area, col 1 = wavelength
        data = np.loadtxt(path, skiprows=2)
        eff_area, wv_val = data[:, 0], data[:, 1]
        # resample onto pixel wavelength grid
        return np.interp(self.wave_sol, wv_val, eff_area)

# ------------------------------

class CuteVisit:
    '''
    One CUTE exposure ("visit")
    '''

    # --------- constants -> class attributes ---------
    GAIN = 1.5                  # electrons per DN
    H = 6.626075540e-27         # erg s
    C = 2.99792458e+10          # cm / s
    N_SCI_PIX = 2048            # science pixels (excludes overscan)

    def __init__(self, fname, reference: CuteReference, base_dir=None):
        self.fname = fname
        self.reference = reference
        self.base_dir = base_dir or reference.base_dir

        self.img = self._load_image()
        self.nx = self.img.shape[1]
        self._build_regions()

        # Processing Steps
        self.spectrum_dn = None     # summed dark-subtracted counts
        self.spectrum_phot = None   # calibrated spectrum

    # ---------- loading ----------
    def _load_image(self):
        path = os.path.join(self.base_dir, self.fname)
        with fits.open(path, memmap=False) as ff:
            # flip left-right so column number increases with wavelength
            return np.fliplr(ff[0].data)

    # ---------- geometry ----------
    def _build_regions(self):
        '''
        Define upper/lower edges of the science trace and of
        the background strip, as a y-value for each science
        column.
        '''
        nx = self.nx

        # science trace edges
        y1_sc, y2_sc = 37 - 1, 69 - 1     # lower edge: left, right
        y3_sc, y4_sc = 59 - 1, 88 - 1     # upper edge: left, right
        # background (dark) strip edges, just below the trace
        y1_dk, y2_dk = 7 - 1, 39 - 1
        y3_dk, y4_dk = 29 - 1, 58 - 1

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

    # ---------- reduction ----------
    def extract_spectrum(self):
        '''
        For each column, take the median dark level,
        then sum dark-subtracted counts inside the trace.
        Returns a 2048-long spectrum already running short->long
        wavelengths (image was flipped)
        '''
        spectrum = np.zeros(self.N_SCI_PIX, dtype=float)

        for i in range(self.N_SCI_PIX):
            col = int(self.xval[i])

            dk_lo, dk_hi = int(self.yval1_dk[i]), int(self.yval2_dk[i])
            dark = np.median(self.img[dk_lo:dk_hi, col])

            sc_lo, sc_hi = int(self.yval1_sc[i]), int(self.yval2_sc[i])
            spectrum[i] = np.sum(self.img[sc_lo:sc_hi, col] - dark)

        self.spectrum_dn = spectrum
        return spectrum

    def to_photons(self) -> NDArray:
        '''
        Convert the DN spectrum to photons using gain + effective area.
        '''
        if self.spectrum_dn is None:
            self.extract_spectrum()

        wave_sol = self.reference.wave_sol
        eff_area = self.reference.eff_area

        spec_e = self.spectrum_dn * (self.GAIN / 100.0)          # DN -> e-
        self.spectrum_phot = spec_e * (self.H * self.C) / (
            (wave_sol * 1e-8) * eff_area)                        # e- -> photons
        return self.spectrum_phot

    def process(self) -> NDArray:
        '''
        Convenience: run the whole reduction
        and return the photon spectrum.
        '''
        self.extract_spectrum()
        return self.to_photons()

    # ---------- diagnostics / output ----------
    '''
    Two '_draw_*' helpers put their content on an axes that is passed in.
    That way standalone views and the combined two-panel view all share
    *one definition of each plot.
    '''
    def _draw_regions(self, ax, vmin=None, vmax=None, title=None):
        '''
        Draw the image with the trace (red) and background (yellow) boundaries
        onto `ax`. If vmin/vmax are None they are chosen from the image, so each
        visit scales to its own levels. Returns the image handle (for colorbar).
        '''
        if vmin is None:
            vmin = np.percentile(self.img, 5)     # dark end
        if vmax is None:
            vmax = np.percentile(self.img, 99)    # bright end
 
        ax.set_title(title or self.fname)
        ax.plot(self.xval, self.yval1_sc, lw=3, color='red')
        ax.plot(self.xval, self.yval2_sc, lw=3, color='red')
        ax.plot(self.xval, self.yval1_dk, lw=3, color='yellow')
        ax.plot(self.xval, self.yval2_dk, lw=3, color='yellow')
        im = ax.imshow(self.img, vmin=vmin, vmax=vmax, origin='lower',
                       aspect='auto', interpolation='none')
        return im
 
    def _draw_spectrum(self, ax, title='Mars NUV Spectra with CUTE', box_pts=15):
        '''Draw the calibrated, smoothed 1D spectrum onto `ax`.'''
        if self.spectrum_phot is None:
            self.to_photons()
 
        ax.grid(color='gray', linestyle='dashed')
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=13)
        ax.set_ylabel(r'Flux (10$^{-9}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)',
                      fontsize=13)
        ax.set_xlim(2500, 3250)
        ax.set_ylim(0., 2.)
        ax.plot(self.reference.wave_sol,
                smooth(self.spectrum_phot, box_pts) * 1e9,
                color='darkred', lw=2)
 
    def plot_regions(self, vmin=None, vmax=None, title=None):
        '''Standalone region-check image.'''
        fig, ax = plt.subplots()
        im = self._draw_regions(ax, vmin, vmax, title)
        fig.colorbar(im, ax=ax)
        plt.show()
 
    def show_spectrum(self, title='Mars NUV Spectra with CUTE', box_pts=15):
        '''Standalone 1D spectrum (no saving).'''
        fig, ax = plt.subplots(figsize=(10, 3.5))
        self._draw_spectrum(ax, title, box_pts)
        fig.tight_layout()
        plt.show()
 
    def show_summary(self, title=None, box_pts=15):
        '''
        Two-panel figure: the region-check image on top and the reduced 1D
        spectrum below, so you can see whether the trace boundaries line up
        with where the signal actually falls.
        '''
        label = title or self.fname
        fig, (ax_img, ax_spec) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={'height_ratios': [1, 1]})
 
        im = self._draw_regions(ax_img, title=f'Mars ({label})')
        fig.colorbar(im, ax=ax_img)
        self._draw_spectrum(ax_spec, title=f'{label} NUV Spectra with CUTE',
                            box_pts=box_pts)
 
        fig.tight_layout()
        plt.show()

# ------------------------------

class CuteMovie:
    '''
    Builds an animated GIf from every FITS frame in one visit folder

    Each movie frame is the two-panel view (region-check on top, 1D spectrum below)
    for one FITS file, played in frame-ID order. 
    Pulls from CuteVisit to loop over frames.
    '''

    def __init__(self, folder, reference: CuteReference, pattern='*.fits'):
        self.folder = folder
        self.reference = reference
        self.files = self._gather_files(pattern)
        if not self.files:
            raise FileNotFoundError(
                f'No files matching {pattern!r} in {folder}')
 
    def _gather_files(self, pattern):
        '''Find the FITS files and order them by frame id (frmid_XXXX).'''
        paths = glob.glob(os.path.join(self.folder, pattern))
 
        def sort_key(path):
            m = re.search(r'frmid_(\d+)', os.path.basename(path))
            # numeric frame id if present, else fall back to the name
            return (0, int(m.group(1))) if m else (1, os.path.basename(path))
 
        return sorted(paths, key=sort_key)
 
    @staticmethod
    def _frame_label(path):
        name = os.path.basename(path)
        m = re.search(r'frmid_(\d+)', name)
        return f'frmid {m.group(1)}' if m else name
 
    def make_gif(self, outfile, fps=5, vmin=None, vmax=None, box_pts=15,
                 view='summary'):
        '''
        Render all frames to an animated GIF.
 
        fps        : frames per second in the movie
        vmin, vmax : image brightness scale. Left as None, they are fixed once
                     from the first frame so the movie doesn't flicker. Pass
                     numbers to force a scale that suits the whole set.
        view       : which panels to animate --
                     'summary'  -> region image + 1D spectrum (default)
                     'spectrum' -> 1D spectrum only
                     'regions'  -> region-check image only
        '''
        if view not in ('summary', 'spectrum', 'regions'):
            raise ValueError(
                f"view must be 'summary', 'spectrum', or 'regions', "
                f"not {view!r}")
 
        show_img = view in ('summary', 'regions')
        show_spec = view in ('summary', 'spectrum')
 
        # a fixed brightness scale is only needed when the image is shown
        if show_img and (vmin is None or vmax is None):
            first = CuteVisit(self.files[0], self.reference)
            if vmin is None:
                vmin = np.percentile(first.img, 5)
            if vmax is None:
                vmax = np.percentile(first.img, 99)
 
        visit_name = os.path.basename(os.path.normpath(self.folder))
        n = len(self.files)
 
        # build a 2-panel or 1-panel figure to match the chosen view
        if view == 'summary':
            fig, (ax_img, ax_spec) = plt.subplots(
                2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [1, 1]})
        elif view == 'regions':
            fig, ax_img = plt.subplots(figsize=(10, 4))
            ax_spec = None
        else:  # 'spectrum'
            fig, ax_spec = plt.subplots(figsize=(10, 3.5))
            ax_img = None
 
        def draw_frame(i):
            path = self.files[i]
            print(f'  frame {i + 1}/{n}: {os.path.basename(path)}')
            visit = CuteVisit(path, self.reference)
            visit.process()
            frame_title = f'{visit_name}  {self._frame_label(path)}'
 
            if ax_img is not None:
                ax_img.clear()
                visit._draw_regions(ax_img, vmin=vmin, vmax=vmax,
                                    title=frame_title)
            if ax_spec is not None:
                ax_spec.clear()
                # in spectrum-only mode, label each frame on the spectrum itself
                spec_title = (frame_title if view == 'spectrum'
                              else 'Mars NUV Spectra with CUTE')
                visit._draw_spectrum(ax_spec, title=spec_title, box_pts=box_pts)
            fig.tight_layout()
 
        # `anim` must stay referenced while the window is open, or Python
        # garbage-collects it and the animation freezes. plt.show() blocks
        # until you close the window, so keeping it local here is enough.
        anim = FuncAnimation(fig, draw_frame, frames=n,
                             interval=1000 / fps)
        print(f'  playing {n} frames (view={view}) -- close the window to continue')
        plt.show()
        plt.close(fig)