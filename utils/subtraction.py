"""
Subtraction logicfor main.py that is able to applied to ALL planets
in the CUTE solar system observation campaign.

2D Extraction via specreduce Horne Extract method (fits cross-dispersion profile
per col, rejects hot pixels), 1D spectra via specutils

Assumes exposure time is 100s constant
(and normalizes flux to that assumption)

@author: jose5987
"""

#from pathlib import Path

import os, re, glob, csv
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from scipy.ndimage import median_filter
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- specreduce / specutils ---
from specreduce.tracing import FitTrace
from specutils import Spectrum
from specutils.manipulation import gaussian_smooth
from specreduce.background import Background
from specreduce.extract import HorneExtract


class Subtraction:
    # ---- instrument constants ----
    GAIN        = 1.5          # e-/DN
    READ_NOISE  = 3.6          # e-/pixel
    H           = 6.626075540e-27
    C           = 2.99792458e+10

    N_SCI_COL   = 2048   # science columns
    XLO_OFF   = 2100     # science cols = [nx-2100 : nx-52] -> 2048
    XHI_OFF   = 52           
    TRACE_BAND = (40, 90) # row band to search for trace

    HOT_WIN   = 7        # Hot pixel window
    HOT_SIGMA = 5.0      # Hot pixel std dev allowance

    # per-planet settings
    PLANET_CFG = {
            'mars':    dict(aper=18, bkg_sep=25, bkg_w=15),
            'uranus':  dict(aper=18, bkg_sep=25, bkg_w=15),
            'jupiter': dict(aper=18, bkg_sep=25, bkg_w=15),   # eff. area still WIP
        }

    # # ---- trace / background geometry ----
    # TRACE_WINDOW = 22          # rows searched around the peak when tracing
    # BKG_SEP      = 25          # rows from trace center to the dark strip
    # BKG_WIDTH    = 15          # dark-strip height (rows)
    # APER_WIDTH   = 18          # science aperture height fed to Horne (≈6σ)

    def __init__(self,planet:str, output_dir:str, obs_dir:str,
                 wv_soln_dir:str, eff_area_dir:str, save:bool = False,):
        self.save         = save
        self.planet       = planet                  # selected planet
        self.output_dir   = output_dir              # output directory for CSV and images
        self.obs_dir      = obs_dir                 # observation directory
        self.eff_area_dir = eff_area_dir            # effective area directory
        self.wv_soln_dir  = wv_soln_dir             # wavelength solution directory

        self.cfg = self.PLANET_CFG[planet]          # planet-specific geometry
        self.wv_soln  = self._get_wv_soln()         # wavelength solution array
        self.eff_area = self._get_effective_area()  # effective area array

    def _get_wv_soln(self):
        path = self.wv_soln_dir
        wave = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i > 0: wave.append(float(line.split()[0]))
        return np.array(wave)
    
    def _get_effective_area(self):
        path = self.eff_area_dir
        eff, wave = [], []
        with open(path) as f:
            for i, line in enumerate(f):
                if i > 1:
                    st = line.split(); eff.append(float(st[0])); wave.append(float(st[1]))
        return np.interp(self.wv_soln, np.array(wave), np.array(eff))

    def _load_frame(self, path):
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            img = np.fliplr(hdul[0].data.astype(float))
            for k in ('EXPTIME', 'EXPOSURE', 'EXPCMD'):
                if k in hdr: exptime = float(hdr[k]) / 1000.0; break
            else: raise KeyError(f"no exposure keyword in {path}")
        return img, exptime

    def _visits(self):
        return sorted(os.path.basename(p) for p in glob.glob(f"{self.obs_dir}/Visit*") if os.path.isdir(p))

    def _frame_paths(self, visit):
        return sorted(glob.glob(f"{self.obs_dir}/{visit}/**/*.fits", recursive=True))

    # ------------------ Reduction ------------------

    def _hot_mask(self, img):
        w = max(int(self.HOT_WIN) | 1, 3)
        medf = median_filter(img, size=(1, w), mode='nearest')
        resid = img - medf
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        return resid > self.HOT_SIGMA * 1.4826 * mad, medf

    def _trace_center(self, img):
        """Smooth trace center per column, robust to the hot-pixel field: suppress hot
        pixels (median along dispersion), background-subtract, centroid ONLY within the
        trace band, then fit a brightness-weighted, sigma-clipped polynomial so faint
        columns inherit the trace from the bright ones."""
        ny, nx = img.shape; rows = np.arange(ny)
        m = median_filter(img, size=(1, 7), mode='nearest')       # hot pixels suppressed
        sig = np.clip(m - np.median(m, axis=0), 0, None)
        lo, hi = self.TRACE_BAND
        w = sig * ((rows >= lo) & (rows <= hi))[:, None]
        tot = w.sum(0)
        cen = (rows[:, None] * w).sum(0) / np.clip(tot, 1e-9, None)
        x = np.arange(nx, dtype=float)
        thr = np.percentile(tot, 20)
        good = tot > thr
        p = np.polyfit(x[good], cen[good], 2, w=np.sqrt(tot[good]))
        for _ in range(3):
            r = cen - np.polyval(p, x)
            s = 1.4826 * np.median(np.abs(r[good] - np.median(r[good]))) + 1e-9
            good = (tot > thr) & (np.abs(r) < 3 * s)
            p = np.polyfit(x[good], cen[good], 2, w=np.sqrt(tot[good]))
        return np.polyval(p, x)

    def _region_rows(self, cen_x):
        a, sep, w = self.cfg['aper'], self.cfg['bkg_sep'], self.cfg['bkg_w']
        sci_lo = np.round(cen_x - a / 2).astype(int); sci_hi = np.round(cen_x + a / 2).astype(int)
        dkm = cen_x - sep
        dk_lo = np.round(dkm - w / 2).astype(int);    dk_hi = np.round(dkm + w / 2).astype(int)
        return sci_lo, sci_hi, dk_lo, dk_hi

    def _to_flux(self, dn, exptime):
        wave_cm = self.wv_soln * 1.0e-8
        conv = (self.GAIN / exptime) * (self.H * self.C) / (wave_cm * self.eff_area) / 1.0e-9
        return dn * conv

    def _profile_inpaint(self, img, hot, cen):
        """Replace each hot pixel with an estimate from an empirical cross-dispersion
        profile: one shared normalized shape (median over columns) scaled by a per-column
        amplitude. Every pixel is kept; only flagged ones are overwritten with the model."""
        ny, nx = img.shape
        a, sep, bw = self.cfg['aper'], self.cfg['bkg_sep'], self.cfg['bkg_w']
        W = a // 2; d = np.arange(-W, W + 1)
        out = img.astype(float).copy()

        # per-column background from the dark strip
        bg = np.empty(nx)
        for c in range(nx):
            dkm = int(round(cen[c] - sep))
            lo = max(dkm - bw // 2, 0); hi = min(dkm + bw // 2 + 1, ny)
            bg[c] = np.median(img[lo:hi, c]) if hi > lo else 0.0

        # shared normalized profile from bright, hot-free columns
        stack = []
        for c in range(nx):
            c0 = int(round(cen[c])); rr = c0 + d; ok = (rr >= 0) & (rr < ny)
            s = np.clip(img[rr[ok], c] - bg[c], 0, None).astype(float)
            s[hot[rr[ok], c]] = np.nan
            tot = np.nansum(s)
            if tot > 0 and np.isfinite(tot):
                row = np.full(d.size, np.nan); row[np.where(ok)[0]] = s / tot
                stack.append(row)
        P = np.nanmedian(np.array(stack), axis=0) if stack else np.ones(d.size)
        P = np.clip(np.nan_to_num(P), 0, None); P = P / P.sum() if P.sum() > 0 else P

        # per column: amplitude from hot-free pixels, build model, inpaint the hot ones
        for c in range(nx):
            c0 = int(round(cen[c])); rr = c0 + d; ok = (rr >= 0) & (rr < ny)
            rows = rr[ok]; Pc = P[ok]; h = hot[rows, c]
            sig = np.clip(img[rows, c] - bg[c], 0, None)
            denom = np.nansum(np.where(h, np.nan, Pc))
            amp = (np.nansum(np.where(h, np.nan, sig)) / denom) if denom > 0 else 0.0
            model = amp * Pc + bg[c]
            col = img[rows, c].astype(float); col[h] = model[h]
            out[rows, c] = col
            # dark strip: replace hot pixels with the local dark median (kills down-spikes)
            dkm = int(round(cen[c] - sep)); dlo = max(dkm - bw // 2, 0); dhi = min(dkm + bw // 2 + 1, ny)
            dcol = img[dlo:dhi, c].astype(float); dh = hot[dlo:dhi, c]
            if dh.any():
                dcol[dh] = np.median(dcol[~dh]) if (~dh).any() else np.median(dcol)
                out[dlo:dhi, c] = dcol
        return out

    def extract(self, path):
        """One frame -> specutils Spectrum + region edges + per-region hot counts."""
        img, exptime = self._load_frame(path)
        nx, nrows = img.shape[1], img.shape[0]
        xcol = np.arange(nx - self.XLO_OFF, nx - self.XHI_OFF)        # 2048 science cols
        hot, medf = self._hot_mask(img)
        cen_full = self._trace_center(img)
        clean = img.astype(float).copy()
        clean[hot] = medf[hot]          # estimate EVERY hot pixel from its dispersion neighbors

        cen_x = cen_full[xcol]

        # --- one-shot diagnostic: is detection catching spikes, is the aperture on the trace? ---
        import matplotlib.pyplot as _plt
        print(f"  hot fraction = {100*hot.mean():.1f}%  | cen range {cen_full[xcol].min():.1f}-{cen_full[xcol].max():.1f}")
        _f,_ax=_plt.subplots(1,2,figsize=(13,4))
        for _a,_im,_t in ((_ax[0],img,'raw'),(_ax[1],clean,'inpainted')):
            _v=np.percentile(img,[20,99]); _a.imshow(_im,origin='lower',aspect='auto',vmin=_v[0],vmax=_v[1])
            _a.plot(xcol,cen_x,'r-',lw=0.8); _a.set_title(_t)
        _plt.show()
    
        sci_lo, sci_hi, dk_lo, dk_hi = self._region_rows(cen_x)
        clamp = lambda v: int(np.clip(v, 0, nrows))
        nhot_sci = np.array([hot[clamp(sci_lo[i]):clamp(sci_hi[i]), xcol[i]].sum() for i in range(len(xcol))])
        nhot_dk  = np.array([hot[clamp(dk_lo[i]):clamp(dk_hi[i]),  xcol[i]].sum() for i in range(len(xcol))])

        dn = np.array([clean[clamp(sci_lo[i]):clamp(sci_hi[i]), xcol[i]].mean()
                       - clean[clamp(dk_lo[i]):clamp(dk_hi[i]), xcol[i]].mean()
                       for i in range(len(xcol))])
        flux = self._to_flux(dn, exptime)
        spec = Spectrum(spectral_axis=self.wv_soln * u.AA,
                          flux=flux * u.Unit('1e-9 erg s-1 cm-2 AA-1'))
        return dict(spec=spec, img=img, xcol=xcol, nhot_sci=nhot_sci, nhot_dk=nhot_dk,
                    sci_lo=sci_lo, sci_hi=sci_hi, dk_lo=dk_lo, dk_hi=dk_hi)

    # ------------------ Outputs ------------------
    @staticmethod
    def _stem(path):
        m = re.search(r'frmid[_-]?(\d+)', os.path.basename(path))
        return f"frmid_{m.group(1)}" if m else os.path.splitext(os.path.basename(path))[0]

    def _emit(self, fig, out):
        if self.save:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
        else:
            plt.show()

    def plot_spectrum(self, R, out, title):
        spec_s = gaussian_smooth(R['spec'], stddev=2)      # ~matches the old boxcar look
        w = R['spec'].spectral_axis.value; fl = R['spec'].flux.value
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(w, fl, color='maroon', lw=0.9)
        ax.set_xlabel('Wavelength (\u00c5)')
        ax.set_ylabel(r'Flux (10$^{-9}$ erg s$^{-1}$ cm$^{-2}$ \u00c5$^{-1}$)')
        ax.set_title(title); ax.grid(True, ls='--', lw=0.5, alpha=0.4)
        self._emit(fig, out)

    def plot_regions(self, R, out, title):
        img = R['img']; x = R['xcol']
        fig, ax = plt.subplots(figsize=(10, 4.5))
        vmn, vmx = np.percentile(img, [20, 99])
        ax.imshow(img, origin='lower', aspect='auto', vmin=vmn, vmax=vmx, cmap='viridis')
        ax.plot(x, R['sci_lo'], 'r-', lw=1); ax.plot(x, R['sci_hi'], 'r-', lw=1, label='science')
        ax.plot(x, R['dk_lo'], 'y-', lw=1); ax.plot(x, R['dk_hi'], 'y-', lw=1, label='dark')
        ax.legend(loc='upper right'); ax.set_title(title)
        ax.set_xlabel('column'); ax.set_ylabel('row')
        self._emit(fig, out)

    def save_csv(self, R, out):
        if not self.save: return
        os.makedirs(os.path.dirname(out), exist_ok=True)
        w = R['spec'].spectral_axis.value; fl = R['spec'].flux.value
        with open(out, 'w', newline='') as f:
            wr = csv.writer(f); wr.writerow(['wave', 'flux', 'n_hotsci', 'n_hotdk'])
            for row in zip(w, fl, R['nhot_sci'], R['nhot_dk']):
                wr.writerow([f"{row[0]:.6e}", f"{row[1]:.6e}", int(row[2]), int(row[3])])

    def run_spectra(self):
        """For every frame of every visit: 1-D spectrum PNG, trace-region PNG, CSV."""
        for visit in self._visits():
            vdir = os.path.join(self.output_dir, visit)
            for p in self._frame_paths(visit):
                R = self.extract(p); stem = self._stem(p)
                ttl = f"{self.planet.capitalize()} - {visit} {stem}"
                self.plot_spectrum(R, os.path.join(vdir, f"{stem}_spectrum.png"), ttl)
                self.plot_regions(R,  os.path.join(vdir, f"{stem}_regions.png"),  ttl)
                self.save_csv(R,      os.path.join(vdir, f"{stem}.csv"))
                print(f"  {visit} {stem}: sci_hot={R['nhot_sci'].sum()} dk_hot={R['nhot_dk'].sum()}")