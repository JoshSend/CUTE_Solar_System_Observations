"""
For each CUTE visit, find the Earth date whose solar hemisphere matches the one
illuminating Mars at the observation time (solar-rotation phasing), pull that
day's MEASURED TSIS-1 SIM solar spectrum from LISIRD (LaTiS), scale it to Mars'
heliocentric distance, trim to the CUTE NUV band, and save one file per visit to
    CUTE_Solar_System_Observations\\Mars_analysis_2026\\Solar_spectra\\
as  <visit>_solar_spectrum.csv.

Input:  a dict  {visit_label: UTC date/time}, e.g.
            {'visit1': '2024-12-22T04:40:31', 'visit2': '2025-01-12T05:24:48', ...}
        The value may be an ISO string ('2025-01-19' or '2025-01-19T05:17:41')
        or an astropy Time / datetime.

Requires:  sunpy, astropy, scipy, numpy, requests
           (pip install sunpy astropy scipy numpy requests)

Solar source:  TSIS-1 SIM Level-3 24-hour SSI via LISIRD LaTiS
    https://lasp.colorado.edu/lisird/latis/dap/tsis_ssi_24hr
This is the measured TSIS product (not a model), covers 2018-present, no login.
Columns returned: time(JD), wavelength(nm), irradiance_1au, irradiance_true_earth,
measurement_stability, measurement_precision, instrument_uncertainty,
additional_uncertainty, quality  (all irradiances in W/m^2/nm).
"""

import os
import io
import numpy as np
import astropy.units as u
from astropy.time import Time
from scipy.optimize import brentq
from sunpy.coordinates import get_body_heliographic_stonyhurst, HeliographicCarrington
import requests
from scipy.ndimage import median_filter

_trap = getattr(np, 'trapezoid', np.trapz)   # band integral, NumPy-version safe

# ============================== CONFIG ==============================

# {visit label : observation UTC}. Edit / replace with your own dict.
OBSERVATIONS = {
    'visit1': '2024-12-22T04:40:31',
    'visit2': '2025-01-12T05:24:48',
    'visit3': '2025-01-19T05:17:41',
    'visit4': '2025-01-27T04:30:56',
    'visit5': '2025-02-03T04:50:58',
    'visit7': '2025-02-17T05:33:26',
    'visit8': '2025-03-02T05:50:56',
    'visit9': '2025-03-30T04:06:56',
}

LATIS_BASE = 'https://lasp.colorado.edu/lisird/latis/dap/tsis_ssi_24hr.csv'
BAND_NM    = (200.0, 335.0)       # CUTE NUV band to keep (nm)
SCALE_TO_MARS = True              # scale 1-AU irradiance to Mars heliocentric dist

# fixed column order of the tsis_ssi_24hr CSV (parsed by index, not header text)
C_TIME, C_WAVE, C_IRR1AU, C_UNC = 0, 1, 2, 6   # time, wavelength, irr_1au, instr_unc

# output dir: ..\Mars_analysis_2026\Solar_spectra relative to THIS file
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(
    os.path.join(_HERE, '..', 'Mars_analysis_2026', 'Solar_spectra'))

# ====================================================================
HSRS_BASE = 'https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs.csv'
R_SIM, R_HSRS = 300.0, 28000.0     # SIM resolution; HSRS native (NUV) ~ R28000

def _gsmooth(y, fwhm_pix):
    sigma = max(fwhm_pix / 2.3548, 1e-6); half = int(np.ceil(4 * sigma))
    x = np.arange(-half, half + 1); k = np.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
    return np.convolve(y, k, mode='same')

def _degrade_to_R(wave, flux, R_from, R_to):
    flux = np.asarray(flux, float)
    if R_to >= R_from: return flux.copy()
    lnw = np.log(wave); lnu = np.linspace(lnw[0], lnw[-1], len(wave)); dln = lnu[1] - lnu[0]
    fu = np.interp(lnu, lnw, flux)
    fwhm = np.sqrt((1.0 / R_to) ** 2 - (1.0 / R_from) ** 2)
    return np.interp(lnw, lnu, _gsmooth(fu, fwhm / dln))

def fetch_hsrs(band_nm, chunk_nm=15.0, timeout=180):
    """Static high-res TSIS-1 HSRS over the band (chunked to avoid size caps)."""
    lo, hi = band_nm
    edges = np.arange(lo, hi + chunk_nm, chunk_nm)
    W, I, U = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        url = f"{HSRS_BASE}?wavelength>={a}&wavelength<{b}"
        arr = np.atleast_2d(np.genfromtxt(io.StringIO(requests.get(url, timeout=timeout).text),
                                          delimiter=',', skip_header=1))
        if arr.size and arr.shape[1] >= 3:
            W.append(arr[:, 0]); I.append(arr[:, 1]); U.append(arr[:, 2])
    wl = np.concatenate(W); irr = np.concatenate(I); unc = np.concatenate(U)
    ok = np.isfinite(wl) & np.isfinite(irr) & (irr > 0)
    wl, irr, unc = wl[ok], irr[ok], unc[ok]
    wl, idx = np.unique(wl, return_index=True)          # sort + dedupe chunk seams
    return wl, irr[idx], unc[idx]

def relevel_hsrs(hw, hirr, sim_w, sim_irr):
    """HSRS high-res shape re-leveled to the day's SIM activity:
       ratio = SIM / (HSRS degraded to SIM res), applied to native HSRS."""
    hsrs_sim = _degrade_to_R(hw, hirr, R_HSRS, R_SIM)           # HSRS at SIM resolution
    ratio_sim = sim_irr / np.interp(sim_w, hw, hsrs_sim)        # smooth, on SIM grid
    return hirr * np.interp(hw, sim_w, ratio_sim)              # day-corrected, high-res (1 AU)

# ----------------------- solar-rotation phasing ----------------------
def sub_carr_lon(body, t):
    """Sub-observer Carrington longitude (deg) of `body` at time `t`."""
    c = get_body_heliographic_stonyhurst(body, t)
    return c.transform_to(
        HeliographicCarrington(obstime=t, observer='self')).lon.to_value(u.deg) % 360


def matched_earth_date(t_obs, span=18, step=0.5):
    """
    UTC when the sub-Earth Carrington longitude equals the sub-Mars longitude
    at t_obs. Scans +-span days for the true zero crossing nearest t_obs
    (robust to the +-180 deg wrap that breaks a bare brentq bracket).
    """
    target = sub_carr_lon('mars', t_obs)

    def resid(dd):
        return (sub_carr_lon('earth', t_obs + dd * u.day) - target + 180) % 360 - 180

    grid = np.arange(-span, span + step, step)
    vals = [resid(dd) for dd in grid]
    roots = []
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0:
            roots.append(grid[i])
        elif a * b < 0 and abs(a - b) < 180:        # real crossing, not a wrap jump
            roots.append(brentq(resid, grid[i], grid[i + 1]))
    if not roots:
        raise RuntimeError(f"no longitude match within +-{span} days of {t_obs.iso}")
    dd = min(roots, key=abs)                          # nearest crossing to t_obs
    return t_obs + dd * u.day, float(dd), target


def mars_helio_dist_au(t):
    """Mars heliocentric distance (AU) at time t."""
    return get_body_heliographic_stonyhurst('mars', t).radius.to_value(u.AU)


# --------------------------- LaTiS / TSIS pull -----------------------
def fetch_solar_day(t_match, band_nm, widen=3, timeout=120):
    """
    Pull the TSIS-1 SIM 24-hour spectrum for the day nearest t_match over the
    CUTE band. Returns (wavelength_nm, irradiance_1au, uncertainty, used_time).
    Widens the date window if the exact day is missing, then picks the nearest.
    """
    jd_target = t_match.jd
    arr = np.empty((0, 9))
    for w in (0, widen, widen * 3):
        d0 = (t_match - w * u.day).datetime.strftime('%Y-%m-%d')
        d1 = (t_match + (w + 1) * u.day).datetime.strftime('%Y-%m-%d')
        url = (f"{LATIS_BASE}?time>={d0}&time<{d1}"
               f"&wavelength>={band_nm[0]}&wavelength<={band_nm[1]}")
        txt = requests.get(url, timeout=timeout).text
        arr = np.atleast_2d(np.genfromtxt(io.StringIO(txt), delimiter=',',
                                          skip_header=1))
        if arr.size and arr.shape[1] >= 7:
            break
    if arr.size == 0:
        raise RuntimeError(f"no TSIS data near {t_match.iso} (check coverage)")

    # choose the daily record (unique JD) nearest the matched time
    times = arr[:, C_TIME]
    uniq = np.unique(times)
    best = uniq[np.argmin(np.abs(uniq - jd_target))]

    sel = times == best
    wl  = arr[sel, C_WAVE]
    irr = arr[sel, C_IRR1AU]
    unc = arr[sel, C_UNC]
    # drop flagged / missing points: TSIS marks them NaN or with a fill value,
    # and a single NaN makes the whole band integral (np.trapz) NaN -- and would
    # also land in the saved CSV.
    ok = np.isfinite(wl) & np.isfinite(irr) & (irr > 0)
    if not ok.any():
        raise RuntimeError(f"no valid TSIS points near {t_match.iso}")
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"    (dropped {n_drop} flagged/missing TSIS points)")
    wl, irr, unc = wl[ok], irr[ok], unc[ok]
    o = np.argsort(wl)
    return wl[o], irr[o], unc[o], Time(best, format='jd')

# ------------------------------- driver ------------------------------
def run(observations=OBSERVATIONS, output_dir=OUTPUT_DIR, band_nm=BAND_NM,
        scale_to_mars=SCALE_TO_MARS):
    os.makedirs(output_dir, exist_ok=True)

    # High-res HSRS is static -- pull it once and reuse for every visit.
    print("fetching TSIS-1 HSRS (high-res reference)...")
    hw, hirr, hunc = fetch_hsrs(band_nm)
    print(f"  HSRS: {hw.size} points, {hw[0]:.3f}-{hw[-1]:.3f} nm "
          f"(~{np.median(np.diff(hw))*1000:.2f} pm sampling)")

    summary = []
    print(f"\n{'visit':>8} {'obs UTC':>20} {'-> TSIS date':>13} {'shift/d':>8} "
          f"{'r/AU':>7} {'Fsun@Mars':>11} {'Fsun@1AU':>10}")
    for label, obs in observations.items():
        t_obs = obs if isinstance(obs, Time) else Time(obs)
        t_match, dd, lon_mars = matched_earth_date(t_obs)

        # TSIS SIM for this visit's phased day -> supplies the daily activity level
        wl, irr1au, unc, used = fetch_solar_day(t_match, band_nm)

        # HSRS high-res shape, re-levelled to the day's SIM level (Mg II etc.)
        solar_1au = relevel_hsrs(hw, hirr, wl, irr1au)

        if scale_to_mars:
            r = mars_helio_dist_au(t_obs)
            solar_mars = solar_1au / r ** 2            # 1 AU -> at Mars
        else:
            r, solar_mars = np.nan, solar_1au.copy()

        # band-integrated solar flux (W/m^2 over the CUTE NUV band)
        flux_mars = float(_trap(solar_mars, hw))       # incident at Mars
        flux_1au  = float(_trap(solar_1au,  hw))       # at 1 AU (reference)

        path = os.path.join(output_dir, f"{label}_solar_spectrum.csv")
        np.savetxt(
            path, np.column_stack([hw, solar_mars, solar_1au, hunc]), delimiter=',',
            header=(f"solar spectrum for {label}; obs {t_obs.iso} UTC; "
                    f"phased TSIS date {used.iso[:10]} (shift {dd:+.2f} d); "
                    f"Mars r={r:.4f} AU; band {band_nm[0]}-{band_nm[1]} nm; "
                    f"integrated F_sun@Mars={flux_mars:.4f} W/m^2, "
                    f"F_sun@1AU={flux_1au:.4f} W/m^2; "
                    f"source: TSIS-1 HSRS (high-res) re-levelled to TSIS-1 SIM 24hr "
                    f"({used.iso[:10]})\n"
                    f"wavelength_nm,irradiance_at_mars_W_m2_nm,"
                    f"irradiance_1au_W_m2_nm,hsrs_uncertainty_W_m2_nm"),
            comments='# ', fmt='%.6e')

        summary.append((label, t_obs.iso, used.iso[:10], dd, r, flux_mars, flux_1au))
        print(f"{label:>8} {t_obs.iso[:19]:>20} {used.iso[:10]:>13} "
              f"{dd:>8.2f} {r:>7.4f} {flux_mars:>11.4f} {flux_1au:>10.4f}")

    print(f"\n(Fsun columns = solar flux integrated over "
          f"{band_nm[0]:.0f}-{band_nm[1]:.0f} nm, in W/m^2)")

    log = os.path.join(output_dir, "phasing_summary.csv")
    with open(log, 'w') as f:
        f.write("visit,obs_utc,tsis_date,shift_days,mars_r_au,"
                "Fsun_at_mars_W_m2,Fsun_1au_W_m2\n")
        for row in summary:
            f.write("{},{},{},{:.3f},{:.4f},{:.5f},{:.5f}\n".format(*row))
    print(f"\nWrote {len(summary)} spectra + phasing_summary.csv -> {output_dir}")


if __name__ == '__main__':
    run()