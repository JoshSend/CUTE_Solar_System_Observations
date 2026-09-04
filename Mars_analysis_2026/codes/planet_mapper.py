"""
Planetmapper wireframe images of Mars.
@author: jose5987
Inputs: date and time in UTC
Outputs: two wireframe images of Mars at specified UTC. Two FOVs, 23x23 arcmin and 5x5 Mars Radii
Dependencies: Planetmapper (pip install) and relevant Mars SpicePy kernels.
"""
# import matplotlib.pyplot as plt
# import planetmapper
# # YYYY-MM-DDTHH:MM:SS
# UTC = '2025-01-12T04:32:26'                       # your CUTE observation time
# body = planetmapper.Body('mars', UTC, observer='earth')
# # Mars angular radius in arcsec at this distance
# mars_radius_arcsec = body.target_diameter_arcsec / 2
# fov_arcmin = 23 * 60
# def zoom(ax, half):
#     """Center the view on Mars with a half-window of half arcsec,
#     preserving PlanetMapper's sky (east-left) x-axis direction."""
#     xlo, xhi = ax.get_xlim()
#     ax.set_xlim((half, -half) if xhi < xlo else (-half, half))
#     ax.set_ylim(-half, half)
#     ax.set_aspect('equal')
#     ax.axis('off')
# # --- Image 1: 5 x 5 Mars radii FOV -------------------------------------
# fig1, ax1 = plt.subplots(figsize=(6, 6))
# body.plot_wireframe_angular(
#     ax=ax1, 
#     indicate_prime_meridian=True,
#     indicate_equator=False, 
#     add_title=False,
# )
# zoom(ax1, 2.5 * mars_radius_arcsec)               # 5 R_Mars across
# ax1.set_title(f'')
# # --- Image 2: 23 x 23 arcmin FOV ---------------------------------------
# fig2, ax2 = plt.subplots(figsize=(6, 6))
# body.plot_wireframe_angular(
#     ax=ax2,
#     indicate_prime_meridian=True,
#     indicate_equator=False, 
#     add_title=False,
#     label_poles=False,
# )
# zoom(ax2, fov_arcmin / 2)                                 # 23 arcmin across
# ax2.set_title('')
# plt.show()
#--------------------------
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use("Agg")            # non-interactive backend (safe inside an IDE)
import planetmapper
 
# ------------------------------------------------------------------ config ---
try:
    CODES_DIR = Path(__file__).resolve().parent          # ...\Mars_analysis_2026\codes
except NameError:
    CODES_DIR = Path.cwd()
 
PROJECT_DIR = CODES_DIR.parent                           # ...\Mars_analysis_2026
CSV_PATH   = PROJECT_DIR / "CUTE_observations" / "CUTE_mars_headers.csv"
OUTPUT_DIR = CODES_DIR / "output" / "Mars_Fits"
 
OBSERVER   = "earth"     # CUTE is in LEO; Earth-centre parallax on Mars is negligible
ABERRATION = "CN"        # SPICE aberration correction: 'CN' (default), 'LT+S', 'CN+S', ...
NPIX       = 256         # image is NPIX x NPIX pixels (bump up for a finer disc)
OVERWRITE  = True
MAX_FRAMES = 40          # 16 reaches Visit 2 / frame 4860; set None for all
 
PRIMARY_BACKPLANE = "INCIDENCE"                          # image in the primary HDU
EXTRA_BACKPLANES  = ["EMISSION", "LON-GRAPHIC", "LAT-GRAPHIC"]
 
COL_VISIT, COL_FRAME, COL_DATE = "Visit#", "frameid", "date (UTC)"
 
 
# --------------------------------------------------------------- functions ---
def parse_utc(date_str):
    """'2024-12-22 3:49:26' (non-padded hour ok) -> ISO 'YYYY-MM-DDTHH:MM:SS'."""
    dt = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")
 
 
def build_wcs_from_backplanes(ra_img, dec_img):
    """WCS from PlanetMapper's own per-pixel RA/Dec, via the local gradient at
    the image centre -- so the WCS reproduces PlanetMapper's pixel->sky mapping
    exactly (linear is essentially exact over these small fields)."""
    ny, nx = ra_img.shape
    jc, ic = ny // 2, nx // 2
    ra0, dec0 = float(ra_img[jc, ic]), float(dec_img[jc, ic])
    cosd = np.cos(np.radians(dec0))
    ra_rel = (((ra_img - ra0 + 180.0) % 360.0) - 180.0) * cosd
    dec_rel = dec_img - dec0
    gj_ra, gi_ra = np.gradient(ra_rel)
    gj_dec, gi_dec = np.gradient(dec_rel)
    cd = np.array([[gi_ra[jc, ic], gj_ra[jc, ic]],
                   [gi_dec[jc, ic], gj_dec[jc, ic]]])
    w = WCS(naxis=2)
    w.wcs.crpix = [ic + 1.0, jc + 1.0]
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.cd = cd
    return w
 
 
def save_navigated_fits(body, utc, fov_arcsec, out_path):
    """Build a BodyXY with PlanetMapper's native orientation and write a WCS FITS."""
    scale = fov_arcsec / NPIX                             # arcsec / pixel
    r0 = (body.target_diameter_arcsec / 2.0) / scale      # Mars radius in pixels
 
    print(f"     {out_path.name}: navigating (native orientation)...", flush=True)
    bxy = planetmapper.BodyXY.from_body(body)
    bxy.set_img_size(NPIX, NPIX)
    bxy.set_r0(r0)
    bxy.set_x0((NPIX - 1) / 2.0)
    bxy.set_y0((NPIX - 1) / 2.0)
    bxy.rotate_north_to_top()                             # SPICE orientation (like plot_wireframe)
 
    print(f"     {out_path.name}: backplanes...", flush=True)
    primary_arr = bxy.get_backplane_img(PRIMARY_BACKPLANE).astype("float32")
    wcs = build_wcs_from_backplanes(bxy.get_ra_img(), bxy.get_dec_img())
    header = wcs.to_header()
 
    primary = fits.PrimaryHDU(data=primary_arr, header=header)
    primary.header["OBJECT"]   = "MARS"
    primary.header["DATE-OBS"] = utc
    primary.header["BACKPLNE"] = PRIMARY_BACKPLANE
    hdus = [primary]
 
    for name in EXTRA_BACKPLANES:
        try:
            arr = bxy.get_backplane_img(name).astype("float32")
            hdus.append(fits.ImageHDU(data=arr, header=header.copy(), name=name))
        except Exception as e:
            print(f"       (skipped {name}: {e})", flush=True)
 
    try:
        c = NPIX // 2
        phase = float(bxy.get_backplane_img("INCIDENCE")[c, c])
        illum = 0.5 * (1.0 + np.cos(np.radians(phase)))
    except Exception:
        phase = illum = float("nan")
 
    print(f"     {out_path.name}: writing...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out_path, overwrite=OVERWRITE)
    return phase, illum
 
 
# --------------------------------------------------------------------- main ---
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}\n"
                                f"Edit CSV_PATH near the top of this script.")
 
    df = pd.read_csv(CSV_PATH).drop_duplicates(subset=[COL_VISIT, COL_FRAME, COL_DATE])
    if MAX_FRAMES is not None:
        df = df.head(MAX_FRAMES)
 
    print(f"{len(df)} observations to process -> {OUTPUT_DIR}", flush=True)
 
    n_ok, n_fail = 0, 0
    for i, (_, r) in enumerate(df.iterrows(), 1):
        visit   = str(r[COL_VISIT]).strip()
        frameid = str(r[COL_FRAME]).strip()
        try:
            utc = parse_utc(str(r[COL_DATE]))
            body = planetmapper.Body("mars", utc, observer=OBSERVER,
                                     aberration_correction=ABERRATION)
            vdir = OUTPUT_DIR / f"Visit{visit}"
 
            # Geometry check -- compare to your PDS reference viewer:
            print(f"     geom  RA={body.target_ra:.6f} Dec={body.target_dec:.6f}  "
                  f"sub-obs={body.subpoint_lon:.3f}W/{body.subpoint_lat:.3f}  "
                  f"sub-sol={body.subsol_lon:.3f}W/{body.subsol_lat:.3f}", flush=True)
 
            phase, illum = save_navigated_fits(
                body, utc, 2.5 * body.target_diameter_arcsec,     # 5 Mars radii across
                vdir / f"frmid{frameid}_5x5MR.fits")
 
            save_navigated_fits(
                body, utc, 23 * 60.0,                             # 23 arcmin across
                vdir / f"frmid{frameid}_23x23arcmin.fits")
 
            n_ok += 1
            print(f"[{i:>3}/{len(df)}]  Visit{visit}  frmid{frameid}  {utc}  "
                  f"phase={phase:5.1f} deg  illum={illum*100:5.1f}%  OK", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[{i:>3}/{len(df)}]  Visit{visit}  frmid{frameid}  FAILED: {e}", flush=True)
 
    print(f"\nDone. {n_ok} observations written, {n_fail} failed. Output: {OUTPUT_DIR}", flush=True)
 
 
if __name__ == "__main__":
    main()