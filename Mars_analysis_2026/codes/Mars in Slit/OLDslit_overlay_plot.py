# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:42:26 2022
@author: amsu4591

Modified for CUTE Mars observations
@author: jose5987
"""

import os
import csv
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from astropy.io import fits
from astropy.wcs import WCS
import pymupdf  
 

import os
import csv
import glob
import argparse
 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.path import Path
 
from astropy.io import fits
from astropy.wcs import WCS
 
# ============================ USER CONFIG ============================
HERE      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(HERE, '..', 'output', 'mars_pointing.csv')
FITS_DIR  = os.path.join(HERE, '..', 'output', 'Mars_Fits')
RAW_DIR   = os.path.join(HERE, '..', '..', 'data')     # L1 cute_TRIM2D_*.fits (optional)
OUT_DIR   = os.path.join(HERE, 'overlay_out')
 
# roll -> on-sky position angle of the slit long axis, see docstring
ROLL_SIGN = +1.0
ROLL_ZERO = 0.0
 
# CUTE slit outline, arcsec in the slit frame.  px = across-slit, py = along-slit.
SLIT_PX = [-60.0, 60.0, 60.0, 30.0, 30.0, 15.0, 15.0, -15.0, -15.0, -30.0, -30.0, -60.0, -60.0]
SLIT_PY = [-591.807, -591.807, -351.805, -351.805, 351.805, 351.805, 680.817,
           680.817, 351.805, 351.805, -351.805, -351.805, -591.807]
 
# which HDU of the PlanetMapper file to show in the zoom panel
ZOOM_HDU  = 0          # 0=INCIDENCE, 1=EMISSION, 2=LON-GRAPHIC, 3=LAT-GRAPHIC
ZOOM_CMAP = 'inferno'
 
C_SLIT, C_MARS, C_BORE, C_COMPASS = '#e8453c', '#f2b705', '#4da3ff', '#7fe3a1'
# =====================================================================
 
SLIT_CLOSED = np.column_stack([SLIT_PX + [SLIT_PX[0]], SLIT_PY + [SLIT_PY[0]]])
SLIT_PATH = Path(SLIT_CLOSED)
 
 
# --------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------
def slit_position_angle(roll, roll_sign=None, roll_zero=None):
    """On-sky position angle (deg east of north) of the slit +py axis."""
    rs = ROLL_SIGN if roll_sign is None else roll_sign
    rz = ROLL_ZERO if roll_zero is None else roll_zero
    return rs * (-roll) + rz
 
 
def slit_to_world(px, py, ra, dec, pa):
    """Slit-frame arcsec -> (RA, Dec) in deg, tangent-plane about (ra, dec)."""
    t = np.deg2rad(pa)
    c, s = np.cos(t), np.sin(t)
    px, py = np.asarray(px, float), np.asarray(py, float)
    east = px * c + py * s          # arcsec toward +RA
    north = -px * s + py * c        # arcsec toward +Dec
    return ra + (east / 3600.0) / np.cos(np.deg2rad(dec)), dec + north / 3600.0
 
 
def world_to_slit(ra_pts, dec_pts, ra, dec, pa):
    """(RA, Dec) in deg -> slit-frame arcsec (px, py). Inverse of slit_to_world."""
    east = (np.asarray(ra_pts, float) - ra) * np.cos(np.deg2rad(dec)) * 3600.0
    north = (np.asarray(dec_pts, float) - dec) * 3600.0
    t = np.deg2rad(pa)
    c, s = np.cos(t), np.sin(t)
    return east * c - north * s, east * s + north * c
 
 
def slit_band(px, py):
    """Which step of the slit funnel a slit-frame point falls in."""
    if not SLIT_PATH.contains_point((px, py)):
        return 'outside slit'
    if py < -351.805:
        return 'wide band (120" across)'
    if py <= 351.805:
        return 'science band (60" across)'
    return 'narrow band (30" across)'
 
 
# --------------------------------------------------------------------
# file handling
# --------------------------------------------------------------------
def find_fits(frmid, pattern):
    """First file under FITS_DIR whose name contains the frmid and the pattern."""
    hits = [f for f in glob.glob(os.path.join(FITS_DIR, '**', '*.fits'), recursive=True)
            if str(frmid) in os.path.basename(f) and pattern in os.path.basename(f)]
    return sorted(hits)[0] if hits else None
 
 
def find_raw(fname):
    if not fname:
        return None
    hits = glob.glob(os.path.join(RAW_DIR, '**', fname), recursive=True)
    return hits[0] if hits else None
 
 
def read_pointing(csv_path):
    """mars_pointing.csv -> list of dicts; headers carry units, so match on the stem."""
    def val(row, key):
        for k, v in row.items():
            if k is not None and k.split(' ')[0] == key:
                return v
        raise KeyError(key)
 
    rows = []
    with open(csv_path, newline='') as fh:
        for raw in csv.DictReader(fh):
            if not raw.get('frmid'):
                continue
            rows.append(dict(visit=raw['visit'], frmid=int(raw['frmid']),
                             file=raw.get('file', ''), date=val(raw, 'date_obs'),
                             ra=float(val(raw, 'ra')), dec=float(val(raw, 'dec')),
                             roll=float(val(raw, 'roll'))))
    # de-duplicate frmids that appear twice (different L1id, same pointing)
    seen, out = set(), []
    for r in rows:
        if r['frmid'] in seen:
            continue
        seen.add(r['frmid'])
        out.append(r)
    return out
 
 
# --------------------------------------------------------------------
# ADCS cross-check
# --------------------------------------------------------------------
def quaternion_check(raw_path, ra_csv, dec_csv, roll_csv):
    """Recover boresight and roll from the L1 XB1 attitude quaternion.
 
    Returns a dict, or None if the header has no quaternion.  Convention found
    for CUTE: scalar-last (AQ3 = scalar); the matrix rows are the body axes
    expressed in ECI; the ADCS boresight is body +X.
    """
    h = fits.getheader(raw_path)
    if 'XB1_AQ3' not in h:
        return None
    q1, q2, q3, q0 = (float(h['XB1_AQ0']), float(h['XB1_AQ1']),
                      float(h['XB1_AQ2']), float(h['XB1_AQ3']))
    A = np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3]])
    bx, by, bz = A[0], A[1], A[2]
    b = bx / np.linalg.norm(bx)
    ra_b = np.arctan2(b[1], b[0])
    dec_b = np.arcsin(b[2])
    E = np.array([-np.sin(ra_b), np.cos(ra_b), 0.0])
    N = np.array([-np.sin(dec_b) * np.cos(ra_b), -np.sin(dec_b) * np.sin(ra_b), np.cos(dec_b)])
    p = bz - (bz @ b) * b
    p /= np.linalg.norm(p)
    pa_z = np.rad2deg(np.arctan2(p @ E, p @ N))
 
    u = np.array([np.cos(np.deg2rad(dec_csv)) * np.cos(np.deg2rad(ra_csv)),
                  np.cos(np.deg2rad(dec_csv)) * np.sin(np.deg2rad(ra_csv)),
                  np.sin(np.deg2rad(dec_csv))])
    v_body = A @ u
    return dict(ra_adcs=np.rad2deg(ra_b) % 360, dec_adcs=np.rad2deg(dec_b),
                pa_z=pa_z, roll_pred=-pa_z, roll_csv=roll_csv,
                roll_resid=((-pa_z) - roll_csv + 180) % 360 - 180,
                sep_deg=np.rad2deg(np.arccos(np.clip(b @ u, -1, 1))),
                body_y_arcmin=np.rad2deg(v_body[1]) * 60,
                body_z_arcmin=np.rad2deg(v_body[2]) * 60)
 
 
# --------------------------------------------------------------------
# one frame
# --------------------------------------------------------------------
def process_frame(row, args):
    frmid = row['frmid']
    f5 = find_fits(frmid, '5x5MR')
    fw = find_fits(frmid, '23x23arcmin')
    if f5 is None and fw is None:
        print(f'  frmid {frmid}: no Mars FITS found in {FITS_DIR}, skipping')
        return None
 
    ra, dec, roll = row['ra'], row['dec'], row['roll']
    pa = slit_position_angle(roll, args.roll_sign, args.roll_zero)
 
    # ---- Mars disk from the 5 R_M file: finite backplane pixels == on-disk
    disk = None
    if f5:
        with fits.open(f5) as hdul:
            hdus5 = [(h.header, h.data) for h in hdul]
        hdr5 = hdus5[0][0]
        w5 = WCS(hdr5)
        mask = np.isfinite(hdus5[0][1])
        try:                                    # works for PC+CDELT, CD, or CDELT-only
            pixscale = float(np.hypot(*w5.pixel_scale_matrix[:, 0])) * 3600.0
        except Exception:
            pixscale = float(np.hypot(hdr5.get('PC1_1', hdr5.get('CDELT1', 1.0)),
                                      hdr5.get('PC1_2', 0.0))) * 3600.0
        yy, xx = np.nonzero(mask)
        ra_d, dec_d = w5.wcs_pix2world(xx, yy, 0)
        dpx, dpy = world_to_slit(ra_d, dec_d, ra, dec, pa)
        inside = SLIT_PATH.contains_points(np.column_stack([dpx, dpy]))
        mars_ra, mars_dec = float(hdr5['CRVAL1']), float(hdr5['CRVAL2'])
        r_mars = np.sqrt(mask.sum() / np.pi) * pixscale
        disk = dict(hdus=hdus5, wcs=w5, xx=xx, yy=yy, dpx=dpx, dpy=dpy,
                    inside=inside, r=r_mars, pixscale=pixscale)
    else:
        with fits.open(fw) as hdul:
            mars_ra, mars_dec = float(hdul[0].header['CRVAL1']), float(hdul[0].header['CRVAL2'])
        r_mars = np.nan
 
    cpx, cpy = world_to_slit(mars_ra, mars_dec, ra, dec, pa)
    cpx, cpy = float(cpx), float(cpy)
    frac = float(disk['inside'].mean()) if disk else np.nan
    sep = np.hypot(cpx, cpy)
 
    metrics = dict(visit=row['visit'], frmid=frmid, date_obs=row['date'],
                   ra=ra, dec=dec, roll=roll, pa_slit=pa,
                   mars_ra=mars_ra, mars_dec=mars_dec, mars_radius_arcsec=r_mars,
                   mars_px_arcsec=cpx, mars_py_arcsec=cpy,
                   boresight_offset_arcsec=sep,
                   offset_in_mars_radii=sep / r_mars if r_mars == r_mars else np.nan,
                   disk_fraction_in_slit=frac, slit_band=slit_band(cpx, cpy))
 
    if args.check_quat:
        raw = find_raw(row['file'])
        if raw:
            q = quaternion_check(raw, ra, dec, roll)
            if q:
                metrics.update({'adcs_' + k: v for k, v in q.items()})
                print(f"  frmid {frmid}: ADCS roll residual {q['roll_resid']:+.5f} deg, "
                      f"boresight-vs-CSV separation {q['sep_deg']:.4f} deg")
        else:
            print(f'  frmid {frmid}: L1 file not found under {RAW_DIR}, skipping quat check')
 
    if not args.no_plot:
        raw_path = find_raw(row['file']) if args.raw_panel else None
        fig_out = os.path.join(OUT_DIR, f"{row['visit']}_frmid{frmid}_slit.png")
        make_figure(row, pa, fw, disk, mars_ra, mars_dec, metrics, raw_path, fig_out)
        print(f'  frmid {frmid}: wrote {os.path.basename(fig_out)}')
    return metrics
 
 
def _draw_slit(ax, wcs, ra, dec, pa, **kw):
    rr, dd = slit_to_world(SLIT_CLOSED[:, 0], SLIT_CLOSED[:, 1], ra, dec, pa)
    x, y = wcs.wcs_world2pix(rr, dd, 0)
    ax.add_patch(Polygon(np.column_stack([x, y]), closed=True, fill=False, **kw))
 
 
def make_figure(row, pa, fw, disk, mars_ra, mars_dec, m, raw_path, fig_out):
    ra, dec = row['ra'], row['dec']
    ncol = 3 if disk else 2
    nrow = 2 if raw_path else 1
    hr = [2.2, 1.0] if raw_path else [1.0]
    fig = plt.figure(figsize=(5.4 * ncol, 5.6 if nrow == 1 else 9.0))
    gs = fig.add_gridspec(nrow, ncol, height_ratios=hr, hspace=0.28, wspace=0.24)
    col = 0
 
    # ---- (a) wide field with the whole slit
    if fw:
        with fits.open(fw) as hdul:
            hdrw, dataw = hdul[0].header, hdul[0].data
        ww = WCS(hdrw)
        axw = fig.add_subplot(gs[0, col]); col += 1
        axw.set_facecolor('#0d0d12')
        axw.imshow(np.nan_to_num(dataw, nan=0.0), origin='lower', cmap='gray')
        _draw_slit(axw, ww, ra, dec, pa, edgecolor=C_SLIT, lw=1.8, zorder=5)
        xm, ym = ww.wcs_world2pix(mars_ra, mars_dec, 0)
        axw.add_patch(Circle((xm, ym), 18, fill=False, ec=C_MARS, lw=1.2, ls='--', zorder=6))
        axw.plot(xm, ym, 'o', color=C_MARS, ms=4, zorder=7, label='Mars')
        xb, yb = ww.wcs_world2pix(ra, dec, 0)
        axw.plot(xb, yb, '+', color=C_BORE, ms=13, mew=1.8, zorder=7, label='boresight')
        for lab, (dE, dN) in [('N', (0, 1)), ('E', (1, 0))]:
            x1, y1 = ww.wcs_world2pix(ra + dE * 150 / 3600.0 / np.cos(np.deg2rad(dec)),
                                      dec + dN * 150 / 3600.0, 0)
            axw.annotate('', xy=(x1, y1), xytext=(xb, yb),
                         arrowprops=dict(arrowstyle='->', color=C_COMPASS, lw=1.4))
            axw.text(float(x1), float(y1), ' ' + lab, color=C_COMPASS, fontsize=9)
        axw.set_xlim(0, dataw.shape[1]); axw.set_ylim(0, dataw.shape[0])
        axw.set_xticks([]); axw.set_yticks([])
        axw.set_title('(a) full slit footprint, 23′ × 23′', fontsize=10)
        axw.legend(loc='upper right', fontsize=7.5, framealpha=0.3, labelcolor='w')
 
    # ---- (b) slit frame
    axs = fig.add_subplot(gs[0, col]); col += 1
    axs.set_facecolor('#0d0d12')
    axs.add_patch(Polygon(SLIT_CLOSED, closed=True, fc=C_SLIT + '18', ec=C_SLIT, lw=1.8, zorder=2))
    if disk:
        s = slice(None, None, max(1, len(disk['dpx']) // 6000))
        axs.scatter(disk['dpx'][s], disk['dpy'][s], s=2, zorder=3,
                    c=disk['hdus'][ZOOM_HDU][1][disk['yy'], disk['xx']][s], cmap=ZOOM_CMAP)
    axs.plot(0, 0, '+', color=C_BORE, ms=14, mew=1.8, zorder=5)
    axs.plot(m['mars_px_arcsec'], m['mars_py_arcsec'], 'x', color=C_MARS, ms=9, mew=1.8, zorder=5)
    lim = max(95.0, 1.6 * np.hypot(m['mars_px_arcsec'], m['mars_py_arcsec']))
    axs.set_xlim(-lim, lim); axs.set_ylim(-lim, lim); axs.set_aspect('equal')
    axs.set_xlabel('across-slit  px  [arcsec]'); axs.set_ylabel('along-slit  py  [arcsec]')
    axs.set_title(f'(b) slit frame, ±{lim:.0f}″', fontsize=10)
    txt = (f"Mars centre  px={m['mars_px_arcsec']:+.2f}″  py={m['mars_py_arcsec']:+.2f}″\n"
           f"offset from boresight  {m['boresight_offset_arcsec']:.2f}″")
    if m['disk_fraction_in_slit'] == m['disk_fraction_in_slit']:
        txt += f"  ({m['offset_in_mars_radii']:.2f} R$_M$)\nin-slit disk fraction  {100*m['disk_fraction_in_slit']:.1f}%"
    axs.text(0.03, 0.03, txt, transform=axs.transAxes, color='w', fontsize=8.5, va='bottom')
 
    # ---- (c) resolved disk
    if disk:
        axz = fig.add_subplot(gs[0, col]); col += 1
        axz.set_facecolor('#0d0d12')
        img = disk['hdus'][ZOOM_HDU][1]
        axz.imshow(img, origin='lower', cmap=ZOOM_CMAP)
        if len(disk['hdus']) > 3:
            axz.contour(disk['hdus'][3][1], levels=np.arange(-90, 91, 30),
                        colors='w', linewidths=0.45, alpha=0.55)
            axz.contour(disk['hdus'][2][1], levels=np.arange(0, 360, 30),
                        colors='w', linewidths=0.45, alpha=0.55)
        # grey out the part of the disk that misses the slit
        if not disk['inside'].all():
            out = ~disk['inside']
            axz.scatter(disk['xx'][out], disk['yy'][out], s=1, c='#2b2b33', zorder=4)
        _draw_slit(axz, disk['wcs'], ra, dec, pa, edgecolor=C_SLIT, lw=2.2, zorder=5)
        xb, yb = disk['wcs'].wcs_world2pix(ra, dec, 0)
        xc, yc = disk['wcs'].wcs_world2pix(mars_ra, mars_dec, 0)
        axz.plot(xb, yb, '+', color=C_BORE, ms=14, mew=1.8, zorder=7)
        axz.annotate('', xy=(xc, yc), xytext=(xb, yb),
                     arrowprops=dict(arrowstyle='->', color=C_BORE, lw=1.3, ls=':'))
        axz.set_xlim(0, img.shape[1]); axz.set_ylim(0, img.shape[0])
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_title('(c) Mars disk, 5 × 5 R$_M$', fontsize=10)
 
    # ---- (d) raw L1 frame
    if raw_path:
        axr = fig.add_subplot(gs[1, :])
        rawd = fits.getdata(raw_path).astype(float)
        v1, v2 = np.percentile(rawd, [5, 99.5])
        axr.imshow(rawd, origin='lower', cmap='viridis', vmin=v1, vmax=v2, aspect='auto')
        axr.set_title(f"(d) L1 raw frame, frmid {row['frmid']}", fontsize=10)
 
    fig.suptitle(f"CUTE Mars slit overlay — {row['visit']}  frmid {row['frmid']}  "
                 f"{row['date']}   roll={row['roll']:.3f}°  PA(+py)={pa:.3f}°", fontsize=12)
    fig.savefig(fig_out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
 
 
# --------------------------------------------------------------------
def main():
    global FITS_DIR, RAW_DIR, OUT_DIR
    p = argparse.ArgumentParser(description='Overlay the CUTE slit on PlanetMapper Mars FITS.')
    p.add_argument('--frmid', type=int, nargs='*', help='frame id(s) to process')
    p.add_argument('--visit', help='process a whole visit, e.g. Visit1')
    p.add_argument('--all', action='store_true', help='process every row in the CSV')
    p.add_argument('--roll-sign', type=float, default=ROLL_SIGN)
    p.add_argument('--roll-zero', type=float, default=ROLL_ZERO)
    p.add_argument('--check-quat', action='store_true', help='cross-check roll against the L1 ADCS quaternion')
    p.add_argument('--raw-panel', action='store_true', default=True, help='include the L1 frame panel')
    p.add_argument('--no-raw-panel', dest='raw_panel', action='store_false')
    p.add_argument('--no-plot', action='store_true', help='metrics only')
    p.add_argument('--csv', default=CSV_PATH)
    p.add_argument('--fits-dir', default=FITS_DIR)
    p.add_argument('--raw-dir', default=RAW_DIR)
    p.add_argument('--out-dir', default=OUT_DIR)
    args = p.parse_args()
 
    FITS_DIR, RAW_DIR, OUT_DIR = args.fits_dir, args.raw_dir, args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
 
    rows = read_pointing(args.csv)
    if args.frmid:
        rows = [r for r in rows if r['frmid'] in args.frmid]
    elif args.visit:
        rows = [r for r in rows if r['visit'] == args.visit]
    elif not args.all:
        rows = rows[:1]
        print('no selection given; doing the first frame only (use --all)')
    if not rows:
        raise SystemExit('no matching rows in ' + args.csv)
 
    out = []
    for r in rows:
        print(f"{r['visit']} frmid {r['frmid']}")
        m = process_frame(r, args)
        if m:
            out.append(m)
 
    if out:
        keys = sorted({k for m in out for k in m})
        head = ['visit', 'frmid', 'date_obs', 'ra', 'dec', 'roll', 'pa_slit']
        keys = head + [k for k in keys if k not in head]
        path = os.path.join(OUT_DIR, 'slit_metrics.csv')
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for m in out:
                w.writerow(m)
        print(f'\nwrote {path}  ({len(out)} frames)')
 
 
if __name__ == '__main__':
    main()