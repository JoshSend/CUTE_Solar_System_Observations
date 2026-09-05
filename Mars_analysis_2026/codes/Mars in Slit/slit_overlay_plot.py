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

fits_path = r"C:\Users\mageb\OneDrive\Documents\CUTE_Solar_System_Observations\Mars_analysis_2026\codes\output\Mars_Images\Visit2\frmid4860.fits"
 
# ============================ USER CONFIG ============================
HERE       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(HERE, '..', 'output', 'mars_pointing.csv')
PDF_DIR    = os.path.join(HERE, 'mars_pdfs')      # holds <frmid>.pdf per frame
OUT_DIR    = os.path.join(HERE, 'overlay_out')    # figures + WCS FITS go here
 
FOV_ARCSEC = 23.0        # the FOV you requested from the viewer (== box width)
DPI        = 150         # rasterisation resolution for the PDF
 
# Roll -> on-sky angle. The roll handedness/zero-point is NOT verified against a
# resolved image (see mars_pointing.py docstring), so these two knobs let you
# calibrate the slit orientation against a frame you trust (e.g. the Visit4
# 180 deg flip). Start at (+1, 0); flip ROLL_SIGN or add ROLL_ZERO if the slit
# is mirrored / rotated the wrong way.
ROLL_SIGN  = +1.0
ROLL_ZERO  = 0.0         # degrees added to (ROLL_SIGN * roll)
 
# The real, stepped CUTE slit outline, in arcsec in the slit frame (from the
# original amsu4591/jose5987 code). px = across-slit, py = along-slit.
SLIT_PX = [-60.0, 60.0, 60.0, 30.0, 30.0, 15.0, 15.0, -15.0, -15.0, -30.0, -30.0, -60.0, -60.0]
SLIT_PY = [-591.807, -591.807, -351.805, -351.805, 351.805, 351.805, 680.817,
           680.817, 351.805, 351.805, -351.805, -351.805, -591.807]
# ====================================================================
 
 
def pdf_to_fits_wcs(pdf_path, fits_out, mars_ra_deg, mars_dec_deg,
                    fov_arcsec=FOV_ARCSEC, page=0, dpi=DPI, dark_thresh=128):
    """
    Rasterise a PDS-viewer Mars PDF, crop to the plotted field, and write a FITS
    with a TAN WCS centred on Mars. Opens in DS9 showing real alpha/delta.
    """
    # 1. render page to a grayscale raster
    pix = pymupdf.open(pdf_path)[page].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(np.float32)
 
    # 2. auto-detect the black axes box (its long frame lines)
    dark = img < dark_thresh
    cols = np.where(dark.sum(axis=0) > 0.4 * img.shape[0])[0]
    rows = np.where(dark.sum(axis=1) > 0.4 * img.shape[1])[0]
    if len(cols) < 2 or len(rows) < 2:
        raise RuntimeError(f"could not find the plot box in {pdf_path}")
    x0, x1, y0, y1 = cols.min(), cols.max(), rows.min(), rows.max()
    crop = img[y0:y1 + 1, x0:x1 + 1]
 
    # 3. flip vertically: image rows run top->bottom, FITS/DS9 run bottom->top
    crop = np.flipud(crop)
    ny, nx = crop.shape
 
    # 4. build the WCS: box width == FOV, RA increases to the LEFT
    scale_deg = (fov_arcsec / nx) / 3600.0
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]   # Mars at box centre (FITS 1-indexed)
    w.wcs.crval = [mars_ra_deg, mars_dec_deg]
    w.wcs.cdelt = [-scale_deg, scale_deg]            # CDELT1 < 0 -> east-left
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
 
    hdu = fits.PrimaryHDU(data=crop.astype(np.float32), header=w.to_header())
    hdu.writeto(fits_out, overwrite=True)
    return fits_out
 
 
def slit_polygon(ra, dec, roll):
    """CUTE slit outline as a closed list of (RA_deg, Dec_deg) points."""
    theta = np.deg2rad(ROLL_SIGN * roll + ROLL_ZERO)
    ct, st = np.cos(theta), np.sin(theta)
    cosd = np.cos(np.deg2rad(dec))
    coord = []
    for px, py in zip(SLIT_PX, SLIT_PY):
        east = px * ct - py * st          # arcsec, +east (= +RA direction)
        north = py * ct + px * st         # arcsec, +north
        ra_pt = ra + (east / 3600.0) / cosd   # cos(dec) correction for RA
        dec_pt = dec + north / 3600.0
        coord.append([ra_pt, dec_pt])
    coord.append(coord[0])                # close the loop
    return coord
 
 
def overlay_frame(row, fits_out, fig_out):
    """Build the WCS FITS for one frame and draw the slit over it."""
    frmid = row['frmid']
    ra, dec, roll = row['ra'], row['dec'], row['roll']
    pdf_path = os.path.join(PDF_DIR, f"{frmid}.pdf")
    if not os.path.exists(pdf_path):
        print(f"  frmid {frmid}: no PDF ({pdf_path}), skipping")
        return
 
    pdf_to_fits_wcs(pdf_path, fits_out, row['mars_ra'], row['mars_dec'])
 
    hdu = fits.open(fits_out)[0]
    wcs = WCS(hdu.header)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection=wcs)
    ax.imshow(hdu.data, origin='lower', cmap='gray')
    ax.set_xlabel('RA'); ax.set_ylabel('Dec')
    ax.set_title(f"{row['visit']}  frmid {frmid}")
    ax.set_autoscale_on(False)
 
    poly = Polygon(slit_polygon(ra, dec, roll), closed=True,
                   edgecolor='red', facecolor='none', lw=1.5,
                   transform=ax.get_transform('fk5'))
    ax.add_patch(poly)
    fig.savefig(fig_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  frmid {frmid}: wrote {os.path.basename(fig_out)}")
 
 
def _f(row, key):
    """Pull a float from a CSV row by the leading token of its unit-carrying header."""
    for k, v in row.items():
        if k.split(' ')[0] == key:
            return float(v)
    raise KeyError(key)
 
 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CSV_PATH) as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = dict(visit=raw['visit'], frmid=raw['frmid'],
                       ra=_f(raw, 'ra'), dec=_f(raw, 'dec'), roll=_f(raw, 'roll'),
                       mars_ra=_f(raw, 'mars_ra'), mars_dec=_f(raw, 'mars_dec'))
            stem = f"{row['visit']}_frmid{row['frmid']}"
            overlay_frame(row,
                          fits_out=os.path.join(OUT_DIR, stem + '.fits'),
                          fig_out=os.path.join(OUT_DIR, stem + '.png'))
 
 
if __name__ == '__main__':
    main()
