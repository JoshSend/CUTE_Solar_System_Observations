# -*- coding: utf-8 -*-
"""
Mars campaign pointing -- FITS headers in, mars_pointing.csv out.

Self-contained: reads CUTE_observations/Visit*/*.fits, converts each frame's
attitude quaternion to the telescope boresight RA / Dec / Roll, and writes one
CSV row per frame.  The only local dependency is cutepoint.py, which must sit
beside this file.

    python3 mars_pointing.py                  -> ./mars_pointing.csv
    python3 mars_pointing.py out/mars.csv     -> that path

--------------------------------------------------------------------------
Where the numbers come from
--------------------------------------------------------------------------
The L1 pipeline stamps the spacecraft attitude and orbit into every header, so
no BCT telemetry CSV and no TLE/SGP4 propagation is needed:

    XB1_AQ0..XB1_AQ3   S/C body-wrt-ECI quaternion (BCT order, scalar last,
                       maps onto cutepoint's e1..e4)
    XB1_OP0..XB1_OP2   orbit position, km, EARTH-FIXED (see below)
    XB1_TAI            seconds since 2000-01-01T00:00:00 TAI
    DATE-OBS           the same instant in UTC (verified identical)

The conversion is the two-step recipe from pld_quat_conversions.py:

    1. cp.sc_ra_dec_roll_from_sc_quat()  -> RA/Dec/Roll of the S/C body +X axis
    2. cp.rotate_coordinates(..., REV_ROT_VEC) -> swing +X onto the true
       telescope boresight

Step 2 is worth 0.7314 deg (2633"), so it is not optional.  Doing only step 1
is what sc_quat_conversions.py does, and it leaves you 0.73 deg off target.
ROLL comes out of step 1 and passes through unchanged -- the boresight
rotation moves the pointing, not the field rotation.

Validated against the JPL Mars ephemeris: median boresight-to-Mars residual is
~18" over the 102 campaign frames, against a 2633" boresight correction.

--------------------------------------------------------------------------
Things that cost time to work out -- do not undo them casually
--------------------------------------------------------------------------
* XB1_OP is EARTH-FIXED (ITRS/ECEF), NOT inertial.  Verified from the orbit
  plane: read as ITRS it reproduces CUTE's inclination (97.59 vs 97.5639 from
  the TLE) and sun-synchronous RAAN precession (1.02 vs 0.9856 deg/day); read
  as GCRS or TEME both give ~100.6 deg and no precession.  So it converts
  straight to an EarthLocation -- do NOT wrap it in GCRS and transform, that
  applies a spurious Earth rotation worth up to 13,700 km.
* Do NOT resurrect the TLE/SGP4 path from the old scripts.  Their newest TLE
  (epoch 2023-02-02) is ~700 days stale at these epochs and misplaces CUTE by
  5,500-12,900 km, inflating the Mars residual from ~13" to ~20".
* The DE432s ephemeris is load-bearing: astropy's analytic 'builtin' differs
  from it by 17" here.  Light-time (3.1") and CUTE's parallax (6.5") are
  accounted for.  Stellar aberration (11.2") is deliberately NOT applied --
  the star tracker frame is already astrometric, and applying it doubles the
  residual.
* ROLL is cutepoint's Tait-Bryan angle, wrapped to (-180, 180].  Its sign and
  zero point relative to a sky position angle are NOT verified against a
  resolved image; treat the handedness as unknown if you convert it to a PA.

--------------------------------------------------------------------------
Known oddities in the data (flagged in the summary, never silently dropped)
--------------------------------------------------------------------------
* Visit3's last frame (frmid 4929) is a valid attitude pointed 119 deg from
  Mars -- CUTE had already slewed off target.  Flagged 'off-target'.
* Visit4 is really TWO pointings: frames 4973-4980 at roll -25.16 deg and
  4981-4987, an orbit later, at +155.16 deg -- a 180.3 deg flip.  The CSV is
  per-frame so it just records both; do not average them.
* There is no Visit6 in the archive.
"""

import os
import sys
import glob
import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import (SkyCoord, EarthLocation, get_body,
                                 solar_system_ephemeris)
import astropy.units as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutepoint as cp

warnings.filterwarnings('ignore')
solar_system_ephemeris.set('de432s')

_HERE = os.path.dirname(os.path.abspath(__file__))

# Telescope boresight in the S/C body frame.  Nominally +X; the small y/z
# terms are the measured misalignment.  From pld_quat_conversions.py.
REV_ROT_VEC = [0.999919, -0.000289, -0.012762]

# XB1_TAI epoch, derived by matching DATE-OBS (not documented in the header).
TAI_EPOCH = Time('2000-01-01T00:00:00', format='isot', scale='tai')

OBS_ROOT = os.path.join(_HERE, '..', 'CUTE_observations')

# A frame further than this from Mars is reported as off target.  Note this
# is a slew-scale threshold, deliberately much looser than the slit.
ON_TARGET_ARCSEC = 300.0

# (row key, CSV header).  The two are kept separate so the header can carry
# units without having to rename the keys used throughout the code.  Header
# text must not contain a comma.
CSV_COLUMNS = [
    ('visit',    'visit'),
    ('file',     'file'),
    ('frmid',    'frmid'),
    ('date_obs', 'date_obs (UTC)'),
    ('tai',      'tai (s since 2000-01-01T00:00:00 TAI)'),
    ('exptime',  'exptime (ms)'),
    ('ra',       'ra (deg)'),
    ('dec',      'dec (deg)'),
    ('roll',     'roll (deg)'),
    ('mars_ra',  'mars_ra (deg)'),      
    ('mars_dec', 'mars_dec (deg)'),     
]


def frame_pointing(path, visit='', with_mars=True):
    """RA / Dec / Roll of the CUTE boresight for one science frame.

    Returns a dict with the CSV columns plus, when with_mars, the Mars
    ephemeris position and the boresight-to-Mars offset used for validation.
    """
    h = fits.getheader(path)

    # 1. S/C body +X axis, then 2. swing it onto the telescope boresight.
    ra_sc, dec_sc, roll = cp.sc_ra_dec_roll_from_sc_quat(
        h['XB1_AQ0'], h['XB1_AQ1'], h['XB1_AQ2'], h['XB1_AQ3'])
    ra, dec = cp.rotate_coordinates(ra_sc, dec_sc, roll, REV_ROT_VEC)

    out = dict(visit=visit, file=os.path.basename(path), frmid=h['FRM_ID'],
               date_obs=h['DATE-OBS'], tai=h['XB1_TAI'], exptime=h['EXPTIME'],
               ra=ra, dec=dec, roll=roll,
               ra_sc=ra_sc, dec_sc=dec_sc,
               n_gd_pkt=h.get('N_GD_PKT', 0), n_filled=h.get('N_FILLED', 0))
    if not with_mars:
        return out

    t = TAI_EPOCH + h['XB1_TAI'] * u.s
    # XB1_OP is Earth-fixed -- straight to EarthLocation, no frame transform.
    loc = EarthLocation.from_geocentric(
        *(np.array([h['XB1_OP0'], h['XB1_OP1'], h['XB1_OP2']]) * u.km))

    mars = get_body('mars', t, loc)
    mars_c = SkyCoord(mars.ra.deg, mars.dec.deg, unit=(u.deg, u.deg))
    c = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    dra, ddec = c.spherical_offsets_to(mars_c)

    out.update(mars_ra=mars.ra.deg, mars_dec=mars.dec.deg,
               sep_arcsec=c.separation(mars_c).arcsec,
               dra_arcsec=dra.arcsec, ddec_arcsec=ddec.arcsec)
    return out


def list_visits(obs_root=OBS_ROOT):
    """Visit names present in the archive, numeric order.  (No Visit6.)"""
    dirs = glob.glob(os.path.join(obs_root, 'Visit*'))
    return sorted((os.path.basename(d) for d in dirs if os.path.isdir(d)),
                  key=lambda v: int(v.replace('Visit', '')))


def read_all(obs_root=OBS_ROOT):
    """Every frame in every visit, time ordered within each visit.

    Nothing is dropped.  Suspect frames are flagged instead:
        off_target  further than ON_TARGET_ARCSEC from Mars
        duplicate   another frame shares this FRM_ID with better packet
                    recovery (highest N_GD_PKT wins, ties by fewest N_FILLED)
    """
    rows = []
    for visit in list_visits(obs_root):
        vrows = [frame_pointing(f, visit)
                 for f in sorted(glob.glob(
                     os.path.join(obs_root, visit, '*.fits')))]
        vrows.sort(key=lambda r: r['tai'])

        # Rank copies of a shared FRM_ID; never just take the first, because
        # filename order puts the LOWEST packet count first.
        best = {}
        for i, r in enumerate(vrows):
            key = (r['n_gd_pkt'], -r['n_filled'], i)
            if r['frmid'] not in best or key > best[r['frmid']][0]:
                best[r['frmid']] = (key, i)
        for i, r in enumerate(vrows):
            r['duplicate'] = best[r['frmid']][1] != i
            r['off_target'] = r['sep_arcsec'] > ON_TARGET_ARCSEC

        rows.extend(vrows)
    return rows


def write_csv(rows, out_csv):
    """One row per frame: identification plus the pointing.

    Column headers carry units -- ra/dec/roll in degrees, exptime in
    milliseconds, tai in seconds from its 2000 epoch.  Read it back with
    e.g. pandas.read_csv(...)['ra (deg)'].
    """
    outdir = os.path.dirname(out_csv)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)

    bad = [hdr for _, hdr in CSV_COLUMNS if ',' in hdr]
    if bad:
        raise ValueError('comma in CSV header would corrupt the file: %s' % bad)

    with open(out_csv, 'w') as fh:
        fh.write(','.join(hdr for _, hdr in CSV_COLUMNS) + '\n')
        for r in rows:
            fh.write(','.join(
                r[k] if isinstance(r[k], str) else
                ('%d' % r[k] if isinstance(r[k], (int, np.integer))
                 else '%.6f' % r[k])
                for k, _ in CSV_COLUMNS) + '\n')


def summarise(rows):
    """Per-visit summary and the end-to-end sanity checks."""
    print('%-8s %3s  %-19s %10s %9s %10s  %11s  %s'
          % ('visit', 'n', 'first DATE-OBS', 'RA (Deg)', 'Dec (Deg)', 'Roll (Deg)',
             'Mars sep"', 'flags'))
    for visit in list_visits():
        rr = [r for r in rows if r['visit'] == visit]
        if not rr:
            continue
        sep = np.array([r['sep_arcsec'] for r in rr])
        good = sep[sep <= ON_TARGET_ARCSEC]
        flags = []
        n_off = sum(1 for r in rr if r['off_target'])
        n_dup = sum(1 for r in rr if r['duplicate'])
        if n_off:
            flags.append('%d off-target' % n_off)
        if n_dup:
            flags.append('%d duplicate' % n_dup)
        rolls = np.array([r['roll'] for r in rr])
        if np.ptp(rolls) > 1.0:
            flags.append('SPLIT: rolls %.1f/%.1f'
                         % (rolls.min(), rolls.max()))
        print('%-8s %3d  %-19s %10.4f %9.4f %10.4f  %11.1f  %s'
              % (visit, len(rr), rr[0]['date_obs'][:19],
                 rr[0]['ra'], rr[0]['dec'], rr[0]['roll'],
                 np.median(good) if len(good) else float('nan'),
                 ', '.join(flags)))

    off = np.array([
        SkyCoord(r['ra_sc'], r['dec_sc'], unit=(u.deg, u.deg)).separation(
            SkyCoord(r['ra'], r['dec'], unit=(u.deg, u.deg))).deg
        for r in rows])
    sep = np.array([r['sep_arcsec'] for r in rows])
    good = sep[sep <= ON_TARGET_ARCSEC]
    print('\nboresight correction applied : %.4f deg (%.1f"), spread %.2g deg'
          % (off.mean(), off.mean() * 3600.0, off.std()))
    print('Mars-to-boresight residual   : median %.1f"  max %.1f"  (%d '
          'on-target frames)' % (np.median(good), good.max(), len(good)))


def main():
    out_csv = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
        else os.path.join(os.getcwd(), 'mars_pointing.csv')

    rows = read_all()
    write_csv(rows, out_csv)
    print('wrote %s  (%d frames, %d bytes)\n'
          % (out_csv, len(rows), os.path.getsize(out_csv)))
    summarise(rows)


if __name__ == '__main__':
    main()
