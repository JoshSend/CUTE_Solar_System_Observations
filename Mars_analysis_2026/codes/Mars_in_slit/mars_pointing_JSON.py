# -*- coding: utf-8 -*-
"""
Mars campaign COMMANDED pointing -- JSON plans in, mars_pointing_JSON.csv out.

Companion to mars_pointing.py.  That script derives RA / Dec / Roll from the
MEASURED attitude quaternion the ADCS stamped into each FITS header
(XB1_AQ0..3).  This one derives the same quantities from the COMMANDED
quaternion in the uplinked plan (Q_CMD_WRT_REF_1..4 of the GOTO_ECI_ATTITUDE
that governs each science block), for every science frame ID the plan
scheduled -- whether or not that frame was ever downlinked.

    python3 mars_pointing_JSON.py                     -> ./mars_pointing_JSON.csv
    python3 mars_pointing_JSON.py out/cmd.csv         -> that path

The CSV has the same nine columns, in the same order with the same headers,
as mars_pointing.csv, so the two can be joined on frmid directly:

    visit, file, frmid, date_obs (UTC), tai (...), exptime (ms),
    ra (deg), dec (deg), roll (deg)

with these differences in meaning:
    file       the FITS file in CUTE_observations whose FRM_ID matches, or
               'N/A' if the plan scheduled the frame but no file exists.
    date_obs   the PLANNED exposure start, not the recorded one.  The plan
               only gives the start time of each block; frames within the
               block are spaced by TIME + READOUT_OVERHEAD_S (see below).
    tai        that planned UTC converted to seconds since the XB1_TAI epoch.
    exptime    the commanded TIME (ms) from the plan.
    ra/dec/roll  from the commanded quaternion, via the identical two-step
               cutepoint conversion mars_pointing.py uses.

--------------------------------------------------------------------------
How each frame is paired with its command
--------------------------------------------------------------------------
A plan's command list runs  GOTO(bias) -> GOTO(dark) -> GOTO(Mars, orbit 1)
-> GOTO(Mars, orbit 2) -> GOTO(dark) -> GOTO(bias), each GOTO followed by a
PLD_CCD_SET_ID and a PLD_CCD_EXPOSE.  A science frame belongs to the block
whose  SET_ID <= FRM_ID < SET_ID + NUM,  and its governing command is the
GOTO issued immediately BEFORE that block's EXPOSE -- not the next one down
the list, and not the one nearest in time (the GOTO leads the exposure by
5-7 min to let the slew settle).  Getting this pairing wrong is the error
that made an earlier hand-built comparison look like a mismatch.

Dark (TARGETID 255) and bias (TARGETID 0, TIME 0) blocks are ignored.

--------------------------------------------------------------------------
Things worth knowing before comparing this with mars_pointing.csv
--------------------------------------------------------------------------
* The commanded and measured attitudes agree to ~8.5" median over the 102
  downlinked frames.  Larger departures are real ACS events, not plan errors:
  4929 (post-slew), 5034 and 5259 (excursions at the exposure-start
  snapshot), 4867, 5043, 5210 (smaller pointing/roll departures).
* The header quaternion is the NEGATIVE of the command (-q) in Visit4 orbit 2
  and all of Visits 5, 7, 8.  q and -q are the same rotation, so RA/Dec/Roll
  come out identical; only the raw components differ in sign.
* Every quaternion here is one value per BLOCK, repeated for each frame in
  it.  Frame-to-frame variation in mars_pointing.csv is ACS jitter that the
  plan, by construction, cannot show.
* Only plans present in JSON_ROOT are read.  Plans that never executed
  (12.29.2024, 01.07.2025, 02.10.2025) were removed from that folder; drop
  them back in and their frames will appear here as N/A rows.
"""

import os
import sys
import glob
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutepoint as cp

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))

# Keep these two identical to mars_pointing.py -- the CSVs are only
# comparable if both scripts use the same boresight vector and time epoch.
REV_ROT_VEC = [0.999919, -0.000289, -0.012762]
TAI_EPOCH = Time('2000-01-01T00:00:00', format='isot', scale='tai')

JSON_ROOT = os.path.join(_HERE, '..', 'Mars_JSON')
OBS_ROOT = os.path.join(_HERE, '..', 'CUTE_observations')

SCIENCE_TARGETID = 340          # Mars.  255 = dark, 0 = bias.

# Seconds from the end of one exposure to the start of the next within a
# block (CCD readout + packetising).  Not in the plan; measured from the
# data: 60 s exposures came 125 s apart, 100 s exposures 165 s apart.  It is
# only used to spread the planned frame times inside a block; summarise()
# checks it against the recorded DATE-OBS of every downlinked frame.
READOUT_OVERHEAD_S = 65.0

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
]


def _plan_time(s):
    """'2025/019-05:09:25' -> datetime."""
    return datetime.strptime(s, '%Y/%j-%H:%M:%S')


def pointing_from_quat(q):
    """RA / Dec / Roll (deg) of the boresight for one quaternion.

    Identical to the conversion in mars_pointing.frame_pointing, so the two
    CSVs differ only in where the quaternion came from.
    """
    ra_sc, dec_sc, roll = cp.sc_ra_dec_roll_from_sc_quat(*q)
    ra, dec = cp.rotate_coordinates(ra_sc, dec_sc, roll, REV_ROT_VEC)
    return ra, dec, roll


def science_blocks(plan_path):
    """Science (TARGETID 340) blocks in one plan, each with its governing GOTO.

    Returns a list of dicts: id0, n, exptime_ms, t0 (planned EXPOSE time),
    q_cmd, goto_t.  Walks the command list in order and remembers the most
    recent GOTO and SET_ID so that each EXPOSE is paired with the commands
    that precede it.
    """
    with open(plan_path) as fh:
        plan = json.load(fh)
    blocks = []
    for sc in plan['storedCommands']:
        last_goto = None
        last_id = None
        for c in sc['commands']:
            m, a = c['mnemonic'], c['args']
            if m == 'GOTO_ECI_ATTITUDE':
                last_goto = (_plan_time(c['utc_time']),
                             [a['Q_CMD_WRT_REF_%d' % i] for i in (1, 2, 3, 4)])
            elif m == 'PLD_CCD_SET_ID':
                last_id = int(a['ID'])
            elif m.startswith('PLD_CCD_EXPOSE'):
                if int(a['TARGETID']) != SCIENCE_TARGETID:
                    continue                      # dark or bias -- skip
                if last_goto is None or last_id is None:
                    raise RuntimeError('%s: EXPOSE at %s has no preceding '
                                       'GOTO/SET_ID' % (plan_path, c['utc_time']))
                blocks.append(dict(id0=last_id, n=int(a['NUM']),
                                   exptime_ms=int(a['TIME']),
                                   t0=_plan_time(c['utc_time']),
                                   goto_t=last_goto[0], q_cmd=last_goto[1]))
    return blocks


def visit_label(plan_path, frmids, index):
    """'VisitN' for a plan.

    Taken from a _VisitN suffix on the file name if there is one; otherwise
    from the CUTE_observations folder that holds any of its frames; otherwise
    'N/A' (a plan that never produced data).
    """
    base = os.path.basename(plan_path)
    if '_Visit' in base:
        return 'Visit' + base.split('_Visit')[1].split('.')[0]
    for fid in frmids:
        if fid in index:
            return index[fid]['visit']
    return 'N/A'


def frame_index(obs_root=OBS_ROOT):
    """FRM_ID -> {file, visit, date_obs, q_hdr, n_gd_pkt} for the archive.

    If two files share a FRM_ID (a re-downlinked frame) the one with more
    good packets is kept.
    """
    index = {}
    for f in glob.glob(os.path.join(obs_root, 'Visit*', '*.fits')):
        h = fits.getheader(f)
        rec = dict(file=os.path.basename(f), visit=f.split(os.sep)[-2],
                   date_obs=h['DATE-OBS'], n_gd_pkt=h.get('N_GD_PKT', 0),
                   q_hdr=[h['XB1_AQ%d' % i] for i in range(4)])
        old = index.get(h['FRM_ID'])
        if old is None or rec['n_gd_pkt'] > old['n_gd_pkt']:
            index[h['FRM_ID']] = rec
    return index


def read_all(json_root=JSON_ROOT, obs_root=OBS_ROOT):
    """One row per planned science frame across every plan in json_root."""
    index = frame_index(obs_root)
    rows = []
    for plan_path in sorted(glob.glob(os.path.join(json_root, '*.json'))):
        blocks = science_blocks(plan_path)
        frmids = [b['id0'] + i for b in blocks for i in range(b['n'])]
        visit = visit_label(plan_path, frmids, index)
        for b in blocks:
            ra, dec, roll = pointing_from_quat(b['q_cmd'])
            cadence = b['exptime_ms'] / 1000.0 + READOUT_OVERHEAD_S
            for i in range(b['n']):
                fid = b['id0'] + i
                t_plan = b['t0'] + timedelta(seconds=i * cadence)
                t_astropy = Time(t_plan, scale='utc')
                rec = index.get(fid)
                rows.append(dict(
                    visit=visit, plan=os.path.basename(plan_path),
                    file=rec['file'] if rec else 'N/A', frmid=fid,
                    date_obs=t_plan.strftime('%Y-%m-%dT%H:%M:%S.000'),
                    tai=int(round((t_astropy.tai - TAI_EPOCH).to(u.s).value)),
                    exptime=b['exptime_ms'],
                    ra=ra, dec=dec, roll=roll,
                    q_cmd=b['q_cmd'], goto_t=b['goto_t'],
                    # kept for the summary only
                    date_obs_actual=rec['date_obs'] if rec else None,
                    q_hdr=rec['q_hdr'] if rec else None))
    rows.sort(key=lambda r: (int(r['visit'][5:]) if r['visit'].startswith('Visit')
                             else 999, r['frmid']))
    return rows


def write_csv(rows, out_csv):
    """Same nine columns as mars_pointing.csv."""
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


def _rotation_arcsec(q1, q2):
    """Angle between two attitudes, sign-blind (q and -q are one rotation)."""
    a = np.asarray(q1, float); a /= np.linalg.norm(a)
    b = np.asarray(q2, float); b /= np.linalg.norm(b)
    return 2.0 * np.degrees(np.arccos(min(1.0, abs(a @ b)))) * 3600.0


def summarise(rows):
    """Per-plan bookkeeping plus two checks against the downlinked frames."""
    print('%-8s %-32s %7s %5s %5s  %s' % (
        'visit', 'plan', 'planned', 'files', 'N/A', 'cmd-vs-header rotation (")'))
    plans = []
    for r in rows:
        if r['plan'] not in plans:
            plans.append(r['plan'])
    for p in plans:
        rr = [r for r in rows if r['plan'] == p]
        have = [r for r in rr if r['q_hdr'] is not None]
        rot = [_rotation_arcsec(r['q_cmd'], r['q_hdr']) for r in have]
        rot_txt = ('median %5.1f  max %7.1f' % (np.median(rot), max(rot))
                   if rot else 'no frames to compare')
        print('%-8s %-32s %7d %5d %5d  %s' % (
            rr[0]['visit'], p, len(rr), len(have), len(rr) - len(have), rot_txt))

    have = [r for r in rows if r['date_obs_actual'] is not None]
    if have:
        dt = np.array([
            (datetime.strptime(r['date_obs_actual'][:19], '%Y-%m-%dT%H:%M:%S')
             - datetime.strptime(r['date_obs'][:19], '%Y-%m-%dT%H:%M:%S')
             ).total_seconds() for r in have])
        print('\nplanned vs recorded DATE-OBS on the %d downlinked frames: '
              '%+.0f .. %+.0f s (median %+.0f s)  '
              '[checks READOUT_OVERHEAD_S = %.0f]'
              % (len(have), dt.min(), dt.max(), np.median(dt),
                 READOUT_OVERHEAD_S))
    n_na = sum(1 for r in rows if r['file'] == 'N/A')
    print('%d planned science frames, %d with a data file, %d N/A'
          % (len(rows), len(rows) - n_na, n_na))


def main():
    out_csv = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
        else os.path.join(os.getcwd(), 'mars_pointing_JSON.csv')

    rows = read_all()
    write_csv(rows, out_csv)
    print('wrote %s  (%d frames, %d bytes)\n'
          % (out_csv, len(rows), os.path.getsize(out_csv)))
    summarise(rows)


if __name__ == '__main__':
    main()
