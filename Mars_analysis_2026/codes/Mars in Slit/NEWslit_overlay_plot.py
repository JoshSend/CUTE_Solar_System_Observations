# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:42:26 2022
@author: amsu4591

Modified for CUTE Mars Campaign in September 2026
@author: jose5987
"""
import io
import re
import time
import traceback
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS

# ------------------------------- Directories -------------------------------
base_dir = Path(__file__).resolve().parent   # ...\codes\Mars In Slit\
project_root = base_dir.parent.parent        # ...\Mars_analysis_2026\
codes_dir = project_root / 'codes'           # ...\codes\
out_dir = codes_dir / 'output'               # ...\codes\output\

csv_dir = out_dir / 'mars_pointing.csv'
fits_dir = out_dir / 'Mars_Fits'             # organised as Mars_Fits\Visit*\
output_dir = out_dir / 'Slit_Overlay'        # written as Slit_Overlay\Visit*\
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------- Settings --------------------------------
# What to produce.  'gif' and 'png' each render every figure exactly once;
# 'both' renders twice per frame (once cropped to disk, once at fixed size for
# the animation) and therefore takes about twice as long.
#     'gif'   one animation per visit per FOV, no PNGs        <- fastest
#     'png'   one PNG per frame, no animation
#     'both'  both, at roughly double the runtime
MODE = 'attitude'

SHOW = True         # pop up each figure.  Leave False for a batch run --
                     # plt.show() on ~200 figures will stall the session.
OVERWRITE = True     # False skips work whose output already exists: in 'png'
                     # mode a frame whose PNG is present, in 'gif' mode an
                     # entire visit whose animation is present (so an
                     # interrupted run resumes without re-rendering).

SAVE_PNG = MODE in ('png', 'both')
MAKE_GIF = MODE in ('gif', 'both')
if MODE not in ('gif', 'png', 'both', 'attitude'):
    raise SystemExit(f"MODE must be 'gif', 'png', 'both' or 'attitude', "
                     f"not {MODE!r}")

# Filters.  None = no filter, i.e. everything found on disk.
VISITS = None        # e.g. ['Visit1', 'Visit2']
FRMIDS = None        # e.g. [3726, 3727]
FOVS = None          # e.g. ['5x5MR'];  None plots every FOV present

# Frames dropped everywhere -- rendering and analysis alike.
# Visit3 frmid 4929: the CSV puts the boresight at RA 227.19 Dec -33.73 while
# the rest of Visit3 sits at RA 117.39 Dec +25.44, i.e. Mars is 119.5 deg away.
# It is also the frame whose NaN projection breaks the VIEW_ARCSEC block.
EXCLUDE_FRMIDS = [4929]

# ---- attitude-error analysis  (MODE = 'attitude')
ATT_CSV_NAME = 'CUTE_mars_attitude_error'
ATT_FOV = '5x5MR'          # whose WCS gives Mars' position; see mars_centre()
ATT_REF_ARCSEC = 30.0      # reference line: half-width of the 60" science band
ATT_GAP_MIN = 15.0         # break a visit's line across a gap longer than this.
                           # Visits run in two orbit passes ~65 min apart; drawing
                           # straight through implies frames that do not exist.

# Categorical palette, fixed order, one slot per visit.  Validated with the
# dataviz validator (light, surface #fcfcfb): lightness band, chroma floor,
# worst adjacent CVD dE 9.1, normal-vision dE 19.6 -- all pass.  Do not cycle:
# a 9th visit needs a deliberate choice, not a generated hue.
ATT_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
              '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'

# Draw a second slit at a fixed roll of 90 deg for comparison (was the
# theta = 90 block at the bottom of the old script).
SHOW_REFERENCE_SLIT = False

# Plotted field of view, as a HALF-width in arcsec centred on the boresight.
# None = the native extent of the image array.  The 5x5MR array is only 33.94"
# across while the nearest slit wall is 30" out, so its native extent shows no
# wall at all; 35" brings the two science walls in with Mars still resolved.
VIEW_ARCSEC = {'5x5MR': 35.0, '23x23arcmin': None}

# ---- animation.  One GIF per visit per FOV, frames in ascending frmid order.
GIF_MS = 500         # milliseconds per frame
GIF_LOOP = 0         # 0 = loop forever
GIF_MIN_FRAMES = 2   # do not bother writing a one-frame animation

# Figure size.  GIF frames come out at FIG_INCHES * GIF_DPI px, so leaving
# these as-is gives 900 px frames with no resampling step at all.
FIG_INCHES = 9.0
GIF_DPI = 100        # animation frames
PNG_DPI = 150        # standalone PNGs

if not SHOW:
    matplotlib.use('Agg')        # no GUI needed for a batch run

# ------------------------------- Slit outline ------------------------------
# arcsec in the slit frame.  px = across-slit, py = along-slit.
px0 = [-60.0, 60.0, 60.0, 30.0, 30.0, 15.0, 15.0, -15.0, -15.0, -30.0, -30.0, -60.0, -60.0]
py0 = [-591.807, -591.807, -351.805, -351.805, 351.805, 351.805, 680.817, 680.817,
       351.805, 351.805, -351.805, -351.805, -591.807]


def slit_polygon(ra, dec, theta):
    """CUTE slit outline as a closed (14, 2) array of [RA, Dec] in degrees.

    ra, dec, theta must be SCALARS -- one frame.  The rotation is the original
    amsu4591 one, which puts the slit long axis (+py) at position
    angle -theta east of north.  That matches the ADCS quaternion: for frmid
    3726 the position angle of S/C +Z is 9.0561 deg against a CSV roll of
    -9.056058 deg.
    """
    ct = np.cos(np.deg2rad(theta))
    st = np.sin(np.deg2rad(theta))
    cosd = np.cos(np.deg2rad(dec))          # RA degrees are compressed at high |dec|

    coord = []
    for i in range(len(px0)):
        d_ra = (px0[i] / 3600 * ct - py0[i] / 3600 * st) / cosd
        d_dec = py0[i] / 3600 * ct + px0[i] / 3600 * st
        coord.append([ra + d_ra, dec + d_dec])
    coord.append(coord[0])                  # repeat the first point -> closed loop
    return np.array(coord)


def safe(text):
    """date_obs contains ':' which Windows will not accept in a filename."""
    return str(text).replace(':', '-')


def visit_key(name):
    """Sort Visit2 before Visit10 instead of after it."""
    m = re.search(r'(\d+)', str(name))
    return (int(m.group(1)) if m else 10**9, str(name))


# ---------------------------- Find the FITS files --------------------------
NAME_RE = re.compile(r'^frmid(\d+)[_-](.+)$', re.IGNORECASE)


def discover_fits():
    """Every PlanetMapper FITS under fits_dir, as (visit_folder, frmid, fov, path).

    Filenames are expected to be frmid<id>_<fov>.fits.  The visit is taken from
    the containing folder when that folder is named Visit*, otherwise it is
    filled in later from the CSV.
    """
    found, unparsed = [], []
    for path in sorted(fits_dir.rglob('*.fits')):
        m = NAME_RE.match(path.stem)
        if not m:
            unparsed.append(path)
            continue
        folder = path.parent.name
        found.append(dict(folder=folder if folder.lower().startswith('visit') else None,
                          frmid=int(m.group(1)), fov=m.group(2), path=path))
    if unparsed:
        print(f'{len(unparsed)} file(s) did not match frmid<id>_<fov>.fits and were skipped:')
        for p in unparsed[:10]:
            print('   ', p.relative_to(fits_dir))
    return found


# --------------------------------- Pointing --------------------------------
df = pd.read_csv(csv_dir)
df = df.drop_duplicates(subset='frmid', keep='first')   # a few frmids repeat
pointing = {int(r['frmid']): r for _, r in df.iterrows()}


# ------------------------------- Plot one frame ----------------------------
def plot_frame(entry, row, save=SAVE_PNG, show=SHOW, capture=MAKE_GIF):
    """Draw the slit over one PlanetMapper image.

    Returns (png_path_or_None, PIL.Image_or_None).  Each figure is rendered to
    disk at most once: PNGs use bbox_inches='tight', which crops to the drawn
    content and so varies in size frame to frame, while GIF needs every frame
    identical -- hence the separate fixed-size render, taken only when MODE
    asks for an animation.
    """
    visit = row['visit']
    frmid = entry['frmid']
    fov = entry['fov']
    date_obs = row['date_obs (UTC)']
    ra = float(row['ra (deg)'])            # scalars, one frame
    dec = float(row['dec (deg)'])
    theta = float(row['roll (deg)'])

    name = (f'overlay_{visit}_frmid{frmid}_{safe(date_obs)}'
            f'_ra{ra:.4f}_dec{dec:.4f}_roll{theta:.4f}_{fov}.png')
    visit_dir = output_dir / visit
    dest = visit_dir / name
    write_png = save and (OVERWRITE or not dest.exists())
    if not write_png and not capture:
        return None, None

    with fits.open(entry['path']) as hdul:
        hdu = hdul[0]
        data = hdu.data
        wcs = WCS(hdu.header)

    fig = plt.figure(figsize=(FIG_INCHES, FIG_INCHES))
    ax = fig.add_subplot(projection=wcs)
    ax.imshow(data, origin='lower', cmap='hot_r')   # origin matters
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_autoscale_on(False)

    coord = slit_polygon(ra, dec, theta)
    ax.add_patch(Polygon(coord, closed=True, edgecolor='k', facecolor='none', lw=1.5,
                         transform=ax.get_transform('world')))

    # boresight
    ax.plot(ra, dec, '+', color='deepskyblue', ms=14, mew=2, alpha=0.5,
            transform=ax.get_transform('world'))

    if SHOW_REFERENCE_SLIT:
        ref = slit_polygon(ra, dec, 90.0)
        ax.add_patch(Polygon(ref, closed=True, edgecolor='w', facecolor='none',
                             lw=1.0, ls='--', transform=ax.get_transform('world')))

    # Field of view.  Limits are in pixels, so convert the requested half-width
    # in arcsec using the WCS pixel scale.  Values larger than the array simply
    # zoom out past the edge of the data.
    half = VIEW_ARCSEC.get(fov)
    if half is not None:
        scale = np.hypot(*wcs.pixel_scale_matrix[:, 0]) * 3600.0   # arcsec/pixel
        xb, yb = wcs.wcs_world2pix(ra, dec, 0)                     # boresight pixel
        ax.set_xlim(float(xb) - half / scale, float(xb) + half / scale)
        ax.set_ylim(float(yb) - half / scale, float(yb) + half / scale)

    ax.set_title(f'{visit}  frmid {frmid}  {date_obs}\n'
                 f'ra={ra:.5f}  dec={dec:.5f}  roll={theta:.4f}  {fov} FOV'
                 + (f'  view ±{half:.0f}″' if half else ''))

    # Grab the canvas before show(), which in a GUI backend blocks and can
    # tear the figure down on close.
    frame = None
    if capture:
        # Render to an in-memory PNG rather than reading fig.canvas.buffer_rgba():
        # that attribute only exists on Agg-family canvases, so it breaks under
        # Spyder/PyCharm inline backends.  savefig with no bbox_inches gives
        # exactly figsize x dpi pixels, identical for every frame.
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=GIF_DPI)
        buf.seek(0)
        frame = Image.open(buf).convert('RGB')

    written = None
    if write_png:
        visit_dir.mkdir(parents=True, exist_ok=True)   # Slit_Overlay\Visit*\
        fig.savefig(dest, dpi=PNG_DPI, bbox_inches='tight')
        written = dest
    if show:
        plt.show()
    plt.close(fig)                                     # or a batch run eats memory
    return written, frame


def write_gif(visit, fov, frames):
    """frames: list of (frmid, PIL.Image).  Writes Slit_Overlay\\<Visit>\\*.gif"""
    frames = sorted(frames, key=lambda f: f[0])        # ascending frmid
    if len(frames) < GIF_MIN_FRAMES:
        print(f'{visit} [{fov}]: only {len(frames)} frame(s), no GIF')
        return None
    sizes = {im.size for _, im in frames}
    if len(sizes) > 1:                                 # should not happen
        w = min(s[0] for s in sizes)
        h = min(s[1] for s in sizes)
        frames = [(i, im.resize((w, h), Image.LANCZOS)) for i, im in frames]
        print(f'{visit} [{fov}]: frames differed in size, cropped to {w}x{h}')

    visit_dir = output_dir / visit
    visit_dir.mkdir(parents=True, exist_ok=True)
    dest = visit_dir / f'animation_{visit}_{fov}.gif'
    images = [im for _, im in frames]
    images[0].save(dest, save_all=True, append_images=images[1:],
                   duration=GIF_MS, loop=GIF_LOOP, disposal=2, optimize=False)
    span = f'{frames[0][0]}-{frames[-1][0]}'
    print(f'{visit} [{fov}]: wrote {visit}\\{dest.name}  '
          f'({len(images)} frames, frmid {span})')
    return dest


# -------------------------- Attitude-error analysis ------------------------
def angular_sep(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees between two sky positions."""
    def unit(a, d):
        a, d = np.radians(a), np.radians(d)
        return np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    return float(np.degrees(np.arccos(np.clip(unit(ra1, dec1) @ unit(ra2, dec2), -1, 1))))


def mars_centre(wcs, shape):
    """Sky position of Mars' centre, in degrees.

    NOT CRVAL.  The generator puts Mars at x0 = y0 = (NPIX-1)/2 = 127.5 but
    writes CRPIX at pixel index 128 -- half a pixel apart.  That is 0.09" at
    the 5x5MR scale and 3.81" in the 23x23arcmin file, which would swamp the
    ~10" signal measured here.  Going through the WCS at 127.5 makes the two
    FOVs agree to 0.000".
    """
    ny, nx = shape
    ra, dec = wcs.wcs_pix2world((nx - 1) / 2.0, (ny - 1) / 2.0, 0)
    return float(ra), float(dec)


def measure_attitude():
    """One row per frame: where Mars sat relative to where CUTE was pointing."""
    groups, missing = collect()
    keys = [k for k in groups if k[1] == ATT_FOV]
    if not keys:
        keys = list(groups)
        print(f'note: no {ATT_FOV} files found, using '
              f'{sorted({k[1] for k in keys})} instead')
    if not keys:
        raise SystemExit('no frames to measure')

    rows = []
    for key in sorted(keys, key=lambda k: (visit_key(k[0]), k[1])):
        visit, fov = key
        for entry, row in groups[key]:
            try:
                with fits.open(entry['path']) as hdul:
                    hdu = hdul[0]
                    wcs = WCS(hdu.header)
                    data = hdu.data
                mra, mdec = mars_centre(wcs, data.shape)

                ra = float(row['ra (deg)'])
                dec = float(row['dec (deg)'])
                theta = float(row['roll (deg)'])

                east = (mra - ra) * np.cos(np.radians(dec)) * 3600.0
                north = (mdec - dec) * 3600.0
                t = np.radians(-theta)              # PA of the +py slit axis is -roll
                c, sn = np.cos(t), np.sin(t)

                rows.append(dict(
                    visit=visit, frmid=entry['frmid'], fov=fov,
                    date_obs=row['date_obs (UTC)'],
                    tai_s=float(row['tai (s since 2000-01-01T00:00:00 TAI)']),
                    boresight_ra_deg=ra, boresight_dec_deg=dec, roll_deg=theta,
                    mars_ra_deg=mra, mars_dec_deg=mdec,
                    offset_arcsec=angular_sep(ra, dec, mra, mdec) * 3600.0,
                    offset_east_arcsec=east, offset_north_arcsec=north,
                    offset_across_slit_arcsec=east * c - north * sn,
                    offset_along_slit_arcsec=east * sn + north * c))
            except Exception:
                print(f"  ERROR measuring {entry['path'].name}")
                traceback.print_exc()

    if not rows:
        raise SystemExit('no frames measured')
    out = pd.DataFrame(rows)
    out['t_min'] = np.nan                      # minutes since the visit's first frame
    for _, g in out.groupby('visit'):
        g = g.sort_values('tai_s')
        out.loc[g.index, 't_min'] = (g['tai_s'] - g['tai_s'].iloc[0]) / 60.0
    return out.sort_values(['visit', 'frmid']).reset_index(drop=True), missing


def summarise(out):
    """Per-visit statistics."""
    g = out.groupby('visit')['offset_arcsec']
    s = pd.DataFrame({
        'n_frames': g.size(),
        'duration_min': out.groupby('visit')['t_min'].max(),
        'mean_offset_arcsec': g.mean(),
        'std_offset_arcsec': g.std(ddof=0),
        'median_offset_arcsec': g.median(),
        'min_offset_arcsec': g.min(),
        'max_offset_arcsec': g.max(),
        'mean_across_slit_arcsec': out.groupby('visit')['offset_across_slit_arcsec'].mean(),
        'mean_along_slit_arcsec': out.groupby('visit')['offset_along_slit_arcsec'].mean(),
    })
    return s.reindex(sorted(s.index, key=visit_key))


def plot_attitude(out, summary, dest):
    """One plot: offset vs minutes into the visit, one line per visit."""
    visits = sorted(out['visit'].unique(), key=visit_key)
    if len(visits) > len(ATT_COLORS):
        raise SystemExit(f'{len(visits)} visits but only {len(ATT_COLORS)} validated '
                         f'colours -- add a slot deliberately, do not cycle')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)

    for i, v in enumerate(visits):
        g = out[out['visit'] == v].sort_values('t_min')
        x = g['t_min'].to_numpy(dtype=float)
        y = g['offset_arcsec'].to_numpy(dtype=float)
        if ATT_GAP_MIN:                    # NaN at each gap breaks the line there
            brk = np.nonzero(np.diff(x) > ATT_GAP_MIN)[0] + 1
            x = np.insert(x, brk, np.nan)
            y = np.insert(y, brk, np.nan)
        mean = summary.loc[v, 'mean_offset_arcsec']
        ax.plot(x, y, '-o', lw=2, ms=6,
                color=ATT_COLORS[i], mec='#fcfcfb', mew=1.0,
                label=f'{v}  (mean {mean:.1f}\u2033)')

    ax.axhline(ATT_REF_ARCSEC, color=INK_2, lw=1.2, ls='--', alpha=0.7)
    ax.annotate(f'{ATT_REF_ARCSEC:.0f}\u2033 \u2014 edge of the 60\u2033 science band',
                xy=(0.995, ATT_REF_ARCSEC), xycoords=('axes fraction', 'data'),
                ha='right', va='bottom', fontsize=9, color=INK_2)

    ax.set_xlabel('minutes since first frame of the visit', color=INK_2, fontsize=11)
    ax.set_ylabel('Mars centre to boresight (arcsec)', color=INK_2, fontsize=11)
    ax.set_title('CUTE Mars campaign \u2014 pointing offset',
                 color=INK, fontsize=14, loc='left')
    # Reserve headroom so the legend never lands on the data.
    ymax = float(np.nanmax(out['offset_arcsec']))
    ax.set_ylim(0, max(ymax * 1.45, ATT_REF_ARCSEC * 1.12))
    ax.legend(frameon=False, ncol=min(len(visits), 4), fontsize=9.5,
              labelcolor=INK_2, loc='upper left')
    if EXCLUDE_FRMIDS:
        fig.text(0.01, 0.005, 'excluded frmid: '
                 + ', '.join(str(f) for f in EXCLUDE_FRMIDS),
                 fontsize=8.5, color=INK_2)
    fig.savefig(dest, dpi=PNG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return dest


def run_attitude():
    print(f'MODE={MODE!r}  attitude-error analysis  (FOV={ATT_FOV})')
    t0 = time.time()
    out, missing = measure_attitude()
    summary = summarise(out)

    csv_path = output_dir / f'{ATT_CSV_NAME}.csv'
    sum_path = output_dir / f'{ATT_CSV_NAME}_summary.csv'
    png_path = output_dir / f'{ATT_CSV_NAME}.png'
    out.to_csv(csv_path, index=False)
    summary.to_csv(sum_path, index_label='visit')
    plot_attitude(out, summary, png_path)

    print()
    print(summary[['n_frames', 'duration_min', 'mean_offset_arcsec',
                   'std_offset_arcsec', 'max_offset_arcsec']].round(2).to_string())
    print(f'\n{len(out)} frame(s) measured in {time.time() - t0:.1f}s')
    for pth in (csv_path, sum_path, png_path):
        print('  ', pth)


# --------------------------------- Batch run -------------------------------
def collect():
    """Discovered FITS joined to their pointing row, grouped by (visit, fov)."""
    groups = defaultdict(list)
    missing = []
    for entry in discover_fits():
        row = pointing.get(entry['frmid'])
        if row is None:
            missing.append(entry['path'].name)
            continue
        visit = row['visit']
        if VISITS is not None and visit not in VISITS:
            continue
        if FRMIDS is not None and entry['frmid'] not in FRMIDS:
            continue
        if EXCLUDE_FRMIDS and entry['frmid'] in EXCLUDE_FRMIDS:
            continue
        if FOVS is not None and entry['fov'] not in FOVS:
            continue
        if entry['folder'] and entry['folder'] != visit:
            print(f"  note: {entry['path'].name} sits in {entry['folder']} "
                  f"but the CSV says {visit}; using {visit}")
        groups[(visit, entry['fov'])].append((entry, row))
    for key in groups:                       # ascending frmid within a group
        groups[key].sort(key=lambda pair: pair[0]['frmid'])
    return groups, missing


def run():
    print(f'MODE={MODE!r}  (SAVE_PNG={SAVE_PNG}, MAKE_GIF={MAKE_GIF})  SHOW={SHOW}  '
          f'backend={matplotlib.get_backend()}  Pillow={Image.__version__}')

    groups, missing = collect()
    if not groups:
        raise SystemExit(f'no usable FITS found under {fits_dir}')

    total = sum(len(v) for v in groups.values())
    print(f'{len(groups)} visit/FOV group(s), {total} frame(s) to render\n')

    done = written = rendered = skipped = errors = gifs = 0
    t0 = time.time()

    # One group at a time, and its GIF is written the moment the group ends --
    # nothing waits for the whole run to finish.
    for (visit, fov) in sorted(groups, key=lambda k: (visit_key(k[0]), k[1])):
        items = groups[(visit, fov)]
        gif_dest = output_dir / visit / f'animation_{visit}_{fov}.gif'
        if MAKE_GIF and not SAVE_PNG and not OVERWRITE and gif_dest.exists():
            print(f'{visit} [{fov}]: animation exists, skipping {len(items)} frame(s)')
            done += len(items)
            skipped += len(items)
            continue

        frames = []
        for entry, row in items:
            done += 1
            tag = f"[{done}/{total}] {visit} frmid {entry['frmid']} [{fov}]"
            try:
                png, frame = plot_frame(entry, row)
            except Exception:
                errors += 1
                print(f'{tag}: ERROR')
                traceback.print_exc()
                continue
            if frame is not None:
                frames.append((entry['frmid'], frame))
            if png is not None:
                written += 1
                print(f'{tag}: {png.name}')
            elif frame is not None:
                rendered += 1                # GIF-only mode: no PNG by design
                print(f'{tag}: rendered')
            else:
                skipped += 1                 # output already existed
                print(f'{tag}: skipped')

        if MAKE_GIF and frames:
            if write_gif(visit, fov, frames):
                gifs += 1
            frames.clear()                   # release before the next group

    dt = time.time() - t0
    print(f'\n{written} PNG(s), {rendered} GIF frame(s), {gifs} GIF(s), '
          f'{skipped} skipped, {errors} error(s), '
          f'{len(missing)} FITS with no CSV row   [{dt:.1f}s, {dt/max(done,1):.2f}s/frame]')
    for n in missing[:10]:
        print('   no pointing for', n)
    print('output root:', output_dir)


# Called unconditionally, not under `if __name__ == '__main__'`: some IDE run
# modes (Spyder cell/selection execution, PyCharm console) do not set __name__
# to '__main__', and the guard then silently does nothing.
if MODE == 'attitude':
    run_attitude()
else:
    run()