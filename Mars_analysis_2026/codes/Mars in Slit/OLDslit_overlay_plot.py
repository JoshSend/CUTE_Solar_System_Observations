# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:42:26 2022
@author: amsu4591
 
Modified for CUTE Mars Campaign in September 2026
@author: jose5987
 
Fixes in this version
---------------------
1. One frame -> one Polygon.  The old code fed whole CSV columns into the
   vertex list, so `coord` came out (14, 2, 104) and Polygon.set_xy raised
   "too many values to unpack".  A Polygon is a single closed shape at a
   single sky position, so ra/dec/roll must be scalars.  Multiple frames are
   handled by looping and making a new Polygon each time.
2. RA offsets divided by cos(dec).  Without it the slit is stretched by
   1/cos(22.6 deg) = 8.3%, ~11.7" at the far end of the slit.
3. imshow(..., origin='lower').  With projection=wcs the matplotlib default
   'upper' flips the array against its own WCS and the overlay is wrong.
4. get_transform('world') instead of 'fk5'.  These headers say RADESYS='ICRS'.
5. Output filenames have ':' stripped out of date_obs -- illegal on Windows.
 
Batch behaviour
---------------
The run is driven by the FITS files on disk, not by the CSV: every
Mars_Fits\\Visit*\\frmid<id>_<fov>.fits is discovered, its frmid is looked up
in mars_pointing.csv for the pointing, and the figure is written to
Slit_Overlay\\<Visit>\\.  Visit folders are created as needed.  A FITS with no
matching CSV row is reported and skipped; a CSV row with no FITS is simply
never reached.
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
MODE = 'gif'
 
SHOW = False         # pop up each figure.  Leave False for a batch run --
                     # plt.show() on ~200 figures will stall the session.
OVERWRITE = True     # False skips work whose output already exists: in 'png'
                     # mode a frame whose PNG is present, in 'gif' mode an
                     # entire visit whose animation is present (so an
                     # interrupted run resumes without re-rendering).
 
SAVE_PNG = MODE in ('png', 'both')
MAKE_GIF = MODE in ('gif', 'both')
if MODE not in ('gif', 'png', 'both'):
    raise SystemExit(f"MODE must be 'gif', 'png' or 'both', not {MODE!r}")
 
# Filters.  None = no filter, i.e. everything found on disk.
VISITS = None        # e.g. ['Visit1', 'Visit2']
FRMIDS = None        # e.g. [3726, 3727]
FOVS = None          # e.g. ['5x5MR'];  None plots every FOV present
 
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
    amsu4591/jose5987 one, which puts the slit long axis (+py) at position
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
    ax.imshow(data, origin='lower', cmap='inferno')   # origin matters
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_autoscale_on(False)
 
    coord = slit_polygon(ra, dec, theta)
    ax.add_patch(Polygon(coord, closed=True, edgecolor='k', facecolor='none', lw=1.5,
                         transform=ax.get_transform('world')))
 
    # boresight
    ax.plot(ra, dec, '+', color='deepskyblue', ms=14, mew=2,
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
run()