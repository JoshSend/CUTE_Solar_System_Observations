"""
Prints select headers from CUTE FITS files
 
@author: jose5987
"""
 
import os
import glob
from astropy.io import fits
 
HERE = os.path.dirname(os.path.abspath(__file__))
 
 
def visit_num(path):
    num = os.path.basename(path).replace('Visit', '')
    return int(num) if num.isdigit() else 10**6
 
 
def main():
    vdirs = sorted(
        (d for d in glob.glob(os.path.join(HERE, 'Visit*')) if os.path.isdir(d)),
        key=visit_num,
    )
 
    print("Visit#\tframeid\tdate (UTC)\tExposure Time (ms)")          # header row
    for vdir in vdirs:
        vnum = visit_num(vdir)
        rows = []
        for f in glob.glob(os.path.join(vdir, '*.fits')):
            try:
                h = fits.getheader(f)
            except Exception:
                continue
            rows.append((h.get('FRM_ID', ''), h.get('DATE-OBS', ''), h.get('EXPTIME', '')))
        # sort by frame id within the visit
        rows.sort(key=lambda r: r[0] if isinstance(r[0], int) else 0)
        for frmid, date_utc, exptime in rows:
            print(f"{vnum}\t{frmid}\t{date_utc}\t\t{exptime}")
 
 
if __name__ == '__main__':
    main()