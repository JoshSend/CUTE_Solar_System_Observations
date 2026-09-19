"""
Prints headers from CUTE FITS files and outputs CUTE_mars_headers.csv
 
@author: jose5987
"""
 
import csv
import glob
import os
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

    #Collect all FITS headers and dynamically find every unique key
    records = []
    all_keys = set()

    for vdir in vdirs:
        vnum = visit_num(vdir)
        for f in glob.glob(os.path.join(vdir, '*.fits')):
            try:
                with fits.open(f) as hdul:
                    # Reads primary header; change hdul[0] to hdul[1] if headers are in extension 1
                    header = hdul[0].header
            except Exception:
                continue

            # Convert header to dictionary and exclude standard blank/comment keywords
            header_dict = {
                k: v for k, v in header.items() if k and k not in ('', 'COMMENT', 'HISTORY')
            }
            header_dict['VISIT_NUM'] = vnum
            header_dict['FILE_PATH'] = f

            records.append(header_dict)
            all_keys.update(header_dict.keys())

    if not records:
        print("No FITS files found or read successfully.")
        return

    # 2. Sort columns: Put Visit/File metadata first, then sorted header keywords
    fieldnames = ['VISIT_NUM', 'FILE_PATH'] + sorted(all_keys - {'VISIT_NUM', 'FILE_PATH'})

    # Optional: Sort rows by FRM_ID or VISIT_NUM
    records.sort(key=lambda r: (r.get('VISIT_NUM', 0), r.get('FRM_ID', 0)))

    # Write output to CSV file
    csv_filename = os.path.join(HERE, 'CUTE_mars_headers.csv')
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Successfully extracted {len(records)} records with {len(fieldnames)} headers to {csv_filename}")


if __name__ == '__main__':
    main()