"""
Converts PDF images to FITS files

@author: jose5987
"""

from astropy.io import fits
from astropy import units as u
import numpy as np
import pymupdf

# Convert Mars PDF file to a fits image
pdf_path = r"C:\Users\mageb\OneDrive\Documents\CUTE_Solar_System_Observations\Mars_analysis_2026\codes\output\Mars_Images\Visit2\frmid4860.pdf"
fits_path = r"C:\Users\mageb\OneDrive\Documents\CUTE_Solar_System_Observations\Mars_analysis_2026\codes\output\Mars_Images\Visit2\frmid4860.fits"

def pdf_to_fits(pdf_path, fits_path=None, page=0, dpi=300):
     # Automatically generate an output .fits filename if none is provided
     if fits_path is None:
          fits_path = pdf_path.rsplit(".", 1)[0] + ".fits"

     doc = pymupdf.open(pdf_path)
     pg = doc[page]
     # render the page straight to a single-channel (grayscale) raster
     pix = pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
     gray = np.frombuffer(pix.samples, dtype=np.uint8) \
               .reshape(pix.height, pix.width).astype(np.float32)
     # FITS/imshow(origin='lower') counts rows from the bottom; image files
     # count from the top, so flip so it displays right-side up.
     gray = np.flipud(gray)
     fits.PrimaryHDU(data=gray).writeto(fits_path, overwrite=True)
     print(f"wrote {fits_path}  shape={gray.shape}  dtype={gray.dtype}")
     return gray

def main():
     pdf_to_fits(pdf_path=pdf_path, fits_path=fits_path)

if __name__ == '__main__':
     main()