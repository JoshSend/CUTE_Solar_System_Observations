"""
Rewrite of CUTE Mars observation processing program
original author: dobh6980
author: jose5987
"""
# ==============================================
# Imports
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
#import scipy
#import pdb

# ==============================================
"""
All input files are read from the folder this script lives in,
so the whole folder can be copied to any machine and run without
editng file paths.
"""
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# ==============================================
"""
Get name of specific CUTE Mars fits file and open
"""

fname = 'cute_TRIM2D_scan_targetID340_2025_02_03_03_58_frmid_5029_V2_nimgpkts_446_L1id19767_botrows_24_midrows_56.fits'
dir_name = base_dir

ff  = fits.open(os.path.join(dir_name,fname),memap=False)

# Flip left-right so the column # increases with wavelength
# (as in the Alpha Cen code)
img = np.fliplr(ff[0].data)

# 2200 columns = 2048 science pixels + overscan on both ends
nx  = img.shape[1]

# ==============================================

def smooth(y, box_pts) -> NDArray:
    """
    Apply smoothing to the spectra
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')

    return y_smooth

# ==============================================
"""
The raw detector runs from long to short wavelengths, so every x-value
below is mirrored onto the flipped image with x_flipped = nx-1-x_raw.

Raw science columns 52-2099 therefore become flipped columns 100-2147.

WIDTH OF SPECTRA REGION
"""

"""
Get the region of image with science spectra
"""
y1_sc = 37.-1
y2_sc = 69.-1
x1_sc = nx-1-(52.-1)
x2_sc = nx-1-(2099.-1)
m1_sc = (y2_sc - y1_sc) / (x2_sc-x1_sc)

y3_sc = 59.-1
y4_sc = 88.-1
x3_sc = nx-1-(52.-1)
x4_sc = nx-1-(2099.-1)
m2_sc = (y4_sc - y3_sc) / (x4_sc - x3_sc)

"""
Get the region of image with background
"""
y1_dk = 7.-1
y2_dk = 39.-1
x1_dk = nx-1-(52.-1)
x2_dk = nx-1-(2099.-1)
m1_dk = (y2_dk - y1_dk) / (x2_dk - x1_dk)

y3_dk = 29.-1
y4_dk = 58.-1
x3_dk = nx-1-(52.-1)
x4_dk = nx-1-(2099.-1)
m2_dk = (y4_dk - y3_dk) / (x4_dk - x3_dk)

"""
Get the boundaries for the spectral region containing Mars spectra
(point-slope form, so the mirrored trace is bit-identical to the one
on the unflipped image)
"""

xval = np.arange(
    start=nx-2100, stop=nx-52, step=1, dtype=int
)
# flipped columns 100-2147, short to long wavelengths

yval1_sc = y1_dk + m1_sc*(xval - x1_dk)
yval2_sc = y3_sc + m2_sc*(xval - x3_sc)

"""
Get boundaries of the dark region just below spectral region
to get the background
"""
yval1_dk = y1_dk + m1_dk*(xval - x1_dk)
yval2_dk = y3_dk = m2_dk*(xval - x3_dk)

# ==============================================
"""
Plot boundary regions to check
be careful with width of spectra region
"""
title = "Mars"
plt.plot(title)
plt.plot(xval, yval1_sc, linewidth=3, color='red')
plt.plot(xval, yval2_sc, linewidth=3, color='red')
plt.plot(xval, yval1_dk, linewidth=3, color='yellow')
plt.plot(xval, yval2_dk, linewidth=3, color='yellow')
plt.imshow(
    img, vmin=0, vmax=18000, origin='lower',
    aspect='auto', interpolation=None
)
plt.colorbar()
plt.show()
#plt.close()

# ==============================================
# Array to contain dark counts
dk_arr = np.zeros(shape=2048, dtype=float)

# Array to contain the Mars spectral counts
sc_arr = np.zeros(shape=2048, dtype=float)

# Start with flipped column 100 (raw pixel 2099, the blue end)
# to avoid overscan region
k = nx-2100

"""
Loop over each x value to add up the counts in between the boundary
region along the y-axis direction.
Assigning each pixel a wavelength
Marking width + adding vertically + converting to 1D spectra
Make a plot of 1D count on the y-axis and pixel number on x-axis
"""
# End at the 2048 pixel to avoid overscan region
for i in range(2048):
# dark region boundaries for the ith x-value
    yy1        = int(yval1_dk[i])  # Bottom boundary
    yy2        = int(yval2_dk[i])  # Top boundary

# Get pixel count array within top and bottom boundary limits
    dkval      = img[yy1: yy2, k]  
    dk_arr[1]  = np.median(dkval)  # Get the median of the count array

# Science region bottom boundary for the ith x-value
    yy3        = int(yval1_sc[i])  
    yy4        = int(yval2_sc[i])

# Get the pixel count array within the top and bottom boundary limits
# and subtract dark
    scval      = img[yy3: yy4, k] - dk_arr[i] 
    sc_arr[1]  = np.sum(scval)     # Sum up all science counts
    k          = k + 1    

# Already runs short to long wavelengths (the image was flipped),
# so no inversion needed here
spectra = sc_arr

# ==============================================
"""
Read in the wavelength solution for CUTE
Convert pixel to wavelength
"""
# How we know what wavelength corresponds to what row
fname = os.path.join(base_dir, 'flight_quad_wavelength_solution_final.dat')

wave_sol = []
f     = open(fname,'r')
lines = f.readlines()
count = 0

for j in lines:
    if count > 0:
        li = j.strip()
        st = j.split()
        ww = float(st[0])
        wave_sol = np.append(wave_sol, ww)
    count = count + 1

wave_sol = np.array(wave_sol)

# ==============================================
"""
Read in the effective area
"""
fname = os.path.join(base_dir, 'cute_recalculated_effa_2025.txt')
eff_area = []
wv_val   = []

f     = open(fname,'r')
lines = f.readlines()
count = 0

for j in lines:
    if count > 1:
        li = j.strip()
        st = j.split()
        ww = float(st[1])
        ee = float(st[0])
        eff_area = np.append(eff_area, ee)
        wv_val   = np.append(wv_val, ww)

eff_area = np.array(eff_area)
wv_val   = np.array(wv_val)

eff_area_mod = np.interp(
    wave_sol, wv_val, eff_area
)

"""
New figure, otherwise plt.close() below kills the image plotted above
"""
plt.figure()
plt.plot(wave_sol, eff_area_mod)
plt.close()

"""
Convert the spectra from DN to electrons / DN
DN = Digital Numbers
Pipeline Digital numbers -> Photons
"""
h = 6.626075540e-27 # ergs * s
c = 2.99792458e+10  # cm/s

# Converts the counts to electrons with 1.5 as the instrument gain
mars_spec_e = spectra*(1.5 / (100.))

# Convert the electron counts to photons
mars_spec = mars_spec_e*((h*c) / ((wave_sol*1E-8)*eff_area_mod))

# ==============================================
"""
Plotting
"""
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.grid(color='gray', linestyle='dashed')

title = "Mars NUV Spectra with CUTE"
ax.set_title(title, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)

ax.set_xlabel('Wavelength ($\AA$)', fontsize=15)
ax.set_ylabel('Flux (10$^{-9}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
ax.set_xlim(2500, 3250)
ax.set_ylim(0., 2.)

ax.plot(
    wave_sol, smooth(mars_spec, 15)*1e9, color='darkred', lw=2
)
figname = os.path.join(base_dir, 'Visit5_eg.jpeg')
plt.savefig(figname, format='jpeg', dpi=200)