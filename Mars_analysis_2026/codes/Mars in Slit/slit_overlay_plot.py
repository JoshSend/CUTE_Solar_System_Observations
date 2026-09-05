
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:42:26 2022
@author: amsu4591

Modified for CUTE Mars Campaign in September 2026
@author: jose5987
"""
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.visualization.wcsaxes import Quadrangle, SphericalCircle
from matplotlib.patches import Rectangle, Polygon
from matplotlib.patches import Circle
import matplotlib as mpl
import numpy as np
from pathlib import Path
import pandas as pd

#-------Directories----

base_dir = Path(__file__).resolve().parent   # '...\codes\Mars In Slit\'
project_root = base_dir.parent.parent        # ...\mars_analysiis_2026\
codes_dir = project_root / 'codes'           # '...\codes\'
out_dir = codes_dir / 'output'               # ...\codes\output\

csv_dir = out_dir / 'mars_pointing.csv'
fits_dir = out_dir / 'Mars_Fits'

# Output Directory
output_dir = out_dir / 'Slit_Overlay'

# # File format for named overlays
# # overlay_Visit*_frmid_date_obs_ra_dec_roll.png

#-------JOSH has to modify this portion to read in an array instead of single numbers for one frame----

df = pd.read_csv(csv_dir)     # Establish pandas dataframe as df

ra  = df['ra (deg)']          # degrees --- This is the Right Ascension angle
dec = df['dec (deg)']         # degrees --- This is the Declination angle
theta = df['roll (deg)']      # degrees --- This is the Roll angle 

px0=[-60.0,60.0,60.0,30.0,30.0,15.0,15.0,-15.0,-15.0,-30.0,-30.0,-60.0,-60.0]
py0=[-591.807,-591.807,-351.805,-351.805,351.805,351.805,680.817,680.817,351.805,
     351.805,-351.805,-351.805,-591.807]
coord = []
coord_x = []
coord_y = []

for i in range(0, 13):
 coord.append([ra + px0[i]/3600*np.cos(np.deg2rad(theta)) - py0[i]/3600*np.sin(np.deg2rad(theta)), 
              dec + py0[i]/3600*np.cos(np.deg2rad(theta)) + px0[i]/3600*np.sin(np.deg2rad(theta))])
 coord_x.append(ra + px0[i]/3600*np.cos(np.deg2rad(theta)) - py0[i]/3600*np.sin(np.deg2rad(theta)))
 coord_y.append(dec + py0[i]/3600*np.cos(np.deg2rad(theta)) + px0[i]/3600*np.sin(np.deg2rad(theta)))
 
 
coord.append(coord[0]) #repeat the first point to create a 'closed loop'

#----JOSH has to modify this section to read in the Mars 2D image file
hdu = fits.open(fits_dir / 'Visit1' / 'frmid3726_5x5MR.fits')[0]
wcs = WCS(hdu.header)
plt.figure(210, (19.5, 8.25))
# print(wcs)
ax = plt.subplot(projection=wcs)


ax.imshow(hdu.data)#,extent=[120.85*u.deg,120.95*u.deg,-40.05*u.deg,-39.95*u.deg])
#ax.grid(color='white', ls='solid')
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.set_autoscale_on(False)
      
p = Polygon(coord, edgecolor = 'k', facecolor='none', transform=ax.get_transform('fk5'))
ax.add_patch(p)

theta = 90
px0=[-60.0,60.0,60.0,30.0,30.0,15.0,15.0,-15.0,-15.0,-30.0,-30.0,-60.0,-60.0]
py0=[-591.807,-591.807,-351.805,-351.805,351.805,351.805,680.817,680.817,351.805,
     351.805,-351.805,-351.805,-591.807]
coord = []
coord_x = []
coord_y = []
for i in range(0, 13):
 coord.append([ra + px0[i]/3600*np.cos(np.deg2rad(theta)) - py0[i]/3600*np.sin(np.deg2rad(theta)), 
              dec + py0[i]/3600*np.cos(np.deg2rad(theta)) + px0[i]/3600*np.sin(np.deg2rad(theta))])
 coord_x.append(ra + px0[i]/3600*np.cos(np.deg2rad(theta)) - py0[i]/3600*np.sin(np.deg2rad(theta)))
 coord_y.append(dec + py0[i]/3600*np.cos(np.deg2rad(theta)) + px0[i]/3600*np.sin(np.deg2rad(theta)))
coord.append(coord[0]) #repeat the first point to create a 'closed loop'

p = Polygon(coord, edgecolor = 'k', facecolor='none', transform=ax.get_transform('fk5'))
#ax.add_patch(p)
ax.show()

