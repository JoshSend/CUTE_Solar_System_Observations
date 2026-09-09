"""
Given UTC, prints Elongation, Phase Angle, and the distances Sun-Mars and Earth-Mars.

@author: jose5987
"""

from skyfield.api import load, Loader
from pathlib import Path

# Base Directory
try:
    base_dir = Path(__file__).resolve().parent          # ...\Mars_analysis_2026\codes
except NameError:
    base_dir = Path.cwd()

# Load the planetary ephemeris data
load = Loader(base_dir)
eph = load('de421.bsp')
sun, earth, mars = eph['sun'], eph['earth'], eph['mars']
ts = load.timescale()

# Chronological Start Date (UTC) timestamps extracted from your table
utc_dates = [
    (2024, 12, 22, 3, 49, 26),  # Visit 1
    (2025, 1, 12, 4, 32, 26),   # Visit 2
    (2025, 1, 19, 5, 9, 26),    # Visit 3
    (2025, 1, 27, 3, 39, 46),   # Visit 4
    (2025, 2, 3, 3, 58, 41),    # Visit 5
    (2025, 2, 17, 4, 41, 11),   # Visit 7
    (2025, 3, 2, 5, 45, 26),    # Visit 8
    (2025, 3, 30, 3, 13, 1)     # Visit 9
]

# Print header formatting
print(f"{'UTC Start Date & Time':<22} | {'Elongation':<11} | {'Phase Angle (θ)':<16} | {'Sun-Mars (AU)':<14} | {'Earth-Mars (AU)':<15}")
print("-" * 92)

# 3. Loop through dates and calculate angles/distances
for date in utc_dates:
    t = ts.utc(*date)
    
    # Observe bodies from Earth's center framework
    mars_from_earth = earth.at(t).observe(mars).apparent()
    sun_from_earth = earth.at(t).observe(sun).apparent()
    
    # A. Calculate Solar Elongation (Angle at vertex E)
    elongation = sun_from_earth.separation_from(mars_from_earth).degrees
    
    # B. Calculate Phase Angle Theta (Angle at vertex M)
    theta_phase = earth.at(t).observe(mars).phase_angle(sun).degrees
    
    # C. Calculate physical distances in AU
    sun_mars_dist = sun.at(t).observe(mars).distance().au
    earth_mars_dist = earth.at(t).observe(mars).distance().au
    
    # Format date display string
    date_str = f"{date[0]}-{date[1]:02d}-{date[2]:02d} {date[3]:02d}:{date[4]:02d}:{date[5]:02d}"
    
    print(f"{date_str:<22} | {elongation:>9.2f}° | {theta_phase:>14.2f}° | {sun_mars_dist:>12.4f} | {earth_mars_dist:>14.4f}")
