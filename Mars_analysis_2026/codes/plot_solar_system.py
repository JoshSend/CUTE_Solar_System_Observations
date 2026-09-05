"""
Planetary position generator at specific UTC time.
Plots a top-down heliocentric 2D map of the solar system.
Current dictionary is of the first exposure for each visit.

Dependencies: astropy, numpy. matplotlib, sunpy

@author: jose5987
"""

from pathlib import Path
from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / 'output' / 'Solar_System'

SAVE: bool = False

# UTC for the first exposure from each visit
UTC = {
    'Visit1_frmid_3726': '2024-12-22 3:49:26',
    'Visit2_frmid_4860': '2025-01-12 4:32:26',
    'Visit3_frmid_4923': '2025-01-19 5:09:26',
    'Visit4_frmid_4973': '2025-01-27 3:39:46',
    'Visit5_frmid_5029': '2025-02-03 3:58:41',
    'Visit7_frmid_5143': '2025-02-17 4:41:11',
    'Visit8_frmid_5209': '2025-03-02 5:45:26',
    'Visit9_frmid_5253': '2025-03-30 3:13:01'
}

planet_list = ['sun', 'mercury', 'venus', 'earth', 'mars',]
planet_colors = {
    'sun': 'gold',
    'mercury': 'brown',
    'venus': 'green',
    'earth': 'mediumblue',
    'mars': 'firebrick',
    'jupiter': 'red'
}

def plot_system(label, utc_string, planet_list, planet_colors, save=SAVE):
    UTC = Time(utc_string)
    planet_coord = [get_body_heliographic_stonyhurst(p, time=UTC)
                    for p in planet_list]

    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(1, 1, 1, projection='polar')

    r_max = max(c.radius.value for c in planet_coord)
    r_pad = 0.1 * r_max

    for this_planet, this_coord in zip(planet_list, planet_coord):
        theta = np.deg2rad(this_coord.lon.value)
        r = this_coord.radius.value

        ax1.plot(theta, r, 'o', color=planet_colors[this_planet])

        if this_planet != 'sun':
            ax1.text(theta, r + r_pad, this_planet.capitalize(),
                     color=planet_colors[this_planet],
                     ha='center', va='center',
                     fontsize=9, fontweight='bold')

    label_strip = label.replace('_', ' ')
    ax1.set_title('{}\n{}'.format(label_strip, UTC))

    if save:
        fig.savefig(output_dir / '{}.png'.format(label), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig

for label, utc_string in UTC.items():
    plot_system(label, utc_string, planet_list, planet_colors)

plt.show()