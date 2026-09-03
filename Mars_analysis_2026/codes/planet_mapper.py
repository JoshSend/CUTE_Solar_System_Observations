"""
Planetmapper wireframe images of Mars.
@author: jose5987

Inputs: date and time in UTC
Outputs: two wireframe images of Mars at specified UTC. Two FOVs, 23x23 arcsec and 5x5 Mars Radii
Dependencies: Planetmapper (pip install) and relevant Mars SpicePy kernels.
"""

import matplotlib.pyplot as plt
import planetmapper

# YYYY-MM-DDTHH:MM:SS
UTC = '2025-01-12T04:32:26'                       # your CUTE observation time
body = planetmapper.Body('mars', UTC, observer='earth')

# Mars angular radius in arcsec at this distance
mars_radius_arcsec = body.target_diameter_arcsec / 2

def zoom(ax, half):
    """Center the view on Mars with a half-window of `half` arcsec,
    preserving PlanetMapper's sky (east-left) x-axis direction."""
    xlo, xhi = ax.get_xlim()
    ax.set_xlim((half, -half) if xhi < xlo else (-half, half))
    ax.set_ylim(-half, half)
    ax.set_aspect('equal')
    ax.axis('off')

# --- Image 1: 5 x 5 Mars radii FOV -------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 6))
body.plot_wireframe_angular(
    ax=ax1, 
    formatting={'terminator': {'color': 'red', 'linestyle': '--', 'linewidth': 1.5}}
)

zoom(ax1, 2.5 * mars_radius_arcsec)               # 5 R_Mars across
ax1.set_title(f'')

# --- Image 2: 23 x 23 arcsec FOV ---------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 6))
body.plot_wireframe_angular(
    ax=ax2,
    formatting={'terminator': {'color': 'red', 'linestyle':'--','linewidth':1.5}}
)
zoom(ax2, 23 / 2)                                 # 23 arcsec across
ax2.set_title('')

plt.show()