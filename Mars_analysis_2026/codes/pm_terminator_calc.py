import planetmapper
import matplotlib.pyplot as plt

body = planetmapper.Body('mars', '2025-01-19 5:09:26', observer='earth', observer_frame='J2000')
ra, dec = body.terminator_radec(npts=360) # returns tuple (ra, dec)
ax = body.plot_wireframe_radec()

ax.plot(ra, dec, color='r', linestyle='-', linewidth=2, label='term')
ax.legend()
plt.show()