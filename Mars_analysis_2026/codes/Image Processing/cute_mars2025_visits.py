'''
Visit analysis for NASA CUTE (Colorado Ultraviolet Transit Experiment)
utilizing image processing logic from cute_mars2025.py
@Author: jose5987
Date Created: 8/20/2026
'''

from cute_mars2025 import FileLoader

fname = 'cute_TRIM2D_scan_targetID340_2025_02_03_03_58_frmid_5029_V2_nimgpkts_446_L1id19767_botrows_24_midrows_56.fits'

#wv_fname = 'flight_quad_wavelength_solution_final.dat'

visit1 = FileLoader(fname)
visit1.get_dir()

print(visit1.dir_name)
print(visit1.wv_fname)