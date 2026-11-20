"""
Driver for subtraction logic that can be applied to all visits for all planets.

@author: jose5987
"""

from pathlib import Path

utils_dir = Path(__file__).parent / 'utils'    # directory for main utilities folder
from utils.subtraction import (
    Subtraction,  # import Subtraction class from subtraction.py
)

Planet:str = "Mars"         # Mars | Jupiter | Uranus
MODE:str   = "subtract"     # subtract | albedo; computes subtraction or albedo
SAVE:bool  = False         # True: writes to output; False: Show on screen

def main():

    # ------ directories ------
    wv_soln_dir = utils_dir / 'flight_quad_wavelength_solution_final.dat'   # wavelength solution directory
    root_dir    = Path(__file__).parent
    base_dir    = root_dir / f'{Planet.capitalize()}_analysis_2026'

    if MODE.lower() == 'subtract':
        if Planet.lower() == 'jupiter':
            eff_area_dir = utils_dir / 'cute_recalculated_effa_2026.txt'
        else:
            eff_area_dir = utils_dir / 'cute_recalculated_effa_2025.txt'

        # Initialize subtraction class
        sub = Subtraction(
            planet       = Planet.lower(),
            obs_dir      = base_dir / 'CUTE_observations',
            output_dir   = base_dir / 'outputs',
            wv_soln_dir  = wv_soln_dir,
            eff_area_dir = eff_area_dir,
            save         = SAVE,
        )
        sub.run_spectra() # computes 1D spectra; plots + generates CSVs

    elif MODE.lower() == 'albedo':
        print("That's super not finished yet.")

    else:
        print("that's not finished yet.")

if __name__ == "__main__":
    main()