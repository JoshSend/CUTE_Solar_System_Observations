"""
Driver for subtraction logic that can be applied to all visits.
Assumes exposure time is 100s constant
(and normalizes flux to that assumption)

@author: jose5987
"""

from pathlib import Path

utils_dir = str(Path(__file__).parent / 'utils')    # directory for main utilities folder
from utils.subtraction import (
    Subtraction,  # import Subtraction class from subtraction.py
)

# --- import from subtraction.py ---


# Select Planet to subtract spectra from:
# Mars | Jupiter | Uranus
Planet:str = "Mars"


def main():

    # assign parameters based off planet selected
    if Planet.lower() == "mars":
        obs_dir:str = str(Path(__file__).parent / # Observation Directory
                             'Mars_analysis_2026' / 'CUTE_observations')
        print("Hello!")

    elif Planet.lower() == 'jupiter':
        obs_dir:str = str(Path(__file__).parent / # Observation Directory
                             'Jupiter_analysis_2026' / 'CUTE_observations')
        #output_csv:str = ""
        print('JUPITER IS WIP')

    elif Planet.lower() == 'uranus':
        obs_dir:str = str(Path(__file__).parent / # Observation Directory
                             'Uranus_analysis_2026' / 'CUTE_observations')
        #output_csv:str = ""
        print('URANUS IS WIP')

    else:
        print('CUTE did not observe that planet :(')

    # Initialize subtraction class based off Planet parameters
    Subtraction(
        planet = Planet.lower(),
        obs_dir = obs_dir,
        # output_dir = output_dir,
        # wv_soln = wv_soln
    )

if __name__ == "__main__":
    main()