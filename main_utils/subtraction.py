"""
Subtraction logicfor main.py that is able to applied to ALL planets
in the CUTE solar system observation campaign.

@author: jose5987
"""

from pathlib import Path

class Subtraction:

    def __init__(self, output_dir = None, obs_dir = None, wv_soln = None,
                 eff_area_dir = None):
        self.output_dir = output_dir

        self.eff_area = self._get_effective_area()

    @classmethod
    def _get_effective_area(self):
        print("effective area!")