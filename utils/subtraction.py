"""
Subtraction logicfor main.py that is able to applied to ALL planets
in the CUTE solar system observation campaign.

@author: jose5987
"""

#from pathlib import Path

import numpy as np
from astropy.io import fits 
from astropy.modeling import fitting, models  # noqa: F401


class Subtraction:

    def __init__(self, output_dir:str = None, obs_dir:str = None, wv_soln = None,
                 eff_area_dir:str = None, planet:str = None):
        self.planet       = planet          # selected planet
        self.output_dir   = output_dir      # output directory for CSV and images
        self.obs_dir      = obs_dir         # observation directory
        self.eff_area_dir = eff_area_dir    # effective area directory

        self.eff_area = self._get_effective_area()

    @classmethod
    def _get_effective_area(self):
        path = self.eff_area_dir

        eff_area  = []
        wave_val  = []
        data      = open(path, 'r')
        lines     = data.readlines()

        count:int = 0
        for i in lines:
            if count > 1:
                li = i.strip()
                st = i.split()
                ww = float(st[1])
                ee = float(st[0])
                eff_area.append(ee)
                eff_area.append(ee)
                wave_val.append(ww)
            count = count + 1

        return np.interp(self.wv_soln, wave_val, eff_area)