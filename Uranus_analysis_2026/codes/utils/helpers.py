"""
Helper functions that are universal for Uranus Observations

@author: jose5987
"""

import os

class Directories:
    """
    Grab all relevant directories
    """

    def __init__(self, base_dir:str=None):
        self.base_dir = base_dir or self._get_default_dir() # base
        self.obs_dir  = self._get_observations_dir()        # observations
        self.proj_dir = self._get_project_dir()             # project
        self.out_dir  = self._get_output_dir()              # outputs

    def _get_default_dir(self)-> str:
        """
        Get the folder this script lives in
        """
        try: 
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        return base_dir
    
    def _get_observations_dir(self)-> str:
        """
        Obtain all Uranus Observations
        they live in C: ".../Uranus_analysis_2026/CUTE_observations"
        """
        return os.path.abspath(os.path.join(self.base_dir, '..', '..', 'CUTE_observations'))

    def _get_project_dir(self)-> str:
        """
        Get the default project directory
        """
        return os.path.abspath(os.path.join(self.base_dir, '..', '..', '..'))

    def _get_output_dir(self)-> str:
        """
        Get the base output directory
        """
        return os.path.abspath(os.path.join(self.base_dir, 'outputs'))

class Observation_info:
    """
    Grabs all relevant observation info
    """

    def __init__(self):
        self.frame_id = self._extract_frame_id
        

    def _extract_frame_id(self):
        """
        Parses the 'frmid_####' token out of
        filename strings
        """
        bare_name = os.path.basename(self.fits_fname)
        parts = bare_name.split('_')

        if "frmid" in parts:
            idx = parts.index('frmid')
            return parts[idx + 1]

        return "Unknown"
