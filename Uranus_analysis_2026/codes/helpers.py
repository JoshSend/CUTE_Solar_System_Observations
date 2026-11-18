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
        self.base_dir = base_dir or self._get_default_dir()
        self.obs_dir  = self._get_observations_dir()
        self.proj_dir = self._get_project_dir()

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
        base = self._get_default_dir()
        return os.path.abspath(os.path.join(base, '..', 'CUTE_observations'))

    def _get_project_dir(self)-> str:
        """
        Get the default project directory
        """
        base = self._get_default_dir()
        return os.path.abspath(os.path.join(base, '..', '..'))

