'''
Image Processing for the NASA CUTE (Colorado Ultraviolet Transit Experiment)
2025 Mars Observations in Near Ultraviolet
@Author: jose5987
Date Created: 8/20/2026

Based off previous processing programs written by dobh6980
'''

# Relevant Imports
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import scipy
import pdb

# ====================================================

class FileLoader():
    '''
    Grabs all necessary files

    get_dir() - returns the base directory of the file
    '''

    def __init__(self, fname):
        self.fname = fname
        self.dir_name = self._set_base_dir()

    @property
    def _set_base_dir(self):
        '''
        All input files are read from the folder this script lives in, 
        so the whole folder can be copied to any machine and run     
        without editing file paths.       
        '''
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

    def get_dir(self):
        return self.dir_name

    def get_wavelength_soln(self):
        wv_fname = os.path.join(base_dir, 'flight_quad_wavelength_solution_final.dat')
        return wv_fname