"""
testing astropy and specutils

Created: 1st October, 2026
@author: jose5987
"""

from pathlib import Path

from specutils import Spectrum

class CuteObservations:
    def __init__(self):
        pass

    def _get_base_dir(self):
        try:
            base_dir = Path(__file__).parent
        except NameError:
            base_dir = Path.cwd
        return base_dir

def main():
    pass

if __name__ == '__main__':
    main()