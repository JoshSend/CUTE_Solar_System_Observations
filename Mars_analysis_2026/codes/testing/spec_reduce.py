"""
testing astropy and specutils

Created: 1st October, 2026
@author: jose5987
"""

from pathlib import Path

from specutils import Spectrum

class CuteObservations:
    def __init__(self):
        self.base_dir = self._get_base_dir()

    def _get_base_dir(self):
        # Acquire base directory for file
        try:
            base_dir = Path(__file__).parent
        except NameError:
            base_dir = Path.cwd
        return base_dir

def main():
    pass

if __name__ == '__main__':
    main()