Data Processing files for NASA CUTE
Author(s){ 
    Dolon Bhattacharyya : dobh6980,
    Joshua Sendgikoski  : jose5987
    }

-----Legacy Processing of single data file-----
cute_mars1: @Author: dobh6980
cute_marsj: @Author: jose5987 # Changed comments, otherwise copy of cute_mars1

-----Multi-scale processing of full visits-----
cute_mars2025: @Author: jose5987

-----Driver for cute_mars2025-----
cute_mars2025_visits: @Author: jose5987

Image Processing logic separated into three classes in cute_mars2025:
    CuteReference  : Shared calibration data for every visit
    CuteVisit      : Individual processing for each visit spectra
    CuteMovie      : Takes a folder input of fits files and outputs a gif

To utilize:
    In cute_mars2025_visit, input relevant visit files into the
    VISIT_FILES dictionary ({label: file_dir}) and run file
Outputs:
    1D NUV Spectra Flux(Wavelength),
    Slit Figure

To utilize:
    In cute_mars2025_visit: