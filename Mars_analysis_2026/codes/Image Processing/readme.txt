Data Processing files for NASA CUTE
Author(s) = { 
    Dolon Bhattacharyya : dobh6980,
    Joshua Sendgikoski  : jose5987
    }

-----Legacy Processing of single data file-----
                TO BE REMOVED
cute_mars1: @Author: dobh6980
cute_marsj: @Author: jose5987 # Changed comments, otherwise copy of cute_mars1

-----Multi-scale processing of full visits-----
cute_mars2025: @Author: jose5987

-----Driver for cute_mars2025-----
cute_mars2025_driver: @Author: jose5987

Image Processing logic separated into two classes in cute_mars2025:
    CuteReference   : Shared calibration data for every visit
    CuteObservation : Individual processing for each visit spectra, with output

To utilize:
    Navigate to cute_mars2025_driver.py
    Enter described inputs into STATIC, VISIT, and FILENAME.
    Run file.
    Get excited.