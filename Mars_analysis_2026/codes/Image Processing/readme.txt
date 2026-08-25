Data Processing files for NASA CUTE (Colorado Ultraviolet Transit Experiment)
Github: https://github.com/JoshSend/CUTE_Solar_System_Observations
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

-----------------------------------
To utilize:
    Navitage to cute_mars2025_driver.py 
    User Inputs:
        MODE         : str          # 'static', 'visit', 'grid', 'sequence'
        VISIT        : str          # Name of visit folder
        FILENAME     : str or int   # File name OR specific frame id
        GRID_VISITS  : list[str]    # Visit folders to skip in animations
        SKIP_FRMID   : list[int]    # FrameIDs to skip in animations
        SAVE         : bool         # Options for saving output to output_dir
        output_dir   : str          # Directory for output
                                    (shouldn't need to change if entire folder is ported. i.e. from Github)