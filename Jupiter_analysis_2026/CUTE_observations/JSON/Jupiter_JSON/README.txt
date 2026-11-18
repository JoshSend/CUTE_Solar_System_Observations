COMMAND PLANS - Jupiter system (CUTE targetID 345)
==============================================================================

Copied from GUI/JSON/Gamma_Lep_Jupiter_zetapup/. These are the plans that
commanded the science frames in this folder.

  Visit1  <- CUTE_Plan_01.25.2026.json   frames 1054-1057,  30 s
  Visit4  <- CUTE_Plan_02.18.2026.json   frames 1137-1140, 350 s
  Visit7  <- CUTE_Plan_03.10.2026.json   frames 1194-1203, 350 s

Five further plans in that folder (02.01, 02.08, 02.25, 03.04, 03.18) command
no targetID 345 at all and are not relevant here.

VERIFICATION
------------------------------------------------------------------------------
Each commanded exposure was compared with the FITS DATE-OBS on disk. A single
PLD_CCD_EXPOSE command with NUM=n starts a sequence, so frame i of a block
falls at t0 + i*cadence.

  Visit1   30 s exposure, cadence  95 s (65 s overhead)   TRIM2D only
  Visit4  350 s exposure, cadence 546 s (196 s overhead)  TRIM2D + NOPROC
  Visit7  350 s exposure, cadence 546 s (196 s overhead)  NOPROC only

Every block-start frame matches its commanded time to within 1 s, and every
frame falls within 2 s of the measured cadence. The larger overhead on Visit4
and Visit7 tracks the NOPROC downlink, which is a much bigger product than
TRIM2D; it is not a timing discrepancy.
