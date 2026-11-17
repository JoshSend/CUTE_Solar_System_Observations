COMMAND PLANS - Uranus (CUTE targetID 330)
==============================================================================

Copied from GUI/JSON/Uranus/. These are the plan files that commanded the
science frames held in this folder.

  Visit3  <- CUTE_Plan_09.27.2024.json
            science frames 2140-2147, first exposure 2024-09-27 08:12:45 UTC
            (data on disk: frmid 2140-2143, first frame 08:12:46 UTC)

  Visit5  <- CUTE_Plan_10.21.2024.json
            science frames 3149-3163, first exposure 2024-10-21 05:48:35 UTC
            (data on disk: frmid 3149-3156, first frame 05:48:36 UTC)

VERIFICATION
------------------------------------------------------------------------------
Each commanded exposure was compared with the FITS DATE-OBS of the frame on
disk. A single PLD_CCD_EXPOSE command with NUM=n starts a sequence of n
exposures, so frame i of a block is expected at t0 + i*(exposure + 65 s).

   Visit3  4 frames, cadence 240 s + 65 s overhead -> all match within 1 s
   Visit5  8 frames, cadence 300 s + 65 s overhead -> all match within 1 s

No frame on disk is absent from its plan.

NOTE ON FRAME-ID COLLISIONS
------------------------------------------------------------------------------
Frame IDs alone do not identify a plan. CUTE_Plan_10.12.2024.json commands the
same frame IDs 3149-3156, but on 2024-10-12 (the Visit4 attempt, whose science
frames never executed). The plans above were matched on commanded UTC time,
which agrees with the FITS DATE-OBS to within one second.

OTHER URANUS PLANS NOT COPIED
------------------------------------------------------------------------------
GUI/JSON/Uranus/ holds seven further plans, all targetID 330, for visits that
produced no science data in the archive:

  CUTE_Plan_09.06.2024.json   Visit1, frames 1026-1033   (science DNE)
  CUTE_Plan_09.14.2024.json   Visit2 plan, frames 1100-1107
  CUTE_Plan_09.20.2024.json   Visit2 re-plan, frames 1101-1109
  CUTE_Plan_10.05.2024.json   frames 3096-3103, not in the exposure list
  CUTE_Plan_10.12.2024.json   Visit4, frames 3149-3156   (superseded by Visit5)
  CUTE_Plan_11.02.2024.json   Visit6, frames 5233-5247   (science DNE)
  CUTE_Plan_11.09.2024.json   Visit6 re-plan, frames 5233-5247 (science DNE)
