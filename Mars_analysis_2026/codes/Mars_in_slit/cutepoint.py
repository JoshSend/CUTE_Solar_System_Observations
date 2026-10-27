# Nicholas Nell
# nicholas.nell@lasp.colorado.edu
# 12/28/2021
#
# Convenience functions for pointing CUTE cubesat.
# required packages:
# astropy
# numpy
# scipy
# jplephem (for ephemeris)
# sgp4 (for TLE)

# CUTE TLE
#1 49263U 21088D   21361.75880348  .00001913  00000-0  15087-3 0  9997
#2 49263  97.6063  68.1056 0017878 292.8538  67.0798 15.01811118 13649

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import ITRS
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, TEME
from astropy.coordinates import get_body
from astropy.time import Time
import astropy.units as u
from sgp4.api import Satrec
from sgp4.api import SGP4_ERRORS

# Used to make plots in roll optimization debug
#import matplotlib.pyplot as mp


# Test tolerance
TOL = 1.0e-10

# Nominal star tracker angle offset
ST_ANGLE = 10.0


# Tait-Bryan Angles

# Tait-Bryan roll
def tb_phi(e1, e2, e3, e4):
    phi = np.arctan2((e2*e3 + e4*e1), 0.5 - (e1*e1 + e2*e2))
    return(phi)

# Tait-Bryan theta [declination (sign inverse)]
def tb_theta(e1, e2, e3, e4):
    theta = np.arcsin(-2.0*(e1*e3 - e4*e2))
    # Convention for declination?
    #theta = -1.0*theta
    return(theta)

# Tait-Bryan PSI (RA)
def tb_psi(e1, e2, e3, e4):
    psi = np.arctan2((e1*e2 + e4*e3), 0.5 - (e2*e2 + e3*e3))
    return(psi)


# Convert Tait-Bryan angles to quaternion
# Input angles in radians
# Output is quaternion in BCT format
def tb_to_quat(phi, theta, psi):
    
    cpsi = np.cos(psi/2.0)
    ctheta = np.cos(theta/2.0)
    cphi = np.cos(phi/2.0)

    spsi = np.sin(psi/2.0)
    stheta = np.sin(theta/2.0)
    sphi = np.sin(phi/2.0)

    e4 = cpsi*ctheta*cphi + spsi*stheta*sphi
    e1 = cpsi*ctheta*sphi - spsi*stheta*cphi
    e2 = cpsi*stheta*cphi + spsi*ctheta*sphi
    e3 = spsi*ctheta*cphi - cpsi*stheta*sphi

    return((e1, e2, e3, e4))

# Convert quaternion into Tait-Bryan angles
# Input is unit quaternion in BCT format
def quat_to_tb(e1, e2, e3, e4):
    phi = tb_phi(e1, e2, e3, e4)
    theta = tb_theta(e1, e2, e3, e4)
    psi = tb_psi(e1, e2, e3, e4)

    return((phi, theta, psi))

# Convert RA, DEC, ROLL (degrees) to Tait-Bryan angles (radians)
def ra_dec_roll_to_tb(ra, dec, roll):
    phi = np.radians(roll)
    theta = -1.0*np.radians(dec)
    psi = np.radians(ra)

    return((phi, theta, psi))

# Convert Tait-Bryan angles (radians) to RA, DEC, ROLL (degrees)
def tb_to_ra_dec_roll(phi, theta, psi):
    roll = np.degrees(phi)
    dec = -1.0*np.degrees(theta)
    ra = np.degrees(psi)

    return((ra, dec, roll))


# RA/DEC/ROLL convenience functions from CUTE telemetry points

# Return RA/DEC in degrees from a unit ECI vector.
def ra_dec_from_eci_vect(x, y, z):
    ra = np.arctan2(y, x)
    if (ra < 0.0):
        ra = ra + np.pi + np.pi
    dec = np.arcsin(z)

    ra = np.degrees(ra)
    dec = np.degrees(dec)

    return((ra, dec))

# Spacecraft quaternion from RA, DEC, and ROLL (in degrees)
def sc_quat_from_ra_dec_roll(ra, dec, roll):
    phi, theta, psi = ra_dec_roll_to_tb(ra, dec, roll)
    q1, q2, q3, q4 = tb_to_quat(phi, theta, psi)
    return((q1, q2, q3, q4))


# Star Tracker RA/DEC/ROLL pointing from T1_ATTITUDE_ST1 -
# T1_ATTITUDE_ST4 telemetry points (do not use other quaternions with
# this!). This factors in the different coordinate convention for the
# Star Tracker. Return RA/DEC/ROLL in degrees.
def st_ra_dec_roll_from_st_quat(e1, e2, e3, e4):
    R = Rotation.from_quat([e1, e2, e3, e4])
    M = R.as_matrix()
    # Star Tracker boresight is Z
    STATT = M.dot([0, 0, 1])
    STRA, STDEC = ra_dec_from_eci_vect(STATT[0], STATT[1], STATT[2])
    # TODO: when we know ST_ROLL orientation 
    STROLL = 0.0

    return((STRA, STDEC, STROLL))


# Star Tracker RA/DEC/ROLL pointing from either commanded or measured
# SPACECRAFT quaternion.
def st_ra_dec_roll_from_sc_quat(e1, e2, e3, e4):
    R = Rotation.from_quat([e1, e2, e3, e4])
    M = R.as_matrix()
    # Star Tracker boresight is 10* off of SC frame
    theta = np.radians(ST_ANGLE)
    vec = [np.cos(theta), np.sin(theta), 0.0]
    STATT = M.dot(vec)
    STRA, STDEC = ra_dec_from_eci_vect(STATT[0], STATT[1], STATT[2])
    # TODO: when we know ST_ROLL orientation 
    STROLL = 0.0

    return((STRA, STDEC, STROLL))

# Get RA, DEC, and ROLL (in degrees) from a spacecraft frame
# quaternion.
def sc_ra_dec_roll_from_sc_quat(e1, e2, e3, e4):
    phi, theta, psi = quat_to_tb(e1, e2, e3, e4)
    ra, dec, roll = tb_to_ra_dec_roll(phi, theta, psi)
    return((ra, dec, roll))
    

# Radiator RA/DEC from a spacecraft quaternion
def rad_ra_dec_from_sc_quat(e1, e2, e3, e4):
    R = Rotation.from_quat([e1, e2, e3, e4])
    M = R.as_matrix()
    # Radiator points in SC +Z direction
    V = [0, 0, 1]
    RATT = M.dot(V)
    R_RA, R_DEC = ra_dec_from_eci_vect(RATT[0], RATT[1], RATT[2])

    return((R_RA, R_DEC))


# Get unit target vector for GOTO_TARGET from an RA/DEC/ROLL.
def target_vec_from_ra_dec_roll(ra, dec, roll):
    # Telescope points in +X direction
    V = [1, 0, 0]
    q1, q2, q3, q4 = sc_quat_from_ra_dec_roll(ra, dec, roll)
    R = Rotation.from_quat([q1, q2, q3, q4])
    M = R.as_matrix()
    target_vec = M.dot(V)

    return(target_vec)

def rotate_coordinates(ra, dec, roll, V):
    q1, q2, q3, q4 = sc_quat_from_ra_dec_roll(ra, dec, roll)
    R = Rotation.from_quat([q1, q2, q3, q4])
    M = R.as_matrix()
    new_eci_vec = M.dot(V)
    new_ra, new_dec = ra_dec_from_eci_vect(*new_eci_vec)
    return(new_ra, new_dec)

# CUTE roll optimizer (keep it cold)
# obs_t: observation time (astropy.Time format)
def cute_roll_opt(ra, dec, roll, obs_t):
    # Use good ephemeris
    solar_system_ephemeris.set('de432s')

    # Use TLE to locate CUTE's "earth position". Last updated 2022.09.19
    s = "1 49263U 21088D   22262.17576613  .00004099  00000+0  30184-3 0  9993"
    t = "2 49263  97.5818 328.2242 0022025 105.2513 255.1150 15.03710872 53501"
 
    cute = Satrec.twoline2rv(s, t)

    error_code, teme_p, teme_v = cute.sgp4(obs_t.jd1, obs_t.jd2)
    if error_code != 0:
        raise RuntimeError(SGP4_ERRORS[error_code])

    teme_p = CartesianRepresentation(teme_p*u.km)
    teme_v = CartesianDifferential(teme_v*u.km/u.s)
    teme = TEME(teme_p.with_differentials(teme_v), obstime = obs_t)

    itrs = teme.transform_to(ITRS(obstime = obs_t))  
    loc = itrs.earth_location
    #location.geodetic

    # The sun.
    sun = get_body('sun', obs_t, loc)
    # NOTE: get_body() object not compatible with skycoord
    # separation... need to make a new skycoord...
    sun_c = SkyCoord(sun.ra.deg, sun.dec.deg, unit = (u.deg, u.deg))

    # Cute's pointing
    c = SkyCoord(ra, dec, unit = (u.deg, u.deg))
    
    # print(c.ra.deg, c.dec.deg)
    # print(sun.ra.deg, sun.dec.deg)
    # print(c.separation(sun))
    #print(roll)

    # Test and debug stuff left in place in case of rework/debug in
    # the future.
    
    # Get quaternion for CUTE pointing
    # q1, q2, q3, q4 = sc_quat_from_ra_dec_roll(ra, dec, roll)
    # rad_ra, rad_dec = rad_ra_dec_from_sc_quat(q1, q2, q3, q4)
    # c_rad = SkyCoord(rad_ra, rad_dec, unit = (u.deg, u.deg))
    # print(rad_ra, rad_dec)
    # print(c_rad.separation(c).deg)
    # print(c_rad.separation(sun_c).deg)
    # print(roll)

    # roll = np.linspace(0.0, 359.0, 200)
    # sep = []
    # for roll_v in roll:
    #     q1, q2, q3, q4 = sc_quat_from_ra_dec_roll(ra, dec, roll_v)
    #     rad_ra, rad_dec = rad_ra_dec_from_sc_quat(q1, q2, q3, q4)
    #     c_rad = SkyCoord(rad_ra, rad_dec, unit = (u.deg, u.deg))
    #     sep.append(c_rad.separation(sun_c).deg)

    # #example of plotting roll dependence
    # mp.plot(roll, sep)
    # mp.grid()
    # mp.xlabel("Roll angle (degrees)")
    # mp.ylabel("Radiator-Sun separation angle (degrees)")
    # mp.title("Pointing CUTE at RA 78.64 DEC -8.20")
    # mp.show()

    # Based on bounds it may be wise to choose a naive roll like 180*
    # to let the minimizer explore decently.
    result = minimize(_cute_roll_min_func, [roll], args = (ra, dec, sun_c), bounds = [(0.0, 359.99)])
    #print(result)
    if (result.status != 0):
        print(result)
        raise Exception("Optimization failed")

    opt_roll = result.x[0]
    #print(opt_roll)

    return(opt_roll)


# Helper minimization function for roll optimization. Do not use
# me. Don't even look at
# me. https://www.youtube.com/watch?v=g7-5io1muSQ
def _cute_roll_min_func(x, ra, dec, sun):
    #print(x)
    roll = x[0]
    q1, q2, q3, q4 = sc_quat_from_ra_dec_roll(ra, dec, roll)
    rad_ra, rad_dec = rad_ra_dec_from_sc_quat(q1, q2, q3, q4)
    c_rad = SkyCoord(rad_ra, rad_dec, unit = (u.deg, u.deg))
    return(180.0 - c_rad.separation(sun).deg)
    
def rotate_coordinates(ra, dec, roll, V):
    q1, q2, q3, q4 = sc_quat_from_ra_dec_roll(ra, dec, roll)
    R = Rotation.from_quat([q1, q2, q3, q4])
    M = R.as_matrix()
    new_eci_vec = M.dot(V)
    new_ra, new_dec = ra_dec_from_eci_vect(*new_eci_vec)
    return(new_ra, new_dec)

# Run some basic verification.
def run_test():
    # test angles to try out...
    phi = 159.54570059967233
    theta = 8.201028142889854
    psi = 78.63973954785719

    # Test the basic quat/tb conversion
    e1, e2, e3, e4 = tb_to_quat(np.radians(phi), np.radians(theta), np.radians(psi))

    # print(e1)
    # print(e2)
    # print(e3)
    # print(e4)
    # Should be roughly...
    #0.7513351529732967
    #0.6318071533638566
    #0.0577772153444556
    #0.1815956018680878

    assert(np.abs(np.sqrt(e1*e1 + e2*e2 + e3*e3 + e4*e4) - 1.0) < TOL)

    phi_n, theta_n, psi_n = quat_to_tb(e1, e2, e3, e4)
    phi_n = np.degrees(phi_n)
    theta_n = np.degrees(theta_n)
    psi_n = np.degrees(psi_n)

    # print(phi_n)
    # print(theta_n)
    # print(psi_n)
    
    # print(phi - phi_n)
    # print(theta - theta_n)
    # print(psi - psi_n)

    # Converting back/forth should leave only numerical roundoff errors...
    assert(np.abs(phi - phi_n) <= TOL)
    assert(np.abs(theta - theta_n) <= TOL)
    assert(np.abs(psi - psi_n) <= TOL)


    # Test RA/DEC conversion
    # Alpha crucis from SIMBAD
    # 186.64956340 -63.09909286
    c = SkyCoord(186.64956340, -63.09909286, unit = (u.deg, u.deg))
    roll = 0.0
    
    phi_a, theta_a, psi_a = ra_dec_roll_to_tb(c.ra.deg, c.dec.deg, roll)

    #print(np.degrees(phi_a))
    #print(np.degrees(theta_a))
    #print(np.degrees(psi_a))

    # More to do here... check large RAs, high/low decs, etc.
    

    # local time
    obst = Time("2021-12-28T06:30:00")
    #def cute_roll_opt(ra, dec, roll, obs_t):
    ra = 78.63973954785719
    dec = -8.2
    opt_roll = cute_roll_opt(ra, dec, 180.0, obst)
    assert(opt_roll < 350.0 and opt_roll > 300.0)
    
    print("TESTS PASSED\n\n")
    

# Demonstrating missing roll information
def example1():
    # An RA, DEC for telescope pointing
    ra = 78.63973954785719
    dec = -8.2

    # Arbitrary roll
    roll = 0.0
    # obs time.
    obst = Time("2021-12-28T06:30:00")

    # Generate an optimal roll orientation for the observation time
    opt_roll = cute_roll_opt(ra, dec, 280.0, obst)
    
    tvec1 = target_vec_from_ra_dec_roll(ra, dec, roll)
    print("Target Vector 1: ", tvec1)
    tvec2 = target_vec_from_ra_dec_roll(ra, dec, opt_roll)
    print("Target Vector 2:", tvec2)
    print("This pointing vector is missing roll information...")

    # These two target vectors are identical -- we're not controlling our roll

    # If we do this with the full quaternion we get different
    # quaternions for different rolls -- allowing us to control
    # instrument roll angle.
    quat1 = sc_quat_from_ra_dec_roll(ra, dec, roll)
    print("Quaternion 1: ", quat1)
    quat2 = sc_quat_from_ra_dec_roll(ra, dec, opt_roll)
    print("Quaternion 2: ", quat2)
    print("Now we're capturing instrument roll and target pointing in a single quaternion")

    # This also serves as an example of how to easily generate a
    # target vector or a target quaternion from an RA, DEC, and roll.
    print("End example 1\n\n\n\n")


# Have a look at instrument and ST RA, DEC
def example2():
    #,time,T1_DEC,T1_RA,T1_ROLL,T1_ST1,T1_ST2,T1_ST3,T1_ST4,CMD1,CMD2,CMD3,CMD4,Q1,Q2,Q3,Q4
    #0,2021-12-22 18:00:01,-4.6145,69.24499999999999,68.497,-0.5230963208007813,0.5163104243164063,0.2744123046875,0.6200739243164062,0.7513351529732967,0.6318071533638566,0.0577772153444556,0.1815956018680878,0.7513325064733496,0.6318093658638124,0.0577820723443585,0.1815973088680537

    # This data is a single point in time from the 30 minute dark.

    # The commanded target vector from the JSON script
    TARG_X = 0.19505203
    TARG_Y = 0.97036262
    TARG_Z = -0.14265724

    # RA, DEC from this target vector
    RA, DEC = ra_dec_from_eci_vect(TARG_X, TARG_Y, TARG_Z)

    print("JSON RA DEC: {0:f} {1:f}".format(RA, DEC))

    # Data from telemetry showing commanded quaternion:
    CMD1 = 0.7513351529732967
    CMD2 = 0.6318071533638566
    CMD3 = 0.0577772153444556
    CMD4 = 0.1815956018680878

    CMD_RA, CMD_DEC, CMD_ROLL = sc_ra_dec_roll_from_sc_quat(CMD1, CMD2, CMD3, CMD4)
    print("CMD RA DEC ROLL: {0:f} {1:f} {2:f}".format(CMD_RA, CMD_DEC, CMD_ROLL))

    # Data from telemetry showing ACS measured pointing quaternion
    Q1 = 0.7513325064733496
    Q2 = 0.6318093658638124
    Q3 = 0.0577820723443585
    Q4 = 0.1815973088680537

    Q_RA, Q_DEC, Q_ROLL = sc_ra_dec_roll_from_sc_quat(Q1, Q2, Q3, Q4)
    print("MEASURED RA DEC ROLL: {0:f} {1:f} {2:f}".format(Q_RA, Q_DEC, Q_ROLL))

    # Star Tracker Quat
    ST1 = -0.5230963208007813
    ST2 = 0.5163104243164063
    ST3 = 0.2744123046875
    ST4 = 0.6200739243164062

    ST_RA, ST_DEC, ST_ROLL = st_ra_dec_roll_from_st_quat(ST1, ST2, ST3, ST4)

    print("ST QUAT RA DEC: {0:f} {1:f}".format(ST_RA, ST_DEC))

    # This is directly from ST telemetry
    ST_RA_TM = 69.24499999999999
    ST_DEC_TM = -4.6145

    print("ST TM RA DEC: {0:f} {1:f}".format(ST_RA_TM, ST_DEC_TM))

    # Here we assume the tracker 10* offset from speacecraft boresight
    # and compure the expected tracker RA and DEC from the measured SC
    # quaternion. This may be useful for minor adjustments in pointing
    # measured off of tracker pointing after a successful scan?
    ST_RA_SC, ST_DEC_SC, ST_ROLL_SC = st_ra_dec_roll_from_sc_quat(Q1, Q2, Q3, Q4)
    print("ST FROM SC QUATERNION RA DEC: {0:f} {1:f}".format(ST_RA_SC, ST_DEC_SC))

    # I think there is cal that nails down the tracker vs SC angle --
    # you can see a fair amount of TM about tracker alignment in the
    # "Calibration" section in the CTDB. But I think we have enough
    # info to figure out where we are.

    cmd_c = SkyCoord(CMD_RA, CMD_DEC, unit = (u.deg, u.deg))
    q_c = SkyCoord(Q_RA, Q_DEC, unit = (u.deg, u.deg))

    alignment_error = cmd_c.separation(q_c)
    print("Alignment error between commanded and measured pointing: {0:f} arcseconds".format(alignment_error.arcsec))
    print("End example 2\n\n")


def main():
    run_test()
    example1()
    example2()

if __name__ == '__main__':
    main()
    