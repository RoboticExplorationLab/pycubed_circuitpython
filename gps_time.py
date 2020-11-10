import math
import ulab as np

def mjd_from_gps(GNSS_week, TOW):
    """Returns MJD from gps time.

    Args:
        GNSS_week: Weeks since 0h January 6th, 1980 (uint16, units: weeks)
        TOW: Seconds into the week (uint16, units: 1/100 seconds)

    Returns:
        MJD_int: first part of UTC MJD (int)
        MJD_float: second part of UTC MJD (float)

    Comments:
        Both of the inputs to this function are the raw GPS time parameters.
        The TOW term is scaled by 0.01 after being input to the function.
        This means the input to the function is the raw gps int.

        This has be tested with <http://leapsecond.com/java/gpsclock.htm>
    """

    # seconds since the week started (GPS is off from UTC by 18 seconds)
    TOW = (TOW // 100) - 18

    # MJD when GPS time started
    MJD_gps_epoch = 44244

    # get days since this epoch
    GNSS_days = GNSS_week * 7 + (TOW / 86400)

    # integer julian days for start of the week since GPS epoch
    GNSS_days_int = GNSS_week * 7

    # float julian days for current time from start of week
    GNSS_days_float = TOW/86400

    # get current MJD int component
    MJD_int = MJD_gps_epoch + GNSS_days_int + math.floor(GNSS_days_float)

    # current mjd float component
    MJD_float = GNSS_days_float % 1

    # return the parts to MJD
    return MJD_int, MJD_float


def earth_rotation_angle_gps(MJD_int,MJD_float):
    """Earth rotation angle from gps data.

    Args:
        MJD_int: integer of MJD (floor(MJD))
        MJD_float: fractional part of MJD (MJD - floor(MJD))

    Returns:
        Era: Earth rotational angle (radians)
        
    Comments:
        Information on the ERA from MJD can be found here:
        <https://www.aanda.org/articles/aa/pdf/2006/45/aa5897-06.pdf>
    """

    # get earth rotation angle at epoch
    era_0 = 1.7557955403696752
    mjd_0 = 59215

    # find the delta MJD since the epoch. This is divided into an int and a float
    delta_mjd_int = MJD_int - mjd_0
    delta_mjd_float = MJD_float

    # (era_0 offset) + (2pi*revolutions since epoch) + (float * radians_p_day)
    Era = era_0 + 2*math.pi*delta_mjd_int*0.0027378119 + delta_mjd_float*6.300387

    return Era

def rveci_from_ecef(r_ecef,v_ecef,era):
    """Get rv in eci from ecef and earth rotation angle.

    Args:
        r_ecef: position in ecef (km)
        v_ecef: velocity in ecef wrt ecef (km/s)
        era: earth rotation angle (radians)

    Returns:
        r_eci: position in eci (km)
        v_eci: velocity in eci wrt eci (km/s)
    """

    sin_theta = math.sin(era)
    cos_theta = math.cos(era)

    r_eci = np.array([cos_theta*r_ecef[0] - sin_theta*r_ecef[1],
                      sin_theta*r_ecef[0] + cos_theta*r_ecef[1],
                      r_ecef[2] ])

    omega_earth = 7.292115146706979e-5
    v_eci = np.array([cos_theta*v_ecef[0] - sin_theta*v_ecef[1] - omega_earth*r_eci[1],
                      sin_theta*v_ecef[0] + cos_theta*v_ecef[1] + omega_earth*r_eci[0],
                      v_ecef[2] ])

    return r_eci, v_eci


# ----------------RAW GPS INFO--------------------
# weeks from midnight january 6th 1980
GNSS_week = 2131

# seconds since the week started (raw gps int)
TOW = 19047200

# ecef position and velocity
r_ecef = (-7384816,-655378593, 149326468)
v_ecef = (682728,44684,249507)
#-------------------------------------------------

def propagatorinfo_from_gps(GNSS_week, TOW, r_ecef, v_ecef):
    """Get propagator starting info from GPS data.

    Args:
        GNSS_week: Weeks since january 6th 1980
        TOW: Seconds into current week
        r_ecef: ecef position (.01 meters)
        v_cef: ecef velocity wrt ecef (0.01 m/s)

    Returns:
        era: Earth rotation angle (radians)
        r_eci: position in eci (km)
        v_eci: velocity in eci wrt eci (km/s)

    Comments:
        Inputs are raw GPS data, no pre-scaling needed.
    """

    # get MJD from gps time info
    MJD_int, MJD_float = mjd_from_gps(GNSS_week, TOW)

    # compute earth rotation angle from the MJD
    era = earth_rotation_angle_gps(MJD_int,MJD_float)

    # scale position and velocity
    r_ecef = np.array([r_ecef[0],r_ecef[1],r_ecef[2]])/100000
    v_ecef = np.array([v_ecef[0],v_ecef[1],v_ecef[2]])/100000

    # convert to inertial position and velocity
    r_eci, v_eci = rveci_from_ecef(r_ecef,v_ecef,era)

    # debugging printouts
    print('MJD int',MJD_int)
    print('MJD float', MJD_float)
    print('Earth Rotation Angle',era)
    print('r_ecef',r_ecef)
    print('v_ecef',v_ecef)
    print('r_eci',r_eci)
    print('v_eci',v_eci)

    return era, r_eci, v_eci



propagatorinfo_from_gps(GNSS_week, TOW, r_ecef, v_ecef)
