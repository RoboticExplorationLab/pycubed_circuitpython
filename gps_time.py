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
    MJD_int = MJD_gps_epoch + GNSS_days_int

    # current mjd float component
    MJD_float = GNSS_days_float

    # return the parts to MJD
    return MJD_int, MJD_float


# here is data from something Max sent on september 11th

# weeks from midnight january 6th 1980
GNSS_week = 2131

# seconds since the week started (raw gps int)
TOW = 16028200

MJD_int, MJD_float = mjd_from_gps(GNSS_week, TOW)

print(MJD_int)
print(MJD_float)

def eci_from_ecef_gps(MJD_int,MJD_float,vector_eci):

    # get earth rotation angle
    era_0 = 1.7557955403696752
    mjd_0 = 59215

    delta_mjd_int = MJD_int - mjd_0
    delta_mjd_float = MJD_float

    w_earth_rad_per_day_p1 = 6.30038
    w_earth_rad_per_day_p2 = .7486754831e-5

    a = delta_mjd_int*w_earth_rad_per_day_p1 % (2*math.pi) + delta_mjd_int*w_earth_rad_per_day_p2 % (2*math.pi)
    b = delta_mjd_float*w_earth_rad_per_day_p1 % (2*math.pi) + delta_mjd_float*w_earth_rad_per_day_p2 % (2*math.pi) + era_0

    # here I get sin and cosine of GMST
    sin_theta = math.sin(a)*math.cos(b) + math.cos(a)*math.sin(b)
    cos_theta = math.cos(a)*math.cos(b) - math.sin(a)*math.sin(b)

    print(sin_theta)
    print(cos_theta)

    print(a+b)
    print(math.sin(a+b))
    print(math.cos(a+b))

    return np.array([cos_theta*vector_eci[0] - sin_theta*vector_eci[1],
                     sin_theta*vector_eci[0] + cos_theta*vector_eci[1],
                     vector_eci[2] ])

def earth_rotation_angle_gps(MJD_int,MJD_float):

    # get earth rotation angle
    era_0 = 1.7557955403696752
    mjd_0 = 59215

    delta_mjd_int = MJD_int - mjd_0
    delta_mjd_float = MJD_float

    w_earth_rad_per_day_p1 = 6.30038
    w_earth_rad_per_day_p2 = .7486754831e-5

    a = delta_mjd_int*w_earth_rad_per_day_p1 % (2*math.pi) + delta_mjd_int*w_earth_rad_per_day_p2 % (2*math.pi)
    b = delta_mjd_float*w_earth_rad_per_day_p1 % (2*math.pi) + delta_mjd_float*w_earth_rad_per_day_p2 % (2*math.pi) + era_0

    return (a+b) % (2*math.pi)

eci_from_ecef_gps(MJD_int,MJD_float,np.array([1,2,3.0]))