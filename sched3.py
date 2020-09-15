import math
import ulab as np

ground_stations = [[37.4241, -122.166], [37.4241, -75], [-30.5595, 22.9375], [90, 0]]

class Earth():
    """
    class for storing all Earth-related parameters
    """
    mu = 3.986004418E5 # specific gravitational parameter
    R = 6371.009 # radius in km
    J2 = 1.08262668E-3 # J2 value
    e = 0.08181919 # elliptical Earth


earth = Earth()

for station in ground_stations:
    # constants
    deg2rad = math.pi/180
    # get station ECEF
    N = earth.R / math.sqrt(1 - earth.e ** 2 * math.sin(station[0]*deg2rad) ** 2)
    r_station = np.array(
    [N * np.vector.cos(station[0]*deg2rad) * np.vector.cos(station[1]*deg2rad),
    N * np.vector.cos(station[0]*deg2rad) * np.vector.sin(station[1]*deg2rad),
    N*(1-earth.e ** 2) * np.vector.sin(station[0]*deg2rad)])
    print(r_station)