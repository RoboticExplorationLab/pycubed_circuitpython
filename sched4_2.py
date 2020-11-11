import math
import ulab as np
import time

import gc

print(gc.mem_alloc())
'''
MAX: need to comment up & add TODO
'''

#----------DEBUGGING FUNCTIONS----
def checkcolvec(x):
    a,b = x.shape

    if b != 1:
        print("this is the problem vector",x)
        print("this is the vector type",type(x))
        print("this is the vector size",x.shape)
        print("this is an element type",type(x[0]))
        raise Exception("Not a column vector")

    if not isinstance(x[0],float):
        print("this is the problem vector",x)
        print("this is the vector type",type(x))
        print("this is the vector size",x.shape)
        print("this is an element type",type(x[0]))
        raise Exception("this is an array of arrays")

def checkrowvec(x):

    a,b = x.shape

    if a != 1:
        print("this is the problem vector",x)
        print("this is the vector type",type(x))
        print("this is the vector size",x.shape)
        print("this is an element type",type(x[0]))
        raise Exception("Not a column vector")

    if not isinstance(x[0],float):
        print("this is the problem vector",x)
        print("this is the vector type",type(x))
        print("this is the vector size",x.shape)
        print("this is an element type",type(x[0]))
        raise Exception("this is an array of arrays")

# ----------- Earth --------------
class Earth():
    """
    class for storing all static Earth-related parameters
    """
    mu = 3.986004415E5 # specific gravitational parameter
    R = 6.3781363e3 # radius in km
    J2 = 0.0010826358191967 # J2 value


# ----------- Dynamics --------------

def rk4_propagate(x,dt,Earth):
    """RK4 for orbital propagation.

    Args:
        x: step [pos;vel] (km,km/s)
        dt: time step (s)
        Earth: Earth class

    Returns:
        x_{t+1}: state at next time step
    """

    k1 = dt*dynamics(x,Earth)
    k2 = dt*dynamics(x+k1/2,Earth)
    k3 = dt*dynamics(x+k2/2,Earth)
    k4 = dt*dynamics(x+k3,Earth)

    return x + (1/6)*(k1+2*k2+2*k3+k4)


def norm(x):
    """norm of an np.array"""
    return math.sqrt(np.numerical.sum(x**2))

def normalize(x):
    """normalize an np.array (not in place)"""
    return x/norm(x)



def dynamics(x,Earth):
    """FODE + J2 dynamics function

    Args:
        x: [rx,ry,rz,vx,vy,vz] in km, km/s
        earth: Earth class

    Returns:
        xdot: [vx, vy, vz, ax, ay, az]
    """

    # precompute a few terms to reduce number of operations
    r = norm(x[0:3])
    Re_r_sqr = 1.5*Earth.J2*(Earth.R/r)**2
    five_z_sqr = 5*x[2]**2/(r**2)

    # two body and J2 acceleration together
    accel = (-Earth.mu/(r**3))*np.array([x[0]*(1 - Re_r_sqr*(five_z_sqr - 1)),
                                         x[1]*(1 - Re_r_sqr*(five_z_sqr - 1)),
                                         x[2]*(1 - Re_r_sqr*(five_z_sqr - 3))])

    return np.array([x[3],x[4],x[5],accel[0],accel[1],accel[2]])



def ecef_from_eci(r_eci,earth_rotation_angle_offset,t_since_epoch):
    """ecef position from eci and earth rotation angle.

    Args:
        r_eci: sc position in ECI (km)
        earth_rotation_angle_offset: angle of earth at time of epoch (rad)
        t_since_epoch: time since epoch (s)

    Returns:
        r_ecef: sc position in ECEF (km)

    Comment 1:
        The math is the following:
            r_ecef = RotZ(GMST)*r_eci
        where
            GMST = earth_rotation_angle_offset + t_since_epoch*w_earth

        I split up GMST into two components for better accuracy

    Comment 2:
        I used the following angle addition formulas:
        sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
        cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    """

    # gmst angle = a + b
    a = t_since_epoch*7.29211e-5 + earth_rotation_angle_offset
    b = t_since_epoch*5.14671e-11

    # here I get sin and cosine of GMST
    sin_theta = math.sin(a)*math.cos(b) + math.cos(a)*math.sin(b)
    cos_theta = math.cos(a)*math.cos(b) - math.sin(a)*math.sin(b)

    # instead of calling a RotZ function, I just do it all here
    return np.array([cos_theta*r_eci[0] + sin_theta*r_eci[1],
                    -sin_theta*r_eci[0] + cos_theta*r_eci[1],
                     r_eci[2] ])


def elevation_dot_product(r_ecef,r_station,earth):
    """Returns dot product of the following two position vectors:
        (1) normalized vector from center of earth to ground station
        (2) normalized vector from ground station to satellite

    Args:
        r_ecef: spacecraft position in ecef (km)
        r_station: ground station position in ecef (km)
        earth: Earth class

    Returns:
        dot product of these two vectors
    """

    return (np.numerical.sum(normalize(r_ecef - r_station)*normalize(r_station)))

# ----------- Scheduler --------------


class Propagator():

    def __init__(self,rv_eci,earth_rotation_angle_offset,t_epoch,ground_stations):
        """Initialize the propagator with the stuff we received from
        the ground station, as well as pre-allocate other parameters.

        Args:
            rv_eci: state [r (km);v (km/s)] at t_epoch
            earth_rotation_angle_offset: GMST at t_epoch (radian)
            t_epoch: on board satellite time when propagator is initialized (s)
        """

        # [r;v] in km, km/s
        self.rv_eci = 1*rv_eci

        # current earth rotation angle
        self.earth_rotation_angle_offset = earth_rotation_angle_offset

        # initial time that propagator was started
        self.t_epoch = t_epoch

        # store ecef location
        self.r_ecef = np.zeros(3)

        # constants
        self.earth = Earth()

        # visible to ground station
        self.visible = False

        # visible to ground station (previous step)
        self.old_visible = False

        # ground stations
        self.ground_stations = ground_stations


    def step(self,dt):
        """Take a step in the propagator.

        Args:
            dt: time step (s) must be greater than 20

        Summary:
            This function integrates the orbital position and velocity
            with respect to a J2-only gravity model
        """

        if float(dt) > 50.0:
            raise Exception("dt is too big (>50 seconds)")

        # send dynamics forward one s
        self.rv_eci = rk4_propagate(self.rv_eci,dt,self.earth)


    def get_r_ecef(self,t_current):
        """Get ecef location of spacecraft.

        Args:
            t_current: on-board spacecraft time (s)
        """

        self.r_ecef = ecef_from_eci(self.rv_eci,
                self.earth_rotation_angle_offset,t_current - self.t_epoch)


    def check_visibility(self):
        """Check if the spacecraft is visible from a ground station.

        Summary: sets self.visible to False, then cycles through the
        ground stations and if one of them is visible, sets self.visible
        to True.
        """

        # put the current reading back in the old reading
        self.old_visible = self.visible

        # find new reading
        self.visible = False
        for gs in self.ground_stations:
            if (elevation_dot_product(self.r_ecef,gs,self.earth) >  0.0):
                self.visible = True

# ---------------------- Testing script --------------------------


#----------Communication from the ground-------------------
# communication from the ground tells us rv_eci, and earth_rotation_angle_offset
rv_eci = np.array([-1861.7015559490976, -6645.09702340011, 0.032941584155793194,
              -0.9555092604431692, 0.29391099680436356, 7.541418280028347])
earth_rotation_angle_offset = 4.273677081313352


#----------Initialize propagator---------------------------
# the spacecraft then initializes the propagator with this information, and
# takes note of the current time as t_epoch
t_epoch = 0 # hard coded now, but would be current on board time at time of transmission
t_epoch = 6856.05
# ground station locations
stanford_ecef = np.array([-2.7001052e3, -4.29272716e3, 3.855177275e3])
ground_stations = [stanford_ecef]
propagator = Propagator(rv_eci, earth_rotation_angle_offset, t_epoch, ground_stations)

# step size for integrator
dt = 30


#----------Run propagator----------------------------------

# empty list (to be list of 2 element lists with start and stop times)
passes = []
n_passes = 0


t1 = time.monotonic()
for i in range(8640):

    # current time
    t_current = dt*i + t_epoch

    # get ecef location
    propagator.get_r_ecef(t_current)

    # check visibility from ground station
    propagator.check_visibility()

    # integrate to the next step
    propagator.step(dt)

    if (not propagator.old_visible) and propagator.visible:
        # if we just entered visibility, add the start time
        passes.append([t_current,0])

        # increment the number of passes by 1
        n_passes += 1


    if propagator.old_visible and (not propagator.visible):
        # if we just left visibility, add the stop time
        passes[n_passes-1][1] = t_current

print("time")
print(time.monotonic()-t1)
#print(propagator.rv_eci)
#print(propagator.r_ecef)

print(passes)
print(n_passes)

gc.collect()
print(gc.mem_alloc())