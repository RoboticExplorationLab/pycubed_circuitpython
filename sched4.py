import math
import ulab as np
import time

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
    """Vanilla RK4 for orbital propagation.

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



def ecef_from_eci(r_eci,earth_rotation_angle_offset,t_current):
    """ecef position from eci and earth rotation angle.

    Args:
        r_eci: sc position in ECI (km)
        earth_rotation_angle_offset: angle of earth at time of epoch (rad)
        t_current: time since epoch (s)

    Returns:
        r_ecef: sc position in ECEF (km)

    Comments:
        The math is the following:
            r_ecef = RotZ(GMST)*r_eci
        where
            GMST = earth_rotation_angle_offset + t_current*w_earth

        I split up GMST into two components for better accuracy
    """

    # gmst angle = a + b
    a = t_current*7.29211e-5 + earth_rotation_angle_offset
    b = t_current*5.14671e-11

    # here I get sin and cosine of GMST
    sin_theta = math.sin(a)*math.cos(b) + math.cos(a)*math.sin(b)
    cos_theta = math.cos(a)*math.cos(b) - math.sin(a)*math.sin(b)

    # instead of calling a RotZ function, I just do it all here
    return np.array([cos_theta*r_eci[0] + sin_theta*r_eci[1],
                    -sin_theta*r_eci[0] + cos_theta*r_eci[1],
                     r_eci[2] ])


def ECEF2ANG(r_ecef,r_station,earth):
    """Returns angle from vertical (90 - abs(el))"""
    return math.acos(np.numerical.sum(normalize(r_ecef - r_station)*normalize(r_station)))
# ----------- Scheduler --------------


class Propagator():

    def __init__(self,rv_eci,earth_rotation_angle_offset,t_epoch):

        # [r;v] in km, km/s
        self.rv_eci = 1*rv_eci

        # current earth rotation angl
        self.earth_rotation_angle_offset = earth_rotation_angle_offset

        # initial time that propagator was started
        self.t_epoch = t_epoch

        # store ecef location
        self.r_ecef = np.zeros(3)

        # constants
        self.earth = Earth()


    def step(self,dt):

        if float(dt) > 20.0:
            raise Exception("dt is too big (>20 seconds)")

        # send dynamics forward one s
        self.rv_eci = rk4_propagate(self.rv_eci,dt,self.earth)

    def get_r_ecef(self,t_current):
        self.r_ecef = ecef_from_eci(self.rv_eci,
        self.earth_rotation_angle_offset,t_current - self.t_epoch)


# ---------------------- Testing script --------------------------


#----------Communication from the ground-------------------
# communication from the ground tells us rv_eci, and earth_rotation_angle_offset
rv_eci = np.array([-1861.7015559490976, -6645.09702340011, 0.032941584155793194,
              -0.9555092604431692, 0.29391099680436356, 7.541418280028347])
earth_rotation_angle_offset = 4.273677081313352


#----------Initialize propagator---------------------------
# the spacecraft then initializes the propagator with this information, and
# takes note of the current time as t_epoch
t_epoch = 0 # hard coded now, but would me current on board time
propagator = Propagator(rv_eci, earth_rotation_angle_offset, t_epoch)

# step size for integrator
dt = 10


#----------Run propagator----------------------------------
t1 = time.monotonic()
for i in range(2000):
    propagator.get_r_ecef(dt*i)
    propagator.step(dt)

print(time.monotonic() - t1)

print(propagator.rv_eci)
print(propagator.r_ecef)