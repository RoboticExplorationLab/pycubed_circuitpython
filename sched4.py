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
    #
    #w_earth_pt1 = 7.29211e-5
    #w_earth_pt2 = 5.14671e-11

    a = t_current*7.29211e-5 + earth_rotation_angle_offset
    b = t_current*5.14671e-11
    sin_theta = math.sin(a)*math.cos(b) + math.cos(a)*math.sin(b)
    cos_theta = math.cos(a)*math.cos(b) - math.sin(a)*math.sin(b)

    return np.array([cos_theta*r_eci[0] + sin_theta*r_eci[1],
                    -sin_theta*r_eci[0] + cos_theta*r_eci[1],
                     r_eci[2] ])


def ECEF2ANG(r_ecef,r_station,earth):
    """Returns angle from vertical (90 - abs(el))"""
    return math.acos(np.numerical.sum(normalize(r_ecef - r_station)*normalize(r_station)))
# ----------- Scheduler --------------


class Propagator():

    def __init__(self,rv_ecef,t_epoch):

        # [r;v] in km, km/s
        self.rv_ecef = rv_ecef

        # initial time that propagator was started
        self.t_epoch = t_epoch

        # store the current time in t_current
        #self.t_current = t_epoch

        # store ecef location
        self.r_ecef = np.zeros(3)

        # store previous time propagator was called
        #self.previous_time = t_epoch


        # constants
        self.earth = Earth()


    def step_in_place(self,dt):


        if float(dt) > 20.0:
            raise Exception("dt is too big (>20 seconds)")

        # set new time to old time
        #self.previous_time = self.current_time

        # send dynamics forward one s
        self.rv_ecef = rk4_propagate(self.rv_ecef,dt,self.earth)

        # reset timing
        #self.t_current += dt


# ----------- Testing script --------------

#earth = Earth()
rv_ecef = np.array([-4577.35003,-5196.57233,-0.002286203,
              -0.733516100,0.67309941644,7.51530773])

# initialize propagator

earth_rotation_angle_offset = 3.7794796
t_0 = 3622269
propagator = Propagator(rv_ecef,t_0)

dt = 10.0

t1 = time.monotonic()
for i in range(2000):
    propagator.r_ecef = ecef_from_eci(propagator.rv_ecef,earth_rotation_angle_offset,dt*i)
    propagator.step_in_place(dt)

print(time.monotonic() - t1)

print(propagator.rv_ecef)
print(propagator.r_ecef)
#print(propagator.t_current)


#print(ecef_from_eci(X,earth_rotation_angle_offset,123456))