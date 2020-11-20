from math import sqrt,sin,cos
import ulab as np
from time import monotonic
import struct


'''
    --------- OG RESULTS -----------
time: 8.25
[[23490, 23730], [28860, 29550], [34620, 35130], [70680, 71130], [76230, 76950], [82020, 82410]]

'''
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
    return sqrt(np.numerical.sum(x**2))

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

def ecef_from_eci(p,r_eci,earth_rotation_angle_offset,t_since_epoch):
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
    sin_theta = sin(a)*cos(b) + cos(a)*sin(b)
    cos_theta = cos(a)*cos(b) - sin(a)*sin(b)

    # instead of calling a RotZ function, I just do it all here
    p.r_ecef[0] = cos_theta*r_eci[0] + sin_theta*r_eci[1]
    p.r_ecef[1] = -sin_theta*r_eci[0] + cos_theta*r_eci[1]
    p.r_ecef[2] = r_eci[2]


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

    # store ecef location
    r_ecef = np.zeros(3)

    # [r;v] in km, km/s
    rv_eci =  np.zeros(6)
    # current earth rotation angle
    earth_rotation_angle_offset=0

    # initial time that propagator was started
    t_epoch=0

    # constants
    earth = Earth()

    def __init__(self,gs,passes):
        """Initialize the propagator with the stuff we received from
        the ground station, as well as pre-allocate other parameters.
        """
        self.ground_stations=gs
        self.passes=passes

        # visible to ground station
        self.visible = False

        self.gs_id=0


    def step(self,dt):
        """Take a step in the propagator.

        Args:
            dt: time step (s) must be greater than 20

        Summary:
            This function integrates the orbital position and velocity
            with respect to a J2-only gravity model
        """

        # if float(dt) > 50.0:
        #     raise Exception("dt is too big (>50 seconds)")

        # send dynamics forward one s
        self.rv_eci = rk4_propagate(self.rv_eci,dt,self.earth)


    def check_visibility(self):
        """Check if the spacecraft is visible from a ground station.

        Summary: sets self.visible to False, then cycles through the
        ground stations and if one of them is visible, sets self.visible
        to True.
        for gs in self.ground_stations:
            if (elevation_dot_product(self.r_ecef,gs,self.earth) >  0.0):
        """

        for gs in self.ground_stations:
            if self.visible ^ (elevation_dot_product(self.r_ecef,self.ground_stations[gs][1],self.earth) >  0.0):
                self.visible ^= 1
                self.gs_id = self.ground_stations[gs][0]
                return True


    def run(self,dt,num_steps):
        n_passes = 0
        n_gs=range(len(self.ground_stations))
        max_passes=len(self.passes)
        buff=[0 for i in n_gs]
        for i in range(num_steps):
            # current time
            t_current = dt*i + self.t_epoch
            # update ecef location
            ecef_from_eci(self,self.rv_eci,self.earth_rotation_angle_offset,t_current - self.t_epoch)
            # check visibility from EACH ground station
            for gs_index in n_gs:
                if elevation_dot_product(self.r_ecef,self.ground_stations[gs_index][1],self.earth) >  0:
                    if not buff[gs_index]:
                        buff[gs_index]=t_current
                        # print('gs index:{}, id:{},\tbuff{}'.format(gs_index,self.ground_stations[gs_index][0],buff))
                elif buff[gs_index]:
                    self.passes[n_passes][0]=buff[gs_index]
                    self.passes[n_passes][1]=t_current
                    self.passes[n_passes][2]=self.ground_stations[gs_index][0]
                    buff[gs_index]=0
                    n_passes += 1
                    # print('gs index:{}, id:{},\tn_passes:{}'.format(gs_index,self.ground_stations[gs_index][0],n_passes))
                    if n_passes > max_passes:
                        break
            if n_passes >= max_passes:
                break
            else:
                # integrate to the next step
                self.step(dt)


    # ---------------------- Testing script --------------------------
    '''

    passes will be preallocated in state machine constructor
    passes=[[0,0] for i in range(6)]


    --------- Initialize propagator---------------------------
    the spacecraft then initializes the propagator with this information, and
    propagator = Propagator()

    --------- Communication from the ground-------------------
    communication from the ground tells us rv_eci, and earth_rotation_angle_offset
    rv_eci = np.array([-1861.7015559490976, -6645.09702340011, 0.032941584155793194,
                  -0.9555092604431692, 0.29391099680436356, 7.541418280028347])
    earth_rotation_angle_offset = 4.273677081313352
    '''

    # def debug_scheduler(self,gs_msg):
    def debug_scheduler(self,gs_msg=None):
        if gs_msg is None:
            print('dummy eci')
            gs_msg = b'\xc0\x9d\x16\xced\xae\xc6\xc1\xc0\xb9\xf5\x18\xd6\x86\x8ak?\xa0\xdd\xb8%AP\x02\xbf\xee\x93\x88(\x14\x1d&?\xd2\xcfp\x11\xcdh{@\x1e*i\x8d\xb8\xb6\x9a'
            gs_msg += b'@\x11\x18>\xce\x07\x9fP'

        #---------- Update propagator---------------------------
        '''
            from ground station message:
                rv_eci: state [r (km);v (km/s)] at t_epoch
                earth_rotation_angle_offset: GMST at t_epoch (radian)

            t_epoch: on board satellite time when propagator is initialized (s)
        '''
        for i in range(6):
            self.rv_eci[i]=struct.unpack_from('>d',gs_msg,offset=i*8)[0]
        self.earth_rotation_angle_offset=struct.unpack_from('>d',gs_msg,offset=-8)[0]
        self.t_epoch=int(monotonic())

        #---------- Run propagator---------------------------
        print('starting time',self.t_epoch)
        self.run(dt=30,num_steps=2880)
        print('total time',(monotonic()-self.t_epoch))
        for i in self.passes:
            print('{},{},{}'.format(i[0],i[1],i[2]))