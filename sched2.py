import math
import ulab as np

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
	mu = 3.986004418E5 # specific gravitational parameter
	R = 6371.009 # radius in km
	J2 = 1.08262668E-3 # J2 value


# ----------- Dynamics --------------

def propagate(x,dt,Earth):
	# this function will utilize a 2-body + J2 dynamics model to deliver
	# the next state using a RK4 step
	# it requires a state of type [pos;vel] in km and km/s

	# generic rk4 method

	k1 = dt*dynamics(x,Earth)
	k2 = dt*dynamics(x+k1/2,Earth)
	k3 = dt*dynamics(x+k2/2,Earth)
	k4 = dt*dynamics(x+k3,Earth)

	return x + (1/6)*(k1+2*k2+2*k3+k4)

def norm(x):
    """norm of an array"""
    return math.sqrt(np.numerical.sum(x**2))

def normalize(x):
    """normalize an array (not in place)"""
    return x/norm(x)



def dynamics(x,Earth):

    # generate the position and velocity
    pos = x[0:3]
    vel = x[3:6]

    # first we find the acceleration from two body
    #pos_norm = np.vector.sqrt(np.numerical.sum(pos ** 2))
    pos_norm = norm(pos)
    accel_twobody = -Earth.mu * \
    (pos_norm ** -3) * pos

    # now we find the J2 acceleration

    accel_J2 = np.zeros(3)
    accel_J2[0] = -3/2 * Earth.J2 * Earth.mu * Earth.R ** 2\
    * pos_norm ** -5 \
    * pos[0] * (1 - 5 * pos[2] ** 2 / pos_norm ** 2)
    accel_J2[1] = -3/2 * Earth.J2 * Earth.mu * Earth.R ** 2\
    * pos_norm ** -5 \
    * pos[1] * (1 - 5 * pos[2] ** 2 / pos_norm ** 2)
    accel_J2[2] = -3/2 * Earth.J2 * Earth.mu * Earth.R ** 2\
    * pos_norm ** -5 \
    * pos[2] * (3 - 5 * pos[2] ** 2 / pos_norm ** 2)

    # add together accelerations
    # accel = accel_twobody + accel_J2
    accel = accel_twobody + accel_J2

    return np.array([vel[0],vel[1],vel[2],accel[0],accel[1],accel[2]])






def ECI2ECEF(r_eci,time):
    # this function takes in an ECI position and a J2000 time

    #find the number of days elapsed since Dec 1 2020
    #now determine the angle of GMST
    # days = (time % (24 * 60 * 60))/(24 * 60 * 60)
    #theta = (1.22719594 + 6.3003880978*days)
    #theta = (1.17719594 + 6.3003880978*days)
    tenday = (time % (861641)) % (86164)
    theta = 2*math.pi*(0.195315806040317 + 1.160576283564814e-5*(tenday))

    # sat state: 6 + timestamp + earth angle = 8 (32 bytes)

    #convert by doing a rotation
    r_ecef = np.linalg.dot(Rz(theta),r_eci.transpose())

    return np.array([r_ecef[0][0], r_ecef[1][0], r_ecef[2][0]])

def Rz(theta):
    # utility function to get the z rotation matrix
    # uses radians

    return np.array([[math.cos(theta),math.sin(theta),0],\
                    [-math.sin(theta),math.cos(theta),0],\
                    [0,               0,              1]])


def ECEF2ANG(r_ecef,r_station,earth):
    # takes in the ecef position and the lat long alt of station
    # station is r_ecef

    diff = r_ecef - r_station
    diff_n = normalize(diff)
    r_station_n = normalize(r_station)

    ang_dot = np.numerical.sum(diff_n*r_station_n)



    ang_from_vert = math.acos(ang_dot)

    return ang_from_vert


# ----------- Scheduler --------------

# We need two functions
# 1) Find elevation compared to lat/long on Earth of
# satellite given ECI position, time

# 2) Scheduler to find when certain threshholds have been
# passed over the horizon

class scheduler():

	def __init__(self,X,time,ground_stations):
		# take in the ECI pos/vel
		# the current JD
		# the lat, long of ground stations we want to view
		self.X = X
		self.time = time
		self.ground_stations = ground_stations

		# constants
		self.dt = 1 # seconds
		self.horizon = 6400 # seconds
		self.earth = Earth()
		# observation constraints
		self.min_ang = 80. # deg from vertical


	def generate(self):
		# first we need to extrapolate forward the pos/vel
		#earth = Earth()

		num_steps = int(self.horizon/self.dt)
		num_stations = len(self.ground_stations)

		# generate the act list and the flags to determine
		# ground passes
		self.act_list = activities()
		event_flags = [0] * num_stations
		obs_start = [0] * num_stations

		current_time = self.time
		current_X = self.X
		for i in range(num_steps):
			#print(current_X)
			# propagate forward a step
			X_next = propagate(current_X,self.dt,self.earth)
			t_next = current_time + self.dt

			# find the ecef position of the sat and az,el
			r_ecef = ECI2ECEF(X_next[0:3],t_next)

			# DEBUG
			#print(norm(X_next[0:3] - np.array([6928,0,0])))
			if i > (num_steps-10):
				print(X_next[0:3])

			if num_stations > 1:
				for j in range(num_stations):
					ang_from_vert = ECEF2ANG(r_ecef,self.ground_stations[j],self.earth)
					if (ang_from_vert*180/math.pi) < self.min_ang and event_flags[j] == 0:
						event_flags[j] = 1 #raise the flag
						obs_start[j] = t_next - 1

					if (ang_from_vert*180/math.pi) >= self.min_ang and event_flags[j] == 1:
						event_flags[j] = 0 #lower the flag

						# create the activity
						center = obs_start[j] + (t_next - 1 - obs_start[j]) / 2
						window = (t_next - 1 - obs_start[j]) / 2
						self.act_list.add_activity(center,window,j)

			if num_stations == 1:
				j = 0
				ang_from_vert = ECEF2ANG(r_ecef,self.ground_stations[j],self.earth)
				if (ang_from_vert*180/math.pi) < self.min_ang and event_flags[j] == 0:
					event_flags[j] = 1 #raise the flag
					obs_start[j] = t_next - 1

				if (ang_from_vert*180/math.pi) >= self.min_ang and event_flags[j] == 1:
					event_flags[j] = 0 #lower the flag

					# create the activity
					center = obs_start[j] + (t_next - 1 - obs_start[j]) / 2
					window = (t_next - 1 - obs_start[j]) / 2
					self.act_list.add_activity(center,window,j)

			# reassign variables
			current_X = 1*X_next
			current_time = 1*t_next


class activities():

	def __init__(self):
		self.act_num = 0
		self.centers = []
		self.windows = []
		self.gs_numbers = []

	def add_activity(self,center,window,gs_number):
		self.act_num += 1
		self.centers.append(center) #J2000 days
		self.windows.append(window) #days
		self.gs_numbers.append(gs_number)



# ----------- Testing script --------------

#earth = Earth()
X = np.array([6928,0,0,0,-.871817082027,7.5348948722])
# time = seconds since launch (Dec 1, 2020) (7640 days J200)
time = 0
# ground station (Stanford)
#ground_stations = np.array([[37.4241,-122.166],[37.4241,-75],[-30.5595,22.9375],[90,0]])
ground_stations = [np.array([-2696.93, -4288.3, 3850.56]), np.array([1311.14, -4893.24, 3850.56]),
                  np.array([5056.68, 2139.92, -3220.33]), np.array([0.00505484, 0.0, 6349.64])]
sched = scheduler(X,time,ground_stations)
sched.generate()
print("Ground station has been set at lat=37.4241, lon=-122.166")
for i in range(len(sched.act_list.centers)):
	print(sched.act_list.centers[i])
	# print("Access: {} | Center: {} | Duration: {}".format(i,\
	# 	sched.act_list.centers[i],sched.act_list.windows[i]*2))
	start_time = sched.act_list.centers[i]- sched.act_list.windows[i]
	stop_time = sched.act_list.centers[i]+ sched.act_list.windows[i]

	start_day = start_time // 86400
	start_hour = (start_time % 86400) // 3600
	start_min = int((start_time / 3600 - start_hour - (start_day * 24)) // (1/60))
	start_sec = start_time - start_min * 60 - start_hour * 3600 - start_day * 86400

	stop_day = stop_time // 86400
	stop_hour = (stop_time % 86400) // 3600
	stop_min = int((stop_time / 3600 - stop_hour - (stop_day * 24)) // (1/60))
	stop_sec = stop_time - stop_min * 60 - stop_hour * 3600 - stop_day * 86400

	print("Access: {} | GS: {} | Start Time: Dec {}, {}:{}:{} | Stop Time: Dec {}, {}:{}:{}".format(i,\
		sched.act_list.gs_numbers[i],\
		1+start_day,start_hour,start_min,start_sec\
		,1+stop_day,stop_hour,stop_min,stop_sec))