import ulab as np

# ECEF_id,(x,y,z)
ground_stations_dict = {
    'su'   :(0x01,np.array([-2701.514, -4293.583, 3853.493])), # 37.407371, -122.178022, 148 MSL
    'aws-w':(0x11,np.array([-2200.201, -3868.026, 4554.08])),  # 45.854611, -119.631972, 101 MSL
    'aws-e':(0x12,np.array([578.498,   -4851.11,  4087.038])), # 40.104278, -83.199556,  294 MSL
}


class ground_station_class:
    def __init__(self,gs_dict):
        self.id = [gs_dict[gs][0] for gs in gs_dict]
        self.ecef = [gs_dict[gs][1] for gs in gs_dict]
        self.visibility = [0 for gs in gs_dict]

#ground_stations = ground_station_class(ground_stations_dict)

ground_stations = ground_stations_dict

import ulab as np
# ECEF id,(x,y,z)
ground_stations = [
        (0x01,np.array([-2701.514, -4293.583, 3853.493])), # 37.407371, -122.178022, 148 MSL
        (0x11,np.array([-2200.201, -3868.026, 4554.080])), # 45.854611, -119.631972, 101 MSL
        (0x12,np.array([578.498,   -4851.11,  4087.038])), # 40.104278, -83.199556,  294 MSL
]