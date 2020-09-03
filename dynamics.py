import time
import board

import ulab as np
import math

def mul2(a, b):
    """Multiply two arrays"""
    return np.linalg.dot(a, b)


def mul3(a, b, c):
    """Multiply three arrays"""
    return np.linalg.dot(mul2(a, b), c)


def pdot_from_w(p,w):
        np2 = np.linalg.dot(p,p.transpose())[0]
        return mul2((.25*(1+np2))*(np.eye(3) + 2*(mul2(hat(p),hat(p)) + hat(p))/(1+np2))      ,w.transpose())


def dcm_from_p(p):
    sp = hat(p)
    np2 = np.linalg.dot(p,p.transpose())[0]
    return np.eye(3) + (8*mul2(sp,sp) + 4*(1-np2)*sp)/(1+np2)**2


def hat(v):
    return np.array([[0,    -v[2],   v[1]  ],
                    [ v[2],     0,  -v[0]  ],
                    [-v[1],  v[0],     0   ]])

def norm(v):
    return math.sqrt(sum(v*v))


def sc_dynamics(t,x,m):

    # spacecraft properties
    J = np.array([[1.959e-4, 2016.333e-9, 269.176e-9],
           [2016.333e-9, 1.999e-4, 2318.659e-9],
           [269.176e-9, 2318.659e-9, 1.064e-4]])

    invJ = 1e3*np.array([[5.105190043647774,  -0.051357738055630,  -0.011796180015288],
      [-0.051357738055630,   5.004282688631113,  -0.108922940075563],
      [-0.011796180015288,  -0.108922940075563,   9.400899721840833]])

    max_moments = np.array([8.8e-3,1.373e-2,8.2e-3])


    # magnetic field in ECI
    B_eci = np.array([1.2,3.4,-5.6])


    # unpack state
    p = x[0:3]
    w = x[3:6]

    # MRP
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]

    # angular velocity
    wx = w[0]
    wy = w[1]
    wz = w[2]

    # magnetic field in the body frame
    B_body = mul2(dcm_from_p(p).transpose(),B_eci.transpose())

    # mrp kinematics
    pdot = pdot_from_w(p,w)

    # scale the magnetic moment with the tanh
    actual_m = max_moments * np.vector.tanh(m)

    # resulting torque on the spacecraft
    actual_tau = mul2(hat(actual_m),B_body)

    # euler dynamics
    wdot = mul2(invJ,actual_tau - mul2(hat(w),mul2(J,w.transpose())))

    # final state derivative
    xdot = np.array([pdot[0][0],pdot[1][0],pdot[2][0],wdot[0][0],wdot[1][0],wdot[2][0]])



    # dpdot_dp
    dpdot_dp = np.array([[ ((p1*wx)/2 + (p2*wy)/2 + (p3*wz)/2),        (wz/2 - (p2*wx)/2 + (p1*wy)/2),      ((p1*wz)/2 - (p3*wx)/2 - wy/2)],
          [((p2*wx)/2 - wz/2 - (p1*wy)/2), ((p1*wx)/2 + (p2*wy)/2 + (p3*wz)/2),      (wx/2 - (p3*wy)/2 + (p2*wz)/2)],
          [(wy/2 + (p3*wx)/2 - (p1*wz)/2),      ((p3*wy)/2 - wx/2 - (p2*wz)/2),   ((p1*wx)/2 + (p2*wy)/2 + (p3*wz)/2)]])

    # dpdot_dw
    dpdot_dw = np.array([[ (p1**2/4 - p2**2/4 - p3**2/4 + 1/4),                 ((p1*p2)/2 - p3/2),                 (p2/2 + (p1*p3)/2)],
                   [(p3/2 + (p1*p2)/2), (- p1**2/4 + p2**2/4 - p3**2/4 + 1/4),                 ((p2*p3)/2 - p1/2)],
                   [((p1*p3)/2 - p2/2),                 (p1/2 + (p2*p3)/2),  ( - p1**2/4 - p2**2/4 + p3**2/4 + 1/4)]])




    # norm(p)^2
    np2 = np.linalg.dot(p,p.transpose())[0]

    # dwdot_dp
    part1 = (1/((1+np2)**2))*(-8*mul2(hat(p),hat(B_eci)) - 8*hat((mul2(hat(p),B_eci.transpose())).transpose()) + 4*(1-np2)*hat(B_eci) + 8*mul3(hat(p),B_eci.transpose(),p))
    part2 = -4*mul2(  (8*mul3(hat(p),hat(p),B_eci.transpose()) - 4*(1-np2)*mul2(hat(p),B_eci.transpose()) )    ,p) /((1+np2)**3)
    dwdot_dp = mul3(invJ,  hat(actual_m),(part1+part2)  )

    # dwdot_dw
    dwdot_dw = mul2(invJ,  hat((mul2(J,w.transpose())).transpose()) -  mul2(hat(w),J)   )

    # A
    A = np.array([  [dpdot_dp[0,0][0], dpdot_dp[0,1][0], dpdot_dp[0,2][0], dpdot_dw[0,0][0], dpdot_dw[0,1][0], dpdot_dw[0,2][0]  ],
                    [dpdot_dp[1,0][0], dpdot_dp[1,1][0], dpdot_dp[1,2][0], dpdot_dw[1,0][0], dpdot_dw[1,1][0], dpdot_dw[1,2][0]  ],
                    [dpdot_dp[2,0][0], dpdot_dp[2,1][0], dpdot_dp[2,2][0], dpdot_dw[2,0][0], dpdot_dw[2,1][0], dpdot_dw[2,2][0]  ],
                    [dwdot_dp[0,0][0], dwdot_dp[0,1][0], dwdot_dp[0,2][0], dwdot_dw[0,0][0], dwdot_dw[0,1][0], dwdot_dw[0,2][0]  ],
                    [dwdot_dp[1,0][0], dwdot_dp[1,1][0], dwdot_dp[1,2][0], dwdot_dw[1,0][0], dwdot_dw[1,1][0], dwdot_dw[1,2][0]  ],
                    [dwdot_dp[2,0][0], dwdot_dp[2,1][0], dwdot_dp[2,2][0], dwdot_dw[2,0][0], dwdot_dw[2,1][0], dwdot_dw[2,2][0]  ],
                    ])


    # B stuff
    tauB_vec = -max_moments*np.array([ (np.vector.tanh(m[0])**2 -1 ), (np.vector.tanh(m[1])**2 -1 ), (np.vector.tanh(m[2])**2 -1 )  ])
    tauB = np.array([  [tauB_vec[0]  ,0,0], [0,tauB_vec[1] ,0],[0,0,tauB_vec[2] ]])
    B_bottom = mul3(invJ,hat(-B_body.transpose()),tauB)

    B = np.array([  [0,0,0],[0,0,0],[0,0,0],
                 [B_bottom[0,0][0],B_bottom[0,1][0],B_bottom[0,2][0]],
                 [B_bottom[1,0][0],B_bottom[1,1][0],B_bottom[1,2][0]],
                 [B_bottom[2,0][0],B_bottom[2,1][0],B_bottom[2,2][0]],
                ])

    return xdot, A, B


#p = np.array([1,2,8.3])
#w = np.array([4,3.4,-4.3])
x = np.array([1,2,8.3,4,3.4,-4.3])
m = np.array([.3,.6,-1.3])
t = 0.0

xdot, A, B = sc_dynamics(t,x,m)

print('xdot',xdot)
print('A',A)
print('B',B)
print('type',type(xdot[0]))