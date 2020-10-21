import time
import board
import math
import ulab as np


def mul2(a, b):
    """Multiply two arrays"""
    return np.linalg.dot(a, b)


def mul3(a, b, c):
    """Multiply three arrays"""
    return np.linalg.dot(mul2(a, b), c)


def mul4(a, b, c, d):
    """Multiply four arrays"""
    return np.linalg.dot(mul3(a, b, c), d)


def mul5(a, b, c, d, e):
    """Multiply five arrays"""
    return np.linalg.dot(mul4(a, b, c, d), e)


def quad(Q, x):
    """Quadratic x'*Q*x"""
    return mul3(x.transpose(), Q, x)


def cost(Q, R, x, u):
    """Quadratic cost function .5*x'Qx + .5*u'Ru"""
    return (0.5 * quad(Q, x) + 0.5 * quad(R, u))[0]

def reset_list(input_list):
    n = len(input_list[0])
    for i in range(len(input_list)):
        input_list[i] = np.zeros(n)

def copy_list(parent, child):
    n = len(parent[0])
    for i in range(len(parent)):
        child[i] = parent[i]


def iLQRsimple_py(x0, xg, utraj0, Q, R, Qf, dt, tol):
    """Simple iLQR with no initial guess"""
    Nx = 6
    Nu = 3
    N = utraj0.shape[1] + 1

    J = 0.0

    # change to edit in place
    xtraj = [1*x0 for _ in range(N)]
    utraj = [np.zeros(Nu) for _ in range(N - 1)]

    # initial cost of an all zeros guess
    J = (N - 1) * 0.5 * quad(Q, (x0 - xg).transpose())[0] + 0.5 * quad(
        Qf, (x0 - xg).transpose()
    )[0]

    print(N)
    print(J)

    S = np.zeros((Nx, Nx))
    s = np.zeros(Nx)
    #K = np.zeros((Nu, Nx * (N - 1)))
    K = [np.zeros((Nu, Nx)) for _ in range(N - 1)]
    l = [np.zeros(Nu) for _ in range(N - 1)]
    xnew = [np.zeros(Nx) for _ in range(N)]
    xnew[0] = x0
    unew = [np.zeros(Nu) for _ in range(N - 1)]

    count = 0
    for ii in range(50):

        count += 1

        S = Qf
        s = mul2(Qf, (xtraj[N - 1] - xg).transpose())

        # Backward pass
        for k in range(N - 2, -1, -1):

            # Calculate cost gradients for this time step
            q = mul2(Q, (xtraj[k] - xg).transpose())
            r = mul2(R, utraj[k].transpose())

            Ak, Bk = rkstep_jacobians(k*dt,xtraj[k], utraj[k], dt)

            # Calculate l and K
            invLH = np.linalg.inv(R + mul3(Bk.transpose(), S, Bk))

            l[k] = mul2(invLH, (r + mul2(Bk.transpose(), s)))
            K[k] = mul2(invLH, mul3(Bk.transpose(), S, Ak))

            # Calculate new S and s

            Snew = Q + mul3(K[k].transpose(), R, K[k]) + quad(S, (Ak - mul2(Bk, K[k])))

            snew = (
                q
                - mul2(K[k].transpose(), r)
                + mul3(K[k].transpose(), R, l[k])
                + mul2((Ak - mul2(Bk, K[k])).transpose(), (s - mul3(S, Bk, l[k])))
            )
            S = 1.0*Snew
            s = 1.0*snew


        alpha = 1.0
        Jnew = J + 1
        #while Jnew > J:
        for iii in range(10):
            Jnew = 0.0

            # forward rollout with new control
            for k in range(0, N - 1):


                unew[k] = (
                    utraj[k]
                    -(alpha * l[k]).transpose()
                    - (mul2(K[k], (xnew[k] - xtraj[k]).transpose())).transpose()
                )
                (xnew[k + 1]) = rkstep_xdot(k*dt,xnew[k], unew[k], dt)

                Jnew += cost(Q, R, (xnew[k] - xg).transpose(), unew[k].transpose())

            # terminal cost
            Jnew += 0.5 * quad(Qf, (xnew[N - 1] - xg).transpose())[0]

            if Jnew<J:
                break
            else:
                alpha = 0.5 * alpha


        dJ = J - Jnew

        copy_list(xnew,xtraj)
        copy_list(unew,utraj)


        J = Jnew


        if dJ < 0.01:
            break

        print(ii)
        print(alpha)
        print(J)

    return xtraj, utraj, K


def rkstep_xdot(t,x1, u0, dt):

    #xdot1 = dynamics_xdot(0, x0, u0)
    #xdot2 = dynamics_xdot(0, x0 + 0.5 * xdot1 * dt, u0)

    #x1 = x0 + dt * xdot2

    #return x1
    #t = 0

    xdot1 = dynamics_xdot(t, x1, u0)
    k1 = xdot1 * dt

    x2 = x1 + 0.5 * k1
    xdot2 = dynamics_xdot(t + dt * 0.5, x2, u0)
    k2 = dt * xdot2

    x3 = x1 + 0.5 * k2
    xdot3 = dynamics_xdot(t + dt * 0.5, x3, u0)
    k3 = dt * xdot3

    x4 = x1 + k3
    xdot4 = dynamics_xdot(t + dt, x4, u0)
    k4 = dt * xdot4

    return x1 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def rkstep_jacobians(t,x1, u0, dt):

    # Define constants
    Nx = 6

    # x1 = x0
    #t = 0.0

    xdot1, A1, B1 = dynamics(t, x1, u0)
    k1 = xdot1 * dt

    x2 = x1 + 0.5 * k1
    xdot2, A2, B2 = dynamics(t + dt * 0.5, x2, u0)
    k2 = dt * xdot2

    x3 = x1 + 0.5 * k2
    xdot3, A3, B3 = dynamics(t + dt * 0.5, x3, u0)
    k3 = dt * xdot3

    x4 = x1 + k3
    xdot4, A4, B4 = dynamics(t + dt, x4, u0)
    # k4 = dt * xdot4

    # x_tp1 = x1 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # RK4 method
    # A_d
    dk1_dx1 = dt * A1
    dx2_dx1 = np.eye(Nx) + 0.5 * dk1_dx1
    dk2_dx1 = dt * mul2(A2, dx2_dx1)
    dx3_dx1 = np.eye(Nx) + 0.5 * dk2_dx1
    dk3_dx1 = dt * mul2(A3, dx3_dx1)
    dx4_dx1 = np.eye(Nx) + dk3_dx1
    dk4_dx1 = dt * mul2(A4, dx4_dx1)
    A_d = np.eye(Nx) + (1 / 6) * (dk1_dx1 + 2 * dk2_dx1 + 2 * dk3_dx1 + dk4_dx1)

    # B_d
    dk1_du = dt * B1
    dx2_du = 0.5 * dk1_du
    dk2_du = dt * mul2(A2, dx2_du) + dt * B2
    dx3_du = 0.5 * dk2_du
    dk3_du = dt * mul2(A3, dx3_du) + dt * B3
    dx4_du = 1.0 * dk3_du
    dk4_du = dt * mul2(A4, dx4_du) + dt * B4
    B_d = (1 / 6) * (dk1_du + 2 * dk2_du + 2 * dk3_du + dk4_du)

    return A_d, B_d





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


def dynamics(t,x,m):

    # spacecraft properties
    J = np.array([[1.959e-4, 2016.333e-9, 269.176e-9],
           [2016.333e-9, 1.999e-4, 2318.659e-9],
           [269.176e-9, 2318.659e-9, 1.064e-4]])

    invJ = 1e3*np.array([[5.105190043647774,  -0.051357738055630,  -0.011796180015288],
      [-0.051357738055630,   5.004282688631113,  -0.108922940075563],
      [-0.011796180015288,  -0.108922940075563,   9.400899721840833]])

    max_moments = np.array([8.8e-3,1.373e-2,8.2e-3])


    # magnetic field in ECI
    T = 50*60
    B_eci = 3e-5*np.array([math.sin(2*math.pi*t/T + math.pi/2),math.sin(2*math.pi*t/T - math.pi/3),math.sin(2*math.pi*t/T + 3*math.pi/2)])


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

def dynamics_xdot(t,x,m):

    # spacecraft properties
    J = np.array([[1.959e-4, 2016.333e-9, 269.176e-9],
           [2016.333e-9, 1.999e-4, 2318.659e-9],
           [269.176e-9, 2318.659e-9, 1.064e-4]])

    invJ = 1e3*np.array([[5.105190043647774,  -0.051357738055630,  -0.011796180015288],
      [-0.051357738055630,   5.004282688631113,  -0.108922940075563],
      [-0.011796180015288,  -0.108922940075563,   9.400899721840833]])

    max_moments = np.array([8.8e-3,1.373e-2,8.2e-3])


    # magnetic field in ECI
    T = 50*60
    B_eci = 3e-5*np.array([math.sin(2*math.pi*t/T + math.pi/2),math.sin(2*math.pi*t/T - math.pi/3),math.sin(2*math.pi*t/T + 3*math.pi/2)])


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

    return xdot
#def dynamics_jacobians(t, x, u):
#    Nx = 2
#
#    m = 1.0
#    b = 0.1
#    lc = 0.5
#    I = 0.25
#    g = 9.81

#    A = np.array([[0, 1], [-m * g * lc * math.cos(x[0]) / I, -b / I]])
#    B = np.array([[0], [1 / I]])

#    return A, B


# Test the algorithm
def main2():

    N = 25
    Nx = 6
    Nu = 3

    qp = 1
    qw = .01
    Q = np.array([[qp, 0.0, 0.0, 0.0, 0.0, 0.0],
	 [0.0, qp, 0.0, 0.0, 0.0, 0.0],
	 [0.0, 0.0, qp, 0.0, 0.0, 0.0],
	 [0.0, 0.0, 0.0, qw, 0.0, 0.0],
	 [0.0, 0.0, 0.0, 0.0, qw, 0.0],
	 [0.0, 0.0, 0.0, 0.0, 0.0, qw]])
    qpf = 100.0
    qwf = 10*qpf
    Qf = np.array([[qpf, 0.0, 0.0, 0.0, 0.0, 0.0],
	 [0.0, qpf, 0.0, 0.0, 0.0, 0.0],
	 [0.0, 0.0, qpf, 0.0, 0.0, 0.0],
	 [0.0, 0.0, 0.0, qwf, 0.0, 0.0],
	 [0.0, 0.0, 0.0, 0.0, qwf, 0.0],
	 [0.0, 0.0, 0.0, 0.0, 0.0, qwf]])

    R = .1*np.eye(3)
    #x0 = np.array([0.07500539420980451, 0.7678669790425526, 0.3299131467179552, 0.0, 0.0, 0.0])
    x0 = np.array([-0.274710632045161, 0.6780834943980876, 0.4108832368302142, 0.0, 0.0, 0.0])
    xg = np.array([0,0,0,0,0,0.0])
    utraj0 = np.zeros((Nu, N - 1))

    dt = 12.0
    tol = 0.5

    xtraj, utraj, K = iLQRsimple_py(x0, xg, utraj0, Q, R, Qf, dt, tol)

    print(xtraj[-1])


main2()