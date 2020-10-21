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
    Nx = 2
    Nu = 1
    N = utraj0.shape[1] + 1

    J = 0.0


    xtraj = [np.zeros(Nx) for _ in range(N)]
    xtraj[0] = x0
    utraj = [np.zeros(Nu) for _ in range(N - 1)]

    # initial cost of an all zeros guess
    J = (N - 1) * 0.5 * quad(Q, (x0 - xg).transpose())[0] + 0.5 * quad(
        Qf, (x0 - xg).transpose()
    )[0]


    S = np.zeros((Nx, Nx))
    s = np.zeros(Nx)
    K = np.zeros((Nu, Nx * (N - 1)))
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

            Ak, Bk = rkstep_jacobians(xtraj[k], utraj[k], dt)

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

        # Forward pass line search with new l and K
        #reset_list(xnew)
        #reset_list(unew)
        #xnew[0] = x0

        alpha = 1.0
        Jnew = J + 1
        while Jnew > J:
            Jnew = 0.0
            for k in range(0, N - 1):
                unew[k] = (
                    utraj[k]
                    - alpha * l[k]
                    - mul2(K[k], (xnew[k] - xtraj[k]).transpose())
                )
                (xnew[k + 1]) = rkstep_xdot(xnew[k], unew[k], dt)

                Jnew += cost(Q, R, (xnew[k] - xg).transpose(), unew[k].transpose())

            Jnew += 0.5 * quad(Qf, (xnew[N - 1] - xg).transpose())[0]
            alpha = 0.5 * alpha

        dJ = J - Jnew

        copy_list(xnew,xtraj)
        copy_list(unew,utraj)


        J = Jnew


        if dJ < 0.1:
            break

        print(ii)
        print(2 * alpha)
        print(J)

    return xtraj, utraj, K


def rkstep_xdot(x1, u0, dt):

    #xdot1 = dynamics_xdot(0, x0, u0)
    #xdot2 = dynamics_xdot(0, x0 + 0.5 * xdot1 * dt, u0)

    #x1 = x0 + dt * xdot2

    #return x1
    t = 0

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

def rkstep_jacobians(x1, u0, dt):

    # Define constants

        # return A, B
    Nx = 2

    # x1 = x0
    t = 0.0

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


def dynamics_xdot(t, x, u):
    Nx = 2

    m = 1.0
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81

    xdot = np.zeros(Nx)

    xdot[0] = x[1]
    xdot[1] = (u - m * g * lc * math.sin(x[0]) - b * x[1]) / I

    return xdot

def dynamics(t, x, u):
    Nx = 2

    m = 1.0
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81

    xdot = np.zeros(Nx)
    xdot[0] = x[1]
    xdot[1] = (u - m * g * lc * math.sin(x[0]) - b * x[1]) / I

    A = np.array([[0, 1], [-m * g * lc * math.cos(x[0]) / I, -b / I]])
    B = np.array([[0], [1 / I]])
    # dxdot = np.hstack((A, B))

    return xdot, A, B

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

    N = 75
    Nx = 2
    Nu = 1

    Qf = np.eye(Nx) * 30
    Q = np.eye(Nx) * 0.01
    R = np.eye(1) * 0.03

    x0 = np.zeros(2)
    xg = np.array([math.pi, 0])
    utraj0 = np.zeros((Nu, N - 1))

    dt = 0.1
    tol = 0.35

    xtraj, utraj, K = iLQRsimple_py(x0, xg, utraj0, Q, R, Qf, dt, tol)


main2()