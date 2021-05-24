import numpy as np


def forward_dynamics(force, dq, link_masses, joint_damping, gravity_acc):
    """
    Computes, using the Newton’s second law, the accelerations (rate of change of velocities) of the coordinates
    in terms of the inertia and forces applied on the skeleton as a set of rigid-bodies. (for each joint)
    Given the trajectory of joint velocities and the forces.

    When working with numpy arrays, make use of broadcasting instead of using a for loop. E.g, it is possible to
    sum two arrays with different dimensions.
    a = np.array([1., 2.]), b = np.array([[1., 1.], [1., 1.]]), (a+b).shape = (2,2)
    As a tip, first construct the loop version and only then move to numpy broadcasting.

    The general equation for a robot's dynamics is given by
    f = M(q) @ ddq + C(q, dq) + G(q),

    q are the generalized joint coordinates, dq the joint velocities,
    ddq the joint accelerations, M mass matrix, C the Centrifugal and Coriolis forces,
    G(q) the gravity force.

    n_links: number of links
    n_joints: number of joints
    N: length of the trajectory

    inputs
        force: np.float64, shape (n_joints, N) -> trajectory of forces to apply to each joint
        dq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint velocities
        link_masses: np.ndarray, np.float64, shape (n_links,) -> mass of each link
        joint_damping: np.ndarray, np.float64, shape (n_joints,) -> damping coefficient for each joint
        gravity_acc: float -> gravity acceleration constant
    returns
        ddq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint accelerations

    """
    # Safe way to convert to numpy arrays
    force = np.asarray(force)
    dq = np.asarray(dq)

    n_links = link_masses.shape[0]
    n_joints = dq.shape[0]
    N = dq.shape[1]

    # TODO
    # fill all the fields with 'None'
    m2, m3, m4 = link_masses[1:]
    d1, d2, d3 = joint_damping
    # Mass matrix
    M = np.array([[m2+m3+m4, 0, 0],
                  [0, m3+m4, 0],
                  [0, 0, m4]])

    # Damping matrix
    C = np.array([[d1, 0, 0],
                  [0, d2, 0],
                  [0, 0, d3]])

    # Gravity
    G = gravity_acc * np.array([[m2+m3+m4, m3+m4, m4]] * N).T

    # Compute the acceleration
    ddq = np.matmul(np.linalg.inv(M), force - np.matmul(C, dq) - G)

    return ddq


def cyclic_analytical(sim_time_vect, amplitude, omega):
    """
    Computes the analytical velocity and acceleration trajectories for each joint, for the given vector
    of simulation times sim_time_vect. Each entry of the joint vector, as a function of time, is given by

    qi(t) = a * (1 - cos(omega*t))

    N: length of the trajectory == len(sim_time_vect)
    n_joints: number of joints

    inputs
        sim_time_vect: np.ndarray, np.float64, shape (N,) -> vector with simulation times
        amplitude: int or float -> joint amplitude
        omega: int or float -> angular frequency
    returns
        dq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint velocities
        ddq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint accelerations

    """
    # TODO
    # fill all the fields with 'None'
    dq = np.array([[omega * np.sin(omega*t)]*3 for t in sim_time_vect]).T
    ddq = np.array([[- omega**2 * np.cos(omega*t)]*3 for t in sim_time_vect]).T
    return dq, ddq


def central_dq(X, dt):
    """
    Computes central difference quotient for the time series in X.
    https://de.wikipedia.org/wiki/Differenzenquotient#Zentraler_Differenzenquotient

    For time step 0 and the last time step,
    the central difference quotient is 0 for all dimensions.

    xdim: X.shape[0]
    N: X.shape[1], length of time series

    inputs
        X: np.ndarray, np.float64, shape (xdim, N) -> time sequence of values
        dt: float -> time step
    returns
        dx: np.ndarray, np.float64, shape (xdim, N) -> central difference quotient

    """
    # TODO
    # fill all the fields with 'None'
    dx = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(1, X.shape[1] - 1):
            dx[r][c] = (X[r][c + 1] - X[r][c - 1]) / (2 * dt)

    return dx


def cyclic_numerical(q, dt):
    """
    Computes the numerical approximation of the velocity and acceleration trajectories for each joint,
    using central difference quotients (central differences).

    N: length of the trajectory
    n_joints: number of joints

    inputs
        q: np.ndarray, np.float64, shape (n_joints, N) -> joint trajectory
        dt: float -> time step for numerical derivative computations
    returns
        dq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint velocities
        ddq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint accelerations

    """
    # TODO
    # fill all the fields with 'None'
    dq = central_dq(q, dt)
    ddq = central_dq(dq, dt)
    return dq, ddq


def fixed_point_iteration_with_relaxation(x, iters, eps, f, df):
    """
    Computes a root of the function f using fixed point iteration method with optimal relaxation matrix.

    Stop the iterative procedure if:
        - The euclidean norm between consecutive points is less than eps
        - The number of iterations (iters) was reached

    xdim: x.shape[0]

    inputs
        x: np.ndarray, np.float64, shape (xdim, 1) -> starting vector
        iters: int -> maximum number of iterations
        eps: float -> tolerance
        f: callable function -> f:R^xdim -> R^xdim. f(x0) gives f at x0
        df: callable function -> df:R^xdim -> R^(xdim, xdim). df(x0) computes the jacobian of f at x0
    returns
        g_x: np.ndarray, np.float64, shape (xdim, 1) -> the root of f

    """
    x = x.copy()
    g_x = x.copy()
    # TODO
    # Computes a root of the function f using fixed point iteration
    """
    write your code here
    """
    for i in range(iters):
        relax = - np.linalg.inv(df(x))
        g_x = x + np.matmul(relax, f(x))
        if np.linalg.norm(g_x - x) < eps:
            break
        else:
            x = g_x

    return g_x


def optimize_vmax(q_target, beta, vmax0, iters, eps, link_masses, joint_damping, gravity_acc):
    """
    Computes the joint velocities that maximize the specified cost function.

    n_joints: number of joints

    inputs
        q_target: np.ndarray, np.float64, shape (n_joints, 1) -> joint target
        beta: np.ndarray, np.float64, shape (n_joints, 1) -> weighting parameter
        vmax0: np.ndarray, np.float64, shape (n_joints, 1) -> initial guess for the joint velocity
        iters: int -> maximum number of iterations for the optimization routine
        eps: float -> tolerance for the optimization routine
        link_masses: np.ndarray, np.float64, shape (n_links,) -> mass of each link
        joint_damping: np.ndarray, np.float64, shape (n_joints,) -> damping coefficient for each joint
        gravity_acc: float -> gravity acceleration constant
    returns
        vmax_opt: np.ndarray, np.float64, shape (n_joints, 1) -> the joint velocity that maximizes the cost function

    """

    # TODO
    # fill all the fields with 'None'
    def dJ(x):
        """
        Jacobian Matrix of the scalar function J, TRANSPOSED.
        The Jacobian of a scalar function is by definition a row vector, but here we return it as a column vector.

        inputs
            x: np.ndarray, np.float64, shape (n_joints, 1)
        returns
            jac: np.ndarray, np.float64, shape (n_joints, 1) -> the Jacobian (TRANSPOSED) of J at x

        """

        n_joints = x.shape[0]
        jac =   - x**2 / beta
        return jac

    def ddJ(x):
        """
        Hessian Matrix of the scalar function J.

        inputs
            x: np.ndarray, np.float64, shape (n_joints, 1)
        returns
            hess: np.ndarray, np.float64, shape (n_joints, n_joints) -> the Hessian of J at x

        """
        n_joints = x.shape[0]
        d1, d2, d3 = joint_damping
        hess =  np.array([[d1**2 * np.e ** x[0], 0 ,0],
                          [0, d2**2 * np.e ** x[1], 0],
                          [0,  0, d3**2 * np.e ** x[2]]])
        return hess

    # Find the optimal velocity with fixed point iteration with relaxation method
    vmax_opt = None

    return vmax_opt