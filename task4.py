import numpy as np
import pymunk
import pyglet
from pymunk.pyglet_util import DrawOptions
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
to install:
pip install pyglet
pip install pymunk
pip install matplotlib
pip install scipy

* without damping:
the constant parameter D = 0

* with damping:
the constant parameter D > 0

you can also drag the body with mouse click on the window 

you can also close the window before the complete step's number reached 
'''


def get_group_num():
    # replace None with your group number
    group_num = 150
    return str(group_num)


# constants
SIMULATION_TIME = 5  # sec
DT = 1/100
STEPS = int(SIMULATION_TIME / DT)

M = 1.
K = 5.
D = 1.

D_UNDAMPED = 0
D_CRITIAL = np.sqrt(4*M*K)  # delta = w0 => d = sqrt(4*M*K) , delta = d/2m and w0 = sqrt(k/m)
D_UNDER = D_CRITIAL - 3. # between D_UNDAMPED and D_CRITIAL
D_OVER = D_CRITIAL + 5.  # greater than D_CRITIAL
GRAVITY = 0.  # or 9.81

REST_SPRING_SIM = 100.  # For visualization only

INITIAL_POS = 10.  # safe check: if gravity=0. and initial position=0., nothing should move
INITIAL_VEL = 0.
EPS = 1e-6
ITER = 200

FRICTION = 0
ANIMATE = True

# ....................... pyglet setting .......................
# set pyglet tools
window = pyglet.window.Window(500, 500, "damped spring–mass system", resizable=False)

space = pymunk.Space()
space.gravity = 0, -GRAVITY
options = DrawOptions()
batch = pyglet.graphics.Batch()

fixer = pymunk.Body(200, 0, pymunk.Body.STATIC)
fixer.position = window.width//2, window.height - 100
poly_fixer = pymunk.Poly.create_box(fixer, size=(30, 30))
poly_fixer.id = 1
poly_fixer.friction = FRICTION * 2.0
space.add(fixer, poly_fixer)

# body
body = pymunk.Body(M, 1, pymunk.Body.DYNAMIC)
body.position = fixer.position.x, fixer.position.y - REST_SPRING_SIM - INITIAL_POS
body.velocity = 0., -INITIAL_VEL
circle = pymunk.Circle(body, 15)
circle.friction = FRICTION * 2.7
space.add(body, circle)

spring = pymunk.DampedSpring(
    fixer, body, (0, 0), (0, 0), rest_length=REST_SPRING_SIM, stiffness=K, damping=D
)

space.add(spring)
t0 = 0.
t = list()
t.append(t0)
time_steps = list()
pos_simulator = list()
pos_simulator.append(circle.body.position.y)
vel_simulator = list()
vel_simulator.append(circle.body.velocity.y)


@window.event()
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    body.position = window.width//2, y


# draw the shapes
@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)
    batch.draw()


# simulator step update
def update(dt):
    # update step
    new_t = t[len(t)-1] + dt
    time_steps.append(dt)
    t.append(new_t)
    pos_simulator.append(circle.body.position.y)
    vel_simulator.append(circle.body.velocity.y)

    space.step(dt)

    if len(t) == STEPS:
        pyglet.app.exit()

# ....................... end pyglet setting .......................


class PointMass:
    # m*d(dx/dt)/dt = -K*x -D*dx/dt + m*g

    def __init__(self, m=1., k=1., d=1., g=9.81):
        self.z0 = np.array([INITIAL_POS, INITIAL_VEL]).reshape((-1, 1))

        self.m = m
        self.K = k
        self.D = d
        self.g = g

    def initial_conds(self):
        return self.z0

    def diff_eq(self, z):
        # z = [x, dx/dt]
        # dz/dt = diff_eq(z)
        return np.array([z[1], -1/self.m*(self.K*z[0] + self.D*z[1]) + self.g]).reshape((-1, 1))

    def diff_eq_analytical_sol(self, t_sim):
        # analytical solution for the diff eq
        # return position, velocity, for n steps

        def dgl_first_order(x0, t, d, k, m, g):
            x, dx = x0[0], x0[1]
            dp = - (k/m) * x - (d/m) * dx + g
            return [dx, dp]

        # initial state:
        x0 = [INITIAL_POS, INITIAL_VEL]
        t = np.linspace(0, t_sim[len(t_sim)-1], STEPS)

        # solving the ODE with different values of the damping ratio
        # numerical using our given D
        #print(f'----- ANALYTICAL -----')
        analytical = odeint(dgl_first_order, x0, t, args=(D, K, M, GRAVITY))
        # undamped
        #print(f'----- UNDAMPED-----')
        undamped = odeint(dgl_first_order, x0, t, args=(D_UNDAMPED, K, M, GRAVITY))
        # under damped
        #print(f'----- UNDER DAMPED-----')
        under_damped = odeint(dgl_first_order, x0, t, args=(D_UNDER, K, M, GRAVITY))
        # critial damping
        #print(f'----- CRITIAL DAMPED-----')
        critial_damping = odeint(dgl_first_order, x0, t, args=(D_CRITIAL, K, M, GRAVITY))
        # over damped
        #print(f'----- OVER DAMPED-----')
        over_damped = odeint(dgl_first_order, x0, t, args=(D_OVER, K, M, GRAVITY))

        return analytical, undamped, under_damped, critial_damping, over_damped


def forward_dq(f, x, d):
    """
    -----------------------------------------------------------------
            This function calculates the Jacobian at x
            using the forward differential method
    -----------------------------------------------------------------
    m: Dimensions of the Function Output
    n: Dimensions of the Function Input
    Inputs
        f       f_handle             Function handle with the function to
                                     be derived
                                     np.ndarray, np.float64, shape (m, 1) ->
                                     np.ndarray, np.float64, shape (m, 1)
        x       np.ndarray((n, 1))   Position at which we calulate the Jacobian
        d       np.ndarray((n, 1))   Step size of the forward difference
    ------------------------------------------------------------------------
    Outpus
        Jx      np.ndarray((m, n))   The Jacobian of f at position x
    ------------------------------------------------------------------------
    """
    # TODO implement your code here
    # subtask 2
    Jx = np.zeros((f(x).shape[0],x.shape[0]),dtype=float)

    for j in range(0, f(x).shape[0]):
        for k in range(0, x.shape[0]):
            v = x.copy()
            v[k] = v[k] + d[k]
            Jx[j][k] = (f(v) - f(x))[j] / d[k]

    return Jx



def newton(x0, iters, eps, f, df):
    """
    Computes a root of the function f using Newton's method.
    https://en.wikipedia.org/wiki/Newton%27s_method

    Stop the iterative procedure if:
        - The euclidean norm between consecutive points is less than eps
        - The number of iterations (iters) was reached

    xdim: x0.shape[0]

    inputs
        x0: np.ndarray, np.float64, shape (xdim, 1) -> starting vector
        iters: int -> maximum number of iterations
        eps: float -> tolerance
        f: callable function -> f:R^xdim -> R^xdim. f(x0) gives f at x0
        df: callable function -> df:R^xdim -> R^(xdim, xdim). df(x0) computes the jacobian of f at x0
    returns
        x1: np.ndarray, np.float64, shape (xdim, 1) -> the root of f

    """
    x0 = x0.copy()
    x1 = x0.copy()
    for k in range(iters):
        x1 = x0 - np.linalg.inv(df(x0)) @ f(x0)

        # Stop criteria
        if np.linalg.norm(x1 - x0) < eps:
            return x1
        x0 = x1

    return x1


def expl_euler(f_sys, z0, dt, n):
    """
    ------------------------------------------------------------------------
            Compute the simulation steps using the Explicit Euler Method
    ------------------------------------------------------------------------
    Inputs
        f_sys   np.ndarray((N, 1))   system function of ODE formulation for
                                     a spring–damping system
        z_0     np.ndarray((N, 1))   the initial state
        dt      float                the time step
        n       int                  the number of steps
    ------------------------------------------------------------------------
    Outputs
        z   np.ndarray((N, n))   the next states x_{k_i}: i= 0 -> n
        (inc. k_0)
    ------------------------------------------------------------------------
    """
    # TODO implement your code here
    # subtask 1

    p = z0.shape[0]
    z = np.zeros((p, n), dtype=float)

    for s in range(0, p):
        z[s][0] = z0[s]

    for x in range(1, n):
        g = np.zeros((p, 1))
        for f in range(0, p):
            g[f] = z[f][x-1]
        b = g + dt*f_sys(g)
        for l in range(0, p):
            z[l][x] = b[l]

    return z


def impl_euler(f_sys, z0, dt, n):
    """
    ------------------------------------------------------------------------
            Compute the simulation steps using the Implicit Euler Method
    ------------------------------------------------------------------------
    Inputs
        f_sys   np.ndarray((N, 1))   system function of ODE formulation for
                                     a spring–damping system
        z_0     np.ndarray((N, 1))   the initial state
        dt      float                the time step
        n       int                  the number of steps
    ------------------------------------------------------------------------
    Outputs
        z   np.ndarray((N, n))   the next states x_{k_i}: i= 0 -> n
        (inc. k_0)
    ------------------------------------------------------------------------
    """
    # TODO implement your code here
    # subtask 3
    p = z0.shape[0]
    d = np.zeros((p, 1), dtype=float)
    for u in range(0, p):
        d[u] = dt

    def F(xk):
        Fxk = z0 + dt * f_sys(xk) - xk
        return Fxk

    def dF(x):
        return forward_dq(F, x, d)

    z = np.zeros((p, n))

    for s in range(0, p):
        z[s][0] = z0[s]

    for t in range(1, n):
        g = np.zeros((p, 1))
        for h in range(0, p):
            g[h] = z[h][t - 1]
        b = newton(g, ITER, EPS, F, dF)
        for l in range(0, p):
            z[l][t] = b[l]

    return z


def heun(f_sys, z0, dt, n):
    """
    ------------------------------------------------------------------------
            Compute the simulation steps using the Heun Method
    ------------------------------------------------------------------------
    Inputs
        f_sys   np.ndarray((N, 1))   system function of ODE formulation for
                                     a spring–damping system
        z_0     np.ndarray((N, 1))   the initial state
        dt      float                the time step
        n       int                  the number of steps
    ------------------------------------------------------------------------
    Outputs
        z   np.ndarray((N, n))   the next states x_{k_i}: i= 0 -> n
        (inc. k_0)
    ------------------------------------------------------------------------
    """
    # TODO implement your code here
    # subtask 4
    p = z0.shape[0]
    z = np.zeros((p, n), dtype=float)

    for s in range(0, p):
        z[s][0] = z0[s]

    for t in range(1, n):
        g = np.zeros((p, 1))
        for h in range(0, p):
            g[h] = z[h][t - 1]
        pre = g + dt * f_sys(g)
        kor = g + (dt/2) * (f_sys(g)+f_sys(pre))
        for l in range(0, p):
            z[l][t] = kor[l]

    return z


def rk4(f_sys, z0, dt, n):
    """
    ------------------------------------------------------------------------
            Compute the simulation steps using the RK4 Method
    ------------------------------------------------------------------------
    Inputs
        f_sys   np.ndarray((N, 1))   system function of ODE formulation for
                                     a spring–damping system
        z_0     np.ndarray((N, 1))   the initial state
        dt      float                the time step
        n       int                  the number of steps
    ------------------------------------------------------------------------
    Outputs
        z   np.ndarray((N, n))   the next states x_{k_i}: i= 0 -> n
        (inc. k_0)
    ------------------------------------------------------------------------
    """
    # TODO implement your code here
    # subtask 5
    p = z0.shape[0]
    z = np.zeros((p, n), dtype=float)

    for s in range(0, p):
        z[s][0] = z0[s]

    for t in range(1, n):
        g = np.zeros((p, 1))
        for h in range(0, p):
            g[h] = z[h][t - 1]
        s1 = f_sys(g)
        s2 = f_sys(g + (dt/2) * s1)
        s3 = f_sys(g + (dt/2) * s2)
        s4 = f_sys(g + dt * s3)
        kor = g + (dt/6) * (s1+ 2*s2 + 2*s3 + s4)
        for l in range(0, p):
            z[l][t] = kor[l]

    return z


def run_numerical_integrators(system, dt, int_steps):
    z0 = system.initial_conds()

    res = {
            'expl_euler': expl_euler(system.diff_eq, z0, dt, int_steps),
            'impl_euler': impl_euler(system.diff_eq, z0, dt, int_steps),
            'heun': heun(system.diff_eq, z0, dt, int_steps),
            'rk4': rk4(system.diff_eq, z0, dt, int_steps)
    }

    return res


# main function to run pyglet app
if __name__ == "__main__":
    # run app
    if ANIMATE:
        pyglet.clock.schedule_interval(update, DT)
        pyglet.app.run()

    pm = PointMass(m=M, d=D, k=K, g=GRAVITY)
    results = run_numerical_integrators(pm, t[len(t) - 1] / STEPS, STEPS)

    analytical, undamped, under_damped, critial_damping, over_damped = pm.diff_eq_analytical_sol(t)

    fig, axs = plt.subplots(4, 2, figsize=(19, 10))
    dt_func = DT
    ts = np.arange(0, STEPS*DT, DT)

    plot_options = {
        'expl_euler': {'color': 'y'},
        'impl_euler': {'color': 'g'},
        'heun': {'color': 'c'},
        'rk4': {'color': 'm'}
    }

    for method in results:
        if results[method] is None:
            results[method] = np.zeros((2, STEPS))
        x, v = results[method][0], results[method][1]
        axs[0, 0].plot(ts, x, color=plot_options[method]['color'], linestyle='-', label=method)
        axs[1, 0].plot(ts, v, color=plot_options[method]['color'], linestyle='-', label=method)

    for method in results:
        x, v = results[method][0], results[method][1]
        axs[0, 1].plot(ts, analytical[:, 0] - x, color=plot_options[method]['color'], linestyle='-', label=method)
        axs[1, 1].plot(ts, analytical[:, 1] - v, color=plot_options[method]['color'], linestyle='-', label=method)

    axs[0, 1].set_ylabel('position error')
    axs[1, 1].set_ylabel('velocity error')

    if ANIMATE:
        offset_pos = pos_simulator[0] + INITIAL_POS
        axs[0, 0].plot(ts, analytical[:, 0], 'r', label="analytical")
        axs[1, 0].plot(ts, analytical[:, 1], 'r', label="analytical")

        axs[2, 1].plot(ts, -(np.array(pos_simulator) - offset_pos), color='b', linestyle='-', label='simulator')
        axs[3, 1].plot(ts, -np.array(vel_simulator), color='b', linestyle='-', label='simulator')

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[2, 1].legend()

    axs[0, 0].set_ylabel('position')
    axs[1, 0].set_ylabel('velocity')
    axs[0, 1].set_ylabel('position error')
    axs[1, 1].set_ylabel('velocity error')
    axs[2, 1].set_ylabel('position simulator')
    axs[3, 1].set_ylabel('velocity simulator')

    axs[2, 0].plot(ts, undamped[:, 0], 'k', label="undamped", linewidth=0.15)
    axs[2, 0].plot(ts, under_damped[:, 0], 'r', label="under damped")
    axs[2, 0].plot(ts, critial_damping[:, 0], 'b', label=r"critical damping")
    axs[2, 0].plot(ts, over_damped[:, 0], 'g', label="over damped")

    axs[3, 0].plot(ts, undamped[:, 1], 'k', label="undamped", linewidth=0.15)
    axs[3, 0].plot(ts, under_damped[:, 1], 'r', label="under damped")
    axs[3, 0].plot(ts, critial_damping[:, 1], 'b', label=r"critical damping")
    axs[3, 0].plot(ts, over_damped[:, 1], 'g', label="over damped")

    axs[2, 0].legend()

    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i, j].set_xlabel('t')
            axs[i, j].grid(True)

    axs[2, 0].set_ylabel('position (analytical)')
    axs[3, 0].set_ylabel('velocity (analytical)')

    plt.show()

    """
    # TODO implement your answer here (Theorie Fragen)
    # subtask 6
   # 1-Die Ergebnisse werden ungenauer, bzw. approximieren die korrekte Lösung nicht genauso gut wie mit
        einem kleineren dt. Dafür kann das Verhalten des Systems über einen längeren Zeitraum betrachtet
        werden als mit einem kleineren dt, vorrausgesetzt die Anzahl der Steps n bleibt gleich. 
        Also: Möchte man z.B. den Systemzustand nach 5 Sekunden simulieren, benötigt es mit einem 
        größeren dt weniger Rechenaufwand als mit einem kleineren dt, jedoch sind dann die Ergebnisse 
        wie gesagt ungenauer/schlechter als mit einem kleineren dt. 
    #
    # 2-Wenn man die Federkonstante vergrößert, werden die Ergebnisse kleiner. Wenn man die Federkonstante
        verkleinert, werden die Ergebnisse größer.
    # 
    """
