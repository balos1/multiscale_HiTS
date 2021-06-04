import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity

class Advection1D():
    """
    Class for the advection problem in 1D space,
        u_t + c * u_x = 0,
    subject to periodic boundary conditions in space and
    initial conditions u(x, 0) = f(x).
    """

    def initial_condition(x):
        """
        f(x) = exp(-x^2)
        """
        return np.exp(-x ** 2)

    def __init__(self, c, domain, nx, u_init=initial_condition):

        self.c = c # advection speed

        # spatial domain
        self.domain = domain
        self.x = np.linspace(self.domain[0], self.domain[1], nx)
        self.x = self.x[0:-1]
        self.nx = nx - 1
        self.dx = self.x[1] - self.x[0]

        self.identity = identity(self.nx, dtype='float', format='csr')
        self.space_disc = self.compute_matrix()

        if callable(u_init):
            self.u_init = u_init(self.x)
        else:
            self.u_init = u_init

    def compute_matrix(self):
        """
        Define spatial discretization matrix for advection problem.
        Discretization is first-order upwind in space.
        """

        fac = self.c / self.dx

        diagonal = np.ones(self.nx) * fac
        lower = np.ones(self.nx) * -fac

        matrix = sp.diags(
            diagonals=[diagonal, lower],
            offsets=[0, -1], shape=(self.nx, self.nx),
            format='lil')
        # set periodic entry
        matrix[0, self.nx - 1] = -fac

        return sp.csr_matrix(matrix)

    def rhs(self, t, x):
        """
        ODE (i.e. the discretized PDE) right hand side.
        """
        return -self.space_disc*x

    def step_backward_euler(self, u_start, t_start, t_stop):
        """
        Backward Euler step.
        """
        return spsolve((t_stop - t_start) * self.space_disc + self.identity, u_start)

    def solve_backward_euler(self, dt, t_start, t_stop):
        """
        Solve the problem with Backward Euler.
        """
        nt = int((t_stop-t_start) / dt + 1)
        u = np.zeros((self.nx, nt))
        u[:,0] = self.u_init
        n = 1; t = t_start+dt
        while t < t_stop:
            # u[:,n] = spsolve(dt * self.space_disc + self.identity, u[:,n-1])
            u[:,n] = self.step_backward_euler(u[:,n-1], t, t+dt)
            t += dt
            n += 1
        return u

    def solve_ivp(self, dt, t_start, t_stop):
        """
        Solve the problem with scipy.solve_ivp.
        """
        nt = int((t_stop-t_start) / dt + 1)
        t = np.linspace(t_start, t_stop, nt)
        u = solve_ivp(self.rhs, [t_start, t_stop], self.u_init, t_eval=t)
        return u