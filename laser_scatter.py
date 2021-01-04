# calculate electron trajectory after scattering by lasers moving along z
import warnings
import numpy as np
from scipy.integrate import odeint, RK45
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

NumberTypes = (int, float, np.int64)
very_small = 1e-6
very_large = 1e6

##################### Class laser_scatter #####################
class laser_scatter():
    def __init__(self, lasers, initial_conditions=[0., 0., 0., 0., 0., 0.], t_bound=None, dt_max=None, q_over_m=-1):
        '''
            lasers: a single laser object or a list (tuple) of laser objects.
            q_over_m: q (normalized to e) over m (normalized to m_e) of the particle.
        '''
        self.lasers = lasers
        self.initial_conditions = initial_conditions
        # Set t_bound only if this parameter is explicility given.
        if t_bound is not None: self.t_bound = t_bound
        # Set dt_max only if this parameter is explicility given.
        if dt_max is not None: self.dt_max = dt_max
        self.q_over_m = q_over_m

    def get_lasers(self):
        return self._lasers
    def set_lasers(self, val):
        if isinstance(val, (list, tuple, np.ndarray)): self._lasers = val
        else:
            # Make laser a list if it is neither a list, a tuple nor an np.ndarray (only when its a single laser object)
            self._lasers = [val]
        # Update t_bound and dt_max
        self.auto_t_bound()
        self.auto_dt_max()
    lasers = property(get_lasers, set_lasers)

    def auto_t_bound(self):
        '''
            Automatically set self.t_bound based on the interaction time estimation of self.lasers
        '''
        t_bound = 0.
        for l in self.lasers:
            tmp = l.interact_time_estimate()
            if tmp>t_bound: t_bound = tmp
        self.t_bound = t_bound

    def auto_dt_max(self):
        '''
            Automatically set self.dt_max based on the omenga0 of self.lasers
        '''
        dt_max = very_large
        for l in self.lasers:
            tmp = 0.1/l.omega0
            if tmp<dt_max: dt_max = tmp
        self.dt_max = dt_max

    def new_fig(self):
        '''
            Creat new fig and axis handles.
        '''
        self.h_fig = plt.figure()
        self.h_ax = self.h_fig.add_subplot(111)

    def func(self, t, p):
        '''
            define the ordinary differential equations to be solved.
            p = [x, y, z, px, py, pz]
        '''
        E = np.zeros(3)
        B = np.zeros(3)
        for l in self.lasers:
            E_buf, B_buf = l.EB(t, p[:3])
            E += E_buf
            B += B_buf
        gamma = np.sqrt(1.+np.sum(np.square(p[3:])))
        v = [p[3]/gamma, p[4]/gamma, p[5]/gamma]
        return [v[0], v[1], v[2],\
                self.q_over_m*(E[0]+v[1]*B[2]-v[2]*B[1]),\
                self.q_over_m*(E[1]+v[2]*B[0]-v[0]*B[2]),\
                self.q_over_m*(E[2]+v[0]*B[1]-v[1]*B[0])]

    def solve_ode(self):
        '''
            solve the ordinary differential equations defined by func, using RK45
        '''
        solver = RK45(self.func, t0=0., y0=self.initial_conditions, t_bound=self.t_bound, max_step=self.dt_max, rtol=1e-6, atol=1e-9)
        solution_list = []
        # Maximum t_bound increase iteration is 10
        for i in range(10):
            while 'running' == solver.status:
                solution_list.append(np.append(solver.t, solver.y))
                solver.step()
            # Check momentum difference of last two results
            p_diff = max([abs(solution_list[-1][4]-solution_list[-2][4]), abs(solution_list[-1][5]-solution_list[-2][5]), abs(solution_list[-1][6]-solution_list[-2][6])])
            # Continue running until momentum difference is negligible
            if p_diff < very_small: break
            else:
                solver.t_bound *= 1.2
                solver.status = 'running'
        if p_diff>=very_small: warnings.warn('Momentum change at t_bound is still not negligible. Maybe you need a larger t_bound.')
        self.solution = np.array(solution_list)

######################## Method plot_line ########################
    def plot_line(self, h_fig=None, h_ax=None, axis_x=0, axis_y=1, **kwargs):
        '''
            Plot line of axis_y vs axis_x.
            Index for axis: 0 - t
                            1 - x
                            2 - y
                            3 - z
                            4 - px
                            5 - py
                            6 - pz
        '''
        if h_fig is None:
            try: h_fig=self.h_fig
            except AttributeError: h_fig = self.h_fig = plt.figure()
        if h_ax is None:
            try: h_ax=self.h_ax
            except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
        h_l, = h_ax.plot(self.solution[:, axis_x], self.solution[:, axis_y], **kwargs)
        labels = ['$t$', '$x$', '$y$', '$z$', '$p_x$', '$p_y$', '$p_z$']
        if 'label' not in kwargs: h_l.set_label(labels[axis_y])
        h_ax.set_xlabel(labels[axis_x])
        h_ax.set_ylabel(labels[axis_y])
        return h_fig, h_ax, h_l

if __name__ == '__main__':
    import laser_functions
    l0 = laser_functions.laser(a0=1., z0=-3.,omega0=30., phase0=90.)
    l1 = laser_functions.laser(a0=0.3, z0=3., omega0=30., dir_polar=180.)
    ls = laser_scatter(lasers=[l0], initial_conditions=[0., 1., 0., 0., 0., 0.])
    ls.solve_ode()
    #ls.plot_line(axis_x=0, axis_y=5, ls=':', label='py, 1 laser')
    #ls.plot_line(axis_x=0, axis_y=6, ls=':', label='pz, 1 laser')
    ls.plot_line(axis_x=5, axis_y=6, ls=':', label='1 laser')
    ls.lasers = [l0, l1]
    ls.solve_ode()
    #ls.plot_line(axis_x=0, axis_y=5, ls='--', label='py, 2 lasers')
    #ls.plot_line(axis_x=0, axis_y=6, ls='--', label='pz, 2 lasers')
    ls.plot_line(axis_x=5, axis_y=6, ls='--', label='2 lasers')
    plt.legend()
    plt.show()
