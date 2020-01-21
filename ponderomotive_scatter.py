# calculate electron trajectory after scattering by laser ponderomotive force
import numpy as np
from scipy.integrate import odeint, RK45
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import my_cmap

NumberTypes = (int, float)

class pondscatter():
    def __init__(self, a0, w, tau, r0, z0, pr0, pz0, ode_method='RK45', t_array=None, t0=0., t_bound=None, dt_max=None, pth=1.):
        '''
            solve the ordinary differential equations dy/dt = func
            y = [r, z, pr, pz]
        '''
        self.a0 = a0
        self.w = w
        self.tau = tau
        self.initial_conditions = np.zeros(4)
        self.r0 = r0
        self.z0 = z0
        self.pr0 = pr0
        self.pz0 = pz0
        self.ode_method = ode_method
        # If self.ode_method is 'RK45', self.solve_ode() will use RK45, in this case self.t0, self.t_bound and self.dt_max will be used; otherwise self.solve_ode() will use odeint, in this case self.t0, self.t_bound and self.dt_max are ignored.
        self.t_array = t_array
        self.t0 = t0
        if t_bound is None: self.t_bound = t0+tau*6.
        else: self.t_bound = t_bound
        if dt_max is None: self.dt_max = tau/8
        else: self.dt_max = dt_max
        self.pth = pth
        self.axis_tuple = ('t', 'r', 'z', 'pr', 'pz', 'pr/pz')

######################## Property r0 ########################
    def get_r0(self):
        return self.initial_conditions[0]

    def set_r0(self, value):
        if not isinstance(value, NumberTypes):
            raiseTypeError('{} is not a number! Cannot set r0'.format(value))
        else: self.initial_conditions[0] = value

    r0 = property(get_r0, set_r0)

######################## Property z0 ########################
    def get_z0(self):
        return self.initial_conditions[1]

    def set_z0(self, value):
        if not isinstance(value, NumberTypes):
            raiseTypeError('{} is not a number! Cannot set z0'.format(value))
        else: self.initial_conditions[1] = value

    z0 = property(get_z0, set_z0)

######################## Property pr0 ########################
    def get_pr0(self):
        return self.initial_conditions[2]

    def set_pr0(self, value):
        if not isinstance(value, NumberTypes):
            raiseTypeError('{} is not a number! Cannot set pr0'.format(value))
        else: self.initial_conditions[2] = value

    pr0 = property(get_pr0, set_pr0)

######################## Property pz0 ########################
    def get_pz0(self):
        return self.initial_conditions[3]

    def set_pz0(self, value):
        if not isinstance(value, NumberTypes):
            raiseTypeError('{} is not a number! Cannot set pz0'.format(value))
        else: self.initial_conditions[3] = value

    pz0 = property(get_pz0, set_pz0)

######################## Method a0 ########################
    def a(self, r, xi):
        '''
            define laser profile. All in normalized units.
            r is radius to axis. xi = z - t. w is laser waist. tau is pulse duration.
            r and xi can be arries or matrix. a0, w and tau shall be scalers.
        '''
        return self.a0*np.exp(-np.square(r)/(self.w*self.w)-np.square(xi)/(self.tau*self.tau))

######################## Method func ########################
    def func(self, t, y):
        '''
            define the ordinary differential equations to be solved.
            y = [r, z, pr, pz]
        '''
        xi = y[1]-t
        # calculate square of a and save to memory
        a_square = (self.a(y[0], xi))**2
        # gamma is the averaged gamma in the relativistic ponderomotive force
        gamma = (1.+y[2]**2+y[3]**2+a_square*0.5)**0.5
        a_square_over_gamma = a_square/gamma
        return [y[2]/gamma, y[3]/gamma, a_square_over_gamma*y[0]/self.w/self.w, a_square_over_gamma*xi/self.tau/self.tau]

######################## Method solve_ode_odeint ########################
    def solve_ode_odeint(self):
        '''
            solve the ordinary differential equations defined by func, using odeint
        '''
        y = odeint(self.func, self.initial_conditions, self.t_array, tfirst=True)
        self.r_array = y[:,0]
        self.z_array = y[:,1]
        self.pr_array = y[:,2]
        self.pz_array = y[:,3]
        # Consider special cases where self.pz_array == 0
        self.pr_over_pz_array = np.zeros_like(self.pz_array)
        np.divide(self.pr_array, self.pz_array, out=self.pr_over_pz_array, where=(self.pz_array!=0))
        self.array_tuple = (self.t_array, self.r_array, self.z_array, self.pr_array, self.pz_array, self.pr_over_pz_array)

######################## Method solve_ode_RK45 ########################
    def solve_ode_RK45(self):
        '''
            solve the ordinary differential equations defined by func, using RK45
        '''
        solver = RK45(self.func, t0=self.t0, y0=self.initial_conditions, t_bound=self.t_bound, max_step=self.dt_max)
        t_list = []
        y_list = []
        while 'running' == solver.status:
            t_list.append(solver.t)
            y_list.append(solver.y)
            solver.step()
        y = np.array(y_list)
        self.t_array = np.array(t_list)
        self.r_array = y[:,0]
        self.z_array = y[:,1]
        self.pr_array = y[:,2]
        self.pz_array = y[:,3]
        # Consider special cases where self.pz_array == 0
        self.pr_over_pz_array = np.zeros_like(self.pz_array)
        np.divide(self.pr_array, self.pz_array, out=self.pr_over_pz_array, where=(self.pz_array!=0))
        self.array_tuple = (self.t_array, self.r_array, self.z_array, self.pr_array, self.pz_array, self.pr_over_pz_array)

######################## Method solve_ode ########################
    def solve_ode(self):
        '''
            solve the ordinary differential equations defined by func.
            If self.t_array is None, use RK45. Otherwise use odeint.
        '''
        if 'RK45' == self.ode_method: self.solve_ode_RK45()
        else: self.solve_ode_odeint()

######################## Method plot_line ########################
    def plot_line(self, h_fig=None, h_ax=None, axis_x='t', axis_y='r', **kwargs):
        '''
            Plot line of axis_y vs axis_x
        '''
        if h_fig is None:
            try: h_fig=self.h_fig
            except AttributeError: h_fig = self.h_fig = plt.figure()
        if h_ax is None:
            try: h_ax=self.h_ax
            except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
        h_ax.plot(self.array_tuple[self.axis_tuple.index(axis_x)], self.array_tuple[self.axis_tuple.index(axis_y)], **kwargs)
        h_ax.set_xlabel(axis_x)
        h_ax.set_ylabel(axis_y)
        return h_fig, h_ax

######################## Method plot_line_r0_array ########################
    def plot_line_r0_array(self, r0_array, axis_x='t', axis_y='r', line_colors=None, h_fig=None, h_ax=None, **kwargs):
        '''
            Plot line of axis_y vs axis_x with r0 in a array
        '''
        len_r0=len(r0_array)
        if line_colors is None: line_colors=cm.rainbow(np.linspace(0, 1, len_r0))
        if h_fig is None:
            try: h_fig=self.h_fig
            except AttributeError: h_fig = self.h_fig = plt.figure()
        if h_ax is None:
            try: h_ax=self.h_ax
            except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
        for i in range(len_r0):
            self.r0 = r0_array[i]
            self.solve_ode()
            self.plot_line(axis_x=axis_x, axis_y=axis_y, color=line_colors[i])
        return h_fig, h_ax

######################## Method pz_pr_vs_r0 ########################
    def pz_pr_vs_r0(self, r0_array, if_plot=False, h_fig=None, h_ax=None, **kwargs):
        '''
            Calculate and return pz_final vs r0 and pr_final vs r0.
            If if_plot is true, plot pz_final vs r0 and pr_final vs r0.
        '''
        len_r0=len(r0_array)
        pz_final=np.zeros(len_r0)
        pr_final=np.zeros(len_r0)
        for i in range(len_r0):
            self.r0 = r0_array[i]
            self.solve_ode()
            pz_final[i] = self.pz_array[-1]
            pr_final[i] = self.pr_array[-1]
        if if_plot:
            if h_fig is None:
                try: h_fig=self.h_fig
                except AttributeError: h_fig = self.h_fig = plt.figure()
            if h_ax is None:
                try: h_ax=self.h_ax
                except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
            h_ax.plot(r0_array, pz_final, label = '$p_z$')
            h_ax.plot(r0_array, pr_final, label = '$p_r$')
            h_ax.set_xlabel('$r_0$')
            h_ax.set_ylabel('$p_z$, $p_r$')
            plt.legend()
        return pz_final, pr_final

######################## Method plot_angle_r0_array ########################
    def plot_angle_r0(self, r0_array, h_fig=None, h_ax=None, **kwargs):
        '''
            Plot scatter angle vs r0
        '''
        len_r0=len(r0_array)
        angle_array=np.zeros(len_r0)
        for i in range(len_r0):
            self.r0 = r0_array[i]
            self.solve_ode()
            angle_array[i] = self.pr_array[-1]/self.pz_array[-1]
            # Maybe a bug: it is unclear why there are some value < 0
            #if angle_array[i]<0.: print('angle_array[i]<0!')
        # Transform to degree
        angle_array = np.arctan(np.abs(angle_array))/np.pi*180
        if h_fig is None:
            try: h_fig=self.h_fig
            except AttributeError: h_fig = self.h_fig = plt.figure()
        if h_ax is None:
            try: h_ax=self.h_ax
            except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
        h_ax.plot(r0_array, angle_array, **kwargs)
        return h_fig, h_ax

######################## Method pcolor_theta_p_projection ########################
    def pcolor_theta_p_projection(self, r0_array, theta_range, p_range, bins=128, h_fig=None, h_ax=None, **kwargs):
        '''
            Pseudocolor plot of p projection to the theta direction vs theta
        '''
        len_r0=len(r0_array)
        angle_array=np.zeros(len_r0)
        for i in range(len_r0):
            self.r0 = r0_array[i]
            self.solve_ode()
            angle_array[i] = self.pr_array[-1]/self.pz_array[-1]
            # Maybe a bug: it is unclear why there are some value < 0
            #if angle_array[i]<0.: print('angle_array[i]<0!')
        # Transform to degree
        angle_array = np.arctan(np.abs(angle_array))/np.pi*180
        if h_fig is None:
            try: h_fig=self.h_fig
            except AttributeError: h_fig = self.h_fig = plt.figure()
        if h_ax is None:
            try: h_ax=self.h_ax
            except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
        h_ax.plot(r0_array, angle_array, **kwargs)
        return h_fig, h_ax

######################## N_pn_gt_pth_vs_pth_theta ########################
    def N_pn_gt_pth_vs_pth_theta(self, d_r0, r0_max, pth_range = [0.1, 1.], theta_range = [0.1, np.pi/2], bins=128, if_plot=False, h_fig=None, h_ax=None, **kwargs):
        '''
            Calculate 2D histogram of N(pn > pth) vs pth and theta.
            Plot pseudocolor if if_plot is true.
        '''
        r0_array = np.arange(d_r0, r0_max, d_r0)
        len_r0 = len(r0_array)
        pz_final, pr_final = self.pz_pr_vs_r0(r0_array)
        pth_array = np.linspace(pth_range[0], pth_range[1], bins)
        theta_array = np.linspace(theta_range[0], theta_range[1], bins)
        len_pth_array = len(pth_array)
        len_theta_array = len(theta_array)
        N = np.zeros((len_pth_array, len_theta_array))
        for theta_ind in range(len_theta_array):
            cos_theta = np.cos(theta_array[theta_ind])
            sin_theta = np.sin(theta_array[theta_ind])
            for pth_ind in range(len_pth_array):
                for r0_ind in range(len_r0):
                    # pn = pz*cos(theta) + pr*sin(theta)*cos(phi)
                    cos_phi = (pth_array[pth_ind] - pz_final[r0_ind]*cos_theta)/pr_final[r0_ind]/sin_theta
                    if cos_phi<=(-1):
                        # phi from -pi to pi satisfy pn>pth
                        N[pth_ind, theta_ind] += r0_array[r0_ind] * (np.pi * 2) # * d_r0 for absolute number
                    elif cos_phi<1.:
                        # -1<cos(phi)<1, phi in a range smaller than pi satisfy pn>pth
                        N[pth_ind, theta_ind] += r0_array[r0_ind] * (np.arccos(cos_phi)*2)
                    # if cos(phi)>1, pn>pth is never satisfied, do not add anything
        if if_plot:
            if h_fig is None:
                h_fig = self.h_fig = plt.figure()
            if h_ax is None:
                h_ax = self.h_ax = h_fig.add_subplot(111)
            h_plot = h_ax.pcolormesh(theta_array*(180./np.pi), pth_array, N, cmap=my_cmap.cmap_lower_range_transparent(transparency_transition_region=[0.,0.02]), **kwargs)
            h_ax.set_ylabel('$p_{{th}}$')
            h_ax.set_xlabel('$\\theta$ [$^{{\circ}}$]')
            h_cb = plt.colorbar(h_plot, ax=h_ax)
            h_cb.set_label('$N_{{p_n > p_{{\\rm th}}}}$')
        return h_fig, h_ax

if __name__ == '__main__':
    # laser parameters
    a0 = 1.
    w = 2.
    L_FWHM = 4.
    tau = L_FWHM/1.665
    # initial conditions
    r0 = w*1.
    z0 = tau*2.
    # time array
    t_array = np.arange(0., 10., 0.1)
    pdsc = pondscatter(a0 = a0, w = w, tau = tau, r0 = r0, z0 = z0, pr0 = 0., pz0 = 0., dt_max = tau/4)
    r0_range = np.arange(0.01*w, 3.*w, 0.01*w)
    #pdsc.plot_line_r0_array(r0_range, axis_x='t', axis_y='r')
    #pdsc.pz_pr_vs_r0(r0_range)
    pdsc.N_pn_gt_pth_vs_pth_theta(0.01*w, 3.*w, theta_range = [0.05, 2.6], bins=128, if_plot=True)
    #pdsc.plot_angle_r0(r0_range)
    plt.tight_layout()
    plt.show()
