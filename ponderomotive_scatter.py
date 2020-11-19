# calculate electron trajectory after scattering by laser ponderomotive force
import warnings
import numpy as np
from scipy.integrate import odeint, RK45
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import my_cmap

NumberTypes = (int, float, np.int64)

##################### Class pondscatter #####################
class pondscatter():
    def __init__(self, a0, w, tau, r0, z0=None, pr0=0., pz0=0., ode_method='RK45', t_array=None, t0=0., t_bound=None, dt_max=None):
        '''
            solve the ordinary differential equations dy/dt = func
            y = [r, z, pr, pz]
        '''
        self.initial_conditions = np.zeros(4)
        self.a0 = a0
        self.w = w
        self.t0 = t0
        self.tau = tau
        self.r0 = r0
        # Setting self.tau will also set self.z0. But if z0 is explicitly given, use this explicite value.
        if z0 is not None: self.z0 = z0
        self.pr0 = pr0
        self.pz0 = pz0
        self.ode_method = ode_method
        # If self.ode_method is 'RK45', self.solve_ode() will use RK45, in this case self.t0, self.t_bound and self.dt_max will be used; otherwise self.solve_ode() will use odeint, in this case self.t0, self.t_bound and self.dt_max are ignored.
        self.t_array = t_array
        # Setting self.tau will also set self.t_bound. But if t_bound is explicitly given, use this explicite value.
        if t_bound is not None: self.t_bound = t_bound
        # Setting self.tau will also set self.dt_max. But if dt_max is explicitly given, use this explicite value.
        if dt_max is not None: self.dt_max = dt_max
        self.axis_tuple = ('t', 'r', 'z', 'pr', 'pz', 'pr/pz')

######################## Property a0 ########################
    def get_a0(self):
        return self._a0

    def set_a0(self, value):
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is not a number! Cannot set a0'.format(value))
        else: self._a0 = value

    a0 = property(get_a0, set_a0)

######################## Property w ########################
    def get_w(self):
        return self._w

    def set_w(self, value):
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is not a number! Cannot set w'.format(value))
        else: self._w = value

    w = property(get_w, set_w)

######################## Property tau ########################
    def get_tau(self):
        return self._tau

    def set_tau(self, value):
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is a {}, not a number! Cannot set tau'.format(value, type(value)))
        else:
            self._tau = value
            # Every time changing tau, z0 t_bound dt_max should also be changed
            self.z0 = self._tau*3 + self.t0 # set z0 to be 3 tau away from the starting point of the laser is enough in most cases
            self.t_bound = self.z0 + self._tau*3 # t_bound-t0 = z0-t0+tau*3
            self.dt_max = self._tau/8 # a smaller d_tmax makes the trajectory moother

    tau = property(get_tau, set_tau)

######################## Property r0 ########################
    def get_r0(self):
        return self.initial_conditions[0]

    def set_r0(self, value):
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is not a number! Cannot set r0'.format(value))
        else: self.initial_conditions[0] = value

    r0 = property(get_r0, set_r0)

######################## Property z0 ########################
    def get_z0(self):
        return self.initial_conditions[1]

    def set_z0(self, value):
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is not a number! Cannot set z0'.format(value))
        else: self.initial_conditions[1] = value

    z0 = property(get_z0, set_z0)

######################## Property pr0 ########################
    def get_pr0(self):
        return self.initial_conditions[2]

    def set_pr0(self, value):
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is not a number! Cannot set pr0'.format(value))
        else: self.initial_conditions[2] = value

    pr0 = property(get_pr0, set_pr0)

######################## Property pz0 ########################
    def get_pz0(self):
        return self.initial_conditions[3]

    def set_pz0(self, value):
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is not a number! Cannot set pz0'.format(value))
        else: self.initial_conditions[3] = value

    pz0 = property(get_pz0, set_pz0)

######################## Method new_fig ########################
    def new_fig(self):
        '''
            Creat new fig.
        '''
        self.h_fig = plt.figure()
        self.h_ax = self.h_fig.add_subplot(111)

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

######################## Method solve_ode ########################
    def solve_ode(self):
        '''
            solve the ordinary differential equations defined by func.
            If 'RK45' == self.ode_method, use RK45. Otherwise use odeint.
        '''
        if 'RK45' == self.ode_method: self.solve_ode_RK45()
        else: self.solve_ode_odeint()
        # Gather the results in a tuple, for the plotting purposes.
        # Consider special cases where self.pz_array == 0
        self.pr_over_pz_array = np.zeros_like(self.pz_array)
        np.divide(self.pr_array, self.pz_array, out=self.pr_over_pz_array, where=(self.pz_array!=0))
        # The sequence in this tuple should be the same as self.axis_tuple
        self.ode_result_array_tuple = (self.t_array, self.r_array, self.z_array, self.pr_array, self.pz_array, self.pr_over_pz_array)

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
        h_ax.plot(self.ode_result_array_tuple[self.axis_tuple.index(axis_x)], self.ode_result_array_tuple[self.axis_tuple.index(axis_y)], **kwargs)
        h_ax.set_xlabel(axis_x)
        h_ax.set_ylabel(axis_y)
        return h_fig, h_ax

######################################################################
##################### Class multi_r0_pondscatter #####################
#####################    Inherite pondscatter    #####################
######################################################################
class multi_r0_pondscatter(pondscatter):
    '''
        Ponderomotive scatterring simulator with r0 in an array.
    '''
    def __init__(self, a0, w, tau, d_r0=None, r0_max=None):
        pondscatter.__init__(self, a0, w, tau, 0., z0=None, pr0=0., pz0=0., ode_method='RK45', t_array=None, t0=0., t_bound=None, dt_max=None)
        # In pondscatter.__init__(), self.set_w() is called which is overloaded in this subclass, so we do not have to set w here.
        #self.w = w
        # Setting self.w will also set self.d_r0. But if d_r0 is explicitly given, use this explicite value.
        if d_r0 is not None:
            self.d_r0 = d_r0
        # Setting self.w will also set self.r0_max. But if r0_max is explicitly given, use this explicite value.
        # Set self.r0_max will update r0_array and len_r0_arry automatically.
        if r0_max is not None: self.r0_max = r0_max
        self.rad_to_degree = 180./np.pi

######################## Property w ########################
    def set_w(self, value):        
        if not isinstance(value, NumberTypes):
            raise TypeError('{} is not a number! Cannot set w'.format(value))
        else:
            self._w = value
            # Every time changing self.w, d_r0 r0_max should also be changed
            self._d_r0 = self._w/128
            # Set self.r0_max will update r0_array and len_r0_arry automatically.
            self.r0_max = self._w*3        

    w = property(pondscatter.get_w, set_w)

######################## Property d_r0 ########################
    def get_d_r0(self):
        return self._d_r0

    def set_d_r0(self, value):
        self._d_r0 = value
        try:
            self._r0_array = np.arange(self.d_r0, self.r0_max, self.d_r0)
            self._len_r0_array = len(self._r0_array)
        except: print('Failed to set r0_array!') # This exception happens if self.r0_max is not set.

    d_r0 = property(get_d_r0, set_d_r0)

######################## Property r0_max ########################
    def get_r0_max(self):
        return self._r0_max

    def set_r0_max(self, value):
        self._r0_max = value
        try:
            self._r0_array = np.arange(self.d_r0, self.r0_max, self.d_r0)
            self._len_r0_array = len(self._r0_array)
        except: print('Failed to set r0_array!')

    r0_max = property(get_r0_max, set_r0_max)

######################## Property r0_array ########################
    def get_r0_array(self):
        return self._r0_array

    def set_r0_array(self, **kwargs):
        raise RuntimeError('You cannot set r0_array! Set d_r0 and r0_max instead.')

    r0_array = property(get_r0_array, set_r0_array)

######################## Property len_r0_array ########################
    def get_len_r0_array(self):
        return self._len_r0_array

    def set_len_r0_array(self, **kwargs):
        raise RuntimeError('You cannot set len_r0_array! Set d_r0 and r0_max instead.')

    len_r0_array = property(get_len_r0_array, set_len_r0_array)

######################## Method plot_line_r0_array ########################
    def plot_line_r0_array(self, axis_x='t', axis_y='r', line_colors=None, h_fig=None, h_ax=None, **kwargs):
        '''
            Plot line of axis_y vs axis_x with r0 in a array.
            Available values of axis_x ans axis_y are listed in self.axis_tuple.
        '''
        if line_colors is None: line_colors=cm.rainbow(np.linspace(0, 1, self.len_r0_array))
        if h_fig is None:
            try: h_fig=self.h_fig
            except AttributeError: h_fig = self.h_fig = plt.figure()
        if h_ax is None:
            try: h_ax=self.h_ax
            except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
        for i in range(self.len_r0_array):
            self.r0 = self.r0_array[i]
            self.solve_ode()
            self.plot_line(axis_x=axis_x, axis_y=axis_y, color=line_colors[i])
        return h_fig, h_ax

######################## Method pz_pr_vs_r0 ########################
    def pz_pr_vs_r0(self, if_plot=False, h_fig=None, h_ax=None, **kwargs):
        '''
            Calculate and return self.pz_final (final p_z_prime after scattering) vs r0 and self.pr_final vs r0.
            If if_plot is true, plot self.pz_final vs r0 and pr_final vs r0.
        '''
        while True:
            # Test the first two r0 to make sure d_r0 is small enough
            self.pz_final=np.zeros(self.len_r0_array)
            self.pr_final=np.zeros(self.len_r0_array)
            for i in range(2):
                self.r0 = self.r0_array[i]
                self.solve_ode()
                self.pz_final[i] = self.pz_array[-1]
                self.pr_final[i] = self.pr_array[-1]
            # Reduce self.d_r0 until the final p is increasing with r_0 at small r_0
            '''if self.pz_final[1]<self.pz_final[0] or self.pr_final[1]<self.pr_final[0]:
                self.d_r0 = self.d_r0/2 # This automatically recalculates self._r0_array and self._len_r0_array
                print('Warning: d_r0 is not small enough for a0 = {}, w = {}, tau = {}. Reducing it to {}.'.format(self.a0, self.w, self.tau, self.d_r0))
            else: break'''
            break
        for i in range(2, self.len_r0_array):
            self.r0 = self.r0_array[i]
            self.solve_ode()
            self.pz_final[i] = self.pz_array[-1]
            self.pr_final[i] = self.pr_array[-1]
        if if_plot:
            if h_fig is None:
                try: h_fig=self.h_fig
                except AttributeError: h_fig = self.h_fig = plt.figure()
            if h_ax is None:
                try: h_ax=self.h_ax
                except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
            h_ax.plot(self.r0_array, self.pz_final, label = '$p_z$')
            h_ax.plot(self.r0_array, self.pr_final, label = '$p_r$')
            h_ax.set_xlabel('$r_0$')
            h_ax.set_ylabel('$p_z$, $p_r$')
            plt.legend()
        return self.pz_final, self.pr_final

######################################################################
######################### Class N_inj_grapher #########################
#################### Inherite multi_r0_pondscatter ###################
######################################################################
class N_inj_grapher(multi_r0_pondscatter):
    '''Inherite multi_r0_pondscatter to make number of injectable particle vs psi_M and theta histogram'''
    def __init__(self, a0=1., w=1., tau=1., d_r0=None, r0_max=None, psi_M_range=None, psi_M_bins=32, theta_range_rad=[0.01, 3.14], theta_range_degree=None, theta_bins=128):
        multi_r0_pondscatter.__init__(self, a0, w, tau, d_r0, r0_max)
        self._psi_M_bins = psi_M_bins
        if psi_M_range is None:
            self.auto_reset_psi_M_range()
        else: self.psi_M_range = psi_M_range
        self._theta_bins = theta_bins
        if theta_range_degree is None: self.theta_range_rad = theta_range_rad
        # Setting theta_range_degree overrides theta_range_rad
        else: self.theta_range_degree = theta_range_degree

######################## Property psi_M_range ########################
    def get_psi_M_range(self):
        return self._psi_M_range

    def set_psi_M_range(self, list_of_2_elements):
        self._psi_M_range = list_of_2_elements
        try:
            self._psi_M_array = np.linspace(self._psi_M_range[0], self._psi_M_range[1], self.psi_M_bins)
        except: print('Failed to set psi_M_array!')

    psi_M_range = property(get_psi_M_range, set_psi_M_range)

######################## Property psi_M_bins ########################
    def get_psi_M_bins(self):
        return self._psi_M_bins

    def set_psi_M_bins(self, n_bins):
        self._psi_M_bins = n_bins
        try:
            self._psi_M_array = np.linspace(self.psi_M_range[0], self.psi_M_range[1], self._psi_M_bins)
        except: print('Failed to set psi_M_array!')

    psi_M_bins = property(get_psi_M_bins, set_psi_M_bins)

######################## Property psi_M_array ########################
    def get_psi_M_array(self):
        return self._psi_M_array

    def set_psi_M_array(self, **kwargs):        
        raise RuntimeError('You cannot set psi_M_array! Set psi_M_range and psi_M_bins instead.')

    psi_M_array = property(get_psi_M_array, set_psi_M_array)

######################## Property theta_range_rad ########################
# theta range in radians
    def get_theta_range_rad(self):
        return self._theta_range_rad

    def set_theta_range_rad(self, list_of_2_elements):
        self._theta_range_rad = list_of_2_elements
        try:
            self._theta_array_rad = np.linspace(self._theta_range_rad[0], self._theta_range_rad[1], self.theta_bins)
        except: print('Failed to set theta_array_rad!')

    theta_range_rad = property(get_theta_range_rad, set_theta_range_rad)

######################## Property theta_range_degree ########################
# theta range in degree
    def get_theta_range_degree(self):
        return [self.theta_range_rad[0]*self.rad_to_degree, self.theta_range_rad[1]*self.rad_to_degree]

    def set_theta_range_degree(self, list_of_2_elements):
        self.theta_range_rad = [list_of_2_elements[0]/self.rad_to_degree, list_of_2_elements[1]/self.rad_to_degree]

    theta_range_degree = property(get_theta_range_degree, set_theta_range_degree)

######################## Property theta_bins ########################
    def get_theta_bins(self):
        return self._theta_bins

    def set_theta_bins(self, n_bins):
        self._theta_bins = n_bins
        try:
            self._theta_array_rad = np.linspace(self.theta_range_rad[0], self.theta_range_rad[1], self._theta_bins)
        except: print('Failed to set theta_array_rad!')

    theta_bins = property(get_theta_bins, set_theta_bins)

######################## Property theta_array_rad ########################
    def get_theta_array_rad(self):
        return self._theta_array_rad

    def set_theta_array_rad(self, **kwargs):        
        raise RuntimeError('You cannot set theta_array_rad! Set theta_range_rad and theta_bins instead.')

    theta_array_rad = property(get_theta_array_rad, set_theta_array_rad)

######################## Property theta_array_degree ########################
    def get_theta_array_degree(self):
        return self._theta_array_rad*self.rad_to_degree

    def set_theta_array_degree(self, **kwargs):        
        raise RuntimeError('You cannot set theta_array_degree! Set theta_range_rad and theta_bins instead.')

    theta_array_degree = property(get_theta_array_degree, set_theta_array_degree)

######################## Method auto_reset_pth_range ########################
    def auto_reset_psi_M_range(self, min_ratios=0.9):
        '''
            Automatically reset psi_M_range
        '''
        self.pz_pr_vs_r0() # obtain self.pz_final and self.pr_final
        max_p_final = np.max(np.sqrt(self.pr_final**2+self.pz_final**2))
        # This automatically set self.pth_array
        self.pth_range = [min_ratios*((1.+max_p_final*max_p_final)**0.5-max_p_final), 1.]
        return

######################## Method N_inj_vs_psi_M_theta ########################
    def N_inj_vs_psi_M_theta(self, if_plot=False, h_fig=None, h_ax=None, **kwargs):
        '''
            Calculate 2D histogram of number of injectable particles vs psi_M and theta and save to self.N_inj.
            Plot pseudocolor if if_plot is true.
            One should have alreadt run self.pz_pr_vs_r0() to obtain self.pz_final and self.pr_final before calling this.
        '''
        # z_prime and r_prime is the laser coordinate, while z is the main wake coordinate.
        p_z_prime = self.pz_final
        p_r_prime = self.pr_final
        gamma_final = np.sqrt(1.+np.square(p_z_prime)+np.square(p_r_prime))
        self.N_inj = np.zeros((self.psi_M_bins, self.theta_bins))
        for theta_ind in range(self.theta_bins):
            cos_theta = np.cos(self.theta_array_rad[theta_ind])
            sin_theta = np.sin(self.theta_array_rad[theta_ind])
            for psi_M_ind in range(self.psi_M_bins):
                for r0_ind in range(self.len_r0_array):
                    # p_z = p_z_prime*cos_theta + p_r_prime*sin_theta cos_phi
                    # particles with cos_phi smaller than this value should be added to N
                    cos_phi = (gamma_final[r0_ind] - self.psi_M_array[psi_M_ind] - p_z_prime[r0_ind]*cos_theta)/p_r_prime[r0_ind]/sin_theta
                    if cos_phi<=(-1):
                        # phi from -pi to pi is injectable
                        # Number proportional to r0
                        self.N_inj[psi_M_ind, theta_ind] += self.r0_array[r0_ind] * (np.pi * 2) # * d_r0 for absolute number
                    elif cos_phi<1.:
                        # -1<cos(phi)<1, phi in a range smaller than pi satisfies pn>pth
                        # Number proportional to r0
                        self.N_inj[psi_M_ind, theta_ind] += self.r0_array[r0_ind] * (np.arccos(cos_phi)*2)
                    # if cos(phi)>=1, none is injectable, do not add anything
        if if_plot:
            if h_fig is None:
                try: h_fig=self.h_fig
                except AttributeError: h_fig = self.h_fig = plt.figure()
            if h_ax is None:
                try: h_ax=self.h_ax
                except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
            h_plot = h_ax.pcolormesh(self.theta_array_degree, self.psi_M_array, self.N_inj, cmap=my_cmap.cmap_lower_range_transparent(transparency_transition_region=[0.,0.02]), **kwargs)
            h_ax.set_ylabel('$\\psi_M$')
            h_ax.set_xlabel('$\\theta$ [$^{{\\circ}}$]')
            h_cb = plt.colorbar(h_plot, ax=h_ax)
            h_cb.set_label('$N_{{\\rm trap}}$ [arb. unit]')
        return

######################## Method psi_M_th_vs_theta ########################
    def psi_M_th_vs_theta(self, if_plot=False, h_fig=None, h_ax=None, **kwargs):
        '''
            Get threshold of psi_M for injection to occur vs. theta.
            One should have alreadt run self.pz_pr_vs_r0() to obtain self.pz_final and self.pr_final before calling this.
        Input:
            if_plot: boolean
                     if True, plot the line of max pth vs theta

        Output:
            self.psi_M_th: numpy array of floats
                           Array of threshold of psi_M for injection to occur
        '''
        # Assuem psi_M_th are never larger than 1.
        self.psi_M_th = np.ones(self.theta_bins)
        # z_prime and r_prime is the laser coordinate, while z is the main wake coordinate.
        p_z_prime = self.pz_final
        p_r_prime = self.pr_final
        gamma_final = np.sqrt(1.+np.square(p_z_prime)+np.square(p_r_prime))
        cos_theta = np.cos(self.theta_array_rad)
        sin_theta = np.sin(self.theta_array_rad)
        for r0_ind in range(self.len_r0_array):
            # p_z = p_z_prime*cos_theta - p_r_prime*sin_theta cos(phi+pi)
            # The maximum p_z has phi = 0
            # p_z array vs theta
            p_z = p_z_prime[r0_ind]*cos_theta + p_r_prime[r0_ind]*sin_theta
            # phi_M_tmp array vs theta
            phi_M_tmp = gamma_final[r0_ind] - p_z
            # (self.psi_M_th array vs theta) takes values of (phi_M_tmp array vs theta) when the later is smaller
            self.psi_M_th = np.where(phi_M_tmp<self.psi_M_th, phi_M_tmp, self.psi_M_th)
        if if_plot:
            if h_fig is None:
                try: h_fig=self.h_fig
                except AttributeError: h_fig = self.h_fig = plt.figure()
            if h_ax is None:
                try: h_ax=self.h_ax
                except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
            h_plot = h_ax.plot(self.theta_array_degree, self.psi_M_th, **kwargs)
        return

if __name__ == '__main__':
    pdsc = N_inj_grapher(a0=1., w=1., d_r0=1./512, tau=2., psi_M_range=[.55, .8], psi_M_bins=64, theta_range_degree=[30., 120.], theta_bins=64)
    pdsc.pz_pr_vs_r0()
    pdsc.N_inj_vs_psi_M_theta(if_plot=True)#, norm=LogNorm())
    pdsc.psi_M_th_vs_theta(if_plot=True)
    #pdsc.N_pn_vs_pth_theta(if_plot=True)
    #pdsc.get_pth_max_vs_theta(if_plot=True)
    #pdsc = multi_r0_pondscatter(a0=1., w=1., d_r0=0.00000001, r0_max=0.000001, tau=20.)
    #pdsc.pz_pr_vs_r0(if_plot=True)
    #pdsc.plot_line_r0_array(axis_x='t', axis_y='pz')
    plt.tight_layout()
    plt.show()
