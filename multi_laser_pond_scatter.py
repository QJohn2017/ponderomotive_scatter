from ponderomotive_scatter import pondscatter
import numpy as np

class m_laser_pd_scatter(pondscatter):
    def __ini__(self, a0, w, tau, r0, z0=None, pr0=0., pz0=0., ode_method='RK45', t_array=None, t0=0., t_bound=None, dt_max=None):
        '''
        Solve the equation of motion by ponderomotive of multiple lasers
        a0, w, tau: list of floats, for the parameters of a series of lasers.
        '''
        super(m_laser_pd_scatter, self).__init__(a0, w, tau, r0, z0, pr0, pz0, ode_method, t_array, t0, t_bound, dt_max)
        
    # Override set_a0t() of super class
    def set_a0(self, val):
        self._a0 = np.array(val)
    a0 = property(pondscatter.get_a0, set_a0)
    # Override set_w of super class
    def set_w(self, val):
        self._w = np.array(val)
    w = property(pondscatter.get_w, set_w)
    # Override set_tau of super class
    def set_tau(self, val):
        self._tau = np.array(val)
        # Every time changing tau, z0 t_bound dt_max should also be changed
        tau_max = max(val)
        self.z0 = tau_max*3 + self.t0 # set z0 to be 3 tau away from the starting point of the laser is enough in most cases
        self.t_bound = self.z0 + tau_max*3 # t_bound-t0 = z0-t0+tau*3
        self.dt_max = min(val)/8 # a smaller d_tmax makes the trajectory moother
    tau = property(pondscatter.get_tau, set_tau)

    def a(self, r, xi):
        '''
            Overriding a() of super class. Return ndarray of a of all lasers.
            define laser profile. All in normalized units.
            r is radius to axis. xi = z - t. w is laser waist. tau is pulse duration.
            r and xi can be arries or matrix. a0, w and tau shall be scalers.
        '''
        return self.a0*np.exp(-np.square(r)/np.square(self.w)-np.square(xi)/np.square(self.tau))

    def func(self, t, y):
        '''
            define the ordinary differential equations to be solved.
            y = [r, z, pr, pz]
        '''
        xi = y[1]-t
        # calculate ndarray of a and save to memory
        a = self.a(y[0], xi)
        a_square = np.square(a)
        # the averaged gamma in the relativistic ponderomotive force
        gamma = (1.+y[2]**2+y[3]**2+np.sum(a_square)/2)**0.5
        return [y[2]/gamma, y[3]/gamma, y[0]/gamma*np.sum(a_square/np.square(self.w)), xi/gamma*np.sum(a_square/np.square(self.tau))]

if "__main__" == __name__:
    import matplotlib.pyplot as plt
    pd = m_laser_pd_scatter(a0=[1.], w=[1.], tau=[1.], r0=1.)
    pd.solve_ode()
    pd.plot_line(axis_x='pr', axis_y='pz', label='1 laser')
    pd.a0=[1.,1./9]
    pd.w=[1.,1.]
    pd.tau=[1.,1.]
    pd.solve_ode()
    pd.plot_line(axis_x='pr', axis_y='pz', label='2 lasers')
    plt.legend()
    plt.show()
