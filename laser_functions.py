# Generate laser functions for ode solving
import warnings
import numpy as np

NumberTypes = (int, float, np.int64)

##################### Class laser #####################
class laser():
    def __init__(self, a0=1., w0=1., tau=1., omega0=20., dir_polar=0., dir_azimuth=0., polarization_angle=0., x0=0., y0=0., z0=0., phase0=0.):
        '''
            For a non-evolving laser definition.
            dir_polar, dir_azimuth: polar and azimuthal angles of the laser direction in degrees. Remember to use self.reset_rot_matrix() every time you set the laser direction.
            polarization_angle: angle (in degrees) of linear polarization plane of E field referring to the x-z plane of laser coordinate.
        '''
        assert(a0>0)
        assert(w0>0)
        assert(tau>0)
        assert(omega0>0)
        self.a0=a0
        self.w0=w0
        self.tau=tau
        self.omega0=omega0
        self.dir_polar=dir_polar
        self.dir_azimuth=dir_azimuth
        self.polarization_angle=polarization_angle
        self.reset_rot_matrix()
        self.x0=x0
        self.y0=y0
        self.z0=z0
        self.phase0=phase0

    def reset_rot_matrix(self):
        '''
            Calculate axis rotation matrix and save to self.rot_matrix.
        '''
        # Transform angles from degrees to radian
        theta = self.dir_polar*(np.pi/180.)
        phi = self.dir_azimuth*(np.pi/180.)
        alpha = self.polarization_angle*(np.pi/180.)
        # Cache cos and sin
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        # Axis rotation, 1st rotate phi along z, 2nd rotate theta along y', 3rd rotate alpha along z'.
        #self.rot_matrix = np.array([[cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta],\
        #                            [-sin_phi, cos_phi, 0.],\
        #                            [sin_theta*cos_phi, sin_theta*sin_phi, cos_theta]])
        self.rot_matrix = np.array([[ cos_alpha*cos_theta*cos_phi-sin_alpha*sin_phi,  cos_alpha*cos_theta*sin_phi+sin_alpha*cos_phi, -cos_alpha*sin_theta],\
                                    [-sin_alpha*cos_theta*cos_phi-cos_alpha*sin_phi, -sin_alpha*cos_theta*sin_phi+cos_alpha*cos_phi,  sin_alpha*sin_theta],\
                                    [                             sin_theta*cos_phi,                              sin_theta*sin_phi,            cos_theta]])

    def EB(self, t, pos):
        '''
            Return E and B field (3 components) at time t.
            t: time
            pos: position, should be a list-like with [x, y, z]
            Either t or (x, y, z) can be array of values, then the return value will be arraies of field components. But do not set both t and (x, y, z) to be array of values.
        '''
        # Shift to relative position of laser center at t=0
        x = pos[0] - self.x0
        y = pos[1] - self.y0
        z = pos[2] - self.z0
        # Coordinate (x, y, z) rotation to laser coordinate (x', y', z') at t=0
        x_prime = self.rot_matrix[0,0]*x + self.rot_matrix[0,1]*y + self.rot_matrix[0,2]*z
        y_prime = self.rot_matrix[1,0]*x + self.rot_matrix[1,1]*y + self.rot_matrix[1,2]*z
        z_prime = self.rot_matrix[2,0]*x + self.rot_matrix[2,1]*y + self.rot_matrix[2,2]*z
        # After axis rotation, laser is polarized in x' direction
        # and A = Ax = a0cos((z'-t)omega0+phi0)exp(-(r/w)**2-((z'-t)/tau)**2)
        # So E = -partial_t A and B = nabla cross A
        profile = self.a0*np.exp(-(np.square(x_prime)+np.square(y_prime))/np.square(self.w0) - np.square((z_prime-t)/self.tau))
        phase = (z_prime-t)*self.omega0+self.phase0*(np.pi/180.)
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        Ex_prime = -self.omega0*sin_phase*profile - 2*(z_prime-t)/self.tau**2*cos_phase*profile
        By_prime = Ex_prime
        # Bz_prime is very important for a correct laser field!
        Bz_prime = 2*y_prime/np.square(self.w0) * cos_phase * profile
        # Coordinate rotation
        # Rotate -theta along y', then rotate -phi along z. Rotation matrix is transpose of self.rot_matrix.
        Ex = self.rot_matrix[0,0]*Ex_prime
        Ey = self.rot_matrix[0,1]*Ex_prime
        Ez = self.rot_matrix[0,2]*Ex_prime
        Bx = self.rot_matrix[1,0]*By_prime + self.rot_matrix[2,0]*Bz_prime
        By = self.rot_matrix[1,1]*By_prime + self.rot_matrix[2,1]*Bz_prime
        Bz = self.rot_matrix[1,2]*By_prime + self.rot_matrix[2,2]*Bz_prime
        return np.array([Ex, Ey, Ez]), np.array([Bx, By, Bz])

    def interact_time_estimate(self, pos=[0., 0., 0.]):
        '''
            Estimate the interaction time at the position pos = [x, y, z].
        '''
        # Shift to relative position of laser center at t=0
        x = pos[0] - self.x0
        y = pos[1] - self.y0
        z = pos[2] - self.z0
        # Coordinate (x, y, z) rotation to laser coordinate (x', y', z') at t=0
        z_prime = self.rot_matrix[2,0]*x + self.rot_matrix[2,1]*y + self.rot_matrix[2,2]*z
        return z_prime+self.tau*3

if "__main__" == __name__:
    # Testing
    import matplotlib.pyplot as plt
    l1 = laser(z0=-2., dir_azimuth=180)
    print(l1.interact_time_estimate())
    t = np.linspace(0.,6.,256)
    E, B = l1.EB(t, [0., 0., 0.])
    plt.plot(t, E[0], ls='--', label = 'Ex')
    plt.plot(t, E[1], ls='--', label = 'Ey')
    plt.plot(t, E[2], ls='--', label = 'Ez')
    '''plt.plot(t, B[0], ls='--', label = 'Bx')
    plt.plot(t, B[1], ls='--', label = 'By')
    plt.plot(t, B[2], ls='--', label = 'Bz')'''
    plt.legend()
    plt.show()
