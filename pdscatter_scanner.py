from ponderomotive_scatter import multi_r0_pondscatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

######################################################################
####################### Class scatter_scanner ########################
#################### Inherite multi_r0_pondscatter ###################
######################################################################
class scatter_scanner_a0_tau(multi_r0_pondscatter):
    '''Inherite multi_r0_pondscatter to perform scan of a0 and tau. Length is normalized to w, and time is normalized to w/c. So tau is actually c*tau/w.'''
    def __init__(self, a0_range=[0.5, 1.5], a0_bins=4, tau_range=[0.5, 1.5], tau_bins=4, d_r0=None, r0_max=None):
        multi_r0_pondscatter.__init__(self, a0=a0_range[0], w=1., tau=tau_range[0], d_r0=d_r0, r0_max=r0_max)
        self._a0_bins = a0_bins
        self.a0_range = a0_range
        self._tau_bins = tau_bins
        self.tau_range = tau_range

######################## Property a0_range ########################
    def get_a0_range(self):
        return self._a0_range

    def set_a0_range(self, list_of_2_elements):
        self._a0_range = list_of_2_elements
        try:
            self._a0_array = np.linspace(self._a0_range[0], self._a0_range[1], self.a0_bins)
        except: print('Failed to set a0_array!')

    a0_range = property(get_a0_range, set_a0_range)

######################## Property a0_bins ########################
    def get_a0_bins(self):
        return self._a0_bins

    def set_a0_bins(self, n_bins):
        self._a0_bins = n_bins
        try:
            self._a0_array = np.linspace(self.a0_range[0], self.a0_range[1], self._a0_bins)
        except: print('Failed to set a0_array!')

    a0_bins = property(get_a0_bins, set_a0_bins)

######################## Property a0_array ########################
    def get_a0_array(self):
        return self._a0_array

    def set_a0_array(self, **ka0args):        
        raise RuntimeError('You cannot set a0_array! Set a0_range and a0_bins instead.')

    a0_array = property(get_a0_array, set_a0_array)

######################## Property tau_range ########################
    def get_tau_range(self):
        return self._tau_range

    def set_tau_range(self, list_of_2_elements):
        self._tau_range = list_of_2_elements
        try:
            self._tau_array = np.linspace(self._tau_range[0], self._tau_range[1], self.tau_bins)
        except: print('Failed to set tau_array!')

    tau_range = property(get_tau_range, set_tau_range)

######################## Property tau_bins ########################
    def get_tau_bins(self):
        return self._tau_bins

    def set_tau_bins(self, n_bins):
        self._tau_bins = n_bins
        try:
            self._tau_array = np.linspace(self.tau_range[0], self.tau_range[1], self._tau_bins)
        except: print('Failed to set tau_array!')

    tau_bins = property(get_tau_bins, set_tau_bins)

######################## Property tau_array ########################
    def get_tau_array(self):
        return self._tau_array

    def set_tau_array(self, **kwargs):        
        raise RuntimeError('You cannot set tau_array! Set tau_range and tau_bins instead.')

    tau_array = property(get_tau_array, set_tau_array)

######################## Method scan_a0_tau ########################
    def scan_a0_tau(self, pmax_or_psiM='psiM', if_plot_2Dpcolor=False, if_plot_lines=False, line_colors=None, **kwargs):
        '''
            For a0 in self.a0_array and tau in self.tau_array, use self.pz_pr_vs_r0() to obtain self.pz_final and self.pr_final. Then calculate p_final= (self.pr_final**2+self.pz_final**2)**0.5 and extract the maximum value of p_final and its corresponding theta.
            If if_plot_2Dpcolor is True, plot the data in 2D pseudocolor.
            If if_plot_lines is True, plot the data in lines, with xaxes of tau, and different a0 are in colors given in line_color. If line_color is None, generate a rainbow color series.
            pmax_or_psiM can be either 'psiM' or 'pmax'. If it is 'psiM', psi_M will be plotted. If it is 'pmax', p_max will be plotted.
        Returns:
            self.opt_theta_degree_vs_a0_tau
            self.max_p_final_vs_a0_tau
        '''
        self.opt_theta_degree_vs_a0_tau = np.zeros((self.tau_bins, self.a0_bins))
        self.max_p_final_vs_a0_tau = np.zeros((self.tau_bins, self.a0_bins))
        for a0_ind in range(self.a0_bins):
            self.a0 = self.a0_array[a0_ind]
            for tau_ind in range(self.tau_bins):
                self.tau = self.tau_array[tau_ind]
                self.pz_pr_vs_r0() # obtain self.pz_final and self.pr_final
                p_final = np.sqrt(self.pr_final**2+self.pz_final**2)
                opt_ind = p_final.argmax()
                self.max_p_final_vs_a0_tau[tau_ind, a0_ind] = p_final[opt_ind]
                sin_opt_theta = self.pr_final[opt_ind]/p_final[opt_ind]
                self.opt_theta_degree_vs_a0_tau[tau_ind, a0_ind] = np.arcsin(sin_opt_theta)*self.rad_to_degree
                print('a0 = {}, tau = {}, p = {}, theta = {}'.format(self.a0, self.tau, self.max_p_final_vs_a0_tau[tau_ind, a0_ind], self.opt_theta_degree_vs_a0_tau[tau_ind, a0_ind]))
        if if_plot_2Dpcolor:
            h_fig = plt.figure(figsize=(4.5, 3))
            h_ax1 = h_fig.add_subplot(121)
            h_plot1 = h_ax1.pcolormesh(self.a0_array, self.tau_array, self.opt_theta_degree_vs_a0_tau, **kwargs)
            h_ax1.set_xlabel('$a_0$')
            h_ax1.set_ylabel('$\\tau/w$')
            h_cb1 = plt.colorbar(h_plot1, ax=h_ax1)
            h_cb1.set_label('$\\theta_{{\\rm opt}}$ [$^\\circ$]')
            h_ax2 = h_fig.add_subplot(122)
            if 'psiM' == pmax_or_psiM:
                h_plot2 = h_ax2.pcolormesh(self.a0_array, self.tau_array, np.sqrt(1.+np.square(self.max_p_final_vs_a0_tau))-self.max_p_final_vs_a0_tau, **kwargs)
                h_cb2 = plt.colorbar(h_plot2, ax=h_ax2)
                h_cb2.set_label('$\\psi_{Mth}$')
            else:
                h_plot2 = h_ax2.pcolormesh(self.a0_array, self.tau_array, self.max_p_final_vs_a0_tau, **kwargs)
                h_cb2 = plt.colorbar(h_plot2, ax=h_ax2)
                h_cb2.set_label('$p_\\max$ [$\\rm m_e c$]')
            h_ax2.set_xlabel('$a_0$')
            h_ax2.set_ylabel('$\\tau/w$')

            a0_square_tau = [.5, 1., 2., 4., 8.]
            for i in range(len(a0_square_tau)):
                a0_min = max(a0_range[0], (a0_square_tau[i]/tau_range[1])**0.5)
                a0_array = np.linspace(a0_min, a0_range[1], 32)
                tau_array = a0_square_tau[i]/np.square(a0_array)
                h_ax1.plot(a0_array, tau_array, 'r-')
                h_ax2.plot(a0_array, tau_array, 'r-')
        if if_plot_lines:
            if line_colors is None: line_colors=cm.rainbow(np.linspace(0, 1, self.a0_bins))
            h_fig = plt.figure(figsize=(4.5, 3))
            h_ax1 = h_fig.add_subplot(121)
            h_ax2 = h_fig.add_subplot(122)
            for a0_ind in range(self.a0_bins):
                h_ax1.plot(self.tau_array, self.opt_theta_degree_vs_a0_tau[:, a0_ind], color=line_colors[a0_ind], label='$a_0 = {:.1f}$'.format(self.a0_array[a0_ind]))
                if 'psiM' == pmax_or_psiM:
                    h_ax2.plot(self.tau_array, np.sqrt(1.+np.square(self.max_p_final_vs_a0_tau[:, a0_ind]))-self.max_p_final_vs_a0_tau[:, a0_ind], color=line_colors[a0_ind], label='$a_0 = {:.1f}$'.format(self.a0_array[a0_ind]))
                else:
                    h_ax2.plot(self.tau_array, self.max_p_final_vs_a0_tau[:, a0_ind], color=line_colors[a0_ind], label='$a_0 = {:.1f}$'.format(self.a0_array[a0_ind]))
            h_ax1.minorticks_on()
            h_ax1.grid(True, which='both', linestyle=':')
            h_ax1.set_xlabel('$\\tau/w$')
            h_ax1.set_ylabel('$\\theta_{{\\rm opt}}$ [$^\\circ$]')
            h_ax1.legend()
            h_ax2.minorticks_on()
            #h_ax2.grid(True, which='both', linestyle=':')
            h_ax2.set_xlabel('$\\tau/w$')
            if 'psiM' == pmax_or_psiM: h_ax2.set_ylabel('$\\psi_{Mth}$')
            else: h_ax2.set_ylabel('$p_\\max$ [$\\rm m_e c$]')
            h_ax2.legend()
        return h_ax1, h_ax2

if __name__ == '__main__':
    '''a0_range = [0.8, 1.0]
    tau_range = [0.5, 7.]
    scanner = scatter_scanner_a0_tau(a0_range=a0_range, a0_bins=2, tau_range=tau_range, tau_bins=16, d_r0=1./1024)
    h_ax1, h_ax2 = scanner.scan_a0_tau(if_plot_2Dpcolor=False, if_plot_lines=True)

    line_colors=cm.rainbow(np.linspace(0, 1, scanner.a0_bins))
    datafilename = './pond_inj/os_beam3D_old.data'
    data = np.loadtxt(datafilename, delimiter=' ')
    tau_over_w = data[:,2]/data[:,1]*0.6005612043932249#*0.56516198
    theta = data[:,3]
    e_theta = data[:,4]
    h_ax1.errorbar(tau_over_w, theta, yerr=e_theta, color=line_colors[1], marker='o', mfc='none', capsize=5, ls='none')

    datafilename = './pond_inj/os_beam3D_new.data'
    data = np.loadtxt(datafilename, delimiter=' ')
    tau_over_w = data[:,2]/data[:,1]
    theta = data[:,3]
    e_theta = data[:,4]
    h_ax1.errorbar(tau_over_w[:2], theta[:2], yerr=e_theta[:2], color=line_colors[0], marker='o', mfc='none', capsize=5, ls='none')
    #h_ax1.errorbar(tau_over_w[2:], theta[2:], yerr=e_theta[2:], color=line_colors[2], marker='o', mfc='none', capsize=5, ls='none')'''

    a0_range = [0.4, 3.]
    tau_range = [0.2, 10.]
    scanner = scatter_scanner_a0_tau(a0_range=a0_range, a0_bins=16, tau_range=tau_range, tau_bins=32, d_r0=1./1024)
    h_ax1, h_ax2 = scanner.scan_a0_tau(if_plot_2Dpcolor=True, if_plot_lines=False)
    plt.tight_layout()
    plt.show()
    
