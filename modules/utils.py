"""Definitions of the functions used in *calculations.py*."""

import kwant
import numpy as np
# from .functions import onsite, onsite_qh, onsite_sc, hopping, hopping_qh, hopping_qh_e, hopping_sc
# from .functions import fermi_momenta, secular_equation, hole_probability, velocity, effective_tau
from .functions import onsite, hopping, onsite_qh, onsite_sc, hopping_qh, hopping_sc
from .functions import fermi_momenta, secular_equation, hole_probability, effective_tau
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import os.path
from joblib import Parallel, delayed
import multiprocessing
from scipy.optimize import minimize
import scipy.optimize as opt
import itertools
from alive_progress import alive_bar

##################################################
# Visualization
##################################################
def plot_device(device):
    r"""Plot the Kwant system.

    Be carefull, the colors are badly defined for negative angles.
    
    :param object device: The device.
    """
    
    def sc_region(pos):
        x, y = pos
        return x >= 1

    def qh_region(pos):
        x, y = pos
        return x <= -1

    def barrier(pos):
        x, y = pos
        return x == 0

    def color(site):
        if sc_region(site.pos):
            return 'orange'
        elif qh_region(site.pos):
            return 'blue'
        elif barrier(site.pos):
            return 'green'

    if device.device_type=='single_corner':
        syst = device.make_system(onsite, onsite_qh, onsite_sc, hopping, hopping_qh, hopping_sc)
    elif device.device_type=='two_corner':
        syst = device.make_system(onsite, hopping)
        
    kwant.plot(syst, site_color=color, show=False)
    plt.xlabel(r'$x [a]$')
    plt.ylabel(r'$y [a]$')
    plt.savefig('files/visualization/system/system_' + str(device.device_type)
                + '_' + str(device.geometry) + '.png', dpi=300)
    plt.show()


def plot_density(device, energy=0., fig_name=False):
    r"""Plot the probability density :math:`|u|^2 - |v|^2` of the incoming electron.

    :param object device: The device.
    :param float energy: Value of the energy (relative to the CAES Fermi level), default to 0.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    device_type = device.device_type
    params = device.params
    if device.device_type=='single_corner':
        syst = device.make_system(onsite, onsite_qh, onsite_sc, hopping, hopping_qh, hopping_sc)
    elif device.device_type=='two_corner':
        syst = device.make_system(onsite, hopping)
    with alive_bar(1, ctrl_c=False, title='Computing wave function') as bar:
        fsyst = syst.finalized()
        wf = kwant.wave_function(fsyst, energy, params=params)
        psi = wf(0)[0]  # wf(lead)[particle_type]
        bar()

    u, v = psi[::2], psi[1::2]
    density = np.abs(u) ** 2 - np.abs(v) ** 2
    cmap = plt.get_cmap('RdBu_r')

    fig, ax = plt.subplots(figsize = (12, 6))
    kwant.plotter.map(fsyst, density, oversampling=10, show=False, cmap=cmap, vmin=-0.3, vmax=0.3, ax=ax)
 
    L_qh = device.dimensions['L_qh']
    L_sc = device.dimensions['L_sc']
    L_interface = device.dimensions["L_interface"]
    
    L_2 = int(round(L_interface / 2))

    patch_top_qh_lead = mpl.patches.FancyBboxPatch(
                        (-L_qh, L_2),
                        L_qh/2,
                        L_qh/4,
                        fc="#bfbf99",
                        zorder=100,)
    
    patch_bot_qh_lead = mpl.patches.FancyBboxPatch(
                        (-L_qh, -L_2),
                        L_qh/2,
                        -L_qh/4,
                        fc='#bfbf99',
                        zorder=100,)

    patch_bot_hybrid_lead = mpl.patches.FancyBboxPatch(
                            (-L_qh, 0),
                            L_qh+2*L_sc/3,
                            -L_qh/6,
                            fc="tan",
                            zorder=100,)
    
        
    if device_type == 'two_corner':
        # theta_1 = device.theta_1
        # theta_2 = device.theta_2
        
        # patch_background = mpl.patches.FancyBboxPatch(
        #                         (-L_qh-10, -(L_2+L_qh/4+10)),
        #                         L_qh+10 + 2*L_sc/3+10,
        #                         L_2+L_qh/4+10 + L_2+L_qh/4+10,
        #                         fc='lightgray',
        #                         linewidth=0,
        #                         zorder=-1,)
        
        # ax.set_ylim([-(L_2+L_qh/4+10), L_2+L_qh/4+10])
        # ax.add_patch(patch_top_qh_lead)  
        # ax.add_patch(patch_bot_qh_lead)
        # ax.add_patch(patch_background)    
        ax.plot([0, 0], [-L_2, L_2], 'k', linewidth=4)


    elif device_type == 'single_corner':
        # theta_qh = device.theta_qh
        # theta_sc = device.theta_sc 
        
        patch_background = mpl.patches.FancyBboxPatch(
                                (-L_qh-10, -L_qh/6-10),
                                L_qh+10 + 2*L_sc/3+10,
                                L_qh/6+10 + L_2+L_qh/4+20,
                                fc='lightgray',
                                linewidth=0,
                                zorder=-1,)
        
        patch_top_background = mpl.patches.FancyBboxPatch(
                                (-L_qh-10,  L_2+L_qh/4+10),
                                L_qh+10 + 2*L_sc/3+10,
                                10,
                                fc='lightgray',
                                linewidth=0,
                                zorder=1,)
        
        patch_right_background = mpl.patches.FancyBboxPatch(
                                (2*L_sc/3,  -L_qh/6-10),
                                10,
                                L_qh/6+10 + L_2+L_qh/4+20,
                                fc='lightgray',
                                linewidth=0,
                                zorder=1,)
        
        ax.set_ylim([-L_qh/6-10, L_2+L_qh/4+20])
        ax.add_patch(patch_top_qh_lead)           
        ax.add_patch(patch_bot_hybrid_lead)
        ax.add_patch(patch_top_background)               
        ax.add_patch(patch_right_background)               
        ax.add_patch(patch_background)               
        ax.plot([0, 0], [0, L_2-1], 'k', linewidth=4)

    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.set_xlabel(r'$x/a$')
    ax.set_ylabel(r'$y/a$')
    ax.set_xlim([-L_qh-10, 2*L_sc/3+10])

    # Add label on the colorbar generated by Kwant
    norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, extend='both')
    cbar.set_label(r'$|\psi_e(\mathbf{r})|^2 - |\psi_h(\mathbf{r})|^2$')
    cbar.ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    cbar.ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')

    if not fig_name:
        fig.savefig('files/visualization/probability_density/density_' + str(device_type)
                    + '_' + str(device.geometry) + str(device.params_name)
                    + '_L=' + str(2*L_2)
                    + '_E=' + str(energy)
                    + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


##################################################
# Energy spectrum
##################################################
def plot_spectrum_TB(device, from_data=True):
    """Plot tight-binding spectrum.

    The data and plots are saved in the directory 'files/energy_spectrum/tight_binding'.

    :param object device: Infinite QH-SC interface.
    :param bool from_data: If true the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist.
    """
    params = device.params
    def TB_spectrum_data():
        flead = device.make_system(onsite, hopping).finalized()
        momenta = np.linspace(-np.pi/params['a'], np.pi/params['a'], 501)
        bands = kwant.physics.Bands(flead, params=params)
        npts = len(momenta)
        energies = []
        with alive_bar(npts, ctrl_c=False, title='Computing TB spectrum') as bar:
            for i in range(npts):
                energies.append(bands(momenta[i]))
                bar()
        energies = np.asarray(energies)
        np.save('files/energy_spectrum/tight_binding/data/TB_momenta' + str(device.params_name) + '.npy', momenta)
        np.save('files/energy_spectrum/tight_binding/data/TB_energies' + str(device.params_name) + '.npy', energies)
        return [momenta, energies]

    if from_data:
        if os.path.isfile('files/energy_spectrum/tight_binding/data/TB_momenta' + str(device.params_name) + '.npy') and \
                os.path.isfile('files/energy_spectrum/tight_binding/data/TB_energies' + str(device.params_name) + '.npy'):
            momenta_TB = np.load('files/energy_spectrum/tight_binding/data/TB_momenta' + str(device.params_name) + '.npy',
                                 allow_pickle=True)
            energies_TB = np.load('files/energy_spectrum/tight_binding/data/TB_energies' + str(device.params_name) + '.npy',
                                  allow_pickle=True)
        else:
            momenta_TB, energies_TB = TB_spectrum_data()
    else:
        momenta_TB, energies_TB = TB_spectrum_data()

    # Plot
    fig, ax = plt.subplots()

    # Edit the major and minor ticks of the x and y axes
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(which='major')
    ax.set_axisbelow(True)    
    ax.set_xlim([-3, 3])
    ax.set_ylim([-6, 6])
    ax.set(xlabel=r'$k/k_F$', ylabel=r'$E/\Delta$')
    ax.plot(momenta_TB / device.kF_qh, energies_TB / device.params['delta'], linewidth=2)
        
    fig.savefig('files/energy_spectrum/tight_binding/plots/spectrum_TB' + str(device.params_name) + '.pdf',
                bbox_inches='tight', transparent=True)
    
    plt.show()


def plot_spectrum_micro(params, from_data=True, fig_name=False, show_k0=False, qp_labels=False):
    """Plot microscopic spectrum.

    The data and plots are saved in the directory 'files/energy_spectrum/microscopic'.

    :param dict params: The system's parameters.
    :param bool from_data: If true the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    :param str show_k0: Show the position of k0, default to False.
    :param str qp_labels: Show the quasiparticles labels, default to False.
    """

    params_name = ''.join(['_%s=%s' % (key, value) for key, value in params.items()])
    m = 1 / (2 * params["t"] * params["a"] ** 2)
    omega = 2 * params["mu_qh"] / params["nu"]
    lB = 1/np.sqrt(m*omega)
    kF_qh = np.sqrt(2 * m * params["mu_qh"])
    vF_qh = kF_qh / m
    V_barrier = params['Z'] * vF_qh / 2
    k0 = fermi_momenta(m, m, params["mu_qh"], params["mu_sc"], params["nu"], params["delta"], V_barrier)[0]

    def microscopic_spectrum_data():
        def f(E, k):
            return secular_equation(k, E, m, m, params['mu_qh'], params['mu_sc'],
                                    omega, params['delta'], V_barrier)

        def find_root(E_min, E_max, k):
            while np.sign(f(E_max, k) * f(E_min, k)) > 0:
                E_min = E_min - 0.1 * params['delta']
                E_max = E_max + 0.1 * params['delta']
            value = opt.brentq(f, E_min, E_max, args=(k,))
            return value

        ky_list = 1/lB * np.linspace(-2.5, 2.5, 5001)
        Elist = []
        dE = 0.999*params['delta']
        Emin_values = np.arange(-0.999*params['delta'], 0.999*params['delta'], dE)
        n_Emin = len(Emin_values)
        n_ky = len(ky_list)
        for j in range(n_Emin):
            Emin = Emin_values[j]
            Evalues = []
            with alive_bar(n_ky, ctrl_c=False, title=f'Computing micro spectrum ({j+1}/{n_Emin})') as bar:
                for ky in ky_list:
                    sol = find_root(E_min=Emin, E_max=Emin + dE, k=ky)
                    Evalues.append(sol)
                    bar()
                Elist.append(Evalues)

        Elist = np.asarray(Elist)
        data = [ky_list, Elist]
        np.save('files/energy_spectrum/microscopic/data/microscopic_spectrum' + str(params_name) + '.npy', data)

        return data

    if from_data:
        if os.path.isfile('files/energy_spectrum/microscopic/data/microscopic_spectrum' + str(params_name) + '.npy'):
            momenta_micro, energies_micro = np.load('files/energy_spectrum/microscopic/data/microscopic_spectrum'
                                                    + str(params_name) + '.npy', allow_pickle=True)
        else:
            momenta_micro, energies_micro = microscopic_spectrum_data()
    else:
        momenta_micro, energies_micro = microscopic_spectrum_data()

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-1, 1])
    ax.grid(which='major')
    ax.set_axisbelow(True)
    ax.set(xlabel=r'$k_y l_B$', ylabel=r'$E/\Delta$')

    for i in range(len(energies_micro)):
        ax.plot(momenta_micro*lB, energies_micro[i] / params['delta'], 'k.', markersize=2)
    
    if show_k0:
        trans = ax.get_xaxis_transform()
        plt.axvline(x=-k0*lB, c='r')
        plt.axvline(x=k0*lB, c='r')
        plt.text(-k0*lB-.38, 0-.12, r'$-k_0 l_B$', c='r', transform=trans)
        plt.text(k0*lB-.2, 0-.12, r'$k_0 l_B$',  c='r', transform=trans)
        
    if qp_labels:
        ax.set_xlim([-2, 2])
        trans = ax.get_xaxis_transform()
        plt.text(-k0*lB-.59, .25-.015, r'qe', c='k', transform=trans)
        plt.text(k0*lB+.38, .75-.015, r'qh',  c='k', transform=trans)
    
    if not fig_name:
        fig.savefig('files/energy_spectrum/microscopic/plots/spectrum_micro' + str(params_name) + '.pdf',
                    bbox_inches='tight', transparent=True)
    
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
        
    plt.show()


def plot_spectrum_comparison(device, params, fig_name=False, from_data=True):
    """Comparison between microscopic and tight-binding spectrums.

    The plots are saved in the directory 'files/energy_spectrum/comparison/plots'.

    :param object device: Infinite QH-SC interface.
    :param dict params: The system's parameters.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    :param bool from_data: If true the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist.
    """

    def TB_spectrum_data():
        flead = device.make_system(onsite, hopping).finalized()
        bands = kwant.physics.Bands(flead, params=params)
        momenta = np.linspace(-np.pi/params['a'], np.pi/params['a'], 501)
        npts = len(momenta)
        energies = []
        with alive_bar(npts, ctrl_c=False, title='Computing TB spectrum') as bar:
            for j in range(npts):
                energies.append(bands(momenta[j]))
                bar()
        energies = np.asarray(energies)
        np.save('files/energy_spectrum/tight_binding/data/TB_momenta' + str(device.params_name) + '.npy', momenta)
        np.save('files/energy_spectrum/tight_binding/data/TB_energies' + str(device.params_name) + '.npy', energies)
        return [momenta, energies]

    def microscopic_spectrum_data():
        def f(E, k):
            return secular_equation(k, E, device.m, device.m, params['mu_qh'], params['mu_sc'],
                                    device.omega, params['delta'], device.V_barrier)

        def find_root(E_min, E_max, k):
            while np.sign(f(E_max, k) * f(E_min, k)) > 0:
                E_min = E_min - 0.1 * params['delta']
                E_max = E_max + 0.1 * params['delta']
            value = opt.brentq(f, E_min, E_max, args=(k,))
            return value

        ky_list = np.linspace(-2 * device.kF_qh, 2 * device.kF_qh, 5001)
        Elist = []
        dE = 0.999*params['delta']
        Emin_values = np.arange(-0.999*params['delta'], 0.999*params['delta'], dE)
        n_Emin = len(Emin_values)
        n_ky = len(ky_list)
        for j in range(n_Emin):
            Emin = Emin_values[j]
            Evalues = []
            with alive_bar(n_ky, ctrl_c=False, title=f'Computing micro spectrum ({j+1}/{n_Emin})') as bar:
                for ky in ky_list:
                    sol = find_root(E_min=Emin, E_max=Emin + dE, k=ky)
                    Evalues.append(sol)
                    bar()
                Elist.append(Evalues)

        Elist = np.asarray(Elist)
        data = [ky_list, Elist]
        np.save('files/energy_spectrum/microscopic/data/microscopic_spectrum' + str(device.params_name) + '.npy', data)

        return data

    if from_data:
        if os.path.isfile('files/energy_spectrum/tight_binding/data/TB_momenta' + str(device.params_name)
                          + '.npy') and \
                os.path.isfile('files/energy_spectrum/tight_binding/data/TB_energies' + str(device.params_name)
                               + '.npy'):
            momenta_TB = np.load('files/energy_spectrum/tight_binding/data/TB_momenta' + str(device.params_name)
                                 + '.npy',
                                 allow_pickle=True)
            energies_TB = np.load('files/energy_spectrum/tight_binding/data/TB_energies' + str(device.params_name)
                                  + '.npy',
                                  allow_pickle=True)
        else:
            momenta_TB, energies_TB = TB_spectrum_data()

        if os.path.isfile('files/energy_spectrum/microscopic/data/microscopic_spectrum' + str(device.params_name)
                          + '.npy'):
            momenta_micro, energies_micro = np.load('files/energy_spectrum/microscopic/data/microscopic_spectrum'
                                                    + str(device.params_name) + '.npy', allow_pickle=True)
        else:
            momenta_micro, energies_micro = microscopic_spectrum_data()

    else:
        momenta_TB, energies_TB = TB_spectrum_data()
        momenta_micro, energies_micro = microscopic_spectrum_data()

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    # Edit the major and minor ticks of the x and y axes
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.grid(which='major')
    ax.set_axisbelow(True)

    for i in range(len(energies_micro)):
        ax.plot(momenta_micro / device.kF_qh, energies_micro[i] / params['delta'], 'r.', markersize=4.)
    ax.plot(momenta_micro / device.kF_qh, energies_micro[0] / params['delta'], 'r.', markersize=4., label='micro')
    ax.plot(momenta_TB[0] / device.kF_qh, energies_TB[0][0] / params['delta'], 'k', markersize=1., label='TB')
    ax.plot(momenta_TB / device.kF_qh, energies_TB / params['delta'], 'k', markersize=1.)

    ax.set(xlabel=r'$k/k_F$', ylabel=r'$E/\Delta$')

    plt.legend(loc='upper right')
    if not fig_name:
        fig.savefig('files/energy_spectrum/comparison/plots/spectrum_comparison' + str(device.params_name) + '.pdf',
                    bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)

    plt.show()


##################################################
# Momentum k0
##################################################
def plot_k0_vs_nu(nus, params, fig_name=False):
    """Plot the momentum at the Fermi level k0 versus nu.

    The plot is saved in the directory 'files/momentum_k0/varying_nu/plots'.

    :param list nus: The values of the filling factor.
    :param dict params: The system's parameters.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """

    m = 1 / (2 * params['t'] * params['a'] ** 2)
    m_qh = m
    m_sc = m
    kF_qh = np.sqrt(2 * m * params["mu_qh"])
    vF_qh = kF_qh / m_qh
    V_barrier = params['Z'] * vF_qh / 2

    
    k0_values = []
    with alive_bar(len(nus), ctrl_c=False,
                   title=f'Computing momenta') as bar:
        for j in range(len(nus)):
            k0 = fermi_momenta(m_qh, m_sc, params['mu_qh'], params['mu_sc'], 
                               nus[j], params['delta'], V_barrier)[0]
            k0_values.append(k0)
            bar()

    k0_values = np.asarray(k0_values)
    
    # Plot
    plot_name = '_a=' + str(params['a']) + '_t=' + str(params['t']) \
                + '_mu_qh=' + str(params['mu_qh']) + '_mu_sc=' + str(params['mu_sc']) + '_delta=' + str(params['delta']) + '_Z=' + str(params['Z'])

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.set_xlim([1, 3])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set(xlabel=r'$\nu$', ylabel=r'$k_0/k_F^{QH}$')
    ax.plot(nus, k0_values/kF_qh, linewidth=2)
    if not fig_name:
        fig.savefig('files/momentum_k0/varying_nu/plots/k0_vs_nu_' 
                    + str(plot_name) + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)

    plt.show()


def plot_k0_vs_Z_various_fillings(nus, Zs, params, fig_name=False):
    """Plot the momentum k0 versus Z for various values of the filling factor.

    The plot is saved in the directory 'files/momentum_k0/varying_Z/plots'.

    :param list nus: The values of the filling factor.
    :param list Zs: The values of the barrier strength.
    :param dict params: The system's parameters.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """

    m = 1 / (2 * params['t'] * params['a'] ** 2)
    m_qh = m
    m_sc = m
    kF_qh = np.sqrt(2 * m_sc * params['mu_qh'])
    omega = 2 * params["mu_qh"] / params["nu"]
    lB = 1/np.sqrt(m*omega)
    vF_qh = kF_qh / m_qh

    Zs_vals = []
    k0_vals = []
    for i in range(len(nus)):
        k0_values = []
        with alive_bar(len(Zs), ctrl_c=False,
                       title=f'Computing momenta for nu = {str(nus[i])} ({i+1}/{len(nus)})') as bar:
            for j in range(len(Zs)):
                V_barrier = Zs[j] * vF_qh / 2
                k0 = fermi_momenta(m_qh, m_sc, params['mu_qh'], params['mu_sc'], nus[i], params['delta'], V_barrier)[0]
                k0_values.append(k0)
                bar()
            Zs_vals.append(Zs)
            k0_vals.append(k0_values)

    k0_vals = np.asarray(k0_vals)

    # Plot
    plot_name = '_a=' + str(params['a']) + '_t=' + str(params['t']) \
                + '_mu_qh=' + str(params['mu_qh']) + '_mu_sc=' + str(params['mu_sc']) + '_delta=' + str(params['delta'])

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.set_xlim([np.min(Zs), np.max(Zs)])
    ax.set_ylim([0, 1.8])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set(xlabel=r'$Z$', ylabel=r'$k_0 l_B$')
    for i in range(len(nus)):
        ax.plot(Zs_vals[i], k0_vals[i]*lB, linewidth=2, label=r'$\nu = $ ' + str(nus[i]))
    if len(nus) > 1:
        plt.legend(ncol=2, fontsize=19, loc=(.2, .53))
    if not fig_name:
        fig.savefig('files/momentum_k0/varying_Z/plots/k0_vs_Z_various_fillings'
                    + str(plot_name) + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


##################################################
# Andreev transmission and hole content
##################################################
def compute_corner_transmissions(device, energy=0.):
    """Compute the corner transmission amplitudes with Kwant.

    :param object device: The QH-SC corner.
    :param float energy: Value of the energy, default to 0.

    :returns: The normal and Andreev amplitudes : t_ee, t_he.
    :rtype: list
    """
    params = device.params
    fsyst = device.make_system(onsite, onsite_qh, onsite_sc, 
                               hopping, hopping_qh, hopping_sc).finalized()
    # compute the S-matrix at zero energy
    sm = kwant.smatrix(fsyst, energy=energy, params=params)
    # compute the transmission matrix
    tm = sm.submatrix(1, 0)
    # compute the transmission amplitudes
    if np.shape(tm) == (2, 2):
        t_he = tm[0][0]
        t_ee = tm[1][0]
    elif np.shape(tm) == (4, 2):    # if track states
        t_he = tm[1][0]
        t_ee = tm[2][0]
    else:
        print('Unknown shape of the transmission matrix.')
        if energy >= 0.:
            t_he = tm[0][0]
        else:
            t_he = tm[-1][-1]
        t_ee = tm[1][0]
    return t_ee, t_he


def plot_tau_vs_theta_qh_various_fillings(nus, thetas, device, fig_name=False, from_data=True, show_only_commensurate=False):
    """Plot the corner's Andreev transmission versus the QH angle for various values of the filling factor.

    The data and the plot are saved in the directory 
    'files/andreev_and_hole_prob/andreev_transmission/varying_theta'.

    :param list nus: The values of the filling factor.
    :param list thetas: The values of the QH angle.
    :param dict params: The system's parameters.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    :param bool from_data: If True the spectrum is plotted using the stored data.
                          If False the data are computed even if they exist.
    :param bool show_only_commensurate: If True only the commensurate angles are shown. Defaults to False.
    """
    params = device.params
    def compute_tau_vs_theta(nu_calc):
        _params = dict(a=params['a'], t=params['t'], mu_qh=params['mu_qh'], mu_sc=params['mu_sc'],
                       delta=params['delta'], nu=nu_calc, Z=params['Z'])
        name_calc = ''.join(['_%s=%s' % (key, value) for key, value in _params.items()])

        def compute_tau_val(theta):
            device.theta = theta
            # plot_device(device)
            fsyst = device.make_system(onsite, onsite_qh, onsite_sc, 
                                       hopping, hopping_qh, hopping_sc).finalized()
            # compute the S-matrix at the Fermi level
            sm = kwant.smatrix(fsyst, energy=0.0, params=_params)
            # compute the transmission matrix
            tm = sm.submatrix(1, 0)
            # compute the transmission amplitudes
            if np.shape(tm) == (2, 2):
                t_he = tm[0][0]
            elif np.shape(tm) == (4, 2):  # if track states cross the Fermi level
                t_he = tm[1][0]
            else:
                print('Unknown shape of the transmission matrix.')
                t_he = tm[0][0]
            tau = np.abs(t_he) ** 2
            return tau

        num_cores = multiprocessing.cpu_count()

        taus = Parallel(n_jobs=num_cores,
                        verbose=len(thetas))(delayed(compute_tau_val)(thetas[k]) for k in range(len(thetas)))
        taus = np.asarray(taus)
        np.save('files/andreev_and_hole_prob/andreev_transmission/varying_theta_qh/data/tau_vs_theta_qh'
                + '_theta_sc=' + str(device.theta_sc) + str(name_calc) + '.npy', [thetas, taus])
        return [thetas, taus]

    theta_vals = []
    tau_vals = []
    i = 0
    for i in range(len(nus)):
        nu_val = nus[i]
        _params2 = dict(a=params['a'], t=params['t'], mu_qh=params['mu_qh'], mu_sc=params['mu_sc'],
                        delta=params['delta'], nu=nu_val, Z=params['Z'])
        name_calc2 = ''.join(['_%s=%s' % (key, value) for key, value in _params2.items()])

        i += 1

        if from_data:
            if os.path.isfile('files/andreev_and_hole_prob/andreev_transmission/varying_theta_qh/data/tau_vs_theta_qh'
                              + '_theta_sc=' + str(device.theta_sc) + str(name_calc2) + '.npy'):
                data = np.load('files/andreev_and_hole_prob/andreev_transmission/varying_theta_qh/data/tau_vs_theta_qh'
                               + '_theta_sc=' + str(device.theta_sc) + str(name_calc2) + '.npy', allow_pickle=True)
            else:
                print('--------------------------------------------------------------------')
                print(f'Computing transmissions for nu = {str(nu_val)} ({i}/{len(nus)})')
                print('--------------------------------------------------------------------')
                data = compute_tau_vs_theta(nu_val)

        else:
            print('--------------------------------------------------------------------')
            print(f'Computing transmissions for nu = {str(nu_val)} ({i}/{len(nus)})')
            print('--------------------------------------------------------------------')
            data = compute_tau_vs_theta(nu_val)

        theta_list, tau_values = data

        theta_vals.append(theta_list)
        tau_vals.append(tau_values)

    plot_name = '_a=' + str(params['a']) + '_t=' + str(params['t']) \
                + '_mu_qh=' + str(params['mu_qh']) \
                + '_mu_sc=' + str(params['mu_sc']) \
                + '_delta=' + str(params['delta']) \
                + '_Z=' + str(params['Z']) \

    markers = ['-o', '-s', '-x', '-v', '-d', '-+', '-^', '-<', '->']
    
    if show_only_commensurate:
        theta_vals_commensurate = []
        tau_vals_commensurate = []
        for j in range(len(nus)):
            indices = [np.where(theta_vals[j]==i) for i in [-45, 0, 45, 45, 90, 135]]
            theta_vals_commensurate.append([theta_vals[j][i] for i in indices])
            tau_vals_commensurate.append([tau_vals[j][i] for i in indices])
        theta_vals = theta_vals_commensurate
        tau_vals = tau_vals_commensurate

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    # ax.set_xlim([-45, 135])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set(xlabel=r'$\theta_{QH}$ $\mathrm{(deg)}$', ylabel=r'$\tau$')
    for i in range(len(nus)):
        ax.plot(theta_vals[i], tau_vals[i], markers[i], label=r'$\nu = $ ' + str(nus[i]))
    fig.legend(ncol=2, fontsize=19)
    if not fig_name:
        fig.savefig('files/andreev_and_hole_prob/andreev_transmission/varying_theta_qh/plots/tau_vs_theta_qh'
                    + '_theta_sc=' + str(device.theta_sc) + str(plot_name) + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


def plot_tau_vs_theta_sc_various_fillings(nus, thetas, device, fig_name=False, from_data=True, show_only_commensurate=False):
    """Plot the corner's Andreev transmission versus the SC angle for various values of the filling factor.

    The data and the plot are saved in the directory 
    'files/andreev_and_hole_prob/andreev_transmission/varying_theta_sc'.

    :param list nus: The values of the filling factor.
    :param list thetas: The values of the SC angle.
    :param dict params: The system's parameters.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    :param bool from_data: If True the spectrum is plotted using the stored data.
                          If False the data are computed even if they exist.
    :param bool show_only_commensurate: If True only the commensurate angles are shown. Defaults to False.
    """
    params = device.params
    def compute_tau_vs_theta(nu_calc):
        _params = dict(a=params['a'], t=params['t'], mu_qh=params['mu_qh'], mu_sc=params['mu_sc'],
                       delta=params['delta'], nu=nu_calc, Z=params['Z'])
        name_calc = ''.join(['_%s=%s' % (key, value) for key, value in _params.items()])

        def compute_tau_val(theta):
            device.theta_sc = theta
            fsyst = device.make_system(onsite, onsite_qh, onsite_sc, 
                                       hopping, hopping_qh, hopping_sc).finalized()
            # compute the S-matrix at the Fermi level
            sm = kwant.smatrix(fsyst, energy=0.0, params=_params)
            # compute the transmission matrix
            tm = sm.submatrix(1, 0)
            # compute the transmission amplitudes
            if np.shape(tm) == (2, 2):
                t_he = tm[0][0]
            elif np.shape(tm) == (4, 2):  # if track states cross the Fermi level
                t_he = tm[1][0]
            else:
                print('Unknown shape of the transmission matrix.')
                t_he = tm[0][0]
            tau = np.abs(t_he) ** 2
            return tau

        num_cores = multiprocessing.cpu_count()

        taus = Parallel(n_jobs=num_cores,
                        verbose=len(thetas))(delayed(compute_tau_val)(thetas[k]) for k in range(len(thetas)))
        taus = np.asarray(taus)
        np.save('files/andreev_and_hole_prob/andreev_transmission/varying_theta_sc/data/tau_vs_theta_sc'
                + '_theta_qh=' + str(device.theta_qh) + str(name_calc) + '.npy', [thetas, taus])
        return [thetas, taus]

    theta_vals = []
    tau_vals = []
    i = 0
    for i in range(len(nus)):
        nu_val = nus[i]
        _params2 = dict(a=params['a'], t=params['t'], mu_qh=params['mu_qh'], mu_sc=params['mu_sc'],
                        delta=params['delta'], nu=nu_val, Z=params['Z'])
        name_calc2 = ''.join(['_%s=%s' % (key, value) for key, value in _params2.items()])

        i += 1

        if from_data:
            if os.path.isfile('files/andreev_and_hole_prob/andreev_transmission/varying_theta_sc/data/tau_vs_theta_sc'
                              + '_theta_qh=' + str(device.theta_qh) + str(name_calc2) + '.npy'):
                data = np.load('files/andreev_and_hole_prob/andreev_transmission/varying_theta_sc/data/tau_vs_theta_sc'
                               + '_theta_qh=' + str(device.theta_qh) + str(name_calc2) + '.npy', allow_pickle=True)
            else:
                print('--------------------------------------------------------------------')
                print(f'Computing transmissions for nu = {str(nu_val)} ({i}/{len(nus)})')
                print('--------------------------------------------------------------------')
                data = compute_tau_vs_theta(nu_val)

        else:
            print('--------------------------------------------------------------------')
            print(f'Computing transmissions for nu = {str(nu_val)} ({i}/{len(nus)})')
            print('--------------------------------------------------------------------')
            data = compute_tau_vs_theta(nu_val)

        theta_list, tau_values = data

        theta_vals.append(theta_list)
        tau_vals.append(tau_values)

    plot_name = '_a=' + str(params['a']) + '_t=' + str(params['t']) \
                + '_mu_qh=' + str(params['mu_qh']) \
                + '_mu_sc=' + str(params['mu_sc']) \
                + '_delta=' + str(params['delta']) \
                + '_Z=' + str(params['Z']) \

    markers = ['-o', '-s', '-x', '-v', '-d', '-+', '-^', '-<', '->']

    if show_only_commensurate:
        theta_vals_commensurate = []
        tau_vals_commensurate = []
        for j in range(len(nus)):
            indices = [np.where(theta_vals[j]==i) for i in [-45, 0, 45, 45, 90, 135]]
            theta_vals_commensurate.append([theta_vals[j][i] for i in indices])
            tau_vals_commensurate.append([tau_vals[j][i] for i in indices])
        theta_vals = theta_vals_commensurate
        tau_vals = tau_vals_commensurate

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    # ax.set_xlim([-45, 135])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set(xlabel=r'$\theta_{SC}$ $\mathrm{(deg)}$', ylabel=r'$\tau$')
    for i in range(len(nus)):
        ax.plot(theta_vals[i], tau_vals[i], markers[i], label=r'$\nu = $ ' + str(nus[i]))
    fig.legend(ncol=2, fontsize=19)
    if not fig_name:
        fig.savefig('files/andreev_and_hole_prob/andreev_transmission/varying_theta_sc/plots/tau_vs_theta_sc'
                    + '_theta_qh=' + str(device.theta_qh) + str(plot_name) + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


def plot_fh_p_vs_Z_various_fillings(nus, Zs, params, fig_name=False):
    """Plot the hole content f_h^+ vs Z for various values of the filling factor.

    The plot is saved in the directory 'files/andreev_and_hole_prob/hole_probability/varying_Z/plots'.

    :param list nus: The values of the filling factor.
    :param list Zs: The values of the barrier strength.
    :param dict params: The system's parameters.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """

    m = 1 / (2 * params['t'] * params['a'] ** 2)
    m_qh = m
    m_sc = m
    kF_qh = np.sqrt(2 * m * params["mu_qh"])
    vF_qh = kF_qh / m

    Zs_vals = []
    B_vals = []
    for i in range(len(nus)):
        B_values = []
        with alive_bar(len(Zs), ctrl_c=False,
                       title=f'Computing hole probs for nu = {str(nus[i])} ({i+1}/{len(nus)})') as bar:
            for j in range(len(Zs)):
                V_barrier = Zs[j] * vF_qh / 2
                B = hole_probability(m_qh, m_sc, params['mu_qh'], params['mu_sc'], nus[i], params['delta'], V_barrier)
                B_values.append(B)
                bar()
            Zs_vals.append(Zs)
            B_vals.append(B_values)

    # Plot
    plot_name = '_a=' + str(params['a']) + '_t=' + str(params['t']) \
                + '_mu_qh=' + str(params['mu_qh']) + '_mu_sc=' + str(params['mu_sc']) + '_delta=' + str(params['delta'])

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.set_xlim([np.min(Zs), np.max(Zs)])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set(xlabel=r'$Z$', ylabel=r'$f_h^+$')
    for i in range(len(nus)):
        ax.plot(Zs_vals[i], B_vals[i], linewidth=2, label=r'$\nu = $ ' + str(nus[i]))
    plt.legend(ncol=1, fontsize=19)
    if not fig_name:
        fig.savefig('files/andreev_and_hole_prob/hole_probability/varying_Z/plots/B_vs_Z_'
                    + str(plot_name) + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


##################################################
# Downstream conductance (at zero T)
##################################################
def compute_downstream_conductance_TB(device):
    """Compute the (zero-temperature) downstream conductance with Kwant.

    :param object device: The QH-SC junction.

    :returns: The value of the downstream conductance.
    :rtype: float
    """
    params = device.params
    try:
        # compute the S-matrix at zero energy
        fsyst = device.make_system(onsite, hopping).finalized()
        smatrix = kwant.smatrix(fsyst, energy=0.0, params=params)
        # Probability of outgoing electron from superconducting contact
        Pe = smatrix.transmission((1, 0), (0, 0))
        # Probability of outgoing hole (Andreev) from superconducting contact
        Ph = smatrix.transmission((1, 1), (0, 0))
        # Gd value
        value = (Pe - Ph)

    except ValueError:
        print('The dimensions and geometry do not allow to build the system. '
              'The downstream conductance is thus taken as Gd = np.nan.')
        value = np.nan

    return value


def compute_delta_b_and_phi_b(device, L_b, v_b, mu_b):
    """Compute the effective barrier parameters delta_b and phi_b.
    
    Here have to give L_b, v_b, and mu_b as inputs. 

    :param object device: The QH-SC junction.
    :param str L_b: Length of the barrier.
    :param str v_b: Velocity in the barrier.
    :param str mu_b: Chemical potential in the barrier.
    
    :returns: The values of delta_b and phi_b.
    :rtype: array
    """
    params = device.params
    m = 1 / (2 * params['t'] * params['a'] ** 2)
    m_qh = m
    m_sc = m
    kF_qh = np.sqrt(2 * m_qh * params["mu_qh"])
    vF_qh = kF_qh / m_qh
    V_barrier = params['Z'] * vF_qh / 2

    mu_qh = params['mu_qh']
    mu_sc = params['mu_sc']
    nu = params['nu']
    delta = params['delta']

    tau_0 = hole_probability(m_qh, m_sc, mu_qh, mu_sc, nu, delta, V_barrier)
    tau_TB = np.abs(compute_corner_transmissions(device)[1])**2
    
    def fun(params):
        delta_b, phi_b = params
        return np.abs(effective_tau(tau_0, L_b, v_b, mu_b, delta_b, phi_b) - tau_TB)

    bnds = ((0, None), (0, 2*np.pi))
    first_guess = [0., 0.]
    res = opt.minimize(fun, first_guess, method = 'SLSQP', bounds=bnds)
    sols=res.x
    deviation = np.abs(fun(sols))
    
    if deviation > 1e-6:
        first_guess = [0., np.pi]
        res = opt.minimize(fun, first_guess, method = 'SLSQP', bounds=bnds)
        sols=res.x
        deviation = np.abs(fun(sols))
        
    print(f'delta = {str(delta)}')
    print(f'delta_b, phi_b = {str(res.x)}')
    print(f'tau_TB: = {str(tau_TB)}')
    print(f'deviation: |tau_ana - tau_TB| = {str(deviation)}')
    return sols


def plot_downstream_conductance_comparison(Ls, device, device_1, device_2, L_b, v_b, mu_b, 
                                           fig_name=False, from_data=True):
    """Comparison between analytical and tight-binding conductance.

    The analytical formula is shifted in order to recover the tight-binding simulation at large L.
    We we need to give L_b, v_b, and mu_b as inputs. 

    The shifted data and the plot are saved in the directory 'files/downstream_conductance/comparison'.

    :param list Ls: The values of the interface's length.
    :param object device: The QH-SC junction.
    :param object device_1: The first QH-SC corner.
    :param object device_2: The second QH-SC corner.
    :param str L_b: Length of the barrier.
    :param str v_b: Velocity in the barrier.
    :param str mu_b: Chemical potential in the barrier.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    :param bool from_data: If True the spectrum is plotted using existing data.
                          If False the data are computed even if they exist.
    """
    params = device.params
    m = 1 / (2 * params["t"] * params["a"] ** 2)
    mu_qh = params["mu_qh"]
    mu_sc = params["mu_sc"]
    delta = params["delta"]
    nu = params["nu"]
    kF_qh = np.sqrt(2 * m * params["mu_qh"])
    vF_qh = kF_qh / m
    V_barrier = params['Z'] * vF_qh / 2
    k0 = fermi_momenta(m, m, mu_qh, mu_sc, nu, delta, V_barrier)[0]
    tau_0 = hole_probability(m, m, mu_qh, mu_sc, nu, delta, V_barrier)

    def calculate_Gd_TB(L):
        device.dimensions = {**device.dimensions, **{"L_interface": L}}
        from modules import utils
        return utils.compute_downstream_conductance_TB(device=device)

    if from_data:
        if os.path.isfile('files/downstream_conductance/comparison/data/conductance_TB_'
                          + str(device.geometry) + str(device.params_name) + '.npy'):
            L_vals_TB, Gd_values_TB = np.load('files/downstream_conductance/comparison/data/conductance_TB_'
                                              + str(device.geometry) + str(device.params_name) + '.npy', allow_pickle=True)
        else:
            print('--------------------------------------------------------------------')
            print(f'Computing TB conductance')
            print('--------------------------------------------------------------------')
            L_vals_TB = Ls
            num_cores = multiprocessing.cpu_count()
            print(f'N_tasks = {str(len(L_vals_TB))}')
            Gd_values_TB = Parallel(n_jobs=num_cores, verbose=10)(delayed(calculate_Gd_TB)(L) for L in L_vals_TB)
            data = [L_vals_TB, Gd_values_TB]
            np.save('files/downstream_conductance/comparison/data/conductance_TB_'
                    + str(device.geometry) + str(device.params_name) + '.npy', data)

        if os.path.isfile('files/downstream_conductance/comparison/data/conductance_ana_shifted_'
                          + str(device_1.geometry) + '_' + str(device_2.geometry) + str(device_1.params_name) + '.npy'):
            L_vals_ana, Gd_values_ana = np.load('files/downstream_conductance/comparison/data/conductance_ana_shifted_'
                                                + str(device_1.geometry) + '_' + str(device_2.geometry)
                                                + str(device_1.params_name) + '.npy',
                                                allow_pickle=True)
        else:
            with alive_bar(2, ctrl_c=False, title=f'Computing corner transmissions') as bar:
                delta_b_1, phi_b_1 = compute_delta_b_and_phi_b(device_1, L_b, v_b, mu_b)
                tau_1 = effective_tau(tau_0, L_b, v_b, mu_b, delta_b_1, phi_b_1)
                bar()
                delta_b_2, phi_b_2 = compute_delta_b_and_phi_b(device_2, L_b, v_b, mu_b)
                tau_2 = effective_tau(tau_0, L_b, v_b, mu_b, delta_b_2, phi_b_2)
                bar()

            def calculate_Gd_ana(L):
                value = (1 - 2 * tau_1) * (1 - 2 * tau_2) \
                        + 4 * np.sqrt(tau_1 * tau_2 * (1 - tau_1) * (1 - tau_2)) * np.cos(2 * k0 * L)
                return value

            def compute_L_min_TB():
                idx_start = (np.abs(L_vals_TB - 20 * device.lB)).argmin()
                indexes = [i for i, x in enumerate(Gd_values_TB) if x == min(Gd_values_TB[idx_start:])]
                min_idx = indexes[-1]
                L_min = L_vals_TB[min_idx]
                return L_min

            def compute_L_min_ana():
                L_start = compute_L_min_TB()
                L_min = minimize(calculate_Gd_ana, L_start)
                return L_min.x[0]

            L_min_TB = compute_L_min_TB()
            L_min_ana = compute_L_min_ana()
            dL = L_min_TB - L_min_ana
            print('phi_12 = ', 2*k0*dL)
            L_vals_ana = Ls
            Gd_values_ana = [calculate_Gd_ana(L-dL) for L in L_vals_ana]
            data = [L_vals_ana, Gd_values_ana]
            np.save('files/downstream_conductance/comparison/data/conductance_ana_shifted_'
                    + str(device_1.geometry) + '_' + str(device_2.geometry)
                    + str(device_1.params_name) + '.npy', data)

    else:
        print('--------------------------------------------------------------------')
        print(f'Computing TB conductance')
        print('--------------------------------------------------------------------')
        L_vals_TB = Ls
        num_cores = multiprocessing.cpu_count()
        print(f'N_tasks = {str(len(L_vals_TB))}')
        Gd_values_TB = Parallel(n_jobs=num_cores, verbose=10)(delayed(calculate_Gd_TB)(L) for L in L_vals_TB)
        data = [L_vals_TB, Gd_values_TB]
        np.save('files/downstream_conductance/comparison/data/conductance_TB_'
                + str(device.geometry) + str(device.params_name) + '.npy', data)

        with alive_bar(2, ctrl_c=False, title=f'Computing corner transmissions') as bar:
            delta_b_1, phi_b_1 = compute_delta_b_and_phi_b(device_1, L_b, v_b, mu_b)
            tau_1 = effective_tau(tau_0, L_b, v_b, mu_b, delta_b_1, phi_b_1)
            bar()
            delta_b_2, phi_b_2 = compute_delta_b_and_phi_b(device_2, L_b, v_b, mu_b)
            tau_2 = effective_tau(tau_0, L_b, v_b, mu_b, delta_b_2, phi_b_2)
            bar()
            
        def calculate_Gd_ana(L):
            value = (1 - 2 * tau_1) * (1 - 2 * tau_2) \
                    + 4 * np.sqrt(tau_1 * tau_2 * (1 - tau_1) * (1 - tau_2)) * np.cos(2 * k0 * L)
            return value

        def compute_L_min_TB():
            idx_start = (np.abs(L_vals_TB - 20 * device.lB)).argmin()
            indexes = [i for i, x in enumerate(Gd_values_TB) if x == min(Gd_values_TB[idx_start:])]
            min_idx = indexes[-1]
            L_min = L_vals_TB[min_idx]
            return L_min

        def compute_L_min_ana():
            L_start = compute_L_min_TB()
            L_min = minimize(calculate_Gd_ana, L_start)
            return L_min.x[0]

        L_min_TB = compute_L_min_TB()
        L_min_ana = compute_L_min_ana()
        dL = L_min_TB - L_min_ana
        L_vals_ana = Ls
        Gd_values_ana = [calculate_Gd_ana(L - dL) for L in L_vals_ana]
        data = [L_vals_ana, Gd_values_ana]
        np.save('files/downstream_conductance/comparison/data/conductance_ana_shifted_'
                + str(device_1.geometry) + '_' + str(device_2.geometry)
                + str(device_1.params_name) + '.npy', data)

    # Plot
    fig, ax = plt.subplots()
    # Edit ticks of the x and y axes
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.set_xlim([0, int(np.max(Ls/device.lB))])
    ax.set_xlim([0, 30])
    ax.set_ylim([-1, 1])
    ax.grid(which='major')
    ax.set_axisbelow(True)
    
    ax.plot(L_vals_TB / device.lB, Gd_values_TB, 'k', linewidth=2, label='TB')
    ax.plot(L_vals_ana / device_1.lB, Gd_values_ana, 'r', linewidth=2, label='ana')
    ax.set(xlabel=r'$L/l_B$', ylabel=r'$G_d/G_0$')
    # plt.legend(loc='upper right')
    fig.legend(fontsize=19)
    if not fig_name:
        fig.savefig('files/downstream_conductance/comparison/plots/conductance_comparison_'
                    + str(device_1.geometry) + '_' + str(device_2.geometry) + str(device_1.params_name)
                    + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


##################################################
# Track states
##################################################
def compute_nu_crit(params, nu_min=1., nu_max=3., tol=1E-6):
    """Compute the value of nu_crit.

    :param dict params: The system's parameters.
    :param float nu_min: Lower bound, defaults to 1.
    :param float nu_max: Higher bound, defaults to 3.
    :param float tol: Precision of the returned value, defaults to 1E-6.

    :returns: The value of nu_crit.
    :rtype: float
    """
    E = 0.
    m = 1 / (2 * params["t"] * params["a"] ** 2)
    m_qh = m
    m_sc = m
    mu_qh = params["mu_qh"]
    mu_sc = params["mu_sc"]
    kF_qh = np.sqrt(2 * m * mu_qh)
    vF_qh = kF_qh / m
    V_barrier = params['Z'] * vF_qh / 2
    delta = params["delta"]

    def check_second_sol(nu_val):
        omega = 2 * mu_qh / nu_val
        k0_value = fermi_momenta(m, m, mu_qh, mu_sc, nu_val, delta, V_barrier)[0]
        k0_min = k0_value + 1e-6 * kF_qh
        k0_max = k0_min + 0.01 * kF_qh
        while np.sign(secular_equation(k0_min, E, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier)
                      * secular_equation(k0_max, E, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier)
                    ) > 0:
            k0_max += 0.005 * kF_qh
        
        sol = opt.brentq(secular_equation, k0_min, k0_max,
                         args=(E, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier))
        
        if sol > k0_value:
            return True
        else:
            return False

    if not check_second_sol(nu_max-tol):
        return np.nan
    else:
        nu_mean = (nu_min + nu_max)/2
        nu_diff = nu_max - nu_min
        while nu_diff > tol:
            check_sol = check_second_sol(nu_mean)
            if check_sol:
                nu_max = nu_mean
            else:
                nu_min = nu_mean
            nu_mean = (nu_min + nu_max)/2
            nu_diff = nu_max - nu_min
        return nu_mean


def plot_nu_crit_vs_mu_qh_delta(params, from_data=True, fig_name=False):
    """Plot nu_crit *v.s.* mu_qh/delta.

    The resulting plot is saved in the 'files/track_states/varying_mu_qh_delta' directory.

    :param dict params: The system's parameters.
    :param bool from_data: If True the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist. Defaults to True.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """
    m = 1 / (2 * params["t"] * params["a"] ** 2)
    mu_qh = params["mu_qh"]
    def compute_nu_c_val(delta):
        _params = {**params, **{'delta': delta}}
        return compute_nu_crit(_params)

    if from_data:
        if os.path.isfile('files/track_states/varying_mu_qh_delta/data/nu_crit_vs_mu_qh_delta'
                          + '_Z=' + str(params["Z"])
                          + '_m=' + str(m)
                          + '_mu_qh=' + str(mu_qh)
                          + '_mu_sc=' + str(params['mu_sc'])
                          + '.npy'):
            delta_list, nu_crit_vals = np.load('files/track_states/varying_mu_qh_delta/data/nu_crit_vs_mu_qh_delta'
                                                 + '_Z=' + str(params["Z"])
                                                 + '_m=' + str(m)
                                                 + '_mu_qh=' + str(mu_qh)
                                                 + '_mu_sc=' + str(params['mu_sc'])
                                                 + '.npy', allow_pickle=True)
        else:
            delta_list = mu_qh / np.linspace(0.1, 4000, 501)
            num_cores = multiprocessing.cpu_count()
            print('N_tasks = ', len(delta_list))
            nu_crit_vals = Parallel(n_jobs=num_cores,
                                    verbose=len(delta_list))(delayed(compute_nu_c_val)(delta_list[k]) for k in range(len(delta_list)))
            nu_crit_vals = np.asarray(nu_crit_vals)
            np.save('files/track_states/varying_mu_qh_delta/data/nu_crit_vs_mu_qh_delta'
                    + '_Z=' + str(params["Z"])
                    + '_m=' + str(m)
                    + '_mu_qh=' + str(mu_qh)
                    + '_mu_sc=' + str(params['mu_sc'])
                    + '.npy', [delta_list, nu_crit_vals])
    else:
        delta_list = mu_qh / np.linspace(0.1, 4000, 501)
        num_cores = multiprocessing.cpu_count()
        print('N_tasks = ', len(delta_list))
        nu_crit_vals = Parallel(n_jobs=num_cores,
                                verbose=len(delta_list))(delayed(compute_nu_c_val)(delta_list[k]) for k in range(len(delta_list)))
        nu_crit_vals = np.asarray(nu_crit_vals)
        np.save('files/track_states/varying_mu_qh_delta/data/nu_crit_vs_mu_qh_delta'
                + '_Z=' + str(params["Z"])
                + '_m=' + str(m)
                + '_mu_qh=' + str(mu_qh)
                + '_mu_sc=' + str(params['mu_sc'])
                + '.npy', [delta_list, nu_crit_vals])
    
    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.set_xlim([0, 4000])
    ax.set_ylim([2.61, 3])

    ax.grid(which='major')
    ax.set_axisbelow(True)
    ax.plot(mu_qh/delta_list, nu_crit_vals, 'k', linewidth=2.5)
    ax.set(xlabel=r'$\mu_{QH}/\Delta$', ylabel=r'$\nu_c$')

    if not fig_name:
        fig.savefig('files/track_states/varying_mu_qh_delta/plots/nu_crit_vs_mu_qh_delta'
                    + '_Z=' + str(params["Z"])
                    + '_m=' + str(m)
                    + '_mu_qh=' + str(mu_qh)
                    + '_mu_sc=' + str(params['mu_sc'])
                    + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)

    plt.show()


def plot_nu_crit_limit_vs_mismatch(params, from_data=True, fig_name=False):
    """Plot the asymptotic value of nu_crit *v.s.* mu_sc/mu_qh.

    The resulting plot is saved in the 'files/track_states/varying_mismatch' directory.
    
    :param dict params: The system's parameters.
    :param bool from_data: If True the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist. Defaults to True.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """
    m = 1 / (2 * params["t"] * params["a"] ** 2)
    mu_qh = params["mu_qh"]
    
    def compute_nu_c_val(mu_sc):
        _params = {**params, **{'mu_sc': mu_sc, 'delta': 1E-6*mu_qh}}
        return compute_nu_crit(_params)
    
    if from_data:
        if os.path.isfile('files/track_states/varying_mismatch/data/nu_crit_limit_vs_mismatch'
                          + '_Z=' + str(params["Z"])
                          + '_m=' + str(m)
                          + '_mu_qh=' + str(mu_qh)
                          + '.npy'):
            mu_sc_list, nu_crit_vals = np.load('files/track_states/varying_mismatch/data/nu_crit_limit_vs_mismatch'
                                                + '_Z=' + str(params["Z"])
                                                + '_m=' + str(m)
                                                + '_mu_qh=' + str(mu_qh)
                                                + '.npy', allow_pickle=True)
        else:
            mu_sc_list = mu_qh * np.arange(1, 10.+.1, .1)
            num_cores = multiprocessing.cpu_count()
            print('N_tasks = ', len(mu_sc_list))
            nu_crit_vals = Parallel(n_jobs=num_cores,
                                    verbose=len(mu_sc_list))(delayed(compute_nu_c_val)(mu_sc_list[k]) for k in range(len(mu_sc_list)))
            nu_crit_vals = np.asarray(nu_crit_vals)
            np.save('files/track_states/varying_mismatch/data/nu_crit_limit_vs_mismatch'
                    + '_Z=' + str(params["Z"])
                    + '_m=' + str(m)
                    + '_mu_qh=' + str(mu_qh)
                    + '.npy', [mu_sc_list, nu_crit_vals])
    else:
        mu_sc_list = mu_qh * np.arange(1, 10.+.1, .1)
        num_cores = multiprocessing.cpu_count()
        print('N_tasks = ', len(mu_sc_list))
        nu_crit_vals = Parallel(n_jobs=num_cores,
                                verbose=len(mu_sc_list))(delayed(compute_nu_c_val)(mu_sc_list[k]) for k in range(len(mu_sc_list)))
        nu_crit_vals = np.asarray(nu_crit_vals)
        np.save('files/track_states/varying_mismatch/data/nu_crit_limit_vs_mismatch'
                + '_Z=' + str(params["Z"])
                + '_m=' + str(m)
                + '_mu_qh=' + str(mu_qh)
                + '.npy', [mu_sc_list, nu_crit_vals])

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.set_xlim([1, 10])
    ax.set_ylim([2.61, 3])
    ax.grid(which='major')
    ax.set_axisbelow(True)
    ax.plot(mu_sc_list / mu_qh, nu_crit_vals, 'k', linewidth=2.5)
    ax.set(xlabel=r'$\mu_{SC}/\mu_{QH}$', ylabel=r'$\nu_c$')

    if not fig_name:
        fig.savefig('files/track_states/varying_mismatch/plots/nu_crit_limit_vs_mismatch'
                    + '_Z=' + str(params["Z"])
                    + '_m=' + str(m)
                    + '_mu_qh=' + str(mu_qh)
                    + '.pdf', bbox_inches='tight', transparent=True)
    else:                
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    
    plt.show()


def plot_nu_crit_limit_vs_Z(params, from_data=True, fig_name=False):
    """Plot the asymptotic value of nu_crit *v.s.* Z.

    The resulting plot is saved in the 'files/track_states/varying_Z' directory.
    
    :param dict params: The system's parameters.
    :param bool from_data: If True the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist. Defaults to True.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """
    m = 1 / (2 * params["t"] * params["a"] ** 2)
    mu_qh = params["mu_qh"]
    mu_sc = params["mu_sc"]
    
    def compute_nu_c_val(Z):
        _params = {**params, **{'delta': 1E-6*mu_qh, 'Z': Z}}
        return compute_nu_crit(_params)

    if from_data:
        if os.path.isfile('files/track_states/varying_Z/data/nu_crit_limit_vs_Z'
                          + '_m=' + str(m)
                          + '_mu_qh=' + str(mu_qh) 
                          + '_mu_sc=' + str(mu_sc)
                          + '.npy'):
            Z_list, nu_crit_vals = np.load('files/track_states/varying_Z/data/nu_crit_limit_vs_Z'
                                                + '_m=' + str(m)
                                                + '_mu_qh=' + str(mu_qh) 
                                                + '_mu_sc=' + str(mu_sc)
                                                + '.npy', allow_pickle=True)
        else:
            Z_list = np.arange(0, 0.8 + .005, .005)
            num_cores = multiprocessing.cpu_count()
            print('N_tasks = ', len(Z_list))
            nu_crit_vals = Parallel(n_jobs=num_cores,
                                    verbose=len(Z_list))(delayed(compute_nu_c_val)(Z_list[k]) for k in range(len(Z_list)))
            nu_crit_vals = np.asarray(nu_crit_vals)
            np.save('files/track_states/varying_Z/data/nu_crit_limit_vs_Z'
                    + '_m=' + str(m)
                    + '_mu_qh=' + str(mu_qh) 
                    + '_mu_sc=' + str(mu_sc)
                    + '.npy', [Z_list, nu_crit_vals])
    else:
        Z_list = np.arange(0, 0.8 + .005, .005)
        num_cores = multiprocessing.cpu_count()
        print('N_tasks = ', len(Z_list))
        nu_crit_vals = Parallel(n_jobs=num_cores,
                                verbose=len(Z_list))(delayed(compute_nu_c_val)(Z_list[k]) for k in range(len(Z_list)))
        nu_crit_vals = np.asarray(nu_crit_vals)
        np.save('files/track_states/varying_Z/data/nu_crit_limit_vs_Z'
                + '_m=' + str(m)
                + '_mu_qh=' + str(mu_qh) 
                + '_mu_sc=' + str(mu_sc)
                + '.npy', [Z_list, nu_crit_vals])

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(0.25))

    ax.set_xlim([0, 0.75])
    ax.set_ylim([2.61, 3])
    ax.grid(which='major')
    ax.set_axisbelow(True)
    ax.plot(Z_list, nu_crit_vals, 'k', linewidth=2.5)
    ax.set(xlabel=r'$Z$', ylabel=r'$\nu_c$')

    if not fig_name:
        fig.savefig('files/track_states/varying_Z/plots/nu_crit_limit_vs_Z'
                    + '_m=' + str(m)
                    + '_mu_qh=' + str(mu_qh) 
                    + '_mu_sc=' + str(mu_sc)
                    + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
        
    plt.show()


##################################################
# Finite-temperature
##################################################
def compute_momentum_difference(params, E):
    """Compute the momentum difference dk.

    :param dict params: The system's parameters.
    :param float E: The energy value.

    :returns: The value of dk.
    :rtype: float
    """
    m = 1 / (2 * params['t'] * params['a'] ** 2)
    nu = params['nu']
    mu_qh = params['mu_qh']
    mu_sc = params['mu_sc']
    kF_qh = np.sqrt(2 * m * params["mu_qh"])
    vF_qh = kF_qh / m
    V_barrier = params['Z'] * vF_qh / 2
    delta = params["delta"]
    omega = 2 * mu_qh / nu

    def sec_eq(k):
        return secular_equation(k, E, m, m, mu_qh, mu_sc, omega, delta, V_barrier)
    
    k0 = fermi_momenta(m, m, mu_qh, mu_sc, nu, delta, V_barrier)[0]
    
    if E >= 0.:
        kp_min = k0 - 0.01*kF_qh
        kp_max = k0 + 0.1*kF_qh
        while np.sign(sec_eq(kp_min) * sec_eq(kp_max)) > 0:
            kp_max += 0.01*kF_qh
        kp_value = opt.brentq(sec_eq, kp_min, kp_max)

        km_max = -k0 + 0.01*kF_qh
        km_min = -k0 - 0.1*kF_qh
        while np.sign(sec_eq(km_min) * sec_eq(km_max)) > 0:
            km_max += 0.01*kF_qh
        km_value = opt.brentq(sec_eq, km_min, km_max)
    
    else:
        kp_min = k0 - 0.01*kF_qh
        kp_max = k0 + 0.1*kF_qh
        while np.sign(sec_eq(kp_min) * sec_eq(kp_max)) > 0:
            kp_min -= 0.01*kF_qh
        kp_value = opt.brentq(sec_eq, kp_min, kp_max)

        km_max = -k0 + 0.01*kF_qh
        km_min = -k0 - 0.1*kF_qh
        while np.sign(sec_eq(km_min) * sec_eq(km_max)) > 0:
            km_min -= 0.01*kF_qh
        km_value = opt.brentq(sec_eq, km_min, km_max)

    delta_k = kp_value - km_value
    
    return delta_k

    
def plot_momentum_difference_vs_energy(params, fig_name=False):
    """Plot momentum difference *v.s.* energy.

    The resulting plot is saved in the 'files/finite_temperature/momentum_difference/varying_energy' directory.

    :param dict params: The system's parameters.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """
    m = 1 / (2 * params['t'] * params['a'] ** 2)
    nu = params['nu']
    mu_qh = params['mu_qh']
    mu_sc = params['mu_sc']
    kF_qh = np.sqrt(2 * m * params["mu_qh"])
    vF_qh = kF_qh / m
    V_barrier = params['Z'] * vF_qh / 2
    delta = params["delta"]
    
    energies = np.linspace(-0.99*delta, 0.99*delta, 101)
    k0 = fermi_momenta(m, m, mu_qh, mu_sc, nu, delta, V_barrier)[0]
    delta_k_0 = 2*k0
    
    delta_k_vals = []
    
    with alive_bar(len(energies), ctrl_c=False, title=f'Computing delta_k values') as bar:
        for E in energies:
            delta_k_vals.append(compute_momentum_difference(params, E))
            bar()


    delta_k_vals = np.asarray(delta_k_vals)
    deviation_vals = np.abs(delta_k_vals - delta_k_0)/delta_k_0

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlim([-1, 1])
    ax.grid(which='major')
    ax.set_axisbelow(True)
    ax.plot(energies/delta, deviation_vals, 'k')
    ax.set(xlabel=r'$E/\Delta$', ylabel=r'$|\delta k - 2k_0|/2k_0$')
    if not fig_name:
        fig.savefig('files/finite_temperature/momentum_difference/varying_energy/delta_k_vs_E'
                    + '_m=' + str(m)
                    + '_mu_qh=' + str(mu_qh)
                    + '_mu_sc=' + str(mu_sc)
                    + '_nu=' + str(nu)
                    + '_delta=' + str(delta)
                    + '_Z=' + str(params["Z"])
                    + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


def plot_tau_vs_energy(device, fig_name=False, from_data=True, tau_label=None):
    """Plot the conversion probabilty tau vs energy.

    The data and the plot are saved in the directory
    'files/finite_temperature/scattering_probabilities/transmissions/'.

    :param object device: The QH-SC junction.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    :param bool from_data: If True the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist. Defaults to True.
    :param int tau_label: None, 1 or 2. Defaults to None.
    """

    params = device.params
    energies = params["delta"] * np.arange(-1., 1.+.05, .05)

    def get_conversions():
        tau_vals = []
        with alive_bar(len(energies), title='Computing conversions') as bar:
            for energy in energies:
                tau_val = np.abs(compute_corner_transmissions(device, energy=energy)[1])**2
                tau_vals.append(tau_val)
                bar()
                
        # def compute_transmissions(energy):
        #     return compute_corner_transmissions(device, energy)   
        # tau_vals = Parallel(n_jobs=8, verbose=10)(delayed(compute_transmissions)(energy) for energy in energies)

        np.save('files/finite_temperature/scattering_probabilities/transmissions/data/'
                'tau_vs_E_'
                + str(device.geometry) + str(device.params_name)
                + '_L=' + str(int(round(device.dimensions["L_interface"])))
                + '.npy', tau_vals)

        return tau_vals

    if from_data:
        if os.path.isfile('files/finite_temperature/scattering_probabilities/transmissions/data/'
                          'tau_vs_E_'
                          + str(device.geometry) + str(device.params_name)
                          + '_L=' + str(int(round(device.dimensions["L_interface"])))
                          + '.npy'):
            tau_vals = np.load('files/finite_temperature/scattering_probabilities/transmissions/data/'
                                'tau_vs_E_'
                                + str(device.geometry) + str(device.params_name)
                                + '_L=' + str(int(round(device.dimensions["L_interface"])))
                                + '.npy', allow_pickle=True)
        else:
            tau_vals = get_conversions()

    else:
        tau_vals = get_conversions()

    fig, ax = plt.subplots()
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.grid()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 0.8])
    ax.set_axisbelow(True)
    plt.plot(energies/params["delta"], tau_vals, 'k')
    
    if tau_label==1: 
        ax.set(xlabel=r'$E/\Delta$', ylabel=r'$\tau_1$')
    elif tau_label==2: 
        ax.set(xlabel=r'$E/\Delta$', ylabel=r'$\tau_2$')
    else:
        ax.set(xlabel=r'$E/\Delta$', ylabel=r'$\tau$')
            
    if not fig_name:
        fig.savefig('files/finite_temperature/scattering_probabilities/transmissions/plots/'
                    'tau_vs_E_'
                    + str(device.geometry) + str(device.params_name)
                    + '_L=' + str(int(round(device.dimensions["L_interface"])))
                    + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    
    plt.show()


def plot_finite_T_conductance_TB_vs_L_various_temps(device, kTs, Ls, from_data=True, fig_name=False):
    """Plot the finite-temperature downstream conductance versus the energy for various temperatures. 
    
    Here we use a full tight-binding calculation and we compare with the zero-temperature result.

    The data and the plot are saved in the directory
    'files/finite_temperature/downstream_conductance/varying_L/tight_binding'.

    :param object device: The QH-SC junction.
    :param list kTs: The values of kB*T.
    :param list Ls: The values of the interface's length.
    :param bool from_data: If True the spectrum is plotted using the existing data.
                          If False the data are computed even if they exist.
    :param str fig_name: The name of the plot used for the manuscript, optional.
    """

    params = device.params
    finite_T_conductances = []
    for kT in kTs:
        beta = 1/kT
        # minus the derivative of Fermi-Dirac
        def dF(energy, Ef):
            return 1 / 4. * beta / (np.cosh(beta * (energy - Ef) / 2.)) ** 2
            
        def compute_trans(L, energy):
            try:
                device.dimensions = {**device.dimensions, **{"L_interface": L}}
                sys = device.make_system(onsite, hopping).finalized()
                smatrix = kwant.smatrix(sys, energy, params=params)
                Pe = smatrix.transmission((1, 0), (0, 0))
                Ph = smatrix.transmission((1, 1), (0, 0))
                value = Pe - Ph
            except ValueError:
                # print('The dimensions and geometry do not allow to build the system. '
                #     'The downstream conductance is thus taken as Gd = np.nan.')
                value = np.nan
            return value
        
        def get_zero_T_conductances(L, Ef):
            zero_energies = [Ef + i / (15 * beta) for i in range(-300, 301)]
            
            def compute_trans_for_one_L(energy):
                return compute_trans(L, energy)
            
            num_cores = multiprocessing.cpu_count()
            print(f'N_tasks = {str(len(zero_energies))}')
            zero_conductances = Parallel(n_jobs=6, verbose=10)(delayed(compute_trans_for_one_L)(energy)
                                                                    for energy in zero_energies)
                
            np.save('files/finite_temperature/downstream_conductance/varying_energy/tight_binding/data/'
                    'zero_T_conductance_TB_vs_E_'
                    + str(device.geometry) + str(device.params_name) + '_delta_kT=' + str(params["delta"]/kT)
                    + '_L=' + str(int(np.round(L)))
                    + '.npy',
                    [zero_energies, zero_conductances])

            return zero_energies, zero_conductances

        def get_zero_and_finite_T_conductances(L):
            Ef=0. # Ef=0. is the energy at which I have a resonance
            if from_data:
                if os.path.isfile('files/finite_temperature/downstream_conductance/varying_energy/tight_binding/data/'
                                'zero_T_conductance_TB_vs_E_'
                                + str(device.geometry) + str(device.params_name) + '_delta_kT=' + str(params["delta"]/kT)
                                + '_L=' + str(int(np.round(L)))
                                + '.npy'):
                    zero_T_energies, zero_T_conductances = np.load('files/finite_temperature/downstream_conductance/'
                                                                'varying_energy/tight_binding/data/'
                                                                'zero_T_conductance_TB_vs_E_'
                                                                + str(device.geometry) + str(device.params_name)
                                                                + '_delta_kT=' + str(params["delta"]/kT)
                                                                + '_L=' + str(int(np.round(L)))
                                                                + '.npy', allow_pickle=True)
                else:
                    zero_T_energies, zero_T_conductances = get_zero_T_conductances(L, Ef)
            else:
                zero_T_energies, zero_T_conductances = get_zero_T_conductances(L, Ef)

            couple_T_E = list(zip(zero_T_conductances, zero_T_energies))  # here, I have a list of (Pe-Ph, E)

            Tr = [T * dF(energy, Ef=0.) for T, energy in couple_T_E]
            g_finite = np.trapz(Tr, zero_T_energies, dx=1. / (15 * beta))  # this is the conductance at one energy Ef.
            
            idx_zero_T_E0 = np.where(np.asarray(zero_T_energies)==0.0)[0][0]
            g_zero = zero_T_conductances[idx_zero_T_E0]
            return g_zero, g_finite
        
        if from_data:
            if os.path.isfile('files/finite_temperature/downstream_conductance/varying_L/tight_binding/data/conductance_TB_vs_L_'
                            + str(device.geometry) + str(device.params_name)
                            + '_delta_kT=' + str(params["delta"]/kT)
                            + '.npy'):
                zero_temp_conductances, finite_temp_conductances = np.load('files/finite_temperature/downstream_conductance/varying_L/tight_binding/data/conductance_TB_vs_L_'
                                                                        + str(device.geometry) + str(device.params_name)
                                                                        + '_delta_kT=' + str(params["delta"]/kT)
                                                                        + '.npy', allow_pickle=True)
            else:
                conductances = []
                i=1
                for L in Ls:
                    print('--------------------------------------------------------------------')
                    print(f'Computing conductances ({str(i)}/{len(Ls)})')
                    print('--------------------------------------------------------------------')
                    conductances.append(get_zero_and_finite_T_conductances(L))  
                    i+=1    
                conductances_to_lists = list(map(list, zip(*conductances)))
                zero_temp_conductances = conductances_to_lists[0]
                finite_temp_conductances = conductances_to_lists[1]
                np.save('files/finite_temperature/downstream_conductance/varying_L/tight_binding/data/conductance_TB_vs_L_'
                        + str(device.geometry) + str(device.params_name)
                        + '_delta_kT=' + str(params["delta"]/kT)
                        + '.npy', [zero_temp_conductances, finite_temp_conductances])
        else:
            conductances = []
            j=1
            for L in Ls:
                print('--------------------------------------------------------------------')
                print(f'Computing conductances ({str(j)}/{len(Ls)})')
                print('--------------------------------------------------------------------')
                conductances.append(get_zero_and_finite_T_conductances(L))  
                j+=1          
            conductances_to_lists = list(map(list, zip(*conductances)))
            zero_temp_conductances = conductances_to_lists[0]
            finite_temp_conductances = conductances_to_lists[1]
            np.save('files/finite_temperature/downstream_conductance/varying_L/tight_binding/data/conductance_TB_vs_L_'
                    + str(device.geometry) + str(device.params_name)
                    + '_delta_kT=' + str(params["delta"]/kT)
                    + '.npy', [zero_temp_conductances, finite_temp_conductances])

        finite_T_conductances.append(finite_temp_conductances)
        
    fig, ax = plt.subplots()
    # Edit ticks of the x and y axes
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlim([0, int(np.max(Ls/device.lB))])
    ax.set_ylim([-1, 1])
    ax.grid(which='major')
    ax.set_axisbelow(True)    
    ax.set(xlabel=r'$L/l_B$', ylabel=r'$G_d/G_0$')
    
    ax.plot(Ls/device.lB, zero_temp_conductances, 'o', markersize=4, label=r"$k_B T = 0$")
    mean_0 = (np.max(zero_temp_conductances[len(zero_temp_conductances)//2:]) 
              +np.min(zero_temp_conductances[len(zero_temp_conductances)//2:]))/2
    print(f'<G_d/G_0> = {str(mean_0)} at kT = 0')
    
    for i in range(len(kTs)):
        mean = (np.max(finite_T_conductances[i][len(finite_T_conductances[i])//2:])
                +np.min(finite_T_conductances[i][len(finite_T_conductances[i])//2:]))/2
        print(f'<G_d/G_0> = {str(mean)} at kT = Delta/{str(int(params["delta"]/kTs[i]))}')
        ax.plot(Ls/device.lB, finite_T_conductances[i],
                label=r"$k_B T = \Delta/$" + str(int(params["delta"]/kTs[i])), linewidth=2)
    
    fig.legend(fontsize=19)
    if not fig_name:
        fig.savefig('files/finite_temperature/downstream_conductance/varying_L/tight_binding/plots/conductance_TB_vs_L_various_temps_'
                + str(device.geometry) + str(device.params_name)
                + '.pdf', bbox_inches='tight', transparent=True)
    else:
        fig.savefig('./figures/' + str(fig_name) + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()
