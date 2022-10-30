"""This script generates the figures as used in the manuscript.

To re-compute the data use the option from_data=False in the plot functions.
Due to the option fig_name the plots are saved in the ‘figures‘ directory.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from modules import system, functions, utils
import numpy as np

plt.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 21
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2

params = dict(a=1, t=1, mu_qh=0.05, mu_sc=0.05, delta=0.005, nu=2., Z=0.)


"""Fig. 2
#####################################################
# FIG. 2
#####################################################
_params = {**params, **{'mu_qh': 1, 'mu_sc': 1, 'delta': .1, 'nu': 2.4}}
utils.plot_spectrum_micro(params=_params, fig_name='fig2', show_k0=True, qp_labels=True)
"""

"""Fig. 3
#####################################################
# FIG. 3
#####################################################
nus = [1.2, 1.6, 2., 2.4, 2.8]
Zs = np.linspace(0, 20, 201)
utils.plot_k0_vs_Z_various_fillings(nus, Zs, params, fig_name='fig3')
"""

"""Fig. 4
#####################################################
# FIG. 4
#####################################################
nus = [1.2, 1.6, 2., 2.4, 2.8]
Zs = np.linspace(0, 20, 201)
utils.plot_fh_p_vs_Z_various_fillings(nus, Zs, params, fig_name='fig4')
"""

"""Fig. 5
#####################################################
# FIG. 5
#####################################################
nus = [1.2, 1.6, 2., 2.4, 2.8]
thetas = [-45, -30, 0, 30, 45, 60, 90, 120, 135]  # multiples of pi/3 and pi/4
device = system.DeviceSingleCorner(theta_qh=90, theta_sc=None, params=params)
utils.plot_tau_vs_theta_sc_various_fillings(nus, thetas, device, fig_name='fig5a', show_only_commensurate=True)
device = system.DeviceSingleCorner(theta_qh=None, theta_sc=90, params=params)
utils.plot_tau_vs_theta_qh_various_fillings(nus, thetas, device, fig_name='fig5b', show_only_commensurate=True)
"""

"""Fig. 6
#####################################################
# FIG. 6
#####################################################
plt.rcParams['font.size'] = 30.75
device = system.DeviceSingleCorner(theta_qh=90, theta_sc=45, params=params)
utils.plot_density(device=device, fig_name='fig6')
"""

"""Fig. 7
#####################################################
# FIG. 7
#####################################################
nus = [1.2, 1.6, 2., 2.4, 2.8]
thetas = [-45, -30, 0, 30, 45, 60, 90, 120, 135]  # multiples of pi/3 and pi/4
_params = {**params, **{'mu_qh': params['mu_qh']/2, 'delta': params['delta']/2, 'Z': 0.7}}
device = system.DeviceSingleCorner(theta_qh=None, theta_sc=90, params=_params)
utils.plot_tau_vs_theta_qh_various_fillings(nus, thetas, device, fig_name='fig7', show_only_commensurate=True)
"""

"""Fig. 8
#####################################################
# FIG. 8
#####################################################
_params = {**params, **{'mu_qh': params['mu_qh']/2, 'delta': params['delta']/2, 'nu': 2.8, 'Z': 0.7}}
m = 1 / (2 * _params['t'] * _params['a'] ** 2)
m_qh = m
m_sc = m
kF_qh = np.sqrt(2 * m_qh * _params['mu_qh'])
kF_sc = np.sqrt(2 * m_sc * _params['mu_sc'])
vF_qh = kF_qh / m_qh
vF_sc = kF_sc / m_sc
V_barrier = _params['Z'] * vF_qh / 2
xi = vF_sc/_params['delta']
v_CAES = functions.velocity(m_qh, m_sc, _params['mu_qh'], _params['mu_sc'],
                            _params['nu'], _params['delta'], V_barrier)

L_b = xi/10
v_b = v_CAES
mu_b = _params['mu_sc']
theta_1, theta_2 = [0, 90]
# with the above values we find: 
# # delta_b_1, phi_b_1 = = [0.25240637, 3.38199987] & delta_b_2, phi_b_2 = [0.00763934, 3.00231003]
# phi_12 = 2k0 * dL = 0.7246063050029221

# for theta_qh in [theta_1, theta_2]:
#     device = system.DeviceSingleCorner(theta_qh=theta_qh, theta_sc=90, params=_params)
#     utils.compute_delta_b_and_phi_b(device, L_b, v_b, mu_b)

device_1 = system.DeviceSingleCorner(theta_qh=theta_1, theta_sc=90, params=_params)
device_2 = system.DeviceSingleCorner(theta_qh=theta_2, theta_sc=90, params=_params)
device =  system.Device(theta_1=theta_1, theta_2=theta_2, params=_params)
Ls = device.lB * np.arange(0, 50+.2, .2)
utils.plot_downstream_conductance_comparison(Ls, device, device_1, device_2, L_b, v_b, mu_b, fig_name='fig8')
"""

"""Fig. 9
#####################################################
# FIG. 9
#####################################################
_params = {**params, **{'nu': 2.8, 'delta': params['mu_qh']/20}}
utils.plot_spectrum_micro(params=_params, fig_name='fig9')
"""

"""Fig. 10
#####################################################
# FIG. 10
#####################################################
plt.rcParams['font.size'] = 47.7
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.linewidth'] = 2.5
utils.plot_nu_crit_vs_mu_qh_delta(params, fig_name='fig10a')
utils.plot_nu_crit_limit_vs_mismatch(params, fig_name='fig10b')
utils.plot_nu_crit_limit_vs_Z(params, fig_name='fig10c')
"""

"""Fig. 11
#####################################################
# FIG. 11
#####################################################
_params = {**params, **{'mu_sc': 2*params['mu_qh'], 'nu': 2.8, 'Z': 0.7}}
plt.rcParams['font.size'] = 31.5
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.linewidth'] = 2

utils.plot_spectrum_micro(params=_params, fig_name='fig11a')
utils.plot_momentum_difference_vs_energy(_params, fig_name='fig11b')

device = system.DeviceSingleCorner(theta_qh=0, theta_sc=90, params=_params)
device.dimensions = {**device.dimensions, **{"L_interface": 20*device.lB}}
utils.plot_tau_vs_energy(device=device, fig_name='fig11c', tau_label=1)

device = system.DeviceSingleCorner(theta_qh=90, theta_sc=90, params=_params)
device.dimensions = {**device.dimensions, **{"L_interface": 20*device.lB}}
utils.plot_tau_vs_energy(device=device, fig_name='fig11d', tau_label=2)
"""

"""Fig. 12
#####################################################
# FIG. 12
#####################################################
_params = {**params, **{'mu_sc': 2*params['mu_qh'], 'nu': 2.8, 'Z': 0.7}}
device = system.Device(theta_1=0, theta_2=90, params=_params)
Ls = device.lB * np.arange(0, 30+.6, .6)
kTs = [_params["delta"]/10, _params["delta"]/2]
utils.plot_finite_T_conductance_TB_vs_L_various_temps(device, kTs, Ls, fig_name='fig12')
"""
