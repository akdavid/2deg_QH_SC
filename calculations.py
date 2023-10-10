"""Calculations.

This script contains different sections, each of which 
generates the data and the plot of a calculation. Comment the quotation
marks by using hashtags to run the calculations.

Contents:

* Visualization

  - Kwant system
  - Density u^2 - v^2

* Energy Spectrum

  - Tight-binding spectrum
  - Microscopic spectrum
  - Spectrum comparison
  
* Momentum at the Fermi level k0

  - k0 *v.s.* nu 
  - k0 *v.s.* Z at various fillings

* Andreev Transmission and Hole Probability

  - tau *v.s.* theta_qh at various fillings
  - tau *v.s.* theta_sc at various fillings
  - tau *v.s.* mu_qh/Delta at various fillings
  - fh_p *v.s.* Z at various fillings

* Downstream conductance

  - Conductance comparison

* Track states

  - nu_crit *v.s.* mu_qh/delta
  - Asymptotic nu_crit *v.s.* mu_sc/mu_qh
  - Asymptotic nu_crit *v.s.* Z

* Finite-temperature

  - Momentum difference *v.s.* energy
  - tau *v.s.* energy
  - Finite-temperature tight-binding conductance *v.s.* L
  - Finite-temperature tight-binding conductance *v.s.* L at various temperatures
  

The resulting data and plots are saved in the ‘files‘ directory.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from modules import system, functions, utils
import numpy as np
import kwant
from kwant.physics import dispersion
import kwantspectrum as ks

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# Default parameters used in the manuscript for an ideal interface.
params = dict(a=1, t=1, mu_qh=0.05, mu_sc=0.05, delta=0.005, nu=2., Z=0.)

# To reduce the calculations' time you can use 
# the following parameters.
# Here we choose higher values for the chemical potential 
# and the superconducting gap in order to reduce 
# the superconducting coherence length and so 
# to reduce the system's dimensions.
params_small = dict(a=1, t=1, mu_qh=0.2, mu_sc=0.2, delta=0.05, nu=2., Z=0.)


##########################################################################################
##########################################################################################
#                                 VISUALIZATION
##########################################################################################
##########################################################################################
"""Kwant system (single-corner)
for theta_qh in [-45, 30, 60]:
  for theta_sc in [90]:
    device_single_corner = system.DeviceSingleCorner(theta_qh=theta_qh, theta_sc=theta_sc, params=params_small)
    utils.plot_device(device=device_single_corner)
"""

"""Kwant system (two-corner)
for theta_1 in [0, 90]:
  for theta_2 in[0, 90, 135]:
    device = system.Device(theta_1=theta_1, theta_2=theta_2, params=params_small)
    utils.plot_device(device=device)
"""


"""Probability density |u|^2 - |v|^2 (single-corner)
for theta_qh in [-45, 0, 45]:
  for theta_sc in [45, 90]:
    device_single_corner = system.DeviceSingleCorner(theta_qh=theta_qh, theta_sc=theta_sc, params=params_small, small=True)
    utils.plot_density(device=device_single_corner, energy=0.)
"""

"""Probability density |u|^2 - |v|^2 (two-corner)
for theta_1 in [0, 90]:
  for theta_2 in[0, 90, 135]:
    device = system.Device(theta_1=theta_1, theta_2=theta_2, params=params_small)
    utils.plot_density(device=device, energy=0.)
"""


##########################################################################################
##########################################################################################
#                                 ENERGY SPECTRUM
##########################################################################################
##########################################################################################
"""Plot tight-binding spectrum
params_1 = params
params_2 = {**params, **{'nu': 2.8}}
params_3 = {**params, **{'mu_sc': 2*params['mu_qh'], 'Z': 0.7}, **{'nu': 2.8}}

for _params in [params_1, params_2, params_3]:
  device = system.DeviceInfinite(params=_params)
  utils.plot_spectrum_TB(device=device, from_data=True)
"""

"""Plot microscopic spectrum
params_1 = params
params_2 = {**params, **{'nu': 2.8}}
params_3 = {**params, **{'mu_sc': 2*params['mu_qh'], 'Z': 0.7}, **{'nu': 2.8}}

for _params in [params_1, params_2, params_3]:
  utils.plot_spectrum_micro(params=_params, from_data=True)
"""

"""Plot spectrum comparison
device = system.DeviceInfinite(params=params)
utils.plot_spectrum_comparison(device=device, params=params, from_data=True)
"""


##########################################################################################
##########################################################################################
#                                 MOMENTUM k0
##########################################################################################
##########################################################################################
"""Plot k0 vs nu
nus = np.linspace(1, 3, 201)
utils.plot_k0_vs_nu(nus, params)
"""

"""Plot k0 vs Z at various fillings
nus = [1.2, 1.6, 2., 2.4, 2.8]
Zs = np.linspace(0, 20, 101)
utils.plot_k0_vs_Z_various_fillings(nus, Zs, params)
"""


##########################################################################################
##########################################################################################
#                   ANDREEV TRANSMISSION AND HOLE CONTENT
##########################################################################################
##########################################################################################
"""Plot tau vs theta_qh at various fillings
nus = [1.2, 1.6, 2., 2.4, 2.8]
# choose angles such that tan(theta) is a rational number
thetas = [-45, 0, 45, 75.96375653207353, 90, 135, 153.43494882292202, 165.96375653207352]
device = system.DeviceSingleCorner(theta_qh=None, theta_sc=90, params=params_small, small=True)
utils.plot_tau_vs_theta_qh_various_fillings(nus, thetas, device, from_data=True)
"""

"""Plot tau vs theta_sc at various fillings
nus = [1.2, 1.6, 2., 2.4, 2.8]
# choose angles such that tan(theta) is a rational number
thetas = [-45, 0, 45, 75.96375653207353, 90, 135, 153.43494882292202, 165.96375653207352]
device = system.DeviceSingleCorner(theta_qh=90, theta_sc=None, params=params_small, small=True)
utils.plot_tau_vs_theta_sc_various_fillings(nus, thetas, device, from_data=True)
"""

"""Plot tau vs mu_qh/delta at various fillings
plt.rcParams['font.size'] = 18
nus = [1.2, 1.6, 2., 2.4, 2.8]
deltas = params['mu_qh'] * np.linspace(1/20, 1, 21)
utils.plot_tau_vs_mu_qh_delta_various_fillings(nus, deltas, theta_qh=0, theta_sc=90, params=params, from_data=True)    
"""

"""Plot fh_p vs Z at various fillings
nus = [1.2, 1.6, 2., 2.4, 2.8]
Zs = np.linspace(0, 20, 101)
utils.plot_fh_p_vs_Z_various_fillings(nus, Zs, params)
"""


##########################################################################################
##########################################################################################
#                           DOWNSTREAM CONDUCTANCE
##########################################################################################
##########################################################################################
"""Plot conductance comparison
_params = params_small
m = 1 / (2 * _params['t'] * _params['a'] ** 2)
m_qh = m
m_sc = m
kF_sc = np.sqrt(2 * m_sc * _params['mu_sc'])
vF_sc = kF_sc / m_sc
kF_qh = np.sqrt(2 * m * _params["mu_qh"])
vF_qh = kF_qh / m
V_barrier = _params['Z'] * vF_qh / 2
xi = vF_sc/_params['delta']
v_CAES = functions.velocity(m_qh, m_sc, _params['mu_qh'], _params['mu_sc'], _params['nu'], _params['delta'], V_barrier)

L_b = xi/10
v_b = v_CAES
mu_b = _params['mu_sc']

for theta_1 in [0]:
    device_1 = system.DeviceSingleCorner(theta_qh=theta_1, theta_sc=90, params=_params)
    for theta_2 in [90]:
        device_2 = system.DeviceSingleCorner(theta_qh=theta_2, theta_sc=90, params=_params)
        device =  system.Device(theta_1=theta_1, theta_2=theta_2, params=_params)
        Ls = device.lB * np.arange(0, 30+.2, .2)
        utils.plot_downstream_conductance_comparison(Ls, device, device_1, device_2, L_b, v_b, mu_b, from_data=True)
"""


##########################################################################################
##########################################################################################
#                           TRACK STATES
##########################################################################################
##########################################################################################
"""Plot the value of nu_crit vs mu_qh/delta
utils.plot_nu_crit_vs_mu_qh_delta(params)
"""

"""Plot the asymptotic value of nu_crit vs mismatch
utils.plot_nu_crit_limit_vs_mismatch(params)
"""

"""Plot the asymptotic value of nu_crit vs Z
utils.plot_nu_crit_limit_vs_Z(params)
"""


##########################################################################################
##########################################################################################
#                           FINITE-TEMPERATURE
##########################################################################################
##########################################################################################
"""Plot momentum difference vs energy
for nu in [1.2, 1.6, 2., 2.4, 2.8]:
  _params = {**params, **{'mu_sc': 2*params['mu_qh'], 'nu': nu, 'Z': 0.7}}
  utils.plot_momentum_difference_vs_energy(_params)
"""

"""Plot tau vs energy
_params = {**params, **{'mu_sc': 2*params['mu_qh'], 'nu': 2.8, 'Z': 0.7}}
device = system.DeviceSingleCorner(theta_qh=0, theta_sc=90, params=_params)
device.dimensions = {**device.dimensions, **{"L_interface": 20*device.lB}}
utils.plot_tau_vs_energy(device=device, from_data=True)

device = system.DeviceSingleCorner(theta_qh=90, theta_sc=90, params=_params)
device.dimensions = {**device.dimensions, **{"L_interface": 20*device.lB}}
utils.plot_tau_vs_energy(device=device, from_data=True)
"""

"""Plot finite-temperature TB conductance vs L at various temperatures
_params = {**params, **{'mu_sc': 2*params['mu_qh'], 'nu': 2.8, 'Z': 0.7}}
device = system.Device(theta_1=0, theta_2=90, params=_params)
Ls = device.lB * np.arange(0, 30+.6, .6)
kTs = [_params["delta"]/10, _params["delta"]/2]
utils.plot_finite_T_conductance_TB_vs_L_various_temps(device, kTs, Ls, fig_name='fig12')
"""
