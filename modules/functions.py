"""Definition of the functions required for the calculations."""
"""
import numpy as np
from .pauli import sigma_0, sigma_1, sigma_3
import scipy.special as special
from scipy.integrate import quad
from scipy.misc import derivative
import scipy.optimize as opt
"""

##################################################
# Microscopic model
##################################################
def U(n, z):
    """Definition of the parabolic cylinder function U(n, z) as defined in Abramowitz & Stegun book.

    :param float n: First argument of the parabolic cylinder function.
    :param float z: Second argument of the parabolic cylinder function.

    :returns: The value of the parabolic cylinder function U(n, z).
    :rtype: float
    """
    return special.pbdv(-n - 1 / 2, z)[0]


def U_p(x, E, k, m_qh, mu_qh, omega):
    """Electron-like solution in QH region.

    :param float x: x-coordinate.
    :param float E: Energy measured from the Fermi level.
    :param float k: Momentum along the QH-SC interface.
    :param float m_qh: Effective mass in the QH region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float omega: Cyclotron frequency.

    :returns: The value of the parabolic cylinder function associated to electrons.
    :rtype: float
    """
    return U(-(mu_qh + E) / omega, -np.sqrt(2 * m_qh * omega) * (x - k / (m_qh * omega)))


def U_m(x, E, k, m_qh, mu_qh, omega):
    """Hole-like solution in QH region.

    :param float x: x-coordinate.
    :param float E: Energy measured from the Fermi level.
    :param float k: Momentum along the QH-SC interface.
    :param float m_qh: Effective mass in the QH region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float omega: Cyclotron frequency.

    :returns: The value of the parabolic cylinder function associated to holes.
    :rtype: float
    """
    return U(-(mu_qh - E) / omega, -np.sqrt(2 * m_qh * omega) * (x + k / (m_qh * omega)))


def chi_p(x, E, k, m_qh, mu_qh, nu):
    """Electron-like wave function in QH region.

    :param float x: x-coordinate.
    :param float E: Energy measured from the Fermi level.
    :param float k: Momentum along the QH-SC interface.
    :param float m_qh: Effective mass in the QH region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float nu: Filling factor.

    :returns: The value of electron-like wave function in QH region.
    :rtype: float
    """
    omega = 2 * mu_qh / nu

    def abs_U_p_squared(s):
        return np.abs(U_p(s, E, k, m_qh, mu_qh, omega)) ** 2

    integral = quad(abs_U_p_squared, -np.inf, 0.)[0]
    N_p_value = 1 / np.sqrt(integral)

    # N_p_value = 1
    return N_p_value * U_p(x, E, k, m_qh, mu_qh, omega)


def chi_m(x, E, k, m_qh, mu_qh, nu):
    """Hole-like wave function in QH region.

    :param float x: x-coordinate.
    :param float E: Energy measured from the Fermi level.
    :param float k: Momentum along the QH-SC interface.
    :param float m_qh: Effective mass in the QH region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float nu: Filling factor.

    :returns: The value of hole-like wave function in QH region.
    :rtype: float
    """
    omega = 2 * mu_qh / nu

    def abs_U_m_squared(s):
        return np.abs(U_m(s, E, k, m_qh, mu_qh, omega)) ** 2

    integral = quad(abs_U_m_squared, -np.inf, 0.)[0]
    N_m_value = 1 / np.sqrt(integral)
    return N_m_value * U_m(x, E, k, m_qh, mu_qh, omega)


def phi_p(x, E, k, m_sc, mu_sc, delta):
    """Electron-like wave function in SC region.

    :param float x: x-coordinate.
    :param float E: Energy measured from the Fermi level.
    :param float k: Momentum along the QH-SC interface.
    :param float m_sc: Effective mass in the SC region.
    :param float mu_sc: Chemical potential in the SC region.
    :param float delta: Superconducting gap.

    :returns: The value of electron-like wave function in SC region.
    :rtype: float
    """
    q_p = np.sqrt(2 * m_sc * mu_sc - k ** 2 + 2 * m_sc * 1j * np.sqrt(np.abs(delta ** 2 - E ** 2)))
    C_value = np.sqrt(2 * np.imag(q_p))
    return C_value * np.exp(+1j * q_p * x)


def phi_m(x, E, k, m_sc, mu_sc, delta):
    """Hole-like wave function in SC region.

    :param float x: x-coordinate.
    :param float E: Energy measured from the Fermi level.
    :param float k: Momentum along the QH-SC interface.
    :param float m_sc: Effective mass in the SC region.
    :param float mu_sc: Chemical potential in the SC region.
    :param float delta: Superconducting gap.

    :returns: The value of hole-like wave function in SC region.
    :rtype: float
    """
    q_p = np.sqrt(2 * m_sc * mu_sc - k ** 2 + 2 * m_sc * 1j * np.sqrt(np.abs(delta ** 2 - E ** 2)))
    q_m = np.sqrt(2 * m_sc * mu_sc - k ** 2 - 2 * m_sc * 1j * np.sqrt(np.abs(delta ** 2 - E ** 2)))
    C_value = np.sqrt(2 * np.imag(q_p))
    return C_value * np.exp(-1j * q_m * x)


def kronecker_delta(x):
    """Kronecker delta function.

    :param float x: x-coordinate.

    :returns: 1. if x=0 and 0. otherwise.
    :rtype: float
    """
    if x == 0:
        return 1.
    else:
        return 0.


def secular_equation(k, E, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier):
    """Value of f(k, E) used to compute the energy spectrum of the CAES
    and the Fermi momenta by solving the secular equation f(k, E) = 0.

    :param float E: Energy measured from the Fermi level.
    :param float k: Momentum along the QH-SC interface.
    :param float m_qh: Effective mass in the QH region.
    :param float m_sc: Effective mass in the SC region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float mu_sc: Chemical potential in the SC region.
    :param float omega: Cyclotron frequency.
    :param float delta: Superconducting gap.
    :param float V_barrier: Height of the delta-potential barrier.

    :returns: The value of f(k, E).
    :rtype: float
    """
    q_calc = np.sqrt(2 * m_sc * mu_sc - k ** 2 + 2 * m_sc * 1j * np.sqrt(np.abs(delta ** 2 - E ** 2)))
    c_calc = q_calc.real * m_qh / m_sc
    d_calc = q_calc.imag * m_qh / m_sc + 2 * m_qh * V_barrier
    G_calc = U_p(0., E, k, m_qh, mu_qh, omega)
    H_calc = U_m(0., E, k, m_qh, mu_qh, omega)
    Gp_calc = derivative(U_p, 0., dx=1e-6, args=(E, k, m_qh, mu_qh, omega))
    Hp_calc = derivative(U_m, 0., dx=1e-6, args=(E, k, m_qh, mu_qh, omega))

    return G_calc * H_calc * (c_calc ** 2 + d_calc ** 2) + Gp_calc * Hp_calc \
        + c_calc * E / np.sqrt(np.abs(delta ** 2 - E ** 2)) * (Gp_calc * H_calc - G_calc * Hp_calc) \
        + d_calc * (Gp_calc * H_calc + G_calc * Hp_calc)


def fermi_momenta(m_qh, m_sc, mu_qh, mu_sc, nu, delta, V_barrier):
    """Positive momentum solutions of the secular equation f(E, k) = 0 at the Fermi level (i.e. at E = 0).

    :param float m_qh: Effective mass in the QH region.
    :param float m_sc: Effective mass in the SC region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float mu_sc: Chemical potential in the SC region.
    :param float nu: Filling factor.
    :param float delta: Superconducting gap.
    :param float V_barrier: Height of the delta-potential barrier.

    :returns: The positive solutions of the equation f(E=0, k) = 0.
    :rtype: list
    """
    E = 0.
    kF_qh = np.sqrt(2 * m_qh * mu_qh)
    omega = 2 * mu_qh / nu
    if (nu // 1) % 2 == 0:  # compute the number of crossings Nc
        Nc = nu // 1
    else:
        Nc = nu // 1 + 1
    N = int(Nc / 2)  # number of positive solutions

    values = []
    k0_value = 0.
    for i in range(N):
        k0_min = k0_value + 1e-6 * kF_qh
        k0_max = k0_min + 0.01 * kF_qh
        while np.sign(secular_equation(k0_min, E, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier)
                      * secular_equation(k0_max, E, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier)
                      ) > 0:
            k0_max += 0.005 * kF_qh
        k0_value = opt.brentq(secular_equation, k0_min, k0_max,
                              args=(E, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier))
        values.append(k0_value)
    return np.asarray(values)


def hole_probability(m_qh, m_sc, mu_qh, mu_sc, nu, delta, V_barrier):
    r"""Compute the hole content f_h^+ at the quasi-electron crossing, i.e., at k = -k0.

    :param float m_qh: Effective mass in the QH region.
    :param float m_sc: Effective mass in the SC region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float mu_sc: Chemical potential in the SC region.
    :param float nu: Filling factor.
    :param float delta: Superconducting gap.
    :param float V_barrier: Height of the delta-potential barrier.

    :returns: Hole probability at the quasi-electron crossing.
    :rtype: float
    """
    k0 = -fermi_momenta(m_qh, m_sc, mu_qh, mu_sc, nu, delta, V_barrier)[0]
    q0 = np.sqrt(2 * m_sc * mu_sc - k0 ** 2 + 2 * m_sc * 1j * delta)

    dx = 1e-6
    Gp_0 = (chi_p(0. + dx, 0., k0, m_qh, mu_qh, nu) - chi_p(0. - dx, 0., k0, m_qh, mu_qh, nu)) / (2 * dx)
    G_0 = chi_p(0., 0., k0, m_qh, mu_qh, nu)
    H_0 = chi_m(0., 0., k0, m_qh, mu_qh, nu)
    
    c0 = np.real(q0)
    q0pp = np.imag(q0)
    d0 = (m_qh/m_sc * np.imag(q0) + 2*m_qh*V_barrier)
    g0 = Gp_0 + d0*G_0
    
    B = 1 - c0**2 * H_0**2 * (1 + 1/(4*q0pp) * (G_0**2 + (g0 + q0pp*G_0)**2/(c0**2 + q0pp**2))) \
        * (g0**2 + c0**2 * H_0**2 * (1 + 1/(2*q0pp) * (G_0**2 + g0**2/c0**2)))**(-1)
    
    return B


def velocity(m_qh, m_sc, mu_qh, mu_sc, nu, delta, V_barrier):
    """Compute the velocity of the CAES.

    :param float m_qh: Effective mass in the QH region.
    :param float m_sc: Effective mass in the SC region.
    :param float mu_qh: Chemical potential in the QH region.
    :param float mu_sc: Chemical potential in the SC region.
    :param float nu: Filling factor.
    :param float delta: Superconducting gap.
    :param float V_barrier: Height of the delta-potential barrier.

    :returns: The value of the velocity.
    :rtype: float
    """
    omega = 2 * mu_qh / nu
    k0 = fermi_momenta(m_qh, m_sc, mu_qh, mu_sc, nu, delta, V_barrier)[0]

    def f(E, k):
        return secular_equation(E, k, m_qh, m_sc, mu_qh, mu_sc, omega, delta, V_barrier)

    def partial_derivative(func, var=0, point=[]):
        args = point[:]

        def wraps(x):
            args[var] = x
            return func(*args)

        return derivative(wraps, point[var], dx=1e-6)

    dE = partial_derivative(f, 0, [0., k0])
    dk = partial_derivative(f, 1, [0., k0])

    return -dk/dE


##################################################
# Tight-binding model
##################################################
def onsite(site, a, t, mu_qh, mu_sc, delta, Z):
    """Define onsite energies in QH and SC regions including a delta-potential barrier.

    :param  site: Kwant site.
    :param float a: Lattice spacing.
    :param float t: Hopping energy at zero field.
    :param float mu_qh: Chemical potential in the QH region.
    :param float mu_sc: Chemical potential in the SC region.
    :param float delta: Superconducting gap.
    :param float Z: Barrier strength.

    :returns: Onsite energy.
    :rtype: float
    """
    m = 1/(2*t*a**2)
    kF_qh = np.sqrt(2 * m * mu_qh)
    vF_qh = kF_qh / m
    V_barrier = Z * vF_qh / 2
    (x, y) = site.pos
    onsite_qh = (4 * t - mu_qh) * sigma_3 * np.heaviside(-x, 1.)
    onsite_sc = ((4 * t - mu_sc) * sigma_3 + delta * sigma_1) * np.heaviside(x, 0.)
    barrier = V_barrier * sigma_3 * kronecker_delta(x)
    return onsite_qh + onsite_sc + barrier


def hopping(site1, site2, a, t, mu_qh, nu):
    """Define hopping energies in QH and SC regions.

    :param  site1: Kwant site.
    :param  site2: Kwant site.
    :param float a: Lattice spacing.
    :param float t: Hopping energy at zero field.
    :param float mu_qh: Chemical potential in the QH region.
    :param float nu: Filling factor.

    :returns: Hopping energy.
    :rtype: float
    """
    m = 1/(2*t*a**2)
    omega = 2 * mu_qh / nu
    B = m * omega
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    xmed = (x1 + x2) / 2
    gauge = B * xmed
    phase = gauge * (y2 - y1)
    hopping_qh = - t * (np.cos(phase) * sigma_3 + 1j * np.sin(phase) * sigma_0) * np.heaviside(-xmed, 1.)
    hopping_sc = - t * sigma_3 * np.heaviside(xmed, .0)
    return hopping_qh + hopping_sc


def onsite_qh(site, t, mu_qh):
    """Define onsite energy in QH region.

    :param  site: Kwant site.
    :param float t: Hopping energy at zero field.
    :param float mu_qh: Chemical potential in the QH region.

    :returns: Onsite energy.
    :rtype: float
    """
    onsite_qh = (4 * t - mu_qh) * sigma_3
    return onsite_qh


def onsite_sc(site, t, mu_sc, delta):
    """Define onsite energy in SC region.

    :param  site: Kwant site.
    :param float t: Hopping energy at zero field.
    :param float mu_sc: Chemical potential in the SC region.
    :param float delta: Superconducting gap.

    :returns: Onsite energy.
    :rtype: float
    """
    onsite_sc = (4 * t - mu_sc) * sigma_3 + delta * sigma_1
    return onsite_sc


def hopping_qh(site1, site2, a, t, mu_qh, nu):
    """Define hopping energy in QH region.

    :param  site1: Kwant site.
    :param  site2: Kwant site.
    :param float a: Lattice spacing.
    :param float t: Hopping energy at zero field.
    :param float mu_qh: Chemical potential in the QH region.
    :param float nu: Filling factor.

    :returns: Hopping energy.
    :rtype: float
    """
    m = 1/(2*t*a**2)
    omega = 2 * mu_qh / nu
    B = m * omega
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    xmed = (x1 + x2) / 2
    gauge = B * xmed
    phase = gauge * (y2 - y1)
    hopping_qh = - t * (np.cos(phase) * sigma_3 + 1j * np.sin(phase) * sigma_0) 
    return hopping_qh


def hopping_sc(site1, site2, t):
    """Define hopping energy in SC region.

    :param  site1: Kwant site.
    :param  site2: Kwant site.
    :param float t: Hopping energy at zero field.

    :returns: Hopping energy.
    :rtype: float
    """
    hopping_sc = - t * sigma_3 
    return hopping_sc


##################################################
# Effective model
##################################################
def effective_tau(tau_0, L_b, v_b, mu_b, delta_b, phi_b):
    """Compute the effective conversion probability at a QH-SS corner.
    
    The value tau_0 is the one obtained from the microscopic model while the parameters
    labelled with '_b' correspond to the effective barrier.

    :param str tau_0: Hole probability computed with the microscopic model.
    :param str L_b: Length of the barrier.
    :param str v_b: Velocity in the barrier.
    :param str mu_b: Chemical potential in the barrier.
    :param str delta_b: Superconducting gap of the barrier.
    :param str phi_b: Superconducting phase of the barrier.
    """
    if mu_b == 0.:
        tau_val = tau_0 + (1-2*tau_0) * np.sin(delta_b*L_b/v_b)**2 \
                  - np.sqrt(tau_0*(1-tau_0)) * np.sin(2*delta_b*L_b/v_b) * np.sin(phi_b)
    
    else:
        alpha_b = np.sqrt(mu_b**2 + delta_b**2)*L_b/v_b
        beta_b = np.arcsin(np.sin(alpha_b)*delta_b/np.sqrt(mu_b**2 + delta_b**2))
        gamma_b = np.arctan(np.sqrt(mu_b**2 + delta_b**2)/(mu_b*np.tan(alpha_b)))
        
        tau_val = (np.sqrt(tau_0)*np.cos(beta_b) + np.sqrt(1-tau_0)*np.sin(beta_b))**2 \
                    - 4*np.sqrt(tau_0*(1-tau_0))*np.sin(beta_b)*np.cos(beta_b)*np.cos((phi_b - gamma_b)/2)**2
    return tau_val
