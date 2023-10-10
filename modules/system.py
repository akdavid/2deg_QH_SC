"""Classes constructing different Kwant's systems.
"""
"""
import kwant
import numpy as np
from .pauli import sigma_2, sigma_3
"""

class DeviceSingleCorner:
    """Construct a semi-infinite junction with a single corner.

    :param float theta_qh: QH angle (in degrees).
    :param str theta_sc: SC angle (in degrees).
    :param dict params: System's parameters.
    :param bool small: Must be set to True when small dimensions are used,
                      default to False.
    """

    def __init__(self, theta_qh, theta_sc, params, small=False):
        """
        The constructor.
        """
        self.params = params
        self.small_val = small
        self.m = 1 / (2 * self.params["t"] * self.params["a"] ** 2)
        self.omega = 2 * self.params["mu_qh"] / self.params["nu"]        
        self.lB = 1 / np.sqrt(self.m * self.omega)
        self.kF_sc = np.sqrt(2 * self.m * self.params["mu_sc"])
        self.kF_qh = np.sqrt(2 * self.m * self.params["mu_qh"])
        self.vF_sc = self.kF_sc / self.m
        self.V_barrier = self.params['Z'] * self.vF_sc / 2
        self.xi = self.vF_sc / self.params["delta"]    
        self.theta = theta_qh
        self.theta_qh = self.theta
        self.theta_sc = theta_sc        
        self.geometry = f'theta={str(theta_qh)}_theta_sc={str(theta_sc)}'
        self.dimensions = dict(
            L_qh = 40 * self.lB,              # Length of the QH region
            L_sc = 6 * self.xi,           # Length of the SC region
            L_interface = 40 * self.lB    # Length of the QH-SC interface
        )
        self.params_name = ''.join(['_%s=%s' % (key, value) for key, value in self.params.items()])
        self.device_type = 'single_corner'

    def make_system(self, onsite, onsite_qh, onsite_sc, hopping, hopping_qh, hopping_sc):
        """Make the (unfinalized) system.

        :param fun onsite: Onsite energy function.
        :param fun onsite_qh: Onsite energy function in QH region.
        :param fun onsite_sc: Onsite energy function in SC region.
        :param fun hopping: Hopping energy function.
        :param fun hopping_qh: Hopping energy function in QH region.
        :param fun hopping_sc: Hopping energy function in SC region.

        :returns: Unfinalized Kwant system.
        """
        lat = kwant.lattice.square(self.params["a"], norbs=2)
        syst = kwant.Builder()
        theta_rad = self.theta * np.pi / 180   
        theta_rad_sc = self.theta_sc * np.pi / 180   
        L_theta_qh = 15*self.lB
        if 0 < self.theta_sc < 90:
            L_theta_sc = 2*self.xi
        else:
            L_theta_sc = 15*self.lB
        L_x = L_theta_qh * np.sin(theta_rad)
        L_y = L_theta_qh * np.cos(theta_rad)
        L_x_sc = L_theta_sc * np.sin(theta_rad_sc)
        L_y_sc = L_theta_sc * np.cos(theta_rad_sc)
        W_sc = 2*L_y_sc + self.dimensions["L_interface"]
        W_sc_large = 8*L_y_sc + 4*self.dimensions["L_interface"]

        def top_qh_angle(pos):
            x, y = pos
            if y >= 0:
                return np.abs(y) <= -L_y / L_x * x + self.dimensions["L_interface"] / 2 and -L_x <= x <= 0
        
        def top_sc_angle(pos):
            x, y = pos
            if y >= 0:
                return np.abs(y) <= L_y_sc / L_x_sc * x + self.dimensions["L_interface"] / 2 and 0 < x <= L_x_sc
        
        def top_sc_angle_large(pos):
            x, y = pos
            if y >= 0:
                return np.abs(y) <= L_y_sc / L_x_sc * x + self.dimensions["L_interface"] / 2 and 0 < x <= 4*L_x_sc
        
        def top_qh_angle_negative(pos):
            x, y = pos
            return np.abs(L_y / L_x) * x + self.dimensions["L_interface"] / 2 <= y <= self.dimensions["L_interface"] / 2 + self.dimensions["L_qh"] \
               and 0 <= x <= self.dimensions["L_qh"]
        
        def top_sc_angle_negative(pos):
            x, y = pos
            return -np.abs(L_y_sc / L_x_sc) * x + self.dimensions["L_interface"] / 2 <= y <= self.dimensions["L_interface"] / 2 + self.dimensions["L_qh"] \
               and -self.dimensions["L_qh"] <= x <= 0

        def top_left_qh(pos):
            x, y = pos
            W_qh = self.dimensions["L_interface"] + 2 * L_y
            if y > 0:
                return np.abs(y) <= W_qh / 2 and -round(self.dimensions["L_qh"]) <= x <= -L_x

        def top_right_sc(pos):
            x, y = pos
            if y > 0:
                return np.abs(y) <= W_sc / 2 and L_x_sc <= x <= round(self.dimensions["L_sc"]) 
                
        def top_right_sc_large(pos):
            x, y = pos
            if y > 0:
                return np.abs(y) <= W_sc_large / 2 and 4*L_x_sc <= x <= max(round(5*L_x_sc), round(self.dimensions["L_sc"]))

        # Onsite
        # SC
        if theta_rad_sc == 0.:
            syst[(lat(x, y) for x in range(1, int(round(self.dimensions["L_sc"])) + 1)
                 for y in range(0, int(self.dimensions["L_interface"] / 2) + 1))] = onsite_sc
            syst[(lat(x, y) for x in range(1, int(round(self.dimensions["L_sc"])) + 1)
                 for y in range(int(self.dimensions["L_interface"] / 2), 
                                int(self.dimensions["L_interface"]) + 1))] = onsite_sc
            
        elif theta_rad_sc == np.pi/2:
            syst[(lat(x, y) for x in range(1, int(round(self.dimensions["L_sc"])) + 1)
                 for y in range(0, int(self.dimensions["L_interface"] / 2) + 1))] = onsite_sc
        
        elif theta_rad_sc < 0:
            syst[(lat(x, y) for x in range(1, int(round(self.dimensions["L_sc"])) + 1)
                  for y in range(0, int(np.round(self.dimensions["L_interface"]/2 
                                                 + self.dimensions["L_qh"])) + 1))] = onsite_sc
            syst[lat.shape(top_sc_angle_negative, (-1, self.dimensions["L_interface"] / 2 
                                                   + L_y_sc - 1))] = onsite_sc
            self.dimensions["L_qh"] = 2*self.dimensions["L_qh"]

        elif 90 < self.theta_sc and self.small_val==False:
            self.dimensions["L_interface"] = 4 * self.dimensions["L_interface"]
            syst[lat.shape(top_sc_angle_large, (2, 2))] = onsite_sc
            syst[lat.shape(top_right_sc_large, (4*L_x_sc + 1, 2))] = onsite_sc

        else:
            syst[lat.shape(top_sc_angle, (2, 2))] = onsite_sc
            syst[lat.shape(top_right_sc, (L_x_sc + 1, 2))] = onsite_sc

        # QH    
        if theta_rad == 0:
            syst[(lat(x, y) for x in range(-int(round(self.dimensions["L_qh"])), 0 + 1)
                 for y in range(0, int(np.round(self.dimensions["L_interface"] / 2)) + 1))] = onsite
            syst[(lat(x, y) for x in range(-int(round(self.dimensions["L_qh"])), 0 + 1)
                 for y in range(int(np.round(self.dimensions["L_interface"] / 2)), 
                                int(np.round(3*self.dimensions["L_interface"] / 4)) + 1))] = onsite_qh
        elif theta_rad < 0:
            syst[(lat(x, y) for x in range(-int(round(self.dimensions["L_qh"])), 0 + 1)
                  for y in range(0, int(np.round(self.dimensions["L_interface"] / 2)) + 1))] = onsite
            syst[(lat(x, y) for x in range(-int(round(self.dimensions["L_qh"])), 0 + 1)
                  for y in range(int(np.round(self.dimensions["L_interface"] / 2)), 
                                 int(np.round(self.dimensions["L_interface"] / 2 + self.dimensions["L_qh"])) + 1))] = onsite_qh
            syst[lat.shape(top_qh_angle_negative, (1, self.dimensions["L_interface"] / 2 + L_y - 1))] = onsite_qh
        elif theta_rad > 0:
            syst[lat.shape(top_qh_angle, (-1, 2))] = onsite
            syst[lat.shape(top_left_qh, (-L_x - 1, 2))] = onsite
        
        # Hopping
        syst[lat.neighbors()] = hopping
        # SC
        if theta_rad_sc < 0:
            for x in range(-int(np.round(self.dimensions["L_qh"])), 0+1):
                for y in range(int(np.round(self.dimensions["L_interface"] / 2)), 
                               int(np.round(self.dimensions["L_interface"] / 2 + self.dimensions["L_qh"]))):
                    if -np.abs(L_y_sc / L_x_sc) * x + self.dimensions["L_interface"] / 2 + 1 < y < self.dimensions["L_interface"] / 2 + self.dimensions["L_qh"]/2-1:
                        syst[lat(x, y), lat(x, y-1)] = hopping_sc
                        syst[lat(x+1, y), lat(x, y)] = hopping_sc
        # QH
        if theta_rad < 0:
            for x in range(0, int(np.round(self.dimensions["L_qh"]))+1):
                for y in range(int(np.round(self.dimensions["L_interface"] / 2)), 
                               int(np.round(self.dimensions["L_interface"] / 2 + self.dimensions["L_qh"]))):
                    if np.abs(L_y / L_x) * x + self.dimensions["L_interface"] / 2 + 1 < y <= self.dimensions["L_interface"] / 2 + self.dimensions["L_qh"]+1:
                        syst[lat(x, y), lat(x, y-1)] = hopping_qh
                        syst[lat(x, y), lat(x-1, y)] = hopping_qh

        sites = list(syst.sites()) 
        sites_pos = [site[1] for site in sites]
        if 90 < self.theta_sc and self.small_val==False:
            y_pos_at_L_sc = [site[1] for site in sites_pos if site[0] == max(int(round(5*L_x_sc)), round(self.dimensions["L_sc"]))]
        else:
            y_pos_at_L_sc = [site[1] for site in sites_pos if site[0] == int(round(self.dimensions["L_sc"]))]
        ymax_at_L_sc = np.max(y_pos_at_L_sc)

        # Leads
        sym_top_qh = kwant.TranslationalSymmetry((0, self.params["a"]))
        top_qh_lead = kwant.Builder(sym_top_qh, conservation_law=-sigma_3, particle_hole=sigma_2)
        top_qh_lead[(lat(x, 0) for x in range(-int(round(self.dimensions["L_qh"])),
                                              -int(round(self.dimensions["L_qh"]/2)) + 1))] = onsite
        top_qh_lead[lat.neighbors()] = hopping

        sym_bottom_hybrid = kwant.TranslationalSymmetry((0, -self.params["a"]))
        bottom_hybrid_lead = kwant.Builder(sym_bottom_hybrid, particle_hole=sigma_2)
        bottom_hybrid_lead[(lat(x, 0) for x in range(-int(round(self.dimensions["L_qh"])),
                                                     int(round(self.dimensions["L_sc"])) + 1))] = onsite
        bottom_hybrid_lead[lat.neighbors()] = hopping

        sym_right_sc = kwant.TranslationalSymmetry((self.params["a"], 0))
        right_sc_lead = kwant.Builder(sym_right_sc)
        right_sc_lead[(lat(0, y) for y in range(0, ymax_at_L_sc + 1))] = onsite_sc
        right_sc_lead[lat.neighbors()] = hopping_sc

        syst.attach_lead(top_qh_lead)
        syst.attach_lead(bottom_hybrid_lead)
        syst.attach_lead(right_sc_lead)

        return syst


class Device:
    """Construct a QH-SC junction with arbitrary QH angles and a rectangle-shaped SC.

    :param float theta_1: First QH angle (in degrees).
    :param float theta_2: Second QH angle (in degrees).
    :param dict params: System's parameters.
    """
    def __init__(self, theta_1, theta_2, params):
        """
        The constructor.
        """
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.params = params
        self.m = 1 / (2 * self.params["t"] * self.params["a"] ** 2)
        self.omega = 2 * self.params["mu_qh"] / self.params["nu"]
        self.lB = 1 / np.sqrt(self.m * self.omega)
        self.kF_sc = np.sqrt(2 * self.m * self.params["mu_sc"])
        self.kF_qh = np.sqrt(2 * self.m * self.params["mu_qh"])
        self.vF_sc = self.kF_sc / self.m
        self.V_barrier = self.params['Z'] * self.vF_sc / 2
        self.xi = self.vF_sc / self.params["delta"]
        self.geometry = f'theta_1={str(theta_1)}_theta_2={str(theta_2)}' # \
                        # + f'_theta_sc_1={str(theta_sc_1)}_theta_sc_2={str(theta_sc_2)}'
        self.dimensions = dict(
            L_qh=20 * self.lB,          # Length of the QH region
            L_sc=4 * self.xi,           # Length of the SC region
            L_interface=40 * self.lB    # Length of the QH-SC interface
        )
        self.params_name = ''.join(['_%s=%s' % (key, value) for key, value in self.params.items()])
        self.device_type = 'two_corner'

    def make_system(self, onsite, hopping):
        """Make the (unfinalized) system.

        :param fun onsite: Onsite energy function.
        :param fun hopping: Hopping energy function.

        :returns: Unfinalized Kwant system.
        """
        lat = kwant.lattice.square(self.params["a"], norbs=2)
        syst = kwant.Builder()

        theta_1_rad = self.theta_1 * np.pi / 180   
        theta_2_rad = self.theta_2 * np.pi / 180   
        L_theta = self.dimensions["L_interface"]/2.5
        L_x_1 = L_theta * np.sin(theta_1_rad)
        L_y_1 = L_theta * np.cos(theta_1_rad)
        L_x_2 = L_theta * np.sin(theta_2_rad)
        L_y_2 = L_theta * np.cos(theta_2_rad)
        W_qh_1 = self.dimensions["L_interface"] + 2 * L_y_1
        W_qh_2 = self.dimensions["L_interface"] + 2 * L_y_2

        def top_qh_angle(pos):
            x, y = pos
            return 0 <= y <= -L_y_1 / L_x_1 * x + self.dimensions["L_interface"] / 2 and -L_x_1 <= x <= 0

        def top_left_qh(pos):
            x, y = pos
            if y > 0:
                return np.abs(y) <= W_qh_1 / 2 and -round(self.dimensions["L_qh"]) <= x <= -L_x_1

        def bottom_qh_angle(pos):
            x, y = pos
            return L_y_2 / L_x_2 * x - self.dimensions["L_interface"] / 2 <= y <= 0 and -L_x_2 <= x <= 0

        def bottom_left_qh(pos):
            x, y = pos
            if y <= 0:
                return np.abs(y) <= W_qh_2 / 2 and -round(self.dimensions["L_qh"]) <= x <= -L_x_2

        syst[(lat(x, y) for x in range(1, int(round(self.dimensions["L_sc"])) + 1)
                for y in range(-int(np.round(self.dimensions["L_interface"] / 2)),
                                int(np.round(self.dimensions["L_interface"] / 2)) + 1))] = onsite
        
        if theta_1_rad == 0:
            syst[(lat(x, y) for x in range(-int(round(self.dimensions["L_qh"])), 0 + 1)
                for y in range(0, int(np.round(3*self.dimensions["L_interface"] / 4)) + 1))] = onsite
        else:
            syst[lat.shape(top_qh_angle, (-1, 2))] = onsite
            syst[lat.shape(top_left_qh, (-L_x_1 - 1, 2))] = onsite
        
        if theta_2_rad == 0:
            syst[(lat(x, y) for x in range(-int(round(self.dimensions["L_qh"])), 0 + 1)
                for y in range(-int(np.round(3*self.dimensions["L_interface"] / 4)), 0 + 1))] = onsite
        else:
            syst[lat.shape(bottom_qh_angle, (-1, -2))] = onsite
            syst[lat.shape(bottom_left_qh, (-L_x_2 - 1, -2))] = onsite

        syst[lat.neighbors()] = hopping

        # Leads
        sym_top_qh = kwant.TranslationalSymmetry((0, self.params["a"]))
        top_qh_lead = kwant.Builder(sym_top_qh, conservation_law=-sigma_3, particle_hole=sigma_2)
        top_qh_lead[(lat(x, 0) for x in range(-int(round(self.dimensions["L_qh"])),
                                              -int(round(self.dimensions["L_qh"] / 2)) + 1))] = onsite
        top_qh_lead[lat.neighbors()] = hopping

        sym_bottom_qh = kwant.TranslationalSymmetry((0, -self.params["a"]))
        bottom_qh_lead = kwant.Builder(sym_bottom_qh, conservation_law=-sigma_3, particle_hole=sigma_2)
        bottom_qh_lead[(lat(x, 0) for x in range(-int(round(self.dimensions["L_qh"])),
                                                 -int(round(self.dimensions["L_qh"] / 2)) + 1))] = onsite
        bottom_qh_lead[lat.neighbors()] = hopping
        
        sym_right_sc = kwant.TranslationalSymmetry((self.params["a"], 0))
        right_sc_lead = kwant.Builder(sym_right_sc)
        right_sc_lead[(lat(0, y) for y in range(-int(round(self.dimensions["L_interface"]/2)),
                                                int(round(self.dimensions["L_interface"]/2)) + 1))] = onsite
        right_sc_lead[lat.neighbors()] = hopping

        syst.attach_lead(top_qh_lead)
        syst.attach_lead(bottom_qh_lead)
        syst.attach_lead(right_sc_lead)

        return syst


class DeviceInfinite:
    """Construct a an infinite QH-SC interface (lead).

    :param dict params: System's parameters.
    """
    def __init__(self, params):
        """
        The constructor.
        """
        self.params = params
        self.m = 1 / (2 * self.params["t"] * self.params["a"] ** 2)
        self.omega = 2 * self.params["mu_qh"] / self.params["nu"]
        self.lB = 1 / np.sqrt(self.m * np.abs(self.omega))
        self.kF_sc = np.sqrt(2 * self.m * self.params["mu_sc"])
        self.kF_qh = np.sqrt(2 * self.m * self.params["mu_qh"])
        self.vF_sc = self.kF_sc / self.m
        self.V_barrier = self.params['Z'] * self.vF_sc / 2
        self.xi = self.vF_sc / self.params["delta"]
        self.dimensions = dict(
            L_qh=10 * self.lB,          # Length of the QH region
            L_sc=4 * self.xi,           # Length of the SC region
        )
        self.params_name = ''.join(['_%s=%s' % (key, value) for key, value in self.params.items()])
        self.device_type = 'infinite_hybrid'

    def make_system(self, onsite, hopping):
        """Make the (unfinalized) lead.

        :param fun onsite: Onsite energy function.
        :param fun hopping: Hopping energy function.

        :returns: Unfinalized Kwant lead.
        """
        lat = kwant.lattice.square(self.params["a"], norbs=2)
        sym = kwant.TranslationalSymmetry((0, -self.params["a"]))
        lead = kwant.Builder(sym)
        lead[(lat(x, 0) for x in range(-int(round(self.dimensions["L_qh"])),
                                       int(round(self.dimensions["L_sc"])) + 1))] = onsite
        lead[lat.neighbors()] = hopping
        return lead
