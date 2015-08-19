# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy
# and Geoinformation. All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Mar 11, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import cmath
import numpy as np


def eps_g(f, S, C, T, SM):
    """
    This functions calculations the dielectricity of soil
    according to Dobson et.al.

    Parameters
    ----------
    f : float
        Frequency [Hz]
    S : float
        Proportion of sand [-]
    C : float
        Proportion of clay [-]
    T : float
        Temperature [deg Celsius]
    SM : float
        Soil moisture [m^3 m^-3]

    Returns
    -------
    eps_g : complex number
        Dielectricity of soil for given input parameters.
    """

    SM = np.where(SM < 0.0001, 0.0001, SM)

    alpha = 0.65
    rho_b = 1.3
    rho_s = 2.664

    beta1 = 1.2748 - 0.519 * S - 0.152 * C

    # DIFFERENCE
    # beta2 = 0.0133797 - 0.603 * S - 0.166 * C
    beta2 = 1.33797 - 0.603 * S - 0.166 * C

    # beta = beta1 - beta2 * 1j
    # dielectricity of solid soil
    e_s = 4.7

    # from Debye equation (or e_fw=79.8-4.3i)

    # where
    # sigma_eff=-1.645+1.939*rho_b-2.013*S+1.594*C;
    # or
    sigma_eff = 0.0467 + 0.2204 * rho_b - 0.4111 * S + 0.6614 * C

    e_0 = 8.854e-12
    e_w_inf = 4.9
    e_w0 = 87.134 - 1.949e-1 * T - 1.276e-2 * T ** 2 + 2.491e-4 * T ** 3
    rt_w = (1.1109e-10 - 3.824e-12 * T + 6.938e-14 * T ** 2 - 5.096e-16 * \
            T ** 3) / (2 * cmath.pi)

    e_fw1 = e_w_inf + (e_w0 - e_w_inf) / (1 + (2 * cmath.pi * f * rt_w) ** 2)
    e_fw2 = 2 * cmath.pi * f * rt_w * (e_w0 - e_w_inf) / \
            (1 + (2 * cmath.pi * f * rt_w) ** 2) + sigma_eff * \
            (rho_s - rho_b) / (2 * cmath.pi * f * e_0 * rho_s * SM)

    # e_fw = e_fw1 - 1j * e_fw2

    # DIFFERENCE
    # eps_g = (1 + (rho_b / rho_s) * (e_s ** alpha - 1) + SM ** beta1 * \
    #         e_fw1 ** alpha - SM) ** (1 / alpha) + 1j * \
    #         (SM ** beta2 * e_fw2 ** alpha) ** (1 / alpha)

    eps_g = (1 + (rho_b / rho_s) * (e_s ** alpha - 1) + SM ** beta1 * \
             e_fw1 ** alpha - SM) ** (1 / alpha) - 1j * \
             (SM ** beta2 * e_fw2 ** alpha) ** (1 / alpha)

    return eps_g


def sigma0_bare(theta, eps_low, f_rms, f, eps_top=1):
    """
    This functions calculations surface backscatter using the
    Oh et al. (1992) surface model.

    References
    Oh et al., 1992, An empirical model and an inversion technique for
    rader scattering from bare soil surfaces. IEEE Trans. Geos. Rem.,
    30, pp. 370-380

    Parameters
    ----------
    theta : float
        Incidence angle [degrees]
    eps_low : complex number
        Complex permittivity of lower medium
    f : float
        Frequency [Hz]
    f_rms: float
        Fractional rms height. rms_height in meters divided by wavelength
    eps_top : complex number, optional
        Complex permittivity of upper(incoming) medium
    """
    k_rms = (2. * cmath.pi) * f_rms
    eps_eff = eps_low / eps_top

    gam0 = gammah(eps_eff, 0)
    gamh = gammah(eps_eff, theta)
    gamv = gammav(eps_eff, theta)
    theta = np.radians(theta)

    # precalulcate cosines of angles
    ct = np.cos(theta)

    # Oh model equations
    g = 0.7 * (1. - np.exp(-0.65 * k_rms ** 1.8))
    root_p = 1. - ((2. * theta / cmath.pi) ** (1. / (3. * gam0))) * \
             np.exp(-k_rms)
#     p = (1 - (2 * theta / cmath.pi) ** (1 / (3 * gam0)) * \
#         np.exp(-1 * k_rms)) ** 2

    q = 0.23 * np.sqrt(gam0) * (1. - np.exp(-k_rms))

    sigma0 = {}

    sigma0['vv'] = g * (ct * (ct * ct)) * (gamv + gamh) / root_p

    # sig0vv = ((g*(cos(theta))^3)/sqrt(p))*(gamv+gamh);

    sigma0['vh'] = q * sigma0['vv']
    sigma0['hh'] = root_p * root_p * sigma0['vv']

    return sigma0


def gammah(eps, theta):
    """
    Incoherent reflectivity for H-pol

    Parameters
    ----------
    eps : complex number
        Complex dielectric constant
    theta : float
        Incidence angle [degrees]

    Returns
    -------
    gammav : complex number
        Incoherent reflectivity.
    """
    theta_rad = np.radians(theta)
    costheta = np.cos(theta_rad)
    neliojuuri = np.sqrt(eps - np.sin(theta_rad) ** 2)
    gammah = np.abs((costheta - neliojuuri) / (costheta + neliojuuri)) ** 2

    return gammah


def gammav(eps, theta):
    """
    Incoherent reflectivity for V-pol

    Parameters
    ----------
    eps : complex number
        Complex dielectric constant
    theta : float
        Incidence angle [degrees]

    Returns
    -------
    gammav : complex number
        Incoherent reflectivity.
    """
    theta_rad = np.radians(theta)
    costheta = np.cos(theta_rad)
    neliojuuri = np.sqrt(eps - np.sin(theta_rad) ** 2)
    gammav = np.abs((eps * costheta - neliojuuri) / \
                    (eps * costheta + neliojuuri)) ** 2

    return gammav


def sigma0_canopy(theta, m_canopy, volume, const=-5.12e-3):
    """
    This function computes the backscatter of forest canopy, according
    to Pulliainen et.al.

    Parameters
    ----------
    theta : float
        Incidence angle [degrees]
    m_canopy : float
        Canopy moisture [m^3 m^-3]
    volume : float
        Forest stem volume [m^3 ha^-1]
    const : float, optional
        Some kind of constant - not documented in original code.

    Returns
    -------
    sigma_0_can : float
        Backscatter of forest canopy [m^2 m^-2]
    """
    theta_rad = np.radians(theta)

    sigma_0_can = 0.131 * m_canopy * np.cos(theta_rad) * (1 -
                  np.exp(const * m_canopy * volume / np.cos(theta_rad)))

    return sigma_0_can


def sigma0_floor(sigma_s, theta, m_canopy, volume, const=-5.12e-3):
    """
    WRONG DOCUMENTATION

    This function computes the backscatter of forest canopy, according
    to Pulliainen et.al.

    Parameters
    ----------
    sigma_s : float
        Backscatter of bare soil [m^2 m^-2]
    theta : float
        Incidence angle [degrees]
    m_canopy : float
        Canopy moisture [m^3 m^-3]
    volume : float
        Forest stem volume [m^3 ha^-1]
    const : float, optional
        Some kind of constant - not documented in original code.

    Returns
    -------
    sigma_0_floor : float
        Backscatter of forest canopy [m^2 m^-2]
    """
    angle_rad = np.radians(theta)

    sigma_0_floor = sigma_s * np.exp(const * m_canopy * \
                                     volume / np.cos(angle_rad))
    return sigma_0_floor


def sigma0_forest(theta, m_canopy, volume, sigma0_b,
                  sigma_0_can=None, sigma_0_floor=None, const=-5.12e-3):
    """
    This function computes the backscattering of a forest, according
    to Pulliainen et.al.

    Parameters
    ----------
    theta : float
        Incidence angle [degrees]
    m_canopy : float
        Canopy moisture [m^3 m^-3]
    volume : float
        Forest stem volume [m^3 ha^-1]
    sigma0_b : float
        Backscatter of bare soil [m^2 m^-2]
    sigma_0_can : float, optional
        Canopy backscatter [m^2 m^-2]
    sigma_0_floor : float, optional
        Floor backscatter [m^2 m^-2]
    const : float, optional
        Some kind of constant - not documented in original code.

    Returns
    -------
    sigma_cvv : float
        Forest backscatter [m^2 m^-2] for VV polarisation.
    """
    sigma_s = sigma0_b

    if sigma_0_can is None:
        sigma_0_can = sigma0_canopy(theta, m_canopy, volume, const)

    if sigma_0_floor is None:
        sigma_0_floor = sigma0_floor(sigma_s, theta, m_canopy, volume, const)

    sigma_cvv = sigma_0_can + sigma_0_floor

    return sigma_cvv.real
