# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
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
    Dielectricity of soil accoriding to Dobson et.al.

    Parameters
    ----------
    f: float
        frequency in hertz
    S : float
        proportion of Sand
    C : float
        proportion of clay
    T : float
        temperature in deg Celsius
    SM : float
        relative soil moisture
    """

    SM = np.where(SM < 0.0001, 0.0001, SM)

    alpha = 0.65
    rho_b = 1.3
    rho_s = 2.664

    beta1 = 1.2748 - 0.519 * S - 0.152 * C
    beta2 = 0.0133797 - 0.603 * S - 0.166 * C
    beta = beta1 - beta2 * 1j
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
    rt_w = (1.1109e-10 - 3.824e-12 * T + 6.938e-14 * T ** 2 - 5.096e-16 * T ** 3) / (2 * cmath.pi)

    e_fw1 = e_w_inf + (e_w0 - e_w_inf) / (1 + (2 * cmath.pi * f * rt_w) ** 2)
    e_fw2 = 2 * cmath.pi * f * rt_w * (e_w0 - e_w_inf) / (1 + (2 * cmath.pi * f * rt_w) ** 2) + sigma_eff * (rho_s - rho_b) / (2 * cmath.pi * f * e_0 * rho_s * SM)

    e_fw = e_fw1 - 1j * e_fw2
    eps_g = (1 + (rho_b / rho_s) * (e_s ** alpha - 1) + SM ** beta1 * e_fw1 ** alpha - SM) ** (1 / alpha) + 1j * (SM ** beta2 * e_fw2 ** alpha) ** (1 / alpha)

    return eps_g


def sigma0_bare(theta, eps_low, f_rms, f, eps_top=1):
    """
    Oh et.al. (1992) surface backscatter calculations

    This functions calculations surface backscatter using the Oh et al. (1992) surface model.
    References
    Oh et al., 1992, An empirical model and an inversion technique for rader scattering
    from bare soil surfaces. IEEE Trans. Geos. Rem., 30, pp. 370-380

    Parameters
    ----------
    theta : float
        incidence angle in degrees
    eps_low : complex number
        complex permittivity of lower medium
    f : float
        frequency in hertz
    f_rms: float
        fractional rms height. rms_height in meters divided by wavelength
    eps_top : complex number, optional
        complex permittivity of upper(incoming) medium
    """

    k_rms = (2. * cmath.pi) * f_rms

    eps_eff = eps_low / eps_top

    gam0 = gammah(eps_eff, 0)

    gamh = gammah(eps_eff, theta)
    gamv = gammav(eps_eff, theta)
    theta = theta / 180.0 * cmath.pi

    # precalulcate cosines of angles
    ct = np.cos(theta)

    # Oh model equations
    g = 0.7 * (1. - np.exp(-0.65 * k_rms ** 1.8))
    root_p = 1. - ((2. * theta / cmath.pi) ** (1. / (3. * gam0))) * np.exp(-k_rms)
    p = (1 - (2 * theta / cmath.pi) ** (1 / (3 * gam0)) * np.exp(-1 * k_rms)) ** 2

    q = 0.23 * np.sqrt(gam0) * (1. - np.exp(-k_rms))

    sigma0 = {}

    sigma0['vv'] = g * (ct * (ct * ct)) * (gamv + gamh) / root_p

    # sig0vv = ((g*(cos(theta))^3)/sqrt(p))*(gamv+gamh);

    sigma0['vh'] = q * sigma0['vv']
    sigma0['hh'] = root_p * root_p * sigma0['vv']

    return sigma0


def gammah(eps, theta):
    """
    incoherent reflectivity for H-pol

    Parameters
    ----------
    eps : complex number
        complex dielectric constant
    theta : float
        angle in degrees
    """

    theta_rad = theta / 180.0 * cmath.pi
    costheta = np.cos(theta_rad)
    neliojuuri = np.sqrt(eps - np.sin(theta_rad) ** 2)
    gammah = np.abs((costheta - neliojuuri) / (costheta + neliojuuri)) ** 2

    return gammah


def gammav(eps, theta):
    """
    incoherent reflectivity for V-pol

    Parameters
    ----------
    eps : complex number
        complex dielectric constant
    theta : float
        angle in degrees
    """
    theta_rad = theta / 180.0 * cmath.pi
    costheta = np.cos(theta_rad)
    neliojuuri = np.sqrt(eps - np.sin(theta_rad) ** 2)
    gammav = np.abs((eps * costheta - neliojuuri) / (eps * costheta + neliojuuri)) ** 2

    return gammav


def sigma0_canopy(theta, m_canopy, volume, const=-5.12e-3):
    """
    backscattering of forest canopy, according to Pulliainen et.al.

    Parameters
    ----------
    theta : float
        incidence angle in degrees
    m_canopy : float
        relative canopy moisture
    volume : float
        forest stem volume
    const : float, optional
        some kind of constant - not documented
    """
    angle_rad = theta / 180.0 * cmath.pi

    sigma_0_can = 0.131 * m_canopy * np.cos(angle_rad) * (1 -
                  np.exp(const * m_canopy * volume / np.cos(angle_rad)))
    return sigma_0_can


def sigma0_floor(sigma_s, theta, m_canopy, volume, const=-5.12e-3):
    """
    backscattering of forest canopy, according to Pulliainen et.al.

    Parameters
    ----------
    sigma_s : float
        backscatter of bare soil
    theta : float
        incidence angle in degrees
    m_canopy : float
        relative canopy moisture
    volume : float
        forest stem volume
    const : float, optional
        some kind of constant - not documented
    """
    angle_rad = theta / 180.0 * cmath.pi

    sigma_0_floor = sigma_s * np.exp(const * m_canopy * volume / np.cos(angle_rad))
    return sigma_0_floor


def sigma0_forest(theta, m_canopy, volume, sigma0_b,
                  sigma_0_can=None, sigma_0_floor=None, const=-5.12e-3):
    """
    backscattering of forest, according to Pulliainen et.al.

    Parameters
    ----------
    theta : float
        incidence angle in degrees
    m_canopy : float
        relative canopy moisture
    volume : float
        forest stem volume
    sigma0_b : float
        backscatter of bare soil
    sigma_0_can : float, optional
        canopy backscatter
    sigma_0_floor : float, optional
        floor backscatter
    const : float, optional
        some kind of constant - not documented
    """

    angle_rad = theta / 180.0 * cmath.pi

    # ASCAT has VV polarization
    sigma_s = sigma0_b

    if sigma_0_can is None:
        sigma_0_can = sigma0_canopy(theta, m_canopy, volume, const=const)

    if sigma_0_floor is None:
        sigma_0_floor = sigma0_floor(sigma_s, theta, m_canopy, volume, const=const)

    sigma_cvv = sigma_0_can + sigma_0_floor

    return sigma_cvv.real
