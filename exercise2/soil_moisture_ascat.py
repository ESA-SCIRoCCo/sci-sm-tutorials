# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

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
Created on Dec 19, 2013

port of the FMI ASCAT soil moisture retrieval code to python
based on the MATLAB code from Tuomo Smolander (tuomo.smolander@fmi.fi)

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''


import netCDF4
import cmath
import numpy as np
from scipy.optimize import minimize
import pandas as pd


def read_ascat_data(filename):
    """
    reads the ascat data from the round robin data package which is saved in
    netCDF

    Parameters
    ----------
    filename : string
    """

    ncdata = netCDF4.Dataset(filename)

    lat = ncdata.variables['lat'][:]
    lon = ncdata.variables['lon'][:]
    time = netCDF4.num2date(ncdata.variables['time'][:], ncdata.variables['time'].units)
    siga = ncdata.variables['siga'][:]
    sigm = ncdata.variables['sigm'][:]
    sigf = ncdata.variables['sigf'][:]
    inca = ncdata.variables['inca'][:]
    incm = ncdata.variables['incm'][:]
    incf = ncdata.variables['incf'][:]

    ncdata.close()

    return lat, lon, time, siga, sigm, sigf, inca, incm, incf


def calculate_soilm(siga, sigm, sigf, inca, incm, incf,
                    forestV=100, T=10, S=0.2, C=0.2):
    """
    calculates soil moisture

    Parameters
    ----------
    siga : numpy.array
        sigma0 of aft beam
    sigm : numpy.array
        sigma0 of mid beam
    sigf : numpy.array
        sigma0 of fore beam
    inca : numpy.array
        incidence angle of aft beam
    incm : numpy.array
        incidence angle of mid beam
    incf : numpy.array
        incidence angle of fore beam
    forestV: int, optional
        forest stem volume
    T : float, optional
        temperature
    S : float, optional
        soil composition sand
    C : float, optional
        soil composition clay
    """

    sigadb = 10. ** (siga / 10.)
    sigmdb = 10. ** (sigm / 10.)
    sigfdb = 10. ** (sigf / 10.)

    soilm = np.empty((siga.size, 3))

    # optimize surface roughness for station
    print soilm.size
    for i in np.arange(soilm.shape[0]):
        # print i
        init_guess = np.array([1, 0.1, 0.01])
        sigma = np.array([sigadb[i], sigmdb[i], sigfdb[i]])
        angle = np.array([inca[i], incm[i], incf[i]])
        result = minimize(forest_model_c_scat_rms, init_guess, args=(sigma, forestV, angle, T, S, C),
                          method='Nelder-Mead')
        soilm[i, :] = result.x

    rms_g = np.median(soilm[:, 2])
    print rms_g

#     c = 2.998e8
#     rms_g = 1 * c / (5.225 * 1e9)

    soilm = np.empty((siga.size, 2))

    # optimize soil moisture using median surface roughness
    for i in np.arange(soilm.shape[0]):
        # print i
        init_guess = np.array([1, 0.1])
        sigma = np.array([sigadb[i], sigmdb[i], sigfdb[i]])
        angle = np.array([inca[i], incm[i], incf[i]])
        result = minimize(forest_model_c_scat_soilm, init_guess, args=(rms_g, sigma, forestV, angle, T, S, C),
                          method='Nelder-Mead')
        soilm[i, :] = result.x

    return soilm


def forest_model_c_scat(m_canopy, m_soil, rms_g, volume, angle, T, S, C,
                            freq=5.255, eps_top=1, const=-5.12e-3):
    """
    backscattering of forest, according to Pulliainen et.al.

    Parameters
    ----------
    m_canopy : float
        relative canopy
    m_soil : float
        relative soil moisture
    rms_g : float
        surface roughness of ground in meters
    volume : float
        forest stem volume
    angle : float
        incidence angle in degrees
    T : float
        temperature in deg Celsius
    S : float
        proportion of sand
    C : float
        proportion of clay
    freq : float, optional
        frequency of EM wave in GHz
    eps_top : float, optional
        dielectricity constant of top layer
    const : float, optional
        some kind of constant - not documented
    """

    freqHz = freq * 1e9

    angle_rad = angle / 180.0 * cmath.pi

    eps_low = eps_g_func(freqHz, S, C, T, m_soil)

    # backscatter soil
    sigma0 = oh_r(eps_top, eps_low, freqHz, angle, rms_g)

    # ASCAT has VV polarization
    sigma_s = sigma0['vv']
    sigma_cvv = 0.131 * m_canopy * cmath.cos(angle_rad) * (1 -
                cmath.exp(const * m_canopy * volume / cmath.cos(angle_rad))) + (
                sigma_s * cmath.exp(const * m_canopy * volume / cmath.cos(angle_rad)))

    return sigma_cvv.real


def forest_model_c_scat_rms(m, sigma, volume, angle, T, S, C,
                            freq=5.255, eps_top=1, const=-5.12e-3):
    """
    backscattering of forest, according to Pulliainen et.al.

    for optimizing rms_g

    Parameters
    ----------
    m : numpy.array
        3 elements
            1. relative canopy
            2. relative soil moisture
            3. surface roughness of ground in meters
    sigma : numpy.array
        3 elements, sigma0 aft, mid and fore beam
    angle : numpy.array
        3 elements, incidence angle aft, mid and fore beam
    """
    return ((forest_model_c_scat(m[0], m[1], m[2], volume, angle[0], T, S, C,
                               freq=freq, eps_top=eps_top, const=const) - sigma[0]) ** 2 +
            (forest_model_c_scat(m[0], m[1], m[2], volume, angle[1], T, S, C,
                               freq=freq, eps_top=eps_top, const=const) - sigma[1]) ** 2 +
            (forest_model_c_scat(m[0], m[1], m[2], volume, angle[2], T, S, C,
                               freq=freq, eps_top=eps_top, const=const) - sigma[2]) ** 2)


def forest_model_c_scat_soilm(m, rms_g, sigma, volume, angle, T, S, C,
                              freq=5.255, eps_top=1, const=-5.12e-3):
    """
    backscattering of forest, according to Pulliainen et.al.

    optimize soil moisture and canopy with fixed rms_g
    Parameters
    ----------
    m : numpy.array
        2 elements,
        0: relative canopy
        1: relative soil moisture
    sigma : numpy.array
        3 elements, sigma0 aft, mid and fore beam
    angle : numpy.array
        3 elements, incidence angle aft, mid and fore beam
    """
    return ((forest_model_c_scat(m[0], m[1], rms_g, volume, angle[0], T, S, C,
                               freq=freq, eps_top=eps_top, const=const) - sigma[0]) ** 2 +
            (forest_model_c_scat(m[0], m[1], rms_g, volume, angle[1], T, S, C,
                               freq=freq, eps_top=eps_top, const=const) - sigma[1]) ** 2 +
            (forest_model_c_scat(m[0], m[1], rms_g, volume, angle[2], T, S, C,
                               freq=freq, eps_top=eps_top, const=const) - sigma[2]) ** 2)


def eps_g_func(f, S, C, T, SM):
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

    alpha = 0.65
    rho_b = 1.3
    rho_s = 2.664

    beta1 = 1.2748 - 0.519 * S - 0.152 * C
    beta2 = 1.33797 - 0.603 * S - 0.166 * C
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

    eps_g = (1 + (rho_b / rho_s) * (e_s ** alpha - 1) + SM ** beta1 * e_fw1 ** alpha - SM) ** (1 / alpha) - 1j * (SM ** beta2 * e_fw2 ** alpha) ** (1 / alpha)

    return eps_g


def oh_r(eps_top, eps_low, f, theta, rms_g):
    """
    Oh et.al. (1992) surface backscatter calculations

    This functions calculations surface backscatter using the Oh et al. (1992) surface model.
    References
    Oh et al., 1992, An empirical model and an inversion technique for rader scattering
    from bare soil surfaces. IEEE Trans. Geos. Rem., 30, pp. 370-380

    Parameters
    ----------
    eps_top : complex number
        complex permittivity of upper(incoming) medium
    eps_low : complex number
        complex permittivity of lower medium
    f : float
        frequency in hertz
    theta : float
        incidence angle in degrees
    rms_g : float
        surface rms height in m
    """

    # speed of light in m/s
    c = 2.998e8

    # calculate wavelength in upper medium by using real part of refarctive index
    n_upper = cmath.sqrt(eps_top)
    wavelength = (c / f) / n_upper.real
    k_rms = (2. * cmath.pi / wavelength) * rms_g

    eps_eff = eps_low / eps_top

    gam0 = gammah(eps_eff, 0)

    gamh = gammah(eps_eff, theta)
    gamv = gammav(eps_eff, theta)
    theta = theta / 180.0 * cmath.pi

    # precalulcate cosines of angles
    ct = cmath.cos(theta)

    # Oh model equations
    g = 0.7 * (1. - cmath.exp(-0.65 * k_rms ** 1.8))
    root_p = 1. - ((2. * theta / cmath.pi) ** (1. / (3. * gam0))) * cmath.exp(-k_rms)
    p = (1 - (2 * theta / cmath.pi) ** (1 / (3 * gam0)) * cmath.exp(-1 * k_rms)) ** 2

    q = 0.23 * cmath.sqrt(gam0) * (1. - cmath.exp(-k_rms))

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
    costheta = cmath.cos(theta_rad)
    neliojuuri = cmath.sqrt(eps - cmath.sin(theta_rad) ** 2)
    gammah = abs((costheta - neliojuuri) / (costheta + neliojuuri)) ** 2

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
    costheta = cmath.cos(theta_rad)
    neliojuuri = cmath.sqrt(eps - cmath.sin(theta_rad) ** 2)
    gammav = abs((eps * costheta - neliojuuri) / (eps * costheta + neliojuuri)) ** 2

    return gammav

if __name__ == '__main__':

    import os
    import matplotlib.pyplot as plt

    filename = os.path.join('/media', 'sf_D', 'small_projects', 'cpa_2013_12_FMI_retrieval',
                            'validation', '20070101205653_20111230215811-ESACCI-SOILMOISTURE-SSMV-ASCAT_AMMA[AF]-f00.1.nc')

    lat, lon, time, siga, sigm, sigf, inca, incm, incf = read_ascat_data(filename)

    soilm = calculate_soilm(siga, sigm, sigf, inca, incm, incf)

    data = pd.DataFrame(soilm, index=time)
    data.to_csv(os.path.join('/media', 'sf_D', 'small_projects', 'cpa_2013_12_FMI_retrieval', 'test.csv'))
    data.plot()
    plt.show()






