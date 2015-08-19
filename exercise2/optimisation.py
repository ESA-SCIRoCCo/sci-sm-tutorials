# # Copyright (c) 2014,Vienna University of Technology, Department of Geodesy
# and Geoinformation. All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Created on Apr 25, 2014

@author: Sebastian Hahn sebastian.hahn@geo.tuwien.ac.at
"""

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

try:
    from RS.dataspecific.WARP.interface import MetopA_WARP55_R21
    import teaching.RS.parameter_retrieval.exercise2.models as models
except ImportError:
    import models as models


def db2lin(val):
    """
    Converting from linear to dB domain.

    Parameters
    ----------
    val : float or numpy.ndarray
        Values in dB domain.

    Returns
    -------
    val : float or numpy.ndarray
        Values in linear domain.
    """
    return 10. ** (val / 10.)


def read_resam(gpi):
    """
    Read raw resampled ASCAT data for given grid point index (GPI).

    Parameters
    ----------
    gpi : int
        Grid point index.

    Returns
    -------
    ts_resam : pandas.DataFrame
        Resampled ASCAT time series.
    """
    w55r21 = MetopA_WARP55_R21(parameter='resam')
    ts_resam = w55r21.read_ts(gpi)

    valid = (ts_resam['sigf'] > -50) & (ts_resam['sigf'] < 10) & \
            (ts_resam['sigm'] > -50) & (ts_resam['sigm'] < 10) & \
            (ts_resam['siga'] > -50) & (ts_resam['siga'] < 10)

    print("#{:} NaN values masked in resampled "
          "ASCAT time series".format(np.sum(~valid)))
#     ts_resam = ts_resam[valid]
#     ts_resam['sigf'].plot()

    return ts_resam


def model_sig_forest(m_soil, theta, sand, clay, f_rms, m_veg, s_vol, freq,
                     temp):
    """
    Modelling backscattering of forest, according to Pulliainen et.al.

    Parameters
    ----------
    m_soil : float
        Soil moisture [m^3 m^-3]
    theta : float
        Incidence angle [degrees]
    sand : float
        Fraction of sand [-]
    clay : float
        Fraction of clay [-]
    f_rms : float
        Fractional root mean square height
        root mean square height / wavelength
    m_veg : float
        Vegetation water content [m^3 m^-3]
    s_vol : float
        Stem volume [m^3 ha^-1]
    freq : float
        frequency [GHz]
    temp : float
        Temperature [degree celsius]

    Returns
    -------
    sig_forest : float
        Forest backscatter [m^3 m^-3].
    """
    eps = models.eps_g(freq * 1e9, sand, clay, temp, m_soil)
    sig_bar = models.sigma0_bare(theta, eps, f_rms, freq * 1e9)
    sig_forest = models.sigma0_forest(theta, m_veg, s_vol, sig_bar['vv'])

    return sig_forest


def sig_sqr_diff(opt_params, ascat_inc, ascat_sig, static_params,
                 optional_param):
    """
    Optimisation equation.

    Parameters
    ----------
    opt_params : numpy.ndarray
        Array with optimisation parameter values.
    ascat_inc : numpy.ndarray
        Incidence angle [degrees]
    ascat_sig : numpy.ndarray
        Backscatter measurements [m^2 m^-2]
    static_params : dict
        Dictionary with static model parameters.
    optional_param : str
        Name of optional optimisation parameter.

    Returns
    -------
    residuals : float
        Differences between modeled and observed backscatter.
    """
    params = {'m_veg': opt_params[0], 'm_soil': opt_params[1], 'freq': 5.255}
    params.update(static_params)

    if optional_param != '':
        params[optional_param] = opt_params[2]

    residuals = 0
    for theta, sig in zip(ascat_inc, ascat_sig):
        params['theta'] = theta
        sig_forest = model_sig_forest(**params)
        residuals += (sig_forest - sig) ** 2

    return residuals


def optimise(params, m_veg_axes_lim=None, m_soil_axes_lim=None,
             timespan=('2009-01', '2009-12'), gpi=None):
    """
    This function is optimising the parameters vegetation water content
    'm_veg', soil moisture 'm_soil' and, if specified, a third optional
    parameter. The third optional parameter can eitehr be sand 'sand',
    clay 'clay', fractional root mean square height 'f_rms',
    stem volume 's_vol' or temperature 'temp'.

    Parameters
    ----------
    params : list of dicts
        Model parameters. At least
        four of the following parameters needs to be specified if an optional
        parameter has been selected, otherwise all of them needs to be
        specified: 'sand', 'clay', 'f_rms', 'temp', 's_vol'
        The optional parameter which should be optimized has to be specified
        using the two dictionary keys 'optional_param' which must be one of
        the four parameters and 'optional_x0' which is the start value of the
        optimization.
    gpi : int, optional
        Grid point index. If specified, it will read data from datapool.

    Returns
    -------
    df : pandas.DataFrame
        Optimised soil moisture, vegetation water concent and, if specified,
        optional optimised parameter.
    """

    if gpi is None:
        ts_resam = pd.read_csv(os.path.join("data", "2011528_2009.csv"), index_col=0,
                               parse_dates=True)[timespan[0]:timespan[1]]
        gpi = 2011528
    else:
        ts_resam = read_resam(gpi)[timespan[0]:timespan[1]]

    columns = ['m_veg', 'm_soil']

    all_params = ['sand', 'clay', 'f_rms', 'temp', 's_vol']

    df_list = []
    optional_x0s = []
    m_soil_x0s = []
    m_veg_x0s = []

    i = 1
    for static_p in params:

        m_veg_x0 = static_p.pop('m_veg_x0')
        m_veg_x0s.append(m_veg_x0)
        m_soil_x0 = static_p.pop('m_soil_x0')
        m_soil_x0s.append(m_soil_x0)

        x0 = np.array([m_veg_x0, m_soil_x0])

        # check if optional parameter has been set and specify default values
        if 'optional_param' in static_p:
            optional_param = static_p.pop('optional_param')
            if optional_param in ['sand', 'clay', 'f_rms', 'temp', 's_vol']:
                if 'optional_x0' in static_p:
                    optional_x0 = static_p.pop('optional_x0')
                    optional_x0s.append(optional_x0)
                    static_p[optional_param] = optional_x0
                    x0_op = np.array([m_veg_x0, m_soil_x0, optional_x0])
                    optional_param_set = True
                else:
                    raise ValueError('Initial guess for optional '
                                     'parameter not set')
            else:
                raise ValueError('Optional parameter unkown')
        else:
            optional_param_set = False

        # check if all parameters have been specified
        for param in all_params:
            if param not in static_p:
                raise KeyError('Parameter {:} not specified'.format(param))

        df = pd.DataFrame(index=ts_resam.index, columns=columns)
        df = df.fillna(np.nan)

        # optimise optional parameter, if specified
        if optional_param_set:
            op_value = []
            for index, row in ts_resam.iterrows():

                ascat_inc = np.array(row[['incf', 'incm', 'inca']].tolist())
                ascat_sig = \
                    db2lin(np.array(row[['sigf', 'sigm', 'siga']].tolist()))

                args = (ascat_inc, ascat_sig, static_p, optional_param)
                res = minimize(sig_sqr_diff, x0_op, args=args,
                               method='Nelder-Mead')

                if res['success'] == True:
                    op_value.append(res['x'][2])

            static_p[optional_param] = np.median(np.array(op_value))

            print("optional_param set#{:}: {:} = "
                  "{:}".format(i, optional_param, static_p[optional_param]))

        # optimise  m_soil and m_veg
        for index, row in ts_resam.iterrows():

            ascat_inc = np.array(row[['incf', 'incm', 'inca']].tolist())
            ascat_sig = \
                db2lin(np.array(row[['sigf', 'sigm', 'siga']].tolist()))

            args = (ascat_inc, ascat_sig, static_p, '')
            res = minimize(sig_sqr_diff, x0, args=args, method='Nelder-Mead')

            if res['success'] == True:
                df['m_veg'][index] = res['x'][0]
                df['m_soil'][index] = res['x'][1]

        df_list.append(df)
        i += 1

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(18, 6))
    fig.suptitle('GPI: {:}'.format(gpi))

    colors = mpl.rcParams["axes.color_cycle"]
    for i, (df, static_p, color) in enumerate(zip(df_list, params, colors)):

        if len(optional_x0s) - 1 >= i:
            str_static_p = \
                ', '.join(["%s: %r" % t for t in static_p.iteritems()
                           if t[0] != optional_param])

            str_static_p += (",\nm_veg_x0 = {:.2f}, m_soil_x0 = "
                             "{:.2f} OPT {:}: {:.3f} x0: {:.2f}").format(m_veg_x0s[i], m_soil_x0s[i],
                                                                         optional_param,
                                                                         static_p[optional_param],
                                                                         optional_x0s[i])
        else:
            str_static_p = \
                ', '.join("%s: %r" % t for t in static_p.iteritems())

            str_static_p += ",\nm_veg_x0 = {:.2f}, m_soil_x0 = {:.2f}".format(m_veg_x0s[i], m_soil_x0s[i])

        label = 'set#{:}: \n'.format(i + 1) + str_static_p

        ax0.plot_date(df.index, df['m_veg'], marker='None', linestyle='-',
                      label=label, color=color)
        ax1.plot_date(df.index, df['m_soil'], marker='None', linestyle='-',
                      label=label, color=color)

    if m_veg_axes_lim is not None:
        ax0.set_ylim(m_veg_axes_lim)

    m_veg_title = 'm_veg'

    ax0.set_title(m_veg_title, fontsize=10)
    ax0.set_ylabel('m_veg [m^3 m^-3]')
    ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
               fontsize=10)

    if m_soil_axes_lim is not None:
        ax1.set_ylim(m_soil_axes_lim)

    m_soil_title = 'm_soil'

    ax1.set_title(m_soil_title, fontsize=10)
    ax1.set_ylabel('m_soil [m^3 m^-3]')
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
    #           fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.68)

    col_names = ['set{:}'.format(i + 1) for i in np.arange(len(df_list))]
    axes_lim = {'m_veg': m_veg_axes_lim, 'm_soil': m_soil_axes_lim}
    titles = {'m_veg': m_veg_title, 'm_soil': m_soil_title}

    for column in columns:
        i = 1
        scatter_df = pd.DataFrame(columns=col_names)
        for df, static_p in zip(df_list, params):
            scatter_df['set{:}'.format(i)] = df[column]
            i += 1
        axes = scatter_matrix(scatter_df, figure=fig)
        axes.flat[0].figure.suptitle(titles[column])

        for j, ax in enumerate(axes.flatten()):
            ax.set_xlim(axes_lim[column])

            # do not scale y axes of the histograms
            if np.remainder(j + 1, len(params) + 1) != 1:
                ax.set_ylim(axes_lim[column])
                min_x, max_x = ax.get_xlim()
                min_y, max_y = ax.get_ylim()
                # find minimum lower left coordinate and maximum upper right
                min_ll = min([min_x, min_y])
                max_ur = max([max_x, max_y])
                ax.plot([min_ll, max_ur], [min_ll, max_ur], '--', c='0.6')

    return df_list


if __name__ == '__main__':

   params = [{'sand': 0.36, 'clay': 0.23, 'temp': 20,
              's_vol': 100, 'f_rms': 0.2,
              'm_veg_x0': 0.5, 'm_soil_x0': 0.01},
             {'sand': 0.36, 'clay': 0.23, 'temp': 20,
              's_vol': 300, 'f_rms': 0.2,
              'm_veg_x0': 0.5, 'm_soil_x0': 0.01}]

   optimise(params, timespan=['2009-06', '2009-12'], m_veg_axes_lim=None,
             m_soil_axes_lim=None)

   timespan = ['2009-01', '2009-06']
   m_veg_axes_lim = [-0.2, 1.6]
   m_soil_axes_lim = [-0.2, 0.4]

   params = [{'sand': 0.36, 'clay': 0.23, 'temp': 20, 's_vol': 300,
              'optional_param': 'f_rms', 'optional_x0': 0.1,
              'm_veg_x0': 0.5, 'm_soil_x0': 0.01},
             {'sand': 0.36, 'clay': 0.23, 'temp': 20, 's_vol': 300,
              'optional_param': 'f_rms', 'optional_x0': 0.4,
              'm_veg_x0': 0.5, 'm_soil_x0': 0.01}]

   optimise(params, timespan=timespan, m_veg_axes_lim=m_veg_axes_lim,
            m_soil_axes_lim=m_soil_axes_lim)


   plt.show()
