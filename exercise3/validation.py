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
import pytesmo.df_metrics as df_metrics
import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.io.ismn.readers as ismn_readers
import os
from collections import OrderedDict

try:
    from RS.dataspecific.WARP.interface import MetopA_WARP55_R21
    import teaching.RS.parameter_retrieval.exercise3.models as models
    from RS.dataspecific.GLDAS_NOAH.interface import GLDAS025v1_nc
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

    residuals = 0
    for theta, sig in zip(ascat_inc, ascat_sig):
        params['theta'] = theta
        sig_forest = model_sig_forest(**params)
        residuals += (sig_forest - sig) ** 2

    return residuals


def std_ratio(x, y):

    return np.std(x) / np.std(y)


def df_std_ratio(df):

    return df_metrics._to_namedtuple(df_metrics.pairwise_apply(df, std_ratio), 'std_ratio')


def validate(params,
             timespan=('2009-01', '2009-12'), gpi=None, rescaling=None,
             y_axis_range=None):
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
    timespan : tuple, optional
        timespan to analyze
    gpi : int, optional
        Grid point index. If specified, it will read data from datapool.
    rescaling : string, optional
        rescaling method, one of 'min_max', 'linreg', 'mean_std' and 'lin_cdf_match'
        Default: None
        insitu is the reference to which is scaled
    y_axis_range : tuple, optional
        specify (min, max) of y axis


    Returns
    -------
    df : pandas.DataFrame
        Optimised soil moisture, vegetation water concent and, if specified,
        optional optimised parameter.
    """

    unit_dict = {'freq': 'GHz',
                 'sand': '',
                 'clay': '',
                 'temp': '$^\circ$C',
                 'eps': '',
                 'theta': '$^\circ$',
                 'f_rms': '',
                 'sig_bare': 'dB',
                 'm_soil': '%',
                 'm_veg': '%',
                 'm_soil_x0': '%',
                 'm_veg_x0': '%',
                 's_vol': '$m^3ha^{-1}$',
                 'sig_canopy': 'dB',
                 'sig_for': 'dB',
                 'sig_floor': 'dB',
                 'polarization': ''}

    param_should = ['sand', 'clay', 'temp',
                    's_vol', 'f_rms',
                    'm_veg_x0', 'm_soil_x0']

    for param in param_should:
        assert param in params.keys()

    if gpi is None:
        ts_resam = pd.read_csv(os.path.join(os.path.split(os.path.abspath(__file__))[0],'data','2011528_2009.csv'), index_col=0,
                               parse_dates=True)[timespan[0]:timespan[1]]
        gpi = 2011528
    else:
        ts_resam = read_resam(gpi)[timespan[0]:timespan[1]]

    m_veg_x0 = params.pop('m_veg_x0')
    m_soil_x0 = params.pop('m_soil_x0')
    columns = ['m_veg', 'm_soil']

    x0 = np.array([m_veg_x0, m_soil_x0])

    df = pd.DataFrame(index=ts_resam.index, columns=columns)
    df = df.fillna(np.nan)
    # optimise  m_soil and m_veg
    for index, row in ts_resam.iterrows():

        ascat_inc = np.array(row[['incf', 'incm', 'inca']].tolist())
        ascat_sig = \
            db2lin(np.array(row[['sigf', 'sigm', 'siga']].tolist()))

        args = (ascat_inc, ascat_sig, params, '')
        res = minimize(sig_sqr_diff, x0, args=args, method='Nelder-Mead')

        if res['success'] == True:
            df['m_veg'][index] = res['x'][0]
            df['m_soil'][index] = res['x'][1]

    str_static_p = \
                ', '.join("%s: %r" % t for t in locals().iteritems())

    str_static_p += ",\nm_veg_x0 = {:.2f}, m_soil_x0 = {:.2f}".format(m_veg_x0, m_soil_x0)

	
    ismn_file = os.path.join(os.path.split(os.path.abspath(__file__))[0],'data','ARM_ARM_Larned_sm_0.050000_0.050000_Water-Matric-Potential-Sensor-229L-W_20090101_20140527.stm')
    ismn_data = ismn_readers.read_data(ismn_file)
    insitu = pd.DataFrame(ismn_data.data['soil moisture']).rename(columns={'soil moisture': 'insitu'})
    gldas = pd.read_csv(os.path.join(os.path.split(os.path.abspath(__file__))[0],'data', 'GLDAS_737602.csv'), parse_dates=True, index_col=0)
    gldas.rename(columns={'086_L1': 'gldas'}, inplace=True)
    gldas = pd.DataFrame(gldas['gldas']) / 100.0
    ascat = pd.DataFrame(df['m_soil']).rename(columns={'m_soil': 'ascat'})

    matched = temp_match.matching(ascat, insitu, gldas)

    if rescaling is not None:
        scaled = scaling.scale(matched, rescaling, reference_index=1)
    else:
        scaled = matched

    metrics = OrderedDict()
    metrics['bias'] = df_metrics.bias(scaled)
    metrics['pearson'] = df_metrics.pearsonr(scaled)
    metrics['spearman'] = df_metrics.spearmanr(scaled)
    metrics['ubrmsd'] = df_metrics.rmsd(scaled)
    metrics['std_ratio'] = df_std_ratio(scaled)
    tcol_error = df_metrics.tcol_error(scaled)._asdict()

    ts_title = "Soil moisture. "
    if rescaling is not None:
        ts_title = ' '.join([ts_title, 'Rescaling: %s.' % rescaling])
        rmsd_title = 'unbiased RMSD'
    else:
        ts_title = ' '.join([ts_title, 'No rescaling.'])
        rmsd_title = 'RMSD'


    axes = scaled.plot(title=ts_title, figsize=(18, 8))
    plt.legend()

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', alpha=0)

    columns = ('ascat-insitu', 'ascat-gldas', 'insitu-gldas')
    row_labels = ['bias', 'pearson R', 'spearman rho', rmsd_title, 'stddev ratio']
    cell_text = []
    for metric in metrics:
        metric_values = metrics[metric]
        if type(metric_values) == tuple:
            metric_values = metric_values[0]
        metric_values = metric_values._asdict()
        cell_text.append(["%.2f" % metric_values['ascat_and_insitu'],
                              "%.2f" % metric_values['ascat_and_gldas'],
                              "%.2f" % metric_values['insitu_and_gldas']])


    table = plt.table(
              cellText=cell_text,
              colLabels=columns,
              colWidths=[0.1, 0.1, 0.1],
              rowLabels=row_labels, loc='bottom',
              bbox=(0.2, -0.5, 0.5, 0.3))

    tcol_table = plt.table(
              cellText=[["%.2f" % tcol_error['ascat'],
                         "%.2f" % tcol_error['gldas'],
                         "%.2f" % tcol_error['insitu']]],
              colLabels=('ascat      ', 'gldas      ', 'insitu      '),
              colWidths=[0.1, 0.1, 0.1],
              rowLabels=['Triple collocation error'], loc='bottom',
              bbox=(0.2, -0.6, 0.5, 0.1))

    plt.subplots_adjust(left=0.08, bottom=0.35, right=0.85)
    plt.draw()
#
    if y_axis_range is not None:
        axes.set_ylim(y_axis_range)

    params['m_veg_x0'] = m_veg_x0
    params['m_soil_x0'] = m_soil_x0

    infotext = []
    for label in sorted(param_should):
        infotext.append('%s = %s %s' % (label, params[label], unit_dict[label]))

    infotext = '\n'.join(infotext)

    # place a text box in upper left in axes coords
    axes.text(1.03, 1, infotext, transform=axes.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    axes = scatter_matrix(scaled)
    axes.flat[0].figure.suptitle(ts_title)

    # only draw 1:1 line if scaling was applied
    for j, ax in enumerate(axes.flatten()):
        if y_axis_range is not None:
            ax.set_xlim(y_axis_range)

        if np.remainder(j + 1, 3 + 1) != 1:
            if y_axis_range is not None:
                ax.set_ylim(y_axis_range)
            min_x, max_x = ax.get_xlim()
            min_y, max_y = ax.get_ylim()
            # find minimum lower left coordinate and maximum upper right
            min_ll = min([min_x, min_y])
            max_ur = max([max_x, max_y])
            ax.plot([min_ll, max_ur], [min_ll, max_ur], '--', c='0.6')


if __name__ == '__main__':

    params = {'sand': 0.36, 'clay': 0.23, 'temp': 20,
              's_vol': 100, 'f_rms': 0.2,
              'm_veg_x0': 0.5, 'm_soil_x0': 0.01}

    validate(params, timespan=['2009-01', '2009-10'], rescaling='mean_std', y_axis_range=[-0.1,0.6])

    plt.show()
