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

import models
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def plot(x_axis, x_range, y_axis, y_range=None, freq=None,
         sand=None, clay=None, temp=None, eps=None,
         theta=None, f_rms=None, sig_bare=None, m_soil=None,
         m_veg=None, s_vol=None, sig_canopy=None, sig_floor=None,
         save_to_file=None, polarization='vv'):
    """
    plot variables against each other.
    Give x_axis as string and y_axis as string and then the
    forest backscatter model is run for the given parameters and the
    range of values given in x_range.

    Exactly one keyword can be a tuple. If a tuple is found then the model is run
    for each element of the tuple.

    Parameters
    ----------
    x_axis : string
        Name of the variable which should be plotted on the x axis.
        valid values are all keywords except y_range and save_to_file and also
        'sig_for' to plot forest backscatter
    x_range : list
        Two element list giving the maximum and minimum value that the variable
        given in x_axis should be varied by.
        [min, max]
    y_axis : string
        Name of the variable which should be plotted on the y axis.
        valid values are all keywords except y_range and save_to_file and also
        'sig_for' to plot forest backscatter
    y_range : list, optional
        Minium and maximum of y axis. This can be set to produce plots that
        only show a specific range of y values. If it is not given the plot
        is automatically scaled
        [min, max]
    freq : float or tuple, optional
        frequency in GHz
    sand : float or tuple, optional
        Fraction of sand from 0 to 1
    clay : float or tuple, optional
        Fraction of clay from 0 to 1
    temp : float or tuple, optional
        Temperature in degree celsius
    eps : float or tuple, optional
        complex permittivity of the soil
    theta : float or tuple, optional
        Incidence angle in degrees
    f_rms : float or tuple, optional
        Fractional root mean square height
        root mean square height / wavelength
    sig_bare : float or tuple, optional
        Backscatter of bare soil in dB.
        If given it will not be calculated but used directly.
        This also means that the values given for eps, f_rms and freq
        will be ignored
    m_soil : float or tuple, optional
        Relative soil moisture in percent
    m_veg : float or tuple, optional
        Relative vegetation water content in percent
    s_vol : float or tuple, optional
        Stem volume in m^3ha^-1
    sig_canopy : float or tuple, optional
        Canopy backscatter
    sig_floor : float or tuple, optional
        floor backscatter, bare soil backscatter + 2 way canopy extraction
    polarization : string or tuple, optional
        polarization can be either 'vv', 'hh' or 'vh'
        or a tuple of strings
    save_to_file : string, optional
        if given the plot is not showed but saved to the given filename
    """
    # variable dictionary for flexible lookup
    vd = dict(locals())

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
                 's_vol': '$m^3ha^{-1}$',
                 'sig_canopy': 'dB',
                 'sig_for': 'dB',
                 'sig_floor': 'dB',
                 'polarization': ''}

    # find tuple in keywords
    nr_tuples = 0
    tuple_key = 'no_variant'
    for key in vd:
        if type(vd[key]) == tuple:
            nr_tuples += 1
            tuple_key = key

    if nr_tuples > 1:
        raise Exception('More than one keyword is a tuple. You can only vary one parameter at a time.')

    if nr_tuples == 0:
        vd[tuple_key] = ['']

    var_x = np.linspace(x_range[0], x_range[1], 200)

    variants = tuple(vd[tuple_key])
    vd[x_axis] = var_x

    results = {}

    for variant in variants:

        vd[tuple_key] = variant

        if eps is None and y_axis in ['eps', 'sig_bare', 'sig_for', 'sig_floor']:
            vd['eps'] = models.eps_g(vd['freq'] * 1e9, vd['sand'], vd['clay'], vd['temp'], vd['m_soil'] / 100.0)

        if sig_bare is None and y_axis in ['sig_bare', 'sig_for', 'sig_floor']:
            vd['sig_bare'] = models.sigma0_bare(vd['theta'], vd['eps'], vd['f_rms'], vd['freq'] * 1e9)[vd['polarization']]

        if sig_canopy is None and y_axis in ['sig_canopy', 'sig_for']:
            vd['sig_canopy'] = models.sigma0_canopy(vd['theta'], vd['m_veg'] / 100.0, vd['s_vol'])

        if sig_floor is None and y_axis in ['sig_floor', 'sig_for']:
            vd['sig_floor'] = models.sigma0_floor(vd['sig_bare'], vd['theta'], vd['m_veg'] / 100.0, vd['s_vol'])

        if y_axis == 'sig_for':
            vd['sig_for'] = models.sigma0_forest(vd['theta'], vd['m_veg'] / 100.0,
                                                 vd['s_vol'], vd['sig_bare'], sigma_0_can=vd['sig_canopy'],
                                                 sigma_0_floor=vd['sig_floor'])

        if nr_tuples == 1:
            results['='.join([tuple_key, str(variant)])] = dict(vd)
        else:
            results[variant] = dict(vd)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for key in sorted(results.keys()):

        vd = results[key]

        x_axis_label = x_axis
        y_axis_label = y_axis

        if x_axis in ['sig_bare', 'sig_for', 'sig_floor', 'sig_canopy']:
            vd[x_axis] = 10 * np.log10(vd[x_axis])
            x_axis_label = '%s' % x_axis

        if y_axis in ['sig_bare', 'sig_for', 'sig_floor', 'sig_canopy']:
            vd[y_axis] = 10 * np.log10(vd[y_axis])
            y_axis_label = '%s' % y_axis

        if nr_tuples == 1:
            ax.plot(vd[x_axis], vd[y_axis].real, label=' '.join([str(key), unit_dict[tuple_key]]))
        else:
            ax.plot(vd[x_axis], vd[y_axis].real, label=' '.join([str(key)]))
        if is_complex(vd[y_axis]):
            # plt.plot(vd[x_axis], vd[y_axis].imag, label=' '.join([y_axis, str(key), 'imaginary part']))
            pass

    ax.set_xlabel(' '.join([x_axis_label, '[%s]' % unit_dict[x_axis]]))
    ax.set_ylabel(' '.join([y_axis_label, '[%s]' % unit_dict[y_axis]]))

    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, 1 - box.height * 0.85 * 1.1, box.width * 0.75, box.height * 0.85])

    if nr_tuples == 1:
        font_p = FontProperties()
        font_p.set_size('medium')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), prop=font_p,
                  ncol=3)

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', alpha=0)

    # prepare text for information box on right side
    labels = dict(results[results.keys()[0]])
    del labels['x_range']
    del labels['y_range']
    del labels['x_axis']
    del labels['y_axis']
    del labels['save_to_file']
    del labels[x_axis]
    del labels[y_axis]
    del labels[tuple_key]
    if 'eps' in labels.keys():
        if labels['eps'] is not None:
            labels['eps'] = labels['eps'].real
    infotext = []
    for label in sorted(labels.keys()):
        if locals()[label] is not None:
            infotext.append('%s = %s %s' % (label, labels[label], unit_dict[label]))

    infotext = '\n'.join(infotext)

    # place a text box in upper left in axes coords
    ax.text(1.03, 1, infotext, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    if y_range is not None:
        plt.ylim(y_range)

    if save_to_file is None:
        plt.show()
    else:
        plt.savefig(save_to_file)
        plt.close()


def is_complex(var):

    is_really_complex = False
    if type(var) == complex:
        is_really_complex = True

    if (type(var) == np.ndarray or
        type(var) == np.memmap or
        type(var) == np.array):

        if var.dtype == np.complex:
            is_really_complex = True

    return is_really_complex


if __name__ == '__main__':
    plot('theta', [25, 65], 'sig_floor', freq=5.25, eps=(30, 60, 50, 60, 71, 80, 90, 100), m_veg=30, s_vol=300, f_rms=0.5)

    plot('theta', [25, 65], 'sig_canopy', freq=5.25,
        m_veg=10, s_vol=100, polarization=('hh', 'vv', 'vh'))

    plot('theta', [25, 65], 'sig_floor', sand=0.2, clay=0.2, temp=23, freq=5.25,
         m_soil=30, sig_canopy=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), m_veg=0.1, s_vol=100, f_rms=1)
