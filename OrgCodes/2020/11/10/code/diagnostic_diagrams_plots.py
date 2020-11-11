import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import rc
from code.misc import color_map_califa_old
from scipy.stats import binned_statistic_2d
from code.misc import Kf_curve_plot, Kw_curve_plot, Gr_curve_plot, Es_curve_plot
from code.misc import SII_AGN_curve_plot, SII_LINERS_curve_plot, Es_SII_curve_plot
from code.misc import OI_AGN_curve_plot, OI_LINERS_curve_plot, Es_OI_curve_plot

rc('mathtext', **{'fontset': 'cm'})
rc('font', **{'size': 14})
rc('xtick', **{'labelsize': 10})
rc('ytick', **{'labelsize': 10})

def main_plot_function(x, y, z, params, ax, bins, statistic):
    counts, xbins, ybins = np.histogram2d(x, y, bins=bins,
                                            range=[params['xlim'],
                                                    params['ylim']])
    counts /= counts.max()
    mask_d = counts.transpose() == 0
    if z is not None:
        bin_means = binned_statistic_2d(x, y, z, bins=bins,
                                        range=[params['xlim'],
                                                params['ylim']],
                                        statistic=statistic).statistic
        if params['norm_value'] is not None:
            bin_means /= params['norm_value']
            bin_means = np.abs(bin_means)
        dens_map = bin_means.T
        dens_map[mask_d] = np.nan
    else:
        dens_map = counts.transpose()
        dens_map[mask_d] = np.nan
    if params['vmin'] is None:
        params['vmin'] = np.nanmin(dens_map)
    if params['vmax'] is None:
        params['vmax'] = np.nanmax(dens_map)
    plot = ax.imshow(dens_map, origin='lower', cmap=params['cmap'],
                        aspect='auto', extent=params['xlim']+params['ylim'],
                        vmin=params['vmin'], vmax=params['vmax'])
    ax.contour(counts.transpose(), params['levels'],
                extent=params['xlim']+params['ylim'],
                colors='k', linestyles='solid')
    if params['dcurves']:
        if params['type_plot'] == 'O3N2':
            Kf_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='dashed',
                      c='k')
            Kw_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='dashdot',
                      c='k')
            Gr_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='dotted',
                      c='k')
            Es_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='solid',
                      c='k')
        if params['type_plot'] == 'O3S2':
            SII_AGN_curve_plot(ax=ax, linestyle='dashdot', c='k')
            SII_LINERS_curve_plot(ax=ax, linestyle='dotted', c='k')
            Es_SII_curve_plot(ax=ax, linestyle='solid', c='k')
        if params['type_plot'] == 'O3O1':
            OI_AGN_curve_plot(ax=ax, linestyle='dashdot', c='k')
            OI_LINERS_curve_plot(ax=ax, linestyle='dotted', c='k')
            Es_OI_curve_plot(ax=ax, linestyle='solid', c='k')
    return plot

def get_params(type_plot):
    params = {}
    params['ylabel'] = r'$\log([\mathrm{OIII}]\lambda 5007/\mathrm{H}\beta)$'
    params['ylim'] = [-2.5, 1.2]
    if type_plot == 'O3N2':
        params['xlabel'] = r'$\log([\mathrm{NII}]\lambda 6583/\mathrm{H}\alpha)$'
        params['xlim'] = [-2.0, 0.5]
    if type_plot == 'O3S2':
        params['xlabel'] = r'$\log([\mathrm{SII}]\lambda 6716+30/\mathrm{H}\alpha)$'
        params['xlim'] = [-1.5, 0.5]
    if type_plot == 'O3O1':
        params['xlabel'] = r'$\log([\mathrm{OI}]\lambda 6300/\mathrm{H}\alpha)$'
        params['xlim'] = [-3.5, 0.2]
    params['cmap'] = color_map_califa_old()
    params['levels'] = [0.05, 0.25, 0.45, 0.65, 0.85]
    params['type_plot'] = type_plot
    return params

def DiagnosticDiagramMapPlot(x, y, z=None, type_plot='O3N2', ax=None,
                             xmin=None, xmax=None, ymin=None, ymax=None,
                             bins=70, vmin=None, vmax=None, labels=False,
                             levels=None, norm_value=None, dcurves=True,
                             statistic='mean', cbar=False,
                             cbar_label=None):
    if ax is None:
        ax_flag = True
        fig, ax = plt.subplots()
    else:
        ax_flag = False

    params = get_params(type_plot)
    params['vmin'] = vmin
    params['vmax'] = vmax
    if levels is not None:
        params['levels'] = levels
    if xmin is not None:
        params['xmin'] = xmin
    if xmax is not None:
        params['xmax'] = xmax 
    if ymin is not None:
        params['ymin'] = ymin
    if ymax is not None:
        params['ymax'] = ymax
    params['norm_value'] = norm_value
    params['dcurves'] = dcurves
    plot = main_plot_function(x, y, z, params, ax, bins, statistic)
    if labels:
        ax.set_xlabel(params['xlabel'])
        ax.set_ylabel(params['ylabel'])
    if cbar:
        cbar = fig.colorbar(plot, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    ax.set_xlim(params['xlim'][0], params['xlim'][1])
    ax.set_ylim(params['ylim'][0], params['ylim'][1])
    if ax_flag:
        fig.tight_layout()
    else:
        return plot
    # plt.show(block=False)

def DiagnosticDiagramScatterPlot(x, y, z=None, type_plot='O3N2', ax=None,
                                 xmin=None, xmax=None, ymin=None, ymax=None,
                                 vmin=None, vmax=None, labels=True,
                                 dcurves=True, alpha=1, color='k',
                                 s=20, marker='s', cbar=False,
                                 cbar_label=None):
    if ax is None:
        ax_flag = True
        fig, ax = plt.subplots()
    else:
        ax_flag = False

    params = get_params(type_plot)
    params['vmin'] = vmin
    params['vmax'] = vmax
    if xmin is not None:
        params['xlim'][0] = xmin
    if xmax is not None:
        params['xlim'][1] = xmax
    if ymin is not None:
        params['ylim'][0] = ymin
    if ymax is not None:
        params['ylim'][1] = ymax
    params['dcurves'] = dcurves
    if z is None:
        plot = ax.scatter(x, y, c=color, marker=marker, s=s, alpha=alpha,
                   rasterized=True)
    else:
        plot = ax.scatter(x, y, c=z, marker=marker, s=s, alpha=alpha,
                   raterized=True)
    if params['dcurves']:
        if params['type_plot'] == 'O3N2':
            Kf_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='dashed',
                      c='k')
            Kw_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='dashdot',
                      c='k')
            Gr_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='dotted',
                      c='k')
            Es_curve_plot(ax=ax, x_min=params['xlim'][0], linestyle='solid',
                      c='k')
        if params['type_plot'] == 'O3S2':
            SII_AGN_curve_plot(ax=ax, linestyle='dashdot', c='k')
            SII_LINERS_curve_plot(ax=ax, linestyle='dotted', c='k')
            Es_SII_curve_plot(ax=ax, linestyle='solid', c='k')
        if params['type_plot'] == 'O3O1':
            OI_AGN_curve_plot(ax=ax, linestyle='dashdot', c='k')
            OI_LINERS_curve_plot(ax=ax, linestyle='dotted', c='k')
            Es_OI_curve_plot(ax=ax, linestyle='solid', c='k')
    if labels:
        ax.set_xlabel(params['xlabel'])
        ax.set_ylabel(params['ylabel'])
    if cbar:
        cbar = fig.colorbar(plot, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    ax.set_xlim(params['xlim'][0], params['xlim'][1])
    ax.set_ylim(params['ylim'][0], params['ylim'][1])
    if ax_flag:
        fig.tight_layout()
    else:
        return plot
