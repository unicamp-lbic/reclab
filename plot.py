# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:41:28 2015

@author: thalita
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import evaluation as evalu



matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
colors = 'bgrcmyk'



def PR_curve(exp_db, conf, sweep, value, args):
    if args.set is None:
        raise ValueError('must use --set valid|test')
    exp_id = exp_db.get_id(conf)
    metric_names = evalu.Metrics.ir_metric_names(args.set)
    df = get_xy([exp_id], exp_db, metric_names)
    if len(df) is 0:
        raise RuntimeError('No metrics available')

    RS_name = conf.get_name()
    plot_PR(df, metric_names, label=RS_name+' '+sweep+'='+str(value))


def plot_PR(df, metric_names,**plot_args):
    metric_names = [m for m in metric_names if m.find('F1') < 0]
    values = defaultdict(dict)
    for metric in metric_names:
        atN = int(metric.split('_')[0].split('@')[1])
        PR = metric.split('_')[0].split('@')[0]
        # values are (mean, std) tuples
        values[atN][PR] = df[metric]
    y = []
    yerr = []
    x = []
    xerr = []
    for atN in values:
        # values are (mean, std) tuples
        y.append(values[atN]['P'][0][0])
        yerr.append(values[atN]['P'][0][1])
        x.append(values[atN]['R'][0][0])
        xerr.append(values[atN]['R'][0][1])


    plt.errorbar(x, y, yerr, xerr,**plot_args)
    plt.grid(which='both')
    plt.ylabel('Precision')
    plt.xlabel('Recall')

def metrics(exp_db, conf, sweep, value, args):
    if args.atN is None:
        raise ValueError('must inform --atN N')
    if args.set is None:
        raise ValueError('must inform --set valid|test')

    if args.type.find('error') > -1:
        metric_names =  evalu.Metrics.error_metric_names(args.set)
    else:
        metric_names = evalu.Metrics.ir_metric_names(args.set, [int(args.atN)]) + \
            evalu.Metrics.error_metric_names(args.set)
    select = conf.as_dict()
    ids = exp_db.get_ids_dict(select)
    if ids is None:
        raise RuntimeError('No corresponding metrics available')
    df = get_xy(ids, exp_db, metric_names)
    if len(df) is 0:
        raise RuntimeError('No corresponding metrics available')
    RS_name = conf.get_name()
    bar_plot_metrics(df, metric_names,
                     label=RS_name+' '+sweep.replace('_',' ')+'='+str(value))
    plt.autoscale()


# static
__plot_count = 0


def bar_plot_metrics(dataframe, metrics, suptitle=None, **plotargs):
    global __plot_count
    width = int(np.ceil(np.sqrt(len(metrics))))
    height = int(np.ceil(len(metrics)/width))
    plt.gcf().set_size_inches(4*width, 3*height, forward=True)
    for i, metric in enumerate(metrics):
        plt.subplot(height, width, i+1)
        single_bar_plot(dataframe, metric, **plotargs)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if suptitle is not None:
        plt.suptitle(suptitle)
    __plot_count += 1


def single_bar_plot(df, metric, **plotargs):
    width = 0.7
    left = __plot_count + 0.5 * (1 - width)
    y = height = df[metric].values[0][0]
    x = left + width/2
    yerr = df[metric].values[0][1]
    plt.bar(left, height, width, color=colors[__plot_count], **plotargs)
    plt.errorbar(x, y, yerr, color='k')
    plt.legend(loc='lower right', fontsize='x-small', framealpha=0.8)
    plt.title(metric.replace('_', ' '), fontsize='small')
    plt.yticks(fontsize='small')
    plt.gca().get_xaxis().set_visible(False)
    plt.grid(axis='y', which='both')


def metrics_xaxis(exp_db, conf, sweep, value, args):
    if args.atN is None:
        raise ValueError('must inform --atN N')
    if args.set is None:
        raise ValueError('must inform --set valid|test')
    if args.xaxis is None:
        raise ValueError('must inform --xaxis param_name')

    if args.type.find('error') > -1:
        metric_names =  evalu.Metrics.error_metric_names(args.set)
    else:
        metric_names = evalu.Metrics.ir_metric_names(args.set, [int(args.atN)]) + \
            evalu.Metrics.error_metric_names(args.set)

    select = conf.as_dict()
    del select[args.xaxis]
    ids = exp_db.get_ids_dict(select)
    if ids is None:
        raise RuntimeError('No corresponding metrics available')
    df = get_xy(ids, exp_db, metric_names, args.xaxis)
    if len(df) is 0:
        raise RuntimeError('No corresponding metrics available')
    RS_name = conf.get_name()
    plt.gcf().set_size_inches(8, 4, forward=True)
    plot_metrics_xaxis(df, args.xaxis , metric_names,
                 label=RS_name+' '+sweep.replace('_', ' ')+'='+str(value))


def get_xy(ids, exp_db, metric_names, x_axis=None):
    metric_values = defaultdict(list)
    x_values = []
    for exp_id in ids:
        df = exp_db.db.loc[exp_id]
        if x_axis is not None:
            x_values.append(df[x_axis].values[0])
        for metric in metric_names:
            metric_values[metric].append(
                (df[metric].values.mean(), df[metric].values.std()))
    # at this point there is a list of points for every metric
    # and a list of x_values
    if x_axis is not None:
        metric_values.update({x_axis: x_values})
    new_df = pd.DataFrame(metric_values)
    return new_df


def plot_metrics_xaxis(dataframe, x_axis, metrics,
                       suptitle=None, **plotargs):
    width = int(np.ceil(np.sqrt(len(metrics))))
    height = int(np.ceil(len(metrics)/width))
    plt.gcf().set_size_inches(4*width, 3*height, forward=True)
    for i, metric in enumerate(metrics):
        plt.subplot(height, width, i+1)
        plot_single_metric_xaxis(dataframe, x_axis, metric, **plotargs)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if suptitle is not None:
        plt.suptitle(suptitle)


def plot_single_metric_xaxis(df, x_axis, metric, **plotargs):
    df.sort(x_axis, inplace=True)
    x = df[x_axis]
    y = [mean for mean, std in df[metric]]
    yerr = [std for mean, std in df[metric]]
    plt.errorbar(x, y, yerr, marker='o', **plotargs)
    plt.legend(loc='best', fontsize='small', framealpha=0.5)
    plt.title(metric.replace('_', ' '), fontsize='small')
    plt.yticks(fontsize='small')
    plt.xticks(fontsize='small')
    plt.xlabel(x_axis.replace('_', ' '), fontsize='small')
    plt.grid(which='both')
