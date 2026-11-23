"""Edit/replot the 24 requested figures using only data/curves.csv.gz.
Examples:
  python plot_figures.py --list
  python plot_figures.py --only m2_addon_risk_all --formats svg pdf
  python plot_figures.py --only 'm_sweep_*' --formats png
Style: edit style.json; custom composition: edit the functions below.
"""
import argparse
import csv
import fnmatch
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
CONTROLS = ['RC', 'PQ', 'QI', 'PS', 'GC']
NEW = ['sm_continuous', 'mem_line', 'gmem_line', '2mem_line_2sweeps',
       'infosynth_trace', 'infosynth_bounded', 'gausscloud_paper_L1',
       'gausscloud_paper_L3', 'gausscloud_paper_L5']
METHODS = CONTROLS + NEW
GROUPS = {'all': METHODS,
          'binary': ['RC'] + NEW[:6],
          'gaussian': ['RC', 'PS', 'GC'] + NEW[6:]}
SWEEPS = {'m': ['m02_b04', 'm04_b04', 'm08_b04', 'm16_b04'],
          'beta': ['m04_b02', 'm04_b04', 'm04_b08', 'm04_b16']}
METRICS = {
    'ellipsoid_eta': ('eta_ellipsoid_approx', 'Approx. ellipsoidal-reference information ratio'),
    'risk': ('risk', 'Full posterior covariance trace'),
    'parameter_error': ('error', 'Squared parameter estimation error'),
    'global_gap': ('relative_global_gap', '(Analytic upper - achieved EIG) / upper'),
    'grid_eta_original': ('eta_grid_original', r'Original auxiliary grid ratio $\eta_{t,\mathrm{grid}}$'),
}
NOTES = {
    'ellipsoid_eta': 'Denominator: achieved integrated EIG from feasible full-dimensional search; global optimum not certified.',
    'risk': 'Original posterior-risk values are unchanged; reevaluation introduces no new answers.',
    'global_gap': 'Upper is log M in these data. This is the unresolved optimization gap for estimated EIG, not a numerical error certificate.',
    'grid_eta_original': 'Original seven-scale ratios are retained and relabeled; they are distinct from ellipsoidal-reference ratios.',
}


def figure_specs():
    result = {}
    for suffix in ['ellipsoid_eta', 'risk', 'global_gap', 'grid_eta_original']:
        for sweep in SWEEPS:
            result[f'{sweep}_sweep_{suffix}'] = ('original', sweep, suffix)
    for group in GROUPS:
        for suffix in ['ellipsoid_eta', 'risk', 'parameter_error', 'global_gap']:
            result[f'm2_addon_{suffix}_{group}'] = ('addon', group, suffix)
    result['m2_addon_ellipsoid_ratio_bounds_all'] = ('bounds', 'all', 'bounds')
    for suffix in ['ellipsoid_eta', 'risk', 'global_gap']:
        result[f'm2_addon_m_sweep_{suffix}'] = ('addon_sweep', 'm', suffix)
    return result


def load_curves(path):
    """A series key is (setting, method, metric); shape (10 interactions, time)."""
    result = {}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            for metric in [v[0] for v in METRICS.values()] + ['eta_lower_plugin', 'eta_upper_plugin']:
                if row[metric] == '':
                    continue
                key = (row['setting'], row['method'], metric)
                length = 129 if metric in ['risk', 'error'] else 128
                if key not in result:
                    result[key] = np.full((10, length), np.nan)
                result[key][int(row['interaction_id']), int(row['t'])] = float(row[metric])
    for key, value in result.items():
        if not np.isfinite(value).all(): raise ValueError(f'Incomplete plotted series: {key}')
    return result


def draw_series(ax, values, color, label, style, family):
    x = np.arange(values.shape[1])
    if style['show_individuals']:
        for v in values:
            ax.plot(x, v, color=color,
                    alpha=style[f'individual_alpha_{family}'],
                    lw=style['individual_width_addon' if family == 'addon' else 'individual_width_original'])
    ax.plot(x, values.mean(axis=0), color=color,
            lw=style[f'mean_width_{family}'], label=label)


def original_plot(curves, style, sweep, suffix, include_sg=False):
    metric, ylabel = METRICS[suffix]
    if suffix == 'ellipsoid_eta': ylabel = 'Approx. ellipsoidal-reference ratio'
    if suffix == 'global_gap': ylabel = '(Analytic upper − achieved EIG) / upper'
    visible = CONTROLS + (['SG'] if include_sg else [])
    fig, axes = plt.subplots(1, 4, figsize=style['figure_size_original'], sharey=True, constrained_layout=True)
    for ax, setting in zip(axes, SWEEPS[sweep]):
        for method in CONTROLS + ['SG']:
            color = style['sg_color'] if method == 'SG' else style['colors'][method]
            if method == 'SG' and not include_sg:
                # The published no-SG figures retained the old shared axis limits.
                # Updating dataLim keeps exactly that scale without drawing SG.
                v = curves[(setting, method, metric)]
                ax.update_datalim([[0, v.min()], [v.shape[1]-1, v.max()]])
                ax.autoscale_view()
            else:
                draw_series(ax, curves[(setting, method, metric)], color,
                            style['original_labels'][method], style, 'original')
        ax.set(title=f'M={int(setting[1:3])}, β={int(setting[5:])}, d=4',
               xlabel='Answers already received, t', xlim=(0, 128 if metric == 'risk' else 127))
        ax.grid(alpha=.15)
    axes[0].set_ylabel(ylabel)
    fig.legend(handles=[Line2D([], [], color=style['sg_color'] if m == 'SG' else style['colors'][m],
                               lw=2, label=style['original_labels'][m]) for m in visible],
               loc='outside upper center', ncol=len(visible), frameon=False)
    note = 'Thin: ten individual interactions. Thick: arithmetic mean; dashed if fewer than ten finite values.\n'
    note += NOTES[suffix] + '\nα=0.5, ρE=16. Raw curves, no ratio clipping.'
    if include_sg: note += ' SG stays in stage 0 for t=0,…,127.'
    fig.supxlabel(note, fontsize=8.5)
    return fig


def addon_plot(curves, style, group, suffix):
    metric, ylabel = METRICS[suffix]
    fig, ax = plt.subplots(figsize=style['figure_size_addon'])
    for method in GROUPS[group]:
        draw_series(ax, curves[('m02_b04', method, metric)], style['colors'][method],
                    style['labels'][method], style, 'addon')
    ax.set(xlabel='Answers already received, t', ylabel=ylabel,
           xlim=(0, 128 if metric in ['risk', 'error'] else 127),
           title='M=2, d=4, beta=4: existing ellipsoid protocol + added methods')
    ax.grid(alpha=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=4, frameon=False)
    note = 'Thin: 10 individual histories; thick: arithmetic mean. No smoothing or ratio clipping.'
    if suffix == 'ellipsoid_eta':
        note += '\nDenominator: achieved full-dimensional EIG; alpha=0.5, radius=16. Global optimum and posterior accuracy not certified.'
    elif suffix == 'global_gap':
        note += '\nThis is the achieved-to-upper gap under posterior approximation; it is not an integration-error certificate.'
    fig.text(.5, .005, note, ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .085, 1, 1])
    return fig


def bounds_plot(curves, style):
    fig, ax = plt.subplots(figsize=style['figure_size_addon'])
    for method in METHODS:
        lo = curves[('m02_b04', method, 'eta_lower_plugin')].mean(axis=0)
        hi = curves[('m02_b04', method, 'eta_upper_plugin')].mean(axis=0)
        x = np.arange(128)
        ax.plot(x, hi, color=style['colors'][method], lw=style['bounds_width'], label=style['labels'][method])
        ax.fill_between(x, lo, hi, color=style['colors'][method], alpha=style['bounds_alpha'])
    ax.set(xlabel='Answers already received, t', ylabel='Numerical ratio interval: N/U to N/L',
           title='M=2 ellipsoid reference: unresolved global optimization gap', xlim=(0, 127))
    ax.grid(alpha=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=4, frameon=False)
    fig.text(.5, .005, 'Intervals are plug-in bounds under numerical posterior approximations, not certified bounds for the continuous posterior.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, 1])
    return fig


def addon_sweep_plot(curves, style, suffix):
    metric, ylabel = METRICS[suffix]
    fig, axes = plt.subplots(1, 4, figsize=style['figure_size_addon_sweep'], sharey=True)
    for ax, M in zip(axes, [2, 4, 8, 16]):
        for method in METHODS if M == 2 else CONTROLS:
            draw_series(ax, curves[(f'm{M:02d}_b04', method, metric)], style['colors'][method],
                        style['labels'][method], style, 'addon_sweep')
        ax.set(title=f'M={M}, beta=4, d=4', xlabel='Answers already received, t',
               xlim=(0, 128 if metric == 'risk' else 127))
        ax.grid(alpha=.15)
    axes[0].set_ylabel(ylabel)
    fig.legend(handles=[Line2D([], [], color=style['colors'][m], lw=2, label=style['labels'][m]) for m in METHODS],
               loc='lower center', ncol=7, bbox_to_anchor=(.5, .07), frameon=False)
    fig.text(.5, .015, 'Only M=2 adds new methods. All old curves are retained; M>2 is not rerun. Same alpha=0.5, radius=16; achieved numerical reference, not certified optimum.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .20, 1, 1])
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true', help='List the 24 available figure names')
    p.add_argument('--only', action='append', help='Filename without extension or a glob; may be repeated')
    p.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['pdf', 'svg'])
    p.add_argument('--style', type=Path, default=ROOT / 'style.json')
    p.add_argument('--output', type=Path, default=ROOT / 'edited_figures')
    p.add_argument('--include-sg', action='store_true', help='Also show SG in original sweeps, matching the with_sg archive')
    args = p.parse_args()
    specs = figure_specs()
    if args.list:
        print('\n'.join(specs)); return
    selected = {k: v for k, v in specs.items() if not args.only or any(fnmatch.fnmatch(k, pattern) for pattern in args.only)}
    if not selected: p.error('--only did not match any figure name; use --list')
    style = json.loads(args.style.read_text())
    plt.rcParams.update({'font.size': style['font_size'], 'font.family': style['font_family'],
                         'svg.fonttype': style['svg_fonttype'], 'pdf.fonttype': style['pdf_fonttype']})
    curves = load_curves(ROOT / 'data/curves.csv.gz')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (kind, group, suffix) in selected.items():
        if kind == 'original': fig = original_plot(curves, style, group, suffix, args.include_sg)
        elif kind == 'addon': fig = addon_plot(curves, style, group, suffix)
        elif kind == 'bounds': fig = bounds_plot(curves, style)
        else: fig = addon_sweep_plot(curves, style, suffix)
        for ax in fig.axes:
            ax.set(**style.get('axis_limits', {}).get(name, {}))
        for ext in args.formats:
            fig.savefig(args.output / f'{name}.{ext}', dpi=style['dpi'], bbox_inches='tight')
        plt.close(fig)
        print(name)


if __name__ == '__main__':
    main()
