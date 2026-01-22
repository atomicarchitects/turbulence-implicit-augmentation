import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_example_SR import visualize_example_predictions, visualize_equivariance, visualize_example_fields
import torch
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import matplotlib.image as mpimg
import ast


torch.set_default_dtype(torch.float32)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9  # Base font size
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
linewidth = 0.5
major_gridlines = 0.5
minor_gridlines = 0.5  
plt.rcParams['axes.linewidth'] = linewidth
plt.rcParams['xtick.major.width'] = linewidth
plt.rcParams['ytick.major.width'] = linewidth

# All user params are here.
FINAL_RESULTS_DIR_1BOX = '/home/rmcconke/implicit_data_augmentation/outputs/dec30_random'
FINAL_RESULTS_DIR_3BOX = '/home/rmcconke/implicit_data_augmentation/outputs/jan1_3box'
PLOT_OUTPUT_DIR = '/home/rmcconke/implicit_data_augmentation/outputs/plots_final'
cmap_magnitude = 'plasma'
vmin_magnitude = 0.8
vmax_magnitude = 1.2
angle_equivariance = torch.tensor([0,0,np.pi/2])
c_nw = 'b'
c_md = 'r'
scatter_size = 3
cdot_scale = 10 # for scatter with dot
marker_1box = 'o'
marker_3box = 's'
# For equivariance plots
vmin_uz = -.1
vmax_uz =.1
vmin_diff = 0
vmax_diff_nw =.03
vmax_diff_md =.03
label_fontsize = 7
cmap_vz = 'PuOr'
cmap_diff = 'viridis'

def scatter_with_dot(ax, x, y, marker, color, **kwargs):
    ax.scatter(x, y, marker = marker, facecolors='none', edgecolors=color, s=scatter_size, linewidths=linewidth, **kwargs)
    ax.scatter(x, y, marker = '+',facecolors=color, s=scatter_size*cdot_scale, linewidths=linewidth, label='_nolegend_')

def detect_nested_dict_columns(df):
    nested_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Only check object columns
            # Check first non-null value
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if sample is not None and isinstance(sample, str):
                # Try to parse it as a dict
                try:
                    parsed = ast.literal_eval(sample)
                    if isinstance(parsed, dict):
                        nested_cols.append(col)
                except (ValueError, SyntaxError):
                    pass
    return nested_cols

def expand_nested_dicts(df, prefix_separator='_',):
    df = df.copy()
    nested_cols = detect_nested_dict_columns(df)    
    for col in nested_cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else x)
        expanded = df[col].apply(pd.Series)
        expanded.columns = [f'{col}{prefix_separator}{subcol}' for subcol in expanded.columns]
        df = pd.concat([df, expanded], axis=1)
        df = df.drop(columns=[col])
    return df

def add_table_legend(fig, x_inches=None, y_inches=None, width_inches=2, height_inches=1):
    """
    Add a table-style legend with fixed size in inches at a specific position.
    
    Parameters:
        fig: matplotlib figure to add legend to
        x_inches: x position in inches from bottom-left of figure (if None, centers horizontally)
        y_inches: y position in inches from bottom-left of figure (if None, centers vertically)
        width_inches: width in inches
        height_inches: height in inches
    """
    from matplotlib.patches import Rectangle
    
    # Get figure size
    fig_width, fig_height = fig.get_size_inches()
    
    # Default to center if not specified
    if x_inches is None:
        x_inches = (fig_width - width_inches) / 2
    if y_inches is None:
        y_inches = (fig_height - height_inches) / 2
    
    # Convert inches to figure fraction
    x_frac = x_inches / fig_width
    y_frac = y_inches / fig_height
    width_frac = width_inches / fig_width
    height_frac = height_inches / fig_height
    
    # Create axes with absolute size
    ax_leg = fig.add_axes([x_frac, y_frac, width_frac, height_frac])
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')
    
    # Layout parameters
    col_spacing = 0.07
    line_len = 0.02
    header_y = 0.85
    row1_y = 0.42
    row2_y = 0.15
    padding_y = 0.12
    center_x = 0.5
    
    label_x = center_x - 0.08
    col_start = center_x - 0.01
    label_ha = 'center'
    box_left_offset = 0.05
    
    marker_x = [col_start + i * col_spacing for i in range(4)]
    box_right = min(0.99, marker_x[-1] + line_len + 0.02)
    box_left = label_x - box_left_offset
    box_bottom = max(0.01, row2_y - padding_y)
    box_top = min(0.99, header_y + padding_y)

    # Draw box around legend
    box = Rectangle((box_left, box_bottom), box_right - box_left, box_top - box_bottom, 
                    transform=ax_leg.transAxes,
                    fill=True, facecolor='white', edgecolor='black', linewidth=0.5, zorder=0)
    ax_leg.add_patch(box)

    # Column headers
    headers = ['1 box\nno aug.', '3 box\nno aug.', '1 box\naug.', '3 box\naug.']
    for xpos, header in zip(marker_x, headers):
        ax_leg.text(xpos, header_y, header, fontsize=6, ha='center', va='top', transform=ax_leg.transAxes)

    # Row labels
    ax_leg.text(label_x, row1_y, 'Near-wall', fontsize=6, ha=label_ha, va='center', transform=ax_leg.transAxes)
    ax_leg.text(label_x, row2_y, 'Middle', fontsize=6, ha=label_ha, va='center', transform=ax_leg.transAxes)

    # Define the marker/line styles for each column: (linestyle, filled)
    styles = [('--', False), ('--', False), ('-', True), ('-', True)]
    markers = ['o', 's', 'o', 's']  # circles for 1box, squares for 3box
    c_nw = 'blue'
    c_md = 'red'
    linewidth = 0.5
    scatter_size = 3

    # Nearwall row
    for i, (ls, filled) in enumerate(styles):
        ax_leg.plot([marker_x[i]-line_len, marker_x[i]+line_len], [row1_y, row1_y], 
                    color=c_nw, linestyle=ls, linewidth=linewidth, transform=ax_leg.transAxes, zorder=2)
        ax_leg.plot(marker_x[i], row1_y, marker=markers[i], color=c_nw, 
                    markerfacecolor=c_nw if filled else 'none',
                    markersize=scatter_size, markeredgewidth=linewidth, 
                    transform=ax_leg.transAxes, linestyle='none', zorder=3)

    # Middle row
    for i, (ls, filled) in enumerate(styles):
        ax_leg.plot([marker_x[i]-line_len, marker_x[i]+line_len], [row2_y, row2_y], 
                    color=c_md, linestyle=ls, linewidth=linewidth, transform=ax_leg.transAxes, zorder=2)
        ax_leg.plot(marker_x[i], row2_y, marker=markers[i], color=c_md, 
                    markerfacecolor=c_md if filled else 'none',
                    markersize=scatter_size, markeredgewidth=linewidth, 
                    transform=ax_leg.transAxes, linestyle='none', zorder=3)
    
    return ax_leg

import ast

def extract_tuple_element(df, column_name, index=0, new_column_name=None):
    """
    Extract a specific element from tuples in a DataFrame column.
    Handles both actual tuples and string representations of tuples.
    """
    if new_column_name is None:
        new_column_name = column_name
    
    def safe_extract(x):
        # If it's a string, convert it to a tuple first
        if isinstance(x, str):
            x = ast.literal_eval(x)
        # Now extract the element
        if isinstance(x, tuple) and len(x) > index:
            return x[index]
        return None
    
    df[new_column_name] = df[column_name].apply(safe_extract)
    
    return df

def compute_wavenumber_bins(N, L):
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    dk = abs(k[1] - k[0])
    k_cut = (2/3) * np.max(np.abs(k))
    k_max = np.sqrt(3) * np.max(np.abs(k))
    edges = np.arange(0.0, k_max + dk, dk)
    centers = 0.5*(edges[1:] + edges[:-1])
    return centers[(centers > 0) & (centers <= k_cut)][1:]


results_1box = pd.read_csv(os.path.join(FINAL_RESULTS_DIR_1BOX, 'final_results.csv'))
results_1box = expand_nested_dicts(results_1box)
results_1box['dataset_train_samples'] = results_1box['dataset_train_samples'].replace('all', 2680).astype(int)

results_nw_1box_aug = results_1box[(results_1box['train_aug_group'] == 'oct') & results_1box['dataset_input_prefix'].str.contains('nearwall')]
results_nw_1box_noaug = results_1box[(results_1box['train_aug_group'] != 'oct') & results_1box['dataset_input_prefix'].str.contains('nearwall')]
results_md_1box_aug = results_1box[(results_1box['train_aug_group'] == 'oct') & results_1box['dataset_input_prefix'].str.contains('middle')]
results_md_1box_noaug = results_1box[(results_1box['train_aug_group'] != 'oct') & results_1box['dataset_input_prefix'].str.contains('middle')]

results_3box = pd.read_csv(os.path.join(FINAL_RESULTS_DIR_3BOX, 'final_results.csv'))
results_3box['dataset_train_samples'] = results_3box['dataset_train_samples'] * 3 # Since there are 3 boxes (config samples is per box)
results_3box = expand_nested_dicts(results_3box)

results_nw_3box_aug = results_3box[(results_3box['train_aug_group'] == 'oct') & results_3box['dataset_input_prefix'].str.contains('nearwall')]
results_nw_3box_noaug = results_3box[(results_3box['train_aug_group'] != 'oct') & results_3box['dataset_input_prefix'].str.contains('nearwall')]
results_md_3box_aug = results_3box[(results_3box['train_aug_group'] == 'oct') & results_3box['dataset_input_prefix'].str.contains('middle')]
results_md_3box_noaug = results_3box[(results_3box['train_aug_group'] != 'oct') & results_3box['dataset_input_prefix'].str.contains('middle')]

nw_1box_noaug = results_nw_1box_noaug.sort_values('dataset_train_samples')
nw_1box_aug = results_nw_1box_aug.sort_values('dataset_train_samples')
nw_3box_noaug = results_nw_3box_noaug.sort_values('dataset_train_samples')
nw_3box_aug = results_nw_3box_aug.sort_values('dataset_train_samples')
md_1box_noaug = results_md_1box_noaug.sort_values('dataset_train_samples')
md_1box_aug = results_md_1box_aug.sort_values('dataset_train_samples')
md_3box_noaug = results_md_3box_noaug.sort_values('dataset_train_samples')
md_3box_aug = results_md_3box_aug.sort_values('dataset_train_samples')

# There are 8 models total: 
# middle/nearwall x 1box/3box x noaug/aug 

fig, ax = plt.subplots(figsize=(4, 3))
add_table_legend(fig, x_inches=None, y_inches=None, height_inches=.55, width_inches=4.8)
plt.savefig(os.path.join(PLOT_OUTPUT_DIR, 'legend.pdf'), bbox_inches='tight', pad_inches=0.1)
plt.close()
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(4, 3))
gs = GridSpec(1, 1, figure=fig, wspace=.1,left=0.1, right=0.99, top=0.77, bottom=0.22)
ax = [fig.add_subplot(gs[0, i]) for i in range(1)]

x_min = 0.8
x_max = 3000
y_min = 8E-4
y_max = 4E-2

for axi in ax:
    axi.grid(True, which='minor', color='lightgray', linewidth=minor_gridlines, zorder=1)
    axi.grid(True, which='major', color='lightgray', linewidth=major_gridlines, zorder=2)

    axi.set_xscale('log')
    axi.set_yscale('log')
    axi.set_xlim([x_min, x_max])
    axi.set_ylim([y_min, y_max])
    axi.set_axisbelow(True)
    
    # Set major locators
    axi.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    axi.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    
    # Set minor locators for log scale
    axi.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
    axi.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
    
    # Turn on minor ticks
    axi.minorticks_on()

# Nearwall
ax[0].plot(nw_1box_noaug['dataset_train_samples'], nw_1box_noaug['abs_equiv_error'], 
        marker=marker_1box, color=c_nw, linestyle='--', markerfacecolor='none', 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax[0].plot(nw_1box_aug['dataset_train_samples'], nw_1box_aug['abs_equiv_error'], 
        marker=marker_1box, color=c_nw, linestyle='-', markerfacecolor=c_nw, 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax[0].plot(nw_3box_noaug['dataset_train_samples'], nw_3box_noaug['abs_equiv_error'], 
        marker=marker_3box, color=c_nw, linestyle='--', markerfacecolor='none', 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax[0].plot(nw_3box_aug['dataset_train_samples'], nw_3box_aug['abs_equiv_error'], 
        marker=marker_3box, color=c_nw, linestyle='-', markerfacecolor=c_nw, 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)

# Middle
ax[0].plot(md_1box_noaug['dataset_train_samples'], md_1box_noaug['abs_equiv_error'], 
        marker=marker_1box, color=c_md, linestyle='--', markerfacecolor='none', 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax[0].plot(md_1box_aug['dataset_train_samples'], md_1box_aug['abs_equiv_error'], 
        marker=marker_1box, color=c_md, linestyle='-', markerfacecolor=c_md, 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax[0].plot(md_3box_noaug['dataset_train_samples'], md_3box_noaug['abs_equiv_error'], 
        marker=marker_3box, color=c_md, linestyle='--', markerfacecolor='none', 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax[0].plot(md_3box_aug['dataset_train_samples'], md_3box_aug['abs_equiv_error'], 
        marker=marker_3box, color=c_md, linestyle='-', markerfacecolor=c_md, 
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)

ax[0].set_ylabel('Test equivariance error $\|\epsilon\|_\mathrm{test}$')
ax[0].set_xlabel('Number of training points, $N_\mathrm{train}$')

#gs.tight_layout(fig, pad=0) 
#add_table_legend(fig, x_inches=2.12, y_inches=None, height_inches=.55, width_inches=4.1)
plt.savefig(os.path.join(PLOT_OUTPUT_DIR, 'equivariance_error_vs_training_samples.pdf'), bbox_inches='tight', pad_inches=0.1)

plt.close()
##
### Test loss vs equivariance error (3 plots)
##
from matplotlib.ticker import FixedLocator, FixedFormatter

fig = plt.figure(figsize=(3, 2))
gs = GridSpec(
    nrows=1, ncols=1,
)

# Legend axis (top, spanning both columns)


# Two identical plots
ax1 = fig.add_subplot(gs[0])

c_md_newsample = 'red'
c_md_newy = 'salmon'
c_md_higherRe = 'darkred'


c_nw_newsample = 'blue'
c_nw_newy = 'lightskyblue'
c_nw_higherRe = 'midnightblue'

# new sample generalization
nw_1box_aug = extract_tuple_element(nw_1box_aug, 'additional_abs_equiv_errors_Re5200_nearwall', index=0, new_column_name='additional_abs_equiv_errors_Re5200_nearwall_0')
md_1box_aug = extract_tuple_element(md_1box_aug, 'additional_abs_equiv_errors_Re5200_middle', index=0, new_column_name='additional_abs_equiv_errors_Re5200_middle_0')
nw_1box_aug = extract_tuple_element(nw_1box_aug, 'additional_abs_equiv_errors_middle', index=0, new_column_name='additional_abs_equiv_errors_middle_0')
md_1box_aug = extract_tuple_element(md_1box_aug, 'additional_abs_equiv_errors_nearwall', index=0, new_column_name='additional_abs_equiv_errors_nearwall_0')

ax1.plot(nw_1box_aug['abs_equiv_error'], nw_1box_aug['test_loss'],
        marker=marker_1box, color=c_nw_newsample, linestyle='-', markerfacecolor=c_nw_newsample,
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)

ax1.plot(md_1box_aug['abs_equiv_error'], md_1box_aug['test_loss'],
        marker=marker_1box, color=c_md_newsample, linestyle='-', markerfacecolor=c_md_newsample,
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)


# new y generalization
ax1.plot(nw_1box_aug['additional_abs_equiv_errors_middle_0'], nw_1box_aug['test_loss_middle'],
        marker=marker_1box, color=c_nw_newy, linestyle='-', markerfacecolor=c_nw_newy,
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)

ax1.plot(md_1box_aug['additional_abs_equiv_errors_nearwall_0'], md_1box_aug['test_loss_nearwall'],
        marker=marker_1box, color=c_md_newy, linestyle='-', markerfacecolor=c_md_newy,
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)


# higher Re generalization
ax1.plot(nw_1box_aug['additional_abs_equiv_errors_Re5200_nearwall_0'], nw_1box_aug['additional_test_losses_Re5200_nearwall'],
        marker=marker_1box,
        color=c_nw_higherRe, linestyle='-', markerfacecolor=c_nw_higherRe,
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)

ax1.plot(md_1box_aug['additional_abs_equiv_errors_Re5200_middle_0'], md_1box_aug['additional_test_losses_Re5200_middle'],
        marker=marker_1box,
        color=c_md_higherRe, linestyle='-', markerfacecolor=c_md_higherRe,
        markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)




ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks([1e-1])
#ax.set_yticks([])
ax1.set_xlim([1E-3,2E-2])
ax1.grid(True, which='minor', color='lightgray', linewidth=minor_gridlines, zorder=1)
ax1.grid(True, which='major', color='lightgray', linewidth=major_gridlines, zorder=2)

ax1.set_xscale('log')
ax1.set_axisbelow(True)


# Set minor locators for log scale
ax1.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))

# Turn on minor ticks
ax1.minorticks_on()


# Axis labels
ax1.set_ylabel('Generalization test loss, $\mathcal{L}_\mathrm{test}$')
ax1.set_xlabel(r'Equivariance error $\left\Vert\epsilon\right\Vert_\mathrm{test}$')

from matplotlib.patches import FancyArrowPatch

# Add arrow
arrow_x_start = 3.5e-3
arrow_y_start = 5e-3
arrow_x_end = 1.5e-3
arrow_y_end = 2.5e-3

arrow = FancyArrowPatch(
    (arrow_x_start, arrow_y_start),
    (arrow_x_end, arrow_y_end),
    arrowstyle='-|>',
    lw=1,  # Line width
    color='black',
    mutation_scale=7  # Controls arrow head size (reduce this to make smaller)
)
ax1.add_patch(arrow)

# Add text annotation
text_x = 2E-3
text_y = 3.8e-3
text_rotation = 35

ax1.text(
    text_x, text_y,
    'Increasing $N_\mathrm{train}$',
    rotation=text_rotation,
    fontsize=8,
    ha='center',
    va='center'
)

plt.savefig(
    os.path.join(PLOT_OUTPUT_DIR, 'test_loss_vs_equivariance_error.pdf'),
    bbox_inches='tight', 
    pad_inches=0
)
plt.close()


#### 
#### Spectra
####
nw_1box_noaug_spectra = nw_1box_noaug[nw_1box_noaug['dataset_train_samples'] == 1500]
nw_1box_noaug_spectra['k_spec_error'] = nw_1box_noaug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
nw_1box_noaug_spectra['spec_error'] = nw_1box_noaug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)

nw_1box_aug_spectra = nw_1box_aug[nw_1box_aug['dataset_train_samples'] == 1500]
nw_1box_aug_spectra['k_spec_error'] = nw_1box_aug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
nw_1box_aug_spectra['spec_error'] = nw_1box_aug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)

nw_3box_noaug_spectra = nw_3box_noaug[nw_3box_noaug['dataset_train_samples'] == 1500]
nw_3box_noaug_spectra['k_spec_error'] = nw_3box_noaug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
nw_3box_noaug_spectra['spec_error'] = nw_3box_noaug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)

nw_3box_aug_spectra = nw_3box_aug[nw_3box_aug['dataset_train_samples'] == 1500]
nw_3box_aug_spectra['k_spec_error'] = nw_3box_aug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
nw_3box_aug_spectra['spec_error'] = nw_3box_aug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)

md_1box_noaug_spectra = md_1box_noaug[md_1box_noaug['dataset_train_samples'] == 1500]
md_1box_noaug_spectra['k_spec_error'] = md_1box_noaug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
md_1box_noaug_spectra['spec_error'] = md_1box_noaug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)

md_1box_aug_spectra = md_1box_aug[md_1box_aug['dataset_train_samples'] == 1500]
md_1box_aug_spectra['k_spec_error'] = md_1box_aug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
md_1box_aug_spectra['spec_error'] = md_1box_aug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)

md_3box_noaug_spectra = md_3box_noaug[md_3box_noaug['dataset_train_samples'] == 1500]
md_3box_noaug_spectra['k_spec_error'] = md_3box_noaug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
md_3box_noaug_spectra['spec_error'] = md_3box_noaug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)

md_3box_aug_spectra = md_3box_aug[md_3box_aug['dataset_train_samples'] == 1500]
md_3box_aug_spectra['k_spec_error'] = md_3box_aug_spectra['k_spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
md_3box_aug_spectra['spec_error'] = md_3box_aug_spectra['spec_error'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
fig, ax = plt.subplots(1,1,figsize=(4,1.5))
k = compute_wavenumber_bins(64, 0.6)


ax.plot(k, nw_1box_noaug_spectra['spec_error'].loc[20], color=c_nw, linestyle='--',
            markerfacecolor='none', marker=marker_1box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax.plot(k, nw_1box_aug_spectra['spec_error'].loc[27], color=c_nw, linestyle='-',
            markerfacecolor='none', marker=marker_1box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax.plot(k, nw_3box_noaug_spectra['spec_error'].loc[20], color=c_nw, linestyle='--',
            markerfacecolor='none', marker=marker_3box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax.plot(k, nw_3box_aug_spectra['spec_error'].loc[27], color=c_nw, linestyle='-',
            markerfacecolor='none', marker=marker_3box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)

ax.plot(k, md_1box_noaug_spectra['spec_error'].loc[6], color=c_md, linestyle='--',
            markerfacecolor='none', marker=marker_1box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax.plot(k, md_1box_aug_spectra['spec_error'].loc[13], color=c_md, linestyle='-',
            markerfacecolor='none', marker=marker_1box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax.plot(k, md_3box_noaug_spectra['spec_error'].loc[6], color=c_md, linestyle='--',
            markerfacecolor='none', marker=marker_3box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)
ax.plot(k, md_3box_aug_spectra['spec_error'].loc[13], color=c_md, linestyle='-',
            markerfacecolor='none', marker=marker_3box, markersize=scatter_size, linewidth=linewidth, markeredgewidth=linewidth)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Wavenumber, $k$ (1/m)')
ax.set_ylabel(r'$E_{\epsilon}(k)$ (m$^3$/s$^2$)')

#ax.set_yticks([])
ax.grid(True, which='minor', color='lightgray', linewidth=minor_gridlines, zorder=1)
ax.grid(True, which='major', color='lightgray', linewidth=major_gridlines, zorder=2)

# Set major locators
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))

# Set minor locators for log scale
ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))

# Turn on minor ticks
ax.minorticks_on()

plt.tight_layout(pad=0,rect=[0, 0, .6, 1])
add_table_legend(fig, x_inches=1, y_inches = None, height_inches = .55, width_inches = 4.1)

fig.savefig(os.path.join(PLOT_OUTPUT_DIR, 'equiv_error_spectra.pdf'),pad_inches=0)


plt.close()


image_paths = [os.path.join(PLOT_OUTPUT_DIR, 'anisotropy_channel_nearwall_box0.png'),
               os.path.join(PLOT_OUTPUT_DIR, 'anisotropy_channel_nearwall_box1.png'),
               os.path.join(PLOT_OUTPUT_DIR, 'anisotropy_channel_nearwall_box2.png'),
               os.path.join(PLOT_OUTPUT_DIR, 'anisotropy_channel_middle_box0.png'),
               os.path.join(PLOT_OUTPUT_DIR, 'anisotropy_channel_middle_box1.png'),
               os.path.join(PLOT_OUTPUT_DIR, 'anisotropy_channel_middle_box2.png')]
crop_top_percent = 7
vmin = 0
vmax = 0.816
cmap = 'magma'

fig = plt.figure(figsize=(6.2, 1))
gs = GridSpec(1, 7, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1, 0.15], hspace=0.0, wspace=0.0)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[0, 4])
ax5 = fig.add_subplot(gs[0, 5])

img0 = mpimg.imread(image_paths[0])
img1 = mpimg.imread(image_paths[1])
img2 = mpimg.imread(image_paths[2])
img3 = mpimg.imread(image_paths[3])
img4 = mpimg.imread(image_paths[4])
img5 = mpimg.imread(image_paths[5])

if crop_top_percent > 0:
    crop_pixels0 = int(img0.shape[0] * crop_top_percent / 100)
    crop_pixels1 = int(img1.shape[0] * crop_top_percent / 100)
    crop_pixels2 = int(img2.shape[0] * crop_top_percent / 100)
    crop_pixels3 = int(img3.shape[0] * crop_top_percent / 100)
    crop_pixels4 = int(img4.shape[0] * crop_top_percent / 100)
    crop_pixels5 = int(img5.shape[0] * crop_top_percent / 100)
    img0 = img0[crop_pixels0:, :, :]
    img1 = img1[crop_pixels1:, :, :]
    img2 = img2[crop_pixels2:, :, :]
    img3 = img3[crop_pixels3:, :, :]
    img4 = img4[crop_pixels4:, :, :]
    img5 = img5[crop_pixels5:, :, :]

im0 = ax0.imshow(img0, vmin=vmin, vmax=vmax, cmap=cmap)
im1 = ax1.imshow(img1, vmin=vmin, vmax=vmax, cmap=cmap)
im2 = ax2.imshow(img2, vmin=vmin, vmax=vmax, cmap=cmap)
im3 = ax3.imshow(img3, vmin=vmin, vmax=vmax, cmap=cmap)
im4 = ax4.imshow(img4, vmin=vmin, vmax=vmax, cmap=cmap)
im5 = ax5.imshow(img5, vmin=vmin, vmax=vmax, cmap=cmap)

ax0.set_xlabel(f'(a)', fontsize=12)
ax1.set_xlabel(f'(b)', fontsize=12)
ax2.set_xlabel(f'(c)', fontsize=12)
ax3.set_xlabel(f'(d)', fontsize=12)
ax4.set_xlabel(f'(e)', fontsize=12)
ax5.set_xlabel(f'(f)', fontsize=12)

for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

cbar_ax = fig.add_subplot(gs[0, 6])
cbar = fig.colorbar(im5, cax=cbar_ax)
cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
cbar.set_ticklabels([r'$0xyz$', r'$\sqrt{\frac{1}{6}}$', r'$\sqrt{\frac{2}{3}}$'])
cbar.set_label('$|| \ \mathbf{\mathsf{b}} \ ||$')

fig.savefig(os.path.join(PLOT_OUTPUT_DIR, 'anisotropy.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

if input("Replot example predictions? (needs some memory) [y/n]: ").lower() == 'y':
    visualize_example_fields(output_dir=PLOT_OUTPUT_DIR,
                             output_file='example_fields.pdf',
                             experiment_dir_nw='outputs/dec30_random/25_12_30_21:27:13_solid_hermit_nearwall_boxfilter_4x_sr',
                             experiment_dir_md='outputs/dec30_random/25_12_30_13:17:35_super_spider_middle_boxfilter_4x_sr',
                             device='cpu',
                             )
    visualize_example_predictions(output_dir=PLOT_OUTPUT_DIR,
                                  output_file='example_predictions.pdf',
                                  experiment_dir_nw='outputs/dec30_random/25_12_31_08:14:58_pet_pika_nearwall_boxfilter_4x_sr',
                                  experiment_dir_md='outputs/dec30_random/25_12_30_13:17:35_super_spider_middle_boxfilter_4x_sr',
                                  device='cpu',
                                  cmap_magnitude=cmap_magnitude,
                                  vmin_magnitude=vmin_magnitude,
                                  vmax_magnitude=vmax_magnitude,
                                  cmap_diff=cmap_diff,
                                  )

    # nearwall
    visualize_equivariance(output_dir=PLOT_OUTPUT_DIR,
                            output_file='example_equivariance_nearwall.pdf',
                            experiment_dir_noaug='outputs/dec30_random/25_12_30_21:27:13_solid_hermit_nearwall_boxfilter_4x_sr',
                            experiment_dir_aug='outputs/dec30_random/25_12_31_08:14:58_pet_pika_nearwall_boxfilter_4x_sr',
                            device='cpu',
                            vmin_uz=vmin_uz,
                            vmax_uz=vmax_uz,
                            vmin_diff=vmin_diff,
                            vmax_diff=vmax_diff_nw,
                            label_fontsize=label_fontsize,
                            cmap_vz=cmap_vz,
                            cmap_diff=cmap_diff,
                            cborder=c_nw,
                            angle_equivariance=angle_equivariance)
    
    # middle
    visualize_equivariance(output_dir=PLOT_OUTPUT_DIR,
                            output_file='example_equivariance_middle.pdf',
                            experiment_dir_noaug='outputs/dec30_random/25_12_30_09:21:54_polite_hog_middle_boxfilter_4x_sr',
                            experiment_dir_aug='outputs/dec30_random/25_12_30_13:17:35_super_spider_middle_boxfilter_4x_sr',
                            device='cpu',
                            vmin_uz=vmin_uz,
                            vmax_uz=vmax_uz,
                            vmin_diff=vmin_diff,
                            vmax_diff=vmax_diff_md,
                            label_fontsize=label_fontsize,
                            cmap_vz=cmap_vz,
                            cmap_diff=cmap_diff,
                            cborder=c_md,
                            angle_equivariance=angle_equivariance)