import sys
sys.path.append("./src/")
from eval import compute_isotropic_spectrum
import numpy as np
import matplotlib.pyplot as plt
import global_config
from utils import get_timesteps_available
from tqdm import tqdm
import matplotlib.patches as patches

middle_prefix = 'channel_middle'
nearwall_prefix = 'channel_nearwall'
full_prefix = 'channel_full'
full_rich_prefix = 'channel_full_rich'
# Set global font to Times New Roman for all text and math
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9  # Base font size
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
from matplotlib.lines import Line2D

timesteps = get_timesteps_available(global_config.numpy_dir, middle_prefix)

for time in tqdm(timesteps):
    middle_data = np.load(f'{global_config.numpy_dir}/{middle_prefix}_box0_time{time}.npy')
    nearwall_data = np.load(f'{global_config.numpy_dir}/{nearwall_prefix}_box0_time{time}.npy')
    full_data = np.load(f'{global_config.numpy_dir}/{full_prefix}_box0_time{time}.npy')
    if time == '0':
        k, middle_spectrum_u = compute_isotropic_spectrum(middle_data[0,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        k, middle_spectrum_v = compute_isotropic_spectrum(middle_data[1,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        k, middle_spectrum_w = compute_isotropic_spectrum(middle_data[2,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        k, nearwall_spectrum_u = compute_isotropic_spectrum(nearwall_data[0,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        k, nearwall_spectrum_v = compute_isotropic_spectrum(nearwall_data[1,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        k, nearwall_spectrum_w = compute_isotropic_spectrum(nearwall_data[2,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
    else:
        pass
        _,middle_spectrum_i_u, = compute_isotropic_spectrum(middle_data[0,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        _,middle_spectrum_i_v, = compute_isotropic_spectrum(middle_data[1,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        _,middle_spectrum_i_w, = compute_isotropic_spectrum(middle_data[2,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        _,nearwall_spectrum_i_u, = compute_isotropic_spectrum(nearwall_data[0,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False)
        _,nearwall_spectrum_i_v, = compute_isotropic_spectrum(nearwall_data[1,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False )
        _,nearwall_spectrum_i_w, = compute_isotropic_spectrum(nearwall_data[2,:,:,:][np.newaxis, :], Lx=0.6, Ly=0.6, Lz=0.6,tke_normalize=False,spectral_dealias=False )
        middle_spectrum_u += middle_spectrum_i_u
        middle_spectrum_v += middle_spectrum_i_v
        middle_spectrum_w += middle_spectrum_i_w
        nearwall_spectrum_u += nearwall_spectrum_i_u
        nearwall_spectrum_v += nearwall_spectrum_i_v
        nearwall_spectrum_w += nearwall_spectrum_i_w

full_data_rich = np.load(f'{global_config.numpy_dir}/{full_rich_prefix}_box0_time0.npy')
k_full_rich, full_spectrum_rich = compute_isotropic_spectrum(full_data_rich, Lx=2, Ly=2, Lz=2,tke_normalize=False,spectral_dealias=False)
# Average the raw spectra
coarse_data_example = np.load(f'{global_config.numpy_dir}/{middle_prefix}_filtered_fs4_box0_time0.npy')
k_coarse, coarse_spectrum = compute_isotropic_spectrum(coarse_data_example, Lx = 0.6, Ly = 0.6, Lz = 0.6, tke_normalize=False, spectral_dealias=False)



middle_spectrum_u /= len(timesteps)
middle_spectrum_v /= len(timesteps)
middle_spectrum_w /= len(timesteps)
nearwall_spectrum_u /= len(timesteps)
nearwall_spectrum_v /= len(timesteps)
nearwall_spectrum_w /= len(timesteps)

# Now normalize each averaged spectrum
#dk = np.diff(k)
#dk_full = np.diff(k_full)
#middle_tke = np.sum(middle_spectrum[:-1] * dk)
#nearwall_tke = np.sum(nearwall_spectrum[:-1] * dk)
#full_tke = np.sum(full_spectrum[:-1] * dk_full)

#middle_spectrum# /= middle_tke
#nearwall_spectrum# /= nearwall_tke
#full_spectrum# /= full_tke
# Verify that the normalized spectra integrate to 1
#middle_integral = np.sum(middle_spectrum[:-1] * dk)
#nearwall_integral = np.sum(nearwall_spectrum[:-1] * dk)
#full_integral = np.sum(full_spectrum[:-1] * dk)


fig,ax = plt.subplots(1,1,figsize=(3,3))

ax.loglog(k, middle_spectrum_u, label='Box B, $u$', linewidth=1, color = 'maroon')
ax.loglog(k, nearwall_spectrum_u, label='Box A, $u$', linewidth=1, color = 'darkblue')
ax.loglog(k_full_rich, full_spectrum_rich, label='Full cross-section', linewidth=2, color='black')

ax.loglog(k, middle_spectrum_v, label='Box B, $v$', linewidth=1, color = 'rosybrown')
ax.loglog(k, nearwall_spectrum_v, label='Box A, $v$', linewidth=1, color = 'royalblue')

ax.loglog(k, middle_spectrum_w, label='Box B, $w$', linewidth=1, color = 'lightcoral')
ax.loglog(k, nearwall_spectrum_w, label='Box A, $w$', linewidth=1, color = 'skyblue')

#ax.loglog(k_coarse, coarse_spectrum, label='Coarse', linewidth=1, color = 'green')

ax.annotate(r'$k^{-5/3}$', xy=(1E2, 5E-3), xytext=(3.8E1, 3E-4), color='black', fontsize=12)

ax.axvspan(min(k), 90, color='gray', lw=0, alpha=0.05,zorder=-100)

ax.axvspan(90, 1E5, color='black', lw=0, alpha=0.1,zorder=-100)

#ax.loglog(k, middle_spectrum, label='middle')
#ax.loglog(k, nearwall_spectrum, label='nearwall')
# Add -5/3 slope line
k_range = np.array([k[0], k[-1]])  # Get range of k values
shift_factor = 1.0  # Adjust this constant to shift the line up/down in log space
ax.loglog([2.5E1, 6E1], [2.5E1**(-5/3) * 10**(-shift_factor), 6E1**(-5/3) * 10**(-shift_factor)], linewidth=1, color='black')
ax.set_ylim([1E-8, 1E-2])
ax.set_xlim([k_full_rich[0], 4E2])
ax.set_xlabel(r'$k$ [1/m]')
ax.set_ylabel(r'$E_i(k)$ [m$^2$/s$^2$]')

# Create table legend with colored line symbols
line_colors = [['maroon', 'rosybrown', 'lightcoral'],      # Box A row colors
               ['darkblue', 'royalblue', 'skyblue']]       # Box B row colors

# Create the table with line symbols as text
table_data = [['—', '—', '—'],
              ['—', '—', '—']]

rect = patches.Rectangle((5.5, 1E-8), 30, 1.5E-6, linewidth=1, edgecolor='black', facecolor='none', alpha=1.0,zorder=-80)
ax.add_patch(rect)

table = ax.table(cellText=table_data,
                rowLabels=['Mid-plane', 'Boundary'],
                colLabels=['$u$', '$v$', '$w$'],
                loc='upper right',
                cellLoc='center',
                bbox=[0.2, 0.1, 0.25, 0.25],zorder=-98)  # [x, y, width, height]

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
# Style the cells
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('none')
    cell.set_linewidth(0)
    if i == 0:  # Header row
        cell.set_text_props(weight='normal', color='black')
        cell.set_facecolor('white')
    elif j == -1:  # Row labels
        cell.set_text_props(weight='normal', color='black')
        cell.set_facecolor('white')
    else:  # Data cells
        cell.set_facecolor('white')
        if i > 0 and j >= 0:  # Only data cells
            cell.set_text_props(color=line_colors[i-1][j], weight='bold')

# Add separate legend entry for full cross-section
full_legend = ax.legend([Line2D([0], [0], color='black', linewidth=2)], 
                       [r'Full channel $E(k)$'], 
                       loc='center left', 
                       bbox_to_anchor=(.078, 0.05),
                       fontsize=12,
                       frameon=False)

ax.annotate('Input scales', xy=(10E1, 1E-3), xytext=(2.7E1, 2E-3), color='black', fontsize=12)
ax.annotate('Superresolved scales', xy=(10E1, 1E-3), xytext=(10E1, 2E-3), color='black', fontsize=12)

fig.tight_layout()

plt.savefig('channel_spectra.pdf')






