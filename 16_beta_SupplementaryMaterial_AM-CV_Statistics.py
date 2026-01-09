# %%
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch
import rasterio
from pathlib import Path
import os,glob,sys
import seaborn as sns
# %%
"""
Code to genrate the plot of the Annual Maxima and the Coefficient of Variation for the 
dissagregated/"big" spatial categories that goes in the supplementary material. 

Ths script is an extention of the script labelled '12_alpha_SpatialStats_AM', but since
it only holds analysisis fo the supplementary material it has a 'beta code' status.

Author: Nathalia Correa-Sánchez
"""
########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################
STORAGE        = Path("/mnt/smb").as_posix()
bd_in_ws       = STORAGE + "/Data/WS_CORDEXFPS/"
bd_out_fig     = "/home/nathalia/Outputs/Plots/WP3_development" # Por ahora en la maquina virtual de procesamiento
bd_out_am      = STORAGE + "/Outputs/AM_ws100m/"
bd_in_eth      = bd_in_ws + "ETH/wsa100m_crop/"
bd_in_cmcc     = bd_in_ws + "CMCC/wsa100m_crop/"
bd_in_cnrm     = bd_in_ws + "CNRM/wsa100m_crop/"
bd_out_tc      = STORAGE + "/Outputs/WP3_SamplingSeries_CPM/"
bd_in_rast     = STORAGE + "/Outputs/Climate_Provinces/Development_Rasters/FinalRasters_In-Out/"
bd_base_raster = STORAGE + "/Outputs/Climate_Provinces/Development_Rasters/ALP3_ETOPO2022_60sArc.tif"

# %%
########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

filas_eliminar    = [0]  # Primera  fila, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada 
columnas_eliminar = [0]  # Primera columna, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada

# %%
########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################
def load_ds(bd_in):
    """
    Function to load the CPMs dataset, by loading multiple related files. 
    INPUTS:
    - ds                : dataset loaded with xarray witht he model information. 
    OUTPUTS:
    - serie_ds : numpy 3D array with all the time series and where each pixel correspond to the lats 
                and lons positon given in the lists. The dimensions are : (time, lat, lon)
    """
    files    = sorted(glob.glob(f"{bd_in}*.nc"))
    ds_org   = xr.open_mfdataset(files, combine='nested', concat_dim='time',   parallel=True, chunks={'time': 1000})
    return ds_org

# Funciones para obtener etiquetas
def get_climate_label(climate_code):
    try:
        idx = int(climate_code) - 1
        if 0 <= idx < len(climate_labels):
            return climate_labels[idx]
        else:
            return f"C{climate_code}"
    except (ValueError, TypeError):
        return "Unknown"

def get_roughness_label(roughness_code):
    try:
        idx = int(roughness_code) - 1
        if 0 <= idx < len(roughness_labels):
            return roughness_labels[idx]
        else:
            return f"R{roughness_code}"
    except (ValueError, TypeError):
        return "Unknown"

def get_slope_label(slope_code):
    try:
        idx = int(slope_code) - 1
        if 0 <= idx < len(topography_labels):
            return topography_labels[idx]
        else:
            return f"T{slope_code}"
    except (ValueError, TypeError):
        return "Unknown"

def decode_category(cat):
    """
    Decode spatial category into climate, roughness, topography
    
    Parameters:
    -----------
    cat : int
        Category code
    
    Returns:
    --------
    climate, roughness, topography : int or None
    """
    if cat == 1:  # Water (special case)
        return None, 1, None
    else:
        # Encoding: climate*100 + roughness*10 + topography
        climate = cat // 100
        roughness = (cat % 100) // 10
        topography = cat % 10
        return climate, roughness, topography

def compute_annual_maxima_from_series(time_series, time_index):
    """
    Compute annual maxima from time series
    
    Parameters:
    -----------
    time_series : array
        Wind speed time series, shape (n_timesteps,)
    time_index : DatetimeIndex
        Time index for the series
    
    Returns:
    --------
    annual_max : array
        Annual maxima, shape (10,) for years 2000-2009
    """
    df         = pd.DataFrame(time_series, index=time_index, columns=['WS'])
    annual_max = df.groupby(df.index.year)['WS'].max().values
    return annual_max

def compute_cv_from_annual_maxima(time_series, time_index):
    """
    Compute CV of annual maxima (consistent with Figure 3)
    
    Returns:
    --------
    cv : float
        Coefficient of variation OF THE 10 ANNUAL MAXIMA
    """
    # Calcular annual maxima (10 valores)
    df         = pd.DataFrame(time_series, index=time_index, columns=['WS'])
    annual_max = df.groupby(df.index.year)['WS'].max().values  # 10 valores
    
    # CV de esos 10 annual maxima
    mean_am = np.nanmean(annual_max)
    std_am  = np.nanstd(annual_max)
    cv      = std_am / mean_am if mean_am > 0 else np.nan
    
    return cv


# %%
#########################################################################################
###---------------------DEFFINING TAGS FOR SPATIAL CATEGORIES-------------------------###
#########################################################################################

# Definir etiquetas descriptivas
climate_labels    = ['Ar', 'Tm', 'Co', 'Td']  # Para códigos 1, 2, 3, 4
roughness_labels  = [r"$R_1$:(water)", r"$R_2$", r"$R_3$", r"$R_4$", r"$R_5$"]  # Para códigos 1-5
topography_labels = [r"$T_1$", r"$T_2$", r"$T_3$", r"$T_4$"]  # Para códigos 1, 2, 3, 4

#########################################################################################
###----------------LOADING AND PROCESSING SPATIAL CATEGORIES--------------------------###
#########################################################################################

print("Loading spatial categories from raster...")

comblay              = rasterio.open(bd_in_rast+"SEA-LANDCropped_Combined_RIX_remCPM_WGS84.tif")
band1_o              = comblay.read(1) ## Solo tiene una banda
band1_o[band1_o < 0] = np.nan          ## Reemplazando los negativos con nan o 0(Tener en cuenta NoData= -3.40282e+38) 
# Ajustillo para que coincida los xarrays (incluso despues del crop)
band1 = np.delete(np.delete(band1_o, filas_eliminar[0], axis=0), columnas_eliminar[0], axis=1) 

# Obteniendo valores unicos de las categorias
unique_vals    = np.unique(band1)
unique_vals    = unique_vals[np.isfinite(unique_vals)]
num_categories = len(unique_vals)

# Contar píxeles para cada valor único
pixel_counts = {}
mask         = np.isfinite(band1)
valid_values = band1[mask]

# Método más directo y confiable
for val in unique_vals:
    pixel_counts[val] = np.sum(valid_values == val)

counts_array = np.array([pixel_counts[val] for val in unique_vals])

df_cats = pd.DataFrame({'value': unique_vals, 'count': counts_array,})

########################################################################################
##--------FILTERING CATEGORIES UNDER THE 25% PERCENTILE: LOW ATYPICAL VALUES----------##
########################################################################################

total_count         = df_cats["count"].sum()
df_cats["rel_freq"] = (df_cats["count"] / total_count) * 100  

percentages = df_cats.rel_freq.values * 100
p25         = np.percentile(percentages, 25)
df_filt     = df_cats[df_cats.rel_freq >= p25]
df_filt     = df_filt.reset_index(drop=True) ## Resetea el indice que se habia dñado luego del filtrado. 
fl_cats     = df_filt['value'].values.astype(int)

# %%
########################################################################################
##--CALCULATING THE SUPPLMENTARY METRICS: ANNUAL MAXIMA AND COEFFICIENT OF VARIATION--##
########################################################################################

print("\n" + "="*80)
print("COMPUTING ANNUAL MAXIMA AND CV FROM SAMPLED TIME SERIES")
print("="*80)

# Crear rango temporal completo
full_range = pd.date_range(start='2000-01-01 00:00:00', end='2009-12-31 23:00:00', freq='1H')

# Preparar diccionarios para almacenar datos
data_annual_max = {
    'climate'   : {label: {'ETH': [], 'CNRM': [], 'CMCC': []} for label in climate_labels},
    'roughness' : {label: {'ETH': [], 'CNRM': [], 'CMCC': []} for label in roughness_labels},
    'topography': {label: {'ETH': [], 'CNRM': [], 'CMCC': []} for label in topography_labels}
}

data_cv = {
    'climate'   : {label: {'ETH': [], 'CNRM': [], 'CMCC': []} for label in climate_labels},
    'roughness' : {label: {'ETH': [], 'CNRM': [], 'CMCC': []} for label in roughness_labels},
    'topography': {label: {'ETH': [], 'CNRM': [], 'CMCC': []} for label in topography_labels}
}

# %%
########################################################################################
##--------------------------EXTRACTING DATA PER SPATIAL CATEGORY----------------------##
########################################################################################

n_categories_processed = 0
n_points_total         = 0

for cat in fl_cats:
    print(f"\n## Processing category {cat}")
    
    # Load time series
    try:
        data_npz    = np.load(f"{bd_out_tc}_TS_cat_{cat}.npz")
        time_series = data_npz['time_series']  # Shape: (n_points, n_models, n_timesteps)
        coordinates = data_npz['coordinates']
        models      = data_npz['models']
    except FileNotFoundError:
        print(f"   Warning: File not found for category {cat}")
        continue
    
    n_points = time_series.shape[0]
    n_points_total += n_points
    print(f"   Loaded {n_points} points")
    
    # Decode category
    climate_cat, roughness_cat, topography_cat = decode_category(cat)
    
    # Get labels
    climate_label    = get_climate_label(climate_cat) if climate_cat is not None else None
    roughness_label  = get_roughness_label(roughness_cat)
    topography_label = get_slope_label(topography_cat) if topography_cat is not None else None
    
    print(f"   Category: Climate={climate_label}, Roughness={roughness_label}, Topo={topography_label}")
    
    # Process each point
    for k in range(n_points):
        # Extract series for each model
        serie_eth  = time_series[k, 0, :]   # Model 0: ETH
        serie_cnrm = time_series[k, 1, :]  # Model 1: CNRM
        serie_cmcc = time_series[k, 2, :]  # Model 2: CMCC
        
        # Skip if too many NaNs
        if np.sum(np.isnan(serie_eth)) > len(serie_eth) * 0.5:
            continue
        if np.sum(np.isnan(serie_cnrm)) > len(serie_cnrm) * 0.5:
            continue
        if np.sum(np.isnan(serie_cmcc)) > len(serie_cmcc) * 0.5:
            continue
        
        # Compute annual maxima (10 values per series)
        am_eth  = compute_annual_maxima_from_series(serie_eth, full_range)
        am_cnrm = compute_annual_maxima_from_series(serie_cnrm, full_range)
        am_cmcc = compute_annual_maxima_from_series(serie_cmcc, full_range)
        
        # Compute CV (1 value per series)
        cv_eth  = compute_cv_from_annual_maxima(serie_eth, full_range)
        cv_cnrm = compute_cv_from_annual_maxima(serie_cnrm, full_range)
        cv_cmcc = compute_cv_from_annual_maxima(serie_cmcc, full_range)
        
       
        # Add to dictionaries
        
        # Climate
        if climate_label is not None and climate_label in data_annual_max['climate']:
            data_annual_max['climate'][climate_label]['ETH'].extend(am_eth.tolist())
            data_annual_max['climate'][climate_label]['CNRM'].extend(am_cnrm.tolist())
            data_annual_max['climate'][climate_label]['CMCC'].extend(am_cmcc.tolist())
            
            data_cv['climate'][climate_label]['ETH'].append(cv_eth)
            data_cv['climate'][climate_label]['CNRM'].append(cv_cnrm)
            data_cv['climate'][climate_label]['CMCC'].append(cv_cmcc)
        
        # Roughness
        if roughness_label in data_annual_max['roughness']:
            data_annual_max['roughness'][roughness_label]['ETH'].extend(am_eth.tolist())
            data_annual_max['roughness'][roughness_label]['CNRM'].extend(am_cnrm.tolist())
            data_annual_max['roughness'][roughness_label]['CMCC'].extend(am_cmcc.tolist())
            
            data_cv['roughness'][roughness_label]['ETH'].append(cv_eth)
            data_cv['roughness'][roughness_label]['CNRM'].append(cv_cnrm)
            data_cv['roughness'][roughness_label]['CMCC'].append(cv_cmcc)
        
        # Topography
        if topography_label is not None and topography_label in data_annual_max['topography']:
            data_annual_max['topography'][topography_label]['ETH'].extend(am_eth.tolist())
            data_annual_max['topography'][topography_label]['CNRM'].extend(am_cnrm.tolist())
            data_annual_max['topography'][topography_label]['CMCC'].extend(am_cmcc.tolist())
            
            data_cv['topography'][topography_label]['ETH'].append(cv_eth)
            data_cv['topography'][topography_label]['CNRM'].append(cv_cnrm)
            data_cv['topography'][topography_label]['CMCC'].append(cv_cmcc)
    
    n_categories_processed += 1
    print(f"   Category {cat} processed ({n_categories_processed}/{len(fl_cats)})")

print(f"\n{'='*80}")
print(f"Data extraction complete!")
print(f"Processed {n_categories_processed} categories with {n_points_total} total points")
print(f"{'='*80}")

# Print summary statistics
print("\n=== DATA SUMMARY ===")
for category_type in ['climate', 'roughness', 'topography']:
    print(f"\n{category_type.upper()}:")
    for label in data_annual_max[category_type].keys():
        n_am = len(data_annual_max[category_type][label]['ETH'])
        n_cv = len(data_cv[category_type][label]['ETH'])
        if n_am > 0:
            print(f"  {label}: {n_am} AM values, {n_cv} CV values per model")

# %%
########################################################################################
##----------------------------PREPARING DATA FOR BOXPLOTS----------------------------##
########################################################################################

print("\nPreparing data for boxplots...")

# ANNUAL MAXIMA DataFrames
am_climate_data    = []
am_roughness_data  = []
am_topography_data = []

for label in climate_labels:
    for model in ['ETH', 'CNRM', 'CMCC']:
        values = data_annual_max['climate'][label][model]
        for val in values:
            am_climate_data.append({'Climate': label, 'Model': model, 'Value': val})

for label in roughness_labels:
    for model in ['ETH', 'CNRM', 'CMCC']:
        values = data_annual_max['roughness'][label][model]
        for val in values:
            am_roughness_data.append({'Roughness': label, 'Model': model, 'Value': val})

for label in topography_labels:
    for model in ['ETH', 'CNRM', 'CMCC']:
        values = data_annual_max['topography'][label][model]
        for val in values:
            am_topography_data.append({'Topography': label, 'Model': model, 'Value': val})

am_climate_df    = pd.DataFrame(am_climate_data)
am_roughness_df  = pd.DataFrame(am_roughness_data)
am_topography_df = pd.DataFrame(am_topography_data)

# CV DataFrames
cv_climate_data    = []
cv_roughness_data  = []
cv_topography_data = []

for label in climate_labels:
    for model in ['ETH', 'CNRM', 'CMCC']:
        values = data_cv['climate'][label][model]
        for val in values:
            cv_climate_data.append({'Climate': label, 'Model': model, 'Value': val})

for label in roughness_labels:
    for model in ['ETH', 'CNRM', 'CMCC']:
        values = data_cv['roughness'][label][model]
        for val in values:
            cv_roughness_data.append({'Roughness': label, 'Model': model, 'Value': val})

for label in topography_labels:
    for model in ['ETH', 'CNRM', 'CMCC']:
        values = data_cv['topography'][label][model]
        for val in values:
            cv_topography_data.append({'Topography': label, 'Model': model, 'Value': val})

cv_climate_df    = pd.DataFrame(cv_climate_data)
cv_roughness_df  = pd.DataFrame(cv_roughness_data)
cv_topography_df = pd.DataFrame(cv_topography_data)

print("DataFrames prepared!")
print(f"  AM Climate: {len(am_climate_df)} rows")
print(f"  AM Roughness: {len(am_roughness_df)} rows")
print(f"  AM Topography: {len(am_topography_df)} rows")
print(f"  CV Climate: {len(cv_climate_df)} rows")

# %%
########################################################################################
###----------------------------CREATING THE PLOT-------------------------------------###
########################################################################################

print("\nCreating boxplots...")

colors = {'ETH': '#edae49', 'CNRM': '#00798c', 'CMCC': '#d1495b'}

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# ---- ROW 1: ANNUAL MAXIMA BOXPLOTS ----

# Panel 1a: Climate - Annual Maxima
ax1 = plt.subplot(gs[0, 0])
sns.boxplot(x='Climate', y='Value', hue='Model', data=am_climate_df, palette=colors, order=climate_labels, ax=ax1,
            linewidth=1.5, fliersize=3)
ax1.set_title('Climate', fontsize=14, fontweight='bold')
ax1.set_ylabel('Annual Maxima [m s$^{-1}$]', fontsize=13, fontweight='bold')
ax1.set_xlabel('')
ax1.legend([], [], frameon=False)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.tick_params(labelsize=11)

# Panel 1b: Roughness - Annual Maxima
ax2 = plt.subplot(gs[0, 1])
sns.boxplot(x='Roughness', y='Value', hue='Model', data=am_roughness_df, palette=colors, order=roughness_labels, ax=ax2,
            linewidth=1.5, fliersize=3)
ax2.set_title('Roughness', fontsize=14, fontweight='bold')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.legend([], [], frameon=False)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.tick_params(labelsize=11)

# Panel 1c: Topography - Annual Maxima
ax3 = plt.subplot(gs[0, 2])
sns.boxplot(x='Topography', y='Value', hue='Model', data=am_topography_df, palette=colors, order=topography_labels, ax=ax3,
            linewidth=1.5, fliersize=3)
ax3.set_title('Topography', fontsize=14, fontweight='bold')
ax3.set_ylabel('')
ax3.set_xlabel('')
ax3.legend([], [], frameon=False)
ax3.grid(axis='y', linestyle='--', alpha=0.3)
ax3.tick_params(labelsize=11)

# Synchronize y-axis for row 1
y_min_row1 = min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0])
y_max_row1 = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1])
for ax in [ax1, ax2, ax3]:
    ax.set_ylim(y_min_row1, y_max_row1)
    ax.yaxis.set_minor_locator(AutoMinorLocator())

# ---- ROW 2: COEFFICIENT OF VARIATION BOXPLOTS ----

# Panel 2a: Climate - CV
ax4 = plt.subplot(gs[1, 0])
sns.boxplot(x='Climate', y='Value', hue='Model', data=cv_climate_df,
            palette=colors, order=climate_labels, ax=ax4,
            linewidth=1.5, fliersize=3)
ax4.set_ylabel('Coefficient of Variation', fontsize=13, fontweight='bold')
ax4.set_xlabel('Climate level', fontsize=13)
ax4.legend([], [], frameon=False)
ax4.grid(axis='y', linestyle='--', alpha=0.3)
ax4.tick_params(labelsize=11)

# Panel 2b: Roughness - CV
ax5 = plt.subplot(gs[1, 1])
sns.boxplot(x='Roughness', y='Value', hue='Model', data=cv_roughness_df,
            palette=colors, order=roughness_labels, ax=ax5,
            linewidth=1.5, fliersize=3)
ax5.set_ylabel('')
ax5.set_xlabel('Roughness level', fontsize=13)
ax5.legend([], [], frameon=False)
ax5.grid(axis='y', linestyle='--', alpha=0.3)
ax5.tick_params(labelsize=11)

# Panel 2c: Topography - CV
ax6 = plt.subplot(gs[1, 2])
sns.boxplot(x='Topography', y='Value', hue='Model', data=cv_topography_df,
            palette=colors, order=topography_labels, ax=ax6,
            linewidth=1.5, fliersize=3)
ax6.set_ylabel('')
ax6.set_xlabel('Topography level', fontsize=13)
ax6.legend([], [], frameon=False)
ax6.grid(axis='y', linestyle='--', alpha=0.3)
ax6.tick_params(labelsize=11)

# Synchronize y-axis for row 2
y_min_row2 = min(ax4.get_ylim()[0], ax5.get_ylim()[0], ax6.get_ylim()[0])
y_max_row2 = max(ax4.get_ylim()[1], ax5.get_ylim()[1], ax6.get_ylim()[1])
for ax in [ax4, ax5, ax6]:
    ax.set_ylim(y_min_row2, y_max_row2)
    ax.yaxis.set_minor_locator(AutoMinorLocator())

# ---- FINAL ADJUSTMENTS ----

subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.text(0.03, 0.97, subplot_labels[i], transform=ax.transAxes, fontsize=15, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='none'))

# Hide y-tick labels on center and right panels
for ax in [ax2, ax3, ax5, ax6]:
    ax.set_yticklabels([])

# Global legend
legend_handles = []
for model, color in colors.items():
    patch = Patch(color=color, label=model)
    legend_handles.append(patch)

fig.legend(legend_handles, list(colors.keys()), loc='upper center', bbox_to_anchor=(0.5, 0.5),
          ncol=3, fontsize=13, frameon=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.08, hspace=0.20, top=0.96, bottom=0.05)
# plt.savefig(bd_out_fig + '/AnnualMaxima_CV_Boxplots_SpatialCategories.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nBoxplot saved successfully!")



# %%