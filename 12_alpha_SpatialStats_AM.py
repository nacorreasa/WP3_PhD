# %%
import numpy as np
import xarray as xr
import dask.array as da
import os,glob,sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mticker 
from matplotlib.ticker import LogLocator, AutoMinorLocator
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
import rasterio
from pathlib import Path

"""
This is an introduction to the high wind speeds, as such, the first plot is the distribution of
the full timeseries of wind speeds for each model. 

Code to compute and map the spatial statitistics of the Annual Maximas for each CPM member.
Statitsitcs such as: Correlation coefficient, Average, standard deviaiton and correlation 
coefficient.

It also computes and visualizes the monthly correlation of the monthly maxima based on the
Pearson correlation coefficient. 

Author : Nathalia Correa-Sánchez
"""
# %%

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

def calcular_maximos_anuales(ds, var_name='wsa100m'):
    """
    Calcula los máximos anuales de velocidad del viento para cada punto de la rejilla
    de forma explícita. 
    
    Inputs:
    - ds: Dataset xarray con datos de velocidad del viento
    - var_name: Nombre de la variable (por defecto 'wsa100m')
    
    Output:
    - ds_max_anual: Dataset con máximos anuales
    """
    # Asegurarnos que time es datetime
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))
    
    # Extraer el año de cada timestamp
    years        = ds.time.dt.year
    unique_years = np.unique(years.values)
    
    # Crear un dataset para almacenar los máximos anuales
    max_vals = []
    times    = []
    
    # Calcular máximos año por año
    for year in unique_years:
        ds_year = ds.where(ds.time.dt.year == year, drop=True)
        max_val = ds_year[var_name].max(dim='time', skipna=True)
        max_vals.append(max_val)
        times.append(np.datetime64(f'{year}-07-01'))  # Mitad del año como referencia
    
    # Concatenar los resultados
    ds_max_anual = xr.concat(max_vals, dim=pd.DatetimeIndex(times, name='time'))
    
    return ds_max_anual

def calcular_correlaciones_espaciales(da_eth, da_cnrm, da_cmcc):
    """
    Versión ultra-rápida para calcular correlaciones espaciales entre máximos anuales
    utilizando xr.corr() que está optimizado para estos cálculos.
    
    Inputs:
    - da_eth : DataArray con máximos anuales del modelo ETH
    - da_cnrm: DataArray con máximos anuales del modelo CNRM
    - da_cmcc: DataArray con máximos anuales del modelo CMCC
    
    Outputs:
    - corr_eth_cnrm, corr_eth_cmcc, corr_cnrm_cmcc: DataArrays de correlaciones entre modelos
    """
    print("Iniciando cálculo rápido de correlaciones...")
    
    # Para ETH vs CNRM
    print("Calculando correlación ETH vs CNRM...")
    eth_cnrm_corr = xr.corr(da_eth, da_cnrm, dim='time')
    
    # Para ETH vs CMCC
    print("Calculando correlación ETH vs CMCC...")
    eth_cmcc_corr = xr.corr(da_eth, da_cmcc, dim='time')
    
    # Para CNRM vs CMCC
    print("Calculando correlación CNRM vs CMCC...")
    cnrm_cmcc_corr = xr.corr(da_cnrm, da_cmcc, dim='time')
    
    print("Cálculo de correlaciones completado!")
    
    return eth_cnrm_corr, eth_cmcc_corr, cnrm_cmcc_corr

def calcular_promedio_maximos_anuales(da_eth, da_cnrm, da_cmcc):
    """
    Calcula el promedio de los máximos anuales para cada modelo en cada píxel.
    
    Inputs:
    - da_eth: DataArray con máximos anuales del modelo ETH
    - da_cnrm: DataArray con máximos anuales del modelo CNRM
    - da_cmcc: DataArray con máximos anuales del modelo CMCC
    
    Outputs:
    - avg_eth, avg_cnrm, avg_cmcc: DataArrays con los promedios de máximos anuales
    """
    print("Calculando promedio de máximos anuales...")
    
    # Cálculo directo usando xarray para vectorizar operaciones
    avg_eth = da_eth.mean(dim='time')
    avg_cnrm = da_cnrm.mean(dim='time')
    avg_cmcc = da_cmcc.mean(dim='time')
    
    print("Cálculo de promedios completado!")
    
    return avg_eth, avg_cnrm, avg_cmcc

def calcular_std_maximos_anuales(da_eth, da_cnrm, da_cmcc):
    """
    Calcula la desviación estándar de los máximos anuales para cada modelo en cada píxel.
    
    Inputs:
    - da_eth: DataArray con máximos anuales del modelo ETH
    - da_cnrm: DataArray con máximos anuales del modelo CNRM
    - da_cmcc: DataArray con máximos anuales del modelo CMCC
    
    Outputs:
    - std_eth, std_cnrm, std_cmcc: DataArrays con las desviaciones estándar de máximos anuales
    """
    print("Calculando desviación estándar de máximos anuales...")
    
    # Cálculo directo usando xarray para vectorizar operaciones
    std_eth  = da_eth.std(dim='time')
    std_cnrm = da_cnrm.std(dim='time')
    std_cmcc = da_cmcc.std(dim='time')
    
    print("Cálculo de desviaciones estándar completado!")
    
    return std_eth, std_cnrm, std_cmcc

def calcular_cv_maximos_anuales(da_eth, da_cnrm, da_cmcc):
    """
    Calcula el coeficiente de variación (CV = std/mean) de los máximos anuales 
    para cada modelo en cada píxel.
    
    Inputs:
    - da_eth: DataArray con máximos anuales del modelo ETH
    - da_cnrm: DataArray con máximos anuales del modelo CNRM
    - da_cmcc: DataArray con máximos anuales del modelo CMCC
    
    Outputs:
    - cv_eth, cv_cnrm, cv_cmcc: DataArrays con los coeficientes de variación de máximos anuales
    """
    print("Calculando coeficiente de variación de máximos anuales...")
    
    # Primero calculamos promedio y desviación estándar
    avg_eth, avg_cnrm, avg_cmcc = calcular_promedio_maximos_anuales(da_eth, da_cnrm, da_cmcc)
    std_eth, std_cnrm, std_cmcc = calcular_std_maximos_anuales(da_eth, da_cnrm, da_cmcc)
    
    # Luego calculamos el coeficiente de variación
    cv_eth  = std_eth / avg_eth
    cv_cnrm = std_cnrm / avg_cnrm
    cv_cmcc = std_cmcc / avg_cmcc
    
    print("Cálculo de coeficientes de variación completado!")
    
    return cv_eth, cv_cnrm, cv_cmcc

def crear_ticks_amigables(vmin, vmax, approx_num_ticks=11):
    """
    Crea ticks redondeados y fáciles de leer basados en el rango de datos.
    """
    # Determinar el rango
    data_range = vmax - vmin
    
    # Determinar el tamaño aproximado del paso
    approx_step = data_range / (approx_num_ticks - 1)
    
    # Encontrar un valor de paso "amigable"
    # 1, 2, 2.5, 5, 10, etc.
    magnitude = 10 ** np.floor(np.log10(approx_step))
    
    for step_size in [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10]:
        if step_size * magnitude >= approx_step:
            break
    
    nice_step = step_size * magnitude
    
    # Redondear el mínimo y máximo a múltiplos del paso
    nice_min = np.floor(vmin / nice_step) * nice_step
    nice_max = np.ceil(vmax / nice_step) * nice_step
    
    # Crear los ticks
    ticks = np.arange(nice_min, nice_max + nice_step/2, nice_step)
    
    return ticks

def add_elevation_contours(ax, colour_lines):
    """
    Añade contornos de elevación al gráfico con colores discretos.
    
    Returns:
        contour_levels: Lista con los niveles de contorno utilizados
        contour_lines: Objeto de contorno para la leyenda
    """
    # Definir los niveles de contorno y los colores 
    # contour_levels = [1000, 2500]
    contour_levels = [1000]
    
    with rasterio.open(bd_base_raster) as src:
        elevation = src.read(1)        
        # Invertir el eje vertical para corregir orientación
        elevation        = np.flipud(elevation)
        bounds           = src.bounds
        elevation_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]        
        # Crear máscara y suavizarla para solo mostrar elevaciones positivas (continentales)
        elevation_masked = np.ma.masked_where(elevation <= 0, elevation)
        elevation_smooth = gaussian_filter(elevation_masked, sigma=1.7)        
        # Añadir líneas de contorno coloreadas sin etiquetas
        contour_lines = ax.contour( elevation_smooth, levels=contour_levels, colors=colour_lines, linewidths=1.1,   alpha=0.7,  extent=elevation_extent,
            transform=ccrs.PlateCarree(), zorder=6, origin='lower', linestyles=['solid', 'solid'])
    return contour_levels, contour_lines

def add_elevation_legend(fig, contour_levels, pos, colour_lines):
    """
    Añade una leyenda horizontal usando líneas en lugar de parches.
    """    
    cax = fig.add_axes(pos)
    cax.axis('off')    
    contour_colors = colour_lines
    line_styles    = ['solid', 'solid']
    
    legend_elements = []
    for i, level in enumerate(contour_levels):
        color = contour_colors[i]
        style = line_styles[i]
        legend_elements.append( mlines.Line2D( [0], [0], color=color, linewidth=1.5, linestyle=style, label=f'{int(level)} m',
                markeredgecolor='black' if color == '#FFFFFF' else None, path_effects=[pe.withStroke(linewidth=2.5, foreground='black')] if color == '#FFFFFF' else None ))
    
    legend = cax.legend( handles=legend_elements, loc='center', ncol=len(contour_levels),frameon=True, framealpha=0.8, facecolor='#F0F0F0', title='Elevation Contours', title_fontsize=14.5, fontsize=14.5 )    
    return cax
# %%
##########################################################################################
###-----------------------------LECTRURA DE ARCHIVOS-----------------------------------###
##########################################################################################

ds_eth  = load_ds(bd_in_eth)
ds_cnrm = load_ds(bd_in_cnrm)
ds_cmcc = load_ds(bd_in_cmcc)

# %%
##########################################################################################
###---------------GENERACION DE LAS DISTRIBUCIONES DEL FULL TIMESERIES-----------------###
##########################################################################################

# Define bins
bins = np.linspace(0, 40, 81)

# Compute histograms using dask (more efficient for large data)
print("Computing Histograms...")
hist_eth, _  = da.histogram(ds_eth.wsa100m.data.flatten(), bins=bins, density=True)
hist_cnrm, _ = da.histogram(ds_cnrm.wsa100m.data.flatten(), bins=bins, density=True)
hist_cmcc, _ = da.histogram(ds_cmcc.wsa100m.data.flatten(), bins=bins, density=True)

# Compute (this triggers the dask computation)
hist_eth  = hist_eth.compute()
hist_cnrm = hist_cnrm.compute()
hist_cmcc = hist_cmcc.compute()

bin_centers = (bins[:-1] + bins[1:]) / 2

# For CDF, compute percentiles instead of full sort (more efficient)
print("Computing percentiles for CDF...")
percentiles    = np.linspace(0, 100, 1001)  # 0.1% resolution
quantiles_eth  = da.percentile(ds_eth.wsa100m.data.flatten(), percentiles).compute()
quantiles_cnrm = da.percentile(ds_cnrm.wsa100m.data.flatten(), percentiles).compute()
quantiles_cmcc = da.percentile(ds_cmcc.wsa100m.data.flatten(), percentiles).compute()
cdf_values     = percentiles / 100

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
# Plot PDF
ax1.plot(bin_centers, hist_eth, linewidth=2.5, color='#edae49', label='ETH', alpha=0.9)
ax1.plot(bin_centers, hist_cnrm, linewidth=2.5, color='#00798c', label='CNRM', alpha=0.9)
ax1.plot(bin_centers, hist_cmcc, linewidth=2.5, color='#d1495b', label='CMCC', alpha=0.9)
ax1.fill_between(bin_centers, hist_eth, alpha=0.15, color='#edae49')
ax1.fill_between(bin_centers, hist_cnrm, alpha=0.15, color='#00798c')
ax1.fill_between(bin_centers, hist_cmcc, alpha=0.15, color='#d1495b')
ax1.set_xlabel('Wind Speed at 100m (m/s)', fontsize=15, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=16, fontweight='bold')
ax1.set_title('(a) PDF', fontsize=19, fontweight='bold')
ax1.legend(fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xlim(0, 40)

# Plot CDF
ax2.plot(quantiles_eth, cdf_values, linewidth=2.3, color='#edae49', label='ETH')
ax2.plot(quantiles_cnrm, cdf_values, linewidth=2.3, color='#00798c', label='CNRM')
ax2.plot(quantiles_cmcc, cdf_values, linewidth=2.3, color='#d1495b', label='CMCC')
ax2.set_xlabel('Wind Speed at 100m (m/s)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=16, fontweight='bold')
ax2.set_title('(b) CDF', fontsize=19, fontweight='bold')
ax2.legend(fontsize=15)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xlim(0, 40)
ax2.set_ylim(0, 1)

for i, ax in enumerate([ax1, ax2]):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', labelsize=15)
    

# plt.suptitle('Full Hourly Time Series Distributions (10 years)', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig(bd_out_fig+'PDF_CDF_FullTimeSeries.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# %%
##########################################################################################
###--------------------------PROCESAMIENTO DE MAXIMOS ANUALES -------------------------###
##########################################################################################

max_anual_eth  = calcular_maximos_anuales(ds_eth)
max_anual_cmcc = calcular_maximos_anuales(ds_cmcc)
max_anual_cnrm = calcular_maximos_anuales(ds_cnrm)
# %%
##########################################################################################
###---------------------CORRELACIONES ESPACIALES DE MÁXIMOS ANUALES--------------------###
##########################################################################################

corr_eth_cnrm, corr_eth_cmcc, corr_cnrm_cmcc = calcular_correlaciones_espaciales( max_anual_eth, max_anual_cnrm, max_anual_cmcc)

ds_corr = xr.Dataset(data_vars={'corr_eth_cnrm': corr_eth_cnrm,     # Correlación entre ETH y CNRM
                                'corr_eth_cmcc': corr_eth_cmcc,     # Correlación entre ETH y CMCC
                                'corr_cnrm_cmcc': corr_cnrm_cmcc})  # Correlación entre CNRM y CMCC 

# Guardar el dataset de correlaciones
# ds_corr.to_netcdf(bd_out_am+'CorrelacionesSpatial_AM_WSA100m.nc')

#########################################################################################
###-----VISUALIZING SPATIAL CORRELATION PATTERNS TRANFORMING THEM TO NUMPY 2D---------###
#########################################################################################

corr_np_eth_cnrm  = corr_eth_cnrm.values
corr_np_eth_cmcc  = corr_eth_cmcc.values
corr_np_cnrm_cmcc = corr_cnrm_cmcc.values

correlaciones = [corr_np_eth_cnrm, corr_np_eth_cmcc, corr_np_cnrm_cmcc]
titles_base   = ['ETH vs CNRM', 'ETH vs CMCC', 'CNRM vs CMCC']
titles_idx    = ["a) ", "b) ","c) "]
colour_lines  = ['#006400', '#FFD700']

vmin, vmax = -1, 1
cmap       = plt.cm.RdBu
levels     = np.linspace(vmin, vmax, 11)
norm       = mpl.colors.BoundaryNorm(levels, cmap.N)

lats = corr_eth_cnrm.lat.values
lons = corr_eth_cnrm.lon.values

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
for i, (ax, corr, title_base) in enumerate(zip(axes, correlaciones, titles_base)):
    mean_corr     = np.nanmean(corr)
    titles_models = f"{titles_idx[i]}{title_base}"
    title         = f"{titles_models}\nMean: {mean_corr:.2f}"

    im    = ax.imshow( corr, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=ccrs.PlateCarree(), origin='lower', cmap=cmap, norm=norm)
    ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())    
    ax.coastlines(resolution='50m', color='black', linewidth=1.3)
    if  i == 0:  # Solo necesitamos obtener estos valores una vez
        contour_levels, contour_lines = add_elevation_contours(ax, colour_lines)
    else:
        add_elevation_contours(ax, colour_lines)
    ax.set_title(title, fontsize=13, fontweight="bold")
    lon_ticks = np.arange(2, 16, 2)  
    lat_ticks = np.arange(42, 50, 2)  
    
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.xaxis.set_tick_params(direction='in', labelsize=10)
    
    if i == 0:
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax.yaxis.set_tick_params(direction='in', labelsize=10)
    else:
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.yaxis.set_tick_params(direction='in', labelleft=False)
    
    ax.grid(False)    
    if i == 0:
        ax.set_ylabel('Latitud', fontsize=11)
    ax.set_xlabel('Longitud', fontsize=11)

cax  = fig.add_axes([0.92, 0.3, 0.015, 0.4])
cbar = fig.colorbar(im, cax=cax, ticks=levels)
cbar.set_label('AM Pearson Correlation', fontsize=11)
cbar.ax.tick_params(labelsize=10)
add_elevation_legend(fig, contour_levels, [0.25, 0.13, 0.5, 0.02], colour_lines)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.10, left=0.05, right=0.9, bottom=0.10, top=0.90)
plt.savefig(bd_out_fig+'SpatialCorr_AM.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

########################################################################################
##-----ABRIENDO EL CROPPED RASTER PARA EXTRAER CADA CLASE & AJUSTANDO EL DATAFRAME----##
########################################################################################

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
mask = np.isfinite(band1)
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
df_filt     = df_filt.reset_index(drop=True) ## Resetea el indice que se habia dañado luego del filtrado. 
fl_cats     = df_filt['value'].values.astype(int)

#########################################################################################
###---------MONTHLY CORRELATION COEFFICIENT BASED ON THE SPATIAL CATEGORIES-----------###
#########################################################################################

full_range = pd.date_range(start='2000-01-01 00:00:00', end='2009-12-31 23:50:00', freq='1H')

# Inicializar diccionarios para almacenar correlaciones de máximos mensuales por categoría
monthly_max_corr_eth_cnrm  = {}
monthly_max_corr_eth_cmcc  = {}
monthly_max_corr_cnrm_cmcc = {}

month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# Para cada categoría, calcular correlaciones de máximos mensuales
for cat in fl_cats:
    print(f"## Processing category {cat} for monthly maximum correlations")
    
    data_npz    = np.load(f"{bd_out_tc}_TS_cat_{cat}.npz")
    time_series = data_npz['time_series']
    coordinates = data_npz['coordinates']
    models      = data_npz['models']
    
    # Inicializar arrays para almacenar máximos mensuales para cada punto y modelo
    # Dimensiones: [punto, año, mes, modelo]
    # 10 años (2000-2009), 12 meses, 3 modelos
    monthly_maxima = np.zeros((len(coordinates), 10, 12, 3))
    
    # Procesar cada punto
    for k in range(len(coordinates)):
        serie_eth  = time_series[k, 0, :]
        serie_cnrm = time_series[k, 1, :]
        serie_cmcc = time_series[k, 2, :]

        df_s_eth  = pd.DataFrame(serie_eth, index=full_range, columns=["WS_h"])
        df_s_cnrm = pd.DataFrame(serie_cnrm, index=full_range, columns=["WS_h"])
        df_s_cmcc = pd.DataFrame(serie_cmcc, index=full_range, columns=["WS_h"])
        
        # Para cada año y mes, encontrar máximos
        for year_idx, year in enumerate(range(2000, 2010)):
            for month_idx, month in enumerate(range(1, 13)):
                # Filtrar por año y mes
                mask = (full_range.year == year) & (full_range.month == month)
                
                # Calcular máximos mensuales para cada modelo
                if mask.any():
                    monthly_maxima[k, year_idx, month_idx, 0] = df_s_eth.loc[mask, "WS_h"].max()
                    monthly_maxima[k, year_idx, month_idx, 1] = df_s_cnrm.loc[mask, "WS_h"].max()
                    monthly_maxima[k, year_idx, month_idx, 2] = df_s_cmcc.loc[mask, "WS_h"].max()
    
    # Calcular correlaciones por mes
    monthly_max_corr_eth_cnrm[cat]  = np.zeros(12)
    monthly_max_corr_eth_cmcc[cat]  = np.zeros(12)
    monthly_max_corr_cnrm_cmcc[cat] = np.zeros(12)
    
    # Para cada mes, calcular correlación de máximos a lo largo de los años
    for month_idx in range(12):
        # Para cada punto, calcular correlación y promediar
        point_corrs_eth_cnrm  = []
        point_corrs_eth_cmcc  = []
        point_corrs_cnrm_cmcc = []
        
        for k in range(len(coordinates)):
            # Obtener máximos mensuales a lo largo de los años para este punto
            eth_maxima  = monthly_maxima[k, :, month_idx, 0]  # 10 años
            cnrm_maxima = monthly_maxima[k, :, month_idx, 1]
            cmcc_maxima = monthly_maxima[k, :, month_idx, 2]
            
            # Calcular correlaciones si hay suficientes datos
            if not np.isnan(eth_maxima).any() and not np.isnan(cnrm_maxima).any():
                corr, _ = stats.pearsonr(eth_maxima, cnrm_maxima)
                point_corrs_eth_cnrm.append(corr)
            
            if not np.isnan(eth_maxima).any() and not np.isnan(cmcc_maxima).any():
                corr, _ = stats.pearsonr(eth_maxima, cmcc_maxima)
                point_corrs_eth_cmcc.append(corr)
            
            if not np.isnan(cnrm_maxima).any() and not np.isnan(cmcc_maxima).any():
                corr, _ = stats.pearsonr(cnrm_maxima, cmcc_maxima)
                point_corrs_cnrm_cmcc.append(corr)
        
        # Calcular promedio de correlaciones para todos los puntos
        if point_corrs_eth_cnrm:
            monthly_max_corr_eth_cnrm[cat][month_idx]  = np.mean(point_corrs_eth_cnrm)
        if point_corrs_eth_cmcc:
            monthly_max_corr_eth_cmcc[cat][month_idx]  = np.mean(point_corrs_eth_cmcc)
        if point_corrs_cnrm_cmcc:
            monthly_max_corr_cnrm_cmcc[cat][month_idx] = np.mean(point_corrs_cnrm_cmcc)

#########################################################################################
###-------VISUALIZING THE CORRELATION OF EACH MONTH FOR EACH SPATIAL CATEGORY--------###
#########################################################################################

color_eth_cnrm  = "#1f77b4"  # Azul
color_eth_cmcc  = "#d62728"  # Rojo
color_cnrm_cmcc = "#2ca02c"  # Verde

# Definir etiquetas simplificadas para categorías climáticas, rugosidad y topografía
climate_labels    = ['Ar', 'Tm', 'Co', 'Td']
roughness_labels  = [r"$R_1$", r"$R_2$", r"$R_3$", r"$R_4$", r"$R_5$"]
topography_labels = [r"$T_1$", r"$T_2$", r"$T_3$", r"$T_4$"]

# Mapeo de códigos de categoría a etiquetas descriptivas
cat_labels = {}
for cat in fl_cats:
    cat_str = str(cat)    
    if len(cat_str) == 3:  
        clim_digit  = int(cat_str[0])
        rough_digit = int(cat_str[1])
        topo_digit  = int(cat_str[2])

        clim_short     = climate_labels[clim_digit-1]
        cat_labels[cat] = f"{clim_short}"+"$R_{"+f"{rough_digit}"+"}T_{"+f"{topo_digit}"+"}$"
    else:  
        cat_labels[cat] = "$R_{"+f"{str(1)}"+"}$ :water"

months      = range(1, 13)
month_names = {1: 'a)January', 2: 'b)February', 3: 'c)March', 4: 'd)April', 5: 'e)May', 6: 'f)June', 7: 'g)July', 8: 'h)August', 9: 'i)September', 10: 'j)October', 11: 'k)November', 12: 'l)December'}

fig, axes = plt.subplots(4,3, figsize=(18, 16), sharex=False, sharey=True)
axes      = axes.flatten()  
# Iterar sobre cada mes
for month_idx, month in enumerate(months):
    ax = axes[month_idx]    
    # Recopilar correlaciones para todas las categorías en este mes
    cats_data = []    
    for cat in fl_cats:
        eth_cnrm_corr  = monthly_max_corr_eth_cnrm[cat][month_idx]
        eth_cmcc_corr  = monthly_max_corr_eth_cmcc[cat][month_idx]
        cnrm_cmcc_corr = monthly_max_corr_cnrm_cmcc[cat][month_idx]
        
        # Calcular correlación promedio para esta categoría para ordenarlos de acuerdo a esto
        avg_corr = np.mean([eth_cnrm_corr, eth_cmcc_corr, cnrm_cmcc_corr])        
        cats_data.append({'cat'      : cat,
                          'label'    : cat_labels[cat],
                          'eth_cnrm' : eth_cnrm_corr,
                          'eth_cmcc' : eth_cmcc_corr,
                          'cnrm_cmcc': cnrm_cmcc_corr,
                          'avg'      : avg_corr })
    
    # Ordenar categorías por correlación promedio (orden descendente)
    cats_data.sort(key=lambda x: x['avg'], reverse=True)
    x_pos      = np.arange(len(cats_data))
    labels     = [item['label'] for item in cats_data]
    markersize = 10
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(x_pos, [item['eth_cnrm'] for item in cats_data], 'o', color=color_eth_cnrm, label='ETH-CNRM', alpha=0.7, markersize=markersize)
    ax.plot(x_pos, [item['eth_cmcc'] for item in cats_data], 's', color=color_eth_cmcc, label='ETH-CMCC', alpha=0.7, markersize=markersize)
    ax.plot(x_pos, [item['cnrm_cmcc'] for item in cats_data], '^', color=color_cnrm_cmcc, label='CNRM-CMCC', alpha=0.7, markersize=markersize)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=13.5)
    ax.set_ylim(0, 1)
    if month_idx in [0,3,6,9]:  # Solo para los gráficos de la izquierda
        ax.set_ylabel('Pearson\nCorrelation (r)', fontsize=15)
        ax.tick_params(axis='y', labelsize=14)  
    ax.set_title(month_names[month], fontsize=15, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=15.5, frameon=False, bbox_to_anchor=(0.5, 0.05),  bbox_transform=fig.transFigure)

plt.tight_layout()
plt.subplots_adjust(wspace=0.20, hspace=0.63, left=0.07, right=0.98, bottom=0.15, top=0.90)
# # fig.suptitle(f"Monthly maximum wind speed correlation by spatial category", fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f"{bd_out_fig}Monthly_Correlation_by_Category.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()


#########################################################################################
###------------VISUALIZING MONTHLY MAXIMA CORRELATION FOR SPATIAL CATEGORY------------###
#########################################################################################

title_base = ['ETH vs CNRM', 'ETH vs CMCC', 'CNRM vs CMCC']
corr_data  = [monthly_max_corr_eth_cnrm, monthly_max_corr_eth_cmcc, monthly_max_corr_cnrm_cmcc]

cmap   = plt.cm.twilight_shifted  # Definir un colormap sin blancos. Alternativas: plasma,  cividis, viridis
colors = cmap(np.linspace(0, 1, 12))

fig, axes  = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
for ax_idx, (ax, title, monthly_corr) in enumerate(zip(axes, title_base, corr_data)):
    ax.set_title(f'{title}', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    
    for month_idx, month in enumerate(range(1, 13)):
        # Extraer datos de este mes para todas las categorías
        cats  = []
        corrs = []        
        for cat in fl_cats:
            cats.append(cat)
            corrs.append(monthly_corr[cat][month_idx])
        
        # Graficar línea para este mes
        ax.plot(np.arange(1, len(cats)+1), corrs, 'o-', color=colors[month_idx], label=f'{month_names[month]}', markersize=8, linewidth=1.5, alpha=0.8)
    
    # Configurar eje Y
    ax.set_ylabel(r"Pearson's $r$", fontsize=12)
    ax.set_ylim(-0.1, 1.1)  # Ajustar según tus datos
    ax.set_xticks(np.arange(1, len(cats)+1),cats)
    ax.axhline(y=0.0, color="grey", linestyle='--', lw=1.5, alpha=0.5)        

    # ax.grid(True, linestyle='--', alpha=0.7)
    if ax_idx == 2:
        ax.set_xlabel('Spatial Category', fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        # Ordenar leyenda por mes
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=6, fontsize=10)

plt.tight_layout()
fig.suptitle('Monthly Maximum Correlation Coefficient per Spatial Categories', fontsize=16, fontweight='bold', y=0.96)
plt.subplots_adjust(wspace=0.13, hspace=0.21, left=0.11, right=0.95, bottom=0.15, top=0.85)
plt.savefig(f"{bd_out_fig}Monthly_Maximum_Correlation_by_Category.png", dpi=300, bbox_inches='tight')
plt.show()

##########################################################################################
###-------------------------DESCRIPTIVE STATISTICS OF THE ANNUAL MAXIMA----------------###
##########################################################################################

# Calcular promedios de máximos anuales
avg_eth, avg_cnrm, avg_cmcc = calcular_promedio_maximos_anuales(max_anual_eth, max_anual_cnrm, max_anual_cmcc)

# Calcular coeficientes de variación de máximos anuales
cv_eth, cv_cnrm, cv_cmcc = calcular_cv_maximos_anuales( max_anual_eth, max_anual_cnrm, max_anual_cmcc)

# Calcular desviaciones estándar de máximos anuales
std_eth, std_cnrm, std_cmcc = calcular_std_maximos_anuales(max_anual_eth, max_anual_cnrm, max_anual_cmcc)


##########################################################################################
###-------------MAPPING SOME DESCRIPTIVE STATISTICS OF THE ANNUAL MAXIMA---------------###
##########################################################################################

# Agrupar métricas por filas como numpy arrar para caragar más facil la imagen abajo
metricas_filas = [[avg_eth.values, avg_cnrm.values, avg_cmcc.values],    # Fila 1: Promedios
                  [cv_eth.values, cv_cnrm.values, cv_cmcc.values]]       # Fila 2: Coeficientes de variación

titulos_metricas = ["Average", "Coefficient\nof Variation"]  # Filas
titulos_modelos  = ["ETH", "CNRM", "CMCC"]                   # Columnas
cmaps            = [plt.cm.viridis, plt.cm.plasma]           # Un colormap por métrica
lons             = avg_eth.lon.values
lats             = avg_eth.lat.values
colour_lines     = ['#FFFFFF'] # just one contour
# colour_lines     = ['#1c1a1a', '#FFFFFF']

# Definir posiciones exactas de los subplots
# [left, bottom, width, height] en coordenadas de figura (0-1)
posiciones = [ # Primera fila (Average)
            [0.08, 0.43, 0.26, 0.35],  # ETH
            [0.37, 0.43, 0.26, 0.35],  # CNRM
            [0.66, 0.43, 0.26, 0.35],  # CMCC
            
            # Segunda fila (Coefficient of Variation)
            [0.08, 0.15, 0.26, 0.38],  # ETH
            [0.37, 0.15, 0.26, 0.38],  # CNRM
            [0.66, 0.15, 0.26, 0.38]]   # CMCC

row_centers   = [0.45 + 0.35/2, 0.15 + 0.35/2]  # Centros de las filas
axes_por_fila = [[], []]
panel_labels  = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
subplot_count = 0 
fig = plt.figure(figsize=(18, 12))
for i, metrica_fila in enumerate(metricas_filas):
    valores_fila = []
    for metrica in metrica_fila:
        valores_fila.extend(metrica.flatten())
    
    vmin = np.nanmin(valores_fila)
    vmax = np.nanmax(valores_fila)
    
    # Crear niveles para colorbar discreta pero que sean más fáciles de leer
    levels = crear_ticks_amigables(vmin, vmax)
    norm   = mpl.colors.BoundaryNorm(levels, cmaps[i].N)
    
    for j, metrica in enumerate(metrica_fila):
        pos_idx = i * 3 + j
        ax      = fig.add_axes(posiciones[pos_idx], projection=ccrs.PlateCarree())
        axes_por_fila[i].append(ax)

        if i == 0:
            ax.set_title(f"{titulos_modelos[j]}", fontsize=14, fontweight="bold")

        im = ax.imshow(metrica, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=ccrs.PlateCarree(), origin='lower', cmap=cmaps[i], norm=norm)
        if j == 0 and i == 0:  # Solo necesitamos obtener estos valores una vez
            contour_levels, contour_lines = add_elevation_contours(ax, colour_lines)
        else:
            add_elevation_contours(ax, colour_lines)
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())    
        ax.coastlines(resolution='10m', color='black', linewidth=1.3)

        gl              = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
        gl.top_labels   = False
        gl.right_labels = False        
        
        if j > 0:
            gl.left_labels = False
        
        if i < len(metricas_filas) - 1:
            gl.bottom_labels = False        
        
        gl.ylabel_style = {'fontsize': 13}
        gl.xlabel_style = {'fontsize': 13}        
        ax.grid(False)

        ax.text(0.02, 0.98, panel_labels[subplot_count], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))
        
        subplot_count += 1  
    
    
    colorbar_height = posiciones[i*3][3] * 0.65  # 65% de la altura del subplot
    if i == 0:  # Fila superior - bajar un poco
        y_pos = row_centers[i] - colorbar_height/2 - 0.02  # Bajar 2%
    elif i == 1:  # Fila inferior - subir un poco  
        y_pos = row_centers[i] - colorbar_height/2 + 0.01  # Subir 1%
    else:
        y_pos = row_centers[i] - colorbar_height/2  # Posición normal

    cax  = fig.add_axes([0.94, y_pos, 0.014, colorbar_height])
    cbar            = fig.colorbar(im, cax=cax, ticks=levels)
    if i == 0:
        cbar.set_label('Wind Speed\n[m/s]', fontsize=14)
    else:
        cbar.set_label('CV\n[dimensionless]', fontsize=14)    
    cbar.ax.tick_params(labelsize=13)

    fig.text(0.03, row_centers[i], titulos_metricas[i], va='center', ha='center', fontsize=14, fontweight='bold', rotation='vertical')
add_elevation_legend(fig, contour_levels, [0.25, 0.15, 0.5, 0.02], colour_lines)
# # fig.suptitle('Spatial Distribution of Annual Maximum Wind Speed Statistics', fontsize=16, fontweight='bold', y=0.83)
plt.savefig(f"{bd_out_fig}Annual_Maximum_Statistics-AvgCV_Maps.png", format='png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


##########################################################################################
###--------------MAPPING ALL DESCRIPTIVE STATISTICS OF THE ANNUAL MAXIMA---------------###
##########################################################################################

# Agrupar métricas por filas como numpy arrar para caragar más facil la imagen abajo
metricas_filas = [[avg_eth.values, avg_cnrm.values, avg_cmcc.values],    # Fila 1: Promedios
                  [std_eth.values, std_cnrm.values, std_cmcc.values],    # Fila 2: Desviaciones estándar
                  [cv_eth.values, cv_cnrm.values, cv_cmcc.values]]       # Fila 3: Coeficientes de variación

titulos_metricas = ["Average", "Standard\nDeviation", "Coefficient\nof Variation"]
titulos_modelos  = ["ETH", "CNRM", "CMCC"]
cmaps            = [plt.cm.viridis, plt.cm.plasma, plt.cm.cividis]
lons             = avg_eth.lon.values
lats             = avg_eth.lat.values
row_centers      = [0.75, 0.5, 0.25]  # Valores aproximados para top, middle, bottom

fig = plt.figure(figsize=(14, 18))
axes_por_fila = [[], [], []]
for i, metrica_fila in enumerate(metricas_filas):
    # Calcular vmin y vmax globales para esta métrica
    valores_fila = []
    for metrica in metrica_fila:
        valores_fila.extend(metrica.flatten())
    
    vmin = np.nanmin(valores_fila)
    vmax = np.nanmax(valores_fila)
    
    # Crear niveles para colorbar discreta
    levels = np.linspace(vmin, vmax, 11)
    norm   = mpl.colors.BoundaryNorm(levels, cmaps[i].N)    
    for j, metrica in enumerate(metrica_fila):
        ax = fig.add_subplot(3, 3, i*3 + j + 1, projection=ccrs.PlateCarree())
        axes_por_fila[i].append(ax)
        im = ax.imshow(metrica,extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=ccrs.PlateCarree(), origin='lower', cmap=cmaps[i], norm=norm )  
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())        
        ax.coastlines(resolution='50m', color='black', linewidth=1.0)
        gl                 = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
        gl.top_labels      = False
        gl.right_labels    = False        
        if j > 0:
            gl.left_labels = False
        
        if i < 2:
            gl.bottom_labels = False            
        gl.ylabel_style = {'fontsize': 9}
        gl.xlabel_style = {'fontsize': 9}
        ax.grid()

        # Añadir colorbar alineada con el centro de la fila
        # Obtener la posición del primer y último subplot de la fila
        first_pos = axes_por_fila[i][0].get_position()
        last_pos = axes_por_fila[i][2].get_position()
        
        # Calcular el centro vertical de la fila
        row_center = (first_pos.y0 + first_pos.y1) / 2
        
        # Configurar colorbar con altura proporcional a los subplots
        colorbar_height = first_pos.height * 0.8  # 80% de la altura del subplot
        
        # Añadir colorbar centrada verticalmente
        cax  = fig.add_axes([0.92, row_center - colorbar_height/2, 0.015, colorbar_height])
        cbar = fig.colorbar(im, cax=cax, ticks=levels)
        
        # Colorbar
        if j == 2:
            # cax  = fig.add_axes([0.92, 0.67 - i * 0.32, 0.015, 0.25])  # Ajustar posición para cada fila
            # cbar = fig.colorbar(im, cax=cax, ticks=levels)

            cbar = fig.colorbar(im, ax=axes[i, j], pad=0.05, ticks=levels)
            
            if i == 0:
                cbar.set_label('Wind Speed (m/s)', fontsize=10)
            elif i == 1:
                cbar.set_label('Standard Deviation (m/s)', fontsize=10)
            else:
                cbar.set_label('CV (dimensionless)', fontsize=10)            
            cbar.ax.tick_params(labelsize=9)

        if i == 0:
            ax.set_title(f"{titulos_modelos[j]}", fontsize=12, fontweight="bold")

fig.suptitle('Spatial Distribution of Annual Maximum Wind Speed Statistics', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(wspace=0.08, hspace=0.21, left=0.11, right=0.95, bottom=0.15, top=0.85)
# plt.savefig(f"{bd_out_fig}Annual_Maximum_Statistics_Maps.png", dpi=300, bbox_inches='tight')
plt.show()
