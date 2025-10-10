import numpy as np
import xarray as xr
import rasterio
import os,glob,sys
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

"""
Code for the implementation onf the Triple Collocation method in the assessment of the 3 CPM
ensemble members. It start with the test over one pixel selected randomly for each category. 
Then, it select the pixels over a regular grid, and then analyze the TC behaviour in the 
categories.

Author : Nathalia Correa-Sánchez
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

bd_in_ws     = "/Dati/Data/WS_CORDEXFPS/"
bd_out_fig   = "/Dati/Outputs/Plots/WP3_development/"
bd_out_int   = "/Dati/Data/WS_CORDEXFPS/Intermediates/"
bd_in_ese    = bd_in_ws + "Ensemble_Mean/wsa100m/"
bd_in_eth    = bd_in_ws + "ETH/wsa100m/"
bd_in_cmcc   = bd_in_ws + "CMCC/wsa100m/"
bd_in_cnrm   = bd_in_ws + "CNRM/wsa100m/"
bd_in_raster = "/Dati/Outputs/Climate_Provinces/Development_Rasters/Combined_RIX_remCPM_WGS84.tif"

########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

years      = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009] 
resolution = 1.5                                 ## For the regular pixels selection
chunk_size = {'time': 100, 'lat': -1, 'lon': 50} ## For the regular pixels selection

########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################

@dataclass
class TCMetrics:
    """Clase para almacenar las métricas del análisis Triple Collocation"""
    error_variances   : Dict[str, float]      # Varianzas de error para cada dataset
    error_std         : Dict[str, float]      # Desviación estándar del error
    snr_db            : Dict[str, float]      # Relación señal-ruido en decibeles
    signal_variance   : float                 # Varianza estimada de la señal verdadera
    quality_flags     : Dict[str, np.ndarray] # Flags de calidad basados en SNR
    error_correlations: Dict[str, float]      # Correlaciones entre errores

def calculate_error_correlations(x, y, z, signal_variance, error_variances, names):
    """
    Calcula las correlaciones de error usando normalización directa
    
    Parameters:
    -----------
    x, y, z : numpy arrays
        Las tres series de mediciones
    signal_variance : float
        Varianza estimada de la señal verdadera
    error_variances : dict
        Diccionario con las varianzas de error para cada serie
    names : tuple
        Nombres de las series de datos
    
    Returns:
    --------
    dict
        Diccionario con las correlaciones de error normalizadas
    """
    # Calculamos las covarianzas entre pares de mediciones
    cov_xy = np.cov(x, y)[0,1]
    cov_xz = np.cov(x, z)[0,1]
    cov_yz = np.cov(y, z)[0,1]
    
    # Estimamos las covarianzas de error
    error_cov_xy = cov_xy - signal_variance
    error_cov_xz = cov_xz - signal_variance
    error_cov_yz = cov_yz - signal_variance
    
    # Calculamos las correlaciones normalizadas
    error_correlations = {}
    
    # Aplicamos la normalización asegurando valores en [-1, 1]
    denominador_xy = np.sqrt(error_variances[names[0]] * error_variances[names[1]])
    denominador_xz = np.sqrt(error_variances[names[0]] * error_variances[names[2]])
    denominador_yz = np.sqrt(error_variances[names[1]] * error_variances[names[2]])
    
    error_correlations[f"{names[0]}-{names[1]}"] = np.clip(
        error_cov_xy / denominador_xy if denominador_xy > 0 else 0,
        -1, 1
    )
    error_correlations[f"{names[0]}-{names[2]}"] = np.clip(
        error_cov_xz / denominador_xz if denominador_xz > 0 else 0,
        -1, 1
    )
    error_correlations[f"{names[1]}-{names[2]}"] = np.clip(
        error_cov_yz / denominador_yz if denominador_yz > 0 else 0,
        -1, 1
    )
    
    return error_correlations

def triple_collocation(x: np.ndarray, y: np.ndarray, z: np.ndarray, snr_thresholds: Tuple[float, float] = (15, 10), names: Tuple[str, str, str] = ('x', 'y', 'z') ) -> TCMetrics:
    """
    Implementa el análisis Triple Collocation calculando métricas clave de error y calidad.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Arrays 1D de igual longitud conteniendo las mediciones colocadas
    snr_thresholds : Tuple[float, float]
        Umbrales de SNR (dB) para flags de calidad (alto, medio)
    names : Tuple[str, str, str]
        Nombres identificadores para cada conjunto de datos
        
    Returns
    -------
    TCMetrics
        Objeto conteniendo todas las métricas calculadas
    """
    # Validación básica de entrada
    if not all(len(arr) == len(x) for arr in [y, z]):
        raise ValueError("Todos los arrays deben tener la misma longitud")
    
    # Remover datos faltantes si existen
    mask    = np.logical_and.reduce([~np.isnan(arr) for arr in [x, y, z]])
    x, y, z = x[mask], y[mask], z[mask]
    
    # 1. Cálculo de covarianzas - base para todas las métricas
    cov_xy = np.cov(x, y)[0,1]
    cov_xz = np.cov(x, z)[0,1]
    cov_yz = np.cov(y, z)[0,1]
    
    # 2. Estimación de la varianza de la señal verdadera
    signal_variance = cov_xy * cov_xz / cov_yz
    
    # 3. Cálculo de varianzas de error - métrica fundamental del TC
    var_ex = np.var(x) - (cov_xy * cov_xz) / cov_yz
    var_ey = np.var(y) - (cov_xy * cov_yz) / cov_xz
    var_ez = np.var(z) - (cov_xz * cov_yz) / cov_xy
    
    # Asegurar que las varianzas no sean negativas (puede ocurrir por errores numéricos)
    error_variances = {
        names[0]: max(0, var_ex),
        names[1]: max(0, var_ey),
        names[2]: max(0, var_ez)
    }
    
    # 4. Cálculo de desviaciones estándar del error
    error_std = {
        name: np.sqrt(var) for name, var in error_variances.items()
    }
    
    # 5. Cálculo de SNR en decibeles
    snr_db = {
        name: 10 * np.log10(signal_variance / var) 
        for name, var in error_variances.items()
    }
    
    # 6. Cálculo de correlaciones de error
    # Esto ayuda a verificar el supuesto de independencia
    error_correlations = calculate_error_correlations(x, y, z, signal_variance, error_variances, names)
    # error_correlations = {
    # f"{names[0]}-{names[1]}": (cov_xy - signal_variance) / 
    #                           np.sqrt(error_variances[names[0]] * error_variances[names[1]]),
    # f"{names[0]}-{names[2]}": (cov_xz - signal_variance) / 
    #                           np.sqrt(error_variances[names[0]] * error_variances[names[2]]),
    # f"{names[1]}-{names[2]}": (cov_yz - signal_variance) / 
    #                           np.sqrt(error_variances[names[1]] * error_variances[names[2]])
    # }
    
    # 7. Asignación de flags de calidad basados en SNR
    quality_flags = {}
    for name in names:
        flags = np.zeros_like(x)
        flags[snr_db[name] > snr_thresholds[0]] = 2  # Alta calidad
        flags[(snr_db[name] <= snr_thresholds[0]) & 
              (snr_db[name] > snr_thresholds[1])] = 1  # Calidad media
        quality_flags[name] = flags
    
    return TCMetrics(
        error_variances=error_variances,
        error_std=error_std,
        snr_db=snr_db,
        signal_variance=signal_variance,
        quality_flags=quality_flags,
        error_correlations=error_correlations
    )

# # Ejemplo de uso con interpretación de resultados:
# if __name__ == "__main__":
#     # Datos de ejemplo
#     np.random.seed(42)
#     true_signal = np.random.normal(10, 2, 1000)
#     x = true_signal + np.random.normal(0, 0.5, 1000)
#     y = true_signal + np.random.normal(0, 0.7, 1000)
#     z = true_signal + np.random.normal(0, 1.0, 1000)
    
#     # Realizar análisis TC
#     results = triple_collocation(x, y, z, 
#                                names=('CPM1', 'CPM2', 'CPM3'))
    
#     # Mostrar resultados principales
#     print("Métricas del Triple Collocation:")
#     print("\nDesviación estándar del error:")
#     for name, std in results.error_std.items():
#         print(f"{name}: {std:.3f}")
    
#     print("\nSNR (dB):")
#     for name, snr in results.snr_db.items():
#         print(f"{name}: {snr:.1f}")
    
#     print("\nCorrelaciones de error:")
#     for pair, corr in results.error_correlations.items():
#         print(f"{pair}: {corr:.3f}")

def create_target_grid(ds, resolution):
    """
    Crea una grilla objetivo basada en la resolución especificada
    """
    # Crear nueva grilla
    new_lats = np.arange(np.ceil(ds.lat.min().values / resolution) * resolution,
                        np.floor(ds.lat.max().values / resolution) * resolution + resolution,
                        resolution)
    new_lons = np.arange(np.ceil(ds.lon.min().values / resolution) * resolution,
                        np.floor(ds.lon.max().values / resolution) * resolution + resolution,
                        resolution)
    return new_lats, new_lons

def process_dataset_in_chunks(filez_ws, target_lats, target_lons):
    """
    Procesa el dataset en chunks y retorna un dataset de xarray con las series temporales
    
    Parameters:
    -----------
    filez_ws : list
        Lista de archivos a procesar
    target_lats : array
        Latitudes objetivo
    target_lons : array
        Longitudes objetivo
    
    Returns:
    --------
    xarray.Dataset
        Dataset combinado con las series temporales extraídas
    """
    processed_datasets = []
    batch_size         = 5  # Número de archivos a procesar simultáneamente (ahorrar memoria RAM)
    
    for i in range(0, len(filez_ws), batch_size):
        batch_files = filez_ws[i:i + batch_size]
        print(f"Processing files {i+1} to {min(i+batch_size, len(filez_ws))} of {len(filez_ws)}")
        
        # Abrir batch de archivos
        ds_batch = xr.open_mfdataset(batch_files, combine='nested', concat_dim='time', chunks={'time': -1, 'lat': -1, 'lon': 50},parallel=True)
        
        # Remuestrear a la grilla objetivo
        ds_batch = ds_batch.sel(lat=target_lats, lon=target_lons, method='nearest')
        
        # Eliminar duplicados de tiempo
        ds_batch = ds_batch.sel(time=~ds_batch.get_index("time").duplicated())
        
        # Computar y almacenar en memoria
        ds_batch = ds_batch.compute()
        
        # Agregar a la lista de datasets procesados
        processed_datasets.append(ds_batch)
        
        # Limpiar memoria explícitamente
        ds_batch.close()
        
    # Combinar todos los datasets
    print("Combining all processed datasets...")
    final_dataset = xr.concat(processed_datasets, dim='time')
    
    # Ordenar por tiempo
    final_dataset = final_dataset.sortby('time')
    
    return final_dataset

def clean_with_xarray(ds, variable_name='wsa100m'):
    """
    Elimina filas y columnas con NaN usando xarray (ay una fila y una columna de nans que sobra)
    
    INPUTS:
    - ds            : xarray.Dataset, Dataset original
    - variable_name : str, Nombre de la variable a limpiar
        
    OUTPUTS:
    - clean_ds : xarray.Dataset, dataset limpio sin las filas/columnas de NaN
    """
    def find_valid_indices(da, dim):
        return ~da.isnull().all(dim=[d for d in da.dims if d != dim])
    
    valid_lats = find_valid_indices(ds[variable_name], 'lat')
    valid_lons = find_valid_indices(ds[variable_name], 'lon')
    
    clean_ds = ds.isel(lat=valid_lats.values.nonzero()[0],lon=valid_lons.values.nonzero()[0] )

    return clean_ds

def style_axis(ax):
    """
    Function to set the format to the plots
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def decompose_spatial_category(category_value):
    """
    Decomposes the 3-digit categorical value into its components.
    
    INPUTS :
        category_value (int): Spatial category code (e.g., 332)
    
    OUTPUTS :
        dict: Decomposed components of the category
    """
    str_category = str(category_value).zfill(3)
    return {
        'climate': {
            'code': int(str_category[0]),
            'description': {
                1: 'Arid', 
                2: 'Temperate', 
                3: 'Cold', 
                4: 'Polar'
            }.get(int(str_category[0]), 'Unknown')
        },
        'roughness': {
            'code': int(str_category[1]),
            'description': {
                1: 'Very Low', 
                2: 'Low', 
                3: 'Moderate', 
                4: 'High', 
                5: 'Very High'
            }.get(int(str_category[1]), 'Unknown')
        },
        'slope_variance': {
            'code': int(str_category[2]),
            'description': {
                1: 'Very Low', 
                2: 'Low', 
                3: 'Moderate', 
                4: 'High', 
                5: 'Very High'
            }.get(int(str_category[2]), 'Unknown')
        }
    }

def create_scatter_legend(fig, gs):
    """
    Scatter legend handling
    """
    ax2 = fig.add_subplot(gs[2, 0])  # Occupy full third row for legend
    ax2.scatter([], [], c='black', s=60, marker="X", label="Regularly selected points")
    ax2.legend(bbox_to_anchor=(0.5, 1.1), loc='lower center', ncol=1)
    ax2.axis('off')
    return ax2

def prepare_stacked_data(df, column, order):
    """
    Prepare stacked data function. It must have the column 'SpatialCat'.
    """
    unique_values = sorted(df['SpatialCat'].unique())
    stacked_df = pd.DataFrame(index=order, columns=unique_values).fillna(0)
    
    for level in order:
        subset = df[df[column] == level]
        value_counts = subset['SpatialCat'].value_counts()
        for val, count in value_counts.items():
            stacked_df.loc[level, val] = count
    
    return stacked_df

def plot_stacked_bar_chart(ax, data, title, colors, ylabel):
    """
    Improved stacked bar plot function
    """
    data.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='none')
    # data.plot(kind='bar', stacked=True, ax=ax, color=[color_map[val] for val in data.columns], edgecolor='none') # Use color_map
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=10, which='both', direction='in')
    
    # Remove floats from x-axis labels
    labels = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    sns.despine(ax=ax, top=True, right=True)
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def smart_plot_boxp_scat(ax, data, positions, color, width=0.25):
    """
    Función para crear boxplot o scatter según el número de puntos.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for pos, d in zip(positions, data):
        if len(d) > 1:
            bp = ax.boxplot([d], positions=[pos],
                          patch_artist=True,
                          medianprops=dict(color="black"),
                          boxprops=dict(facecolor=color, alpha=0.6),
                          widths=width)
        else:
            if len(d) == 1:
                ax.scatter([pos], d, color=color, marker='8', s=60, zorder=5)



########################################################################################
##------------------ABRIENDO EL RASTER PARA EXTRAER CADA CLASE------------------------##
########################################################################################

comblay          = rasterio.open(bd_in_raster)
band1            = comblay.read(1) ## Solo tiene una banda
band1[band1 < 0] = np.nan          ## Reemplazando los negativos con nan o 0(Tener en cuenta NoData= -3.40282e+38) 

layclass = np.delete(np.unique(band1), np.where(np.unique(band1) == 0)) ## revisar exceso de clases.

###--------------------------RANDOM PIXELS FOR ALL CATEGORIES------------------------###
########################################################################################
##-----------SELECCIONANDO PIXELES CADA CLASE SOBRE COORDENADAS ALEATORIAS------------##
########################################################################################

random_lat = []
random_lon = []
random_idx = []
for i in range(len(layclass)):
    lat_idxs, lon_idxs = np.where(band1 == layclass[i]) ## Asume la grilla típica geoespacia [latitud, longitud].
    num_indices        = len(lat_idxs)                  ## Ambos lat_indices, lon_indice tiene la misma longitud
    random_idxs        = np.random.randint(0, num_indices)
    print (random_idxs)                                 ## Recordar que hay clases muy amplias entonces peuden haber inces del orden de miles.
    random_lat.append(lat_idxs[random_idxs])
    random_lon.append(lon_idxs[random_idxs])
    random_idx.append(random_idxs)
        
random_idx = np.array(random_idx)
np.save(bd_out_int+"RandomIdx_ClassificationLayers.npy", random_idx) ## En caso en el que se enecesiten más pruebas sobre los mismos pixeles

########################################################################################
### ------------ITERATING ACCROSS CPM DATA SELECTING RANDOM PIXELS DATA--------------###
########################################################################################

filez_ws_eth  = sorted(glob.glob(f"{bd_in_eth}*.nc"))
ds_eth_o      = xr.open_mfdataset(filez_ws_eth, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
ds_eth        = clean_with_xarray(ds_eth_o)

filez_ws_cmcc = sorted(glob.glob(f"{bd_in_cmcc}*.nc"))
ds_cmcc_o     = xr.open_mfdataset(filez_ws_cmcc, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
ds_cmcc       = clean_with_xarray(ds_cmcc_o)

filez_ws_cnrm = sorted(glob.glob(f"{bd_in_cnrm}*.nc"))
ds_cnrm_o     = xr.open_mfdataset(filez_ws_cnrm, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
ds_cnrm       = clean_with_xarray(ds_cnrm_o)

results_tc = []
for i in range(len(layclass)):
    serie_ds_eth  = ds_eth ['wsa100m'].isel(lat=random_lat[i], lon=random_lon[i], drop=True).to_numpy()
    serie_ds_cnrm = ds_cnrm ['wsa100m'].isel(lat=random_lat[i], lon=random_lon[i], drop=True).to_numpy()
    serie_ds_cmcc = ds_cmcc ['wsa100m'].isel(lat=random_lat[i], lon=random_lon[i], drop=True).to_numpy()

    results = triple_collocation(x=serie_ds_eth, y=serie_ds_cnrm, z=serie_ds_cmcc, names=('CPM1:ETH', 'CPM2:CNRM', 'CPM3:CMCC'))

    results_tc.append(results)
    print("## TC in: "+str(i))

ds_eth.close() 
ds_cmcc.close()
ds_cnrm.close()

########################################################################################
###---EXTRACTING METRICS FOR GENERATING THE TC PLOT OVER RANDOM PIXELS PER CATEGORY--###
########################################################################################

error_variances    = {key: [res.error_variances[key] for res in results_tc] for key in results_tc[0].error_variances.keys()}
error_correlations = {key: [res.error_correlations[key] for res in results_tc] for key in results_tc[0].error_correlations.keys()}
signal_variance    = [res.signal_variance for res in results_tc]
snr_db             = {key: [res.snr_db[key] for res in results_tc] for key in results_tc[0].snr_db.keys()}
colours            = {"CPM1:ETH": "#edae49",    # Amarillo
                      "CPM2:CNRM": "#00798c",   # Azul
                      "CPM3:CMCC": "#d1495b",}  # Rojo

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Subplot 1: Error variance
style_axis(axes[0])
for key, values in error_variances.items():
    axes[0].plot(np.arange(len(layclass)), values, label=key, marker='o', color =colours.get(key))
axes[0].set_ylabel("Error Variance")
axes[0].grid(True, linestyle='--', alpha=0.7)

# Subplot 2: Signal variance
style_axis(axes[1])
axes[1].plot(np.arange(len(layclass)), signal_variance, label="Signal Variance", color="purple", marker='s')
axes[1].set_ylabel("Signal Variance")
axes[1].grid(True, linestyle='--', alpha=0.7)

# Subplot 3: SNR
style_axis(axes[2])
for key, values in snr_db.items():
    axes[2].plot(np.arange(len(layclass)), values, label=key, marker='x', color =colours.get(key))
axes[2].set_ylabel("SNR (dB)")
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend()

# Subplot 4: Error correlations
style_axis(axes[3])
for key, values in error_correlations.items():
    axes[3].plot(np.arange(len(layclass)), values, label=key, marker='^')
axes[3].set_ylabel("Error Correlation")
axes[3].grid(True, linestyle='--', alpha=0.7)
axes[3].set_xticks(np.arange(len(layclass)))  
axes[3].set_xticklabels(layclass.astype(int), rotation=90, ha='right')  
axes[3].legend()
axes[3].set_xlabel("Spatial category")

plt.suptitle("Triple Collocation Metrics Across Spatial Cathegories", weight='bold', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar espacios para no sobreponer el título
plt.subplots_adjust(wspace=0.30, hspace=0.15, left=0.10, right =0.97, bottom = 0.08) 
# plt.savefig(bd_out_fig+"TC_Relative_CPMs.png", format='png', dpi=300, transparent=True)
plt.show()

########################################################################################
###---------TC PLOT WITHOUT ERROR CORRELATION OVER RANDOM PIXELS PER CATEGORY--------###
########################################################################################

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Subplot 1: Error variance
style_axis(axes[0])
for key, values in error_variances.items():
    axes[0].plot(np.arange(len(layclass)), values, label=key, marker='o', color =colours.get(key))
axes[0].set_ylabel("Error Variance")
axes[0].grid(True, linestyle='--', alpha=0.7)

# Subplot 2: Signal variance
style_axis(axes[1])
axes[1].plot(np.arange(len(layclass)), signal_variance, label="Signal Variance", color="purple", marker='s')
axes[1].set_ylabel("Signal Variance")
axes[1].grid(True, linestyle='--', alpha=0.7)

# Subplot 3: SNR
style_axis(axes[2])
for key, values in snr_db.items():
    axes[2].plot(np.arange(len(layclass)), values, label=key, marker='x', color =colours.get(key))
axes[2].set_ylabel("SNR (dB)")
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend()
axes[2].set_xticks(np.arange(len(layclass)))  
axes[2].set_xticklabels(layclass.astype(int), rotation=90, ha='right')  
axes[2].set_xlabel("Spatial category")

plt.suptitle("Triple Collocation Metrics Across Spatial Categories", weight='bold', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar espacios para no sobreponer el título
plt.subplots_adjust(wspace=0.30, hspace=0.15, left=0.10, right =0.97, bottom = 0.08) 
plt.savefig(bd_out_fig+"TC_Relative_CPMs_SinErrorCorr.png", format='png', dpi=300, transparent=True)
plt.show()

###--------------------------------REGULAR GRID PIXELSS------------------------------###
########################################################################################
##--------SELECCIONANDO PIXELES EN UNA MALLA REGULAR CATEGORIAS ALEATORIAS------------##
########################################################################################

# Obtener lista de archivos
bd_ws_ese    = f"{bd_in_ese}"
filez_ws_ese = sorted(glob.glob(f"{bd_ws_ese}*.nc"))

# Obtener grilla objetivo del primer archivo
with xr.open_dataset(filez_ws_ese[0]) as ds:
    target_lats, target_lons = create_target_grid(ds, resolution)

# Procesar datos para conseguir las lats y lons de referencia
result_ds   = process_dataset_in_chunks(filez_ws_ese, target_lats, target_lons)
regular_lat = result_ds.lat.values
regular_lon = result_ds.lon.values

print("Processing completed! regular lats and lons based on enesemble dataset grid")

########################################################################################
##---ADJUSTING AND MAPPING SELECTED VALUES IN THE GRID AND STACKED PLOTS OF THOSE-----##
########################################################################################

# Definir los valores mínimos y máximos de latitud y longitud para im
ds_latlon          = xr.open_mfdataset(filez_ws_ese[0], combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
lon_min, lon_max   = ds_latlon.lon.values.min(), ds_latlon.lon.values.max()
lat_min, lat_max   = ds_latlon.lat.values.min(), ds_latlon.lat.values.max()
lon_mesh, lat_mesh = np.meshgrid(regular_lon, regular_lat)

## Data extraction and analysis (common for all 3 plots)
data_for_bars = []
lon_flat      = lon_mesh.flatten()
lat_flat      = lat_mesh.flatten()

for lon, lat in zip(lon_flat, lat_flat):
    # Encuentra el índice más cercano en la malla original (ds_latlon)
    lat_idx = np.abs(ds_latlon.lat.values - lat).argmin()
    lon_idx = np.abs(ds_latlon.lon.values - lon).argmin()

    value = band1[lat_idx, lon_idx]

    if np.isfinite(value):
        data_for_bars.append(int(value))
    else:
        print("NAN-ocean region") ## No se cuentan en df

df = pd.DataFrame({'SpatialCat': data_for_bars})

# Add decomposition columns to the DataFrame
df['climate']        = df['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['climate']['description'])
df['roughness']      = df['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['roughness']['description'])
df['slope_variance'] = df['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['slope_variance']['description']) 

# Define order of levels
climate_orden   = ['Arid', 'Temperate', 'Cold', 'Polar']
roughness_orden = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
variance_orden  = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

# Prepare stacked DataFrames
df_stacked_climate   = prepare_stacked_data(df, 'climate', climate_orden)
df_stacked_roughness = prepare_stacked_data(df, 'roughness', roughness_orden)
df_stacked_variance  = prepare_stacked_data(df, 'slope_variance', variance_orden)

unique_vals    = np.unique(band1)
unique_vals    = unique_vals[np.isfinite(unique_vals)]
num_categories = len(unique_vals)

# Crear un colormap discreto y normalizar
if num_categories <= 20:
    cmap = plt.get_cmap('tab20', num_categories)
else:
    cmap = plt.get_cmap('nipy_spectral', num_categories)
bounds = np.arange(num_categories + 1)
norm = BoundaryNorm(bounds, cmap.N) ## Para ax1. Con esto cada valor en bounbds tiene un color

colors = sns.color_palette("husl", len(list(df_stacked_climate.columns))) ## Para ax3, 4, 5.  Ojala pudiera modificar esto


# Create figure with GridSpec
fig = plt.figure(figsize=(15, 6))
gs  = GridSpec(3, 2, width_ratios=[1, 1], figure=fig)

# Subplot 1: Mapa de categorías espaciales con imshow
ax1                     = fig.add_subplot(gs[:2, 0], projection=ccrs.PlateCarree())
band1_categorized       = np.full(band1.shape, np.nan) # Crea un array del mismo tamaño que band1 lleno de NaN
mask                    = np.isfinite(band1)           # Mascara de valores no NaN - necesario para la colorbar
band1_categorized[mask] = np.digitize(band1[mask], bins=unique_vals, right=True) - 1 #Categoriza solo los valores no NaN
im                      = ax1.imshow(band1_categorized, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", cmap=cmap, norm=norm, alpha=0.7, transform=ccrs.PlateCarree())
scatter            = ax1.scatter(lon_mesh.flatten(), lat_mesh.flatten(), c='black', transform=ccrs.PlateCarree(), s=60, marker="X")
ax1.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
gl              = ax1.gridlines(draw_labels=True, linewidth=0.8, color="gray", alpha=0.8, linestyle="--")
gl.right_labels = False
gl.top_labels   = False
gl.ylocator     = MaxNLocator(nbins=5)
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
ax1.set_title("a) Regular grid pixels selection \n over all spatial categories", fontsize=12, fontweight="bold")
cax = fig.add_axes([ax1.get_position().x0-0.07, ax1.get_position().y0 - 0.07, ax1.get_position().width +0.09, 0.02])  # Ajusta la posición y tamaño
cbar = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds[:-1] + 0.5)
cax.minorticks_off() # 0 para no mostrar minor ticks
cbar.set_ticklabels([str(int(val)) if np.isclose(val % 1, 0) else str(val) for val in unique_vals], fontsize = 7, rotation=90, ha='right')
cbar.ax.xaxis.set_ticks_position('bottom')  # Poner ticks abajo
cbar.set_label("Spatial category (Total :"+str(num_categories)+")", fontsize=10)

ax2 = fig.add_subplot(gs[2, 0])  # Occupy full third row for legend
ax2.scatter([], [], c='black', s=60, marker="X", label="Regularly selected points")
ax2.legend(bbox_to_anchor=(0.5, 1.1), loc='lower center', ncol=1)
ax2.axis('off')

# Subplots for stacked bar charts
ax3 = fig.add_subplot(gs[0, 1])  # Climate
ax4 = fig.add_subplot(gs[1, 1])  # Roughness
ax5 = fig.add_subplot(gs[2, 1])  # Slope Variance

# Plot stacked bar charts
plot_stacked_bar_chart(ax3, df_stacked_climate, "b) Distribution of Pixels by Climate", colors, "Pixel Count")
plot_stacked_bar_chart(ax4, df_stacked_roughness, "c) Distribution of Pixels by Roughness", colors, "Pixel Count")
plot_stacked_bar_chart(ax5, df_stacked_variance, "d) Distribution of Pixels by Slope Variance", colors, "Pixel Count")

# Add global legend for spatial categories
legend_handles = [Patch(facecolor=colors[i], label=str(val)) for i, val in enumerate(sorted(df['SpatialCat'].unique()))]
fig.legend(handles=legend_handles, title="Spatial category \n (after selection)", fontsize=10, title_fontsize=10, bbox_to_anchor=(0.98, 0.80), loc='upper right')

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.subplots_adjust(wspace=0.3, hspace=0.35, left=0.10, right=0.88, bottom=0.10, top=0.89)
plt.savefig(bd_out_fig+"MapStackedBars_RegularSlect_SpaCat.png", format='png', dpi=300, transparent=True)
plt.show()


########################################################################################
###---IERATING ACCROSS CPM DATA SELECTING REGULAR PIXELS DATA TO COMPUTE TC METRICS--###
########################################################################################

filez_ws_eth  = sorted(glob.glob(f"{bd_in_eth}*.nc"))
ds_eth_o      = xr.open_mfdataset(filez_ws_eth, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
ds_eth        = clean_with_xarray(ds_eth_o)

filez_ws_cmcc = sorted(glob.glob(f"{bd_in_cmcc}*.nc"))
ds_cmcc_o     = xr.open_mfdataset(filez_ws_cmcc, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
ds_cmcc       = clean_with_xarray(ds_cmcc_o)

filez_ws_cnrm = sorted(glob.glob(f"{bd_in_cnrm}*.nc"))
ds_cnrm_o     = xr.open_mfdataset(filez_ws_cnrm, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
ds_cnrm       = clean_with_xarray(ds_cnrm_o)

results_tc_regular =  np.empty((len(regular_lat), len(regular_lon)), dtype=object)
for i, lat in enumerate(regular_lat):
    for j, lon in enumerate(regular_lon):

        serie_ds_eth  = ds_eth ['wsa100m'].sel(lat=lat, lon=lon, method='nearest', drop=True).to_numpy()
        serie_ds_cnrm = ds_cnrm['wsa100m'].sel(lat=lat, lon=lon, method='nearest', drop=True).to_numpy()
        serie_ds_cmcc = ds_cmcc['wsa100m'].sel(lat=lat, lon=lon, method='nearest', drop=True).to_numpy()

        results_tc_regular[i,j] = triple_collocation(x=serie_ds_eth, y=serie_ds_cnrm, z=serie_ds_cmcc, names=('CPM1:ETH', 'CPM2:CNRM', 'CPM3:CMCC'))

        print("## TC in: "+str(i))

ds_eth.close() 
ds_cmcc.close()
ds_cnrm.close()

########################################################################################
###-----------------------EXTRACTING REGULAR PIXELS TC RESULTS TO PLOT---------------###
########################################################################################

shape_2d = results_tc_regular.shape  # (7, 13)

# Para error variances
error_var_reg = {}
for key in results_tc_regular[0,0].error_variances.keys():
    error_var_reg[key] = np.empty(shape_2d)
    for i in range(shape_2d[0]):
        for j in range(shape_2d[1]):
            error_var_reg[key][i,j] = results_tc_regular[i,j].error_variances[key]

# Para error correlations
error_corr_reg = {}
for key in results_tc_regular[0,0].error_correlations.keys():
    error_corr_reg[key] = np.empty(shape_2d)
    for i in range(shape_2d[0]):
        for j in range(shape_2d[1]):
            error_corr_reg[key][i,j] = results_tc_regular[i,j].error_correlations[key]

# Para signal variance
signal_var_reg = np.empty(shape_2d)
for i in range(shape_2d[0]):
    for j in range(shape_2d[1]):
        signal_var_reg[i,j] = results_tc_regular[i,j].signal_variance

# Para SNR en dB
snr_db_reg = {}
for key in results_tc_regular[0,0].snr_db.keys():
    snr_db_reg[key] = np.empty(shape_2d)
    for i in range(shape_2d[0]):
        for j in range(shape_2d[1]):
            snr_db_reg[key][i,j] = results_tc_regular[i,j].snr_db[key]

########################################################################################
###----------------------PLOTTING REGULAR PIXELS TC RESULTS TO METRICS---------------###
########################################################################################

ds_latlon   = xr.open_mfdataset(filez_ws_ese[0], combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
# Crear DataFrame con todos los resultados
data_points = []
for i in range(shape_2d[0]):
    for j in range(shape_2d[1]):
        lon = regular_lon[j]
        lat = regular_lat[i]
        
        # Encuentra el índice más cercano en la malla original
        lat_idx = np.abs(ds_latlon.lat.values - lat).argmin()
        lon_idx = np.abs(ds_latlon.lon.values - lon).argmin()
        
        # Verificar si el punto está en tierra (no es océano)
        if np.isfinite(band1[lat_idx, lon_idx]) and band1[lat_idx, lon_idx]!= 0 :  # Acá oceanos salen como 0
            point_data = { 'lon': lon, 'lat': lat, 'signal_variance': signal_var_reg[i,j] }

            #Añadir spatial category
            point_data[f'SpatialCat'] = int(band1[lat_idx, lon_idx])
            
            # Añadir error variances
            for key in error_var_reg.keys():
                point_data[f'error_var_{key}'] = error_var_reg[key][i,j]
            
            # Añadir SNR
            for key in snr_db_reg.keys():
                point_data[f'snr_{key}'] = snr_db_reg[key][i,j]
                
            data_points.append(point_data)

df = pd.DataFrame(data_points)
df = df.dropna(subset=["signal_variance"]) # No nan metrics porque en algunos casos no esta el CPM en los bordes

cat_counts   = df.SpatialCat.value_counts().reindex(spatial_cats).fillna(0) # Contar puntos por categoría
spatial_cats = np.sort(np.unique(df.SpatialCat.values))                     # Ordenar las categorías espaciales

colours = { "CPM1:ETH": "#edae49",    # Amarillo
            "CPM2:CNRM": "#00798c",   # Azul
            "CPM3:CMCC": "#d1495b" }  # Rojo

fig = plt.figure(figsize=(12, 14))
gs  = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.4])

# 1. Signal Variance 
ax1         = fig.add_subplot(gs[0])
data_signal = [df[df.SpatialCat == cat]['signal_variance'] for cat in spatial_cats]
smart_plot_boxp_scat(ax1, data_signal, range(len(spatial_cats)), 'purple')
ax1.set_ylabel("Signal Variance", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(-0.5, len(spatial_cats)-0.5)

# 2. Error Variance 
ax2       = fig.add_subplot(gs[1])
positions = np.arange(len(spatial_cats))
width     = 0.25
for i, (model, color) in enumerate(colours.items()):
    data_error = [df[df.SpatialCat == cat][f'error_var_{model}'] for cat in spatial_cats]
    smart_plot_boxp_scat(ax2, data_error, positions + (i-1)*width, color, width)
ax2.set_ylabel("Error Variance", fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(-0.5, len(spatial_cats)-0.5)

# 3. SNR 
ax3 = fig.add_subplot(gs[2])
for i, (model, color) in enumerate(colours.items()):
    data_snr = [df[df.SpatialCat == cat][f'snr_{model}'] for cat in spatial_cats]
    smart_plot_boxp_scat(ax3, data_snr, positions + (i-1)*width, color, width)
ax3.set_ylabel("SNR (dB)", fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_xlim(-0.5, len(spatial_cats)-0.5)

# 4. Frequency plot 
ax4   = fig.add_subplot(gs[3])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
bars = ax4.bar(range(len(spatial_cats)), cat_counts, color='gray', alpha=0.6)
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')
ax4.set_ylabel("Frequency", fontsize=10)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.set_xlim(-0.5, len(spatial_cats)-0.5)
# Configurar ejes x para que sean homogeneos siempre
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(range(len(spatial_cats)))
    if ax == ax4:  # Solo mostrar etiquetas en el último subplot
        ax.set_xticklabels(spatial_cats, rotation=0)
    else:
        ax.set_xticklabels([])
ax4.set_xlabel("Spatial Category", fontsize=10)

# Crear leyenda fuera de los subplots
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6, label=model)for model, color in colours.items()]
# fig.legend(handles=legend_elements, loc='center right', box_to_anchor=(0.99, 0.5),title="Models")
fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5), title="Models")
plt.suptitle("Triple Collocation Metrics Distribution by Spatial Category", fontweight="bold", fontsize=13, y=0.95)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.20, left=0.10, right=0.88, bottom=0.10, top=0.89)
plt.savefig(bd_out_fig+"TC_Relative_CPMs_SinErrorCor_RegularSlect_SpaCat.png", format='png', dpi=300, transparent=True)
plt.show()

########################################################################################
###-PLOTTING REGULAR PIXELS TC RESULTS TO METRICS AS CATEGORIES AND LEVELS FUNCTION--###
########################################################################################

# Add decomposition columns to the DataFrame
df['climate']        = df['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['climate']['description'])
df['roughness']      = df['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['roughness']['description'])
df['slope_variance'] = df['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['slope_variance']['description']) 


# Definir categorías para cada columna
categories = ['climate', 'roughness', 'slope_variance']
cat_values = {'climate': ['Arid', 'Temperate', 'Cold', 'Polar'],
              'roughness': ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
              'slope_variance': ['Very Low', 'Low', 'Moderate', 'High', 'Very High']}

# Títulos para las filas
row_titles = ['Signal Variance', 'Error Variance', 'SNR (dB)']

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# Iterar sobre las columnas (categorías)
for j, category in enumerate(categories):
    # Obtener valores únicos ordenados para esta categoría
    values = cat_values[category]
    positions = np.arange(len(values))
    
    # Primera fila: Signal Variance
    data_signal = [df[df[category] == cat]['signal_variance'] for cat in values]
    smart_plot_boxp_scat(axes[0,j], data_signal, positions, 'purple')
    if j == 0:
        axes[0,j].set_ylabel(row_titles[0], fontsize=10)
    axes[0,j].grid(True, linestyle='--', alpha=0.7)
    axes[0,j].set_title(category.replace('_', ' ').title(), fontsize=12, pad=10)
    
    # Segunda fila: Error Variance
    width = 0.25
    for k, (model, color) in enumerate(colours.items()):
        data_error = [df[df[category] == cat][f'error_var_{model}'] for cat in values]
        smart_plot_boxp_scat(axes[1,j], data_error, positions + (k-1)*width, color, width)
    if j == 0:
        axes[1,j].set_ylabel(row_titles[1], fontsize=10)
    axes[1,j].grid(True, linestyle='--', alpha=0.7)
    
    # Tercera fila: SNR
    for k, (model, color) in enumerate(colours.items()):
        data_snr = [df[df[category] == cat][f'snr_{model}'] for cat in values]
        smart_plot_boxp_scat(axes[2,j], data_snr, positions + (k-1)*width, color, width)
    if j == 0:
        axes[2,j].set_ylabel(row_titles[2], fontsize=10)
    axes[2,j].grid(True, linestyle='--', alpha=0.7)
    
    # Configurar ejes x para cada columna
    for i in range(3):
        ax = axes[i,j]
        ax.set_xticks(positions)
        if i == 2:  # Solo mostrar etiquetas en la última fila
            ax.set_xticklabels(values, rotation=0)
        else:
            ax.set_xticklabels([])
        ax.set_xlim(-0.5, len(values)-0.5)

legend_elements   = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6, label=model) for model, color in colours.items()]
fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5), title="Models")
plt.suptitle("Triple Collocation Metrics Distribution by Categories", fontsize=13, fontweight="bold", y=0.97)

plt.tight_layout()
plt.subplots_adjust(wspace=0.17, hspace=0.20, left=0.05, right=0.88, bottom=0.10, top=0.89)
plt.savefig(bd_out_fig+"TC_Relative_CPMs_SinErrorCor_RegularSlect_SpaCat-Levels.png", format='png', dpi=300, transparent=True)
plt.show()


















