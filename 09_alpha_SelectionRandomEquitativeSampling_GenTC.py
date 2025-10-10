
import numpy as np
import numpy.ma as ma
import xarray as xr
import rasterio
import random
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
import os,glob,sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import pickle


"""
Code to select N_pixels randomly from each of the most frequent spatial categories, 
and compute the Triple Collocation (Generalized) from all those samples, in order to
compare those performance metrics across categories.

Spatial categories whose relative frequency exceeded the first quartile (25th percentile) 
of the frequency distribution were selected to ensure sufficient statistical representativeness
in subsequent triple-collocation analyses.

The updated routine uses the cropped CORDEX domain. 

Author : Nathalia Correa-Sánchez
"""


########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

bd_in_ws     = "/Dati/Data/WS_CORDEXFPS/"
bd_out_fig   = "/Dati/Outputs/Plots/WP3_development/"
bd_out_int   = "/Dati/Data/WS_CORDEXFPS/Intermediates/"
bd_out_tc    = "/Dati/Outputs/WP3_SamplingSeries_CPM/"
bd_in_eth    = bd_in_ws + "ETH/wsa100m_crop/"
bd_in_cmcc   = bd_in_ws + "CMCC/wsa100m_crop/"
bd_in_cnrm   = bd_in_ws + "CNRM/wsa100m_crop/"
bd_in_raster = "/Dati/Outputs/Climate_Provinces/Development_Rasters/FinalRasters_In-Out/" 
bd_ras_input = bd_in_raster+"SEA-LAND_Combined_RIX_remCPM_WGS84.tif" # Hay que cortarlo

########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

years             = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009] 
chunk_size        = {'time': 100, 'lat': -1, 'lon': 50} ## For the regular pixels selection
filas_eliminar    = [0]  # Primera  fila, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada 
columnas_eliminar = [0]  # Primera columna, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada
N_pixels          = 100  # Cantidad de puntos en cada muestra

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

# Enfoque alternativo más robusto
def calculate_error_correlations_corrected(x, y, z, names):
    # Calculamos los errores estimados
    e_x = x - (x + y + z)/3  # Una aproximación simple
    e_y = y - (x + y + z)/3
    e_z = z - (x + y + z)/3
    
    # Calculamos directamente las correlaciones entre los errores estimados
    corr_xy = np.corrcoef(e_x, e_y)[0,1]
    corr_xz = np.corrcoef(e_x, e_z)[0,1]
    corr_yz = np.corrcoef(e_y, e_z)[0,1]
    
    error_correlations = {
        f"{names[0]}-{names[1]}": corr_xy,
        f"{names[0]}-{names[2]}": corr_xz,
        f"{names[1]}-{names[2]}": corr_yz
    }
    
    return error_correlations

def triple_collocation(x: np.ndarray, y: np.ndarray, z: np.ndarray, snr_thresholds: Tuple[float, float] = (15, 10), names: Tuple[str, str, str] = ('x', 'y', 'z') ) -> TCMetrics:
    """
    Implementa el análisis Triple Collocation calculando métricas clave de error y calidad.
    
    INPUTS:
    ----------
    x, y, z : np.ndarray
        Arrays 1D de igual longitud conteniendo las mediciones colocadas
    snr_thresholds : Tuple[float, float]
        Umbrales de SNR (dB) para flags de calidad (alto, medio)
    names : Tuple[str, str, str]
        Nombres identificadores para cada conjunto de datos
        
    OUTPUTS:
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
    
    # 2. Estimación de la varianza de la señal verdadera tratando 'x' como referencia
    signal_variance = cov_xy * cov_xz / cov_yz
    
    # 3. Cálculo de varianzas de error - métrica fundamental del TC.
    #    Será la misma independientemente de cuál se use como referencia para rotar el análisis.
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
    error_correlations = calculate_error_correlations_corrected (x, y, z, names)
    # Verificar correlaciones directas entre series
    print("Correlación directa X-Y:", np.corrcoef(x, y)[0,1])
    print("Correlación directa X-Z:", np.corrcoef(x, z)[0,1])
    print("Correlación directa Y-Z:", np.corrcoef(y, z)[0,1])

    
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


def ds_sel_coords(ds, lat_indices, lon_indices):
    """
    Function to extract the time series of specific lats and lon vectors, which are given by the 
    lists of the indexes with the positions of the desired lats and lons. The function accounts 
    for efficiency 
    INPUTS:
    - bd_in       : str path to all the files for the membeers of that model. 
    - lat_indices : List with the indexes of latitudes selected. 
    - lon_indices : List with the indexes of longitudes selected. 

    OUTPUTS:
    - serie_ds : numpy 3D array with all the time series and where each pixel correspond to the lats 
                and lons positon given in the lists. The dimensions are : (time, lat, lon)
    - lat_ds   : numpy 1D array  with all the selected latitudinal coordinates. 
    - lon_ds   : numpy 1D array  with all the selected longitudinal coordinates. 
    """
    serie_ds = ds['wsa100m'].isel(lat=lat_indices, lon=lon_indices, drop=True).compute() # Extracción vectorizada de los datos en la serie de tiempo, NO Cartesian product = NO LAS COMBINA,  sin combinarlos
    lon_ds   = ds.isel(lat=lat_indices, lon=lon_indices, drop=True).lon.values
    lat_ds   = ds.isel(lat=lat_indices, lon=lon_indices, drop=True).lat.values
   
    return serie_ds, lon_ds, lat_ds

def extract_tc_metrics(results_tc_by_category):
    """
    Extrae los resultados de TC metrics del diccionario anidado.
    
    Parameters:
    -----------
    results_tc_by_category : dict
        Diccionario con estructura {cat: {(i,j): TCMetrics}}
    
    Returns:
    --------
    dict
        Diccionario con los resultados organizados por categoría y métrica
    """
    # Diccionario para almacenar resultados por categoría
    results_by_category = {}
    
    for cat, tc_results in results_tc_by_category.items():
        # Obtener las claves de la primera TCMetrics para inicializar las estructuras
        first_metric = next(iter(tc_results.values()))
        
        # Inicializar diccionarios para cada métrica
        error_variances = {model: [] for model in first_metric.error_variances.keys()}
        error_correlations = {corr: [] for corr in first_metric.error_correlations.keys()}
        snr_db = {model: [] for model in first_metric.snr_db.keys()}
        signal_variance = []
        coordinates = []
        
        # Extraer resultados para cada punto
        for coords, metrics in tc_results.items():
            # Guardar coordenadas
            coordinates.append(coords)
            
            # Error variances
            for model in error_variances.keys():
                error_variances[model].append(metrics.error_variances[model])
            
            # Error correlations
            for corr in error_correlations.keys():
                error_correlations[corr].append(metrics.error_correlations[corr])
            
            # SNR
            for model in snr_db.keys():
                snr_db[model].append(metrics.snr_db[model])
            
            # Signal variance
            signal_variance.append(metrics.signal_variance)
        
        # Convertir listas a arrays numpy
        results_by_category[cat] = {
            'error_variances': {model: np.array(values) for model, values in error_variances.items()},
            'error_correlations': {corr: np.array(values) for corr, values in error_correlations.items()},
            'snr_db': {model: np.array(values) for model, values in snr_db.items()},
            'signal_variance': np.array(signal_variance),
            'coordinates': np.array(coordinates)
        }
    
    return results_by_category

def df_tc_results_adjustment_spat_cats(results_tc_by_category, fl_cats):
    """
    Function to extract the Triple Collocation (TC) analysis results form the ductionary
    and setting a dataframe with the results of each spatial category for later analysis.
    The function considers the actegories given by: climate, roughness, and slope_variance
    combination. 
    
    The function calls other functions such as 'extract_tc_metrics' and 'decompose_spatial_category'

    INPUTS :
    - results_tc_by_category : nested dictionay with the results of the TC. 
    - fl_cats                : numpy 1D array qith the 3digits integers ID for the spatial 
                               categories.

    OUTPUTS :
    - df_tc                  : pandas dataframe with the tc results per spatial category and with
                               the caregories decomposition. 
    """
    results_tc = extract_tc_metrics(results_tc_by_category )
    df_tc      = pd.DataFrame(fl_cats, columns=["SpatialCat"])

    # Add decomposition columns to the DataFrame
    df_tc['climate']        = df_tc['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['climate']['description'])
    df_tc['roughness']      = df_tc['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['roughness']['description'])
    df_tc['slope_variance'] = df_tc['SpatialCat'].apply(lambda x: decompose_spatial_category(x)['slope_variance']['description']) 

    error_var = []
    sign_var  = []
    snr_db    = []
    error_cor = []
    for k in range(len(df_tc.SpatialCat.values)):
        cat = df_tc.SpatialCat.values [k]
        error_var.append(results_tc[cat]['error_variances'])
        sign_var.append(results_tc[cat]['signal_variance'])
        snr_db.append(results_tc[cat]['snr_db'])
        error_cor.append(results_tc[cat]['error_correlations'])

    df_tc["error_var"] = error_var
    df_tc["sign_var"]  = sign_var
    df_tc["snr_db"]    = snr_db
    df_tc["error_cor"] = error_cor

    return df_tc


########################################################################################
##-----------ABRIENDO EL DATAFRAME DE LAS CATEGORIAS COMBINADAS CORTADAS--------------##
########################################################################################

# Definir las coordenadas de recorte
crop_coords = {'lon_min': 0.5,     # Límite occidental   : 0.5°E
               'lat_min': 40.2,    # Límite meridional   : 40.2°N
               'lon_max': 16.3,    # Límite oriental     : 16.3°E 
               'lat_max': 49.7  }  # Límite septentrional: 49.6°N


comblay              = rasterio.open(bd_in_raster+"SEA-LANDCropped_Combined_RIX_remCPM_WGS84.tif")
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

if np.issubdtype(valid_values.dtype, np.integer): # Asegurarse de que los valores son enteros para np.bincount
    counts = np.bincount(valid_values.astype(int))
    for val in unique_vals:
        pixel_counts[val] = counts[int(val)] if int(val) < len(counts) else 0
else:
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

########################################################################################
##--------------SELECTING N RANDOM COORDINATES FROM EACH SPATIAL CATEGORY-------------##
########################################################################################

fl_cats  = df_filt['value'].values.astype(int)

ds_eth  = load_ds(bd_in_eth)
ds_cnrm = load_ds(bd_in_cnrm)
ds_cmcc = load_ds(bd_in_cmcc)

coords_dict = {}
for i in range(len(fl_cats)):
    cat         = fl_cats[i] 
    idx, idy    = np.where(band1 == cat)  # Devuelve (fila, columna)  OJO, con el ajustillo para los exattars
    indices     = random.sample(range(len(idx)), N_pixels) # Seleccionar aleatoriamente N_pixels índices, len(idx) es == len(idy)
    sampled_idx = idx[indices]
    sampled_idy = idy[indices]  

    # # Verificación adicional (opcional)
    # for er in range(len(sampled_idx)):
    #     assert band1[sampled_idx[er], sampled_idy[er]] == cat, \
    #         f"Error: El punto ({sampled_idx[er]}, {sampled_idy[er]}) no pertenece a la categoría {cat}"

    coords_dict[cat] = {'idx': sampled_idx,
                        'idy': sampled_idy,
                        'N_pixels': N_pixels}

# # Guardar las coordenadas
# with open(bd_out_tc+'Random_coords_N100.pkl', 'wb') as f:
#     pickle.dump(coords_dict, f)

# Abrir las coordenadas
with open(bd_out_tc + 'Random_coords_N100.pkl', 'rb') as f:
    coords_dict = pickle.load(f)

########################################################################################
##-------------------------MAPPING N RANDOM PIXELS SELECTION--------------------------##
########################################################################################

water_color     = '#345F77'  # Azul petróleo oscuro con toque grisáceo para el agua
band1_transform = comblay.transform
band1_crs       = comblay.crs
west            = band1_transform[2]  # x_0
east            = west + band1_transform[0] * comblay.width  # x_0 + width * pixel_width
north           = band1_transform[5]  # y_0
south           = north + band1_transform[4] * comblay.height  # y_0 + height * pixel_height
if north < south:
    north, south = south, north

# Crear colormap personalizado agregando el color del agua
colors = []
if 1 in unique_vals:
    colors.append(water_color)  # Color para agua    
    # Obtener colores de nipy_spectral para el resto
    nipy_colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_categories))
    for i in range(1, num_categories):
        colors.append(nipy_colors[i])
else:
    nipy_colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_categories))
    colors = nipy_colors
cmap   = ListedColormap(colors)
bounds = np.arange(num_categories + 1)
norm   = BoundaryNorm(bounds, cmap.N)

# Enmascarar valores no datos
nodata            = -3.40282e+38  # Original NoData value from the raster
masked_data       = ma.masked_where(band1 == nodata, band1)
band1_categorized = np.full(band1.shape, np.nan)
mask              = np.isfinite(band1)
for i, val in enumerate(unique_vals):
    band1_categorized[band1 == val] = i

fig  = plt.figure(figsize=(10, 6))
gs   = fig.add_gridspec(1, 1)
ax1  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
img1 = ax1.imshow(band1_categorized, extent=[west, east, south, north], origin="upper",  cmap=cmap, norm=norm, alpha=0.7, transform=ccrs.PlateCarree())
ax1.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.set_title('Spatial categories', fontsize=12, fontweight="bold")
cbar = plt.colorbar(img1, ax=ax1, shrink=0.9, ticks=bounds[:-1] + 0.5)
cbar.set_ticks([])
cbar.set_label("Spatial category (Total :" + str(num_categories) + ")", fontsize=10)

# Usar directamente la transformación geoespacial en lugar de cálculos manuales de rasterior para extraer las coordenadas
all_lons       = []
all_lats       = []
all_categories = []

# Verificar antes de plotear
valid_count   = 0
invalid_count = 0

for i, cat in enumerate(fl_cats):
    if cat in coords_dict:
        # Obtener los índices
        sampled_idx = coords_dict[cat]['idx']
        sampled_idy = coords_dict[cat]['idy']
        
        # Convertir índices a coordenadas geográficas usando la transformación exacta de rasterio
        xs, ys = [], []
        for j in range(len(sampled_idx)):
            row, col = sampled_idx[j], sampled_idy[j]
            
            # Verificar que el punto realmente tiene el valor de categoría esperado
            if band1[row, col] != cat:
                print(f"Error: Punto ({row}, {col}) esperaba cat={cat}, pero tiene valor={band1[row, col]}")
                invalid_count += 1
                continue
                
            # Usar la transformación exacta de rasterio para convertir fila/columna a coordenadas
            x, y = band1_transform * (col + 0.5, row + 0.5)
            xs.append(x)
            ys.append(y)
            
            # Añadir a los arrays generales
            all_lons.append(x)
            all_lats.append(y)
            all_categories.append(cat)
            valid_count += 1
        
        # Plotear los puntos de esta categoría
        ax1.scatter(xs, ys, c="black", s=15, alpha=0.8, transform=ccrs.PlateCarree(), edgecolors='white', linewidths=0.5)

print(f"Puntos válidos: {valid_count}, Puntos inválidos: {invalid_count}")
ax1.text(0.5, 1.05, f'Total: {len(all_lons)} random points from {len(coords_dict)} categories',  transform=ax1.transAxes, ha='center', fontsize=14)

plt.tight_layout()
# plt.savefig(bd_out_fig+'Map_RandomPixelsSelection_SpatialCategories.png', dpi=300, bbox_inches='tight')
plt.show()

########################################################################################
##----EXTRANTING RELEVANT TIME SERIES PROCESSING G-TRIPLE COLLOCATION FOR SELECTED POINTS -------------##
########################################################################################

results_tc_by_category = {}
for cat, coords in coords_dict.items():
    print(f"## Processing category {cat} with {coords['N_pixels']} points")

    series_data = {'time_series' : [],   # Lista para almacenar todas las series
                    'coordinates': [],   # Lista para almacenar coordenadas originales
                    'models'     : ['ETH', 'CNRM', 'CMCC']}
    
    lat_indices = np.array(coords['idx']) # filas
    lon_indices = np.array(coords['idy']) # columns

    serie_ds_eth, lon_ds_eth, lat_ds_eth    = ds_sel_coords(ds_eth, lat_indices, lon_indices)
    serie_ds_cmcc, lon_ds_cmcc, lat_ds_cmcc = ds_sel_coords(ds_cmcc, lat_indices, lon_indices)
    serie_ds_cnrm, lon_ds_cnrm, lat_ds_cnrm = ds_sel_coords(ds_cnrm, lat_indices, lon_indices)
    
    print(f"## Loaded category {cat} with {coords['N_pixels']} points")

    results_tc_category = {} ## Diccionario con los resultados de tc para cada punto
    for k in range(N_pixels):
        la = lat_ds_eth[k]
        lo = lon_ds_eth[k]

        # Encontrar índices exactos para la latitud y longitud actuales - SOLO UN PUNTO
        idx_lat = np.where(serie_ds_eth.lat.values == la)[0][0]
        idx_lon = np.where(serie_ds_eth.lon.values == lo)[0][0]

        serie_eth  = serie_ds_eth.isel(lat=idx_lat, lon=idx_lon).values
        serie_cmcc = serie_ds_cmcc.isel(lat=idx_lat, lon=idx_lon).values
        serie_cnrm = serie_ds_cnrm.isel(lat=idx_lat, lon=idx_lon).values

        series = np.array([serie_eth, serie_cnrm, serie_cmcc])
        series_data['time_series'].append(series)
        series_data['coordinates'].append([lat_indices[k], lon_indices[k]])
            
        # Calcular la triple colocacion
        results_tc_category [(k)] = triple_collocation(x=serie_eth, y=serie_cnrm, z=serie_cmcc, names=('CPM1:ETH', 'CPM2:CNRM', 'CPM3:CMCC'))   ## Se agregan en el i, j NO SON COMBINATORIA
        
    # Convertir a arrays numpy y guardarlas para estimar los RL y TC rotado
    series_data['time_series'] = np.array(series_data['time_series'])
    series_data['coordinates'] = np.array(series_data['coordinates'])
    np.savez( f"{bd_out_tc}_TS_cat_{cat}.npz", time_series=series_data['time_series'], coordinates=series_data['coordinates'], models=series_data['models'])

    results_tc_by_category[cat] = results_tc_category  ## Un diccionario anidado con los resultados de tc por categoria.
    print(f"## Finished category {cat} with {coords['N_pixels']} points")
     

ds_eth.close() 
ds_cmcc.close()
ds_cnrm.close()


# Guardar los resultados
with open(bd_out_tc+'Results_GenTC_by_category_N100.pkl', 'wb') as f:
    pickle.dump(results_tc_by_category, f)

########################################################################################
###--EXTRACTING METRICS FOR GENERATING THE TC PLOT OVER SELECTED PIXELS PER CATEGORY-###
########################################################################################

# # Para cargarlos más tarde
# with open(bd_out_tc+'Results_GenTC_by_category_N100.pkl', 'rb') as f:
#     results_tc_by_category = pickle.load(f)

df = df_tc_results_adjustment_spat_cats(results_tc_by_category, fl_cats)

########################################################################################
###-PLOTTING SELECTED PIXELS TC RESULTS TO METRICS AS CATEGORIES AND LEVELS FUNCTION-###
########################################################################################

# Definir categorías para cada columna
categories = ['climate', 'roughness', 'slope_variance']
cat_values = {'climate'       : ['Arid', 'Temperate', 'Cold', 'Polar'],
              'roughness'     : ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
              'slope_variance': ['Very Low', 'Low', 'Moderate', 'High', 'Very High']}

# Títulos para las filas
row_titles = ['Signal\n Variance [$(m/s)^2$]', 'Error\n Variance [$(m/s)^2$]', 'SNR [$dB$]', 'Error\n Correlations']

colours    = {"CPM1:ETH": "#edae49",   # Amarillo
            "CPM2:CNRM" : "#00798c",   # Azul
            "CPM3:CMCC" : "#d1495b",}  # Rojo
width      = 0.25


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# Iterar sobre las columnas (categorías)
for j, category in enumerate(categories):
    # Obtener valores únicos ordenados para esta categoría
    values    = cat_values[category]
    positions = np.arange(len(values))
    
    # Primera fila: Signal Variance
    for i, val in enumerate(values):
        mask = df[category] == val
        if mask.any():
            data = []
            for row in df[mask]['sign_var']:
                data.extend([x for x in row if not np.isnan(x)])
            axes[0,j].boxplot([data],
                            positions   = [i],
                            patch_artist= True,
                            medianprops = dict(color="black"),
                            boxprops    = dict(facecolor='purple', alpha=0.6),
                            widths      = width)
    
    # Segunda fila: Error Variance
    for k, (model, color) in enumerate(colours.items()):
        for i, val in enumerate(values):
            mask = df[category] == val
            if mask.any():
                data = []
                for row in df[mask]['error_var']: 
                    if isinstance(row, dict) and model in row:   ## Para garatizar que estamos trabajando con un diccionario (y con el adecuado)
                        data.extend([x for x in row[model] if not np.isnan(x)])
                if data:
                    pos = i + (k-1)*width
                    axes[1,j].boxplot([data],
                                    positions   = [pos],
                                    patch_artist= True,
                                    medianprops = dict(color="black"),
                                    boxprops    = dict(facecolor=color, alpha=0.6),
                                    widths      = width)
    
    # Tercera fila: SNR
    for k, (model, color) in enumerate(colours.items()):
        for i, val in enumerate(values):
            mask = df[category] == val
            if mask.any():
                data = []
                for row in df[mask]['snr_db']:
                    if isinstance(row, dict) and model in row:   ## Para garatizar que estamos trabajando con un diccionario (y con el adecuado)
                        data.extend([x for x in row[model] if not np.isnan(x)])
                if data:
                    pos = i + (k-1)*width
                    axes[2,j].boxplot([data],
                                    positions   = [pos],
                                    patch_artist= True,
                                    medianprops = dict(color="black"),
                                    boxprops    = dict(facecolor=color, alpha=0.6),
                                    widths      = width)
    
    # Configurar el formato de cada subplot
    for i in range(3):
        ax = axes[i,j]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.set_xticks(positions)
        ax.tick_params(axis='y', labelsize=12)
        if i == 2:  # Solo mostrar etiquetas en la última fila
            ax.set_xticklabels(values, fontsize=13, rotation=0)
        else:
            ax.set_xticklabels([])
        ax.set_xlim(-0.5, len(values)-0.5)
        if j == 0:
            ax.set_ylabel(row_titles[i], fontsize=15)
        if i == 0:
            ax.set_title(category.replace('_', ' ').title(), fontsize=15, pad=10)
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6, label=model) for model, color in colours.items()]
fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.999, 0.5), title="Models")

plt.suptitle("Triple Collocation Metrics Distribution by Categories", fontsize=16, fontweight="bold", y=0.95)
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, hspace=0.19, left=0.09, right=0.9, bottom=0.10, top=0.89)
plt.savefig(bd_out_fig+"TC_Relative_CPMs_SinErrorCor_FinalSelect_SpaCat-Levels.png", format='png', dpi=300, transparent=True)
plt.show()
