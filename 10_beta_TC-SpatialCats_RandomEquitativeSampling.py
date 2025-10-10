
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import os,glob,sys
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import seaborn as sns
import scipy.linalg as la
from typing import Dict, Tuple, List, NamedTuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import itertools
from collections import defaultdict



"""
Code to compute the  Triple Collocation from all those samples, in order to compare those
performance metrics across categories.

Spatial categories whose relative frequency exceeded the first quartile (25th percentile) 
selected in the precedant script. Series pixels follow a startified equivallent sandom sampling.

It also conducts and visualizes the verification of the TC assumptions. 

The updated routine uses the cropped CORDEX domain. 

Author : Nathalia Correa-Sánchez
"""


########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

bd_in_ws     = "/Dati/Data/WS_CORDEXFPS/"
bd_out_fig   = "/Dati/Outputs/Plots/WP3_development/"
bd_out_int   = "/Dati/Data/WS_CORDEXFPS/Intermediates/"
bd_out_tc    = "/Dati/Outputs/Triple_Collocation/" 
bd_in_eth    = bd_in_ws + "ETH/wsa100m_crop/"
bd_in_cmcc   = bd_in_ws + "CMCC/wsa100m_crop/"
bd_in_cnrm   = bd_in_ws + "CNRM/wsa100m_crop/"
bd_in_raster = "/Dati/Outputs/Climate_Provinces/Development_Rasters/band1_cropped.tif" # Antes : Combined_RIX_remCPM_WGS84.tif
file_cats    = "/Dati/Outputs/Climate_Provinces/CSVs/Combination_RIX.csv"

########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

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

def calculate_error_correlations_mccoll(x, y, z, signal_variance, error_vars, names):
    return {
        f"{names[0]}-{names[1]}": (np.cov(x, y)[0,1] - signal_variance) / 
                                  np.sqrt(error_vars[names[0]] * error_vars[names[1]]),
        f"{names[0]}-{names[2]}": (np.cov(x, z)[0,1] - signal_variance) / 
                                  np.sqrt(error_vars[names[0]] * error_vars[names[2]]),
        f"{names[1]}-{names[2]}": (np.cov(y, z)[0,1] - signal_variance) / 
                                  np.sqrt(error_vars[names[1]] * error_vars[names[2]])
    }

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
    # error_correlations = calculate_error_correlations_corrected (x, y, z, names) # Mejor Mc Coll que deriva las correlaciones de error directamente del modelo matemático de TC
    error_correlations = calculate_error_correlations_mccoll(x, y, z, signal_variance, error_variances, names)
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

def df_tc_results_adjustment_spat_cats(results_tc, fl_cats):
    # Crear listas para almacenar los datos
    all_categories               = []
    all_climate                  = []
    all_roughness                = []
    all_slope_variance           = []
    all_signal_variance          = []
    all_signal_variance_original = []  # Nuevo para los valores transformados de vuelta
    all_error_var                = []
    all_error_var_original       = []  # Nuevo para los valores transformados de vuelta
    all_snr_db                   = []
    all_error_cor                = []

    # Iterar a través de las categorías y extraer los datos
    for cat in fl_cats:
        # Extraer los dígitos de la categoría
        climate_level   = int(str(cat)[0])
        roughness_level = int(str(cat)[1])
        slope_level     = int(str(cat)[2])
        
        # Mapear a nombres de categoría más descriptivos
        climate_name   = ['Arid', 'Temperate', 'Cold', 'Polar'][climate_level-1]
        roughness_name = ['Very Low', 'Low', 'Moderate', 'High', 'Very High'][roughness_level-1]
        slope_name     = ['Very Low', 'Low', 'Moderate', 'High', 'Very High'][slope_level-1]
        
        # Extraer resultados para esta categoría
        results_cat = results_tc[cat]
        
        for k in results_cat.keys():
            # Añadir a las listas
            all_categories.append(cat)
            all_climate.append(climate_name)
            all_roughness.append(roughness_name)
            all_slope_variance.append(slope_name)
            
            # Varianza de señal - tanto original como transformada
            try:
                all_signal_variance.append([results_cat[k].signal_variance])
                # Usar el valor transformado si existe, sino usar NaN
                signal_var_original = getattr(results_cat[k], 'signal_variance_original', np.nan)
                all_signal_variance_original.append([signal_var_original])
            except:
                all_signal_variance.append([np.nan])
                all_signal_variance_original.append([np.nan])
            
            # Varianza de error - tanto original como transformada
            try:
                all_error_var.append(results_cat[k].error_variances)
                # Usar valores transformados si existen, sino usar diccionario vacío
                error_var_original = getattr(results_cat[k], 'error_variances_original', {})
                all_error_var_original.append(error_var_original)
            except:
                all_error_var.append({})
                all_error_var_original.append({})
            
            # SNR (no necesita transformación)
            try:
                all_snr_db.append(results_cat[k].snr_db)
            except:
                all_snr_db.append({})
            
            # Correlaciones de error (no necesitan transformación)
            try:
                all_error_cor.append(results_cat[k].error_correlations)
            except:
                all_error_cor.append({})
    
    # Crear un DataFrame con los datos extraídos
    df = pd.DataFrame({
        'category'          : all_categories,
        'climate'           : all_climate,
        'roughness'         : all_roughness,
        'slope_variance'    : all_slope_variance,
        'sign_var'          : all_signal_variance,
        'sign_var_original' : all_signal_variance_original,  # Nueva columna
        'error_var'         : all_error_var,
        'error_var_original': all_error_var_original,       # Nueva columna
        'snr_db'            : all_snr_db,
        'error_cor'         : all_error_cor
    })
    
    return df

########################################################################################
##-----ABRIENDO EL CROPPED RASTER PARA EXTRAER CADA CLASE & AJUSTANDO EL DATAFRAME----##
########################################################################################

comblay              = rasterio.open(bd_in_raster)
band1_o              = comblay.read(1) ## Solo tiene una banda
band1_o[band1_o < 0] = np.nan          ## Reemplazando los negativos con nan o 0(Tener en cuenta NoData= -3.40282e+38) 
band1                = np.delete(np.delete(band1_o, filas_eliminar[0], axis=0), columnas_eliminar[0], axis=1) ## Ajustillo para los xarrays

# Obteniendo valores unicos de las categorias
unique_vals    = np.unique(band1_o)
unique_vals    = unique_vals[np.isfinite(unique_vals)]
num_categories = len(unique_vals)

# Contar píxeles para cada valor único
pixel_counts = {}
mask         = np.isfinite(band1_o)
valid_values = band1_o[mask]

if np.issubdtype(valid_values.dtype, np.integer): # Asegurarse de que los valores son enteros para np.bincount
    counts = np.bincount(valid_values.astype(int))
    for val in unique_vals:
        pixel_counts[val] = counts[int(val)] if int(val) < len(counts) else 0
else:
    for val in unique_vals:
        pixel_counts[val] = np.sum(valid_values == val)
counts_array = np.array([pixel_counts[val] for val in unique_vals])
df_cats      = pd.DataFrame({'value': unique_vals, 'count': counts_array,})

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

########################################################################################
##--------TC ROTATING THE REFERENCE FROM THE THREE MODELS FOR DAILY WIND GUSTS--------##
########################################################################################

full_range  = pd.date_range(start='2000-01-01 00:00:00', end='2009-12-31 23:50:00', freq='1H')      

results_tc_rot_by_category_eth  = {}
results_tc_rot_by_category_cnrm = {}
results_tc_rot_by_category_cmcc = {}

for cat in fl_cats:
    print(f"## Processing category {cat} TC rotated")
    
    data_npz    = np.load(f"{bd_out_tc}_TS_cat_{cat}.npz")
    time_series = data_npz['time_series']
    coordinates = data_npz['coordinates']
    models      = data_npz['models']

    results_tc_rotated_eth  = {} ## Diccionario con los resultados de tc para cada punto
    results_tc_rotated_cnrm = {} ## Diccionario con los resultados de tc para cada punto
    results_tc_rotated_cmcc = {} ## Diccionario con los resultados de tc para cada punto
    
    for k in range(N_pixels):
        serie_eth  = time_series[k, 0, :]        
        serie_cnrm = time_series[k, 1, :]
        serie_cmcc = time_series[k, 2, :]

        df_s_eth  = pd.DataFrame(serie_eth, index = full_range, columns=["WS_h"])
        df_s_cnrm = pd.DataFrame(serie_cnrm, index = full_range, columns=["WS_h"])
        df_s_cmcc = pd.DataFrame(serie_cmcc, index = full_range, columns=["WS_h"])

        # Obtener time series de daily wind gust
        wg_s_eth  = df_s_eth.groupby(df_s_eth.index.date)["WS_h"].max().values
        wg_s_cnrm = df_s_cnrm.groupby(df_s_cnrm.index.date)["WS_h"].max().values
        wg_s_cmcc = df_s_cmcc.groupby(df_s_cmcc.index.date)["WS_h"].max().values
        
        # Aplicar transformación logarítmica para estabilizar la varianza
        log_wg_s_eth  = np.log1p(wg_s_eth)   # log1p es ln(x+1)
        log_wg_s_cnrm = np.log1p(wg_s_cnrm)
        log_wg_s_cmcc = np.log1p(wg_s_cmcc)
            
        # Calcular la triple colocacion rotada con datos transformados
        results_tc_rotated_eth [(k)]  = triple_collocation(x=log_wg_s_eth, y=log_wg_s_cnrm, z=log_wg_s_cmcc, names=('CPM1:ETH', 'CPM2:CNRM', 'CPM3:CMCC'))
        results_tc_rotated_cnrm [(k)] = triple_collocation(x=log_wg_s_cnrm, y=log_wg_s_cmcc, z=log_wg_s_eth, names=('CPM2:CNRM', 'CPM3:CMCC', 'CPM1:ETH'))
        results_tc_rotated_cmcc [(k)] = triple_collocation(x=log_wg_s_cmcc, y=log_wg_s_eth, z=log_wg_s_cnrm, names=('CPM3:CMCC', 'CPM1:ETH', 'CPM2:CNRM'))

        # Calcular medias de series logarítmicas para transformación inversa
        mean_log_values = np.mean([log_wg_s_eth, log_wg_s_cnrm, log_wg_s_cmcc])
        exp_factor      = np.exp(2 * mean_log_values)
        
        # TRANSFORMACIÓN INVERSA PARA ETH COMO REFERENCIA
        # 1. Transformar varianza de señal de vuelta a escala original
        signal_var_log      = results_tc_rotated_eth[k].signal_variance
        signal_var_original = exp_factor * (np.exp(signal_var_log) - 1)
        
        # Crear un atributo nuevo para almacenar el valor transformado
        # Usamos setattr para evitar modificar la clase TCMetrics
        setattr(results_tc_rotated_eth[k], 'signal_variance_original', signal_var_original)
        
        # 2. Transformar varianzas de error de vuelta a escala original
        error_var_original = {}
        for model in ['CPM1:ETH', 'CPM2:CNRM', 'CPM3:CMCC']:
            error_var_log             = results_tc_rotated_eth[k].error_variances[model]
            error_var_original[model] = exp_factor * (np.exp(error_var_log) - 1)
        
        # Almacenar las varianzas de error transformadas
        setattr(results_tc_rotated_eth[k], 'error_variances_original', error_var_original)
        
        # 3. Transformar SNR de vuelta (manteniendo la escala dB)
        # El SNR en dB no necesita transformación inversa ya que es una relación logarítmica
        
        # REPETIR PARA CNRM COMO REFERENCIA
        signal_var_log      = results_tc_rotated_cnrm[k].signal_variance
        signal_var_original = exp_factor * (np.exp(signal_var_log) - 1)
        setattr(results_tc_rotated_cnrm[k], 'signal_variance_original', signal_var_original)
        
        error_var_original = {}
        for model in ['CPM2:CNRM', 'CPM3:CMCC', 'CPM1:ETH']:
            error_var_log             = results_tc_rotated_cnrm[k].error_variances[model]
            error_var_original[model] = exp_factor * (np.exp(error_var_log) - 1)
        setattr(results_tc_rotated_cnrm[k], 'error_variances_original', error_var_original)
        
        # REPETIR PARA CMCC COMO REFERENCIA
        signal_var_log      = results_tc_rotated_cmcc[k].signal_variance
        signal_var_original = exp_factor * (np.exp(signal_var_log) - 1)
        setattr(results_tc_rotated_cmcc[k], 'signal_variance_original', signal_var_original)
        
        error_var_original = {}
        for model in ['CPM3:CMCC', 'CPM1:ETH', 'CPM2:CNRM']:
            error_var_log             = results_tc_rotated_cmcc[k].error_variances[model]
            error_var_original[model] = exp_factor * (np.exp(error_var_log) - 1)
        setattr(results_tc_rotated_cmcc[k], 'error_variances_original', error_var_original)

    results_tc_rot_by_category_eth[cat]  = results_tc_rotated_eth   ## Un diccionario anidado con los resultados de tc por categoria.
    results_tc_rot_by_category_cnrm[cat] = results_tc_rotated_cnrm  ## Un diccionario anidado con los resultados de tc por categoria.
    results_tc_rot_by_category_cmcc[cat] = results_tc_rotated_cmcc  ## Un diccionario anidado con los resultados de tc por categoria.

    print(f"## Finished TC-Rotated category {cat} with {str(N_pixels)} points")

########################################################################################
###--EXTRACTING METRICS FOR GENERATING THE TC PLOT OVER SELECTED PIXELS PER CATEGORY-###
########################################################################################

df_tc_rot_eth  = df_tc_results_adjustment_spat_cats(results_tc_rot_by_category_eth, fl_cats)
df_tc_rot_cnrm = df_tc_results_adjustment_spat_cats(results_tc_rot_by_category_cnrm, fl_cats)
df_tc_rot_cmcc = df_tc_results_adjustment_spat_cats(results_tc_rot_by_category_cmcc, fl_cats)

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

models_key = ["CPM1:ETH", "CPM2:CNRM",  "CPM3:CMCC" ]
colours    = {"CPM1:ETH": "#edae49",     # Amarillo           ## Para los modelos, no las diferencias
              "CPM2:CNRM" : "#00798c",   # Azul
              "CPM3:CMCC" : "#d1495b",}  # Rojo
dfs_tc_key  = [df_tc_rot_eth, df_tc_rot_cnrm, df_tc_rot_cmcc] 

diff_colours_eth  = {'CPM1:ETH-CPM2:CNRM' : "#8338ec",  # Púrpura  ## Para las diferencias, no los modelos
                     'CPM1:ETH-CPM3:CMCC' : "#3a86ff",  # Azul celeste               
                     'CPM2:CNRM-CPM3:CMCC': "#06d6a0"}  # Turquesa brillante
diff_colours_cnrm = {'CPM2:CNRM-CPM3:CMCC': "#06d6a0",  # Turquesa brillante
                    'CPM2:CNRM-CPM1:ETH' : "#8338ec",   # Púrpura           
                    'CPM3:CMCC-CPM1:ETH' : "#3a86ff"}   # Azul celeste
diff_colours_cmcc = {'CPM3:CMCC-CPM1:ETH' : "#3a86ff",  # Azul celeste  
                    'CPM3:CMCC-CPM2:CNRM': "#06d6a0",   # Turquesa brillante         
                    'CPM1:ETH-CPM2:CNRM' : "#8338ec"}   # Púrpura                       

list_dif_colours= [diff_colours_eth, diff_colours_cnrm, diff_colours_cmcc]

for m in range (len(models_key)):
    ref_model    = models_key[m]
    df_ref       = dfs_tc_key [m]
    diff_colours = list_dif_colours [m]

    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    # Iterar sobre las columnas (categorías)
    for j, category in enumerate(categories):
        # Obtener valores únicos ordenados para esta categoría
        values    = cat_values[category]
        positions = np.arange(len(values))
        
        # Primera fila: Signal Variance
        for i, val in enumerate(values):
            mask = df_ref[category] == val
            if mask.any():
                data = []
                for row in df_ref[mask]['sign_var_original']:
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
                mask = df_ref[category] == val
                if mask.any():
                    data = []
                    for row in df_ref[mask]['error_var_original']: 
                        if isinstance(row, dict) and model in row:   ## Para garatizar que estamos trabajando con un diccionario (y con el adecuado)
                            if not np.isnan(row[model]):
                                data.append(row[model])
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
                mask = df_ref[category] == val
                if mask.any():
                    data = []
                    for row in df_ref[mask]['snr_db']:
                        if isinstance(row, dict) and model in row:   ## Para garatizar que estamos trabajando con un diccionario (y con el adecuado)
                            if not np.isnan(row[model]):
                                data.append(row[model])
                    if data:
                        pos = i + (k-1)*width
                        axes[2,j].boxplot([data],
                                        positions   = [pos],
                                        patch_artist= True,
                                        medianprops = dict(color="black"),
                                        boxprops    = dict(facecolor=color, alpha=0.6),
                                        widths      = width)

        # Cuarta fila: Error correlations
        for k, (diff_pair, color_d) in enumerate(diff_colours.items()):
            for i, val in enumerate(values):
                mask = df_ref[category] == val
                if mask.any():
                    data = []
                    for row in df_ref[mask]['error_cor']:
                        if isinstance(row, dict) and diff_pair in row:   ## Para garatizar que estamos trabajando con un diccionario (y con el adecuado)
                            if not np.isnan(row[diff_pair]):
                                data.append(row[diff_pair])
                    if data:
                        pos = i + (k-1)*width
                        axes[3,j].boxplot([data],
                                        positions   = [pos],
                                        patch_artist= True,
                                        medianprops = dict(color="black"),
                                        boxprops    = dict(facecolor=color_d, alpha=0.6),
                                        widths      = width)
        
        # Configurar el formato de cada subplot
        for i in range(4):
            ax = axes[i,j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.set_xticks(positions)
            ax.tick_params(axis='y', labelsize=11)
            if i == 3:  # Solo mostrar etiquetas en la última fila
                ax.set_xticklabels(values, fontsize=13, rotation=0)
            else:
                ax.set_xticklabels([])
            ax.set_xlim(-0.5, len(values)-0.5)
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=14)
            if i == 0:
                ax.set_title(category.replace('_', ' ').title(), fontsize=14, pad=10)
    
    # Leyenda para los modelos (filas 2 y 3)
    model_legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6, label=model) for model, color in colours.items()]
    
    # Leyenda para las diferencias (fila 4)
    diff_legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6, label=diff_pair) for diff_pair, color in diff_colours.items()]
    
    # Posicionamos ambas leyendas
    first_legend = fig.legend(handles=model_legend_elements, loc='center right', bbox_to_anchor=(0.999, 0.62), title="Models", fontsize=10)
    fig.add_artist(first_legend)
    fig.legend(handles=diff_legend_elements, loc='center right', bbox_to_anchor=(1.01, 0.19), title="Model Differences", fontsize=9)

    plt.suptitle("Triple Collocation Metrics Distribution by Categories \n with " +ref_model[5:]+" as reference", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.13, hspace=0.19, left=0.09, right=0.9, bottom=0.10, top=0.89)
    plt.savefig(bd_out_fig+"TC_Relative_"+ref_model[5:]+"_CPMs_ConErrorCor_FinalSelect_SpaCat-Levels.png", format='png', dpi=300, transparent=True)
    plt.show()


########################################################################################
###--------------VERIFICACIÓN CONDENSADA DE SUPUESTOS DE TRIPLE COLOCACIÓN-----------###
########################################################################################

all_linearity_results        = []
all_homoscedasticity_results = []
all_stationarity_results     = []

# Colores para los diferentes pares de modelos
model_colors = {
    'ETH-CNRM': "#8338ec",   # Púrpura 
    'ETH-CMCC': "#3a86ff",   # Azul celeste 
    'CNRM-CMCC': "#06d6a0"}  # Turquesa brillante


fig = plt.figure(figsize=(20, 15))
gs  = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])

sample_cats  = fl_cats # Usar TODAS las categorías, sin muestreo aleatorio
total_points = 0

# 1. PRIMER PANEL: LINEALIDAD
lineal_data = defaultdict(list)
corr_data   = defaultdict(list)
reset_data  = defaultdict(list)

# 2. SEGUNDO PANEL: HOMOSCEDASTICIDAD
bp_pvalues     = defaultdict(list)
residuals_data = defaultdict(list)
predicted_data = defaultdict(list)

# 3. TERCER PANEL: ESTACIONARIEDAD
adf_pvalues    = defaultdict(list)
means_by_month = defaultdict(lambda: defaultdict(list))

# Iterar por categorías seleccionadas
for cat_idx, cat in enumerate(sample_cats):
    data_npz    = np.load(f"{bd_out_tc}_TS_cat_{cat}.npz")
    time_series = data_npz['time_series']
    coordinates = data_npz['coordinates']
    
    # Tomar 5 puntos de muestra de cada categoría (todas tienen 100 datos)
    n_sample_points = 5
    sample_indices  = np.random.choice(time_series.shape[0], n_sample_points, replace=False)
    
    # Procesar cada punto
    for sample_idx in sample_indices:
        total_points += 1
        
        # Extraer y procesar las series temporales
        serie_eth  = time_series[sample_idx, 0, :]
        serie_cnrm = time_series[sample_idx, 1, :]
        serie_cmcc = time_series[sample_idx, 2, :]
        
        # Convertir a dataframes y obtener máximos diarios
        df_s_eth  = pd.DataFrame(serie_eth, index=full_range, columns=["WS_h"])
        df_s_cnrm = pd.DataFrame(serie_cnrm, index=full_range, columns=["WS_h"])
        df_s_cmcc = pd.DataFrame(serie_cmcc, index=full_range, columns=["WS_h"])
        
        wg_s_eth  = df_s_eth.groupby(df_s_eth.index.date)["WS_h"].max()
        wg_s_cnrm = df_s_cnrm.groupby(df_s_cnrm.index.date)["WS_h"].max()
        wg_s_cmcc = df_s_cmcc.groupby(df_s_cmcc.index.date)["WS_h"].max()
        
        # Aplicar transformación logarítmica
        log_wg_s_eth  = np.log1p(wg_s_eth)
        log_wg_s_cnrm = np.log1p(wg_s_cnrm)
        log_wg_s_cmcc = np.log1p(wg_s_cmcc)
        
        # Dataframe combinado
        df_combined = pd.DataFrame({
            'ETH': log_wg_s_eth.values,
            'CNRM': log_wg_s_cnrm.values,
            'CMCC': log_wg_s_cmcc.values
        }, index=wg_s_eth.index)
        
        # ANÁLISIS DE LINEALIDAD
        for (col1, col2), name in zip([('ETH', 'CNRM'), ('ETH', 'CMCC'), ('CNRM', 'CMCC')],  ['ETH-CNRM', 'ETH-CMCC', 'CNRM-CMCC']):
            # Correlación
            corr = np.corrcoef(df_combined[col1], df_combined[col2])[0,1]
            corr_data[name].append(corr)
            
            # Datos para scatter plot
            lineal_data[name].extend(list(zip(df_combined[col1], df_combined[col2])))
            
            # RESET test
            x, y     = df_combined[col1].values, df_combined[col2].values
            X        = x.reshape(-1, 1)
            model    = LinearRegression().fit(X, y)
            y_pred   = model.predict(X)
            X_nl     = np.column_stack([X, y_pred**2])
            model_nl = LinearRegression().fit(X_nl, y)
            r2_base  = model.score(X, y)
            r2_nl    = model_nl.score(X_nl, y)
            reset_data[name].append(r2_nl - r2_base)
        
        # ANÁLISIS DE HOMOSCEDASTICIDAD
        for (col1, col2), name in zip([('ETH', 'CNRM'), ('ETH', 'CMCC'), ('CNRM', 'CMCC')], ['ETH-CNRM', 'ETH-CMCC', 'CNRM-CMCC']):
            # Regresión lineal
            slope, intercept, _, _, _ = stats.linregress(df_combined[col1], df_combined[col2])
            residuals                 = df_combined[col2] - (slope * df_combined[col1] + intercept)
            predicted                 = slope * df_combined[col1] + intercept
            
            # Guardar algunos residuos y valores predichos (muestreo para evitar sobrecarga)
            sample_indices = np.random.choice(len(residuals), min(10, len(residuals)), replace=False)
            residuals_data[name].extend(residuals.iloc[sample_indices])
            predicted_data[name].extend(predicted.iloc[sample_indices])
            
            # Test de Breusch-Pagan
            x_with_const = sm.add_constant(df_combined[col1])
            model        = sm.OLS(df_combined[col2], x_with_const).fit()
            bp_test      = het_breuschpagan(model.resid, model.model.exog)
            bp_pvalues[name].append(bp_test[1])
        
        # ANÁLISIS DE ESTACIONARIEDAD
        df_combined.index = pd.to_datetime(df_combined.index)
        
        # Test ADF para cada modelo
        for col in ['ETH', 'CNRM', 'CMCC']:
            adf_result = adfuller(df_combined[col].dropna())
            adf_pvalues[col].append(adf_result[1])
        
        # Calcular medias mensuales para cada modelo
        df_combined['month'] = df_combined.index.month
        for col in ['ETH', 'CNRM', 'CMCC']:
            monthly_means = df_combined.groupby('month')[col].mean()
            for month, mean_val in monthly_means.items():
                means_by_month[col][month].append(mean_val)

# CREAR PANELES CONDENSADOS
# 1. PANEL DE LINEALIDAD
ax_lineality = fig.add_subplot(gs[0, :2])
for name, color in model_colors.items():
    data = np.array(lineal_data[name])
    if len(data) > 1000:  # Si hay muchos puntos, tomar una muestra
        indices = np.random.choice(len(data), 1000, replace=False)
        data    = data[indices]
    ax_lineality.scatter(data[:, 0], data[:, 1], alpha=0.3, color=color, label=name)
    
    # Añadir línea de regresión
    if len(data) > 0:
        x, y = data[:, 0], data[:, 1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        ax_lineality.plot(x_line, y_line, color=color, linestyle='-', linewidth=2)

ax_lineality.set_title('Linearity between models (log-transformed data)', fontsize=14)
ax_lineality.set_xlabel('log(WS) Model 1', fontsize=12)
ax_lineality.set_ylabel('log(WS) Model 2', fontsize=12)
ax_lineality.legend()
ax_lineality.grid(True, alpha=0.3)

# Gráfico de correlaciones
ax_corr = fig.add_subplot(gs[0, 2])
for i, name in enumerate(model_colors.keys()):
    ax_corr.boxplot(corr_data[name], positions=[i], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=model_colors[name], alpha=0.6))
ax_corr.set_title('Correlations between models', fontsize=14)
ax_corr.set_ylabel('Correlation Coefficient', fontsize=12)
ax_corr.set_xticks(range(len(model_colors)))
ax_corr.set_xticklabels(list(model_colors.keys()), fontsize=7, rotation=0)
ax_corr.grid(True, alpha=0.3)
ax_corr.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
ax_corr.text(0.5, 0.72, 'Threshold  0.7', color='red', ha='center')

# Gráfico de RESET test
ax_reset = fig.add_subplot(gs[0, 3])
for i, name in enumerate(model_colors.keys()):
    ax_reset.boxplot(reset_data[name], positions=[i], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=model_colors[name], alpha=0.6))
ax_reset.set_title('RESET test', fontsize=14)
ax_reset.set_ylabel('Improvement of R²', fontsize=12)
ax_reset.set_xticks(range(len(model_colors)))
ax_reset.set_xticklabels(list(model_colors.keys()), fontsize=7, rotation=0)
ax_reset.grid(True, alpha=0.3)
ax_reset.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
ax_reset.text(1.5, 0.055, 'Threshold 0.05', color='red', ha='center')

# 2. PANEL DE HOMOSCEDASTICIDAD
ax_homo = fig.add_subplot(gs[1, :2])
for name, color in model_colors.items():
    residuals = residuals_data[name]
    predicted = predicted_data[name]
    if len(residuals) > 0:
        ax_homo.scatter(predicted, residuals, alpha=0.4, color=color, label=name)
ax_homo.set_title('Homoscedasticity: Residuals vs Predicted Values', fontsize=14)
ax_homo.set_xlabel('Predicted Values (log)', fontsize=12)
ax_homo.set_ylabel('Residuals', fontsize=12)
# ax_homo.legend()
ax_homo.grid(True, alpha=0.3)
ax_homo.axhline(y=0, color='black', linestyle='--')

# Gráfico de p-valores Breusch-Pagan
ax_bp = fig.add_subplot(gs[1, 2:])
for i, name in enumerate(model_colors.keys()):
    ax_bp.boxplot(bp_pvalues[name], positions=[i], widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor=model_colors[name], alpha=0.6))
ax_bp.set_title('Breusch-Pagan Test (p-values)', fontsize=14)
ax_bp.set_ylabel('p-value', fontsize=12)
ax_bp.set_xticks(range(len(model_colors)))
ax_bp.set_xticklabels(list(model_colors.keys()), rotation=0)
ax_bp.grid(True, alpha=0.3)
ax_bp.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
ax_bp.text(1.5, 0.06, 'Threshold  0.05', color='red', ha='center')

# 3. PANEL DE ESTACIONARIEDAD
# Gráfico de p-valores ADF
ax_adf = fig.add_subplot(gs[2, :2])
model_colors_adf = {'ETH': '#edae49', 'CNRM': '#00798c', 'CMCC': '#d1495b'}
for i, (model, color) in enumerate(model_colors_adf.items()):
    ax_adf.boxplot(adf_pvalues[model], positions=[i], widths=0.6, patch_artist=True,  boxprops=dict(facecolor=color, alpha=0.6))
ax_adf.set_title('Test de Dickey-Fuller (p-values)', fontsize=14)
ax_adf.set_ylabel('p-values', fontsize=12)
ax_adf.set_xticks(range(len(model_colors_adf)))
ax_adf.set_xticklabels(list(model_colors_adf.keys()))
ax_adf.grid(True, alpha=0.3)
ax_adf.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
ax_adf.text(1, 0.05, 'Threshold 0.05', color='red', ha='center')

# Gráfico de ciclo anual (medias mensuales)
ax_seasonal = fig.add_subplot(gs[2, 2:])
months = range(1, 13)
for model, color in model_colors_adf.items():
    monthly_means = []
    for month in months:
        if month in means_by_month[model]:
            monthly_means.append(np.mean(means_by_month[model][month]))
        else:
            monthly_means.append(np.nan)
    ax_seasonal.plot(months, monthly_means, 'o-', color=color, label=model, linewidth=2)

ax_seasonal.set_title('Annual cicle (Monthly mean)', fontsize=14)
ax_seasonal.set_xlabel('Month', fontsize=12)
ax_seasonal.set_ylabel('log(WS) mean', fontsize=12)
ax_seasonal.set_xticks(months)
ax_seasonal.grid(True, alpha=0.3)
ax_seasonal.legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0.40, hspace=0.40, left=0.10, right=0.9, bottom=0.09, top =0.92) 
plt.savefig(f"{bd_out_fig}TC_Assumptions_Condensed_Summary.png", dpi=300, bbox_inches='tight')
plt.show()



# # # Añadir resumen de estadísticas clave
# # stats_text = (
# #     f"ASSUMPTION CHECK SUMMARY (N={total_points} points)\n"
# #     f"LINEARITY: Average correlation ETH-CNRM: {np.mean(corr_data['ETH-CNRM']):.3f}, "
# #     f"ETH-CMCC: {np.mean(corr_data['ETH-CMCC']):.3f}, CNRM-CMCC: {np.mean(corr_data['CNRM-CMCC']):.3f}\n"
# #     f"HOMOSCEDASTICITY: Ratio with heteroscedasticity (p<0.05): "
# #     f"{sum(p < 0.05 for ps in bp_pvalues.values() for p in ps) / sum(len(ps) for ps in bp_pvalues.values()):.1%}\n"
# #     f"STATIONARIETY: Proportion of non-stationary series(p>0.05): "
# #     f"{sum(p > 0.05 for ps in adf_pvalues.values() for p in ps) / sum(len(ps) for ps in adf_pvalues.values()):.1%}"
# # )
# # fig.text(0.5, 0.01, stats_text, ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=12)

# Guardar figura



























for cat in fl_cats:
    print(f"## Processing category {cat} TC rotated")
    
    data_npz    = np.load(f"{bd_out_tc}_TS_cat_{cat}.npz")

    time_series = data_npz['time_series']
    coordinates = data_npz['coordinates']
    models = data_npz['models']

    # Seleccionar algunos puntos representativos para análisis detallado (evitar analizar todos por eficiencia)
    # Puedes ajustar el número según tus necesidades
    n_sample_points = min(5, time_series.shape[0])
    sample_indices  = np.random.choice(time_series.shape[0], n_sample_points, replace=False)

    # Crear figura para todos los análisis de supuestos
    fig_assumptions = plt.figure(figsize=(15, 18))
    gs              = gridspec.GridSpec(n_sample_points, 3, figure=fig_assumptions)

    # Almacenar resultados de pruebas estadísticas para análisis posterior
    linearity_results        = []
    homoscedasticity_results = []
    stationarity_results     = []

    # Analizar cada punto de muestra
    for idx, sample_idx in enumerate(sample_indices):
        # Extraer series temporales para este punto
        serie_eth = time_series[sample_idx, 0, :]
        serie_cnrm = time_series[sample_idx, 1, :]
        serie_cmcc = time_series[sample_idx, 2, :]
        
        # Convertir a dataframes para facilitar agrupación
        df_s_eth = pd.DataFrame(serie_eth, index=full_range, columns=["WS_h"])
        df_s_cnrm = pd.DataFrame(serie_cnrm, index=full_range, columns=["WS_h"])
        df_s_cmcc = pd.DataFrame(serie_cmcc, index=full_range, columns=["WS_h"])
        
        # Obtener máximos diarios de wind gust (como en tu código original)
        wg_s_eth = df_s_eth.groupby(df_s_eth.index.date)["WS_h"].max()
        wg_s_cnrm = df_s_cnrm.groupby(df_s_cnrm.index.date)["WS_h"].max()
        wg_s_cmcc = df_s_cmcc.groupby(df_s_cmcc.index.date)["WS_h"].max()

        # Aplicar transformación logarítmica para estabilizar la varianza
        log_wg_s_eth  = np.log1p(wg_s_eth)   # log1p es ln(x+1)
        log_wg_s_cnrm = np.log1p(wg_s_cnrm)
        log_wg_s_cmcc = np.log1p(wg_s_cmcc)
            
        
        # Crear dataframe combinado con fechas alineadas
        df_combined = pd.DataFrame({
            'ETH': log_wg_s_eth.values,
            'CNRM': log_wg_s_cnrm.values,
            'CMCC': log_wg_s_cmcc.values
        }, index=wg_s_eth.index)
        
        # 1. VERIFICACIÓN DE LINEALIDAD
        # ---------------------------
        ax_linearity = fig_assumptions.add_subplot(gs[idx, 0])
        
        # Scatter plots con líneas de regresión
        sns.regplot(x='ETH', y='CNRM', data=df_combined, scatter_kws={'alpha':0.5}, ax=ax_linearity, label='ETH-CNRM', color='blue')
        sns.regplot(x='ETH', y='CMCC', data=df_combined, scatter_kws={'alpha':0.5}, ax=ax_linearity, label='ETH-CMCC', color='red')
        sns.regplot(x='CNRM', y='CMCC', data=df_combined, scatter_kws={'alpha':0.5}, ax=ax_linearity, label='CNRM-CMCC', color='green')
        
        # Calcular coeficientes de correlación y R²
        r_eth_cnrm = np.corrcoef(df_combined['ETH'], df_combined['CNRM'])[0,1]
        r_eth_cmcc = np.corrcoef(df_combined['ETH'], df_combined['CMCC'])[0,1]
        r_cnrm_cmcc = np.corrcoef(df_combined['CNRM'], df_combined['CMCC'])[0,1]
        
        # Realizar prueba más formal de linealidad con RESET test (Ramsey's Regression Equation Specification Error Test)
        # Implementación simplificada usando regresión lineal y términos cuadráticos
        from sklearn.linear_model import LinearRegression
        
        # Función para RESET test simplificado
        def simplified_reset_test(x, y, power=2):
            X = x.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            
            # Crear términos no lineales
            X_nl = np.column_stack([X, y_pred**power])
            model_nl = LinearRegression().fit(X_nl, y)
            
            # Comparar R² entre modelos
            r2_base = model.score(X, y)
            r2_nl = model_nl.score(X_nl, y)
            
            # Si R² mejora significativamente con términos no lineales, hay evidencia de no linealidad
            return r2_nl - r2_base
        
        # Ejecutar pruebas RESET para cada par
        reset_eth_cnrm = simplified_reset_test(df_combined['ETH'].values, df_combined['CNRM'].values)
        reset_eth_cmcc = simplified_reset_test(df_combined['ETH'].values, df_combined['CMCC'].values)
        reset_cnrm_cmcc = simplified_reset_test(df_combined['CNRM'].values, df_combined['CMCC'].values)
        
        # Guardar resultados
        linearity_results.append({
            'punto': sample_idx,
            'coordenadas': coordinates[sample_idx],
            'r_eth_cnrm': r_eth_cnrm,
            'r_eth_cmcc': r_eth_cmcc,
            'r_cnrm_cmcc': r_cnrm_cmcc,
            'reset_eth_cnrm': reset_eth_cnrm,
            'reset_eth_cmcc': reset_eth_cmcc,
            'reset_cnrm_cmcc': reset_cnrm_cmcc
        })
        
        # Añadir título y leyenda al gráfico
        ax_linearity.set_title(f'Linealidad - Punto {sample_idx}\nCoordenadas: {coordinates[sample_idx]}')
        ax_linearity.set_xlabel('Velocidad (m/s)')
        ax_linearity.set_ylabel('Velocidad (m/s)')
        ax_linearity.legend()
        ax_linearity.text(0.05, 0.95, f'r(ETH-CNRM): {r_eth_cnrm:.2f}\nr(ETH-CMCC): {r_eth_cmcc:.2f}\nr(CNRM-CMCC): {r_cnrm_cmcc:.2f}', 
                        transform=ax_linearity.transAxes, fontsize=8, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 2. VERIFICACIÓN DE HOMOSCEDASTICIDAD
        # -----------------------------------
        ax_homo = fig_assumptions.add_subplot(gs[idx, 1])
        
        # Calculamos residuos para cada par de modelos
        from scipy import stats
        
        # ETH vs CNRM
        slope_eth_cnrm, intercept_eth_cnrm, _, _, _ = stats.linregress(df_combined['ETH'], df_combined['CNRM'])
        residuals_eth_cnrm = df_combined['CNRM'] - (slope_eth_cnrm * df_combined['ETH'] + intercept_eth_cnrm)
        
        # ETH vs CMCC
        slope_eth_cmcc, intercept_eth_cmcc, _, _, _ = stats.linregress(df_combined['ETH'], df_combined['CMCC'])
        residuals_eth_cmcc = df_combined['CMCC'] - (slope_eth_cmcc * df_combined['ETH'] + intercept_eth_cmcc)
        
        # CNRM vs CMCC
        slope_cnrm_cmcc, intercept_cnrm_cmcc, _, _, _ = stats.linregress(df_combined['CNRM'], df_combined['CMCC'])
        residuals_cnrm_cmcc = df_combined['CMCC'] - (slope_cnrm_cmcc * df_combined['CNRM'] + intercept_cnrm_cmcc)
        
        # Graficamos residuos vs valores predichos
        ax_homo.scatter(slope_eth_cnrm * df_combined['ETH'] + intercept_eth_cnrm, residuals_eth_cnrm, 
                    alpha=0.5, label='ETH-CNRM', color='blue')
        ax_homo.scatter(slope_eth_cmcc * df_combined['ETH'] + intercept_eth_cmcc, residuals_eth_cmcc, 
                    alpha=0.5, label='ETH-CMCC', color='red')
        ax_homo.scatter(slope_cnrm_cmcc * df_combined['CNRM'] + intercept_cnrm_cmcc, residuals_cnrm_cmcc, 
                    alpha=0.5, label='CNRM-CMCC', color='green')
        
        # Test formal de homoscedasticidad: Test de Breusch-Pagan
        from statsmodels.stats.diagnostic import het_breuschpagan
        
        def simple_bp_test(y, x):
            # Regresión lineal simple
            x_with_const = sm.add_constant(x)
            model = sm.OLS(y, x_with_const).fit()
            
            # Test de Breusch-Pagan
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            return bp_test[1]  # p-valor
        
        import statsmodels.api as sm
        
        # Ejecutar tests
        bp_eth_cnrm = simple_bp_test(df_combined['CNRM'], df_combined['ETH'])
        bp_eth_cmcc = simple_bp_test(df_combined['CMCC'], df_combined['ETH'])
        bp_cnrm_cmcc = simple_bp_test(df_combined['CMCC'], df_combined['CNRM'])
        
        # Guardar resultados
        homoscedasticity_results.append({
            'punto': sample_idx,
            'coordenadas': coordinates[sample_idx],
            'bp_eth_cnrm': bp_eth_cnrm,
            'bp_eth_cmcc': bp_eth_cmcc,
            'bp_cnrm_cmcc': bp_cnrm_cmcc
        })
        
        # Añadir línea horizontal en y=0 para referencia
        ax_homo.axhline(y=0, color='black', linestyle='--')
        
        # Añadir título y leyenda
        ax_homo.set_title(f'Homoscedasticidad - Punto {sample_idx}')
        ax_homo.set_xlabel('Valores predichos (m/s)')
        ax_homo.set_ylabel('Residuos')
        ax_homo.legend()
        ax_homo.text(0.05, 0.95, f'BP p-val(ETH-CNRM): {bp_eth_cnrm:.3f}\nBP p-val(ETH-CMCC): {bp_eth_cmcc:.3f}\nBP p-val(CNRM-CMCC): {bp_cnrm_cmcc:.3f}',
                    transform=ax_homo.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 3. VERIFICACIÓN DE ESTACIONARIEDAD
        # --------------------------------
        ax_station = fig_assumptions.add_subplot(gs[idx, 2])
        
        # Asegurarse de que el índice es datetime
        df_combined.index = pd.to_datetime(df_combined.index)
        
        # Análisis de ventana móvil para evaluar estacionariedad
        window_size = min(30, len(df_combined) // 4)  # Ventana de 30 días o 1/4 de la serie si es más corta
        
        # Calcular media y desviación estándar móviles
        rolling_mean_eth = df_combined['ETH'].rolling(window=window_size).mean()
        rolling_std_eth = df_combined['ETH'].rolling(window=window_size).std()
        rolling_mean_cnrm = df_combined['CNRM'].rolling(window=window_size).mean()
        rolling_std_cnrm = df_combined['CNRM'].rolling(window=window_size).std()
        rolling_mean_cmcc = df_combined['CMCC'].rolling(window=window_size).mean()
        rolling_std_cmcc = df_combined['CMCC'].rolling(window=window_size).std()
        
        # Graficar medias móviles
        ax_station.plot(df_combined.index, rolling_mean_eth, label='Media móvil ETH', color='blue')
        ax_station.plot(df_combined.index, rolling_mean_cnrm, label='Media móvil CNRM', color='red')
        ax_station.plot(df_combined.index, rolling_mean_cmcc, label='Media móvil CMCC', color='green')
        
        # Prueba formal: Test de Dickey-Fuller aumentado
        from statsmodels.tsa.stattools import adfuller
        
        adf_eth  = adfuller(df_combined['ETH'].dropna())
        adf_cnrm = adfuller(df_combined['CNRM'].dropna())
        adf_cmcc = adfuller(df_combined['CMCC'].dropna())
        
        # Guardar resultados
        stationarity_results.append({
            'punto': sample_idx,
            'coordenadas': coordinates[sample_idx],
            'adf_eth_pvalue': adf_eth[1],
            'adf_cnrm_pvalue': adf_cnrm[1],
            'adf_cmcc_pvalue': adf_cmcc[1],
        })
        
        # Añadir título y leyenda
        ax_station.set_title(f'Estacionariedad - Punto {sample_idx}')
        ax_station.set_xlabel('Fecha')
        ax_station.set_ylabel('Media móvil (m/s)')
        ax_station.legend()
        # Formatear fechas en el eje x
        ax_station.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        ax_station.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_station.xaxis.get_majorticklabels(), rotation=45)
        
        # Añadir resultados de test ADF
        ax_station.text(0.05, 0.95, f'ADF p-val ETH: {adf_eth[1]:.3f}\nADF p-val CNRM: {adf_cnrm[1]:.3f}\nADF p-val CMCC: {adf_cmcc[1]:.3f}',
                    transform=ax_station.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Ajustar layout
    plt.tight_layout()
    fig_assumptions.suptitle(f'Verificación de Supuestos TC - {cat}', fontsize=16, y=0.99)
    plt.subplots_adjust(top=0.95)

    # Guardar figura
    # plt.savefig(f"{bd_out_fig}TC_Assumptions_Check_{cat.png", dpi=300, bbox_inches='tight')
    plt.close()

########################################################################################
### ---------------RESUMEN DE VERIFICACIÓN DE SUPUESTOS DE TC---------------------###
########################################################################################

# Convertir resultados a dataframes para análisis
df_linearity        = pd.DataFrame(linearity_results)
df_homoscedasticity = pd.DataFrame(homoscedasticity_results)
df_stationarity     = pd.DataFrame(stationarity_results)

# Resumir resultados de linealidad
print("\n== RESUMEN DE VERIFICACIÓN DE LINEALIDAD ==")
print(f"Correlaciones medias: ETH-CNRM: {df_linearity['r_eth_cnrm'].mean():.3f}, ETH-CMCC: {df_linearity['r_eth_cmcc'].mean():.3f}, CNRM-CMCC: {df_linearity['r_cnrm_cmcc'].mean():.3f}")
print(f"Mejora R² RESET Test: ETH-CNRM: {df_linearity['reset_eth_cnrm'].mean():.3f}, ETH-CMCC: {df_linearity['reset_eth_cmcc'].mean():.3f}, CNRM-CMCC: {df_linearity['reset_cnrm_cmcc'].mean():.3f}")
print(f"Linealidad problemática (mejora R² > 0.05): {((df_linearity['reset_eth_cnrm'] > 0.05) | (df_linearity['reset_eth_cmcc'] > 0.05) | (df_linearity['reset_cnrm_cmcc'] > 0.05)).sum()} de {len(df_linearity)} puntos")

# Resumir resultados de homoscedasticidad
print("\n== RESUMEN DE VERIFICACIÓN DE HOMOSCEDASTICIDAD ==")
print(f"p-valores medios Breusch-Pagan: ETH-CNRM: {df_homoscedasticity['bp_eth_cnrm'].mean():.3f}, ETH-CMCC: {df_homoscedasticity['bp_eth_cmcc'].mean():.3f}, CNRM-CMCC: {df_homoscedasticity['bp_cnrm_cmcc'].mean():.3f}")
print(f"Heteroscedasticidad detectada (p < 0.05): {((df_homoscedasticity['bp_eth_cnrm'] < 0.05) | (df_homoscedasticity['bp_eth_cmcc'] < 0.05) | (df_homoscedasticity['bp_cnrm_cmcc'] < 0.05)).sum()} de {len(df_homoscedasticity)} puntos")

# Resumir resultados de estacionariedad
print("\n== RESUMEN DE VERIFICACIÓN DE ESTACIONARIEDAD ==")
print(f"p-valores medios ADF: ETH: {df_stationarity['adf_eth_pvalue'].mean():.3f}, CNRM: {df_stationarity['adf_cnrm_pvalue'].mean():.3f}, CMCC: {df_stationarity['adf_cmcc_pvalue'].mean():.3f}")
print(f"No estacionariedad detectada (p > 0.05): {((df_stationarity['adf_eth_pvalue'] > 0.05) | (df_stationarity['adf_cnrm_pvalue'] > 0.05) | (df_stationarity['adf_cmcc_pvalue'] > 0.05)).sum()} de {len(df_stationarity)} puntos")

# Implicaciones para el análisis de TC
print("\n== IMPLICACIONES PARA TRIPLE COLOCACIÓN ==")
linearity_issue        = ((df_linearity['reset_eth_cnrm'] > 0.05) | (df_linearity['reset_eth_cmcc'] > 0.05) | (df_linearity['reset_cnrm_cmcc'] > 0.05)).mean() > 0.25
homoscedasticity_issue = ((df_homoscedasticity['bp_eth_cnrm'] < 0.05) | (df_homoscedasticity['bp_eth_cmcc'] < 0.05) | (df_homoscedasticity['bp_cnrm_cmcc'] < 0.05)).mean() > 0.25
stationarity_issue     = ((df_stationarity['adf_eth_pvalue'] > 0.05) | (df_stationarity['adf_cnrm_pvalue'] > 0.05) | (df_stationarity['adf_cmcc_pvalue'] > 0.05)).mean() > 0.25

if linearity_issue:
    print("⚠️ ADVERTENCIA: Posibles problemas de linealidad detectados. Considerar transformaciones o TC extendido.")
if homoscedasticity_issue:
    print("⚠️ ADVERTENCIA: Heteroscedasticidad detectada. Los resultados de TC podrían estar sesgados.")
if stationarity_issue:
    print("⚠️ ADVERTENCIA: Series no estacionarias detectadas. Considerar análisis por ventanas temporales más cortas o diferenciación.")

# Recomendaciones basadas en los resultados
print("\n== RECOMENDACIONES ==")
if linearity_issue or homoscedasticity_issue or stationarity_issue:
    print("Basado en los resultados de verificación de supuestos, se recomienda:")
    if linearity_issue:
        print("1. Evaluar transformaciones logarítmicas o potenciales para mejorar la linealidad")
        print("2. Considerar implementar una versión extendida de Triple Colocación que permita relaciones no lineales")
    if homoscedasticity_issue:
        print("3. Analizar por separado diferentes rangos de valores para manejar la heteroscedasticidad")
        print("4. Considerar transformaciones para estabilizar la varianza (log, raíz cuadrada)")
    if stationarity_issue:
        print("5. Realizar TC por ventanas temporales más cortas (estacional o mensual)")
        print("6. Considerar preprocesamientos para hacer las series más estacionarias (diferenciación, eliminación de tendencia)")
else:
    print("✓ Los supuestos de TC parecen cumplirse razonablemente bien en los puntos analizados.")
    print("Se recomienda continuar con el análisis TC según lo planeado.")













