
# %%
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
import os,glob,sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from smev_class import SMEV
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.stats import genextreme
from scipy.stats import rankdata
from scipy import integrate
from scipy.stats import gumbel_r
from affine import Affine
from shapely.geometry import Point, Polygon

"""
Code for analysingthe parametres of the U50 estimated in contrasted points over water
vs terrestial surfaces. The comparison is also distributional and geographycal.

The aim in to produce analysis for understanding the U50 patterns in water surfaces.

Author: Nathalia Correa-Sánchez
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################
STORAGE    = Path("/mnt/smb").as_posix()
file_cats  = STORAGE + "/Outputs/Climate_Provinces/CSVs/Combination_RIX.csv"
bd_out_fig = STORAGE + "/Outputs/Plots/WP3_development/"
bd_in_rast = STORAGE + "/Outputs/Climate_Provinces/Development_Rasters/FinalRasters_In-Out/"
ras_comb   = "SEA-LAND_Combined_RIX_remCPM_WGS84.tif"
bd_out_tc  = STORAGE + "/Outputs/WP3_SamplingSeries_CPM/"
bd_out_rl  = STORAGE + "/Outputs/RL_ws100m/"
# %%
########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################
filas_eliminar    = [0]  # Primera  fila, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada 
columnas_eliminar = [0]  # Primera columna, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada
rp_idx            = 4    # CAMBIAR ESTE: OJOOOOOOOO Índice para el período de retorno !!!!! (VER LA LISTA DE LOS PERIODOS DE RETORNO)
ret_per           = 50    # CAMBIAR ESTE: Valor para etiquetas
return_periods    = [2, 5, 10, 20, 50] # List of return periods 

########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################
    
def separation_ws_events_max(s_arr, h_corr, dt_arr):
    """
    Function to obtain the list of maximum values extracted in every loop for the wind speeds array, 
    by also deleting the 'h_corr' preceding and following hours. It also delivers the index positon
    for each maximum value, which ususally is a datetime. The aim is to identify the storms
    INPUTS:
    - s_arr  =  numpy array with the ordinary events to separate, at the same temporal frequency represented 
                by h_corr.
    - h_corr = integer representing the time of when the correlation is low and for instance the separation
               time.
    - dt_arr = numpy array with the date time index (or any kind of index) for each element in s_arr to keep
                more informaiton ont he events.

    OUTPUTS:
    - sep_event = numpy array with the selected ordinary events.
    - idx_eents = numpy array with the indexes (i.e. datetime) of each selected ordinary events.

    """

    sep_events = []
    idx_events = []
    
    while len(s_arr) > 0:

        # Finding the max value and its index and adding it to the events array, also for the related index.
        max_index = np.argmax(s_arr)
        sep_events.append(s_arr[max_index])

        idx_events.append(dt_arr[max_index]) #Usually datetimes
        
        # Computing the bordsrs of the before and after window to be deleted
        start_index = max(0, max_index - h_corr)              # Avoiding negative indexes if the max is in the first values
        end_index   = min(len(s_arr), max_index + h_corr + 1) # Avoiding excedances in the indexes if the max is in the last values
        
        # Deleting teh range windiw of data before and after
        s_arr  = np.delete(s_arr, slice(start_index, end_index))
        dt_arr = np.delete(dt_arr, slice(start_index, end_index))

    sep_events = np.array(sep_events)
    idx_events = np.array(idx_events)
    
    return sep_events, idx_events

def simple_mev_newsep(s_arr, dates, return_periods, threshold_measure, separation, durations, time_resolution, data_portion):
    # Convert dates to numpy datetime64 array
    dates_np = dates.to_numpy()

    #######################ORDINARY EVENTS SELECTION################################
    sep_events, idt_events = separation_ws_events_max(s_arr, separation, dates_np)
    ordinary_values        = sep_events
    ordinary_dates         = [pd.to_datetime(idt_events[i]) for i in range(len(idt_events))]

    ordinary_events_new_df = pd.DataFrame({'value': ordinary_values}, index = ordinary_dates)
    n_ordinary_per_year    = ordinary_events_new_df.groupby(pd.Grouper(freq="Y")).count() 
    n_ordinary             = n_ordinary_per_year['value'].mean()
    ###############################################################################

    return_levels = []
    shape_param = None  # Initialize
    scale_param = None  # Initialize    
    for i in range(len(return_periods)):
        return_period = return_periods[i]

        # Initialize SMEV object
        rl_arr = SMEV(threshold_measure, separation, return_period, durations, time_resolution)

        # Estimate SMEV parameters
        shape_new, scale_new = rl_arr.estimate_smev_parameters(ordinary_events_new_df['value'], data_portion)
        
        # Save parameters (same for all return periods, so we only need to save once)
        if i == 0:
            shape_param = shape_new  # This is kappa (κ)
            scale_param = scale_new  # This is lambda (λ)

        # Calculate return values
        intensity = rl_arr.smev_return_values(rl_arr.return_period, shape_new, scale_new, n_ordinary)
        return_levels.append(intensity)

    return_values = np.array(return_levels)

    # MODIFICACIÓN: Retornar también shape, scale, y n_ordinary
    return return_values, ordinary_events_new_df, shape_param, scale_param, n_ordinary

def style_axis(ax):
    """
    Function to set the format to the plots
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def exp_func(x, a, b):
    """
    Function for the exponential fit
    """
    return a * np.exp(-b * x)

def linear_func(x, a, b):
    """
    Function for the linear fit
    """
    return a * x + b

def gevd_fit(data, return_periods):
    # Extraer máximos anuales
    annual_max = np.array([np.max(data[i*8760:(i+1)*8760]) for i in range(10)])
    n          = len(annual_max)

    # Ajuste de la distribución GEV a los máximos anuales
    shape, loc, scale = genextreme.fit(annual_max)

    list_ut    = []
    for i in range(len(return_periods)):
        return_period = return_periods[i]

        ut = genextreme.ppf(1 - 1/return_period, shape, loc=loc, scale=scale)
        
        list_ut.append(ut)
        # list_sigma_ut.append(sigma_ut)
    
    return list_ut, (shape, loc, scale)

def gpd_fit(data, threshold, return_periods, h_corr, dt_arr):
    # 1. Obtener eventos independientes basando en funcion precedente
    sep_events, idx_events = separation_ws_events_max(data, h_corr, dt_arr)
    
    # 2. Seleccionar excedencias sobre el umbral
    exceedances = sep_events[sep_events > threshold] - threshold
    
    # 3. Ajustar GPD
    shape, loc, scale = stats.genpareto.fit(exceedances)
    
    # 4. Calcular tasa de excedencia (usando longitud original de datos)
    years = len(data) / 8760
    rate  = len(exceedances) / years  # Eventos por año
    
    list_ut       = []
    list_sigma_ut = []
    for return_period in return_periods:
        # Calcular el cuantil
        q = 1 - 1 / (return_period * rate)
        if 0 < q < 1:
            ut = threshold + stats.genpareto.ppf(q, shape, loc, scale)
            
            # Incertidumbre
            sigma_ut = scale / np.sqrt(years) * np.sqrt(1 + np.log(return_period)**2)
            
            list_ut.append(ut)
            list_sigma_ut.append(sigma_ut)
        else:
            list_ut.append(np.nan)
            list_sigma_ut.append(np.nan)
    
    return list_ut, list_sigma_ut

def spectral_correction(s_arr, return_periods=[2, 5, 10, 20]):
    """
    Estima valores extremos de viento usando el método de Corrección Espectral (SC).
    
    INPUTS:
    -----------
    s_arr          : numpy.ndarray
        Serie temporal de velocidad del viento (datos horarios)
    h_corr         : float
        Tiempo de decorrelación promedio en horas
    return_periods : list
        Lista de períodos de retorno en años para los cuales estimar valores extremos
    
    OUTPUTS:
    --------
    extreme_values :
        Lista con los valores extremos estimados para cada período de retorno
    R              :
        Float de la relacion Umax_corrected / Umax_origina
        
    corrected_annual_maxima:
        Lista con los valores corregidos de annual maxima para cada año de registro

    fc            :
        Float con el valor del cutoff frequency
    """

    def calculate_spectrum_smoothed(data, window_size=10):
        detrended_data = data - np.mean(data)
        n              = len(detrended_data)
        fft_values     = np.fft.rfft(detrended_data)
        freqs          = np.fft.rfftfreq(n, d=1/24)  # Frecuencia en días^-1
        psd            = 2 * np.abs(fft_values)**2 / (n * 24)

        # Suavizar el espectro usando una convolución
        kernel       = np.ones(window_size) / window_size
        smoothed_psd = np.convolve(psd, kernel, mode='same')
        
        return freqs[1:], psd[1:], smoothed_psd[1:]  # Eliminamos la frecuencia cero


    # Determinar fc y S(fc) con ajuste log-log
    def find_cutoff_frequency(freqs, psd):
        # Ajuste lineal en el rango 0.6 < f < 0.9 días^-1 --> esto viene de la teoria
        mask      = (freqs > 0.6) & (freqs < 0.9)
        log_freqs = np.log(freqs[mask])
        log_psd   = np.log(psd[mask])
        slope, intercept = np.polyfit(log_freqs, log_psd, 1)
        
        # Determinar f_c y S(f_c)
        fc     = np.exp((np.log(1) - intercept) / slope)  
        fc_idx = np.argmin(np.abs(freqs - fc))
        return freqs[fc_idx], psd[fc_idx]

    # Aplicar corrección espectral
    def correct_spectrum(freqs, psd, fc, s_fc):
        fc_idx                 = np.argmin(np.abs(freqs - fc))
        corrected_psd          = psd.copy()
        corrected_psd[fc_idx:] = s_fc * (freqs[fc_idx:] / fc)**(-5/3)
        return corrected_psd

    def calculate_spectral_moments(freqs, psd):
        m0 = np.trapezoid(psd, freqs)
        m2 = np.trapezoid(psd * freqs**2, freqs)
        return m0, m2
    
    def calculate_annual_maximum(mean_speed, m0, m2):
        T0          = 365  # 1 año en días
        nu          = np.sqrt(m2 / m0) / (2 * np.pi)
        peak_factor = np.sqrt(2 * np.log(2 * nu * T0))
        Umax        = mean_speed + np.sqrt(m0) * peak_factor
        return Umax

    # Verificación de datos
    if len(s_arr) < 8760:
        raise ValueError("Se requiere al menos un año de datos horarios")
    
    mean_speed = np.mean(s_arr)

    # Calcular espectro
    freqs, psd, smoothed_psd = calculate_spectrum_smoothed(s_arr)

    # Determinar f_c y corregir espectro
    fc, s_fc      = find_cutoff_frequency(freqs, smoothed_psd)
    corrected_psd = correct_spectrum(freqs, smoothed_psd, fc, s_fc)

    # Calcular momentos espectrales
    m0_original, m2_original   = calculate_spectral_moments(freqs, smoothed_psd)
    m0_corrected, m2_corrected = calculate_spectral_moments(freqs, corrected_psd)

    # Estimar velocidad máxima anual media
    Umax_original  = calculate_annual_maximum(mean_speed, m0_original, m2_original)
    Umax_corrected = calculate_annual_maximum(mean_speed, m0_corrected, m2_corrected)

    # Relación R
    R = Umax_corrected / Umax_original

    # Extraer velocidades máximas anuales
    annual_maxima = []
    for year in range(len(s_arr) // 8760):
        start = year * 8760
        end   = start + 8760
        annual_maxima.append(np.max(s_arr[start:end]))
    
    # Corregir las velocidades máximas anuales usando R
    corrected_annual_maxima = [R * value for value in annual_maxima]

    # Ajuste de Gumbel
    corrected_mean = np.mean(corrected_annual_maxima)
    corrected_std  = np.std(corrected_annual_maxima)
    beta           = np.sqrt(6) * corrected_std / np.pi
    mu             = corrected_mean - 0.5772 * beta
    extreme_values = [mu - beta * np.log(-np.log(1 - 1/T)) for T in return_periods]

    return extreme_values, R, corrected_annual_maxima, fc

def decorr_time_potential(serie, lags):
    """
    Function to compute the de-correlation time for the events separation, 
    to guarantee their independence based on the potential approach which relies in
    the decorrelation time of each time serie.
    """
    # Usar statsmodels para calcular la autocorrelación
    rs_values = acf(serie, nlags=max(lags), fft=True)[lags]
    
    # Ajuste spline cúbico
    spline = UnivariateSpline(lags, rs_values, s=0.01)  # s es el parámetro de suavizado
    rs_spline = spline(lags)
    
    # Calcular la integral (área bajo la curva)
    tau_d = np.trapezoid(rs_spline, lags)
    p_cor = int(round(tau_d, 0))
    
    return p_cor

def process_time_series(s_arr, p_cor, dates, dates_np, return_periods):
    """
    Procesa una serie de tiempo aplicando diferentes métodos de análisis de valores extremos.
    
    Parámetros:
    s_arr         : Serie de tiempo a analizar
    p_cor         : Tiempo de decorrelación
    dates         : Fechas en formato datetime
    dates_np      : Fechas en formato timestamp
    return_periods: Periodos de retorno a calcular
    
    Retorna:
    Un diccionario con todos los resultados de los diferentes métodos
    """
    results = {}
    
    # --- SMEV con decaimiento potencial y top 10% ---
    p_cens                                                  = 0.90
    smev_rl, df_ord_events_p, kappa, lambda_scale, n_annual = simple_mev_newsep( s_arr, full_range, return_periods, threshold_meas, p_cor, durations, time_resolution, [p_cens, 1] )
    results['smev_rl']                                      = smev_rl
    
    results['smev_rl']      = smev_rl
    results['kappa']        = kappa        # Shape parameter (forma)
    results['lambda_scale'] = lambda_scale # Scale parameter (escala)
    results['n_annual']     = n_annual     # Frecuencia anual de eventos
    
    print("### --- SMEV 10-pot results:")
    for period, value in zip(return_periods, smev_rl):
        print(f"Return period {period} years: {value:.2f}")
    print(f"    Weibull parameters: κ={kappa:.3f}, λ={lambda_scale:.2f}, n={n_annual:.1f}")  # AÑADIDO    
    
    # --- GEVD-PMM ---
    gev_rl, gev_param    = gevd_fit(s_arr, return_periods)
    results['gev_rl']    = gev_rl
    results['gev_param'] = gev_param
    
    print("### --- GEVD-PMM results:")
    for period, value in zip(return_periods, gev_rl):
        print(f"Return period {period} years: {value:.2f} m/s shape, loc, scale: {gev_param}")
    
    # --- GPD-POT eventos separados ---
    top_ord_events = 90
    threshold_gpd = np.percentile(s_arr, top_ord_events)
    gpd_rl, sigma_rl_gpd = gpd_fit(s_arr, threshold_gpd, return_periods, p_cor, dates_np)
    results['gpd_rl'] = gpd_rl
    results['sigma_rl_gpd'] = sigma_rl_gpd
    
    print("### --- GPD-POT results:")
    for period, value, sigma in zip(return_periods, gpd_rl, sigma_rl_gpd):
        print(f"Return period {period} years: {value:.2f} ± {1.96*sigma:.2f} m/s")
    
    # --- SC-Gumbel fit ---
    spcr_rl, _, annual_max_cr, _ = spectral_correction(s_arr, return_periods)
    results['spcr_rl'] = spcr_rl
    results['annual_max_cr'] = annual_max_cr
    
    print("### --- SC-Gumbel results:")
    for period, value in zip(return_periods, spcr_rl):
        print(f"Return period {period} years: {value:.2f} m/s")
    
    # --- Posiciones de plotting de probabilidad para máximos anuales ---
    annual_max = np.array([np.max(s_arr[i*8760:(i+1)*8760]) for i in range(10)])
    results['annual_max'] = annual_max
    
    n = len(annual_max)
    rank_am = rankdata(annual_max)
    pp = rank_am / (n + 1)
    rt_pp = 1 / (1 - pp)
    results['pp_annual_max'] = rt_pp
    
    # --- Posiciones de plotting para máximos corregidos ---
    n_cr = len(annual_max_cr)
    rank_am_cr = rankdata(annual_max_cr)
    pp_cr = rank_am_cr / (n_cr + 1)
    rt_pp_cr = 1 / (1 - pp_cr)
    results['pp_annual_max_cr'] = rt_pp_cr
    
    return results

# Función para decodificar una categoría en sus componentes
def decode_category(cat_code):
    cat_str   = str(cat_code).zfill(3)
    climate   = int(cat_str[0])
    roughness = int(cat_str[1])
    slope     = int(cat_str[2])
    return climate, roughness, slope

# Función para obtener la etiqueta completa de una categoría
def get_category_label(cat_code):
    if cat_code ==1:
        cat_label = f"$R_1$"
    elif cat_code > 100:
        climate, roughness, slope = decode_category(cat_code)
        cat_label =  f"{climate_names[climate]}, {roughness_names[roughness]}, {slope_names[slope]}"
    return cat_label

# Función modificada para preparar datos de visualización
def prepare_visualization_data(return_period_idx):
    data = []
    
    for cat in fl_cats:
        # Manejar caso especial de categoría 1 (representa solo rugosidad R_1: agua)
        if cat == 1:
            # Para categoría 1, solo consideramos rugosidad, no clima ni pendiente
            climate = None  # No aplicable para esta categoría
            roughness = 1   # R_1 es agua
            slope = None    # No aplicable para esta categoría
            
            cat_label = r"$R_1$:(water)"
        else:
            # Caso normal: decodificar la categoría de 3 dígitos
            climate, roughness, slope = decode_category(cat)
            cat_label = get_category_label(cat)
        
        # Extraer los valores de retorno para este período específico
        if cat in results_rl_eth and 'results' in results_rl_eth[cat]:
            eth_values = [res['smev_rl'][return_period_idx] for res in results_rl_eth[cat]['results'] if 'smev_rl' in res]
            cnrm_values = [res['smev_rl'][return_period_idx] for res in results_rl_cnrm[cat]['results'] if 'smev_rl' in res]
            cmcc_values = [res['smev_rl'][return_period_idx] for res in results_rl_cmcc[cat]['results'] if 'smev_rl' in res]
            
            # Calcular estadísticas
            eth_mean  = np.mean(eth_values) if eth_values else np.nan
            eth_std   = np.std(eth_values) if eth_values else np.nan
            cnrm_mean = np.mean(cnrm_values) if cnrm_values else np.nan
            cnrm_std  = np.std(cnrm_values) if cnrm_values else np.nan
            cmcc_mean = np.mean(cmcc_values) if cmcc_values else np.nan
            cmcc_std  = np.std(cmcc_values) if cmcc_values else np.nan
            
            # Añadir a la lista de datos - solo incluir en categorías relevantes
            entry = {
                'Category': cat,
                'Category_Label': cat_label,
                'ETH_Mean': eth_mean,
                'ETH_Std': eth_std,
                'CNRM_Mean': cnrm_mean,
                'CNRM_Std': cnrm_std,
                'CMCC_Mean': cmcc_mean,
                'CMCC_Std': cmcc_std
            }
            
            # Añadir información de clima solo si es aplicable
            if climate is not None:
                entry['Climate'] = climate
                entry['Climate_Name'] = climate_names[climate]
            
            # Añadir información de rugosidad (siempre aplicable)
            entry['Roughness'] = roughness
            entry['Roughness_Name'] = roughness_names[roughness]
            
            # Añadir información de pendiente solo si es aplicable
            if slope is not None:
                entry['Slope'] = slope
                entry['Slope_Name'] = slope_names[slope]
            
            data.append(entry)
    
    return pd.DataFrame(data)

# Función para calcular intervalo de confianza bootstrap
def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """
    Calcula intervalos de confianza mediante bootstrap
    
    Parámetros:
    data (array): Datos para calcular bootstrap
    n_bootstrap (int): Número de muestras bootstrap
    ci (float): Nivel de confianza (ej. 95 para 95%)
    
    Retorna:
    lower, upper: Límites inferior y superior del intervalo de confianza
    """
    if len(data) < 2:
        return np.nan, np.nan
    
    # Inicializar array para almacenar medias bootstrap
    bootstrap_means = np.zeros(n_bootstrap)
    
    # Generar muestras bootstrap
    for i in range(n_bootstrap):
        # Muestreo con reemplazo
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    # Calcular percentiles para el intervalo de confianza
    alpha = (100 - ci) / 2
    lower_percentile = alpha
    upper_percentile = 100 - alpha
    
    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)
    
    return lower, upper

# Calcular intervalos de confianza para cada categoría
def add_bootstrap_ci_to_df(df, category_col, value_col, n_bootstrap=1000):
    """
    Añade intervalos de confianza bootstrap a un DataFrame
    
    Parámetros:
    df (DataFrame): DataFrame con los datos
    category_col (str): Nombre de la columna de categoría
    value_col (str): Nombre de la columna de valor
    n_bootstrap (int): Número de muestras bootstrap
    
    Retorna:
    DataFrame con columnas adicionales para los intervalos de confianza
    """
    # Obtener categorías únicas
    categories = df[category_col].unique()
    
    # Inicializar listas para almacenar resultados
    cat_list   = []
    lower_list = []
    upper_list = []
    
    # Calcular intervalos de confianza para cada categoría
    for category in categories:
        # Filtrar datos para esta categoría
        cat_data = df[df[category_col] == category][value_col].values
        
        # Calcular intervalo de confianza
        lower, upper = bootstrap_ci(cat_data, n_bootstrap)
        
        # Almacenar resultados
        cat_list.append(category)
        lower_list.append(lower)
        upper_list.append(upper)
    
    # Crear DataFrame con los resultados
    result_df = pd.DataFrame({ category_col: cat_list,
                            'ci_lower': lower_list,
                            'ci_upper': upper_list })
    
    return result_df

# Función para obtener etiqueta descriptiva de forma segura
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

# Función para crear etiquetas descriptivas a partir de códigos numéricos
def create_descriptive_label(climate, roughness, slope):
    # Caso especial para la categoría 1 (agua)
    if np.isnan(climate) and roughness == 1 and np.isnan(slope):
        cat_label = roughness_names[1]  # Solo mostrar R₁:(water)
   
    # Caso normal: usar los nombres descriptivos
    else :
        climate_label   = climate_names.get(climate, str(climate))
        roughness_label = roughness_names.get(roughness, f"R_{roughness}")
        slope_label     = slope_names.get(slope, f"T_{slope}")
        
        # Combinar en formato descriptivo
        cat_label = f"{climate_label}{roughness_label}{slope_label}"

    return cat_label 

# Función para estimar densidad de kernel
def get_kde_values(data, x_min, x_max):
    kde = stats.gaussian_kde(data)
    x   = np.linspace(x_min, x_max, 1000)
    y   = kde(x)
    return x, y

def decode_category(cat_code):
    """
    Decode 3-digit category code into climate, roughness, slope
    Special case: cat_code=1 means only R1 (water)
    """
    if cat_code == 1:
        return None, 1, None  # R1 only
    else:
        cat_str = str(cat_code).zfill(3)
        climate = int(cat_str[0])
        roughness = int(cat_str[1])
        slope = int(cat_str[2])
        return climate, roughness, slope

# %%

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
df_filt     = df_filt.reset_index(drop=True) ## Resetea el indice que se habia dñado luego del filtrado. 
fl_cats     = df_filt['value'].values.astype(int)
# %%
#############################################################################
##-------SETTING UP SMEV PARAMETERS FOR SMEV RETURN LEVELS ESTIMATION------##
#############################################################################

# Setting up the SMEV  single parameters
threshold_meas  = 0  # 0 For the valid measurements, 0.1 mm in precipitation records in winds could be 0 m/s
durations       = 60 # In Minutes
time_resolution = 60 # In Minutes

#########################################################################################
###----------------EXTREME VALUES PER EACH POINT IN SPATIAL CATEGORIES----------------###
#########################################################################################
lags       = np.arange(1, 201, 1) 
full_range = pd.date_range(start='2000-01-01 00:00:00', end='2009-12-31 23:50:00', freq='1H')
dates      = np.array(full_range) ## As an array datetime
dates_np   = np.array([d.timestamp() for d in full_range]) ## As an array to,estamp

# Inicializar diccionarios para almacenar clos return levels de cada modelo
results_rl_eth  = {}
results_rl_cnrm = {}
results_rl_cmcc = {}

# Para cada categoría, calcular correlaciones de máximos mensuales
for cat in fl_cats:
    print(f"## Processing category {cat} for monthly maximum correlations")
    
    data_npz    = np.load(f"{bd_out_tc}_TS_cat_{cat}.npz")
    time_series = data_npz['time_series']
    coordinates = data_npz['coordinates']
    models      = data_npz['models']

    cat_results_eth = []
    cat_results_cnrm = []
    cat_results_cmcc = []
      
    # Procesar cada punto
    for k in range(len(coordinates)):
        point_coord = coordinates[k]
        print(f"Processing point {k+1}/{len(coordinates)}: {point_coord}")

        serie_eth  = time_series[k, 0, :]
        serie_cnrm = time_series[k, 1, :]
        serie_cmcc = time_series[k, 2, :]

        p_cor_eth  = decorr_time_potential(serie_eth, lags)
        p_cor_cnrm = decorr_time_potential(serie_cnrm, lags)
        p_cor_cmcc = decorr_time_potential(serie_cmcc, lags)

        try:
            # Procesar ETH
            print(f"Processing ETH model for point {k}")
            eth_results = process_time_series(serie_eth, p_cor_eth, dates, dates_np, return_periods)
            cat_results_eth.append(eth_results)
            
            # Procesar CNRM
            print(f"Processing CNRM model for point {k}")
            cnrm_results = process_time_series(serie_cnrm, p_cor_cnrm, dates, dates_np, return_periods)
            cat_results_cnrm.append(cnrm_results)
            
            # Procesar CMCC
            print(f"Processing CMCC model for point {k}")
            cmcc_results = process_time_series(serie_cmcc, p_cor_cmcc, dates, dates_np, return_periods)
            cat_results_cmcc.append(cmcc_results)
            
        except Exception as e:
            print(f"Error processing point {k} in category {cat}: {e}")
            # Continuar con el siguiente punto
            continue
    
    # Almacenar resultados en el diccionario principal
    results_rl_eth[cat] = {'results': cat_results_eth,
                           'coordinates': coordinates }
    
    results_rl_cnrm[cat] = {'results': cat_results_cnrm,
                           'coordinates': coordinates }
    
    results_rl_cmcc[cat] = {'results': cat_results_cmcc,
                           'coordinates': coordinates }
    
    # Extraer y organizar los return levels para esta categoría
    eth_smev_rl = np.array([res['smev_rl'] for res in cat_results_eth])
    eth_gev_rl  = np.array([res['gev_rl'] for res in cat_results_eth])
    eth_gpd_rl  = np.array([res['gpd_rl'] for res in cat_results_eth])
    eth_spcr_rl = np.array([res['spcr_rl'] for res in cat_results_eth])
    
    cnrm_smev_rl = np.array([res['smev_rl'] for res in cat_results_cnrm])
    cnrm_gev_rl  = np.array([res['gev_rl'] for res in cat_results_cnrm])
    cnrm_gpd_rl  = np.array([res['gpd_rl'] for res in cat_results_cnrm])
    cnrm_spcr_rl = np.array([res['spcr_rl'] for res in cat_results_cnrm])
    
    cmcc_smev_rl = np.array([res['smev_rl'] for res in cat_results_cmcc])
    cmcc_gev_rl  = np.array([res['gev_rl'] for res in cat_results_cmcc])
    cmcc_gpd_rl  = np.array([res['gpd_rl'] for res in cat_results_cmcc])
    cmcc_spcr_rl = np.array([res['spcr_rl'] for res in cat_results_cmcc])

results_rl_summary = {'eth'           : results_rl_eth,
                      'cnrm'          : results_rl_cnrm,
                      'cmcc'          : results_rl_cmcc,
                      'categories'    : fl_cats,
                      'return_periods': return_periods}
# %%

# #########################################################################################
# ###-------------------- LOADING PREVIOUSLY COMPUTED RETURN LEVELS --------------------###
# #########################################################################################
results_rl_eth  = results_rl_summary['eth']
results_rl_cnrm = results_rl_summary['cnrm']
results_rl_cmcc = results_rl_summary['cmcc']

# %%
###############################################################################
# LABELS AND COLORS
###############################################################################

# Climate labels (1=Arid, 2=Temperate, 3=Cold, 4=Tundra)
climate_labels = ['Ar', 'Tm', 'Co', 'Td']
climate_colors = {1: '#F5A500', 2: '#64C864', 3: '#4B92DB', 4: '#A8A8A8'}
# climate_colors = {1: '#FF6400', 2: '#64C864', 3: '#3264FF', 4: '#969696' }

# Roughness labels (1=water, 2-5=land)
roughness_labels = [r'$R_1$:(water)', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
roughness_colors = {1: '#345F77', 2: '#FFDA8A', 3: '#F2A05D', 4: '#D95D30', 5: '#8C2A04'}
# roughness_colors = {1: '#2C7BB6', 2: '#ABD9E9', 3: '#FFFFBF', 4: '#FDAE61', 5: '#D7191C'}

# Topography labels
topography_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$']
topography_colors = { 1: '#E5E0CB', 2: '#B8D6BE', 3: '#73AE80', 4: '#2A6B3D'}
# topography_colors = { 1: '#0000FF', 2: '#00B400', 3: '#FF0000', 4: '#FFA500'}

# %%
###############################################################################
###---------------------- EXTRACT WEIBULL PARAMETERS -----------------------###
###############################################################################

print("\nExtracting Weibull parameters...")

# Initialize lists to store data
all_data = []

for cat in fl_cats:
    climate, roughness, slope = decode_category(cat)
    
    # Extract parameters from all three models
    for model_name, results_dict in [('ETH', results_rl_eth), ('CNRM', results_rl_cnrm), ('CMCC', results_rl_cmcc)]:
        
        if cat in results_dict and 'results' in results_dict[cat]:
            results_list = results_dict[cat]['results']
            
            for point_result in results_list:
                if 'kappa' in point_result and 'lambda_scale' in point_result:
                    all_data.append({'category'    : cat,
                                     'climate'     : climate,
                                     'roughness'   : roughness,
                                     'slope'       : slope,
                                     'model'       : model_name,
                                     'kappa'       : point_result['kappa'],
                                     'lambda_scale': point_result['lambda_scale'],
                                     'n_annual'    : point_result['n_annual'] })

df_all = pd.DataFrame(all_data)

print(f"Extracted {len(df_all)} parameter sets")
print(f"Total points: {len(df_all)//3} (across 3 models)")

# Quick stats
print("\nQuick stats:")
print(f"  Kappa range: {df_all.kappa.min():.2f} - {df_all.kappa.max():.2f}")
print(f"  Lambda range: {df_all.lambda_scale.min():.2f} - {df_all.lambda_scale.max():.2f}")

###############################################################################
### ----------PREPARE DATA FOR PLOTTING - DISAGGREGATED BY LAYER -----------###
###############################################################################

print("\nPreparing data for disaggregated plots...")

# Climate data (excluding R1 which has climate=None)
df_climate    = df_all[df_all.climate.notna()].copy()
climate_order = [1, 2, 3, 4]  # Ar, Tm, Co, Td

# Roughness data (all categories)
df_roughness    = df_all.copy()
roughness_order = [1, 2, 3, 4, 5]

# Slope data (excluding R1 which has slope=None)
df_slope    = df_all[df_all.slope.notna()].copy()
slope_order = [1, 2, 3, 4]

print(f"  Climate: {len(df_climate)} values across {df_climate.climate.nunique()} levels")
print(f"  Roughness: {len(df_roughness)} values across {df_roughness.roughness.nunique()} levels")
print(f"  Slope: {len(df_slope)} values across {df_slope.slope.nunique()} levels")

# %%
###############################################################################
### -- CREATE PARAMETERS FIGURE: (KAPPA, LAMBDA) FOR DISSAGREGATED LAYERS-- ###
###############################################################################

print("\nCreating figure...")

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.25)

# ============== ROW 1: KAPPA (SHAPE PARAMETER) ==============

ax1                = plt.subplot(gs[0, 0])
data_kappa_climate = [df_climate[df_climate.climate == c]['kappa'].values for c in climate_order]
bp1                = ax1.boxplot(data_kappa_climate, positions=range(len(climate_order)), widths=0.6, patch_artist=True, showfliers=True,
                   flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, climate in zip(bp1['boxes'], climate_order):
    patch.set_facecolor(climate_colors[climate])
    patch.set_alpha(0.7)
ax1.axhline(y=2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(0.02, 2.05, r'$\kappa=2$ (Rayleigh)', transform=ax1.get_yaxis_transform(), fontsize=9, color='gray') 
ax1.set_xticks(range(len(climate_order)))
ax1.set_xticklabels(climate_labels, fontsize=11)
ax1.set_ylabel(r'Shape parameter $\kappa$', fontsize=13)
ax1.set_title('a) Weibull Shape by Climate', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
# Add sample sizes
for i, c in enumerate(climate_order):
    n      = len(data_kappa_climate[i])
    median = np.median(data_kappa_climate[i]) if len(data_kappa_climate[i]) > 0 else np.nan
    # # ax1.text(i, ax1.get_ylim()[0]*0.7, f'n={n}', ha='center', va='top', fontsize=9, color='gray')
    if not np.isnan(median):
        ax1.text(i, median, f'{median:.2f}', ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

ax2              = plt.subplot(gs[0, 1])
data_kappa_rough = [df_roughness[df_roughness.roughness == r]['kappa'].values for r in roughness_order]
bp2              = ax2.boxplot(data_kappa_rough, positions=range(len(roughness_order)), widths=0.6, patch_artist=True, showfliers=True,
                  flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, rough in zip(bp2['boxes'], roughness_order):
    patch.set_facecolor(roughness_colors[rough])
    patch.set_alpha(0.7)
ax2.axhline(y=2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xticks(range(len(roughness_order)))
ax2.set_xticklabels(roughness_labels, fontsize=11)
ax2.set_ylabel(r' ', fontsize=12)
ax2.set_title('b) Weibull Shape by Roughness', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
# Add sample sizes
for i, r in enumerate(roughness_order):
    n      = len(data_kappa_rough[i])
    median = np.median(data_kappa_rough[i]) if len(data_kappa_rough[i]) > 0 else np.nan
    # # ax2.text(i, ax2.get_ylim()[0]*0.7, f'n={n}', ha='center', va='top', fontsize=9, color='gray')
    if not np.isnan(median):
        ax2.text(i, median, f'{median:.2f}', ha='center', va='bottom', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

ax3              = plt.subplot(gs[0, 2])
data_kappa_slope = [df_slope[df_slope.slope == s]['kappa'].values for s in slope_order]
bp3              = ax3.boxplot(data_kappa_slope, positions=range(len(slope_order)), widths=0.6, patch_artist=True, showfliers=True,
                  flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, slope in zip(bp3['boxes'], slope_order):
    patch.set_facecolor(topography_colors[slope])
    patch.set_alpha(0.7)
ax3.axhline(y=2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xticks(range(len(slope_order)))
ax3.set_xticklabels(topography_labels, fontsize=11)
ax3.set_ylabel(r' ', fontsize=12)
ax3.set_title('c) Weibull Shape by Topography', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
# Add sample sizes
for i, s in enumerate(slope_order):
    n      = len(data_kappa_slope[i])
    median = np.median(data_kappa_slope[i]) if len(data_kappa_slope[i]) > 0 else np.nan
    # # ax3.text(i, ax3.get_ylim()[0]*0.7, f'n={n}',ha='center', va='top', fontsize=9, color='gray')
    if not np.isnan(median):
        ax3.text(i, median, f'{median:.2f}', ha='center', va='bottom', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

# ============== ROW 2: LAMBDA (SCALE PARAMETER) ==============

ax4                 = plt.subplot(gs[1, 0])
data_lambda_climate = [df_climate[df_climate.climate == c]['lambda_scale'].values for c in climate_order]
bp4                 = ax4.boxplot(data_lambda_climate, positions=range(len(climate_order)), widths=0.6, patch_artist=True, showfliers=True,
                      flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, climate in zip(bp4['boxes'], climate_order):
    patch.set_facecolor(climate_colors[climate])
    patch.set_alpha(0.7)
ax4.set_xticks(range(len(climate_order)))
ax4.set_xticklabels(climate_labels, fontsize=11)
ax4.set_ylabel(r'Scale parameter $\lambda$ [m/s]', fontsize=13)
ax4.set_title('d) Weibull Scale by Climate', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, c in enumerate(climate_order):
    n      = len(data_lambda_climate[i])
    median = np.median(data_lambda_climate[i]) if len(data_lambda_climate[i]) > 0 else np.nan
    # # ax4.text(i, ax4.get_ylim()[0]*0.7, f'n={n}', ha='center', va='top', fontsize=9, color='gray')
    if not np.isnan(median):
        ax4.text(i, median, f'{median:.1f}', ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

ax5               = plt.subplot(gs[1, 1]) 
data_lambda_rough = [df_roughness[df_roughness.roughness == r]['lambda_scale'].values for r in roughness_order]
bp5               = ax5.boxplot(data_lambda_rough, positions=range(len(roughness_order)), widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, rough in zip(bp5['boxes'], roughness_order):
    patch.set_facecolor(roughness_colors[rough])
    patch.set_alpha(0.7)
ax5.set_xticks(range(len(roughness_order)))
ax5.set_xticklabels(roughness_labels, fontsize=11)
ax5.set_ylabel(r' ', fontsize=12)
ax5.set_title('e) Weibull Scale by Roughness', fontsize=13, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for i, r in enumerate(roughness_order):
    n      = len(data_lambda_rough[i])
    median = np.median(data_lambda_rough[i]) if len(data_lambda_rough[i]) > 0 else np.nan
    # # ax5.text(i, ax5.get_ylim()[0]*0.7, f'n={n}', ha='center', va='top', fontsize=9, color='gray')
    if not np.isnan(median):
        ax5.text(i, median, f'{median:.1f}', ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

ax6               = plt.subplot(gs[1, 2])
data_lambda_slope = [df_slope[df_slope.slope == s]['lambda_scale'].values for s in slope_order]
bp6               = ax6.boxplot(data_lambda_slope, positions=range(len(slope_order)), widths=0.6, patch_artist=True, showfliers=True,
                   flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, slope in zip(bp6['boxes'], slope_order):
    patch.set_facecolor(topography_colors[slope])
    patch.set_alpha(0.7)
ax6.set_xticks(range(len(slope_order)))
ax6.set_xticklabels(topography_labels, fontsize=11)
ax6.set_ylabel(r' ', fontsize=12)
ax6.set_title('f) Weibull Scale by Topography', fontsize=13, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
for i, s in enumerate(slope_order):
    n      = len(data_lambda_slope[i])
    median = np.median(data_lambda_slope[i]) if len(data_lambda_slope[i]) > 0 else np.nan
    # # ax6.text(i, ax6.get_ylim()[0]*0.7, f'n={n}', ha='center', va='top', fontsize=9, color='gray')
    if not np.isnan(median):
        ax6.text(i, median, f'{median:.1f}', ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')

plt.subplots_adjust(wspace=0.09, hspace=0.02, left=0.15, right=0.85, bottom=0.20, top=0.90)
plt.savefig(f"{bd_out_fig}BoxPlots_Weibull_Parameters_Disaggregated_SpatialCategories.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# %%

###############################################################################
###------------------------ PRINT SUMMARY STATISTICS------------------------###
###############################################################################

print("\n" + "="*70)
print("SUMMARY STATISTICS - WEIBULL PARAMETERS")
print("="*70)

print("\n--- BY ROUGHNESS ---")
for r in roughness_order:
    df_r = df_roughness[df_roughness.roughness == r]
    if len(df_r) > 0:
        print(f"\n{roughness_labels[r-1]}:")
        print(f"  κ (shape): {df_r.kappa.mean():.2f} ± {df_r.kappa.std():.2f} (median: {df_r.kappa.median():.2f})")
        print(f"  λ (scale): {df_r.lambda_scale.mean():.1f} ± {df_r.lambda_scale.std():.1f} m/s (median: {df_r.lambda_scale.median():.1f})")
        print(f"  n_annual: {df_r.n_annual.mean():.0f} ± {df_r.n_annual.std():.0f}")
        print(f"  Sample size: {len(df_r)} values")

print("\n--- BY CLIMATE (excluding R1) ---")
for c in climate_order:
    df_c = df_climate[df_climate.climate == c]
    if len(df_c) > 0:
        print(f"\n{climate_labels[c-1]}:")
        print(f"  κ: {df_c.kappa.mean():.2f} ± {df_c.kappa.std():.2f}")
        print(f"  λ: {df_c.lambda_scale.mean():.1f} ± {df_c.lambda_scale.std():.1f} m/s")
        print(f"  Sample size: {len(df_c)}")

print("\n" + "="*70)

# %%
########################################################################################
##---------LOADING THE COORDINATES TO BE USED TO ALOCATE THE RANDOM POINTS------------##
########################################################################################
# Abrir las coordenadas
with open(bd_out_tc + 'Random_coords_N100.pkl', 'rb') as f:
    coords_dict = pickle.load(f)

cat_r1 = 1  # Category R1 (water)

print("Loaded SMEV results")

###############################################################################
###----------------- EXTRACT R1 COORDINATES USING YOUR METHOD---------------###
###############################################################################

print("\n" + "="*70)
print("EXTRACTING R1 COORDINATES")
print("="*70)

# Extract coordinates for R1 only
sampled_idx = coords_dict[cat_r1]['idx']
sampled_idy = coords_dict[cat_r1]['idy']

band1_transform    = comblay.transform # Obtener la transformación geoespacial del raster
adjusted_transform = band1_transform * Affine.translation(1, 1) #shifts the transformation reference point.

lons_r1 = []
lats_r1 = []

for j in range(len(sampled_idx)):
    row, col = sampled_idx[j], sampled_idy[j]
    
    # Verify point has correct category value
    if band1[row, col] != cat_r1:
        print(f"Warning: Point ({row}, {col}) expected cat={cat_r1}, has value={band1[row, col]}")
        continue
    
    # Use rasterio transform to convert row/col to coordinates
    x, y = adjusted_transform * (col + 0.5, row + 0.5)
    lons_r1.append(x)
    lats_r1.append(y)

lons_r1 = np.array(lons_r1)
lats_r1 = np.array(lats_r1)

print(f"Extracted {len(lons_r1)} R1 points")
print(f"Longitude range: {lons_r1.min():.2f}°E - {lons_r1.max():.2f}°E")
print(f"Latitude range: {lats_r1.min():.2f}°N - {lats_r1.max():.2f}°N")

# %%
###############################################################################
###----------------------- DEFINE 3 WATER BODY GROUPS-----------------------###
###############################################################################

print("\n" + "="*70)
print("DEFINING WATER BODY GROUPS")
print("="*70)

# GROUP 1: ADRIATIC SEA
adriatic = Polygon([
    (12.0, 39.5),   # South (Strait of Otranto)
    (19.5, 39.5),   # SE corner
    (19.5, 42.0),   # Mid-east
    (18.5, 44.0),   # NE
    (16.5, 45.7),   # North (Trieste)
    (13.5, 45.7),   # NW (Venice)
    (12.5, 44.5),   # Central west
    (12.0, 43.0),   # SW
    (12.0, 39.5)    # Close
])

# GROUP 2: MEDITERRANEAN (Tyrrhenian, Ligurian, Western Med combined)
mediterranean = Polygon([
    (0.5, 39.0),    # Spanish coast
    (8.0, 37.5),    # South Sardinia/Sicily area
    (15.5, 38.5),   # Strait of Messina
    (16.0, 40.2),   # South Italy (before Adriatic starts)
    (14.5, 41.0),   # Naples area
    (12.5, 42.0),   # Rome area
    (11.5, 43.5),   # Tuscan coast
    (10.0, 44.5),   # Genoa/Liguria
    (8.0, 44.5),    # Nice/Monaco
    (7.0, 43.5),    # French Riviera
    (5.0, 43.5),    # Gulf of Lion
    (3.0, 42.5),    # French coast
    (1.0, 41.5),    # Catalonia
    (0.5, 40.0),    # East Spain
    (0.5, 39.0)     # Close
])

# GROUP 3: INLAND WATERS
# Everything else (alpine lakes, rivers, Po delta, lagoons)
# Will be classified as points NOT in Adriatic or Mediterranean

print("Defined 3 water body groups:")
print("  1. Adriatic Sea")
print("  2. Mediterranean (Tyrrhenian + Ligurian + Western Med)")
print("  3. Inland Waters (alpine lakes + rivers + other)")

###############################################################################
###--------------------- CLASSIFY R1 POINTS INTO 3 GROUPS-------------------###
###############################################################################

print("\n" + "="*70)
print("CLASSIFYING R1 POINTS")
print("="*70)

classifications = []

for i, (lon, lat) in enumerate(zip(lons_r1, lats_r1)):
    point = Point(lon, lat)
    
    if point.within(adriatic):
        classifications.append('Adriatic')
    elif point.within(mediterranean):
        classifications.append('Mediterranean')
    else:
        classifications.append('Inland_Waters')

# Create DataFrame
df_r1 = pd.DataFrame({'lon': lons_r1,'lat': lats_r1, 'group': classifications})

print("\nClassification results:")
for group in ['Adriatic', 'Mediterranean', 'Inland_Waters']:
    n   = (df_r1.group == group).sum()
    pct = 100 * n / len(df_r1)
    print(f"  {group:20s}: {n:3d} points ({pct:5.1f}%)")

# %%