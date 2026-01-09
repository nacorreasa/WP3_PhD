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

"""
Code to generate the extreme values for different return periods , absed on SMEV and other traditional 
methods for the time series extracted from the soatial categories. 

Author : Nathalia Correa-Sánchez
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

bd_in_ws     = "/Dati/Data/WS_CORDEXFPS/"
bd_out_fig   = "/Dati/Outputs/Plots/WP3_development/"
bd_out_rl    = "/Dati/Outputs/RL_ws100m/"
bd_out_tc    = "/Dati/Outputs/WP3_SamplingSeries_CPM/" 
bd_in_rast   = "/Dati/Outputs/Climate_Provinces/Development_Rasters/FinalRasters_In-Out/"  # Antes : Combined_RIX_remCPM_WGS84.tif

########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

filas_eliminar    = [0]  # Primera  fila, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada 
columnas_eliminar = [0]  # Primera columna, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada
rp_idx            = 0    # CAMBIAR ESTE: OJOOOOOOOO Índice para el período de retorno !!!!! (VER LA LISTA DE LOS PERIODOS DE RETORNO)
ret_per           = 2    # CAMBIAR ESTE: Valor para etiquetas
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
    for i in range(len(return_periods)):
        return_period = return_periods[i]

        # Initialize SMEV object
        rl_arr = SMEV(threshold_measure, separation, return_period, durations, time_resolution)

        # Estimate SMEV parameters
        shape_new, scale_new = rl_arr.estimate_smev_parameters(ordinary_events_new_df['value'], data_portion) # left_sensoring as None, [0, 1] it it using all the ordinary events

        # Calculate return values
        intensity = rl_arr.smev_return_values(rl_arr.return_period, shape_new, scale_new, n_ordinary)
        ###############################################################################

        return_levels.append(intensity)

    return_values = np.array(return_levels)

    return return_values, ordinary_events_new_df

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
        m0 = integrate.trapz(psd, freqs)
        m2 = integrate.trapz(psd * freqs**2, freqs)
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
    tau_d = np.trapz(rs_spline, lags)
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
    p_cens = 0.90
    smev_rl, df_ord_events_p = simple_mev_newsep(
        s_arr, full_range, return_periods, threshold_meas, 
        p_cor, durations, time_resolution, [p_cens, 1]
    )
    results['smev_rl'] = smev_rl
    
    print("### --- SMEV 10-pot results:")
    for period, value in zip(return_periods, smev_rl):
        print(f"Return period {period} years: {value:.2f}")
    
    # --- GEVD-PMM ---
    gev_rl, gev_param = gevd_fit(s_arr, return_periods)
    results['gev_rl'] = gev_rl
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
            eth_mean = np.mean(eth_values) if eth_values else np.nan
            eth_std = np.std(eth_values) if eth_values else np.nan
            cnrm_mean = np.mean(cnrm_values) if cnrm_values else np.nan
            cnrm_std = np.std(cnrm_values) if cnrm_values else np.nan
            cmcc_mean = np.mean(cmcc_values) if cmcc_values else np.nan
            cmcc_std = np.std(cmcc_values) if cmcc_values else np.nan
            
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
    cat_list = []
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

def custom_KGclimate_colormap(level):
    arids_color      = [255/255, 100/255, 0/255]    # Naranja intenso
    temperates_color = [100/255, 200/255, 100/255]  # Verde suave
    cold_color       = [50/255, 100/255, 255/255]   # Azul medio
    polar_color      = [150/255, 150/255, 150/255]  # Gris medio

    if level == 1:  # Arids
        return arids_color
    elif level == 2:  # Temperates
        return temperates_color
    elif level == 3:  # Cold
        return cold_color
    elif level == 4:  # Polar
        return polar_color
    else:
        return [0, 0, 0]  # Negro por defecto para niveles no definidos
    
# Función de colores contrastantes para Roughness y Slope Variance
def custom_level_colormap(level):
    colors = {
        1: [0/255, 0/255, 255/255],       # Very Low - Azul
        2: [0/255, 180/255, 0/255],       # Low - Verde
        3: [255/255, 0/255, 0/255],       # Moderate - Rojo
        4: [255/255, 165/255, 0/255],     # High - Naranja
        5: [128/255, 0/255, 128/255]      # Very High - Púrpura
    }
    return colors.get(level, [0, 0, 0])   # Negro por defecto

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

#########################################################################################
###------------RETURN LEVELS SAVING AND LOADING TO SPEED UP POST-PROCESSING-----------###
#########################################################################################

# ## Guardar en caso de que sea necesario para el futuro:
np.savez(os.path.join(bd_out_rl, "RL_allModels_SpatialCat.npz"), results=results_rl_summary)

## Abriendo los Rl que habia guardado  previamente:

npz_file_path      = os.path.join(bd_out_rl, "RL_allModels_SpatialCat.npz")
loaded_data        = np.load(npz_file_path, allow_pickle=True)
results_rl_summary = loaded_data['results'].item()

results_rl_eth  = results_rl_summary['eth']
results_rl_cnrm = results_rl_summary['cnrm']
results_rl_cmcc = results_rl_summary['cmcc']

#########################################################################################
###---------------TO EXPLORE THE DATA ESTRUCTURE OF THE RETURN LEVELS-----------------###
#########################################################################################

# Para explorar la estructura
for key in results_rl_summary.keys():
    print(f"Clave: {key}, Tipo: {type(results_rl_summary[key])}")
    
    # Si es un diccionario, explora un nivel más
    if isinstance(results_rl_summary[key], dict):
        for subkey in results_rl_summary[key].keys():
            print(f"  Subclave: {subkey}, Tipo: {type(results_rl_summary[key][subkey])}")

#########################################################################################
###------VISUALIZATION EXTREME VALUES PER EACH POINT IN ALL SPATIAL CATEGORIES--------###
#########################################################################################

# Definir los códigos y sus nombres descriptivos
climate_names   = {1:'Ar', 2:'Tm', 3:'Co', 4:'Td'}
roughness_names = {1:r"$R_1$:(water)", 2: r"$R_2$", 3: r"$R_3$", 4: r"$R_4$", 5:r"$R_4$"}
slope_names     = {1:r"$T_1$", 2:r"$T_2$", 3:r"$T_3$", 4:r"$T_4$"}

df = prepare_visualization_data(rp_idx)
df = df.sort_values(by=['Climate', 'Roughness', 'Slope'])

x_labels = []
for row in df.itertuples():
    label = create_descriptive_label(row.Climate, row.Roughness, row.Slope)
    x_labels.append(label)

x          = np.arange(len(df))
width      = 0.25
marker_siz = 90

# Gráfico de valroes medios y barras de std de RL  con marcadores para plotting positions
fig, (ax1) = plt.subplots(1, 1, figsize=(14, 8))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(which='both', direction='in')
ax1.yaxis.set_minor_locator(AutoMinorLocator())
plt.scatter(x - width, df['ETH_Mean'], s=marker_siz, color='#edae49', marker='s', label='ETH (Mean)', alpha=0.8, linewidths=1, edgecolors='black')
plt.scatter(x, df['CNRM_Mean'], s=marker_siz, color='#00798c', marker='s', label='CNRM (Mean)', alpha=0.8, linewidths=1, edgecolors='black')
plt.scatter(x + width, df['CMCC_Mean'], s=marker_siz, color='#d1495b',  marker='s', label='CMCC (Mean)', alpha=0.8, linewidths=1, edgecolors='black')
plt.errorbar(x - width, df['ETH_Mean'], yerr=df['ETH_Std'], fmt='none',ecolor='#edae49', capsize=5, alpha=0.9)
plt.errorbar(x, df['CNRM_Mean'], yerr=df['CNRM_Std'], fmt='none', ecolor= '#00798c', capsize=5, alpha=0.9)
plt.errorbar(x + width, df['CMCC_Mean'], yerr=df['CMCC_Std'], fmt='none',  ecolor='#d1495b', capsize=5, alpha=0.9)
# Añadir marcadores para valores extremos de plotting positions
# Iteramos por cada categoría (índice en x)
for i, cat in enumerate(df['Category']):
    # Para cada modelo, obtener los valores extremos de plotting positions
    if cat in results_rl_eth:
        # Conseguir el promedio de los pp_annual_max para esta categoría y periodo de retorno
        eth_pp_values = []
        for res in results_rl_eth[cat]['results']:
            if 'pp_annual_max' in res and len(res['pp_annual_max']) > 0:
                # Ordenar los valores por periodo de retorno para asegurar correspondencia
                sorted_indices = np.argsort(res['pp_annual_max'])
                # Seleccionar el valor que corresponde aproximadamente al periodo de retorno deseado
                closest_idx = np.argmin(np.abs(res['pp_annual_max'] - ret_per))
                if 'annual_max' in res and len(res['annual_max']) > closest_idx:
                    eth_pp_values.append(res['annual_max'][closest_idx])
        
        eth_pp_mean = np.mean(eth_pp_values)
        plt.scatter(i - width, eth_pp_mean, color='grey', marker='X', s=50, )
    
    # Repetir para CNRM
    if cat in results_rl_cnrm:
        cnrm_pp_values = []
        for res in results_rl_cnrm[cat]['results']:
            if 'pp_annual_max' in res and len(res['pp_annual_max']) > 0:
                closest_idx = np.argmin(np.abs(res['pp_annual_max'] - ret_per))
                if 'annual_max' in res and len(res['annual_max']) > closest_idx:
                    cnrm_pp_values.append(res['annual_max'][closest_idx])
        
        cnrm_pp_mean = np.mean(cnrm_pp_values)
        plt.scatter(i, cnrm_pp_mean, color='grey', marker='X',s=50, )
    
    # Repetir para CMCC
    if cat in results_rl_cmcc:
        cmcc_pp_values = []
        for res in results_rl_cmcc[cat]['results']:
            if 'pp_annual_max' in res and len(res['pp_annual_max']) > 0:
                closest_idx = np.argmin(np.abs(res['pp_annual_max'] - ret_per))
                if 'annual_max' in res and len(res['annual_max']) > closest_idx:
                    cmcc_pp_values.append(res['annual_max'][closest_idx])
        
        cmcc_pp_mean = np.mean(cmcc_pp_values)
        plt.scatter(i + width, cmcc_pp_mean, color='grey', marker='X', s=50, label='Mean PP Value ' if i == 0 else "")

plt.xlabel('Spatial category', fontsize=12)
plt.ylabel(f'Wind Speed (m/s)', fontsize=12)
plt.title(f'Comparison of {ret_per}-year Return Levels by spatial category', fontsize=14, fontweight="bold", y=1.03)
plt.xticks(x, x_labels, rotation=30, ha='right')
plt.grid(True, axis='x', linestyle='--', alpha=0.3)
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.10, left=0.1, right=0.98, bottom=0.15, top=0.90)
# plt.savefig(bd_out_fig+'MeanRL_SMEV_AllSpatialCats_with_PP_'+str(ret_per)+'.png', dpi=300, bbox_inches='tight')
plt.show()

##############################################################################################
###-SINGLE MEMBER AND ENSEMBLE BOXPLOT EXTRATION OF EXTREME VALUES PER SPATIAL CATEGORIES-###
##############################################################################################

# Definir etiquetas descriptivas nuevas
climate_labels    = ['Ar', 'Tm', 'Co', 'Td']  # Valores en orden 1, 2, 3, 4 
roughness_labels  = [r"$R_1$:(water)", r"$R_2$", r"$R_3$", r"$R_4$", r"$R_5$"]  # Valores en orden 1-5
topography_labels = [r"$T_1$", r"$T_2$", r"$T_3$", r"$T_4$"]  # Valores en orden 1, 2, 3, 4

# Preparar datos para visualización
climate_data            = []
roughness_data          = []
slope_data              = []
ensemble_climate_data   = []
ensemble_roughness_data = []
ensemble_slope_data     = []

# Procesar cada categoría
for cat in fl_cats:
    if cat == 1:  # Caso especial para la categoría 1 (agua)
        climate   = None
        roughness = 1
        slope     = None
        
        climate_name   = None
        roughness_name = get_roughness_label(roughness)
        slope_name     = None
    else:
        climate, roughness, slope = decode_category(cat)
        
        climate_name   = get_climate_label(climate)
        roughness_name = get_roughness_label(roughness)
        slope_name     = get_slope_label(slope)
    
    # Datos para cada modelo en esta categoría
    eth_values  = []
    cnrm_values = []
    cmcc_values = []
    
    # Obtener datos ETH (TODOS LOS PUNTOS INDIVIDUALES)
    if cat in results_rl_eth and 'results' in results_rl_eth[cat]:
        for res in results_rl_eth[cat]['results']:
            if 'smev_rl' in res and len(res['smev_rl']) > rp_idx:
                eth_values.append(res['smev_rl'][rp_idx])
    
    # Obtener datos CNRM (TODOS LOS PUNTOS INDIVIDUALES)
    if cat in results_rl_cnrm and 'results' in results_rl_cnrm[cat]:
        for res in results_rl_cnrm[cat]['results']:
            if 'smev_rl' in res and len(res['smev_rl']) > rp_idx:
                cnrm_values.append(res['smev_rl'][rp_idx])
    
    # Obtener datos CMCC (TODOS LOS PUNTOS INDIVIDUALES)
    if cat in results_rl_cmcc and 'results' in results_rl_cmcc[cat]:
        for res in results_rl_cmcc[cat]['results']:
            if 'smev_rl' in res and len(res['smev_rl']) > rp_idx:
                cmcc_values.append(res['smev_rl'][rp_idx])
    
    # Calcular ensemble para cada punto (promedio de los 3 modelos)
    ensemble_values = []
    min_len = min(len(eth_values), len(cnrm_values), len(cmcc_values))
    for i in range(min_len):
        model_values = [eth_values[i], cnrm_values[i], cmcc_values[i]]
        ensemble_values.append(np.mean(model_values))
    
    # Agregar datos a las listas correspondientes por clima
    if climate_name is not None:
        for val in eth_values:
            climate_data.append({'Climate': climate_name, 'Climate_Label': climate_name, 'Model': 'ETH', 'Value': val})
        for val in cnrm_values:
            climate_data.append({'Climate': climate_name, 'Climate_Label': climate_name, 'Model': 'CNRM', 'Value': val})
        for val in cmcc_values:
            climate_data.append({'Climate': climate_name, 'Climate_Label': climate_name, 'Model': 'CMCC', 'Value': val})
        for val in ensemble_values:
            ensemble_climate_data.append({'Climate': climate_name, 'Climate_Label': climate_name, 'Value': val})
    
    # Agregar datos a las listas correspondientes por rugosidad
    for val in eth_values:
        roughness_data.append({'Roughness': roughness_name, 'Roughness_Label': roughness_name, 'Model': 'ETH', 'Value': val})
    for val in cnrm_values:
        roughness_data.append({'Roughness': roughness_name, 'Roughness_Label': roughness_name, 'Model': 'CNRM', 'Value': val})
    for val in cmcc_values:
        roughness_data.append({'Roughness': roughness_name, 'Roughness_Label': roughness_name, 'Model': 'CMCC', 'Value': val})
    for val in ensemble_values:
        ensemble_roughness_data.append({'Roughness': roughness_name, 'Roughness_Label': roughness_name, 'Value': val})
    
    # Agregar datos a las listas correspondientes por pendiente
    if slope_name is not None:
        for val in eth_values:
            slope_data.append({'Slope': slope_name, 'Slope_Label': slope_name, 'Model': 'ETH', 'Value': val})
        for val in cnrm_values:
            slope_data.append({'Slope': slope_name, 'Slope_Label': slope_name, 'Model': 'CNRM', 'Value': val})
        for val in cmcc_values:
            slope_data.append({'Slope': slope_name, 'Slope_Label': slope_name, 'Model': 'CMCC', 'Value': val})
        for val in ensemble_values:
            ensemble_slope_data.append({'Slope': slope_name, 'Slope_Label': slope_name, 'Value': val})

# Convertir listas a DataFrames
climate_df            = pd.DataFrame(climate_data)
roughness_df          = pd.DataFrame(roughness_data)
slope_df              = pd.DataFrame(slope_data)
ensemble_climate_df   = pd.DataFrame(ensemble_climate_data)
ensemble_roughness_df = pd.DataFrame(ensemble_roughness_data)
ensemble_slope_df     = pd.DataFrame(ensemble_slope_data)

# Definir órdenes para las categorías usando las etiquetas descriptivas
climate_order_desc   = climate_labels     # ['Ar', 'Tm', 'Co', 'Td']
roughness_order_desc = roughness_labels   # [r"$R_1$:(water)", r"$R_2$", r"$R_3$", r"$R_4$", r"$R_5$"]
slope_order_desc     = topography_labels  # [r"$T_1$", r"$T_2$", r"$T_3$", r"$T_4$"]

# Contar el número de puntos por categoría usando las etiquetas descriptivas
# Cada entrada en ensemble_climate_df representa UN PUNTO, no un promedio por categoría
climate_counts   = ensemble_climate_df['Climate_Label'].value_counts().reindex(climate_order_desc)
roughness_counts = ensemble_roughness_df['Roughness_Label'].value_counts().reindex(roughness_order_desc)
slope_counts     = ensemble_slope_df['Slope_Label'].value_counts().reindex(slope_order_desc)

# Rellenar valores NaN con cero para evitar errores en los gráficos -en las categorias donde no hay datos
climate_count    = climate_counts.fillna(0)
roughness_counts = roughness_counts.fillna(0)
slope_counts     = slope_counts.fillna(0)


##############################################################################################
###----APPLING BOOSTRAP-CI TO THE MEAN RETURN LEVELS OF THE ENSEMBLE PER SPATIAL CATEGORIES---###
##############################################################################################

# Aplicar bootstrap a los DataFrames de ensemble
n_bootstrap = 1000  # Número de muestras bootstrap

# Calcular intervalos de confianza para clima
climate_ci = add_bootstrap_ci_to_df(ensemble_climate_df, 'Climate', 'Value', n_bootstrap)

# Calcular intervalos de confianza para rugosidad
roughness_ci = add_bootstrap_ci_to_df(ensemble_roughness_df, 'Roughness', 'Value', n_bootstrap)

# Calcular intervalos de confianza para pendiente
slope_ci = add_bootstrap_ci_to_df(ensemble_slope_df, 'Slope', 'Value', n_bootstrap)

##############################################################################################
###-SINGLE MEMBER AND ENSEMBLE BOXPLOT VISUALIZATION EXTREME VALUES PER SPATIAL CATEGORIES-###
##############################################################################################

# Paleta de colores para los modelos
colors = {'ETH': '#edae49', 'CNRM': '#00798c', 'CMCC': '#d1495b', 'Ensemble': '#9b19f5'}

fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.5]) # Tercera fila mas pequeña
# ---- FILA 1: Boxplots por modelo ----
# Columna 1: Clima
ax1 = plt.subplot(gs[0, 0])
sns.boxplot(x='Climate', y='Value', hue='Model', data=climate_df, palette=colors, order=climate_order_desc, ax=ax1)
ax1.set_title('Climate', fontsize=13, fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylabel(r'$U_{' + str(ret_per) + r'}$ [m s$^{-1}$]', fontsize=12.5)
ax1.legend([],[], frameon=False)  # Ocultar leyenda

# Columna 2: Roughness
ax2 = plt.subplot(gs[0, 1])
sns.boxplot(x='Roughness', y='Value', hue='Model', data=roughness_df, palette=colors, order=roughness_order_desc, ax=ax2)
ax2.set_title('Roughness', fontsize=13, fontweight='bold')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.legend([],[], frameon=False)  # Ocultar leyenda

# Columna 3: Slope
ax3 = plt.subplot(gs[0, 2]) 
sns.boxplot(x='Slope', y='Value', hue='Model', data=slope_df, palette=colors, order=slope_order_desc, ax=ax3)
ax3.set_title('Topography', fontsize=13, fontweight='bold')
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.legend([],[], frameon=False)  # Ocultar leyenda

# Synchronize limits, major and minor ticks for row 1
y_min_row1 = min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0])
y_max_row1 = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1])
ax1.set_ylim(3, y_max_row1)
ax2.set_ylim(3, y_max_row1)
ax3.set_ylim(3, y_max_row1)
major_ticks_row1 = np.arange(10, y_max_row1 + 10, 10)  
minor_ticks_row1 = np.arange(5, y_max_row1 + 5, 5)    
for ax in [ax1, ax2, ax3]:
    ax.set_yticks(major_ticks_row1)
    ax.set_yticks(minor_ticks_row1, minor=True)

# ---- FILA 2: Ensemble ----
# Columna 1: Clima (Ensemble)
ax4 = plt.subplot(gs[1, 0])
sns.boxplot(x='Climate', y='Value', data=ensemble_climate_df, color=colors['Ensemble'], order=climate_order_desc, ax=ax4)
for patch in ax4.patches: ## Para la transparencia de las boxes
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.6))
# Añadir intervalo de confianza como rectángulos sombreados
for i, climate in enumerate(climate_order_desc):
    if climate in climate_ci['Climate'].values:
        ci_row = climate_ci[climate_ci['Climate'] == climate].iloc[0]
        lower  = ci_row['ci_lower']
        upper  = ci_row['ci_upper']

        ax4.add_patch(plt.Rectangle((i - 0.4, lower),  # Posición (x, y)
                                    0.8,               # Ancho
                                    upper - lower,     # Alto
                                    alpha=0.7,         # Transparencia
                                    color='gray',      # Color
                                    zorder=5))         # Capa (detrás del boxplot)
ax4.set_ylabel(r'$U_{' + str(ret_per) + r'}$ [m s$^{-1}$]', fontsize=12.5)
ax4.set_xlabel('')

# Columna 2: Roughness (Ensemble)
ax5 = plt.subplot(gs[1, 1])
sns.boxplot(x='Roughness', y='Value', data=ensemble_roughness_df, color=colors['Ensemble'], order=roughness_order_desc, ax=ax5)
for patch in ax5.patches: ## Para la transparencia de las boxes
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.6))
for patch in ax5.artists: ## Para la transparencia de las boxes
    fc = patch.get_facecolor()
    patch.set_facecolor((fc, 0.6))
# Añadir intervalo de confianza
for i, roughness in enumerate(roughness_order_desc):
    if roughness in roughness_ci['Roughness'].values:
        ci_row = roughness_ci[roughness_ci['Roughness'] == roughness].iloc[0]
        lower  = ci_row['ci_lower']
        upper  = ci_row['ci_upper']
        ax5.add_patch(plt.Rectangle((i - 0.4, lower),  # Posición
                                    0.8,               # Ancho
                                    upper - lower,     # Alto
                                    alpha=0.7,         # Transparencia
                                    color='gray',      # Color
                                    zorder=5 ))        # Capa (detrás del boxplot)        
ax5.set_ylabel('')
ax5.set_xlabel('')

# Columna 3: Slope (Ensemble)
ax6 = plt.subplot(gs[1, 2])
sns.boxplot(x='Slope', y='Value', data=ensemble_slope_df, color=colors['Ensemble'], order=slope_order_desc, ax=ax6)
for patch in ax6.patches: ## Para la transparencia de las boxes
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.6))
# Añadir intervalo de confianza
for i, slope in enumerate(slope_order_desc):
    if slope in slope_ci['Slope'].values:
        ci_row = slope_ci[slope_ci['Slope'] == slope].iloc[0]
        lower  = ci_row['ci_lower']
        upper  = ci_row['ci_upper']
        ax6.add_patch(plt.Rectangle((i - 0.4, lower),  # Posición
                                    0.8,               # Ancho
                                    upper - lower,     # Alto
                                    alpha=0.7,         # Transparencia
                                    color='gray',      # Color
                                    zorder=5))         # Capa (detrás del boxplot)   
ax6.set_ylabel('')
ax6.set_xlabel('')

# Set same fixed limits for all subplots in row 2
y_min_fixed = 10
y_max_fixed = 50
ax4.set_ylim(9, y_max_fixed)
ax5.set_ylim(9, y_max_fixed)
ax6.set_ylim(9, y_max_fixed)
major_ticks_row2 = np.arange(y_min_fixed, y_max_fixed + 1, 10)  
minor_ticks_row2 = np.arange(y_min_fixed, y_max_fixed + 1, 5)   
for ax in [ax4, ax5, ax6]:
    ax.set_yticks(major_ticks_row2)
    ax.set_yticks(minor_ticks_row2, minor=True)


# ---- FILA 3: Conteo de puntos por categoría ----
# Columna 1: Conteo para Clima
ax7  = plt.subplot(gs[2, 0])
bars = ax7.bar(range(len(climate_counts)), climate_counts, color='lightgray', edgecolor='black')
# Añadir etiquetas con el conteo sobre cada barra
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=12)
ax7.set_xticks(range(len(climate_order_desc)))
ax7.set_xticklabels(climate_order_desc, fontsize=12.5)
ax7.set_ylabel('Number\nof points', fontsize=12.5)
ax7.set_xlabel('Climate level', fontsize=12.5)
ax7.grid(axis='y', linestyle='--', alpha=0.3)

# Columna 2: Conteo para Roughness
ax8 = plt.subplot(gs[2, 1])
bars = ax8.bar(range(len(roughness_counts)), roughness_counts, color='lightgray', edgecolor='black')
# Añadir etiquetas con el conteo
for bar in bars:
    height = bar.get_height()
    if np.isnan(height):
        ax8.text(bar.get_x() + bar.get_width()/2., 0.5, '0', ha='center', va='bottom', fontsize=12)
    else:
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=12)
ax8.set_xticks(range(len(roughness_order_desc)))
ax8.set_xticklabels(roughness_order_desc, fontsize=12.5)
ax8.set_xlabel('Roughness level', fontsize=12.5)
ax8.set_ylabel('')
ax8.grid(axis='y', linestyle='--', alpha=0.3)

# Columna 3: Conteo para Slope
ax9 = plt.subplot(gs[2, 2])
bars = ax9.bar(range(len(slope_counts)), slope_counts, color='lightgray', edgecolor='black')
# Añadir etiquetas con el conteo
for bar in bars:
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=12)
ax9.set_xticks(range(len(slope_order_desc)))
ax9.set_xticklabels(slope_order_desc, fontsize=12.5)
ax9.set_xlabel('Topography level', fontsize=12.5)
ax9.set_ylabel('')
ax9.grid(axis='y', linestyle='--', alpha=0.3)

# Synchronize limits and ticks for row 3
y_min_fixed = 0
y_max_fixed = 1000
major_ticks_row3 = np.array([0, 500, 1000])
for ax in [ax7, ax8, ax9]:
    ax.set_ylim(y_min_fixed, y_max_fixed)
    ax.set_yticks(major_ticks_row3)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# ---- Ajustes generales finales ----#
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    ax.text(0.08, 1.001, subplot_labels[i], transform=ax.transAxes, fontsize=14, fontweight='bold', ha='right', va='top')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xticklabels([]) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)

for ax in [ax2, ax3, ax5, ax6, ax8, ax9]:
    ax.set_yticklabels([])

for ax in [ax1, ax4, ax7]:
    ax.tick_params(axis='y', labelsize=12)

legend_handles = []
legend_labels  = []
# Crear patches para cada modelo
for model, color in colors.items():
    if model != 'Ensemble':  # Solo para los 3 modelos, no para Ensemble
        patch = Patch(color=color, label=model)
        legend_handles.append(patch)
        legend_labels.append(model)        
# Añadir patch para Ensemble, descompuesto en RGB para poder modificar el color ocn el alpha
ensemble_rgba = mcolors.to_rgba(colors['Ensemble'], alpha=0.6)
patch         = Patch(facecolor=ensemble_rgba, label='Ensemble')
legend_handles.append(patch)
legend_labels.append('Ensemble')
confidence_patch = Patch(color='gray', alpha=0.3, label='95% CI (Bootstrap)')
legend_handles.append(confidence_patch)
legend_labels.append('95% CI (Bootstrap)')
fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=5, fontsize=12.5)

# fig.suptitle(f'{ret_per}-year Return Levels by Spatial Factors (SMEV)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(wspace=0.10, hspace=0.10, left=0.1, right=0.98, bottom=0.15, top=0.90)
plt.savefig(bd_out_fig+'RL_Boxplots_SMEV_SpatialCategories_CPM&Ensemble_'+str(ret_per)+'.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

############################################################################################
###------ENSEMBLE FREQUENCY VISUALIZATION EXTREME VALUES PER SPATIAL CATEGORIES----------###
############################################################################################

# Crear diccionario para mapear nombres a niveles
level_mapping = {'Very Low' : 1,
                 'Low'      : 2,
                 'Moderate' : 3,
                 'High'     : 4,
                 'Very High': 5}

climate_numbers = { 'Arid'     : 1,
                    'Temperate': 2,
                    'Cold'     : 3,
                    'Polar'    : 4}

rp_idx  = 4  # Índice para el período de retorno de 50 años
ret_per = 50  # Valor para etiquetas
y_limit = 0.31

fig = plt.figure(figsize=(18, 6))
gs  = gridspec.GridSpec(1, 3)

# Columna 1: Clima (Densidad)
# Encontrar valores mínimos y máximos para el eje x
all_climate_values = ensemble_climate_df['Value'].values
x_min = min(all_climate_values) - 2
x_max = max(all_climate_values) + 2
ax1   = plt.subplot(gs[0, 0])
# Graficar densidad para cada clima con colores personalizados
for climate in climate_order:
    climate_values = ensemble_climate_df[ensemble_climate_df['Climate'] == climate]['Value'].values
    if len(climate_values) > 1:  # Necesitamos al menos 2 puntos para KDE
        x, y = get_kde_values(climate_values, x_min, x_max)        
        # Obtener el número correspondiente al clima
        climate_num = climate_numbers.get(climate, 0)        
        # Obtener el color personalizado
        color       = custom_KGclimate_colormap(climate_num)        
        # Graficar la línea y el relleno con el color personalizado
        ax1.plot(x, y, label=climate, color=color)
        ax1.fill_between(x, y, alpha=0.2, color=color)

ax1.set_title('Climate', fontweight='bold')
ax1.set_xlabel('Wind Speed (m/s)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_ylim(0, y_limit)
ax1.legend()
ax1.grid(axis='both', linestyle='--', alpha=0.3)
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', direction='in')

# Columna 2: Roughness (Densidad)
# Encontrar valores mínimos y máximos para el eje x
all_roughness_values = ensemble_roughness_df['Value'].values
x_min = min(all_roughness_values) - 2
x_max = max(all_roughness_values) + 2
ax2   = plt.subplot(gs[0, 1])
# Graficar densidad para cada rugosidad con colores contrastantes
for roughness in roughness_order:
    roughness_values = ensemble_roughness_df[ensemble_roughness_df['Roughness'] == roughness]['Value'].values
    if len(roughness_values) > 1:
        x, y = get_kde_values(roughness_values, x_min, x_max)        
        # Obtener nivel y color
        level = level_mapping.get(roughness, 0)
        color = custom_level_colormap(level)        
        ax2.plot(x, y, label=roughness, color=color, linewidth=2)
        ax2.fill_between(x, y, alpha=0.15, color=color)

ax2.set_title('Roughness', fontweight='bold')
ax2.set_xlabel('Wind Speed (m/s)', fontsize=12)
ax2.set_ylabel('')
ax2.set_ylim(0, y_limit)
ax2.legend()
ax2.grid(axis='both', linestyle='--', alpha=0.3)
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='both', direction='in')


# Columna 3: Slope (Densidad)
# Encontrar valores mínimos y máximos para el eje x
all_slope_values = ensemble_slope_df['Value'].values
x_min = min(all_slope_values) - 2
x_max = max(all_slope_values) + 2
ax3   = plt.subplot(gs[0, 2])
# Graficar densidad para cada pendiente con colores contrastantes
for slope in slope_order:
    slope_values = ensemble_slope_df[ensemble_slope_df['Slope'] == slope]['Value'].values
    if len(slope_values) > 1:
        x, y = get_kde_values(slope_values, x_min, x_max)        
        # Obtener nivel y color
        level = level_mapping.get(slope, 0)
        color = custom_level_colormap(level)        
        ax3.plot(x, y, label=slope, color=color, linewidth=2)
        ax3.fill_between(x, y, alpha=0.15, color=color)

ax3.set_title('Slope Variance', fontweight='bold')
ax3.set_xlabel('Wind Speed (m/s)', fontsize=12)
ax3.set_ylabel('')
ax3.set_ylim(0, y_limit)
ax3.legend()
ax3.grid(axis='both', linestyle='--', alpha=0.3)
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(which='both', direction='in')


fig.suptitle(f'Frequency Distribution of {ret_per}-year Return Levels (Ensemble)', fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, left=0.1, right=0.98, bottom=0.20, top=0.80)
plt.savefig(bd_out_fig+'RL_Frequency_SMEV_SpatialCategories_Ensemble.png', dpi=300, bbox_inches='tight')
plt.show()


############################################################################################
###--------ENSEMBLE RETURN VALUES FOR ALL RETURN PERIODS PER SPATIAL CATEGORIES----------###
############################################################################################

# Preparar diccionarios para almacenar datos por categoría
climate_data   = {climate: np.zeros(len(return_periods)) for climate in climate_order}
roughness_data = {roughness: np.zeros(len(return_periods)) for roughness in roughness_order}
slope_data     = {slope: np.zeros(len(return_periods)) for slope in slope_order}

# Contadores para calcular promedios
climate_counts   = {climate: 0 for climate in climate_order}
roughness_counts = {roughness: 0 for roughness in roughness_order}
slope_counts     = {slope: 0 for slope in slope_order}

# Estructuras para los plotting positions
climate_pp   = {climate: {'x': [], 'y': []} for climate in climate_order}
roughness_pp = {roughness: {'x': [], 'y': []} for roughness in roughness_order}
slope_pp     = {slope: {'x': [], 'y': []} for slope in slope_order}

# Procesar datos por categoría
for cat in fl_cats:
    cat_climate, cat_roughness, cat_slope = decode_category(cat)
    climate_name   = climate_names[cat_climate]
    roughness_name = roughness_names[cat_roughness]
    slope_name     = slope_names[cat_slope]
    
    # Recolectar valores de smev_rl para cada punto, para todos los periodos de retorno
    for point_idx in range(len(results_rl_eth[cat]['results'])):
        if point_idx < len(results_rl_eth[cat]['results']) and point_idx < len(results_rl_cnrm[cat]['results']) and point_idx < len(results_rl_cmcc[cat]['results']):
            if 'smev_rl' in results_rl_eth[cat]['results'][point_idx] and 'smev_rl' in results_rl_cnrm[cat]['results'][point_idx] and 'smev_rl' in results_rl_cmcc[cat]['results'][point_idx]:
                eth_vals  = results_rl_eth[cat]['results'][point_idx]['smev_rl']
                cnrm_vals = results_rl_cnrm[cat]['results'][point_idx]['smev_rl']
                cmcc_vals = results_rl_cmcc[cat]['results'][point_idx]['smev_rl']
                
                # Verificar que tenemos valores para todos los periodos
                if len(eth_vals) >= len(return_periods) and len(cnrm_vals) >= len(return_periods) and len(cmcc_vals) >= len(return_periods):
                    # Calcular ensemble para cada periodo de retorno
                    for i, rp in enumerate(return_periods):
                        ensemble_val = np.mean([eth_vals[i], cnrm_vals[i], cmcc_vals[i]])
                        
                        # Sumar a los acumuladores
                        climate_data[climate_name][i]     += ensemble_val
                        roughness_data[roughness_name][i] += ensemble_val
                        slope_data[slope_name][i]         += ensemble_val
                    
                    # Incrementar contadores
                    climate_counts[climate_name]     += 1
                    roughness_counts[roughness_name] += 1
                    slope_counts[slope_name]         += 1
            
            # Recopilar plotting positions para este punto (solo de ETH)
            if 'annual_max' in results_rl_eth[cat]['results'][point_idx] and 'pp_annual_max' in results_rl_eth[cat]['results'][point_idx]:
                pp_periods = results_rl_eth[cat]['results'][point_idx]['pp_annual_max']
                pp_values = results_rl_eth[cat]['results'][point_idx]['annual_max']
                
                if len(pp_periods) == 10 and len(pp_values) == 10:
                    climate_pp[climate_name]['x'].extend(pp_periods)
                    climate_pp[climate_name]['y'].extend(pp_values)
                    
                    roughness_pp[roughness_name]['x'].extend(pp_periods)
                    roughness_pp[roughness_name]['y'].extend(pp_values)
                    
                    slope_pp[slope_name]['x'].extend(pp_periods)
                    slope_pp[slope_name]['y'].extend(pp_values)

# Calcular promedios dividiendo por los contadores
for climate in climate_order:
    if climate_counts[climate] > 0:
        climate_data[climate] = climate_data[climate] / climate_counts[climate]

for roughness in roughness_order:
    if roughness_counts[roughness] > 0:
        roughness_data[roughness] = roughness_data[roughness] / roughness_counts[roughness]

for slope in slope_order:
    if slope_counts[slope] > 0:
        slope_data[slope] = slope_data[slope] / slope_counts[slope]

y_min = 12  # Ajustar según los datos
y_max = 35  # Ajustar según los datos

fig = plt.figure(figsize=(18, 6))
gs  = gridspec.GridSpec(1, 3)

# Columna 1: Clima (Return Levels vs Return Period)
ax1 = plt.subplot(gs[0, 0])
for climate in climate_order:
    # Obtener color
    climate_num = climate_numbers[climate]
    color       = custom_KGclimate_colormap(climate_num)
    
    # Graficar línea de return levels
    if climate_counts[climate] > 0:
        ax1.plot(return_periods, climate_data[climate], color=color, linewidth=2, label=climate)
        ax1.scatter(return_periods, climate_data[climate], color=color, s=50, zorder=3)
    
    # # Graficar plotting positions
    # if climate_pp[climate]['x']:
    #     ax1.scatter(climate_pp[climate]['x'], climate_pp[climate]['y'], 
    #                color=color, s=80, alpha=0.7, marker='x', linewidth=2)

ax1.set_title('Climate', fontweight='bold')
ax1.set_xlabel('Return Period (years)')
ax1.set_ylabel('Wind Speed (m/s)')
ax1.set_xscale('log')
ax1.grid(True, which="both", ls="-", alpha=0.3)
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', direction='in')
ax1.legend()

# Repetir para roughness y slope
ax2 = plt.subplot(gs[0, 1])
for roughness in roughness_order:
    roughness_num = level_mapping[roughness]
    color         = custom_level_colormap(roughness_num)
    
    if roughness_counts[roughness] > 0:
        ax2.plot(return_periods, roughness_data[roughness], color=color, linewidth=2, label=roughness)
        ax2.scatter(return_periods, roughness_data[roughness], color=color, s=50, zorder=3)
        
    # if roughness_pp[roughness]['x']:
    #     ax2.scatter(roughness_pp[roughness]['x'], roughness_pp[roughness]['y'], 
    #                color=color, s=80, alpha=0.7, marker='x', linewidth=2)

ax2.set_title('Roughness', fontweight='bold')
ax2.set_xlabel('Return Period (years)')
ax2.set_xscale('log')
ax2.grid(True, which="both", ls="-", alpha=0.3)
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='both', direction='in')
ax2.legend()

ax3 = plt.subplot(gs[0, 2])
for slope in slope_order:
    slope_num = level_mapping[slope]
    color     = custom_level_colormap(slope_num)
    
    if slope_counts[slope] > 0:
        ax3.plot(return_periods, slope_data[slope], color=color, linewidth=2, label=slope)
        ax3.scatter(return_periods, slope_data[slope], color=color, s=50, zorder=3)

ax3.set_title('Slope Variance', fontweight='bold')
ax3.set_xlabel('Return Period (years)')
ax3.set_xscale('log')
ax3.grid(True, which="both", ls="-", alpha=0.3)
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(which='both', direction='in')
ax3.legend()


for ax in [ax1, ax2, ax3]:
    ax.set_ylim(y_min, y_max)
    ax.set_xscale('log')
    ax.set_xticks(return_periods)
    ax.set_xticklabels(return_periods, fontsize=10)
fig.suptitle('Return Levels vs Return Period by Spatial Factors (Ensemble Average)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, left=0.1, right=0.98, bottom=0.20, top=0.80)
plt.savefig(bd_out_fig+'RL_vs_ReturnPeriod_SMEV_SpatialCategories_Ensemble.png', dpi=300, bbox_inches='tight')
plt.show()



############################################################################################
###------------------------NORMALIZED LINES PER SPATIAL CATEGORY-------------------------###
############################################################################################

# Definir etiquetas descriptivas y colores
water_color  = '#345F77'
titles       = ['a) Climate', 'b) Roughness', 'c) Topography']
colour_clima = ['#F5A500', '#64C864', '#4B92DB', '#A8A8A8']              # Valores en orden 1, 2, 3, 4 
colour_rough = [water_color, '#FFDA8A', '#F2A05D', '#D95D30', '#8C2A04'] # Valores en orden 1-mar, 2, 3, 4, 5 
colour_topog = ['#E5E0CB', '#B8D6BE', '#73AE80', '#2A6B3D']              # Valores en orden 1, 2, 3, 4

# Definir etiquetas descriptivas
climate_labels    = ['Ar', 'Tm', 'Co', 'Td']  # Valores en orden 1, 2, 3, 4 
roughness_labels  = [r"$R_1$:(water)", r"$R_2$", r"$R_3$", r"$R_4$", r"$R_5$"]  # Valores en orden 1-5
topography_labels = [r"$T_1$", r"$T_2$", r"$T_3$", r"$T_4$"]  # Valores en orden 1, 2, 3, 4

# Crear diccionarios para mapear códigos a etiquetas y colores
climate_map   = {i+1: {'label': climate_labels[i], 'color': colour_clima[i]} for i in range(len(climate_labels))}
roughness_map = {i+1: {'label': roughness_labels[i], 'color': colour_rough[i]} for i in range(len(roughness_labels))}
slope_map     = {i+1: {'label': topography_labels[i], 'color': colour_topog[i]} for i in range(len(topography_labels))}

# Función para decodificar categorías con manejo especial para cat=1
def decode_category(cat_code):
    if cat_code == 1:
        return None, 1, None
    
    cat_str = str(cat_code).zfill(3)
    climate = int(cat_str[0])
    roughness = int(cat_str[1])
    slope = int(cat_str[2])
    return climate, roughness, slope

# Preparar diccionarios para acumular valores por categoría y periodo de retorno
climate_values   = {climate: {period: [] for period in return_periods} for climate in climate_labels}
roughness_values = {roughness: {period: [] for period in return_periods} for roughness in roughness_labels}
slope_values     = {slope: {period: [] for period in return_periods} for slope in topography_labels}

# Metodo sobre el cual computar la normalizacion: 'smev_rl', 'gev_rl', 'gpd_rl', 'spcr_rl'
method_rl     = 'gev_rl'
title_method  = 'GEV'

for cat in fl_cats:
    # Decodificar categoría
    climate, roughness, slope = decode_category(cat)
    
    # Obtener etiquetas descriptivas
    climate_label   = climate_map[climate]['label'] if climate is not None else None
    roughness_label = roughness_map[roughness]['label']
    slope_label     = slope_map[slope]['label'] if slope is not None else None
    
    # Obtener datos para cada punto en esta categoría
    if cat in results_rl_eth and 'results' in results_rl_eth[cat]:
        for point_idx in range(len(results_rl_eth[cat]['results'])):
            if (point_idx < len(results_rl_eth[cat]['results']) and 
                point_idx < len(results_rl_cnrm[cat]['results']) and 
                point_idx < len(results_rl_cmcc[cat]['results'])):
                
                # Obtener valores para los tres modelos
                eth_vals  = results_rl_eth[cat]['results'][point_idx][method_rl]
                cnrm_vals = results_rl_cnrm[cat]['results'][point_idx][method_rl]
                cmcc_vals = results_rl_cmcc[cat]['results'][point_idx][method_rl]
                
                if len(eth_vals) >= len(return_periods) and len(cnrm_vals) >= len(return_periods) and len(cmcc_vals) >= len(return_periods):
                    # Promediar los tres modelos para cada periodo de retorno
                    ensemble_vals = []
                    for i in range(len(return_periods)):
                        ensemble_val = np.mean([eth_vals[i], cnrm_vals[i], cmcc_vals[i]])
                        ensemble_vals.append(ensemble_val)
                    
                    # Normalizar por el valor de 2 años
                    if ensemble_vals[0] > 0:  # Asegurar que no hay división por cero
                        normalized_vals = [val / ensemble_vals[0] for val in ensemble_vals]
                        
                        # Agregar a las listas correspondientes
                        if climate_label:
                            for i, period in enumerate(return_periods):
                                climate_values[climate_label][period].append(normalized_vals[i])
                        
                        for i, period in enumerate(return_periods):
                            roughness_values[roughness_label][period].append(normalized_vals[i])
                        
                        if slope_label:
                            for i, period in enumerate(return_periods):
                                slope_values[slope_label][period].append(normalized_vals[i])

# Calcular promedios por categoría y periodo
climate_means   = {climate: {period: np.mean(values) if values else 1.0 for period, values in periods.items()} for climate, periods in climate_values.items()}
roughness_means = {roughness: {period: np.mean(values) if values else 1.0 for period, values in periods.items()}   for roughness, periods in roughness_values.items()}
slope_means     = {slope: {period: np.mean(values) if values else 1.0 for period, values in periods.items()} for slope, periods in slope_values.items()}

fig = plt.figure(figsize=(18, 6))
gs  = gridspec.GridSpec(1, 3)

# Columna 1: Clima
ax1 = plt.subplot(gs[0, 0])
for i, climate in enumerate(climate_labels):
    if climate in climate_means:
        means = [climate_means[climate][period] for period in return_periods]
        ax1.plot(return_periods, means, '-o', color=colour_clima[i], linewidth=2, markersize=8, label=climate)
ax1.set_title(titles[0], fontweight='bold')
ax1.set_ylabel('RL / RL(2yr)')
ax1.set_xscale('log')
ax1.set_xticks(return_periods)
ax1.set_xticklabels(return_periods)
ax1.grid(True, which="both", ls="-", alpha=0.3)
ax1.set_ylim(0.95, 1.6)
ax1.legend(loc='upper left')
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', direction='in')

# Columna 2: Roughness
ax2 = plt.subplot(gs[0, 1])
for i, roughness in enumerate(roughness_labels):
    if roughness in roughness_means and roughness != '$R_2$': ## Porque no tenemos datos de R2
        means = [roughness_means[roughness][period] for period in return_periods]
        ax2.plot(return_periods, means, '-o', color=colour_rough[i],linewidth=2, markersize=8, label=roughness)
    else:
        pass
ax2.set_title(titles[1], fontweight='bold')
ax2.set_xscale('log')
ax2.set_xticks(return_periods)
ax2.set_xticklabels(return_periods)
ax2.grid(True, which="both", ls="-", alpha=0.3)
ax2.set_ylim(0.95, 1.6)
ax2.legend(loc='upper left')
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='both', direction='in')

# Columna 3: Slope/Topography
ax3 = plt.subplot(gs[0, 2])
for i, slope in enumerate(topography_labels):
    if slope in slope_means:
        means = [slope_means[slope][period] for period in return_periods]
        ax3.plot(return_periods, means, '-o', color=colour_topog[i], linewidth=2, markersize=8, label=slope)
ax3.set_title(titles[2], fontweight='bold')
ax3.set_xscale('log')
ax3.set_xticks(return_periods)
ax3.set_xticklabels(return_periods)
ax3.grid(True, which="both", ls="-", alpha=0.3)
ax3.set_ylim(0.95, 1.6)
ax3.legend(loc='upper left')
ax3.yaxis.set_minor_locator(AutoMinorLocator())
for ax in [ax1, ax2, ax3]: 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Return Period [years]')

fig.suptitle('Normalized return levels-'+title_method+'- (Ensemble average)', fontsize=16, fontweight='bold', y=0.98) 
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, left=0.05, right=0.98, bottom=0.15, top=0.85)
plt.savefig(bd_out_fig + 'RL_Normalized_Lines_by_SpatialCategories_'+method_rl+'.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()