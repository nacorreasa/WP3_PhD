import xarray as xr
import numpy as np
import pandas as pd
import os,glob,sys
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from smev_class import SMEV
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.stats import genextreme
from scipy.stats import rankdata
from scipy import integrate
from scipy.stats import gumbel_r

"""
Codigo para la exploracion y prueba de los tiempos de autocorrelacion, el ajuste Weibull
con left-censoring, la comparacion de RL estimados por diferentes metodos, y la generacion 
del espectro de potencia a partir de 12 series aleatorias del ensamble de datos CPM. 
El código tambien genera el mapa con la localizacion de los puntos aleatorios seleccionados. 

Author : Nathalia Correa-Sánchez
"""

#############################################################################
##-------------------------DEFINING IMPORTANT PATHS------------------------##
#############################################################################
bd_in_ws   = "/Dati/Data/WS_CORDEXFPS/"
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"
bd_in_ese  = bd_in_ws + "Ensemble_Mean/wsa100m/"
bd_in_eth  = bd_in_ws + "ETH/wsa100m/"
bd_in_cmcc = bd_in_ws + "CMCC/wsa100m/"
bd_in_cnrm = bd_in_ws + "CNRM/wsa100m/"

#############################################################################
##-----------DEFINNING BOUNDARY COORDINATES TO PERFOMR THE TEST------------##
#############################################################################
min_lon =  0.689
min_lat =  43.521
max_lon =  8.478
max_lat =  46.067

#############################################################################
##-------------------------DEFINNING RELEVANT INPUTS-----------------------##
#############################################################################
years    = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009] 
corr_lim = 1/np.e
colors   = [
            '#7F00FF',  # Púrpura
            '#00FFFF',  # Cian
            '#00FF00',  # Verde
            '#FFFF00',  # Amarillo
            '#FF00FF',  # Magenta
            '#FFA500',  # Naranja
            '#800080',  # Morado oscuro
            '#008080',  # Verde azulado
            '#008000',  # Verde oscuro
            '#808000',  # Oliva
            '#800000',  # Marrón
            '#008080'   # Verde azulado
            ]

#############################################################################
##------------------------DEFINING RELEVANT FUNCTIONS----------------------##
#############################################################################

def cut_concat_netcdf_files(bd_in_ese, min_lon, max_lon, min_lat, max_lat):
    
    bd_ws    = f"{bd_in_ese}"
    filez_ws = sorted(glob.glob(f"{bd_ws}*.nc"))
    arr_cut  = []
    for file in filez_ws:
        # Open the dataset
        ds = xr.open_dataset(file)
        
        # Cut the dataset to the specified boundaries
        ds_cut = ds.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat)
        )
        
        # Append the cut dataset to the list
        arr_cut.append(ds_cut)
        
        # Close the original dataset to free up memory
        ds.close()
        print(file)
    
    # Combine all cut datasets into a single xarray Dataset
    combined_ds = xr.concat(arr_cut, dim='time')
    print("FILES COMBINED")

    # Correction of duplicated dates (ex: 1st Jan every year)
    combined_ds = combined_ds.sel(time=~combined_ds.get_index("time").duplicated()) 
    
    # Close individual cut datasets to free up memory
    for ds in arr_cut:
        ds.close()
    
    return combined_ds


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Taken from: https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))
    
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

def select_unique_pixels(sample_arr, num_pixels):
    """
    Selecciona aleatoriamente num_pixels píxeles únicos de un arreglo 3D (tiempo, latitud, longitud).
    
    Parámetros:
    sample_arr (numpy.ndarray): Arreglo 3D de datos a muestrear.
    num_pixels (int): Número de píxeles únicos a seleccionar (por defecto 12).
    
    Retorna:
    numpy.ndarray: Arreglo 2D con los num_pixels píxeles seleccionados.
    """
    time_dim, lat_dim, lon_dim = sample_arr.shape
    
    # Generar índices aleatorios sin repetición
    all_indices    = np.arange(lon_dim * lat_dim)
    np.random.shuffle(all_indices)
    unique_indices = all_indices[:num_pixels]
    
    # Convertir los índices 1D a índices 2D (longitud, latitud)
    lon_indices = (unique_indices // lat_dim).astype(int)
    lat_indices = (unique_indices % lat_dim).astype(int)
    
    # Seleccionar los píxeles únicos
    subset_arr = sample_arr[:, lat_indices, lon_indices]
    
    return subset_arr, lon_indices, lat_indices


def plot_selected_pixels_map(lat_indices, lon_indices, lat_array, lon_array):
    """
    Crea un mapa mostrando la ubicación de los píxeles seleccionados.
    
    Parámetros:
    lat_indices: índices de latitud de los píxeles seleccionados
    lon_indices: índices de longitud de los píxeles seleccionados
    lat_array: array completo de latitudes
    lon_array: array completo de longitudes
    """
    
    # Crear la figura y los ejes con la proyección deseada
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Añadir características del mapa
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    
    # Obtener las coordenadas reales de los píxeles seleccionados
    selected_lats = lat_array[lat_indices]
    selected_lons = lon_array[lon_indices]
    
    # Plotear los puntos seleccionados
    ax.plot(selected_lons, selected_lats, 'rx', markersize=10, 
            transform=ccrs.PlateCarree(), label='Selected pixels')
    
    # Configurar los límites del mapa
    # Ajustar estos valores según tu región de interés
    buffer = 5  # grados de buffer alrededor de los puntos
    lon_min, lon_max = min(selected_lons) - buffer, max(selected_lons) + buffer
    lat_min, lat_max = min(selected_lats) - buffer, max(selected_lats) + buffer
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Añadir grid
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Añadir título y leyenda
    plt.title('Location of Selected Pixels', pad=20)
    plt.legend()
    
    return fig, ax

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


def weibull_subplot_censored(data, fig_title, n_r, n_c, pos_f, censoring_percentages=[75, 80, 85], confidence_level=0.95, method = 'Trad', positive_x = False):
    """
    - method : 'Trad' for ln(-ln(1-p)), any other string for ln(ln(1 / (1 - p)))
    """
    # Ensure data is a 1D numpy array
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values.flatten()
    else:
        data = np.array(data).flatten()

    # Sort the data from lowest to highest
    sorted_data = np.sort(data)
    n           = len(sorted_data)

    # Calculate non-exceedance probabilities (p = ECDF in SMEV code)
    p = np.arange(1, n + 1) / (n + 1)

    # Calculate Weibull plot coordinates 
    if method == 'Trad':
        x       = np.log(-np.log(1 - p))
        x_label = 'ln(-ln(1-p)) [p: ECDF]'
    else:
        x = (np.log(np.log(1 / (1 - p))))  
        x_label = 'ln(ln(1 / (1 - p))) [p: ECDF]'
    # y = sorted_data
    y = np.log(sorted_data)
    # # x = np.log(np.log(p))  ## CON ESTO NO DA NADA, SE DAÑAA
    # # x_label = 'ln(ln(p)) [p: ECDF]'

    # # plt.figure(figsize=(12, 8))
    # fig, ax = plt.subplots(figsize=(10, 8))
    ax = Fig.add_subplot(n_r,n_c,pos_f)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(x, y, alpha=0.4, color='gray', label='All Data')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(censoring_percentages)))

    global_shapes = []
    global_scales = []
    ci_shapes = []
    ci_scales = []
    for censoring_percentage, color in zip(censoring_percentages, colors):
        # Determine the number of points to include (top percentage)
        n_include = int(n * (100 - censoring_percentage) / 100)

        # Select the top percentage of data (to use in SMEV code) Remember they are in log scale
        x_top = x[-n_include:]
        y_top = y[-n_include:]

        # Perform linear regression
        slope, intercept, r_value, _, std_err = stats.linregress(x_top, y_top)

        # Getting Weibull parameters as in SMEV code
        shape         = 1 / slope
        scale         = np.exp(intercept)        
        weibull_param = [shape, scale]

        global_shapes.append(shape)
        global_scales.append(scale)

        # Calculate confidence intervals for the slope (shape parameter)
        t_value = stats.t.ppf((1 + confidence_level) / 2, n_include - 2)
        conf_int_slope = t_value * std_err

        # Confidence interval for shape (1/slope)
        ci_shape_lower = 1 / (slope + conf_int_slope)
        ci_shape_upper = 1 / (slope - conf_int_slope)
        ci_shapes.append((ci_shape_lower, ci_shape_upper))

        # Confidence interval for scale (exp(intercept))
        conf_int_intercept = t_value * std_err * np.sqrt(1/n_include + np.mean(x_top)**2 / np.sum((x_top - np.mean(x_top))**2))
        ci_scale_lower = np.exp(intercept - conf_int_intercept)
        ci_scale_upper = np.exp(intercept + conf_int_intercept)
        ci_scales.append((ci_scale_lower, ci_scale_upper))

        # Generate points for the fitted line over the entire range
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept

        # Calculate confidence intervals
        conf_int = stats.t.ppf((1 + confidence_level) / 2, n_include - 2) * std_err * \
                   np.sqrt(1/n_include + (x_fit - np.mean(x_top))**2 / np.sum((x_top - np.mean(x_top))**2))
        # Ensure that the confidence bands are plotted based on the fitted line
        y_fit_lower = y_fit - conf_int
        y_fit_upper = y_fit + conf_int

        # Plot the top percentage and regression line
        if positive_x:
            alp = 0.3
        else:
            alp = 0.7
        plt.scatter(x_top, y_top, marker='x', label=f'Top {100-censoring_percentage:.0f}%', color=color, alpha=alp)
        plt.plot(x_fit, y_fit, ls ='--', label=f'Fitted Line (Top {100-censoring_percentage:.0f}%)', color=color, linewidth=2)
        # plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color=color, alpha=0.2, label=f'95% Confidence Interval (Top {100-censoring_percentage:.0f}%)')

        print(f"Top {100-censoring_percentage:.0f}%:")
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"Weibull shape: {weibull_param[0]:.4f}")
        print(f"Weibull scale: {weibull_param[1]:.4f}\n")
    if pos_f > n_c:
        plt.xlabel(x_label, fontsize=12)
    else:
        pass
    if pos_f == 1 or pos_f ==7 :
        plt.ylabel('ln(Wind Speed)', fontsize=12)
    else:
        pass    
    plt.title(fig_title, fontsize=14)
    if positive_x:
        plt.xlim(left=0)  # Set x-axis to start from 0 -- Logatirmic range of values of WS
        plt.ylim(2, 3.5) 
        # plt.ylim(bottom=0)  # Set y-axis to start from 0
    else:
        pass
    # plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tick_params(which='both', direction='in')
    # plt.savefig(bd_out_fig+name_fig, format='png', dpi=300, transparent=True)
    # plt.show()

    return global_shapes, global_scales, ci_shapes, ci_scales

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



#######################################################################################
##--------EXTRACTING SAMPLE OF PIXELS BY CONCATENATING THE REGION -ENSEMBLE ---------##
#######################################################################################

arr_cut_com = cut_concat_netcdf_files(bd_in_ese, min_lon, max_lon, min_lat, max_lat)
sample_arr  = arr_cut_com.wsa100m.values
lat_arr     = arr_cut_com.lat.values
lon_arr     = arr_cut_com.lon.values

num_pixels                           = 12 # Choosing 12 random pixels
subset_arr, lon_indices, lat_indices = select_unique_pixels(sample_arr, num_pixels)

##############################################################################
##---------MAPPING THE SELECTED POINTS OVER THE DATASET DOMIAN AREA---------##
##############################################################################

# Crear la figura y los ejes con la proyección deseada
fig = plt.figure(figsize=(12, 8))
ax  = plt.axes(projection=ccrs.PlateCarree())

# Añadir características del mapa
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Obtener las coordenadas reales de los píxeles seleccionados
selected_lats = lat_arr[lat_indices]
selected_lons = lon_arr[lon_indices]

# Plotear los puntos seleccionados
ax.plot(selected_lons, selected_lats, 'rx', markersize=10, 
        transform=ccrs.PlateCarree(), label='Selected pixels')

# Configurar los límites del mapa
# Ajustar estos valores según tu región de interés
buffer = 5  # grados de buffer alrededor de los puntos
lon_min, lon_max = min(selected_lons) - buffer, max(selected_lons) + buffer
lat_min, lat_max = min(selected_lats) - buffer, max(selected_lats) + buffer
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Añadir grid
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Añadir título y leyenda
plt.title('Location of Selected Pixels', pad=20)
plt.legend()
# plt.savefig(bd_out_fig+"Test_MapRandomPixels.png", format='png', dpi=300, transparent=True)
plt.show()

#############################################################################
##---------------------GENERATING THE NUMPY ARRAY OF DATETIMES ------------##
#############################################################################

start_date = '2000-01-01'    # Adjust this to the dataset start date
dates      = pd.date_range(start=start_date, periods=len(subset_arr), freq='H')
dates_np   = dates.to_numpy() # Convert dates to numpy datetime64 array

#############################################################################
##-----COMPUTTING THE OUTOCORRELOGRAM TO FIND THE MOST SUITABLE HOURS------##
#############################################################################
lags = np.arange(1, 201, 6) 

all_rs = []
for i in range(subset_arr.shape[1]):
    d1 = pd.Series(subset_arr[:, i])
    d2 = pd.Series(subset_arr[:, i])
    rs = [crosscorr(d1, d2, lag).round(3) for lag in lags]
    all_rs.append(rs)

all_rs = np.array(all_rs)
print(all_rs.shape)  # Debería ser (12, 34)

all_rs_filtered   = []
all_lags_filtered = []
all_rs_log        = []
all_lags_log      = []
for i in range(all_rs.shape[0]):
    print(i)
    rs = all_rs[i]
    
    # # Filtrar valores no positivos
    # mask_positive = rs > 0
    # lags_filtered = np.array(lags)[mask_positive]
    # rs_filtered   = rs[mask_positive]

    lags_filtered = np.array(lags)
    rs_filtered   = np.array(rs)

    all_lags_filtered.append(lags_filtered)
    all_rs_filtered.append(rs_filtered)
    
    # Transformación logarítmica
    rs_log = np.log(rs_filtered)
    rs_log[~np.isfinite(rs_log)] = np.nan
    all_rs_log.append(rs_log)
    all_lags_log.append(np.log(lags_filtered))


# # Filtrar valores no positivos
mask_positive   = rs > 0
lags_filtered_m = lags[mask_positive]
rs_filtered_m   = np.mean(all_rs,axis=0,keepdims=True)[0]

# Transformación logarítmica
rs_log_m = np.log(rs_filtered_m)
log_lags = np.log(lags_filtered_m)

#############################################################################
##-------PLOTTTING AND VISUALIZING THE AUTOCORRELATION IN THREE WAYS-------##
#############################################################################

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# 1. Gráfico original
style_axis(ax1)
for i in range(all_rs.shape[0]):
    ax1.scatter(lags, all_rs[i], c=colors[i], s=40, alpha =0.7, edgecolor='black', zorder=3)
    ax1.plot(lags, all_rs[i], c=colors[i], label=f"Serie {i+1}" , linewidth=2, zorder=2)
# ax1.scatter(lags, rs, c='#7f8eb7', s=40, zorder=3)
# ax1.plot(lags, rs, c='#7f8eb7', label='Autocorrelation', linewidth=2, zorder=2)
ax1.set_xlabel("Time [hours]", fontsize=14)
ax1.set_ylabel("Correlation coefficient", fontsize=14)
ax1.set_title("Lagged temporal autocorrelation", fontsize=13, fontweight="bold")
ax1.set_ylim(-0.01, 1.1)
ax1.legend(loc ="upper right" , fontsize=9)

# 2. Gráfico logarítmico con ajuste exponencial
style_axis(ax2)
for i in range(all_rs.shape[0]):
    ax2.scatter(lags_filtered, all_rs_log[i], c=colors[i], s=40, alpha =0.7, edgecolor='black', zorder=3)
    ax2.plot(lags_filtered, all_rs_log[i], c=colors[i], linewidth=2, zorder=2)
# ax2.scatter(lags_filtered, rs_log, c='#7f8eb7', s=40, zorder=3)
# ax2.plot(lags_filtered, rs_log, c='#7f8eb7', label='Ord.Events', linewidth=2, zorder=2)
ax2.set_xlabel("Time [hours]", fontsize=14)
ax2.set_ylabel("Log(Correlation coefficient)", fontsize=14)
ax2.set_title("Logarithmic Autocorrelation", fontsize=13, fontweight="bold")
ax2.axhline(y=np.log(corr_lim), color='grey', linestyle='--', label='1/e threshold', zorder=1)
# Ajuste exponencial para las primeras 60 horas
mask_60h = lags_filtered <= 60
if np.sum(mask_60h) > 2:  # Asegurarse de que hay suficientes puntos para el ajuste
    popt_exp, _ = curve_fit(exp_func, lags_filtered[mask_60h], rs_filtered_m[mask_60h], p0=[1, 0.1])
    ax2.plot(lags_filtered[mask_60h], np.log(exp_func(lags_filtered[mask_60h], *popt_exp)), c='r', ls = '--',
             label=f'Exp fit: a*exp(-bx), b={popt_exp[1]:.4f}', linewidth=2, zorder=4)
    ax2.legend(loc ="upper right" , fontsize=9)

# 3. Gráfico Log-Log con ajuste lineal
style_axis(ax3)
for i in range(all_rs.shape[0]):
    ax3.scatter(all_lags_log[0], all_rs_log[i], c=colors[i], s=40, alpha =0.7, edgecolor='black', zorder=3)
    ax3.plot(all_lags_log[0], all_rs_log[i], c=colors[i], linewidth=2, zorder=2)
# ax3.scatter(log_lags, rs_log, c='#7f8eb7', s=40, zorder=3)
# ax3.plot(log_lags, rs_log, c='#7f8eb7', linewidth=2, label='Ord.Events')
# Ajuste lineal
popt_linear, _ = curve_fit(linear_func, all_lags_log[0], rs_log_m)
ax3.plot(all_lags_log[0], linear_func(all_lags_log[0], *popt_linear), c='r', ls = '--', 
         label=f'Linear fit: y = {popt_linear[0]:.4f}x + {popt_linear[1]:.4f}', linewidth=2, zorder=4)

ax3.set_xlabel("Log(Time) [hours]", fontsize=14)
ax3.set_ylabel("Log(Correlation coefficient)", fontsize=14)
ax3.set_title("Log-Log Autocorrelation", fontsize=13, fontweight="bold")
ax3.legend(loc ="upper right" , fontsize=9)

plt.tight_layout()
plt.savefig(bd_out_fig+"Test_TypesAutocorrelation.png", format='png', dpi=300, transparent=True)
plt.show()

#############################################################################
##------COMPUTTING THE TIMES OF NO CORRELATION WITH THE TWO APPROACHES-----##
#############################################################################

h_corr_list = []
tau_d_list  = []
for i in range(all_rs.shape[0]):
    rs = all_rs[i]

    # Decaimiento exponenecial
    indice = np.where(rs < corr_lim)[0][0] # Correlation threshold chosen indicating the temporal structure is week/Minimum influenceof the past values
    h_corr = lags[indice]
    h_corr_list.append(h_corr)

    # Ajuste spline cúbico (Integral del tiempo)
    spline    = UnivariateSpline(lags, rs, s=0.01)  # s es el parámetro de suavizado
    rs_spline = spline(lags)
    tau_d     = np.trapz(rs_spline, lags)
    tau_d_list.append(round(tau_d, 1))

    # ax1.scatter(lags, all_rs[i], c=colors[i], s=40, alpha =0.7, edgecolor='black', zorder=3)
    # ax1.plot(lags, all_rs[i], c=colors[i], label=f"Serie {i+1}" , linewidth=2, zorder=2)
h_corr_list = np.array(h_corr_list)
tau_d_list  = np.array(tau_d_list)

h_corr_m = h_corr_list.mean()
tau_d_m  = tau_d_list.mean()

series_labels = [str (n) for n in range(1, num_pixels+1)]

fig, (ax1) = plt.subplots(1, 1,figsize=(8, 5))
style_axis(ax1)
plt.scatter( np.arange(1, len(series_labels)+1), h_corr_list,  c='#bc4f5e')
plt.plot( np.arange(1, len(series_labels)+1), h_corr_list, '#bc4f5e',  label=f'[Exp]Mean Dec.Time ≈ {h_corr_m:.1f} hours')
plt.scatter( np.arange(1, len(series_labels)+1), tau_d_list,  c='#7f8eb7')
plt.plot( np.arange(1, len(series_labels)+1), tau_d_list, '#7f8eb7',  label=f'[Pot]Mean Dec.Time ≈ {tau_d_m:.1f} hours')
plt.axhline(y=h_corr_m, color='#bc4f5e', linestyle='--', linewidth=0.8)
plt.axhline(y=tau_d_m, color='#7f8eb7', linestyle='--', linewidth=0.8)
ax1.set_xticks(np.arange(1, len(series_labels)+1))
ax1.set_xticklabels(series_labels, fontsize=11)
plt.xlabel("Random Series ", fontsize=14)
plt.ylabel("Time [hours]", fontsize=14)
plt.legend(loc ="upper right", fontsize=11)
plt.tight_layout()
# plt.savefig(bd_out_fig+"Test_ExpCubicSplineFit_DecorrTime.png", format='png', dpi=300, transparent=True)
plt.show()

########################################################################################
##-ITERATING OVER THE TIME SERIE TO SEPARATE THE EVENTS BASED ON THE AOUTOCORRELOGRAM-##
########################################################################################

n_eve_e = []
n_eve_p = []
for i in range(len(series_labels)):
    s_arr = subset_arr[:, i]
    e_cor = h_corr_list [i]
    p_cor = int(round(tau_d_list[i], 0))

    sep_events_e, idt_events_e = separation_ws_events_max(s_arr, e_cor, dates_np)
    sep_events_p, idt_events_p = separation_ws_events_max(s_arr, p_cor, dates_np)

    n_eve_e.append(len(sep_events_e))
    n_eve_p.append(len(sep_events_p))

n_eve_e = np.array(n_eve_e)
n_eve_p = np.array(n_eve_p)

n_eve_e_m = n_eve_e.mean()
n_eve_p_m  = n_eve_p.mean()

fig, (ax1) = plt.subplots(1, 1,figsize=(8, 5))
style_axis(ax1)
plt.scatter( np.arange(1, len(series_labels)+1), n_eve_e,  c='#bc4f5e')
plt.plot( np.arange(1, len(series_labels)+1), n_eve_e, '#bc4f5e',  label=f'[Exp]N.Events ≈ {n_eve_e_m:.1f}')
plt.scatter( np.arange(1, len(series_labels)+1), n_eve_p,  c='#7f8eb7')
plt.plot( np.arange(1, len(series_labels)+1), n_eve_p, '#7f8eb7',  label=f'[Pot]N.Events ≈ {n_eve_p_m:.1f}')
ax1.set_xticks(np.arange(1, len(series_labels)+1))
ax1.set_xticklabels(series_labels, fontsize=11)
plt.xlabel("Random Series ", fontsize=14)
plt.ylabel("Num.Events", fontsize=14)
plt.legend(loc ="upper left", fontsize=11)
plt.tight_layout()
# plt.savefig(bd_out_fig+"Test_NEvents_SeparationStorms.png", format='png', dpi=300, transparent=True)
plt.show()

#############################################################################
##-------SETTING UP SMEV PARAMETERS FOR SMEV RETURN LEVELS ESTIMATION------##
#############################################################################

# Setting up the SMEV  single parameters
threshold_meas  = 0  # For the valid measurements 0.1 mm in precipitation records in winds could be 0 m/s
durations       = 60 # In Minutes
time_resolution = 60 # In Minutes

# List of return periods as other SMEV parameters
return_periods = [2, 5, 10, 20, 50]

##########################################################################################
##---RETURN LEVELS SMEV-WEIBULL PLOTS WITH DIFFERENT THRESHOLDS FOR LEFT SENSORING------##
##########################################################################################

# port_censored = [0.75, 0.80, 0.85, 0.90, 0.95]

g_shape_e = []
g_scale_e = []
Fig  = plt.figure(figsize=(12, 6))
for i in range(len(series_labels)):
    s_arr  = subset_arr[:, i]
    e_cor  = h_corr_list [i]
    p_cens = 0.75 # Se escoge arbitario considerando los objetivos acá. Para obtener los eventos ordinarios desde SMEV no es relevante el data portion, en esta funcion esto solo es relevante para estimar los RL. 

    smev_r_e, df_ord_events_e    = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, e_cor, durations, time_resolution, [p_cens, 1])
    oe_smev_e                    = df_ord_events_e['value'].values
    list_shape, list_scale, _, _ = weibull_subplot_censored(oe_smev_e, f"Serie {series_labels[i]}", 2, 6, i+1, censoring_percentages=[75, 80, 85, 90, 95], confidence_level=0.95, method = 'Trad', positive_x = True)
    
    g_shape_e.append(list_shape)
    g_scale_e.append(list_scale)

plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.title('Weibull Plot of Ordinary events [Pot]', fontsize=14)
# plt.savefig(bd_out_fig+"Test_WeibullPlot_SMEV_e_positiveX.png", format='png', dpi=300, transparent=True)
plt.show()

g_shape_p = []
g_scale_p = []
Fig  = plt.figure(figsize=(12, 6))
for i in range(len(series_labels)):
    s_arr  = subset_arr[:, i]
    p_cor  = int(round(tau_d_list [i], 0))
    p_cens = 0.75 # Se escoge arbitario considerando los objetivos acá. Para obtener los eventos ordinarios desde SMEV no es relevante el data portion, en esta funcion esto solo es relevante para estimar los RL. 

    smev_r_p, df_ord_events_p    = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, p_cor, durations, time_resolution, [p_cens, 1])
    oe_smev_p                    = df_ord_events_p['value'].values
    list_shape, list_scale, _, _ = weibull_subplot_censored(oe_smev_p, f"Serie {series_labels[i]}", 2, 6, i+1, censoring_percentages=[75, 80, 85, 90, 95], confidence_level=0.95, method = 'Trad', positive_x = True)
    
    g_shape_p.append(list_shape)
    g_scale_p.append(list_scale)

plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.title('Weibull Plot of Ordinary events [Pot]', fontsize=14)
# plt.savefig(bd_out_fig+"Test_WeibullPlot_SMEV_p_positiveX.png", format='png', dpi=300, transparent=True)
plt.show()

###############################################################################
##------PLOTTING SHAPE AND SCALE PARAMETER DOR EACH CENSORING THERSHOLD------##
###############################################################################

g_shape_p = np.array(g_shape_p)
g_scale_p = np.array(g_scale_p)

x_tick_labels = ['Top 25%', 'Top 20%', 'Top 15%', 'Top 10%', 'Top 05%'] 
x_tick        = np.arange(1, len(x_tick_labels)+1)

Fig = plt.figure(figsize=(12, 4))

ax = Fig.add_subplot(121)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
for i in range(g_shape_p.shape[0]):
    ax.scatter(x_tick, g_shape_p[i], c=colors[i], marker='o', s=40, alpha=0.7, edgecolor='black', zorder=3)
    ax.plot(x_tick, g_shape_p[i], c=colors[i], linewidth=2, zorder=2, label=f"Serie {i+1}")
ax.set_xticks(x_tick, x_tick_labels, fontsize=15, rotation=0)
ax.set_title(u'a) Shape parameter per portion', fontsize=15, weight='bold', loc='left')
ax.set_ylabel(u'[-]', fontsize=15)
ax.set_xlabel(u'Top portion Ord.Events', fontsize=15)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(which='both', direction='in', labelsize=12)

ax1 = Fig.add_subplot(122)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
for i in range(g_scale_p.shape[0]):
    ax1.scatter(x_tick, g_scale_p[i], c=colors[i], marker='o', s=40, alpha=0.7, edgecolor='black', zorder=3)
    ax1.plot(x_tick, g_scale_p[i], c=colors[i], linewidth=2, zorder=2)
ax1.set_xticks(x_tick, x_tick_labels, fontsize=15, rotation=0)
ax1.set_title(u'b) Scale parameter per portion', fontsize=15, weight='bold', loc='left')
ax1.set_ylabel(u'[m/s]', fontsize=15)
ax1.set_xlabel(u'Top portion Ord.Events', fontsize=15)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.tick_params(which='both', direction='in', labelsize=12)

legend = ax.legend(bbox_to_anchor=(0.1, -0.2), loc='upper left', ncol=6, fontsize=11)
plt.subplots_adjust(wspace=0.25, hspace=0.45, left=0.10, right=0.97, bottom=0.30)
# plt.savefig(bd_out_fig+'Test_ShapeScale_CensoredPortions.png', format='png', dpi=300, transparent=True)
plt.show()

###############################################################################
##-------PLOTTING WEIBULL DISTRIBUTION AND FIT FOR THE ORDINARY EVENTS-------##
###############################################################################

n_c       = 6
n_r       = 2
Fig       = plt.figure(figsize=(12, 6))
for i in range(len(series_labels)):
    s_arr  = subset_arr[:, i]
    p_cor  = int(round(tau_d_list [i], 0))
    p_cens = 0.75 # Se escoge arbitario considerando los objetivos acá. Para obtener los eventos ordinarios desde SMEV no es relevante el data portion, en esta funcion esto solo es relevante para estimar los RL. 
    pos_f  = i+1

    smev_r_p, df_ord_events_p    = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, p_cor, durations, time_resolution, [p_cens, 1])
    oe_smev_p                    = df_ord_events_p['value'].values

    ax = Fig.add_subplot(n_r, n_c, pos_f)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ajuste de distribución de Weibull
    shape, loc, scale = stats.weibull_min.fit(oe_smev_p, floc=0)
    
    # Histograma de los datos
    n, bins, patches = ax.hist(oe_smev_p, bins='auto', density=True, alpha=0.7, color='skyblue', label='Ord.Events')
  
    # Generar puntos para la PDF de Weibull
    x   = np.linspace(oe_smev_p.min(), oe_smev_p.max(), 100)
    pdf = stats.weibull_min.pdf(x, shape, loc, scale)
    ax.plot(x, pdf, 'r-', lw=2, label=f'Weibull fit\n(k={shape:.2f}, l={scale:.2f})')
    ax.set_title(f"Serie {series_labels[i]}", fontsize=14, y=1.09)
    if pos_f > n_c:
        plt.xlabel("Wind Speed [m/s]", fontsize=10)
    else:
        pass
    if pos_f == 1 or pos_f ==7 :
        plt.ylabel("Density", fontsize=12)
    else:
        pass 
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.savefig(bd_out_fig+"Test_RandomPixesl_OrdWeibullFitDistribution.png", format='png', dpi=300, transparent=True)
plt.show()

###############################################################################
##--------------COMPUTTING RL FROM SMEV AND TRADITIONAL METHODS--------------##
###############################################################################

smev_rl_total = []
gev_rl_total  = []
gpd_rl_total  = []
spcr_rl_total = []
anumax_total  = []
ppamax_total  = []
amaxcr_total  = []
pamaxcr_total = []
for i in range(len(series_labels)):
    s_arr  = subset_arr[:, i]

    #######--PROBABILITY PLOTTING POSITIONS--######
    annual_max = np.array([np.max(s_arr[i*8760:(i+1)*8760]) for i in range(10)])
    anumax_total.append(annual_max)

    n          = len(annual_max)
    rank_am    = rankdata(annual_max)
    pp         = rank_am / (n + 1)
    rt_pp      = 1 / ( 1- pp)
    ppamax_total.append(rt_pp)
    ###############################################

    ##--SMEV WITH POTENTIAL DECAY APPROACH AND TOP 10 %
    p_cor                    = int(round(tau_d_list [i], 0))
    p_cens                   = 0.90
    smev_rl, df_ord_events_p = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, p_cor, durations, time_resolution, [p_cens, 1])
    smev_rl_total.append (smev_rl)
    print("### --- SMEV 10-pot results:")
    for period, value in zip(return_periods, smev_rl):
        print(f"Return period {period} years: {value:.2f}")

    ##--GEVD-PMM 
    gev_rl, gev_param = gevd_fit(s_arr, return_periods)
    gev_rl_total.append (gev_rl)
    print("### --- GEVD-PMM results:")
    for period, value in zip(return_periods, gev_rl):
        print(f"Return period {period} years: {value:.2f} m/s shape, loc, scale: {gev_param}")

    ##--GPD-POT separeted events:
    # sep_events, idt_events = separation_ws_events_max(s_arr, h_corr, dates_np)
    top_ord_events       = 90 
    threshold_gpd        = np.percentile(s_arr, top_ord_events)
    gpd_rl, sigma_rl_gpd = gpd_fit(s_arr, threshold_gpd, return_periods, p_cor, dates_np)
    gpd_rl_total.append (gpd_rl)
    print("### --- GPD-POT results:")
    for period, value, sigma in zip(return_periods, gpd_rl, sigma_rl_gpd):
        print(f"Return period {period} years: {value:.2f} ± {1.96*sigma:.2f} m/s")

    ##--SC-Gumbel fit:
    spcr_rl, _, annual_max_cr, _ = spectral_correction(s_arr,  return_periods)
    spcr_rl_total.append(spcr_rl)
    amaxcr_total.append(annual_max_cr)
    print("### --- SC-Gumbel results:")
    for period, value in zip(return_periods, spcr_rl):
        print(f"Return period  {period} years: {value:.2f} m/s")
    #######--PROBABILITY PLOTTING POSITIONS--######
    n_cr       = len(annual_max_cr)
    rank_am_cr = rankdata(annual_max_cr)
    pp_cr      = rank_am_cr / (n_cr + 1)
    rt_pp_cr   = 1 / ( 1- pp_cr)
    pamaxcr_total.append(rt_pp_cr)
    ###############################################

smev_rl_total = np.array(smev_rl_total)
gev_rl_total  = np.array(gev_rl_total )
gpd_rl_total  = np.array(gpd_rl_total)
spcr_rl_total = np.array(spcr_rl_total)
anumax_total  = np.array(anumax_total)
ppamax_total  = np.array(ppamax_total)
amaxcr_total  = np.array(amaxcr_total)
pamaxcr_total = np.array(pamaxcr_total)

###############################################################################
##---------------PLOTTING RL FROM SMEV AND TRADITIONAL METHODS---------------##
###############################################################################

colors_rl = ['#f39800', '#05badd', '#2b4871', '#00f398']
n_c       = 6
n_r       = 2
Fig       = plt.figure(figsize=(12, 6))
for i in range(len(series_labels)):

    pos_f = i+1

    ax = Fig.add_subplot(n_r, n_c, pos_f)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.scatter(return_periods, smev_rl_total[i,:], c=colors_rl[0], s=50, zorder = 3)
    ax.plot(return_periods, smev_rl_total[i,:], c=colors_rl[0], label = 'SMEV Top10%', linewidth=2, zorder = 3)
    ax.scatter(return_periods, gev_rl_total[i,:], c=colors_rl[1], s=50, zorder = 3)
    ax.plot(return_periods, gev_rl_total[i,:], c=colors_rl[1], label = 'GEV', linewidth=2, zorder = 3)
    ax.scatter(return_periods, gpd_rl_total[i,:], c=colors_rl[2], s=50, zorder = 3)
    ax.plot(return_periods, gpd_rl_total[i,:], c=colors_rl[2], label = 'GPD', linewidth=2, zorder = 3)
    ax.scatter(return_periods, spcr_rl_total[i,:], c=colors_rl[3], s=50, zorder = 3)
    ax.plot(return_periods, spcr_rl_total[i,:], c=colors_rl[3], label = 'SC', linewidth=2, zorder = 3)
    ax.scatter(ppamax_total[i,:], anumax_total[i,:], c='red', s=60, alpha = 0.5, label = 'PPP', zorder = 1)
    ax.scatter(pamaxcr_total[i,:], amaxcr_total[i,:], c='gray', marker = 'X', alpha = 0.7, label = 'PPP-SC', zorder = 2)
    ax.set_title(f"Serie {series_labels[i]}", fontsize=14)
    ax.set_ylim(10, 25)
    if pos_f > n_c:
        plt.xlabel("Return Period [years]", fontsize=10)
    else:
        pass
    if pos_f == 1 or pos_f ==7 :
        plt.ylabel("Wind Speed [m/s]", fontsize=12)
    else:
        pass 
    ax.set_xticks(return_periods)
    ax.set_xticklabels(return_periods, fontsize=10)
    ax.set_yticks(np.arange(7, 29, 7))
    ax.set_yticklabels(np.arange(7, 29, 7), fontsize=11)
    ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

legend = ax.legend(bbox_to_anchor=(-0.5, -0.3), loc='upper right', ncol=6, fontsize=12)
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
# plt.savefig(bd_out_fig+"Test_RL_Estimtion_MultiMethods.png", format='png', dpi=300, transparent=True)
plt.show()



###############################################################################
##---------------RANDOM SERIES POWER SPECTRA AND CORRECTION------------------##
###############################################################################

n_c       = 6
n_r       = 2
Fig       = plt.figure(figsize=(12, 6))
for i in range(len(series_labels)):
    s_arr      = subset_arr[:, i]

    mean_speed = np.mean(s_arr)

    # Cálculo del espectro
    freqs, psd, smoothed_psd = calculate_spectrum_smoothed(s_arr, window_size=60)
    # Determinación de fc y corrección
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

    pos_f = i+1

    ax = Fig.add_subplot(n_r, n_c, pos_f)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    line1, = ax.plot(np.log(freqs), np.log(psd), alpha=0.5, linestyle="--")
    line2, = ax.plot(np.log(freqs), np.log(smoothed_psd), linewidth=2)
    vline  = ax.axvline(np.log(fc), color='r', linestyle='--', label=f"$f_c$ = {fc:.2f} days$^{{-1}}$")
    line3, = ax.plot(np.log(freqs), np.log(corrected_psd), linewidth=2, color='green', alpha =0.5)
    ax.set_title(f"Serie {series_labels[i]}", fontsize=14)
    if pos_f > n_c:
        plt.xlabel("ln(Frequency) (ln(days$^{-1}$))", fontsize=10)
    else:
        pass
    if pos_f == 1 or pos_f ==7 :
        plt.ylabel("ln(S(f))", fontsize=12)
    else:
        pass 
    ax.legend(handles=[vline], fontsize=8)
    ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

legend = ax.legend([line1, line2, line3], ["Original PSD (noisy)", "PSD smoothed", "PSD corrected"],bbox_to_anchor=(-0.5, -0.3), loc='upper right', ncol=6, fontsize=12)
## Para que tambien salga la leyenda en el ultimo subplot, ponemos:
ax.add_artist(legend)
ax.legend(handles=[vline], fontsize=8) 
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
# plt.savefig(bd_out_fig+"Test_Spectrum_Power_EnsemSeries.png", format='png', dpi=300, transparent=True)
plt.show()




















