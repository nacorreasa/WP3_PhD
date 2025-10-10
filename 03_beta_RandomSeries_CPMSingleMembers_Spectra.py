import xarray as xr
import numpy as np
import pandas as pd
import os,glob,sys
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
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
Codigo para la generacion de los espectros de potencia originales y correfidos de 
12 series de datos aleatorias provenientes de los miembros CPM previamente 
guardadas.

Author : Nathalia Correa-Sánchez
"""

#############################################################################
##-------------------------DEFINING IMPORTANT PATHS------------------------##
#############################################################################

bd_in_npy  = "/Dati/Outputs/Random_SeriePixels_WS/"
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"
bd_in_eth  = bd_in_npy + "RandomSerie_ETH.npy"
bd_in_cmcc = bd_in_npy + "RandomSerie_CMCC.npy"
bd_in_cnrm = bd_in_npy + "RandomSerie_CNRM.npy"

#############################################################################
##------------------------DEFINING RELEVANT FUNCTIONS----------------------##
#############################################################################

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
    mask             = (freqs > 0.6) & (freqs < 0.9)
    log_freqs        = np.log(freqs[mask])
    log_psd          = np.log(psd[mask])
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

def obtain_spectra_params_corr (s_arr, window_size = 60):
    """
    Funcion que resume los calculos del espectro y su correccion para
    series de cada modelo.    
    """
    # Cálculo del espectro
    freqs, psd, smoothed_psd = calculate_spectrum_smoothed(s_arr, window_size)

    # Determinación de fc y corrección
    fc, s_fc      = find_cutoff_frequency(freqs, smoothed_psd)
    corrected_psd = correct_spectrum(freqs, smoothed_psd, fc, s_fc)

    return freqs, psd, smoothed_psd, fc, corrected_psd

#######################################################################################
##------------LOADING THE ARRAY WITH THE RANDOM SERIES FOR EACH CPM MEMBER-----------##
#######################################################################################

subset_arr_eth  = np.load (bd_in_eth)
subset_arr_cmcc = np.load (bd_in_cmcc)
subset_arr_cnrm = np.load (bd_in_cnrm)

#######################################################################################
##-------------------DERIVING & PLOTTING SPECTRA ORIGINAL AND CORRECTED--------------##
#######################################################################################

colours_l     = ["#edae49", "#d1495b", "#00798c"]
series_labels = [str (n) for n in range(1, subset_arr_eth.shape[1]+1)]
n_c           = 6
n_r           = 2
Fig           = plt.figure(figsize=(12, 6))
for i in range(len(series_labels)):
    s_arr_eth  = subset_arr_eth[:, i]
    s_arr_cmcc = subset_arr_cmcc[:, i]
    s_arr_cnrm = subset_arr_cnrm[:, i]

    freqs_eth, psd_eth, smoothed_psd_eth, fc_eth, corrected_psd_eth      = obtain_spectra_params_corr (s_arr_eth, window_size = 60)
    freqs_cmcc, psd_cmcc, smoothed_psd_cmcc, fc_cmcc, corrected_psd_cmcc = obtain_spectra_params_corr (s_arr_cmcc, window_size = 60)
    freqs_cnrm, psd_cnrm, smoothed_psd_cnrm, fc_cnrm, corrected_psd_cnrm = obtain_spectra_params_corr (s_arr_cnrm, window_size = 60)
    
    pos_f = i+1
    ax    = Fig.add_subplot(n_r, n_c, pos_f)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ## ETH 
    line2_eth, = ax.plot(np.log(freqs_eth), np.log(smoothed_psd_eth),color=colours_l[0], linewidth=2, alpha=0.3, linestyle="--")
    vline_eth  = ax.axvline(np.log(fc_eth), color=colours_l[0], linestyle='dotted', label=f"$f_c$ = {fc_eth:.2f} d$^{{-1}}$")
    line3_eth, = ax.plot(np.log(freqs_eth), np.log(corrected_psd_eth), linewidth=1, color=colours_l[0])

    ## CMCC 
    line2_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(smoothed_psd_cmcc),color=colours_l[1], linewidth=2, alpha=0.3, linestyle="--")
    vline_cmcc  = ax.axvline(np.log(fc_cmcc), color=colours_l[1], linestyle='dotted', label=f"$f_c$ = {fc_cmcc:.2f} d$^{{-1}}$")
    line3_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(corrected_psd_cmcc), linewidth=1, color=colours_l[1])

    ## CNRM 
    line2_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(smoothed_psd_cnrm),color=colours_l[2], linewidth=2, alpha=0.3, linestyle="--")
    vline_cnrm  = ax.axvline(np.log(fc_cnrm), color=colours_l[2], linestyle='dotted', label=f"$f_c$ = {fc_cnrm:.2f} d$^{{-1}}$")
    line3_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(corrected_psd_cnrm), linewidth=1, color=colours_l[2])


    ax.set_title(f"Serie {series_labels[i]}", fontsize=14)
    if pos_f > n_c:
        plt.xlabel("(ln(days$^{-1}$))", fontsize=10)
    else:
        pass
    if pos_f == 1 or pos_f ==7 :
        plt.ylabel("ln(S(f))", fontsize=12)
    else:
        pass 

    custom_line_eth  = Line2D([0], [0], color=colours_l[0], linestyle='dotted', linewidth=0.9)
    custom_line_cmcc = Line2D([0], [0], color=colours_l[1], linestyle='dotted', linewidth=0.9)
    custom_line_cnrm = Line2D([0], [0], color=colours_l[2], linestyle='dotted', linewidth=0.9)
    ax.legend(handles=[custom_line_eth, custom_line_cmcc, custom_line_cnrm], 
              labels=[f"$f_c$:{fc_eth:.2f} d$^{{-1}}$", f"$f_c$:{fc_cmcc:.2f} d$^{{-1}}$", f"$f_c$:{fc_cnrm:.2f} d$^{{-1}}$"], 
              fontsize=7, loc='lower left')
    ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

legend = ax.legend([line2_eth, line3_eth, line2_cmcc, line3_cmcc, line2_cnrm, line3_cnrm ], 
                   ["PSD smoothed ETH", "PSD corrected ETH","PSD smoothed CMCC", "PSD corrected CMCC", "PSD smoothed CNRM", "PSD corrected CNRM"],
                   bbox_to_anchor=(0., -0.3), loc='upper right', ncol=3, fontsize=12)
## Para que tambien salga la leyenda en el ultimo subplot, ponemos:
ax.add_artist(legend)
ax.legend(handles=[vline_eth, vline_cmcc, vline_cnrm], fontsize=8) 
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.savefig(bd_out_fig+"Test_Spectrum_Power_CPMMembersSeries.png", format='png', dpi=300, transparent=True)
plt.show()



