import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as tck
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import pyarrow as pa
from smev_class import SMEV
import glob
import numpy as np

#############################################################################
##-------------------------DEFINING IMPORTANT PATHS------------------------##
#############################################################################
bd_in_ws   = "/Dati/Data/COSMO-REA/Germany_WP2/OutDownload/WS_100/"
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"

#############################################################################
##-----------DEFINNING BOUNDARY COORDINATES TO PERFOMR THE TEST------------##
#############################################################################
min_lon =  0.689
min_lat =  46.521
max_lon =  1.478
max_lat =  47.067

#############################################################################
##-------------------------DEFINNING RELEVANT INPUTS-----------------------##
#############################################################################
years    = [2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009] 
corr_lim = 1/np.e
# corr_lim = 1.15

#############################################################################
##------------------------DEFINING RELEVANT FUNCTIONS----------------------##
#############################################################################

def cut_concat_netcdf_files(bd_in_ws, years, min_lon, max_lon, min_lat, max_lat):
    arr_cut = []
    
    for year in years:
        bd_ws = f"{bd_in_ws}{year}/"
        filez_ws = sorted(glob.glob(f"{bd_ws}*.nc"))
        
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


#############################################################################
##-----------EXTRACTING SINGLE PIXEL BY  CONCATENATING THE REGION----------##
#############################################################################

arr_cut_com = cut_concat_netcdf_files(bd_in_ws, years, min_lon, max_lon, min_lat, max_lat)

sample_arr = arr_cut_com.wind_speed.values
s_arr      = sample_arr[:, 0, 0]

start_date = '2009-01-01'    # Adjust this to the dataset start date
dates      = pd.date_range(start=start_date, periods=len(s_arr), freq='H')
dates_np   = dates.to_numpy() # Convert dates to numpy datetime64 array

#############################################################################
##-----COMPUTTING THE OUTOCORRELOGRAM TO FIND THE MOST SUITABLE HOURS------##
#############################################################################
lags = [0,6, 12, 24, 36,48, 60, 72, 84,96, 108, 120, 132, 144, 156, 168] # In hours
lags = np.array(lags)

df_ws = pd.DataFrame({'S1': s_arr, 'S2': s_arr})
d1    = df_ws['S1']
d2    = df_ws['S2']
rs    = [crosscorr(d1,d2, lag).round(3) for lag in np.array(lags)]
rs    = np.array(rs)

lags = np.arange(1, 201, 6) 

rs = [crosscorr(d1, d2, lag).round(3) for lag in lags]
rs = np.array(rs)

# Filtrar valores no positivos
mask_positive = rs > 0
lags_filtered = lags[mask_positive]
rs_filtered   = rs[mask_positive]

# Transformación logarítmica
rs_log   = np.log(rs_filtered)
log_lags = np.log(lags_filtered)

#############################################################################
##-------PLOTTTING AND VISUALIZING THE AUTOCORRELATION IN THREE WAYS-------##
#############################################################################

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# 1. Gráfico original
style_axis(ax1)
ax1.scatter(lags, rs, c='#7f8eb7', s=50, zorder=3)
ax1.plot(lags, rs, c='#7f8eb7', label='Autocorrelation', linewidth=2, zorder=2)
ax1.set_xlabel("Time [hours]", fontsize=14)
ax1.set_ylabel("Correlation coefficient", fontsize=14)
ax1.set_title("Lagged temporal autocorrelation", fontsize=13, fontweight="bold")
ax1.set_ylim(-0.01, 1.1)

# 2. Gráfico logarítmico con ajuste exponencial
style_axis(ax2)
ax2.scatter(lags_filtered, rs_log, c='#7f8eb7', s=50, zorder=3)
ax2.plot(lags_filtered, rs_log, c='#7f8eb7', label='Ord.Events', linewidth=2, zorder=2)
ax2.set_xlabel("Time [hours]", fontsize=14)
ax2.set_ylabel("Log(Correlation coefficient)", fontsize=14)
ax2.set_title("Logarithmic Autocorrelation", fontsize=13, fontweight="bold")
ax2.axhline(y=np.log(corr_lim), color='grey', linestyle='--', label='1/e threshold', zorder=1)
# Ajuste exponencial para las primeras 60 horas
mask_60h = lags_filtered <= 60
if np.sum(mask_60h) > 2:  # Asegurarse de que hay suficientes puntos para el ajuste
    popt_exp, _ = curve_fit(exp_func, lags_filtered[mask_60h], rs_filtered[mask_60h], p0=[1, 0.1])
    ax2.plot(lags_filtered[mask_60h], np.log(exp_func(lags_filtered[mask_60h], *popt_exp)), c='r', ls = '--',
             label=f'Exp fit: a*exp(-bx), b={popt_exp[1]:.4f}', linewidth=2)
    ax2.legend(loc ="upper right" , fontsize=9)

# 3. Gráfico Log-Log con ajuste lineal
style_axis(ax3)
ax3.scatter(log_lags, rs_log, c='#7f8eb7', s=50, zorder=3)
ax3.plot(log_lags, rs_log, c='#7f8eb7', linewidth=2, label='Ord.Events')
# Ajuste lineal
popt_linear, _ = curve_fit(linear_func, log_lags, rs_log)
ax3.plot(log_lags, linear_func(log_lags, *popt_linear), c='r', ls = '--', 
         label=f'Linear fit: y = {popt_linear[0]:.4f}x + {popt_linear[1]:.4f}', linewidth=2)

ax3.set_xlabel("Log(Time) [hours]", fontsize=14)
ax3.set_ylabel("Log(Correlation coefficient)", fontsize=14)
ax3.set_title("Log-Log Autocorrelation", fontsize=13, fontweight="bold")
ax3.legend(loc ="upper right" , fontsize=9)

plt.tight_layout()
# plt.savefig(bd_out_fig+"TestSinglePixel_TypesAutocorrelation.png", format='png', dpi=300, transparent=True)
plt.show()

#############################################################################
##------COMPUTTING THE TIMES OF NO CORRELATION WITH THE TWO APPROACHES-----##
#############################################################################

# Decaimiento exponenecial
indice = np.where(rs < corr_lim)[0][0] # Correlation threshold chosen indicating the temporal structure is week/Minimum influenceof the past values
h_corr = lags[indice]
print(f"Tiempo de 1/e: {h_corr} horas")

# Ajuste spline cúbico
spline    = UnivariateSpline(lags, rs, s=0.01)  # s es el parámetro de suavizado
rs_spline = spline(lags)
tau_d     = np.trapz(rs_spline, lags)
print(f"Tiempo de Decorrelación: {tau_d} horas")

fig, (ax1) = plt.subplots(1, 1,figsize=(8, 5))
style_axis(ax1)
plt.scatter(lags, rs,  c='#7f8eb7', zorder =3, label='Original Autocorrelation')
plt.plot(lags, rs_spline, 'r-', zorder =2, label='Cubic spline fit')
plt.fill_between(lags, rs_spline, color='lightgrey', alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=tau_d, color='g', linestyle='--', label=f'Decorrelation time ≈ {tau_d:.2f} hours')
plt.ylim(-0.01, 1.1)
plt.xlabel("Time [hours]", fontsize=14)
plt.ylabel("Correlation coefficient", fontsize=14)
plt.legend(loc ="upper right", fontsize=11)
plt.tight_layout()
plt.show()


########################################################################################
##-ITERATING OVER THE TIME SERIE TO SEPARATE THE EVENTS BASED ON THE AOUTOCORRELOGRAM-##
########################################################################################

sep_events_thr, idt_events_thr = separation_ws_events_max(s_arr, h_corr, dates_np)
sep_events_int, idt_events_int = separation_ws_events_max(s_arr, int(round(tau_d, 0)), dates_np)

########################################################################################
##----------------------PLOTTING THE DATA AND THE SELECTED ORDINARY EVENTS------------##
########################################################################################

y_labels = np.array([0, 5, 10, 15, 20, 25, 30])

fig = plt.figure(figsize=(12, 6))
ax  = fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(dates_np, s_arr, label='Wind speed', color='#898989', alpha=0.5)
plt.scatter(idt_events_thr, np.array(sep_events_thr), color='#bc4f5e', label='[Exp] Time = '+str(h_corr)+' hours-'+str(len(sep_events_thr))+' events', zorder=4)
plt.scatter(idt_events_int, np.array(sep_events_int), color='#7f8eb7',alpha =0.6, label='[Pot] Time = '+str(int(round(tau_d, 0)))+' hours-'+str(len(sep_events_int))+' events', zorder=5)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Wind speed [m/s]', fontsize=15)
plt.title('Selected maximum events', fontsize=14, fontweight="bold")
plt.ylim(0,31)
ax.set_yticks(y_labels)
ax.set_yticklabels([f'{y:.0f}' for y in y_labels], fontsize=15, rotation=0)
plt.tick_params(which='both', direction='in')
ax.legend(fontsize=15, loc='best')
plt.grid(True, alpha=0.3, linestyle = '--')
plt.tight_layout()
plt.savefig(bd_out_fig+"TestSinglePixel_SeparationStorms.png", format='png', dpi=300, transparent=True)
plt.show()

#############################################################################
##---------------------------HOTOGRAM PLOT TEST----------------------------##
#############################################################################

# Define bins based on the entire data (ord_events)
bins = np.histogram_bin_edges(sep_events_thr, bins=15)

fig, ax = plt.subplots(figsize=(9, 7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.hist(sep_events_thr, bins=bins, alpha=0.6, color='#bc4f5e', edgecolor='black', label='Exponential')
ax.hist(sep_events_int, bins=bins, alpha=0.6, color='#7f8eb7', edgecolor='black', label='Potential')
bin_centers = 0.5 * (bins[:-1] + bins[1:])  
ax.set_xticks(bin_centers)
ax.set_xticklabels([f'{int(center)}' for center in bin_centers], fontsize=11)
ax.set_xlabel('Wind Speed [m/s]', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('Histogram of Wind Speed Ordinary Events with two separation approaches', fontsize=14, fontweight="bold")
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.legend(fontsize=11)
plt.tick_params(which='both', direction='in')
plt.savefig(bd_out_fig + "Test_SinglePixel_HistogramOrdEvt_SMEV.png", format='png', dpi=300, transparent=True)
plt.show()

#############################################################################
##----------------EMPIRIC CUMMULATIVE PROBABILITY FUNCTION-----------------##
#############################################################################

# Calcular la CDF empírica
ord_events_cdf_exp = np.sort(sep_events_thr)
cdf_exp            = np.arange(1, len(ord_events_cdf_exp)+1) / len(ord_events_cdf_exp) # Con el denominador así, llega excatamente a 1

ord_events_cdf_pot = np.sort(sep_events_int)
cdf_pot            = np.arange(1, len(ord_events_cdf_pot)+1) / len(ord_events_cdf_pot) # Con el denominador así, llega excatamente a 1

y_ticks     = np.arange(0, 1.1, 0.2)
y_ticks_lbs = [str(round(y_ticks[i], 1)) for i in range(len(y_ticks))]

x_ticks     = np.arange(0, 31, 5)
x_ticks_lbs = [str(round(x_ticks[i], 1)) for i in range(len(x_ticks))]

fig, ax = plt.subplots(figsize=(10, 8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(ord_events_cdf_exp, cdf_exp, marker='o', linestyle='-', color='#bc4f5e', alpha=0.7, label='Exponential')
ax.plot(ord_events_cdf_pot, cdf_pot, marker='o', linestyle='-', color='#7f8eb7', alpha=0.7, label='Potential')
ax.set_xlabel('Wind Speed [m/s]', fontsize=13)
ax.set_ylabel('Cumulative Probability', fontsize=13)
ax.set_title('Cumulative Distribution Function (CDF) of Ordinary events', fontsize=14, fontweight="bold")
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks_lbs, fontsize=11)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks_lbs, fontsize=11)
plt.tick_params(which='both', direction='in')
plt.savefig(bd_out_fig + "Test_SinglePixel_EmpriCDF_OrdEvents_SMEV.png", format='png', dpi=300, transparent=True)
plt.show()

