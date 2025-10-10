
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.ticker as ticker
from scipy import stats
import statsmodels.api as sm
import pyarrow as pa
from smev_class import SMEV
import glob
import numpy as np
from scipy.stats import genextreme

"""
Code to select one single pixel of the COSMO-REA6 Dataset and condunct the test for 
applying SMEV on high wind speeds. Initial code fort he exploration of the SMEV 
feasibility for winds.

Author: Nathalia Correa-Sánchez
"""

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
years = [2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009] 

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


def simple_mev(s_arr, dates, return_periods, threshold, separation, durations, time_resolution):

    # Convert dates to numpy datetime64 array
    dates_np = dates.to_numpy()

    return_levels = []
    for i in range(len(return_periods)):
        return_period = return_periods[i]
        # Initialize SMEV object
        rl_arr = SMEV(threshold, separation, return_period, durations, time_resolution)

        ordinary_events                          = rl_arr.get_ordinary_events(s_arr, dates_np, name_col=None)
        arr_vals, arr_dates, n_ordinary_per_year = rl_arr.remove_short(ordinary_events, rl_arr.min_duration)

        # Prepare data for parameter estimation
        ordinary_values = []
        for event in ordinary_events:
            event_indices = np.where(np.isin(dates_np, event))[0]
            ordinary_values.extend(s_arr[event_indices])

        ordinary_events_df = pd.DataFrame({'value': ordinary_values})

        # Estimate SMEV parameters
        shape, scale = rl_arr.estimate_smev_parameters(ordinary_events_df['value'], rl_arr.left_censoring)

        # Calculate return values
        n         = n_ordinary_per_year.mean().values.item()  # Average number of events per year
        intensity = rl_arr.smev_return_values(rl_arr.return_period, shape, scale, n)

        return_levels.append(intensity)

    return_values = np.array(return_levels)

    return return_values

    
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


def gpd_fit_scipy(data, threshold, return_periods):
    # Extraer excedencias sobre el umbral
    exceedances = data
    
    # Ajustar GPD
    shape, loc, scale = stats.genpareto.fit(exceedances)
    
    # Calcular la tasa de excedencia
    rate = len(exceedances) / len(data)

    list_ut = []
    list_sigma_ut = []
    for return_period in return_periods:
        # Calcular el cuantil (valor de retorno) usando SciPy
        q = 1 - 1 / (return_period * rate)
        ut = threshold + stats.genpareto.ppf(q, shape, loc, scale)
        
        # Calcular la incertidumbre (ecuación 13 del paper)
        L = len(data) / 8760  # Longitud de los datos en años
        sigma_ut = scale / np.sqrt(L) * np.sqrt(1 + np.log(return_period)**2)

        list_ut.append(ut)
        list_sigma_ut.append(sigma_ut)
    
    return list_ut, list_sigma_ut

def gpd_fit_scipy_threshold(data, threshold, return_periods):
    # Extraer excedencias sobre el umbral
    exceedances = data[data > threshold] - threshold
    
    # Ajustar GPD
    shape, loc, scale = stats.genpareto.fit(exceedances)
    
    # Calcular la tasa de excedencia
    rate = len(exceedances) / len(data)

    list_ut = []
    list_sigma_ut = []
    for return_period in return_periods:
        # Calcular el cuantil (valor de retorno) usando SciPy
        q = 1 - 1 / (return_period * rate)
        ut = threshold + stats.genpareto.ppf(q, shape, loc, scale)
        
        # Calcular la incertidumbre (ecuación 13 del paper)
        L = len(data) / 8760  # Longitud de los datos en años
        sigma_ut = scale / np.sqrt(L) * np.sqrt(1 + np.log(return_period)**2)

        list_ut.append(ut)
        list_sigma_ut.append(sigma_ut)
    
    return list_ut, list_sigma_ut

def weibull_shape_cv(data, num_bootstrap=100):
    """
    Funcion to compute the Coefficient of Variation (CV) from a bootstrap sample. 
    The function generates bootstrap samples with random coice and replacement, 
    the amount of samples by deflout is 100. For each sample, the functions fits
    the Wibull distribution and gget the shape parameter. From the shape parametters
    collections it computes the CV.
    """
    shape_params = []
    for _ in range(num_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        shape, _, _ = stats.weibull_min.fit(bootstrap_sample, floc=0)
        shape_params.append(shape)
    return np.std(shape_params) / np.mean(shape_params)



#############################################################################
##-----------------EXTRACTING AND CONCATENATING THE REGION-----------------##
#############################################################################

arr_cut_com = cut_concat_netcdf_files(bd_in_ws, years, min_lon, max_lon, min_lat, max_lat)

#############################################################################
##------LOOPING OVER THE RETURN PERIODS TO OBTAIN THE RETURN LEVELS--------##
#############################################################################

# Selecting the sample array
sample_arr      = arr_cut_com.wind_speed.values
s_arr           = sample_arr[:, 0, 0]

# Get the dates
start_date = '2009-01-01'  # Adjust this to the dataset start date
dates      = pd.date_range(start=start_date, periods=len(s_arr), freq='H')

# Decorrelation separation - Exponential decay
h_corr  = 25 # In Hours
# Decorrelation separation - Potential decay
t_corr  = 37 # In Hours

# Setting up the SMEV  single parameters
threshold_meas  = 0  # For the valid measurements 0.1 mm in precipitation records in winds could be 0 m/s
durations       = 60 # In Minutes
time_resolution = 60 # In Minutes

# List of return periods as other SMEV parameters
return_periods = [2, 5, 10, 20]

#############################################################################
##----RETURN LEVELS SMEV WITH DIFFERENT THRESHOLDS FOR LEFT SENSORING------##
#############################################################################

# SMEV - Return levels (Exponential decay + Top 25):
smev_results_exp25, df_ord_events_exp25 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, h_corr, durations, time_resolution, [0.75, 1])
print("SMEV Top 25-exp results:")
for period, value in zip(return_periods, smev_results_exp25):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Exponential decay + Top 20):
smev_results_exp20, df_ord_events_exp20 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, h_corr, durations, time_resolution, [0.80, 1])
print("SMEV 20-exp results:")
for period, value in zip(return_periods, smev_results_exp20):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Exponential decay + Top 15):
smev_results_exp15, df_ord_events_exp15 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, h_corr, durations, time_resolution, [0.85, 1])
print("SMEV 15-exp results:")
for period, value in zip(return_periods, smev_results_exp15):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Exponential decay + Top 10):
smev_results_exp10, df_ord_events_exp10 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, h_corr, durations, time_resolution, [0.90, 1])
print("SMEV 10-exp results:")
for period, value in zip(return_periods, smev_results_exp10):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Exponential decay + Top 05):
smev_results_exp05, df_ord_events_exp05 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, h_corr, durations, time_resolution, [0.95, 1])
print("SMEV 05-exp results:")
for period, value in zip(return_periods, smev_results_exp05):
    print(f"Return period {period} years: {value:.2f}")


# SMEV - Return levels (Potential decay + Top 25):
smev_results_pot25, df_ord_events_pot25 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, t_corr, durations, time_resolution, [0.75, 1])
print("SMEV Top 25-pot results:")
for period, value in zip(return_periods, smev_results_pot25):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Potential decay + Top 20):
smev_results_pot20, df_ord_events_pot20 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, t_corr, durations, time_resolution, [0.80, 1])
print("SMEV 20-pot results:")
for period, value in zip(return_periods, smev_results_pot20):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Potential decay + Top 15):
smev_results_pot15, df_ord_events_pot15 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, t_corr, durations, time_resolution, [0.85, 1])
print("SMEV 15-pot results:")
for period, value in zip(return_periods, smev_results_pot15):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Potential decay + Top 10):
smev_results_pot10, df_ord_events_pot10 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, t_corr, durations, time_resolution, [0.90, 1])
print("SMEV 10-pot results:")
for period, value in zip(return_periods, smev_results_pot10):
    print(f"Return period {period} years: {value:.2f}")

# SMEV - Return levels (Potential decay + Top 05):
smev_results_pot05, df_ord_events_pot05 = simple_mev_newsep(s_arr, dates, return_periods, threshold_meas, t_corr, durations, time_resolution, [0.95, 1])
print("SMEV 05-pot results:")
for period, value in zip(return_periods, smev_results_pot05):
    print(f"Return period {period} years: {value:.2f}")

smev_exp = [smev_results_exp25, smev_results_exp20, smev_results_exp15, smev_results_exp10, smev_results_exp05]
smev_pot = [smev_results_pot25, smev_results_pot20, smev_results_pot15, smev_results_pot10, smev_results_pot05]


#############################################################################
##------------------RETURN LEVELS TRADITIONAL METHODS----------------------##
#############################################################################

# Convert dates to numpy datetime64 array
dates_np        = dates.to_numpy()
# Threhold for separated max events in GPD
top_ord_events  = 80 

# GEVD-PMM - Return levels:
ut_gevd, gev_param = gevd_fit(s_arr, return_periods)
print("GEVD-PMM results:")
for period, value in zip(return_periods, ut_gevd):
    print(f"Return period {period} years: {value:.2f} m/s shape, loc, scale: {gev_param}")

# GPD-POT separeted events:
sep_events, idt_events = separation_ws_events_max(s_arr, h_corr, dates_np)
threshold_gpd          = np.percentile(sep_events, top_ord_events)

# 1) GPD-POT - Return levels:
ut_gpd, sigma_ut_gpd   = gpd_fit_scipy(sep_events, threshold_gpd, return_periods)
print("GPD-POT results:")
for period, value, sigma in zip(return_periods, ut_gpd, sigma_ut_gpd):
    print(f"Return period {period} years: {value:.2f} ± {1.96*sigma:.2f} m/s")

# # 2) GPD-POT 2nd threhold - Return levels:
# ut_gpd_2thr, sigma_ut_gpd_2thr = gpd_fit_scipy_threshold(sep_events, threshold_gpd, return_periods)
# print("GPD-POT 2nd threshold results:")
# for period, value, sigma in zip(return_periods, ut_gpd_2thr, sigma_ut_gpd_2thr):
#     print(f"Return period gpd 2nd Threshold {period} years: {value:.2f} ± {1.96*sigma:.2f} m/s")

#############################################################################
##-------------------PROBABILITY PLOTTING POSITION------------------------##
#############################################################################
# Extraer máximos anuales
annual_max = np.array([np.max(s_arr[i*8760:(i+1)*8760]) for i in range(10)])
n          = len(annual_max)

from scipy.stats import rankdata
rank_am = rankdata(annual_max)
pp      = rank_am / (n + 1)
rt_pp   = 1 / ( 1- pp)

#############################################################################
##------PLOTTING THE RESULTS FOR EXPONENTIAL AND POTENTIAL APROACH---------##
#############################################################################

colors_pot = ['#bdc5d9', '#9ea8c8', '#7f8eb7', '#6375a6', '#485c95']
colors_exp = ['#e1a2aa', '#cf7883', '#bc4f5e', '#a93a48', '#962532']

# Crear el gráfico
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.fill_between(return_periods, smev_exp[0], smev_exp[-1], color='#F3D9DD', alpha=0.9, label='SMEV. Exp')
for i, smev_results in enumerate(smev_exp):
    label = f'SMEV[Exp] Top {25-i*5}%'
    ax.scatter(return_periods, smev_results, c=colors_exp[i], s=50)
    ax.plot(return_periods, smev_results, c=colors_exp[i], label=label, linewidth=2)
ax.scatter(return_periods, ut_gevd, c='#94bb8b', s=50)
ax.plot(return_periods, ut_gevd, c='#94bb8b', label='GEV', linewidth=2)
ax.scatter(return_periods, ut_gpd, c='#9c7762', s=50)
ax.plot(return_periods, ut_gpd, c='#9c7762', label='GPD', linewidth=2)
ax.scatter(rt_pp, annual_max, c='red', s=80)
ax.set_xlabel("Return Period [years]", fontsize=13)
ax.set_ylabel("Wind Speed [m/s]", fontsize=13)
ax.set_title("Comparison Methods [Exp]", fontsize=14, fontweight="bold")
# ax.set_xscale('log')
ax.set_xticks(return_periods)
ax.set_xticklabels(return_periods, fontsize=12)
ax.set_yticklabels([f'{int(y)}' for y in ax.get_yticks()], fontsize=12)
ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in')
ax.legend(fontsize=8.5)
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

ax1 = fig.add_subplot(122)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.fill_between(return_periods, smev_pot[0], smev_pot[-1], color='gray', alpha=0.2, label='SMEV. Pot')
for i, smev_results in enumerate(smev_pot):
    label = f'SMEV[Pot] Top {25-i*5}%'
    ax1.scatter(return_periods, smev_results, c=colors_pot[i], s=50)
    ax1.plot(return_periods, smev_results, c=colors_pot[i], ls = '--', label=label, linewidth=2)
ax1.scatter(return_periods, ut_gevd, c='#94bb8b', s=50)
ax1.plot(return_periods, ut_gevd, c='#94bb8b', label='GEV', linewidth=2)
ax1.scatter(return_periods, ut_gpd, c='#9c7762', s=50)
ax1.plot(return_periods, ut_gpd, c='#9c7762', label='GPD', linewidth=2)
ax1.scatter(rt_pp, annual_max, c='red', s=80)
ax1.set_xlabel("Return Period [years]", fontsize=13)
ax1.set_ylabel("Wind Speed [m/s]", fontsize=13)
ax1.set_title("Comparison Methods [Pot]", fontsize=14, fontweight="bold")
# ax1.set_xscale('log')
ax1.set_xticks(return_periods)
ax1.set_xticklabels(return_periods, fontsize=12)
ax1.set_yticklabels([f'{int(y)}' for y in ax1.get_yticks()], fontsize=12)
ax1.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', direction='in')
ax1.legend(fontsize=8.5)
ax1.grid(True, which='major', linestyle='-', alpha=0.7)
ax1.grid(True, which='minor', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.subplots_adjust(wspace=0.20, hspace=0.45) 
plt.savefig(bd_out_fig+"Test_SMEV_GEV_GPD_SepEv.png", format='png', dpi=300, transparent=True)
plt.show()
