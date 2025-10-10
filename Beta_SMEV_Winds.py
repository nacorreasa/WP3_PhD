
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator
from scipy import stats
import statsmodels.api as sm
import pyarrow as pa
from smev_class import SMEV
import glob
import numpy as np
from scipy.stats import genextreme


#############################################################################
##-------------------------DEFINING IMPORTANT PATHS------------------------##
#############################################################################
bd_in_ws   = "/Dati/Data/COSMO-REA/Germany_WP2/OutDownload/WS_100/"
# bd_out_fig = "/Dati/Outputs/Plots/WP2_development/"

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

def gev_weibull(sample_arr, return_periods):
    # Ajustar la distribución GEV
    shape, loc, scale = stats.genextreme.fit(sample_arr)
    
    # Calcular los cuantiles para los períodos de retorno
    quantiles     = 1 - 1 / np.array(return_periods)
    return_values = stats.genextreme.ppf(quantiles, shape, loc, scale)

    return return_values

def weibull_extreme_value(sample_arr, return_periods):
    # Ajustar la distribución de Weibull
    shape, loc, scale = stats.weibull_min.fit(sample_arr, floc=0)
    
    # Calcular los cuantiles para los períodos de retorno
    quantiles     = 1 - 1 / np.array(return_periods)
    return_values = stats.weibull_min.ppf(quantiles, shape, loc, scale)
    
    return return_values, (shape, loc, scale)

def annual_maxima_gumbel(sample_arr, dates, return_periods):
    # Crear un DataFrame con las fechas y los valores
    df         = pd.DataFrame({'date': dates, 'wind_speed': sample_arr})
    df['date'] = pd.to_datetime(df['date'])
    
    # Calcular los máximos anuales
    annual_max = df.resample('Y', on='date')['wind_speed'].max()
    
    # Ajustar la distribución de Gumbel a los máximos anuales
    loc, scale = stats.gumbel_r.fit(annual_max)
    
    # Calcular los cuantiles para los períodos de retorno
    quantiles     = 1 - 1 / np.array(return_periods)
    return_values = stats.gumbel_r.ppf(quantiles, loc, scale)
    
    return return_values, annual_max

def annual_maxima_weibull(sample_arr, dates, return_periods):
    # Crear un DataFrame con las fechas y los valores
    df = pd.DataFrame({'date': dates, 'wind_speed': sample_arr})
    df['date'] = pd.to_datetime(df['date'])
    
    # Calcular los máximos anuales
    annual_max = df.resample('Y', on='date')['wind_speed'].max()
    
    # Ajustar la distribución de Weibull a los máximos anuales
    shape, loc, scale = stats.weibull_min.fit(annual_max, floc=0)
    
    # Calcular los cuantiles para los períodos de retorno
    quantiles     = 1 - 1 / np.array(return_periods)
    return_values = stats.weibull_min.ppf(quantiles, shape, loc, scale)
    
    return return_values, annual_max, (shape, loc, scale)

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


def simple_mev_newsep(s_arr, dates, return_periods, threshold, separation, durations, time_resolution, h_corr):

    # Convert dates to numpy datetime64 array
    dates_np = dates.to_numpy()

    return_levels = []
    for i in range(len(return_periods)):
        return_period = return_periods[i]
        # Initialize SMEV object
        rl_arr = SMEV(threshold, separation, return_period, durations, time_resolution)

        ###############################################################################
        sep_events, idt_events = separation_ws_events_max(s_arr, h_corr, dates_np)
        threshold              = np.percentile(sep_events, 90)
        ordinary_values        = sep_events[sep_events > threshold]
        ordinary_dates         = idt_events[sep_events > threshold]

        ordinary_events_new_df = pd.DataFrame({'value': ordinary_values})

        # Estimate SMEV parameters
        shape_new, scale_new = rl_arr.estimate_smev_parameters(ordinary_events_new_df['value'], rl_arr.left_censoring)

        # Calculate return values
        n_new     = len(ordinary_events_new_df)/10  # Average number of events per year
        intensity = rl_arr.smev_return_values(rl_arr.return_period, shape_new, scale_new, n_new)
        ###############################################################################

        return_levels.append(intensity)

    return_values = np.array(return_levels)

    return return_values

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

def gpd_fit(data, threshold, return_periods):
    # Extraer excedencias sobre el umbral
    exceedances = data[data > threshold] - threshold
    
    # Ajustar GPD
    shape, loc, scale = stats.genpareto.fit(exceedances)
    
    # Calcular la tasa de excedencia
    rate = len(exceedances) / len(data)

    list_sigma_ut = []
    list_ut       = []
    for i in range(len(return_periods)):
        return_period = return_periods[i]
    
        # Calcular el viento de retorno de 50 años (ecuación 10 del paper)
        if shape != 0:
            ut = threshold + scale * ((return_period * rate)**shape - 1) / shape
        else:
            ut = threshold + scale * np.log(return_period * rate)
        
        # Calcular la incertidumbre (ecuación 13 del paper)
        L = len(data) / 8760  # Longitud de los datos en años
        sigma_ut = scale / np.sqrt(L) * np.sqrt(1 + np.log(return_period)**2)

        list_ut.append(ut)
        list_sigma_ut.append(sigma_ut)
    
    return list_ut, list_sigma_ut

def gpd_fit_newsep(data, threshold, return_periods):
    # Extraer excedencias sobre el umbral
    exceedances = data[data > threshold] - threshold
    
    # Ajustar GPD
    shape, loc, scale = stats.genpareto.fit(exceedances)
    
    # Calcular la tasa de excedencia
    rate = len(exceedances) / len(data)

    list_sigma_ut = []
    list_ut       = []
    for i in range(len(return_periods)):
        return_period = return_periods[i]
    
        # Calcular el viento de retorno de 50 años (ecuación 10 del paper)
        if shape != 0:
            ut = threshold + scale * ((return_period * rate)**shape - 1) / shape
        else:
            ut = threshold + scale * np.log(return_period * rate)
        
        # Calcular la incertidumbre (ecuación 13 del paper)
        L = len(data) / 8760  # Longitud de los datos en años
        sigma_ut = scale / np.sqrt(L) * np.sqrt(1 + np.log(return_period)**2)

        list_ut.append(ut)
        list_sigma_ut.append(sigma_ut)
    
    return list_ut, list_sigma_ut

#############################################################################
##-----------------EXTRACTING AND CONCATENATING THE REGION-----------------##
#############################################################################

arr_cut_com = cut_concat_netcdf_files(bd_in_ws, years, min_lon, max_lon, min_lat, max_lat)

#############################################################################
##---------------------SELECTING ONE PIXEL TO TEST SMEV--------------------##
#############################################################################

sample_arr      = arr_cut_com.wind_speed.values
s_arr           = sample_arr[:, 0, 0]
threshold       = np.percentile(s_arr, 90)
separation      = 48
return_period   = 100
durations       = 60 # In Minutes
time_resolution = 60 # In Minutes

# Get ordinary events
start_date = '2009-01-01'  # Adjust this to the dataset start date
dates      = pd.date_range(start=start_date, periods=len(s_arr), freq='H')

# Convert dates to numpy datetime64 array
dates_np = dates.to_numpy()

# Initialize SMEV object
rl_arr = SMEV(threshold, separation, return_period, durations, time_resolution)

# Get ordinary events
ordinary_events_old = rl_arr.get_ordinary_events(s_arr, dates_np, name_col=None)

###############################################################################
sep_events, idt_events = separation_ws_events_max(s_arr, h_corr, dates_np)
# threshold              = np.percentile(sep_events, 90)
# ordinary_values        = sep_events[sep_events > threshold]
# ordinary_dates         = idt_events[sep_events > threshold]
ordinary_values        = sep_events
ordinary_dates         = [pd.to_datetime(idt_events[i]) for i in range(len(idt_events))]

ordinary_events_new_df = pd.DataFrame({'value': ordinary_values}, index = ordinary_dates)
n_ordinary_per_year    = ordinary_events_new_df.groupby(pd.Grouper(freq="Y")).count() 
n_ordinary             = n_ordinary_per_year['value'].mean().values.item()

# Estimate SMEV parameters
shape_new, scale_new = rl_arr.estimate_smev_parameters(ordinary_events_new_df['value'], rl_arr.left_censoring)

# Calculate return values
# n_new     = len(ordinary_events_new_df)/10  # Average number of events per year
intensity = rl_arr.smev_return_values(rl_arr.return_period, shape_new, scale_new, n_ordinary)
###############################################################################



# Check if ordinary_events is empty or None
if ordinary_events_old and len(ordinary_events_old) > 0:
    print("No ordinary events found. Check your threshold.")
    # For example:
    # rl_arr.threshold = np.percentile(s_arr, 75)  # Using 75th percentile
    # ordinary_events_old = rl_arr.get_ordinary_events_old(s_arr, dates_np, name_col=None)


    arr_vals, arr_dates, n_ordinary_per_year = rl_arr.remove_short(ordinary_events_old, rl_arr.min_duration)

    # Prepare data for parameter estimation
    ordinary_values = []
    for event in ordinary_events_old:
        event_indices = np.where(np.isin(dates_np, event))[0]
        ordinary_values.extend(s_arr[event_indices])

    ordinary_events_old_df = pd.DataFrame({'value': ordinary_values})

    # Estimate SMEV parameters
    shape, scale = rl_arr.estimate_smev_parameters(ordinary_events_old_df['value'], rl_arr.left_censoring)

    # Calculate return values
    n = n_ordinary_per_year.mean().values.item()  # Average number of events per year
    intensity = rl_arr.smev_return_values(rl_arr.return_period, shape, scale, n)

    print(f"Return level intensity for {return_period}-year return period: {intensity}")
else:
    print("Analysis could not be completed due to lack of ordinary events.")

#############################################################################
##------------PERFORMING SOME CHECKS ON THE THRESHOLDS---------------------##
#############################################################################

print(f"90th percentile threshold: {np.percentile(s_arr, 90)}")

rl_arr.threshold = np.percentile(s_arr, 90)
ordinary_events_90 = rl_arr.get_ordinary_events(s_arr, dates_np)
print(f"Number of events (90th percentile): {len(ordinary_events_90)}")

#############################################################################
##------LOOPING OVER THE RETURN PERIODS TO OBTAIN THE RETURN LEVELS--------##
#############################################################################

# Selecting the sample array
sample_arr      = arr_cut_com.wind_speed.values
s_arr           = sample_arr[:, 0, 0]

# Get the dates
start_date = '2009-01-01'  # Adjust this to the dataset start date
dates      = pd.date_range(start=start_date, periods=len(s_arr), freq='H')

# Setting up the SMEV parameters
threshold       = np.percentile(s_arr, 95)
separation      = 48
return_period   = 100
durations       = 60 # In Minutes
time_resolution = 60 # In Minutes
h_corr          = 24 # In Hours


# List of return periods
return_periods = [2, 5, 10, 20]

# # Annual Maxima - Return levels:
# am_results, annual_maxima_series, weibull_params = annual_maxima_weibull(s_arr, dates, return_periods)
# # am_results, annual_maxima_series = annual_maxima_gumbel(s_arr, dates, return_periods)
# print("Annual Maxima results:")
# for period, value in zip(return_periods, am_results):
#     print(f"Return period {period} years: {value:.2f}")

# print("\nMáximos anuales:")
# print(annual_maxima_series)

# # GEV general - Return levels:
# gev_results = gev_weibull(s_arr, return_periods)
# print("GEV results:")
# for period, value in zip(return_periods, gev_results):
#     print(f"Return period {period} years: {value:.2f}")

# # GEV Weibull - Return levels: 
# weibull_results, weibull_params = weibull_extreme_value(s_arr, return_periods)
# print("GEV-Weibull results:")
# for period, value in zip(return_periods, weibull_results):
#     print(f"Return period {period} years: {value:.2f}")
# print(f"Fit-to-all Weibull  shape, loc, scale: {weibull_params}")

# GEVD-PMM - Return levels:
ut_gevd, gev_param = gevd_fit(s_arr, return_periods)
print("GEVD-PMM results:")
for period, value in zip(return_periods, ut_gevd):
    # print(f"Return period {period} years: {value:.2f} ± {1.96*sigma:.2f} m/s")
    print(f"Return period {period} years: {value:.2f} m/s shape, loc, scale: {gev_param}")

# SMEV - Return levels:
smev_results =simple_mev_newsep(s_arr, dates, return_periods, threshold, separation, durations, time_resolution, h_corr)
print("SMEV results:")
for period, value in zip(return_periods, smev_results):
    print(f"Return period {period} years: {value:.2f}")

# GPD-POT - Return levels:
sep_events, idt_events = separation_ws_events_max(s_arr, h_corr, dates_np)
threshold_gpd          = np.percentile(sep_events, 90)
ut_gpd, sigma_ut_gpd   = gpd_fit(sep_events, threshold_gpd, return_periods)
print("GPD-POT results:")
for period, value, sigma in zip(return_periods, ut_gpd, sigma_ut_gpd):
    print(f"Return period {period} years: {value:.2f} ± {1.96*sigma:.2f} m/s")


#############################################################################
##-------------------PROBABILITY PLOTTING POSITION------------------------##
#############################################################################
# Extraer máximos anuales
annual_max = np.array([np.max(s_arr[i*8760:(i+1)*8760]) for i in range(10)])
n          = len(annual_max)

from scipy.stats import rankdata
rank_am = rankdata(annual_max)

pp = rank_am / (n + 1)

rt_pp = 1 / ( 1- pp)


#############################################################################
##---------------------------PLOTTING THE RESULTS--------------------------##
#############################################################################

fig = plt.figure(figsize=(8, 6))
ax   = fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.scatter(return_periods, smev_results, c='#7f8eb7', s=50)
ax.plot(return_periods, smev_results, c='#7f8eb7', label='SMEV', linewidth=2)
ax.scatter(return_periods, ut_gevd, c='#94bb8b', s=50)
ax.plot(return_periods, ut_gevd, c='#94bb8b', label='GEV', linewidth=2)
ax.scatter(return_periods, ut_gpd, c='#9c7762', s=50)
ax.plot(return_periods, ut_gpd, c='#9c7762', label='GPD', linewidth=2)
ax.scatter(rt_pp, annual_max, c='red', s=80)
ax.set_xlabel("Return Period [years]", fontsize=13)
ax.set_ylabel("Wind Speed [m/s]", fontsize=13)
ax.set_title("Comparison of Extreme Value Analysis Methods", fontsize=14, fontweight="bold")
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xticks(return_periods)
ax.set_xticklabels(return_periods, fontsize=12)
ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(which='both', direction='in')
ax.legend(fontsize=8.5)
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()

