
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

    cv_params = np.std(shape_params) / np.mean(shape_params)
    return cv_params

def cv_weibull_distribution(ord_events, num_fractions=20, num_repetitions=50):
    """
    Function to generate a distribution of CV based on the number of repetitions
    of the calculus of the CV for the shape parameters generated with Boostrap 
    in the weibull_shape_cv fucntion. i.e., generates a CV distribution 
    (for the boxplots) for each fraction of ordianry events. 
    """
    fractions = np.linspace(0.05, 1, num_fractions)
    cv_values = []

    for fraction in fractions:
        n = int(len(ord_events) * fraction)
        fraction_cvs = []
        for _ in range(num_repetitions):
            sorted_data = np.sort(ord_events)[::-1]
            data_subset = sorted_data[:n]
            cv = weibull_shape_cv(data_subset)
            fraction_cvs.append(cv)
        cv_values.append(fraction_cvs)
        print(fraction)
    return cv_values

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

#############################################################################
##------WEIBULL PLOT FOR COMBINATIONS OF LEFT CENSORING AND SEPARATION-----##
#############################################################################

def weibull_plot_censored(data, fig_title, name_fig, censoring_percentages=[75, 80, 85], confidence_level=0.95, method = 'Trad', positive_x = False):
    """
    -method : 'Trad' for ln(-ln(1-p)), any other string for ln(ln(1 / (1 - p)))
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
    fig, ax = plt.subplots(figsize=(10, 8))
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
        plt.scatter(x_top, y_top, marker='x', label=f'Top {100-censoring_percentage:.0f}%', alpha=0.7, color=color)
        plt.plot(x_fit, y_fit, ls ='--', label=f'Fitted Line (Top {100-censoring_percentage:.0f}%)', color=color, linewidth=2)
        # plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color=color, alpha=0.2, label=f'95% Confidence Interval (Top {100-censoring_percentage:.0f}%)')

        print(f"Top {100-censoring_percentage:.0f}%:")
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"Weibull shape: {weibull_param[0]:.4f}")
        print(f"Weibull scale: {weibull_param[1]:.4f}\n")

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('ln(Wind Speed)', fontsize=14)
    plt.title(fig_title, fontsize=14)
    if positive_x:
        plt.xlim(left=0)  # Set x-axis to start from 0 -- Logatirmic range of values of WS
        plt.ylim(2, 3.5) 
        # plt.ylim(bottom=0)  # Set y-axis to start from 0
    else:
        pass
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tick_params(which='both', direction='in')
    plt.savefig(bd_out_fig+name_fig, format='png', dpi=300, transparent=True)
    plt.show()

    return global_shapes, global_scales, ci_shapes, ci_scales


## EXPONENTIAL DECAY - SEPARATION
df_ord_events_exp = df_ord_events_exp20 # Because the selections of the ordinary evens comes first than the RL computation
wind_speeds_exp   = df_ord_events_exp['value'].values

_, _, _, _                                                             = weibull_plot_censored(wind_speeds_exp, 'Weibull Plot of Ordinary events [Exp. Approach]', "Test_SinglePixel_WeibullPlot_SMEV.png",  censoring_percentages=[75, 80, 85, 90, 95], method = 'Trad', positive_x =False)
list_shape_exp, list_scale_exp, list_ci_shape_exp, list_ci_scale_exp  = weibull_plot_censored(wind_speeds_exp, 'Weibull Plot of Ordinary events [Exp. Approach] (Positive)', "Test_SinglePixel_WeibullPlot_SMEV_PositiveDomain.png", censoring_percentages=[75, 80, 85, 90, 95], method = 'Trad', positive_x =True)

## POTENTIAL DECAY - SEPARATION
df_ord_events_pot = df_ord_events_pot20 # Because the selections of the ordinary evens comes first than the RL computation
wind_speeds_pot   = df_ord_events_pot['value'].values

_, _, _, _                                                             = weibull_plot_censored(wind_speeds_pot, 'Weibull Plot of Ordinary events [Pot. Approach]', "Test_SinglePixel_WeibullPlot_PotSMEV.png",  censoring_percentages=[75, 80, 85, 90, 95], method = 'Trad', positive_x =False)
list_shape_pot, list_scale_pot, list_ci_shape_pot, list_ci_scale_pot  = weibull_plot_censored(wind_speeds_pot, 'Weibull Plot of Ordinary events [Pot. Approach] (Positive)', "Test_SinglePixel_WeibullPlot_PotSMEV_PositiveDomain.png", censoring_percentages=[75, 80, 85, 90, 95], method = 'Trad', positive_x =True)


#############################################################################
##------PLOTTING SHAPE AND SCALE PARAMETER DOR EACH CENRING THERSHOLD------##
#############################################################################

x_tick_labels = ['Top 25%', 'Top 20%', 'Top 15%', 'Top 10%', 'Top 05%'] 
x_tick        = np.arange(1, len(x_tick_labels)+1)

ci_shape_exp_lower = [ci[0] for ci in list_ci_shape_exp]
ci_shape_exp_upper = [ci[1] for ci in list_ci_shape_exp]

ci_shape_pot_lower = [ci[0] for ci in list_ci_shape_pot]
ci_shape_pot_upper = [ci[1] for ci in list_ci_shape_pot]

ci_scale_exp_lower = [ci[0] for ci in list_ci_scale_exp]
ci_scale_exp_upper = [ci[1] for ci in list_ci_scale_exp]

ci_scale_pot_lower = [ci[0] for ci in list_ci_scale_pot]
ci_scale_pot_upper = [ci[1] for ci in list_ci_scale_pot]

# y_tick_labels = np.arange(-200, 201, 100)
# y_tick        = np.arange(-200, 201, 100)

Fig  = plt.figure(figsize=(10, 4))
ax   = Fig.add_subplot(121)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.scatter(x_tick, list_shape_pot, marker='o', alpha=0.7, color='#7f8eb7')
ax.plot(x_tick, list_shape_pot, label=f'Potential',  color='#7f8eb7', linewidth=2)
ax.fill_between(x_tick, ci_shape_pot_lower, ci_shape_pot_upper, color='#7f8eb7', alpha=0.2, label='95% CI [Pot]')
ax.scatter(x_tick, list_shape_exp, marker='o', alpha=0.7, color='#bc4f5e')
ax.plot(x_tick, list_shape_exp, label=f'Exponential',  color='#bc4f5e', linewidth=2)
ax.fill_between(x_tick, ci_shape_exp_lower, ci_shape_exp_upper, color='#bc4f5e', alpha=0.2, label='95% CI [Exp]')
ax.set_xticks(x_tick, x_tick_labels,fontsize = 15, rotation = 0)
# ax.set_yticks(y_percent, y_percent,fontsize = 15, rotation = 0)
ax.set_title(u'a) Shape parameter per portion',fontsize = 15, weight='bold', loc='left')
ax.set_ylabel(u'[-]',fontsize = 15)
ax.set_xlabel(u'Top portion Ord.Events',fontsize = 15)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
# ax.yaxis.set_major_locator(ticker.FixedLocator(y_percent))
ax.tick_params(which='both', direction='in', labelsize = 12)

ax1   = Fig.add_subplot(122)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.scatter(x_tick, list_scale_pot, marker='o', alpha=0.7, color='#7f8eb7')
ax1.plot(x_tick, list_scale_pot, label=f'Potential',  color='#7f8eb7', linewidth=2)
ax1.fill_between(x_tick, ci_scale_pot_lower, ci_scale_pot_upper, color='#7f8eb7', alpha=0.2, label='95% CI [Pot]')
ax1.scatter(x_tick, list_scale_exp, marker='o', alpha=0.7, color='#bc4f5e')
ax1.plot(x_tick, list_scale_exp, label=f'Exponential',  color='#bc4f5e', linewidth=2)
ax1.fill_between(x_tick, ci_scale_exp_lower, ci_scale_exp_upper, color='#bc4f5e', alpha=0.2, label='95% CI [Exp]')
ax1.set_xticks(x_tick, x_tick_labels,fontsize = 15, rotation = 0)
# ax1.set_yticks(y_percent, y_percent,fontsize = 15, rotation = 0)
ax1.set_title(u'b) Scale parameter per portion',fontsize = 15, weight='bold', loc='left')
ax1.set_ylabel(u'[m/s]',fontsize = 15)
ax1.set_xlabel(u'Top portion Ord.Events',fontsize = 15)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
# ax1.yaxis.set_major_locator(ticker.FixedLocator(y_percent))
ax1.tick_params(which='both', direction='in', labelsize = 12)
ax1.legend()

plt.subplots_adjust(wspace=0.30, hspace=0.45, left=0.10, right =0.97, bottom = 0.20) 
# plt.savefig(bd_out_fig+'TestSinglePixel_ShapeScale_CensoredPortions.png', format='png', dpi=300, transparent=True)
plt.show()


############################################################################################
##--DISTRIBUTION CV SHAPE FROM BOOTSTRAPED SAMPLES FOR EACH PORTION OF ORDINARY EVENTS----##
############################################################################################

num_fractions = 20
cv_values_exp = cv_weibull_distribution(wind_speeds_exp)
cv_values_pot = cv_weibull_distribution(wind_speeds_pot)

all_values = cv_values_exp + cv_values_pot
min_val    = min(min(box) for box in all_values)
max_val    = max(max(box) for box in all_values)
y_ticks    = np.logspace(np.log10(min_val), np.log10(max_val), num=6)

xticks       = [1, 5, 10, 15, 20]
xtick_labels = ['top 5%', 'top 25%', 'top 50%', 'top 75%', 'All']

color_exp = '#bc4f5e'
color_pot = '#7f8eb7'

fig, ax = plt.subplots(figsize=(10, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
box_exp = ax.boxplot(cv_values_exp, positions=range(1, num_fractions + 1), widths=0.6,
                     boxprops=dict(color=color_exp), medianprops=dict(color=color_exp),
                     whiskerprops=dict(color=color_exp), capprops=dict(color=color_exp),
                     flierprops=dict(marker='o', markerfacecolor=color_exp, markersize=5,
                                     linestyle='none', markeredgecolor=color_exp))
box_pot = ax.boxplot(cv_values_pot, positions=range(1, num_fractions + 1), widths=0.6,
                     boxprops=dict(color=color_pot), medianprops=dict(color=color_pot),
                     whiskerprops=dict(color=color_pot), capprops=dict(color=color_pot),
                     flierprops=dict(marker='o', markerfacecolor=color_pot, markersize=5,
                                     linestyle='none', markeredgecolor=color_pot))
ax.set_xlabel('Fraction of values used for the fit', fontsize=14, labelpad=15)
ax.set_ylabel('CV within 5 %-wide bins', fontsize=14, labelpad=15)
ax.set_title('Coefficient of Variation of Weibull Shape Parameter')
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels, fontsize=12)
ax.set_ylim(min_val * 0.9, max_val * 1.1)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{x:.2e}' for x in y_ticks])
ax.plot([], [], color=color_exp, label='Exponential')
ax.plot([], [], color=color_pot, label='Potencial')
ax.legend(fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.tick_params(which='both', direction='in')
plt.tight_layout()
plt.savefig(bd_out_fig+'TestSinglePixel_CVShapePramWeibull.png', format='png', dpi=300, transparent=True)
plt.show()





# def ord_events_weibull_plot(df_ord_events, column_name):
#     """
#     Function to fit the Weibull distribution to the values in a pandas dataframe
#     and transofrm the values in logatihims for the Wibull Plot.

#     INPUTS:
#     - df_ord_events (pd.DataFrame) : Input DataFrame
#     - column_name (str)            : Name of the column to fit
    
#     OUTPUTS:
#     - log_sorted_data: 
#     - log_y_fit      :
#     - x              :
#     - x_fit          :

#     """
#     ord_events = df_ord_events[column_name].values

#     # Sort the data in descending order
#     sorted_data = np.sort(ord_events)[::-1]

#     # Calculate empirical exceedance probabilities
#     n = len(sorted_data)
#     p = np.arange(1, n + 1) / (n + 1)

#     # Calculate x-axis values for Weibull plot
#     x = np.log(np.log(1/p))

#     # # Fit Weibull distribution
#     shape, loc, scale = stats.weibull_min.fit(sorted_data, floc=0)
#     # Generate points for the fitted Weibull line
#     x_fit = np.linspace(x.min(), x.max(), 100)
#     y_fit = scale * np.exp(x_fit/shape) + loc

#     # Logarithmic transformation to sorted values
#     log_sorted_data = np.log(sorted_data)
#     log_y_fit       = np.log(y_fit)

#     return log_sorted_data, log_y_fit, x, x_fit


# # Definir los valores enteros deseados para Wind Speed [m/s] en eje Y
# ytick_labels_int = np.arange(1, int(np.ceil(np.exp(log_sorted_data_pot05).max())) + 1, 5)  # Incremento de 5
# yticks           = np.log(ytick_labels_int)

# fig, ax = plt.subplots(figsize=(10, 8))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.scatter(x_exp25, log_sorted_data_exp25, c='#7fcdbb', marker = 'x', label='Ord. Events (L.C = 25%)')
# ax.scatter(x_exp20, log_sorted_data_exp20, c='#41b6c4', marker = 'x', label='Ord. Events (L.C = 20%)')
# ax.scatter(x_exp15, log_sorted_data_exp15, c='#1d91c0', marker = 'x', label='Ord. Events (L.C = 15%)')
# ax.scatter(x_exp10, log_sorted_data_exp10, c='#225ea8', marker = 'x', label='Ord. Events (L.C = 10%)')
# ax.scatter(x_exp05, log_sorted_data_exp05, c='#0c2c84', marker = 'x', label='Ord. Events (L.C = 05%)')
# ax.plot(x_fit_exp25, log_y_fit_exp25, c='#7fcdbb', ls = '--', linewidth=2, label='Fitted Weibull (Top 25%)')
# ax.plot(x_fit_exp05, log_y_fit_exp05, c='#0c2c84', ls = '--', linewidth=2, label='Fitted Weibull (Top 05%)')
# ax.set_xlabel('log(log(1/p))', fontsize=13)
# ax.set_ylabel('Wind Speed [m/s]', fontsize=13)
# ax.set_title('Weibull Plot for Extreme Wind Analysis (Exponential decay)', fontsize=14, fontweight="bold")
# ax.set_yticks(yticks)
# ax.set_yticklabels([f'{val}' for val in ytick_labels_int], fontsize=11)
# ax.legend(fontsize=10)
# ax.grid(True, which="both", ls="-", alpha=0.2)
# plt.tick_params(which='both', direction='in')
# # plt.savefig(bd_out_fig+"Test_SinglePixel_WeibullPlot_SMEV.png", format='png', dpi=300, transparent=True)
# plt.show()

