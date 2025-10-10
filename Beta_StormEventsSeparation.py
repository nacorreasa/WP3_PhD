import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as tck
from scipy import stats
import statsmodels.api as sm
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

indice = np.where(rs < corr_lim)[0][0] # Correlation threshold chosen indicating the temporal structure is week/Minimum influenceof the past values
h_corr = lags[indice]

y_labels    = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
y_positions = np.maximum(y_labels, 0.1) 

fig, ax = plt.subplots(figsize=(10, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Time [hours]", fontsize=15)
ax.set_ylabel("Correlation coefficient", fontsize=15)
ax.set_title("Lagged temporal autocorrelation", fontsize=14, fontweight="bold")
ax.scatter(lags, rs, c='#7f8eb7', s=50, zorder=3)
ax.plot(lags, rs, c='#7f8eb7', label='Autocorrelation', linewidth=2, zorder=2)
ax.axhline(y=corr_lim, color='red', linestyle='--', label='1/e threshold', zorder=1)
ax.set_yscale('log')
ax.set_ylim(0.1, 1)
ax.set_xticks(lags)
ax.set_xticklabels(lags, fontsize=15, rotation=0)
ax.set_yticks(y_positions)
ax.set_yticklabels([f'{y:.1f}' for y in y_labels], fontsize=15, rotation=0)
ax.yaxis.set_major_locator(tck.FixedLocator(y_positions))
ax.yaxis.set_minor_locator(NullLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, which='major', linestyle='-', alpha=0.3, zorder=0)
ax.grid(True, which='minor', linestyle=':', alpha=0.15, zorder=0)
plt.tick_params(which='both', direction='in')
ax.legend(fontsize=15, loc='best')
plt.tight_layout()
# plt.savefig(bd_out_fig+"TestSinglePixel_TimeLagged_CrossCorr.png", format='png', dpi=300, transparent=True)
plt.show()

########################################################################################
##-ITERATING OVER THE TIME SERIE TO SEPARATE THE EVENTS BASED ON THE AOUTOCORRELOGRAM-##
########################################################################################

sep_events, idt_events = separation_ws_events_max(s_arr, h_corr, dates_np)

########################################################################################
##----------------------PLOTTING THE DATA AND THE SELECTED ORDINARY EVENTS------------##
########################################################################################

y_labels = np.array([0, 5, 10, 15, 20, 25, 30])

fig = plt.figure(figsize=(12, 6))
ax  = fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(dates_np, s_arr, label='Wind speed', color='#898989', alpha=0.7)
plt.scatter(idt_events, np.array(sep_events), color='#950606', label='Max. events ('+str(len (sep_events))+')', zorder=5)
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