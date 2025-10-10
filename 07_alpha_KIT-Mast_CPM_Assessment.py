import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import os,glob,sys
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.dates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

"""
Code for producing analysis for the assessment of the CPM ensemble and members in the
location of the KIT mast, the reference observational dataset. It also conduncst the 
comparison of the spectrums between all the datasets, to provide insights on the 
variability. Besides the observed information from the KIT mast, and the CPM separate
and as an ensemble, the script uses data from NEWA at that point privided by Niel.

Author : Nathalia Correa-Sánchez
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

bd_in_ws   = "/Dati/Data/WS_CORDEXFPS/"
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"
bd_out_int = "/Dati/Data/WS_CORDEXFPS/Intermediates/"
bd_in_ese  = bd_in_ws + "Ensemble_Mean/wsa100m/"
bd_in_eth  = bd_in_ws + "ETH/wsa100m/"
bd_in_cmcc = bd_in_ws + "CMCC/wsa100m/"
bd_in_cnrm = bd_in_ws + "CNRM/wsa100m/"
bd_in_mast = "/Dati/Data/WindsMasts_TallTowers/WS_KIT-Mast_2000-01-01_2010-01-02_100m.csv"
bd_in_newa = "/Dati/Data/WindsMasts_TallTowers/NEWA_KIT/newa_ts_DE_for_Nathalia_2.nc"

########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

lat_kit  = 49.0925 ## Coordenada lat de KIT Mast
lon_kit  = 8.4259  ## Coordenada lon de KIT Mast
years    = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009] 

########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################

def style_axis(ax):
    """
    Function to set the format to the plots
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

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

    # Calculate raw spectral slope in high-frequency region
    # Using the same frequency range used to find cutoff frequency
    mask         = (freqs > 0.6) & (freqs < 0.9)
    log_freqs    = np.log(freqs[mask])
    log_psd      = np.log(smoothed_psd[mask])
    raw_slope, _ = np.polyfit(log_freqs, log_psd, 1)
    
    return freqs, psd, smoothed_psd, fc, corrected_psd, raw_slope


########################################################################################
### --------------------EXTRACTING TIME SERIES FROM CPM IN KIT-----------------------###
########################################################################################

filez_ws_eth  = sorted(glob.glob(f"{bd_in_eth}*.nc"))
ds_eth        = xr.open_mfdataset(filez_ws_eth, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
serie_ds_eth  = ds_eth ['wsa100m'].sel(lat=lat_kit, lon=lon_kit, method = 'nearest', drop=True).to_numpy()
time_eth      = ds_eth.time.values
print("## Extracted TS: ETH")
ds_eth.close() 

filez_ws_cmcc = sorted(glob.glob(f"{bd_in_cmcc}*.nc"))
ds_cmcc       = xr.open_mfdataset(filez_ws_cmcc, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
serie_ds_cmcc = ds_cmcc ['wsa100m'].sel(lat=lat_kit, lon=lon_kit, method = 'nearest', drop=True).to_numpy()
time_cmcc     = ds_cmcc.time.values
print("## Extracted TS: CMCC")
ds_cmcc.close()

filez_ws_cnrm = sorted(glob.glob(f"{bd_in_cnrm}*.nc"))
ds_cnrm       = xr.open_mfdataset(filez_ws_cnrm, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
serie_ds_cnrm = ds_cnrm ['wsa100m'].sel(lat=lat_kit, lon=lon_kit, method = 'nearest', drop=True).to_numpy()
time_cnrm     = ds_cnrm.time.values
print("## Extracted TS: CNRM")
ds_cnrm.close()

# filez_ws_ese  = sorted(glob.glob(f"{bd_in_ese}*.nc"))
# ds_ese        = xr.open_mfdataset(filez_ws_ese, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1000})
# serie_ds_ese  = ds_ese ['wsa100m'].sel(lat=lat_kit, lon=lon_kit, method = 'nearest', drop=True).to_numpy()
# time_ese      = ds_ese.time.values
# print("## Extracted TS: Ensemble")
# ds_ese.close()

df_eth  = pd.DataFrame(serie_ds_eth, index = time_eth, columns = ['ETH'])
df_cmcc = pd.DataFrame(serie_ds_cmcc, index = time_cmcc, columns = ['CMCC'])
df_cnrm = pd.DataFrame(serie_ds_cnrm, index = time_cnrm, columns = ['CNRM'])
# df_ese  = pd.DataFrame(serie_ds_ese, index = time_ese, columns = ['Ensemble'])

df_eth  = df_eth[df_eth.index.year != 2010]
df_cmcc = df_cmcc[df_cmcc.index.year != 2010]
df_cnrm = df_cnrm[df_cnrm.index.year != 2010]
# df_ese  = df_ese[df_ese.index.year != 2010]

df_eth  = df_eth[~df_eth.index.duplicated(keep='first')] 
df_cmcc = df_cmcc[~df_cmcc.index.duplicated(keep='first')]     
df_cnrm = df_cnrm[~df_cnrm.index.duplicated(keep='first')]  
# df_ese  = df_ese[~df_ese.index.duplicated(keep='first')]  

df_eth  = df_eth.dropna()   
df_cmcc = df_cmcc.dropna() 
df_cnrm = df_cnrm.dropna() 
# df_ese  = df_ese.dropna() 

########################################################################################
### --------------------------LOADING THE OBSERVED DATA SET--------------------------###
########################################################################################

df_mast          = pd.read_csv(bd_in_mast, sep=";")
df_mast["Datum"] =  pd.to_datetime(df_mast["Datum"], format= '%Y-%m-%d %H:%M:%S')
df_mast.set_index("Datum", inplace=True)
df_mast          = df_mast[df_mast.index.year != 2010]

########################################################################################
### --------------------ASESSING THE QUALITY OF THE OBSERVATIONS---------------------###
########################################################################################

##----------------------------NEGATIVE-WRONG DATA--------------------------------------##
df_wrong = df_mast[df_mast["Ws100m(m s-1)"]<0]
if len(df_wrong)>0:
    df_mast[df_mast["Ws100m(m s-1)"]<0] = 0  ## Given the range of the negative values the desicion is to turn them into 0. WMO: If absolute value < 0.2 m/s: Set to 0
    print("##### WARNING! NEGATIVE WS!! Percentage of negatives: "+str(round((len(df_wrong)/len(df_mast))*100, 4))+"%")
    
    df_wrong.index   = pd.to_datetime(df_wrong.index)
    x_values         = np.arange(1, len(df_wrong.index) + 1)
    formatted_labels = [dt.strftime('%y-%m-%d\n%H:%M') for dt in df_wrong.index]

    # fig, (ax1) = plt.subplots(1, 1,figsize=(9, 4))
    # style_axis(ax1)
    # ax1.plot(x_values, df_wrong["Ws100m(m s-1)"])
    # ax1.scatter(x_values, df_wrong["Ws100m(m s-1)"])
    # plt.title('Range of wrong records', weight='bold', pad =20)
    # plt.xlabel('Date [y-m-d h:m]')
    # plt.ylabel('Wrong records [m/s]')
    # plt.ylim(-0.2, 0.2)
    # plt.axhline(y=0, ls='--', color='lightgray')
    # plt.xticks(x_values, formatted_labels, rotation=0, fontsize=7)
    # plt.tight_layout()
    # plt.savefig(bd_out_fig+"KIT-Mast_NegativesWS.png", format='png', dpi=300, transparent=True)
    # plt.show()

elif len(df_wrong)==0:
    print("##### OK: Not negative WS")
    pass

##----------------------------MISSING DATA------------------------------------##

full_range        = pd.date_range(start='2000-01-01 00:10:00', end='2009-12-31 23:50:00', freq='10T')                                   # 10-minute frequency
df_mast_res       = df_mast.reindex(full_range, copy=True)                                                                              # Compeltting missing datetimes with nan
df_mast_res       = df_mast_res[~df_mast_res.index.duplicated(keep='first')]                                                            # Eliminating duplicated datetime indexes
df_count_nan      = df_mast_res["Ws100m(m s-1)"].isnull().groupby([df_mast_res.index.year]).sum().astype(int).reset_index(name='count') # Counting the amount of nans every year
df_count_all      = df_mast_res["Ws100m(m s-1)"].groupby(df_mast_res.index.year).size().astype(int).reset_index(name='count')           # Counting the total nomimal data every year
df_perc_nan       = 100-((df_count_nan / df_count_all)*100)
df_perc_nan       = df_perc_nan.drop(["index"], axis =1) 
df_perc_nan.index = years


fig, (ax1) = plt.subplots(1, 1,figsize=(6, 4))
style_axis(ax1)
bars       = ax1.bar(df_perc_nan.index, df_perc_nan['count'])
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
plt.title('Percentage of data per year', weight='bold', pad =20)
plt.xlabel('Year')
plt.ylabel('[%] Data')
plt.ylim(0, 100)
plt.xticks(df_perc_nan.index.values, rotation=45)
plt.tight_layout()
# plt.savefig(bd_out_fig+"KIT-Mast_PercData-Year.png", format='png', dpi=300, transparent=True)
plt.show()

########################################################################################
### ------------------------LOADING THE NEWA DATASET AT KIT--------------------------###
########################################################################################

ds_newa     = xr.open_dataset(bd_in_newa)  # Lat X  Lon = 49.101406 X 8.431763
newa_ws100m = ds_newa.WS.sel(height=100, method='nearest').values
newa_time   = ds_newa.time.values
df_newa     = pd.DataFrame( newa_ws100m, columns = ["NEWA"], index=newa_time)

##----------------------------NEWA MISSING DATA------------------------------------##

df_newa_nan            = df_newa["NEWA"].isnull().groupby([df_newa.index.year]).sum().astype(int).reset_index(name='count') # Counting the amount of nans every year
df_newa_all            = df_newa["NEWA"].groupby(df_newa.index.year).size().astype(int).reset_index(name='count')           # Counting the total nomimal data every year
df_newa_perc_nan       = 100-((df_newa_nan / df_newa_all)*100)
df_newa_perc_nan       = df_newa_perc_nan.drop(["index"], axis =1) 
df_newa_perc_nan.index = years

fig, (ax1) = plt.subplots(1, 1,figsize=(6, 4))
style_axis(ax1)
bars       = ax1.bar(df_newa_perc_nan.index, df_newa_perc_nan['count'])
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
plt.title('Percentage of data per year - NEWA', weight='bold', pad =20)
plt.xlabel('Year')
plt.ylabel('[%] Data')
plt.ylim(0, 100)
plt.xticks(df_newa_perc_nan.index.values, rotation=45)
plt.tight_layout()
plt.savefig(bd_out_fig+"NEWA-KIT-Mast_PercData-Year.png", format='png', dpi=300, transparent=True)
plt.show()

##----------------------------RESAMPLING DATA AT 1H------------------------------------##
df_newa     = df_newa.dropna()    
df_newa_h   = df_newa.resample('1H').mean()                  # To CPM temoporal resolution- It returns the nans


########################################################################################
### ---------------------------KIT DATAFRAME FORMATTING------------------------------###
########################################################################################

df_mast_res = df_mast_res.dropna()                               # To eliminate nan values for the hourly mean aggregation.
df_kit_h    = df_mast_res.resample('1H').mean()                  # To CPM temoporal resolution- It returns the nans
df_kit_h    = df_kit_h.rename(columns={'Ws100m(m s-1)': 'Observations'})  # It fills gaps with nan again.

df_all       = pd.concat([df_kit_h, df_eth, df_cmcc, df_cnrm, df_newa_h], axis=1)
df_all.index = pd.to_datetime(df_all.index)   # To be sure it is a DataFrme Index.
df_all.to_csv("/Dati/Data/WindsMasts_TallTowers/WS_all_datasets_KIT_2000-2009.csv", sep=";")
# df_all[df_all.isnull().any(axis=1) == True]
# df_all       = df_all.fillna(0)               # Ajustillo para el primer y el ultimo valor.

########################################################################################
### --------------ASESSING THE TIME SERIES OVER THE KIT MAST POINT-------------------###
########################################################################################
colours_l = ["black", "#edae49", "#d1495b", "#00798c",  "blue"]
# colours_l = ["black", "#edae49", "#d1495b", "#00798c", "#99a7bf", "red"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
style_axis(ax1)
bp = df_all.boxplot(column=list(df_all.columns), ax=ax1, patch_artist=True, boxprops=dict(alpha=0.7), 
     medianprops=dict(color="red"), flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))
ax1.set_title('Comparison of Wind Speed Distributions at the KIT mast', weight='bold', pad=10)
ax1.set_ylabel('Wind Speed [w/s]')
for i, box in enumerate(bp.patches):
    box.set_facecolor(colours_l[i])
ax1.grid(True, linestyle='--', alpha=0.7)

min_val = df_all.min().min()
max_val = df_all.max().max()
x_kde   = np.linspace(min_val, max_val, 200)
bins    = np.linspace(np.floor(min_val), np.ceil(max_val), 50)
style_axis(ax2)
for (column, color) in zip(df_all.columns, colours_l):
    if column == "Observations":
        ax2.hist(df_all[column], bins=bins, alpha=0.3, color=color, density=True, edgecolor=color, linewidth=2)
    else:
        ax2.hist(df_all[column], bins=bins, alpha=0.3, color=color, density=True)
    kernel = stats.gaussian_kde(df_all[column].dropna())
    y_kde  = kernel(x_kde)
    ax2.plot(x_kde, y_kde, color=color, linewidth=2, label=f'{column}',linestyle='-')
ax2.set_title('Wind Speed Distribution at the KIT mast', weight='bold', pad=15)
ax2.set_xlabel('Wind Speed [w/s]')
ax2.set_ylabel('Frequency Density')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.08) 
plt.savefig(bd_out_fig+"KIT-Mast_CPMs_HistoBoxplots.png", format='png', dpi=300, transparent=True)
plt.show()

gen_stats = df_all.describe()
print("\nEstadísticas descriptivas:")
print(gen_stats)

########################################################################################
### ------------------GETTING THE ANNUAL MAXIMA OF ALL TEH DATASETS------------------###
########################################################################################

df_AM     = df_all.groupby(df_all.index.year).max()

fig, (ax1) = plt.subplots(1, 1,figsize=(6, 4))
style_axis(ax1)
ax1.plot(years, df_AM["Observations"], color = colours_l[0], label= "Observations")
ax1.scatter(years, df_AM["Observations"], color = colours_l[0])
ax1.scatter(years, df_AM["ETH"], marker ="X", color = colours_l[1], label= "ETH")
ax1.scatter(years, df_AM["CMCC"], marker ="X", color = colours_l[2], label= "CMCC")
ax1.scatter(years, df_AM["CNRM"], marker ="X", color = colours_l[3], label= "CNRM")
ax1.plot(years, df_AM["Ensemble"], color = colours_l[4], label= "Ensemble")
ax1.scatter(years, df_AM["Ensemble"], color = colours_l[4])
ax1.plot(years, df_AM["NEWA"], color = colours_l[5], label= "NEWA")
ax1.scatter(years, df_AM["NEWA"], color = colours_l[5])
ax1.grid(True, linestyle='--', alpha=0.7)
plt.title('Annual Maxima comparison at the KIT mast', weight='bold', pad =20)
plt.xlabel('Years')
plt.ylabel('Wind Speed [m/s]')
plt.ylim(np.nanmin(df_AM.values)-1, np.nanmax(df_AM.values)+1)
plt.xticks(years, [str(i) for i in years], rotation=0, fontsize=7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(bd_out_fig+"KIT-Mast_CPMs_AnnualMax.png", format='png', dpi=300, transparent=True)
plt.show()

#######################################################################################
##--------------DERIVING & PLOTTING SPECTRA OBSERVED AND MODELLED DATASETS ----------##
#######################################################################################

s_arr_obs  = df_all['Observations'].values[~np.isnan(df_all['Observations'].values)]  # Eliminting nan values from each array
s_arr_eth  = df_all['ETH'].values[~np.isnan(df_all['ETH'].values)]                    # Eliminting nan values from each array
s_arr_cmcc = df_all['CMCC'].values[~np.isnan(df_all['CMCC'].values)]                  # Eliminting nan values from each array
s_arr_cnrm = df_all['CNRM'].values[~np.isnan(df_all['CNRM'].values)]                  # Eliminting nan values from each array
# s_arr_ese  = df_all['Ensemble'].values[~np.isnan(df_all['Ensemble'].values)]          # Eliminting nan values from each array
s_arr_newa = df_all['NEWA'].values[~np.isnan(df_all['NEWA'].values)]                  # Eliminting nan values from each array

freqs_obs, psd_obs, smoothed_psd_obs, fc_obs, corrected_psd_obs, raw_slope_obs        = obtain_spectra_params_corr (s_arr_obs, window_size = 130)
freqs_eth, psd_eth, smoothed_psd_eth, fc_eth, corrected_psd_eth, raw_slope_eth        = obtain_spectra_params_corr (s_arr_eth, window_size = 130)
freqs_cmcc, psd_cmcc, smoothed_psd_cmcc, fc_cmcc, corrected_psd_cmcc, raw_slope_cmcc  = obtain_spectra_params_corr (s_arr_cmcc, window_size = 130)
freqs_cnrm, psd_cnrm, smoothed_psd_cnrm, fc_cnrm, corrected_psd_cnrm, raw_slope_cnrm  = obtain_spectra_params_corr (s_arr_cnrm, window_size = 130)
# freqs_ese, psd_ese, smoothed_psd_ese, fc_ese, corrected_psd_ese, raw_slope_ese      = obtain_spectra_params_corr (s_arr_ese, window_size = 130)
freqs_newa, psd_newa, smoothed_psd_newa, fc_newa, corrected_psd_newa, raw_slope_newa  = obtain_spectra_params_corr (s_arr_newa, window_size = 130)


###-----------------------------ALL SPECTRUMS-------------------------###

Fig = plt.figure(figsize=(7, 5))
ax  = Fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

## Observations 
line2_obs, = ax.plot(np.log(freqs_obs), np.log(smoothed_psd_obs),color=colours_l[0], linewidth=2, alpha=0.6, linestyle="--")
# vline_obs  = ax.axvline(np.log(fc_obs), color=colours_l[0], linestyle='dotted', label=f"$f_c$ = {fc_obs:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_obs, = ax.plot(np.log(freqs_obs), np.log(corrected_psd_obs), linewidth=1, color=colours_l[0])  ## Esta es la correccion

## ETH 
line2_eth, = ax.plot(np.log(freqs_eth), np.log(smoothed_psd_eth),color=colours_l[1], linewidth=2, alpha=0.6, linestyle="--")
# vline_eth  = ax.axvline(np.log(fc_eth), color=colours_l[1], linestyle='dotted', label=f"$f_c$ = {fc_eth:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_eth, = ax.plot(np.log(freqs_eth), np.log(corrected_psd_eth), linewidth=1, color=colours_l[1])  ## Esta es la correccion

## CMCC 
line2_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(smoothed_psd_cmcc),color=colours_l[2], linewidth=2, alpha=0.6, linestyle="--")
# vline_cmcc  = ax.axvline(np.log(fc_cmcc), color=colours_l[2], linestyle='dotted', label=f"$f_c$ = {fc_cmcc:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(corrected_psd_cmcc), linewidth=1, color=colours_l[2])  ## Esta es la correccion

## CNRM 
line2_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(smoothed_psd_cnrm),color=colours_l[3], linewidth=2, alpha=0.6, linestyle="--")
# vline_cnrm  = ax.axvline(np.log(fc_cnrm), color=colours_l[3], linestyle='dotted', label=f"$f_c$ = {fc_cnrm:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(corrected_psd_cnrm), linewidth=1, color=colours_l[3])  ## Esta es la correccion

## Ensemble
# line2_ese,  = ax.plot(np.log(freqs_ese), np.log(smoothed_psd_ese),color=colours_l[4], linewidth=2, alpha=0.6, linestyle="--")
# vline_ese   = ax.axvline(np.log(fc_ese), color=colours_l[4], linestyle='dotted', label=f"$f_c$ = {fc_ese:.2f} d$^{{-1}}$")
# line3_ese,  = ax.plot(np.log(freqs_ese), np.log(corrected_psd_ese), linewidth=1, color=colours_l[4])  ## Esta es la correccion

## NEWA 
line2_newa, = ax.plot(np.log(freqs_newa), np.log(smoothed_psd_newa),color=colours_l[5], linewidth=2, alpha=0.6, linestyle="--")
# vline_newa  = ax.axvline(np.log(fc_newa), color=colours_l[5], linestyle='dotted', label=f"$f_c$ = {fc_newa:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_newa, = ax.plot(np.log(freqs_newa), np.log(corrected_psd_newa), linewidth=1, color=colours_l[5])


ax.set_title(f"Power spectrum comparison at the KIT mast", weight='bold', pad =20, fontsize=14)
plt.xlabel("(ln(days$^{-1}$))", fontsize=10)
plt.ylabel("ln(S(f))", fontsize=12)

custom_line_eth  = Line2D([0], [0], color=colours_l[0], linestyle='dotted', linewidth=0.9)
custom_line_cmcc = Line2D([0], [0], color=colours_l[1], linestyle='dotted', linewidth=0.9)
custom_line_cnrm = Line2D([0], [0], color=colours_l[2], linestyle='dotted', linewidth=0.9)
# ax.legend(handles=[custom_line_eth, custom_line_cmcc, custom_line_cnrm], 
#             labels=[f"$f_c$:{fc_eth:.2f} d$^{{-1}}$", f"$f_c$:{fc_cmcc:.2f} d$^{{-1}}$", f"$f_c$:{fc_cnrm:.2f} d$^{{-1}}$"], 
#             fontsize=7, loc='lower left')
ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in')
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)


legend = ax.legend([line2_obs, line2_eth, line2_cmcc, line2_cnrm, line2_ese, line2_newa], 
                   ["PSD smoothed Observations", "PSD smoothed ETH", "PSD smoothed CMCC",  "PSD smoothed CNRM", "PSD smoothed Esemble", "PSD smoothed NEWA"],
                   bbox_to_anchor=(0.96, -0.21), loc='center right', ncol=3, fontsize=8)
# ax.add_artist(legend)
# ax.legend(handles=[vline_obs, vline_eth, vline_cmcc, vline_cnrm, vline_ese, vline_newa], fontsize=8) 
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.savefig(bd_out_fig+"KIT-Mast_CPMs_PowerSpectrum.png", format='png', dpi=300, transparent=True)
plt.show()

###-----------------------------CPM ENSEMBLE MEMBERS SPECTRUMS-------------------------###

Fig = plt.figure(figsize=(7, 5))
ax  = Fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

## Observations 
line2_obs, = ax.plot(np.log(freqs_obs), np.log(smoothed_psd_obs),color=colours_l[0], linewidth=2, alpha=0.6, linestyle="--")
# vline_obs  = ax.axvline(np.log(fc_obs), color=colours_l[0], linestyle='dotted', label=f"$f_c$ = {fc_obs:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_obs, = ax.plot(np.log(freqs_obs), np.log(corrected_psd_obs), linewidth=1, color=colours_l[0])  ## Esta es la correccion

## ETH 
line2_eth, = ax.plot(np.log(freqs_eth), np.log(smoothed_psd_eth),color=colours_l[1], linewidth=2, alpha=0.6, linestyle="--")
# vline_eth  = ax.axvline(np.log(fc_eth), color=colours_l[1], linestyle='dotted', label=f"$f_c$ = {fc_eth:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_eth, = ax.plot(np.log(freqs_eth), np.log(corrected_psd_eth), linewidth=1, color=colours_l[1])  ## Esta es la correccion

## CMCC 
line2_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(smoothed_psd_cmcc),color=colours_l[2], linewidth=2, alpha=0.6, linestyle="--")
# vline_cmcc  = ax.axvline(np.log(fc_cmcc), color=colours_l[2], linestyle='dotted', label=f"$f_c$ = {fc_cmcc:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(corrected_psd_cmcc), linewidth=1, color=colours_l[2])  ## Esta es la correccion

## CNRM 
line2_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(smoothed_psd_cnrm),color=colours_l[3], linewidth=2, alpha=0.6, linestyle="--")
# vline_cnrm  = ax.axvline(np.log(fc_cnrm), color=colours_l[3], linestyle='dotted', label=f"$f_c$ = {fc_cnrm:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(corrected_psd_cnrm), linewidth=1, color=colours_l[3])  ## Esta es la correccion


ax.set_title(f"Power spectrum comparison at the KIT mast \n single members", weight='bold', pad =15, fontsize=14)
plt.xlabel("(ln(days$^{-1}$))", fontsize=10)
plt.ylabel("ln(S(f))", fontsize=12)

custom_line_eth  = Line2D([0], [0], color=colours_l[0], linestyle='dotted', linewidth=0.9)
custom_line_cmcc = Line2D([0], [0], color=colours_l[1], linestyle='dotted', linewidth=0.9)
custom_line_cnrm = Line2D([0], [0], color=colours_l[2], linestyle='dotted', linewidth=0.9)
# ax.legend(handles=[custom_line_eth, custom_line_cmcc, custom_line_cnrm], 
#             labels=[f"$f_c$:{fc_eth:.2f} d$^{{-1}}$", f"$f_c$:{fc_cmcc:.2f} d$^{{-1}}$", f"$f_c$:{fc_cnrm:.2f} d$^{{-1}}$"], 
#             fontsize=7, loc='lower left')
ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in')
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)


legend = ax.legend([line2_obs, line2_eth, line2_cmcc, line2_cnrm, line3_obs, line3_eth, line3_cmcc, line3_cnrm], 
                   ["PSD smoothed Obs", "PSD smoothed ETH", "PSD smoothed CMCC",  "PSD smoothed CNRM", 
                    "PSD corrected Obs", "PSD corrected ETH", "PSD corrected CMCC",  "PSD corrected CNRM"],
                   bbox_to_anchor=(1.01, -0.21), loc='center right', ncol=4, fontsize=8)
# ax.add_artist(legend)
# ax.legend(handles=[vline_obs, vline_eth, vline_cmcc, vline_cnrm, vline_ese, vline_newa], fontsize=8) 
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.savefig(bd_out_fig+"KIT-Mast_CPMsMenmbers_PowerSpectrum.png", format='png', dpi=300, transparent=True)
plt.show()

###-----------------------------ENSEMBLE AND NEWA SPECTRUMS-------------------------###

Fig = plt.figure(figsize=(7, 5))
ax  = Fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

## Observations 
line2_obs, = ax.plot(np.log(freqs_obs), np.log(smoothed_psd_obs),color=colours_l[0], linewidth=2, alpha=0.3, linestyle="--")
# vline_obs  = ax.axvline(np.log(fc_obs), color=colours_l[0], linestyle='dotted', label=f"$f_c$ = {fc_obs:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_obs, = ax.plot(np.log(freqs_obs), np.log(corrected_psd_obs), linewidth=1.5, color=colours_l[0])  ## Esta es la correccion

# # # Ensemble
# # line2_ese,  = ax.plot(np.log(freqs_ese), np.log(smoothed_psd_ese),color=colours_l[4], linewidth=2, alpha=0.3, linestyle="--")
# # vline_ese   = ax.axvline(np.log(fc_ese), color=colours_l[4], linestyle='dotted', label=f"$f_c$ = {fc_ese:.2f} d$^{{-1}}$")
# # line3_ese,  = ax.plot(np.log(freqs_ese), np.log(corrected_psd_ese), linewidth=1.5, color=colours_l[4])  ## Esta es la correccion

## NEWA 
line2_newa, = ax.plot(np.log(freqs_newa), np.log(smoothed_psd_newa),color=colours_l[4], linewidth=2, alpha=0.3, linestyle="--")
# vline_newa  = ax.axvline(np.log(fc_newa), color=colours_l[4], linestyle='dotted', label=f"$f_c$ = {fc_newa:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_newa, = ax.plot(np.log(freqs_newa), np.log(corrected_psd_newa), linewidth=1.5, color=colours_l[4])


ax.set_title(f"Power spectrum comparison at the KIT mast\n ensemble and NEWA", weight='bold', pad =15, fontsize=14)
plt.xlabel("(ln(days$^{-1}$))", fontsize=10)
plt.ylabel("ln(S(f))", fontsize=12)

custom_line_eth  = Line2D([0], [0], color=colours_l[0], linestyle='dotted', linewidth=0.9)
custom_line_cmcc = Line2D([0], [0], color=colours_l[1], linestyle='dotted', linewidth=0.9)
custom_line_cnrm = Line2D([0], [0], color=colours_l[2], linestyle='dotted', linewidth=0.9)
# ax.legend(handles=[custom_line_eth, custom_line_cmcc, custom_line_cnrm], 
#             labels=[f"$f_c$:{fc_eth:.2f} d$^{{-1}}$", f"$f_c$:{fc_cmcc:.2f} d$^{{-1}}$", f"$f_c$:{fc_cnrm:.2f} d$^{{-1}}$"], 
#             fontsize=7, loc='lower left')
ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in')
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)


legend = ax.legend([line2_obs, line2_newa, line3_obs,  line3_newa, ], 
                   ["PSD smoothed Obs", "PSD smoothed Ensemble", "PSD smoothed NEWA",   
                    "PSD corrected Obs", "PSD corrected Ensemble", "PSD corrected NEWA"],
                   bbox_to_anchor=(0.98, -0.21), loc='center right', ncol=2, fontsize=8)
# ax.add_artist(legend)
# ax.legend(handles=[vline_obs, vline_eth, vline_cmcc, vline_cnrm, vline_ese, vline_newa], fontsize=8) 
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.savefig(bd_out_fig+"KIT-Mast_CPMsObs-NEWA_PowerSpectrum.png", format='png', dpi=300, transparent=True)
plt.show()


##########################################################################################################################
###-----------------------------VISUALIZING POWER SPECTRA AT KIT POINT FOR COMPARISON-----------------------------------##
##########################################################################################################################

theoretical_slope = -5/3  # ≈ -1.67
# Format for annotation
slope_text1 = (f"Spectral slopes:\n"
              f"Theoretical: -5/3 \u2248 {theoretical_slope:.2f}\n"
              f"Obs: {raw_slope_obs:.2f}\n"
              f"ETH: {raw_slope_eth:.2f}\n"
              f"CMCC: {raw_slope_cmcc:.2f}\n"
              f"CNRM: {raw_slope_cnrm:.2f}")

slope_text2 = (f"Spectral slopes:\n"
              f"Theoretical: -5/3 \u2248 {theoretical_slope:.2f}\n"
              f"Obs: {raw_slope_obs:.2f}\n"
              f"NEWA: {raw_slope_newa:.2f}")

# Define the range for the theoretical line to be plotted and generating the points for that reference line
x_start = np.log(1.0)    # log(10^0) = 0, corresponds to 1 day^-1
x_end   = np.log(100.0)  # log(10^2) = 4.6, corresponds to 14 min  ~10 min
x_ref = np.array([x_start, x_end])

# Calculate y values for the reference line
# y = mx + b where m = -5/3 and we need to set b appropriately
# Offset is chosen to make the line visible without overlapping too much with data
y_offset = 1.5  # Adjust as needed for visibility

y_ref_mid_first = np.log(smoothed_psd_obs[np.argmin(np.abs(freqs_obs - 1.0))])
y_start_first   = y_ref_mid_first + y_offset
b_first         = y_start_first - theoretical_slope * x_start
y_ref_first     = theoretical_slope * x_ref + b_first


Fig = plt.figure(figsize=(10, 5))
ax  = Fig.add_subplot(1, 2, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## Observations 
line2_obs, = ax.plot(np.log(freqs_obs), np.log(smoothed_psd_obs),color=colours_l[0], linewidth=2, alpha=0.4, linestyle= ":")
# vline_obs  = ax.axvline(np.log(fc_obs), color=colours_l[0], linestyle='dotted', label=f"$f_c$ = {fc_obs:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_obs, = ax.plot(np.log(freqs_obs), np.log(corrected_psd_obs), linewidth=1.7, color=colours_l[0])  ## Esta es la correccion
## ETH 
line2_eth, = ax.plot(np.log(freqs_eth), np.log(smoothed_psd_eth),color=colours_l[1], linewidth=2, alpha=0.4, linestyle= ":")
# vline_eth  = ax.axvline(np.log(fc_eth), color=colours_l[1], linestyle='dotted', label=f"$f_c$ = {fc_eth:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_eth, = ax.plot(np.log(freqs_eth), np.log(corrected_psd_eth), linewidth=1.7, color=colours_l[1])  ## Esta es la correccion
## CMCC 
line2_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(smoothed_psd_cmcc),color=colours_l[2], linewidth=2, alpha=0.4, linestyle= ":")
# vline_cmcc  = ax.axvline(np.log(fc_cmcc), color=colours_l[2], linestyle='dotted', label=f"$f_c$ = {fc_cmcc:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_cmcc, = ax.plot(np.log(freqs_cmcc), np.log(corrected_psd_cmcc), linewidth=1.7, color=colours_l[2])  ## Esta es la correccion
## CNRM 
line2_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(smoothed_psd_cnrm),color=colours_l[3], linewidth=2, alpha=0.4, linestyle= ":")
# vline_cnrm  = ax.axvline(np.log(fc_cnrm), color=colours_l[3], linestyle='dotted', label=f"$f_c$ = {fc_cnrm:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_cnrm, = ax.plot(np.log(freqs_cnrm), np.log(corrected_psd_cnrm), linewidth=1.7, color=colours_l[3])  ## Esta es la correccion
# ax.text(1.01, 1.01, slope_text1, transform=ax.transAxes, ha='right', va='top', fontsize=8,  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
line_theory = ax.plot(x_ref, y_ref_first, 'k--', linewidth=1.5, alpha=0.7)
ax.set_title(f"a) Power spectrum observations \n and single CPM members", weight='bold', pad =14, fontsize=12)
ax.set_xlabel("(ln(days$^{-1}$))", fontsize=10)
ax.set_ylabel("ln(S(f))", fontsize=12)
ax.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in')
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)
legend1 = ax.legend([line2_obs, line2_eth, line2_cmcc, line2_cnrm, line3_obs, line3_eth, line3_cmcc, line3_cnrm], 
                   ["Obs (raw)", "ETH (raw)", "CMCC (raw)",  "CNRM (raw)", 
                    "Obs (corr.)", "ETH (corr.)", "CMCC  (corr.)",  "CNRM  (corr.)"],
                   bbox_to_anchor=(1.09, -0.21), loc='center right', ncol=4, fontsize=8)

ax1 = Fig.add_subplot(1, 2, 2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
## Observations 
line2_obs, = ax1.plot(np.log(freqs_obs), np.log(smoothed_psd_obs),color=colours_l[0], linewidth=2, alpha=0.4, linestyle= ":")
# vline_obs  = ax1.axvline(np.log(fc_obs), color=colours_l[0], linestyle='dotted', label=f"$f_c$ = {fc_obs:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_obs, = ax1.plot(np.log(freqs_obs), np.log(corrected_psd_obs), linewidth=1.7, color=colours_l[0])  ## Esta es la correccion
## NEWA 
line2_newa, = ax1.plot(np.log(freqs_newa), np.log(smoothed_psd_newa),color=colours_l[4], linewidth=2, alpha=0.4, linestyle= ":")
# vline_newa  = ax1.axvline(np.log(fc_newa), color=colours_l[4], linestyle='dotted', label=f"$f_c$ = {fc_newa:.2f} d$^{{-1}}$") ## Este es el cut-off frequency
line3_newa, = ax1.plot(np.log(freqs_newa), np.log(corrected_psd_newa), linewidth=1.7, color=colours_l[4])
# ax1.text(1.01, 1.01, slope_text2, transform=ax1.transAxes, ha='right', va='top', fontsize=8,  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
line_theory = ax1.plot(x_ref, y_ref_first, 'k--', linewidth=1.5, alpha=0.7)
ax1.set_title(f"b) Power spectrum observations \n  and NEWA", weight='bold', pad =14, fontsize=12)
ax1.set_xlabel("(ln(days$^{-1}$))", fontsize=10)
ax1.set_ylabel("ln(S(f))", fontsize=12)
ax1.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9)))
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', direction='in')
ax1.grid(True, which='major', linestyle='-', alpha=0.7)
ax1.grid(True, which='minor', linestyle=':', alpha=0.4)
legend2 = ax1.legend([line2_obs, line2_newa, line3_obs,  line3_newa, ], 
                   ["Obs (raw)",  "NEWA (raw)",   
                    "Obs (corr.)", "NEWA (corr.)"],
                    bbox_to_anchor=(0.8, -0.21), loc='center right', ncol=2, fontsize=8)

# custom_line_eth  = Line2D([0], [0], color=colours_l[0], linestyle='dotted', linewidth=0.9)
# custom_line_cmcc = Line2D([0], [0], color=colours_l[1], linestyle='dotted', linewidth=0.9)
# custom_line_cnrm = Line2D([0], [0], color=colours_l[2], linestyle='dotted', linewidth=0.9)              

plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right =0.97, bottom = 0.20) 
plt.savefig(bd_out_fig+"KIT-Mast_SingleCPMs-Obs-NEWA_Comparison.png", format='png', dpi=300, transparent=True)
plt.show()

