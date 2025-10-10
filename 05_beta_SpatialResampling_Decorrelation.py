
import xarray as xr
import numpy as np
import pandas as pd
import glob, os
from numba import njit
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import dask as da
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, AutoMinorLocator, MaxNLocator
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
Code to compute a spatial resampling of the three CPMs, create ensemble mean,
and calculate temporal decorrelation. Finally it produces a map of the results.

Author : Nathalia Correa-Sánchez
"""

#############################################################################
##-------------------------DEFINING IMPORTANT PATHS------------------------##
#############################################################################
bd_in_ws   = "/Dati/Data/WS_CORDEXFPS/"
bd_out_ese = "/Dati/Data/WS_CORDEXFPS/Ensemble_Mean/wsa100m_SpatRes_Temporal/"
bd_in_eth  = bd_in_ws + "ETH/wsa100m_crop/"
bd_in_cmcc = bd_in_ws + "CMCC/wsa100m_crop/"
bd_in_cnrm = bd_in_ws + "CNRM/wsa100m_crop/"
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"

#############################################################################
##------------------------DEFINING INPUT PARAMETERS------------------------##
#############################################################################
resolution = 1
chunk_size = {'time': 100, 'lat': -1, 'lon': 50}

#############################################################################
##-----------------------DEFINING RELEVANT FUNCTIONS-----------------------##
#############################################################################

def load_ds(bd_in):
    """
    Function to load the CPMs dataset, by loading multiple related files. 
    INPUTS:
    - bd_in : path to the dataset directory
    OUTPUTS:
    - ds_org : xarray dataset with the model information
    """
    files    = sorted(glob.glob(f"{bd_in}*.nc"))
    ds_org   = xr.open_mfdataset(files, combine='nested', concat_dim='time', 
                                parallel=True, chunks={'time': 1000})
    return ds_org

def create_target_grid(ds, resolution):
    """
    Create target grid based on specified resolution
    """
    # Create new grid
    new_lats = np.arange(np.ceil(ds.lat.min().values / resolution) * resolution,
                        np.floor(ds.lat.max().values / resolution) * resolution + resolution,
                        resolution)
    new_lons = np.arange(np.ceil(ds.lon.min().values / resolution) * resolution,
                        np.floor(ds.lon.max().values / resolution) * resolution + resolution,
                        resolution)
    return new_lats, new_lons

def process_multiple_cpms_to_ensemble(bd_paths, target_lats, target_lons):
    """
    Process multiple CPMs and create ensemble mean at target grid points
    
    Parameters:
    -----------
    bd_paths : dict
        Dictionary with paths to each CPM dataset
    target_lats : array
        Target latitudes
    target_lons : array
        Target longitudes
    
    Returns:
    --------
    xarray.Dataset
        Ensemble mean dataset with time series at target grid points
    """
    print("Loading and processing CPM datasets...")
    
    # Load all three CPMs
    print("Loading ETH...")
    ds_eth = load_ds(bd_paths['ETH'])
    print("Loading CMCC...")
    ds_cmcc = load_ds(bd_paths['CMCC'])
    print("Loading CNRM...")
    ds_cnrm = load_ds(bd_paths['CNRM'])
    
    # Interpolate all datasets to target grid
    print("Interpolating to target grid...")
    ds_eth_target = ds_eth.sel(lat=target_lats, lon=target_lons, method='nearest')
    ds_cmcc_target = ds_cmcc.sel(lat=target_lats, lon=target_lons, method='nearest')
    ds_cnrm_target = ds_cnrm.sel(lat=target_lats, lon=target_lons, method='nearest')
    
    # Ensure all datasets have the same time dimension
    print("Aligning time dimensions...")
    common_times = np.intersect1d(np.intersect1d(ds_eth_target.time.values, 
                                                ds_cmcc_target.time.values), 
                                 ds_cnrm_target.time.values)
    
    ds_eth_aligned = ds_eth_target.sel(time=common_times)
    ds_cmcc_aligned = ds_cmcc_target.sel(time=common_times)
    ds_cnrm_aligned = ds_cnrm_target.sel(time=common_times)
    
    # Create ensemble mean
    print("Creating ensemble mean...")
    ensemble_data = (ds_eth_aligned.wsa100m.values + 
                    ds_cmcc_aligned.wsa100m.values + 
                    ds_cnrm_aligned.wsa100m.values) / 3.0
    
    # Create ensemble dataset
    ensemble_ds = xr.Dataset(
        {'wsa100m': (('time', 'lat', 'lon'), ensemble_data)},
        coords={
            'time': ds_eth_aligned.time,
            'lat': ds_eth_aligned.lat,
            'lon': ds_eth_aligned.lon
        }
    )
    
    # Clean up memory
    ds_eth.close()
    ds_cmcc.close()
    ds_cnrm.close()
    
    print("Ensemble dataset created successfully!")
    return ensemble_ds

def fill_nans_with_neighborhood_mean(array):
    """Fill NaN values with neighborhood mean"""
    array_filled = array.copy()
    for t in range(array.shape[0]):
        for i in range(array.shape[1]):
            for j in range(array.shape[2]):
                if np.isnan(array[t,i,j]):
                    # Extract neighborhood
                    neighborhood = array[t, 
                        max(0,i-1):min(array.shape[1],i+2), 
                        max(0,j-1):min(array.shape[2],j+2)
                    ]
                    # Calculate mean of valid neighbors
                    valid_neighbors = neighborhood[~np.isnan(neighborhood)]
                    if len(valid_neighbors) > 0:
                        array_filled[t,i,j] = np.mean(valid_neighbors)
    return array_filled

@njit
def crosscorr(a, b, lag):
    """Numba-accelerated cross-correlation function"""
    return np.corrcoef(a[:-lag], b[lag:])[0, 1]

def calculate_temporal_decorrelation(pixel_series, lags):
    """
    Calculate temporal decorrelation for a single pixel time series
    """
    # Compute cross-correlations
    rs = np.array([crosscorr(pixel_series, pixel_series, lag) for lag in lags])
    
    # Cubic spline interpolation
    spline    = UnivariateSpline(lags, rs, s=0.01)
    rs_spline = spline(lags)
    
    # Compute integral (temporal decorrelation)
    return np.trapz(rs_spline, lags)

def style_axis(ax):
    """
    Function to set the format to the plots
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

#############################################################################
##----------------PROCESSING CPMS TO CREATE ENSEMBLE MEAN------------------##
#############################################################################

# Define CPM paths
cpm_paths = {
    'ETH': bd_in_eth,
    'CMCC': bd_in_cmcc,
    'CNRM': bd_in_cnrm
}

# Get target grid from first available dataset (ETH)
print("Determining target grid...")
with xr.open_dataset(sorted(glob.glob(f"{bd_in_eth}*.nc"))[0]) as ds:
    target_lats, target_lons = create_target_grid(ds, resolution)

print(f"Target grid: {len(target_lats)} lats x {len(target_lons)} lons")

# Process CPMs and create ensemble mean
result_ds = process_multiple_cpms_to_ensemble(cpm_paths, target_lats, target_lons)

print("Processing completed!")

#############################################################################
##------------CORRECTING SPURIOUS NANs IN THE RESULTING ARRAY--------------##
#############################################################################

wsa100m_array = result_ds.wsa100m.values
lat_array     = result_ds.lat.values
lon_array     = result_ds.lon.values

print(f"Original NaNs: {np.isnan(wsa100m_array).sum()}")
wsa100m_array_filled = fill_nans_with_neighborhood_mean(wsa100m_array)
print(f"Remaining NaNs after filling: {np.isnan(wsa100m_array_filled).sum()}")

#############################################################################
##-------------------COMPUTING TEMPORAL DECORRELATION----------------------##
#############################################################################

tau_dec       = np.zeros((wsa100m_array_filled.shape[1], wsa100m_array_filled.shape[2]), dtype=np.float32)
  
# Definir rango de lags
lags = np.arange(1, 201, 1)

# Iterar sobre latitudes y longitudes
for lat_idx in range(wsa100m_array_filled.shape[1]):
    for lon_idx in range(wsa100m_array_filled.shape[2]):
        # Extraer la serie temporal para un píxel específico
        pixel_series = wsa100m_array_filled[:, lat_idx, lon_idx]
        
        # Calcular el tiempo de decorrelación
        tau_dec[lat_idx, lon_idx] = calculate_temporal_decorrelation(pixel_series, lags)

    print(lat_idx)

print("MinDec:"+str(tau_dec.min()))
print("MaxDec:"+str(tau_dec.max()))

tau_dec_meanlon = np.mean(tau_dec, axis=1)
#############################################################################
##----------------------MAPPING DECORRELATION VALUES-----------------------##
#############################################################################

# Create figure with improved proportions
fig = plt.figure(figsize=(15, 6))
gs  = GridSpec(1, 2, width_ratios=[1, 3.5], height_ratios=[1], figure=fig)

ax1 = fig.add_subplot(gs[0])
style_axis(ax1)
ax1.plot(tau_dec_meanlon, lat_array, '-k', linewidth=2.5)
ax1.set_ylabel('Latitude [°N]', fontsize=13.5)
ax1.set_xlabel('Mean Decorrelation [h]', fontsize=13.5)
ax1.set_title("a) Mean decorrelation profile", fontsize=14, fontweight="bold")
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim([lat_array.min(), lat_array.max()])
ax1.set_xlim([tau_dec_meanlon.min() - 2, tau_dec_meanlon.max() + 2])
ax1.tick_params(labelsize=13)

# Subplot 2: Spatial distribution map 
ax2                = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
lon_mesh, lat_mesh = np.meshgrid(lon_array, lat_array)
scatter            = ax2.scatter( lon_mesh.flatten(), lat_mesh.flatten(), c=tau_dec.flatten(), transform=ccrs.PlateCarree(), cmap="viridis", s=100 )
ax2.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black", linewidth=0.8)
ax2.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")
gl = ax2.gridlines(draw_labels=True, alpha=0.3, linestyle="--")
gl.left_labels  = False
gl.top_labels   = False
gl.right_labels = True
gl.bottom_labels = True
y_ticks_ax1 = ax1.get_yticks()
gl.ylocator = ticker.FixedLocator(y_ticks_ax1)
gl.xlocator = MaxNLocator(nbins=8)
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}
ax2.set_title("b) Spatial distribution of decorrelation time", fontsize=14, fontweight="bold")
ax2.set_extent([lon_array.min(), lon_array.max(), lat_array.min(), lat_array.max()],  crs=ccrs.PlateCarree())


cbar = plt.colorbar(scatter, ax=ax2, orientation="vertical", shrink=0.8, pad=0.06, aspect=30)
cbar.set_label("Decorrelation [h]", fontsize=13)
cbar.ax.tick_params(labelsize=12)

# Asegurar igualdad de alturas
ax2.set_aspect('auto') 
plt.subplots_adjust(wspace=0.05) 
plt.tight_layout()

plt.subplots_adjust(wspace=0.08, left=0.08, right=0.95, top=0.92, bottom=0.12)
# plt.savefig(bd_out_fig + "MapDecorrelation_MultiCPM.png", format='png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
