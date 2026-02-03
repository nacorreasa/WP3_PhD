 # %%
import xarray as xr
import numpy as np
import pandas as pd
import glob, os
# from numba import njit
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import dask as da
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, AutoMinorLocator, MaxNLocator
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path


"""
Code to compute a spatial resampling of the three CPMs, create ensemble mean,
and calculate temporal decorrelation. Finally it produces a map of the results.

Author : Nathalia Correa-Sánchez
"""
 # %%
#############################################################################
##-------------------------DEFINING IMPORTANT PATHS------------------------##
#############################################################################
STORAGE        = Path("/mnt/smb").as_posix()
bd_in_ws       = STORAGE + "/Data/WS_CORDEXFPS/"
bd_out_fig     = "/home/nathalia/Outputs/Plots/WP3_development/" # Por ahora en la maquina virtual de procesamiento
bd_out_ese     = STORAGE + "Data/WS_CORDEXFPS/Ensemble_Mean/wsa100m_SpatRes_Temporal/"
bd_in_eth      = bd_in_ws + "ETH/wsa100m_crop/"
bd_in_cmcc     = bd_in_ws + "CMCC/wsa100m_crop/"
bd_in_cnrm     = bd_in_ws + "CNRM/wsa100m_crop/"

 # %%
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

# # @njit
def crosscorr(a, b, lag):
    """Numba-accelerated cross-correlation function

    Calculate the Pearson correlation coefficient matrix (already normalised).
    
    """
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
    return np.trapezoid(rs_spline, lags)

def computing_decorrelation_per_array(wsa100m_array):
    """
    Function to compute the decorrelation time in a 3D numpy array grid.
    
    INPUTS:
     - wsa100m_array: 3D numpy array grid the time series in a grid
    """
    tau_dec = np.zeros((wsa100m_array.shape[1], wsa100m_array.shape[2]), dtype=np.float32)
    
    # Definir rango de lags
    lags = np.arange(1, 201, 1)

    # Iterar sobre latitudes y longitudes
    for lat_idx in range(wsa100m_array.shape[1]):
        for lon_idx in range(wsa100m_array.shape[2]):
            # Extraer la serie temporal para un píxel específico
            pixel_series = wsa100m_array[:, lat_idx, lon_idx]
            
            # Calcular el tiempo de decorrelación
            tau_dec[lat_idx, lon_idx] = calculate_temporal_decorrelation(pixel_series, lags)
        print(lat_idx)
   
    tau_dec_meanlon = np.mean(tau_dec, axis=1)

    return tau_dec, tau_dec_meanlon

def style_axis(ax):
    """
    Function to set the format to the plots
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
# %%
##########################################################################################
###-----------------------------LECTRURA DE ARCHIVOS-----------------------------------###
##########################################################################################

ds_eth  = load_ds(bd_in_eth)
ds_cnrm = load_ds(bd_in_cnrm)
ds_cmcc = load_ds(bd_in_cmcc)

# %%
##########################################################################################
###------------AJUSTE A LOS DS PARA QUE TENGAN EL MIMO RANGO TEMPORAL------------------###
##########################################################################################

# Asegurar que todos los datasets comiencen desde 2000-01-01 01:00:00
ds_eth  = ds_eth.sel(time=slice('2000-01-01 01:00:00', None))
ds_cmcc = ds_cmcc.sel(time=slice('2000-01-01 01:00:00', None))

# Asegurar que todos los datasets terminen el 2009-12-31 23:00:00
ds_cnrm = ds_cnrm.sel(time=slice(None, '2009-12-31 23:00:00'))

# %%
################################################################################
###----------------CREATING THE TARGET GRID IN EACH CPM----------------------###
################################################################################
# Get target grid from first available dataset (ETH)
print("Determining target grid...")
with xr.open_dataset(sorted(glob.glob(f"{bd_in_eth}*.nc"))[0]) as ds:
    target_lats, target_lons = create_target_grid(ds, resolution)

print(f"Target grid: {len(target_lats)} lats x {len(target_lons)} lons")

# Interpolate all datasets to target grid
print("Interpolating to target grid...")
ds_eth_target  = ds_eth.sel(lat=target_lats, lon=target_lons, method='nearest')
ds_cmcc_target = ds_cmcc.sel(lat=target_lats, lon=target_lons, method='nearest')
ds_cnrm_target = ds_cnrm.sel(lat=target_lats, lon=target_lons, method='nearest')

# Ensure all datasets have the same time dimension (Intersection). To complement the DQ done before. 
print("Aligning time dimensions...")
common_times = np.intersect1d(np.intersect1d(ds_eth_target.time.values, ds_cmcc_target.time.values), ds_cnrm_target.time.values)

ds_eth_aligned  = ds_eth_target.sel(time=common_times)
ds_cmcc_aligned = ds_cmcc_target.sel(time=common_times)
ds_cnrm_aligned = ds_cnrm_target.sel(time=common_times)

# %%
#############################################################################
##-------------------SHAPPING THE ARRAYS FOR EACH CPM----------------------##
#############################################################################
eth_100m_array  = ds_eth_aligned.wsa100m.values
cnrm_100m_array = ds_cnrm_aligned.wsa100m.values
cmcc_100m_array = ds_cmcc_aligned.wsa100m.values

lat_array   = ds_eth_aligned.lat.values
lon_array  = ds_eth_aligned.lon.values
# %%
#############################################################################
##-------------COMPUTING TEMPORAL DECORRELATION FOR EACH CPM---------------##
#############################################################################

tau_dec_eth, tau_dec_meanlon_eth   = computing_decorrelation_per_array(eth_100m_array)
tau_dec_cnrm, tau_dec_meanlon_cnrm = computing_decorrelation_per_array(cnrm_100m_array)
tau_dec_cmcc, tau_dec_meanlon_cmcc = computing_decorrelation_per_array(cmcc_100m_array)

print("MinDec ETH:"+str(tau_dec_eth.min()))
print("MaxDec ETH:"+str(tau_dec_eth.max()))

print("MinDec CNRM:"+str(tau_dec_cnrm.min()))
print("MaxDec CNRM:"+str(tau_dec_cnrm.max()))

print("MinDec CMCC:"+str(tau_dec_cmcc.min()))
print("MaxDec CMCC:"+str(tau_dec_cmcc.max()))

# %%
#############################################################################
##----------------------MAPPING DECORRELATION VALUES-----------------------##
#############################################################################

# Create figure with improved proportions
fig = plt.figure(figsize=(15, 6))
gs  = GridSpec(1, 2, width_ratios=[1, 3.5], height_ratios=[1], figure=fig)

ax1 = fig.add_subplot(gs[0])
style_axis(ax1)
ax1.plot(tau_dec_meanlon_eth, lat_array, '#edae49', linewidth=2.5, label = "ETH")
ax1.plot(tau_dec_meanlon_cnrm, lat_array, '#00798c', linewidth=2.5, label = "CNRM")
ax1.plot(tau_dec_meanlon_cmcc, lat_array, '#d1495b', linewidth=2.5, label = "CMCC")
ax1.set_ylabel('Latitude [°N]', fontsize=13.5)
ax1.set_xlabel('Mean Decorrelation [h]', fontsize=13.5)
ax1.set_title("a) Mean decorrelation profile", fontsize=14, fontweight="bold")
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim([lat_array.min(), lat_array.max()])
ax1.set_xlim([tau_dec_meanlon_cmcc.min() - 2, tau_dec_meanlon_eth.max() + 2])
ax1.tick_params(labelsize=13)
ax1.legend(fontsize=13)

# Subplot 2: Spatial distribution map 
ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

# Calculate averaged decorrelation time across the 3 models
# ASUMIENDO que tienes arrays 2D: tau_d_eth, tau_d_cnrm, tau_d_cmcc
# Si no los tienes, necesitas crearlos primero
tau_dec_averaged = (tau_dec_eth + tau_dec_cnrm + tau_dec_cmcc) / 3

# Create mesh grid
lon_mesh, lat_mesh = np.meshgrid(lon_array, lat_array)
scatter = ax2.scatter( lon_mesh.flatten(), lat_mesh.flatten(), c=tau_dec_averaged.flatten(), cmap='viridis', marker='o', s=100,                          # Tamaño
         edgecolors='black', linewidths=0.5, transform=ccrs.PlateCarree(), vmin=tau_dec_averaged.min(), vmax=tau_dec_averaged.max())
ax2.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black", linewidth=0.8)
ax2.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")
gl = ax2.gridlines(draw_labels=True, alpha=0.3, linestyle="--")
gl.left_labels   = False
gl.top_labels    = False
gl.right_labels  = True
gl.bottom_labels = True
y_ticks_ax1 = ax1.get_yticks()
gl.ylocator = ticker.FixedLocator(y_ticks_ax1)
gl.xlocator = MaxNLocator(nbins=8)
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}
ax2.set_title("b) Spatial distribution of sample points with decorrelation time", fontsize=14, fontweight="bold")
ax2.set_extent([lon_array.min(), lon_array.max(), lat_array.min(), lat_array.max()], crs=ccrs.PlateCarree())
cbar = plt.colorbar(scatter, ax=ax2, orientation="vertical", shrink=0.8, pad=0.06, aspect=30)
cbar.set_label("Multi-Model Mean Decorrelation Time [h]", fontsize=13) 
cbar.ax.tick_params(labelsize=12)

# Asegurar igualdad de alturas
ax2.set_aspect('auto') 
plt.subplots_adjust(wspace=0.05) 
plt.tight_layout()

plt.subplots_adjust(wspace=0.08, left=0.08, right=0.95, top=0.92, bottom=0.12)
plt.savefig(bd_out_fig + "MapDecorrelation_SingleMember.png", format='png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# %%
