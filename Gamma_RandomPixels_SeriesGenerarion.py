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
Codigo para generar el archivo con el numpy array de las series aleatorias
seleccionadas de cada miembro CPM para otros análisis. Se hace porque la 
lectura de los archivos es lenta y consume muchos recursos de memoria. Por eso, 
sólo se peude correr una vez por CPM member.

Author : Nathalia Correa-Sánchez  
"""

#############################################################################
##-------------------------DEFINING IMPORTANT PATHS------------------------##
#############################################################################
bd_in_ws   = "/Dati/Data/WS_CORDEXFPS/"
bd_out_fil = "/Dati/Outputs/Random_SeriePixels_WS/"
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

bd_in_cpm = bd_in_eth
fil_name  = "RandomSerie_ETH.npy"

## Ingreso los indices solo temporalmente para las pruebas
lon_indices = np.array([ 35,  88,  21, 172, 159, 138,  68, 143, 173,  56, 156, 102])
lat_indices = np.array([67, 76, 76, 70, 92, 59,  1, 57, 79, 50, 85, 29])

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
    all_indices = np.arange(lon_dim * lat_dim)
    np.random.shuffle(all_indices)
    unique_indices = all_indices[:num_pixels]
    
    # Convertir los índices 1D a índices 2D (longitud, latitud)
    lon_indices = (unique_indices // lat_dim).astype(int)
    lat_indices = (unique_indices % lat_dim).astype(int)
    
    # Seleccionar los píxeles únicos
    subset_arr = sample_arr[:, lat_indices, lon_indices]
    
    return subset_arr, lon_indices, lat_indices

#######################################################################################
##------EXTRACTING SAMPLE OF PIXELS BY CONCATENATING THE REGION - SINGLE MEMBER------##
#######################################################################################

arr_cut_cpm    = cut_concat_netcdf_files(bd_in_cpm, min_lon, max_lon, min_lat, max_lat)
sample_arr_cpm = arr_cut_cpm.wsa100m.values
subset_arr_cpm = sample_arr_cpm[:, lat_indices, lon_indices] # Shepe: (87672, 12)

#######################################################################################
##------------SAVING RANDOM SERIES FILES FOR THE CPM IN EXTERNAL NPY FILES-----------##
#######################################################################################

np.save(bd_out_fil+fil_name, subset_arr_cpm)

print("#####----FIN DE LA CREACION DEL ARCHIVO .NPY----####")