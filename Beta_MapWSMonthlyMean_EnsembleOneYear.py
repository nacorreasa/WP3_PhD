import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

#############################################################################
##-------------------------DEFINNING IMPORTANT PATHS-----------------------##
#############################################################################
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"
bd_in_nc   = '/Dati/Data/WS_CORDEXFPS/Ensemble_Mean/wsa100m/'

#############################################################################
##-------------------------READING THEINPUT FILE---------------------------##
#############################################################################
ds = xr.open_dataset(bd_in_nc+'ensemble_mean_ws100m_2000.nc')





# Calcular promedios mensuales - Método corregido para datos horarios
# Primero convertimos los datos a un Dataset si es un DataArray
if isinstance(ds.wsa100m, xr.DataArray):
    ds_temp = ds.wsa100m.to_dataset()
else:
    ds_temp = ds

# Ahora hacemos el resample mensual
monthly_means = ds_temp.resample(time='1M', closed='left', label='left').mean()

# Configurar el subplot
fig = plt.figure(figsize=(15, 12))
months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
          'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

# Definir niveles discretos para el colorbar
levels = np.linspace(0, 30, 16)  # 15 intervalos entre 0 y 30
norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

# Crear una figura común para todos los subplots
proj = ccrs.PlateCarree()

# Crear los subplots
for i, month in enumerate(monthly_means.wsa100m.values):
    ax = plt.subplot(3, 4, i+1, projection=proj)
    
    # Plotear los datos
    im = plt.pcolormesh(monthly_means.lon, monthly_means.lat, month,
                       transform=ccrs.PlateCarree(),
                       cmap='viridis',
                       norm=norm)
    
    # Añadir características del mapa
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    # Ajustar la extensión del mapa según tus datos
    ax.set_extent([-2, 17, 39, 50])  # Ajustado según las coordenadas de tus datos
    
    # Añadir título
    ax.set_title(months[i])

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar    = plt.colorbar(im, cax=cbar_ax, orientation='vertical', 
                   label='Velocidad del viento (m/s)',
                   ticks=levels[::2])  # Mostrar cada segundo nivel para no sobrecargar la colorbar
plt.subplots_adjust(right=0.9)
plt.suptitle('Promedio Mensual de Velocidad del Viento a 100m - 2000', y=0.95)
# plt.savefig(bd_out_fig+'Temporal_promedios_mensuales_viento.png', bbox_inches='tight', dpi=300)
plt.show()


































# Calcular promedios mensuales
monthly_means = ds.resample(time='M').mean()

# Configurar el subplot
fig = plt.figure(figsize=(15, 12))
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# Definir niveles discretos para el colorbar
levels = np.arange(0, 31, 2)  # De 0 a 30 m/s en intervalos de 2

# Crear una figura común para todos los subplots
proj = ccrs.PlateCarree()

# Crear los subplots
for i, month in enumerate(months, 1):
    ax = plt.subplot(3, 4, i, projection=proj)
    
    # Plotear los datos
    im = monthly_means.wsa100m[i-1].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0,
        vmax=30,
        levels=levels,
        add_colorbar=False
    )
    
    # Añadir características del mapa
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    # Ajustar la extensión del mapa según tus datos
    # Modifica estos valores según tu área de interés
    ax.set_extent([-10, 5, 35, 45])  # [lon_min, lon_max, lat_min, lat_max]
    
    # Añadir título
    ax.set_title(month)

# Añadir colorbar común
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar    = plt.colorbar(im, cax=cbar_ax, orientation='vertical', label='Velocidad del viento (m/s)')
plt.subplots_adjust(right=0.9)
plt.suptitle('Promedio Mensual de Velocidad del Viento a 100m - 2000', y=0.95)
