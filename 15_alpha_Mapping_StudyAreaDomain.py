import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.patches as mpatches


"""
Code to generate the map of the study area for the WP3. 
It is based in Cartopy and uses the ETOPO 2022 raster to define the 
elevations because we are interested into shwing a full descriotion of 
the geographical features in teh study domain. 

Author : Nathalia Correa-Sánchez
"""
########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

bd_out_fig     = "/Dati/Outputs/Plots/WP3_development/"
bd_base_raster = "/Dati/Outputs/Climate_Provinces/Development_Rasters/ALP3_ETOPO2022_60sArc.tif"

########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

# Definir coordenadas del área de estudio
crop_coords = { 'lon_min': 0.5,    # Límite occidental   : 0.5°E
                'lat_min': 40.2,   # Límite meridional   : 40.2°N
                'lon_max': 16.3,   # Límite oriental     : 16.3°E
                'lat_max': 49.7  }   # Límite septentrional: 49.7°N

# Definir coordenadas del mapa (1° más grande en todas las direcciones)
map_coords = { 'lon_min': crop_coords['lon_min'] - 0.5,
               'lat_min': crop_coords['lat_min'] - 0.5,
               'lon_max': crop_coords['lon_max'] + 0.5,
               'lat_max': crop_coords['lat_max'] + 0.5 }
cmap_terrain = plt.cm.terrain
########################################################################################
##-----------------------------CREATING THE VISUALIZATION-----------------------------##
########################################################################################

elevation_file = bd_base_raster # Cargar el archivo de elevación

fig = plt.figure(figsize=(8, 6))
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([map_coords['lon_min'], map_coords['lon_max'], map_coords['lat_min'], map_coords['lat_max']], crs=ccrs.PlateCarree()) 
with rasterio.open(elevation_file) as src:
    elevation     = src.read(1)
    elevation_max = np.ceil(np.max(elevation) / 500) * 500
    levels        = np.linspace(0, elevation_max, 11)
    norm          = colors.BoundaryNorm(levels, cmap_terrain.N)

    img = ax.imshow(elevation,extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
                    transform=ccrs.PlateCarree(),cmap=cmap_terrain, norm=norm, origin='upper', zorder=1)
    ocean = ax.add_feature(cfeature.OCEAN, facecolor='#bedae6', edgecolor='black', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, color='black', zorder=3)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.9, color='black')
ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.9, color='blue', alpha=0.7)
ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='lightblue', edgecolor='blue', alpha=0.7)
# ax.add_feature(cfeature.OCEAN, facecolor='#bedae6', edgecolor='black', zorder=0)
ax.add_feature(cfeature.LAND, facecolor='none', zorder=0) # Asegurarse que el océano se dibuje antes que los datos de elevación
gl              = ax.gridlines(draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
gl.top_labels   = False
gl.right_labels = False
gl.xlines       = True
gl.ylines       = True
gl.xformatter   = LONGITUDE_FORMATTER
gl.yformatter   = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}

# Añadir un recuadro rojo para marcar el dominio del área de estudio
rectangle = mpatches.Rectangle( (crop_coords['lon_min'], crop_coords['lat_min']), crop_coords['lon_max'] - crop_coords['lon_min'], 
    crop_coords['lat_max'] - crop_coords['lat_min'], edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
ax.add_patch(rectangle)
# Añadir nombres de lugares importantes 
important_places = {'Italy'            : (12.5, 42.5),
                    'Switzerland'      : (8.2, 47),
                    'Austria'          : (13.5, 47.5),
                    'France'           : (5.0, 46.0),  
                    'Germany'          : (10.0, 49.0),  
                    'Po Valley'        : (10.0, 45.0),
                    'Alps'             : (9.0, 46.5),
                    'Mediterranean Sea': (8.0, 41.0),
                    'Adriatic Sea'     : (15.0, 43.0), 
                    'Pyrenees'         : (2.0,43.0)}

for place, coords in important_places.items():
    if (map_coords['lon_min'] <= coords[0] <= map_coords['lon_max'] and 
        map_coords['lat_min'] <= coords[1] <= map_coords['lat_max']):
        ax.text(coords[0], coords[1], place, 
                horizontalalignment='center', 
                transform=ccrs.PlateCarree(),
                fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

cbar      = fig.colorbar(img, ax=ax, shrink=0.7, pad=0.05, ticks=levels)
cbar.set_label('Elevation [m]', fontsize=10)
red_patch = mpatches.Patch(color='red', label='Study Domain', fill=False, linewidth=2)
ax.legend(handles=[red_patch], loc='lower right', framealpha=0.9)
plt.title('Study area', fontsize=14, fontweight='bold', pad=20)
coord_text = (f"Domain: {crop_coords['lon_min']}°E to {crop_coords['lon_max']}°E, " f"{crop_coords['lat_min']}°N to {crop_coords['lat_max']}°N")
plt.figtext(0.5, 0.09, coord_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))

plt.tight_layout()
plt.savefig(bd_out_fig+'MapWP3_StudyDomain.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
