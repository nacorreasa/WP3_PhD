import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
import numpy as np
import numpy.ma as ma
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle

"""
Code for visualizing the four maps of spatial categories and the combination of them
into a single figure for the publication. 

Author: Nathalia Correa-Sánchez
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

file_cats  = "/Dati/Outputs/Climate_Provinces/CSVs/Combination_RIX.csv"
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"
bd_in_rast = "/Dati/Outputs/Climate_Provinces/Development_Rasters/FinalRasters_In-Out/"
ras_clim   = "ReducedBeck-KG_present_ALP3_remCPM_WGS84.tif"
ras_rough  = "CMCC_Z0-Class_ALP3_remCPM_WGS84.tif"
ras_topog  = "ReducedSurface_VarianceSlope_ALP3_remCPM_WGS84.tif"
ras_comb   = "SEA-LAND_Combined_RIX_remCPM_WGS84.tif"
bd_out_tc  = "/Dati/Outputs/WP3_SamplingSeries_CPM/"

########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################
filas_eliminar    = [0]  # Primera  fila, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada 
columnas_eliminar = [0]  # Primera columna, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada


########################################################################################
##------------------LOADING INPUT RASTERS AND EXPLORING DATA STRUCTURE----------------##
########################################################################################
raster_files = [bd_in_rast+ras_clim, bd_in_rast+ras_rough, bd_in_rast+ras_topog, bd_in_rast+ras_comb]

print("Diagnóstico de rasters:")
for i, raster_file in enumerate(raster_files):
    with rasterio.open(raster_file) as src:
        print(f"\nRaster {i+1}: {raster_file}")
        print(f"Dimensiones: {src.height} x {src.width}")
        print(f"Resolución: {src.res}")
        print(f"Extensión: {src.bounds}")
        print(f"CRS: {src.crs}")

########################################################################################
##-----------------------------DEFINING MAP PARAMETERS-------------------------------##
########################################################################################

# Define default nodata value based on your project's pattern
DEFAULT_NODATA = -3.40282e+38  # Based on your other scripts

water_color = '#345F77' 

titles       = ['a) Climate', 'b) Roughness', 'c) Topography', 'd) Combined Classification']
colour_clima = ['#F5A500', '#64C864', '#4B92DB', '#A8A8A8']           # Values in order 1, 2, 3, 4
colour_rough = [water_color,'#FFDA8A', '#F2A05D', '#D95D30', '#8C2A04'] # Values in order 1-sea, 2, 3, 4, 5
colour_topog = ['#E5E0CB','#B8D6BE', '#73AE80', '#2A6B3D']            # Values in order 1, 2, 3, 4, 

climate_labels    = ['Ar:Arid', 'Tm:Temperate', 'Co:Cold', 'Td:Tundra'] # Values in order 1, 2, 3, 4
roughness_labels  = [r"$R_1$:Very smooth (water)", r"$R_2$:Smooth", r"$R_3$:Moderate", r"$R_4$:Rough", r"$R_5$:Very rough"]  # Values in order 1-sea, 2, 3, 4, 5
topography_labels = [r"$T_1$:Flat", r"$T_2$:Gentle", r"$T_3$:Moderate", r"$T_4$:Complex"] # Values in order 1, 2, 3, 4, 

crop_coords = {'lon_min': 0.5,     # Western limit   : 0.5°E
               'lat_min': 40.2,    # Southern limit  : 40.2°N
               'lon_max': 16.3,    # Eastern limit   : 16.3°E 
               'lat_max': 49.7  }  # Northern limit  : 49.6°N

extent = [crop_coords['lon_min'], crop_coords['lon_max'], crop_coords['lat_min'], crop_coords['lat_max']]

########################################################################################
##-----------------------------DEFINING RELEVANT FUNCTIONS----------------------------##
########################################################################################

def decode_category(cat_value):
    """Decode category: 1.0 = water, 3-digit = climate-roughness-slope"""
    if cat_value == 1.0:  # Special case: water (only single digit)
        return None, 1, None  # Water is roughness category 1
    
    cat_str = str(int(cat_value))
    if len(cat_str) == 3:
        climate = int(cat_str[0])
        roughness = int(cat_str[1])
        slope = int(cat_str[2])
        return climate, roughness, slope
    else:
        return None, None, None

########################################################################################
##-----ABRIENDO EL CROPPED RASTER PARA EXTRAER CADA CLASE & AJUSTANDO EL DATAFRAME----##
########################################################################################


comblay              = rasterio.open(bd_in_rast+"SEA-LANDCropped_Combined_RIX_remCPM_WGS84.tif")
band1_o              = comblay.read(1) ## Solo tiene una banda
band1_o[band1_o < 0] = np.nan          ## Reemplazando los negativos con nan o 0(Tener en cuenta NoData= -3.40282e+38) 
# Ajustillo para que coincida los xarrays (incluso despues del crop)
band1 = np.delete(np.delete(band1_o, filas_eliminar[0], axis=0), columnas_eliminar[0], axis=1) 

# Obteniendo valores unicos de las categorias
unique_vals    = np.unique(band1)
unique_vals    = unique_vals[np.isfinite(unique_vals)]
num_categories = len(unique_vals)

# Contar píxeles para cada valor único
pixel_counts = {}
mask = np.isfinite(band1)
valid_values = band1[mask]

# Método más directo y confiable
for val in unique_vals:
    pixel_counts[val] = np.sum(valid_values == val)

counts_array = np.array([pixel_counts[val] for val in unique_vals])

df_cats = pd.DataFrame({'value': unique_vals, 'count': counts_array,})

########################################################################################
##--------FILTERING CATEGORIES UNDER THE 25% PERCENTILE: LOW ATYPICAL VALUES----------##
########################################################################################

total_count         = df_cats["count"].sum()
df_cats["rel_freq"] = (df_cats["count"] / total_count) * 100  

percentages = df_cats.rel_freq.values * 100
p25         = np.percentile(percentages, 25)
df_filt     = df_cats[df_cats.rel_freq >= p25]
df_filt     = df_filt.reset_index(drop=True) ## Resetea el indice que se habia dañado luego del filtrado. 
fl_cats     = df_filt['value'].values.astype(int)

########################################################################################
##----------------LOADING THE COORDINATES TO BE USED TO ALOCATE THE RANDOM POINTS----------------##
########################################################################################


# Abrir las coordenadas
with open(bd_out_tc + 'Random_coords_N100.pkl', 'rb') as f:
    coords_dict = pickle.load(f)


########################################################################################
##-----------------MAPPING RASTERS FOR VISUALIZATION OF THE CATEGORIES----------------##
########################################################################################

cmap_clima = ListedColormap(colour_clima)
cmap_rough = ListedColormap(colour_rough)
cmap_topog = ListedColormap(colour_topog)

# Define common extent
common_extent = [crop_coords['lon_min'], crop_coords['lon_max'],  crop_coords['lat_min'], crop_coords['lat_max']]

fig = plt.figure(figsize=(15, 12))
# Manually define position and size of each subplot
# [left, bottom, width, height] in figure coordinates (0-1)
ax1 = fig.add_axes([0.07, 0.50, 0.34, 0.35], projection=ccrs.PlateCarree())  # Climate
ax2 = fig.add_axes([0.60, 0.50, 0.34, 0.35], projection=ccrs.PlateCarree())  # Roughness
ax3 = fig.add_axes([0.07, 0.15, 0.34, 0.35], projection=ccrs.PlateCarree())  # Topography
ax4 = fig.add_axes([0.60, 0.15, 0.34, 0.35], projection=ccrs.PlateCarree())  # Combined

# Process the first three rasters
axs = [ax1, ax2, ax3]  # List to iterate over the first 3 subplots of categories
for i, (raster_file, title, ax) in enumerate(zip(raster_files[:3], titles[:3], axs)):
    with rasterio.open(raster_file) as src:
        data = src.read(1)
        # Get nodata value from raster metadata, use default if None
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        
        # Convert to float to allow NaN values
        data = data.astype(np.float64)
        
        # Convert negative values to NaN (following your existing pattern)
        data[data < 0] = np.nan
        
        # Create masked array for invalid values (NaN, inf) - consistent with your workflow
        masked_data = ma.masked_invalid(data)

        if i == 0:  # Climate
            cmap   = cmap_clima
            bounds = np.arange(0.5, 4.5 + 1)
            norm   = BoundaryNorm(bounds, cmap.N)
            labels = climate_labels
        elif i == 1:  # Roughness
            cmap   = cmap_rough
            bounds = np.arange(0.5, 5.5 + 1)
            norm   = BoundaryNorm(bounds, cmap.N)
            labels = roughness_labels
        elif i == 2:  # Topography
            cmap   = cmap_topog
            bounds = np.arange(0.5, 4.5 + 1)
            norm   = BoundaryNorm(bounds, cmap.N)
            labels = topography_labels

        ax.set_extent(common_extent)
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, alpha=0.7)

        bounds = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        img    = ax.imshow(masked_data, extent=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), origin='upper')

        legend_elements = [Patch(facecolor=color, edgecolor='k', label=lab) for color, lab in zip(cmap.colors, labels)]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=13, framealpha=0.7)

        gl              = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', xlabel_style={'fontsize': 13}, ylabel_style={'fontsize': 12})
        gl.top_labels   = False
        gl.right_labels = False
        
        ax.set_title(title, fontsize=14, fontweight='bold')

# Process combined categories raster

water_color     = '#345F77'  # Azul petróleo oscuro con toque grisáceo para el agua
band1_transform = comblay.transform
band1_crs       = comblay.crs
west            = band1_transform[2]  # x_0
east            = west + band1_transform[0] * comblay.width  # x_0 + width * pixel_width
north           = band1_transform[5]  # y_0
south           = north + band1_transform[4] * comblay.height  # y_0 + height * pixel_height
if north < south:
    north, south = south, north

# Crear colormap personalizado agregando el color del agua
colors = []
if 1 in unique_vals:
    colors.append(water_color)  # Color para agua    
    # Obtener colores de nipy_spectral para el resto
    nipy_colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_categories))
    for i in range(1, num_categories):
        colors.append(nipy_colors[i])
else:
    nipy_colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_categories))
    colors = nipy_colors
cmap   = ListedColormap(colors)
bounds = np.arange(num_categories + 1)
norm   = BoundaryNorm(bounds, cmap.N)

# Enmascarar valores no datos
nodata            = -3.40282e+38  # Original NoData value from the raster
masked_data       = ma.masked_where(band1 == nodata, band1)
band1_categorized = np.full(band1.shape, np.nan)
mask              = np.isfinite(band1)
for i, val in enumerate(unique_vals):
    band1_categorized[band1 == val] = i

img1 = ax4.imshow(band1_categorized, extent=[west, east, south, north], origin="upper",  cmap=cmap, norm=norm, alpha=0.7, transform=ccrs.PlateCarree())
ax4.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
ax4.coastlines(resolution='50m')
ax4.add_feature(cfeature.BORDERS, linestyle=':')
ax4.set_aspect('equal')
ax4.set_title('d) Combined Classification', fontsize=14, fontweight="bold")
gl              = ax4.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', xlabel_style={'fontsize': 13}, ylabel_style={'fontsize': 13})
gl.top_labels   = False
gl.right_labels = False
cax  = fig.add_axes([0.96, 0.19, 0.015, 0.27])  # Position and size of colorbar # [left, bottom, width, height]
cbar = plt.colorbar(img1, cax=cax)  # Remove specified ticks
cbar.set_ticks([])  # Remove numeric ticks
cbar.set_label("Spatial category (Total: "+str(num_categories)+")", fontsize=13)


# Usar directamente la transformación geoespacial en lugar de cálculos manuales de rasterior para extraer las coordenadas
all_lons       = []
all_lats       = []
all_categories = []

# Verificar antes de plotear
valid_count   = 0
invalid_count = 0

for i, cat in enumerate(fl_cats):
    if cat in coords_dict:
        # Obtener los índices
        sampled_idx = coords_dict[cat]['idx']
        sampled_idy = coords_dict[cat]['idy']
        
        # Convertir índices a coordenadas geográficas usando la transformación exacta de rasterio
        xs, ys = [], []
        for j in range(len(sampled_idx)):
            row, col = sampled_idx[j], sampled_idy[j]
            
            # Verificar que el punto realmente tiene el valor de categoría esperado
            if band1[row, col] != cat:
                print(f"Error: Punto ({row}, {col}) esperaba cat={cat}, pero tiene valor={band1[row, col]}")
                invalid_count += 1
                continue
                
            # Usar la transformación exacta de rasterio para convertir fila/columna a coordenadas
            x, y = band1_transform * (col + 0.5, row + 0.5)
            xs.append(x)
            ys.append(y)
            
            # Añadir a los arrays generales
            all_lons.append(x)
            all_lats.append(y)
            all_categories.append(cat)
            valid_count += 1
        
        # Plotear los puntos de esta categoría y la leyenda solo en uno
        if i == 0:  # Solo en la primera categoría
            ax4.scatter(xs, ys, c="black", s=15, alpha=0.8, transform=ccrs.PlateCarree(), edgecolors='white', linewidths=0.5, label="Samples location")
        else:
            ax4.scatter(xs, ys, c="black", s=15, alpha=0.8, transform=ccrs.PlateCarree(), edgecolors='white', linewidths=0.5)

print(f"Puntos válidos: {valid_count}, Puntos inválidos: {invalid_count}")
ax4.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', fontsize=13, framealpha=0.8, ncol=1)

plt.savefig(bd_out_fig+'Maps_StudyAreas_SpatialCats.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

########################################################################################
##--------------------VISUALIZING DISTRIBUTION OF SPATIAL CATEGORIES------------------##
########################################################################################

climate_names    = {1: 'Ar', 2: 'Tm', 3: 'Co', 4: 'Td'}  # Valores en orden 1, 2, 3, 4 


fig = plt.figure(figsize=(18, 6))

# Panel A: Ranked distribution of all categories (ocupando espacio de A y B)
ax1 = plt.subplot(1, 3, (1, 2))  # Ocupa las primeras 2 columnas
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False) 
df_sorted = df_cats.sort_values('rel_freq', ascending=False).reset_index(drop=True)
colors    = ['#2E86AB' if freq >= p25 else '#E63946' for freq in df_sorted['rel_freq']]
bars      = ax1.bar(range(len(df_sorted)), df_sorted['rel_freq'], color=colors, alpha=0.8, width=0.8)  

ax1.axhline(y=p25, color='black', linestyle='--', linewidth=2, label=f'25th percentile ({p25:.3f}%)')
ax1.set_xlabel('Spatial category rank', fontsize=13, fontweight='bold')
ax1.set_ylabel('Relative frequency (%)', fontsize=13, fontweight='bold')
ax1.set_title('a) Distribution of pixels across all spatial categories', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.set_xlim(-1, len(df_sorted))
ax1.grid(False)
ax1.tick_params(axis='y', direction='in', labelsize=12)  
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 

for i, (idx, row) in enumerate(df_sorted.iterrows()):
    if row['value'] == 1.0:
        category_label = 'Water'
    else:
        climate, roughness, slope = decode_category(row['value'])
        climate_str               = climate_names.get(climate, f'C{climate}') if climate else ''
        roughness_str             = f"R_{{{roughness}}}" if roughness else ''
        slope_str                 = f"T_{{{slope}}}"     if slope else ''

        category_label = rf"${climate_str}{roughness_str}{slope_str}$"

    ax1.text(i, row['rel_freq'] * 3.9, category_label,ha='center', va='top', fontsize=10.5, rotation=90, color='black', fontweight='bold')
    
# Legend using YOUR dataframes
above_patch       = mpatches.Patch(color='#2E86AB', label=f'Above 25th percentile (n={len(df_filt)})')
below_patch       = mpatches.Patch(color='#E63946', label=f'Below 25th percentile (n={len(df_cats)-len(df_filt)})')
threshold_line    = mpatches.Patch(color='black', label=f'25th percentile threshold')
ax1.legend(handles=[above_patch, below_patch, threshold_line], loc='upper right', fontsize=12)

# Panel B: Filtering effectiveness 
ax2 = plt.subplot(2, 3, 3)  
categories      = ['All categories', 'Filtered\n(>25th percentile)']
n_categories    = [len(df_cats), len(df_filt)]
cumulative_freq = [df_cats['rel_freq'].sum(), df_filt['rel_freq'].sum()]

# Create grouped bar chart
x     = np.arange(len(categories))
width = 0.35
bars1 = ax2.bar(x - width/2, n_categories, width, label='Number of categories',   color='#1f77b4', alpha=0.8)

# Secondary y-axis for cumulative frequency
ax2_twin = ax2.twinx()
bars2    = ax2_twin.bar(x + width/2, cumulative_freq, width, label='Cumulative frequency [%]', color='#F28E2C', alpha=0.8)

for bar, count in zip(bars1, n_categories):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(n_categories)*0.02, f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)

for bar, freq in zip(bars2, cumulative_freq):
    ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{freq:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)


ax2.set_xlabel('Category sets', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of categories', fontsize=12, fontweight='bold', color='#1f77b4')
ax2_twin.set_ylabel('Cumulative frequency [%]', fontsize=12, fontweight='bold', color='#F28E2C')
ax2.set_title('b) Filtering effectiveness', fontsize=14, fontweight='bold', pad=10)
ax2_twin.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=12)
# Color the y-axis labels
ax2.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)
ax2_twin.tick_params(axis='y', labelcolor='#F28E2C', labelsize=11)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.tick_params(which='both', direction='in')

# Panel C: Top filtered categories (más pequeño, abajo en la tercera columna)  
ax3 = plt.subplot(2, 3, 6)  # Fila 2, columna 3

# Show top 10 from  df_filt
top_categories  = df_filt.sort_values('rel_freq', ascending=True).tail(10)
detailed_labels = []
for _, row in top_categories.iterrows():
    if row['value'] == 1.0:
        detailed_labels.append('Water')
    else:
        climate, roughness, slope  = decode_category(row['value'])
        climate_str                = climate_names.get(climate, f'C{climate}') if climate else ''
        roughness_str              = f"R_{{{roughness}}}" if roughness else ''
        slope_str                  = f"T_{{{slope}}}"     if slope else ''
        detailed_labels.append(rf"${climate_str}{roughness_str}{slope_str}$")


# Horizontal bar chart
bars = ax3.barh(range(len(top_categories)), top_categories['rel_freq'], color='#2E86AB', alpha=0.8)

# Add value labels on bars
for i, (bar, freq) in enumerate(zip(bars, top_categories['rel_freq'])):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,  f'{freq:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False) 
ax3.set_yticks(range(len(top_categories)))
ax3.set_yticklabels(detailed_labels, fontsize=12)
ax3.tick_params(axis='x', labelsize=12)  
ax3.set_xlabel('Relative frequency [%]', fontsize=13, fontweight='bold')
ax3.set_title('c) Top 10 spatial categories', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
ax3.tick_params(which='both', direction='in')

plt.tight_layout()
# plt.savefig(bd_out_fig+'Stats_SpatCat_TotFilt.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

########################################################################################
##------------VISUALIZING DISTRIBUTION OF SPATIAL CATEGORIES WITH CDF LIKE------------##
########################################################################################

climate_names    = {1: 'Ar', 2: 'Tm', 3: 'Co', 4: 'Td'}  # Valores en orden 1, 2, 3, 4 


fig = plt.figure(figsize=(18, 6))

# Panel A: Ranked distribution of all categories (ocupando espacio de A y B)
ax1 = plt.subplot(1, 3, (1, 2))  # Ocupa las primeras 2 columnas
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False) 
df_sorted = df_cats.sort_values('rel_freq', ascending=False).reset_index(drop=True)
colors    = ['#2E86AB' if freq >= p25 else '#E63946' for freq in df_sorted['rel_freq']]
bars      = ax1.bar(range(len(df_sorted)), df_sorted['rel_freq'], color=colors, alpha=0.8, width=0.8)  

ax1.axhline(y=p25, color='black', linestyle='--', linewidth=2, label=f'25th percentile ({p25:.3f}%)')
ax1.set_xlabel('Spatial category rank', fontsize=13, fontweight='bold')
ax1.set_ylabel('Relative frequency (%)', fontsize=13, fontweight='bold')
ax1.set_title('a) Distribution of pixels across all spatial categories', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.set_xlim(-1, len(df_sorted))
ax1.grid(False)
ax1.tick_params(axis='y', direction='in', labelsize=12)  
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 

for i, (idx, row) in enumerate(df_sorted.iterrows()):
    if row['value'] == 1.0:
        category_label = 'Water'
    else:
        climate, roughness, slope = decode_category(row['value'])
        climate_str               = climate_names.get(climate, f'C{climate}') if climate else ''
        roughness_str             = f"R_{{{roughness}}}" if roughness else ''
        slope_str                 = f"T_{{{slope}}}"     if slope else ''

        category_label = rf"${climate_str}{roughness_str}{slope_str}$"

    ax1.text(i, row['rel_freq'] * 3.9, category_label,ha='center', va='top', fontsize=10.5, rotation=90, color='black', fontweight='bold')
    
# Legend using YOUR dataframes
above_patch       = mpatches.Patch(color='#2E86AB', label=f'Above 25th percentile (n={len(df_filt)})')
below_patch       = mpatches.Patch(color='#E63946', label=f'Below 25th percentile (n={len(df_cats)-len(df_filt)})')
threshold_line    = mpatches.Patch(color='black', label=f'25th percentile threshold')
ax1.legend(handles=[above_patch, below_patch, threshold_line], loc='upper right', fontsize=12)

# Panel B: Cumulative Distribution Function
ax2 = plt.subplot(1, 3, 3)

# Usar todas las categorías ordenadas por frecuencia (df_sorted)
# Pero mostrar solo las primeras 25-30 para claridad
df_show = df_sorted.head(25).copy()
df_show['cumulative_freq'] = df_show['rel_freq'].cumsum()

# Colores: azul para las 17 filtradas, rojo para el resto
n_filtered = len(df_filt)  # = 17
colors = ['#2E86AB' if i < n_filtered else '#E63946' for i in range(len(df_show))]

# Plot CDF
bars = ax2.bar(range(len(df_show)), df_show['cumulative_freq'], 
               color=colors, alpha=0.8, width=0.8)

# Línea horizontal en 97.7% y línea vertical en categoría 17
cumulative_17 = df_show.iloc[16]['cumulative_freq']  # Categoría 17
ax2.axhline(y=cumulative_17, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=16, color='red', linestyle='--', linewidth=2)

# Anotación
ax2.annotate(f'Top {n_filtered} categories\n{cumulative_17:.1f}% of domain', 
             xy=(16, cumulative_17), 
             xytext=(19, cumulative_17-8),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, fontweight='bold', color='red',
             ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Etiquetas TODAS las categorías mostradas - DENTRO de las barras, hacia arriba
for i in range(len(df_show)):
    row = df_show.iloc[i]
    
    # Generar etiqueta base (SIN notación + porque era incorrecta)
    if row['value'] == 1.0:
        category_label = 'Water'
    else:
        climate, roughness, slope = decode_category(row['value'])
        climate_str               = climate_names.get(climate, f'C{climate}') if climate else ''
        roughness_str             = f"R_{{{roughness}}}" if roughness else ''
        slope_str                 = f"T_{{{slope}}}" if slope else ''
        category_label = rf"${climate_str}{roughness_str}{slope_str}$"
    
    # Color de etiqueta según tipo
    label_color = 'black' if i < n_filtered else 'black'  # Blanco para contraste dentro de barras
    
    # Posición Y: dentro de la barra, en la parte superior
    bar_height = df_show.iloc[i]['cumulative_freq']
    y_position = 11
    
    # TODAS las etiquetas, posicionadas correctamente
    ax2.text(i, y_position, category_label, ha='center', va='top',  fontsize=9.5, rotation=90, color=label_color, fontweight='bold')

# Configuración
ax2.set_xlabel('Categories ranked by frequency (cumulative)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative frequency [%]', fontsize=12, fontweight='bold')
ax2.set_title('b) Cumulative coverage by ranked categories', fontsize=14, fontweight='bold')

ax2.set_xlim(-0.5, len(df_show)-0.5)
ax2.set_ylim(0, 105)

# Grid y estilo
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', direction='in', labelsize=11)

# ELIMINAR los xticks números - solo mostrar etiquetas de categorías
ax2.set_xticks([])  # Elimina los números del eje X
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
plt.tight_layout()
plt.savefig(bd_out_fig+'Stats_SpatCat_CDF.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Print summary using dataframes
print(f"Total spatial categories: {len(df_cats)}")
print(f"Categories above 25th percentile: {len(df_filt)}")
print(f"Area coverage by filtered categories: {df_filt['rel_freq'].sum():.1f}%")
print(f"25th percentile threshold: {p25:.3f}%")
