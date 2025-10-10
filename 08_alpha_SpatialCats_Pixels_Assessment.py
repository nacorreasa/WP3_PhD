import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, AutoMinorLocator, FuncFormatter, NullLocator

"""
Code for analisying the statistisc of the spatial categories derived from the combination
og the B-KG climate zones, the Z0 CMCC, the variance slope  layers in the ALP3 domain. 
It starts by a generneral description of the relative frequencies of all cathegories to
then filter those whuch relative frequency are under the p25 ( lower outliers in a IQR 
point of view). However given that we could be loosing some inormacion for those
low-frequency categories, the script also conduncst an co-ocurrence analysis of the sum 
of those low frecuencie among categories in order to dectect patterns.  Finally, it generates
stacked bar charts on the most dominant spatial categories to show how they are distributed. 

Author : Nathalia Correa-Sánchez
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

file_cats  = "/Dati/Outputs/Climate_Provinces/CSVs/Combination_RIX.csv"
bd_out_fig = "/Dati/Outputs/Plots/WP3_development/"

########################################################################################
##--------------------READING PANDAS DATAFRAME WITH THE INFORMAITON-------------------##
########################################################################################

df_cats = pd.read_csv (file_cats)

########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################

def mapear_valor(valor):
    try:
        str_valor = str(int(valor))
        return {
            "clasificacion_climatica": int(str_valor[0]),
            "rugosidad": int(str_valor[1]),
            "varianza_pendiente": int(str_valor[2]),
        }
    except (ValueError, IndexError):
        print(f"Advertencia: Valor inválido '{valor}'. Se omitirá este valor.")
        return None

def style_axis(ax):
    """
    Function to set the format to the plots
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator()) ## No en este script de categorias discretas
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def custom_KGclimate_colormap(level):
    arids_color      = [255/255, 100/255, 0/255]    # Naranja intenso
    temperates_color = [100/255, 200/255, 100/255]  # Verde suave
    cold_color       = [50/255, 100/255, 255/255]   # Azul medio
    polar_color      = [150/255, 150/255, 150/255]  # Gris medio

    if level == 1:  # Arids
        return arids_color
    elif level == 2:  # Temperates
        return temperates_color
    elif level == 3:  # Cold
        return cold_color
    elif level == 4:  # Polar
        return polar_color
    else:
        return [0, 0, 0]  # Negro por defecto para niveles no definidos

def adjust_color_intensity(cmap, index, max_levels, min_intensity=0.3):
    """
    Ajusta la intensidad del color para mejorar visibilidad.
    Para aumentar intensidad/contraste.
    
    INPUTS :
    - cmap: Mapa de colores
    - index: Nivel actual (1-based)
    - max_levels: Número máximo de niveles
    - min_intensity: Intensidad mínima para el primer nivel
    
    OUTPUTS :
    - Color ajustado
    """
    normalized_index = (index - 1) / (max_levels - 1)
    color = cmap(normalized_index)
    
    # Asegura que el primer nivel no sea muy claro
    if index == 1:
        color = list(color)
        color = [max(c, min_intensity) for c in color[:3]] + [color[3]]
    
    return color
 

def decompose_spatial_category(category_value):
    """
    Decomposes the 3-digit categorical value into its components.
    
    INPUTS :
        category_value (int): Spatial category code (e.g., 332)
    
    OUTPUTS :
        dict: Decomposed components of the category
    """
    str_category = str(category_value).zfill(3)
    return {
        'climate': {
            'code': int(str_category[0]),
            'description': {
                1: 'Arid', 
                2: 'Temperate', 
                3: 'Cold', 
                4: 'Polar'
            }.get(int(str_category[0]), 'Unknown')
        },
        'roughness': {
            'code': int(str_category[1]),
            'description': {
                1: 'Very Low', 
                2: 'Low', 
                3: 'Moderate', 
                4: 'High', 
                5: 'Very High'
            }.get(int(str_category[1]), 'Unknown')
        },
        'slope_variance': {
            'code': int(str_category[2]),
            'description': {
                1: 'Very Low', 
                2: 'Low', 
                3: 'Moderate', 
                4: 'High', 
                5: 'Very High'
            }.get(int(str_category[2]), 'Unknown')
        }
    }

def analyze_low_frequency_categories(df_categories, frequency_threshold):
    """
    Analyzes correlations and patterns in low-frequency categories.
    
    INPUTS:
        df_categories (pd.DataFrame): DataFrame with categories
        frequency_threshold (float): Frequency threshold to consider 'low frequency'
    
    OUTPUTS:
        dict: Results of the pattern analysis
    """
    df_low = df_categories[df_categories['rel_freq'] < frequency_threshold].copy()
    df_low['components'] = df_low['value'].apply(decompose_spatial_category)
    df_low['climate'] = df_low['components'].apply(lambda x: x['climate']['description'])
    df_low['roughness'] = df_low['components'].apply(lambda x: x['roughness']['description'])
    df_low['variance'] = df_low['components'].apply(lambda x: x['slope_variance']['description'])

    return df_low

def create_cooccurrence_matrix(df, x_col, y_col, orden_x, orden_y):
    """
    Crea la matriz de co-ocurrencia.
    """
    matrix = np.zeros((len(orden_x), len(orden_y)), dtype=int)
    for i, x in enumerate(orden_x):
        for j, y in enumerate(orden_y):
            count = len(df[(df[x_col] == x) & (df[y_col] == y)])
            matrix[i, j] = count
    return matrix

def plot_heatmap(ax, matrix, orden_x, orden_y, title, xlabel, ylabel, cmap_name, num_bins=10):  # Añadido num_bins
    """
    Crea el heatmap con ordenamiento correcto, barra de color discreta y etiquetas enteras.
    """

    # Crear colormap discreto
    cmap = ListedColormap(plt.cm.get_cmap(cmap_name)(np.linspace(0, 1, num_bins)))  # num_bins controla la granularidad

    im = ax.imshow(matrix, cmap=cmap, aspect='equal')
    ax.set_xticks(np.arange(len(orden_y)))
    ax.set_yticks(np.arange(len(orden_x)))
    ax.set_xticklabels(orden_y)
    ax.set_yticklabels(orden_x)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Añadir barra de color discreta
    cbar = ax.figure.colorbar(im, ax=ax, ticks=np.arange(0, np.max(matrix) + 1, step=1))  # Etiquetas enteras
    cbar.ax.tick_params(labelsize=9)

    # Anotaciones (enteros)
    for i in range(len(orden_x)):
        for j in range(len(orden_y)):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w", fontsize=9)

    bbox = cbar.ax.get_position()  # cbar.ax.set_position([x0, y0, width, height])
    cbar.ax.set_position([bbox.x0 + bbox.width * 0.1, 0.25, bbox.width * 0.8, 0.85])  # Ejemplo de ajuste


    return im


def plot_stacked_bar_chart(ax, data, title, colors, ylabel):
    """
    Función para crear un gráfico de barras apiladas sin leyenda (la leyenda se añadirá de forma global).
    Grafica el DataFrame de las categorias separadas; las filas (índice) se usarán como etiquetas del eje x.
    """
    data.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='none')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=11, which='both', direction='in')
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    sns.despine(ax=ax, top=True, right=True)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

########################################################################################
##-----------DEFINING DICTIONARIES PER CATEGORIES FOR MAPPING AND PLOTTING------------##
########################################################################################

climate_levels = {  1: "Arids",
                    2: "Temperates", 
                    3: "Cold",
                    4: "Polar"}

roughness_levels = {    1: "Very low",
                        2: "Low", 
                        3: "Moderate",
                        4: "High",
                        5: "Very high"}

slope_variance_levels = {   1: "Very low",
                            2: "Low", 
                            3: "Moderate",
                            4: "High",
                            5: "Very high"}

########################################################################################
##----------------CALCULATING COVER PERCENTAGE OF THE TOTAL CATEGORIES----------------##
########################################################################################

total_count         = df_cats["count"].sum()
df_cats["rel_freq"] = (df_cats["count"] / total_count) * 100  

df_cats["componentes"] = df_cats["value"].apply(mapear_valor)
df_cats                = df_cats.dropna(subset=["componentes"])

width = 0.45
x     = np.arange(len(df_cats))

cmap_rugosidad = plt.cm.YlOrBr   # Amarillo-Naranja-Marrón
cmap_varianza  = plt.cm.YlGnBu   # Amarillo-Verde-Azul

fig, ax = plt.subplots(figsize=(14, 7))
style_axis(ax)

# Diccionarios para guardar handles y labels únicos y en orden
handles = {"KG-Climate": [],
           "Roughness": [],
           "Slope variance": []}
labels = {"KG-Climate": [],
          "Roughness": [],
          "Slope variance": []}

# Crear barras para cada nivel de cada capa
for i, row in df_cats.iterrows():
    climatic            = row["componentes"]["clasificacion_climatica"]
    rugosidad           = row["componentes"]["rugosidad"]
    varianza            = row["componentes"]["varianza_pendiente"]
    frecuencia_relativa = row["rel_freq"]
    value               = row["value"]

    # Obtener etiquetas usando los diccionarios de mapeo
    clim    = climate_levels.get(climatic, f"Unknown ({climatic})")
    rough   = roughness_levels.get(rugosidad, f"Unknown ({rugosidad})")
    varslop = slope_variance_levels.get(varianza, f"Unknown ({varianza})")

    # Escalar colores para el degradado (min=1, max=5)
    color_climatic  = custom_KGclimate_colormap(climatic)
    color_rugosidad = adjust_color_intensity(cmap_rugosidad, rugosidad, 5)
    color_varianza  = adjust_color_intensity(cmap_varianza, varianza, 5)
    # color_rugosidad = cmap_rugosidad((rugosidad - 1) / 4)              
    # color_varianza  = cmap_varianza((varianza - 1) / 4)   

    # Barras
    ax.bar(x[i] - width, frecuencia_relativa, width, label=clim, color=color_climatic)
    ax.bar(x[i], frecuencia_relativa, width, label=rough, color=color_rugosidad)
    ax.bar(x[i] + width, frecuencia_relativa, width, label=varslop, color=color_varianza)

    # Guardar handles y labels únicos
    if clim not in labels["KG-Climate"]:
        handles["KG-Climate"].append(plt.Rectangle((0,0),1,1, color=color_climatic))
        labels["KG-Climate"].append(clim)
    
    if rough not in labels["Roughness"]:
        handles["Roughness"].append(plt.Rectangle((0,0),1,1, color=color_rugosidad))
        labels["Roughness"].append(rough)
    
    if varslop not in labels["Slope variance"]:
        handles["Slope variance"].append(plt.Rectangle((0,0),1,1, color=color_varianza))
        labels["Slope variance"].append(varslop)
        
ax.set_xticks(x)
ax.set_xticklabels(df_cats["value"].astype(int), rotation=90, ha="right")  
plt.margins(x=0.01)  # Un margen muy pequeño

ax.set_ylabel("Relative frequency [%]", fontsize=12)  
ax.set_xlabel("Spatial category", fontsize=12) 
ax.set_title("Relative frequency of each spatial category \n (Total = "+str(len(df_cats))+")", fontsize=13, fontweight="bold")

# Preparar leyenda con subtítulos
handles_list = [] 
labels_list  = [] 

# Añadir sección de KG-Climate
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append(r'$\mathbf{KG-Climate\ Levels}$')

handles_list.extend(handles["KG-Climate"])
labels_list.extend(labels["KG-Climate"])

# Separador
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append('')

# Añadir sección de Roughness
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append(r'$\mathbf{Roughness\ Levels}$')

handles_list.extend(handles["Roughness"])
labels_list.extend(labels["Roughness"])

# Separador
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append('')

# Añadir sección de Slope Variance
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append(r'$\mathbf{Slope\ Variance\ Levels}$')

handles_list.extend(handles["Slope variance"])
labels_list.extend(labels["Slope variance"])

# Leyenda con subtítulos
ax.legend(handles_list, labels_list, bbox_to_anchor=(0.84, 0.93), loc='upper left',)
plt.tight_layout()
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right=0.8, bottom=0.15) 
plt.savefig(bd_out_fig+"FreqBars_RIXCats_Total.png", format='png', dpi=300, transparent=True)
plt.show()


########################################################################################
##--------FILTERING CATEGORIES UNDER THE 25% PERCENTILE: LOW ATYPICAL VALUES----------##
########################################################################################

percentages = df_cats.rel_freq.values * 100
p25         = np.percentile(percentages, 25)
df_filt     = df_cats[df_cats.rel_freq >= p25]
df_filt     = df_filt.reset_index(drop=True) ## Resetea el indice que se habia dñado luego del filtrado. 

width = 0.30
x     = np.arange(len(df_filt))

cmap_rugosidad = plt.cm.YlOrBr   # Amarillo-Naranja-Marrón
cmap_varianza  = plt.cm.YlGnBu   # Amarillo-Verde-Azul

fig, ax = plt.subplots(figsize=(11, 6))
style_axis(ax)

# Diccionarios para guardar handles y labels únicos y en orden
handles = {"KG-Climate": [],
           "Roughness": [],
           "Slope variance": []}
labels = {"KG-Climate": [],
          "Roughness": [],
          "Slope variance": []}

# Crear barras para cada nivel de cada capa
for i, row in df_filt.iterrows():
    climatic            = row["componentes"]["clasificacion_climatica"]
    rugosidad           = row["componentes"]["rugosidad"]
    varianza            = row["componentes"]["varianza_pendiente"]
    frecuencia_relativa = row["rel_freq"]
    value               = row["value"]

    # Obtener etiquetas usando los diccionarios de mapeo
    clim    = climate_levels.get(climatic, f"Unknown ({climatic})")
    rough   = roughness_levels.get(rugosidad, f"Unknown ({rugosidad})")
    varslop = slope_variance_levels.get(varianza, f"Unknown ({varianza})")

    # Escalar colores para el degradado (min=1, max=5)
    color_climatic  = custom_KGclimate_colormap(climatic)
    color_rugosidad = adjust_color_intensity(cmap_rugosidad, rugosidad, 5)
    color_varianza  = adjust_color_intensity(cmap_varianza, varianza, 5)
    # color_rugosidad = cmap_rugosidad((rugosidad - 1) / 4)              
    # color_varianza  = cmap_varianza((varianza - 1) / 4)   

    # Barras
    ax.bar(x[i] - width, frecuencia_relativa, width, label=clim, color=color_climatic)
    ax.bar(x[i], frecuencia_relativa, width, label=rough, color=color_rugosidad)
    ax.bar(x[i] + width, frecuencia_relativa, width, label=varslop, color=color_varianza)

    # Guardar handles y labels únicos
    if clim not in labels["KG-Climate"]:
        handles["KG-Climate"].append(plt.Rectangle((0,0),1,1, color=color_climatic))
        labels["KG-Climate"].append(clim)
    
    if rough not in labels["Roughness"]:
        handles["Roughness"].append(plt.Rectangle((0,0),1,1, color=color_rugosidad))
        labels["Roughness"].append(rough)
    
    if varslop not in labels["Slope variance"]:
        handles["Slope variance"].append(plt.Rectangle((0,0),1,1, color=color_varianza))
        labels["Slope variance"].append(varslop)
        
ax.set_xticks(x)
ax.set_xticklabels(df_filt["value"].astype(int), rotation=0, ha="right")  
plt.margins(x=0.01)  # Un margen muy pequeño

ax.set_ylabel("Relative frequency [%]", fontsize=12)  
ax.set_xlabel("Spatial category", fontsize=12) 
ax.set_title("Relative frequency of each spatial category \n (Total : "+str(len(df_filt))+" cats ∼ "+str(round(df_filt.rel_freq .values.sum(), 2))+"%)", fontsize=13, fontweight="bold")

# Preparar leyenda con subtítulos
handles_list = [] 
labels_list  = [] 

# Añadir sección de KG-Climate
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append(r'$\mathbf{KG-Climate\ Levels}$')

handles_list.extend(handles["KG-Climate"])
labels_list.extend(labels["KG-Climate"])

# Separador
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append('')

# Añadir sección de Roughness
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append(r'$\mathbf{Roughness\ Levels}$')

handles_list.extend(handles["Roughness"])
labels_list.extend(labels["Roughness"])

# Separador
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append('')

# Añadir sección de Slope Variance
handles_list.append(plt.Rectangle((0,0),1,1, color='none', alpha=0))
labels_list.append(r'$\mathbf{Slope\ Variance\ Levels}$')

handles_list.extend(handles["Slope variance"])
labels_list.extend(labels["Slope variance"])

# Leyenda con subtítulos
ax.legend(handles_list, labels_list, bbox_to_anchor=(0.84, 0.93), loc='upper left',)
plt.tight_layout()
plt.subplots_adjust(wspace=0.30, hspace=0.4, left=0.10, right=0.8, bottom=0.15) 
plt.savefig(bd_out_fig+"FreqBars_RIXCats_Filtered.png", format='png', dpi=300, transparent=True)
plt.show()

########################################################################################
##-------CO-OCCURENCE CATEGORIES UNDER THE 25% PERCENTILE: LOW ATYPICAL VALUES--------##
########################################################################################

climate_orden   = ['Arid', 'Temperate', 'Cold', 'Polar']
roughness_orden = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
variance_orden  = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

df_low = analyze_low_frequency_categories(df_cats, frequency_threshold=p25)

co_climate_roughness_matrix  = create_cooccurrence_matrix(df_low, 'climate', 'roughness', climate_orden, roughness_orden)
co_climate_variance_matrix   = create_cooccurrence_matrix(df_low, 'climate', 'variance', climate_orden, variance_orden)
co_roughness_variance_matrix = create_cooccurrence_matrix(df_low, 'roughness', 'variance', roughness_orden, variance_orden)


plt.figure(figsize=(15, 5))

ax1 = plt.subplot(131)
plot_heatmap(ax1, co_climate_roughness_matrix, climate_orden, roughness_orden,
             'Climate-Roughness\nCo-occurrence', 'Roughness', 'Climate', 'YlGnBu')

ax2 = plt.subplot(132)
plot_heatmap(ax2, co_climate_variance_matrix, climate_orden, variance_orden,
             'Climate-Slope Variance\nCo-occurrence', 'Slope Variance', 'Climate', 'YlOrRd')

ax3 = plt.subplot(133)
plot_heatmap(ax3, co_roughness_variance_matrix, roughness_orden, variance_orden,
             'Roughness-Slope Variance\nCo-occurrence', 'Slope Variance', 'Roughness', 'PuRd')

plt.tight_layout()
plt.subplots_adjust(wspace=0.40, hspace=0.4, left=0.10, right=0.9, bottom=0.25)
plt.savefig(bd_out_fig+"Co-Ocurrence_Heatmaps_LowfreqCats.png", format='png', dpi=300, transparent=True)
plt.show()

########################################################################################
##-------------------STAKED BAR PLOT FOR PREDONMINANT CATEGORIES----------------------##
########################################################################################
use_rel_freq = False  # Cambia a False para usar conteos

if use_rel_freq:
    df_grouped_climate_counts = df_filt.groupby(['climate', 'value'])['count'].sum().unstack(fill_value=0)
    df_grouped_climate        = df_grouped_climate_counts.div(df_grouped_climate_counts.sum(axis=1), axis=0)
    df_grouped_climate        = df_grouped_climate.reindex(climate_orden, fill_value=0)

    df_grouped_roughness_counts = df_filt.groupby(['roughness', 'value'])['count'].sum().unstack(fill_value=0)
    df_grouped_roughness        = df_grouped_roughness_counts.div(df_grouped_roughness_counts.sum(axis=1), axis=0)
    df_grouped_roughness        = df_grouped_roughness.reindex(roughness_orden, fill_value=0)
    
    df_grouped_variance_counts = df_filt.groupby(['slope_variance', 'value'])['count'].sum().unstack(fill_value=0)
    df_grouped_variance        = df_grouped_variance_counts.div(df_grouped_variance_counts.sum(axis=1), axis=0)
    df_grouped_variance        = df_grouped_variance.reindex(variance_orden, fill_value=0)
else:
    # Usar conteos directamente
    df_grouped_climate = df_filt.groupby(['climate', 'value'])['count'].sum().unstack(fill_value=0)
    df_grouped_climate = df_grouped_climate.reindex(climate_orden, fill_value=0)
    
    df_grouped_roughness = df_filt.groupby(['roughness', 'value'])['count'].sum().unstack(fill_value=0)
    df_grouped_roughness = df_grouped_roughness.reindex(roughness_orden, fill_value=0)
    
    df_grouped_variance = df_filt.groupby(['slope_variance', 'value'])['count'].sum().unstack(fill_value=0)
    df_grouped_variance = df_grouped_variance.reindex(variance_orden, fill_value=0)

value_labels = sorted(df_filt['value'].unique().astype(str))
colors       = sns.color_palette("husl", len(value_labels)) # Paleta de colores personalizada: utilizar "husl" para garantizar colores diferentes sin repetir tonalidades.

fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
plot_stacked_bar_chart(axes[0], df_grouped_climate, "Distribution of Pixels by Climate", colors, "Pixel Count" if not use_rel_freq else "Relative Frequency")
plot_stacked_bar_chart(axes[1], df_grouped_roughness, "Distribution of Pixels by Roughness", colors, "Pixel Count" if not use_rel_freq else "Relative Frequency")
plot_stacked_bar_chart(axes[2], df_grouped_variance, "Distribution of Pixels by Slope Variance", colors, "Pixel Count" if not use_rel_freq else "Relative Frequency")
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.10, right=0.88, bottom=0.10, top=0.89)
legend_handles = [Patch(facecolor=colors[i], label=value_labels[i]) for i in range(len(value_labels))]
fig.legend(handles=legend_handles, title="Spatial Category", fontsize=11, title_fontsize=11, bbox_to_anchor=(1.0, 0.68), loc='upper right')
plt.savefig(bd_out_fig+"SpatialCats_StackBars_HighFreqCats.png", format='png', dpi=300, transparent=True)
plt.show()

























