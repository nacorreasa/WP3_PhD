
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from collections import defaultdict
import os
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import rasterio


"""
Code to conduct the Principal Components Analysis from all those samples, in order to compare those
performance metrics across the spatial categories.

Spatial categories whose relative frequency exceeded the first quartile (25th percentile) 
selected in the precedant script. Series pixels follow a startified equivallent random sampling.

It also conducts and visualizes the divergence (CP2) accross categories. 

The updated routine uses the cropped CORDEX domain. 

Conducts an analysis where :
- the number of variables (models) is small = 3
- We are interested in comparing behaviour between spatial categories. 
- The temporal variability is high
Thus, the analysis of loadings and their patterns tends to be more revealing than the analysis of scores. 
This loadings-focused perspective has allowed you to identify subtle but systematic patterns in how models 
diverge in different spatial categories, revealing valuable information that conventional score-based 
analysis might have missed.

Author : Nathalia Correa-Sánchez
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

bd_out_tc    = "/Dati/Outputs/WP3_SamplingSeries_CPM/"
bd_out_fig   = "/Dati/Outputs/Plots/WP3_development/"
bd_out_pca   = "/Dati/Outputs/PCA_Analysis/"
bd_in_raster = "/Dati/Outputs/Climate_Provinces/Development_Rasters/FinalRasters_In-Out/"  # Antes : Combined_RIX_remCPM_WGS84.tif


########################################################################################
##------------------------------DEFINNING RELEVANT INPUTS-----------------------------##
########################################################################################

filas_eliminar    = [0]  # Primera  fila, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada 
columnas_eliminar = [0]  # Primera columna, para ajuste de CNRM en todos los 2D array o xarrays con datos de entrada
N_pixels          = 100  # Cantidad de puntos en cada muestra

########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################

def decompose_spatial_category(category_value):
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

def interpret_pc3_pattern(eth_sign, cnrm_sign, cmcc_sign):
    """
    Interpreta el patrón de signos de loadings de PC3.
    
    Args:
        eth_sign, cnrm_sign, cmcc_sign: Signos de los loadings de cada modelo
        
    Returns:
        Descripción interpretativa del patrón
    """
    if eth_sign == cnrm_sign == cmcc_sign:
        return "Todos los modelos divergen en la misma dirección"
    
    if eth_sign == cnrm_sign and eth_sign != cmcc_sign:
        return "ETH y CNRM divergen juntos, opuestos a CMCC"
    
    if eth_sign == cmcc_sign and eth_sign != cnrm_sign:
        return "ETH y CMCC divergen juntos, opuestos a CNRM"
    
    if cnrm_sign == cmcc_sign and eth_sign != cnrm_sign:
        return "CNRM y CMCC divergen juntos, opuestos a ETH"
    
    return "Patrón complejo sin agrupación clara"

########################################################################################
##-----ABRIENDO EL CROPPED RASTER PARA EXTRAER CADA CLASE & AJUSTANDO EL DATAFRAME----##
########################################################################################

comblay              = rasterio.open(bd_in_raster+"SEA-LANDCropped_Combined_RIX_remCPM_WGS84.tif")
band1_o              = comblay.read(1) ## Solo tiene una banda
band1_o[band1_o < 0] = np.nan          ## Reemplazando los negativos con nan o 0(Tener en cuenta NoData= -3.40282e+38) 
band1                = np.delete(np.delete(band1_o, filas_eliminar[0], axis=0), columnas_eliminar[0], axis=1) ## Ajustillo para los xarrays

# Obteniendo valores unicos de las categorias
unique_vals    = np.unique(band1)
unique_vals    = unique_vals[np.isfinite(unique_vals)]
num_categories = len(unique_vals)

# Contar píxeles para cada valor único
pixel_counts = {}
mask         = np.isfinite(band1_o)
valid_values = band1_o[mask]

if np.issubdtype(valid_values.dtype, np.integer): # Asegurarse de que los valores son enteros para np.bincount
    counts = np.bincount(valid_values.astype(int))
    for val in unique_vals:
        pixel_counts[val] = counts[int(val)] if int(val) < len(counts) else 0
else:
    for val in unique_vals:
        pixel_counts[val] = np.sum(valid_values == val)
counts_array = np.array([pixel_counts[val] for val in unique_vals])
df_cats      = pd.DataFrame({'value': unique_vals, 'count': counts_array,})

########################################################################################
##--------FILTERING CATEGORIES UNDER THE 25% PERCENTILE: LOW ATYPICAL VALUES----------##
########################################################################################

total_count         = df_cats["count"].sum()
df_cats["rel_freq"] = (df_cats["count"] / total_count) * 100  

percentages = df_cats.rel_freq.values * 100
p25         = np.percentile(percentages, 25)
df_filt     = df_cats[df_cats.rel_freq >= p25]
df_filt     = df_filt.reset_index(drop=True) ## Resetea el indice que se habia dñado luego del filtrado. 
fl_cats     = df_filt['value'].values.astype(int)

########################################################################################
###------------------NIVEL 1: ANÁLISIS POR PUNTO INDIVIDUAL--------------------------###
########################################################################################
"""
 Necesitamos analizar el comportamiento de los tres modelos en cada ubicación específica 
 para entender sus patrones de similitud y diferencia a nivel local.
 Para comparar cómo los tres modelos representan la velocidad del viento en exactamente
 la misma ubicación geográfica.

 Los wind gusts (máximos diarios) son más relevantes para elobjetivo de evaluar eventos 
 extremos y reducen el ruido en los datos. Para analizar la variable que realmente nos 
 interesa: los vientos máximos diarios, que son críticos para aplicaciones de riesgo 
 y seguridad.
"""


# Diccionario para almacenar resultados por categoría
all_pca_results = {}
full_range       = pd.date_range(start='2000-01-01 00:00:00', end='2009-12-31 23:50:00', freq='1H')

# Procesar cada categoría
for cat in fl_cats:
    print(f"## Processing category {cat} for PCA analysis")
    
    # Cargar datos
    data_npz    = np.load(f"{bd_out_tc}_TS_cat_{cat}.npz")
    time_series = data_npz['time_series']
    coordinates = data_npz['coordinates']
    models      = data_npz['models']
    
    # Inicializar contenedor para resultados de esta categoría
    all_pca_results[cat] = []
    
    # Procesar cada punto
    for k in range(len(coordinates)):
        # Extraer series temporales
        serie_eth  = time_series[k, 0, :]        
        serie_cnrm = time_series[k, 1, :]
        serie_cmcc = time_series[k, 2, :]

        # Convertir a DataFrame
        df_s_eth  = pd.DataFrame(serie_eth, index=full_range, columns=["WS_h"])
        df_s_cnrm = pd.DataFrame(serie_cnrm, index=full_range, columns=["WS_h"]) 
        df_s_cmcc = pd.DataFrame(serie_cmcc, index=full_range, columns=["WS_h"])

        # Obtener wind gusts diarios
        wg_s_eth  = df_s_eth.groupby(df_s_eth.index.date)["WS_h"].max()
        wg_s_cnrm = df_s_cnrm.groupby(df_s_cnrm.index.date)["WS_h"].max()
        wg_s_cmcc = df_s_cmcc.groupby(df_s_cmcc.index.date)["WS_h"].max()

        # Transformación logarítmica --> Evaluar si si es necesaria para mejorar las diferencias relativas en este tipo de datos sezgados. 
        log_wg_s_eth  = np.log1p(wg_s_eth)
        log_wg_s_cnrm = np.log1p(wg_s_cnrm)
        log_wg_s_cmcc = np.log1p(wg_s_cmcc)
        
        # Crear matriz para PCA - cada fila es un día y cada columna es un modelo
        X = np.column_stack([log_wg_s_eth, log_wg_s_cnrm, log_wg_s_cmcc])
        
        # Manejo de NaN - PCA requiere una matriz completa sin valores faltantes
        mask    = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        
        # Verificar si hay suficientes datos(aunque por como se extraen las series de tiempo siempre hay)
        if len(X_clean) < 30:  # Umbral mínimo
            print(f"  Warning: Point {k} has insufficient data ({len(X_clean)} days)")
            continue
        
        # Centrar y escalar datos -cada columna (modelo) restando su media y dividiendo por su desviación estándar.
        # Esto porque PCA es sensible a la escala de las variables. La estandarización asegura que cada modelo 
        # contribuya por igual al análisis, independientemente de su magnitud. Para evitar que un modelo domine 
        # el análisis simplemente porque tiene valores numéricamente mayores, garantizando un análisis justo 
        # de la estructura de covarianza.
        X_norm = (X_clean - X_clean.mean(axis=0)) / X_clean.std(axis=0)
        
        # Aplicar PCA: PCA identifica direcciones de máxima varianza en los datos, encontrando combinaciones 
        # lineales de los modelos originales que explican la mayor cantidad de variabilidad
        pca = PCA()
        pca.fit(X_norm)
        
        # Extraer métricas principales
        explained_variance = pca.explained_variance_ratio_
        loadings           = pca.components_
        scores             = pca.transform(X_norm)
        
        # Identificar percentiles altos (para análisis de extremos)
        p95_mask = scores[:, 0] > np.percentile(scores[:, 0], 95)
        
        # Calcular métricas específicas
        point_results = {
            'point_idx': k,
            'coordinates': coordinates[k],
            'explained_variance': explained_variance,
            'loadings': loadings,
            'pc1_var': explained_variance[0],
            'pc2_var': explained_variance[1] if len(explained_variance) > 1 else 0,
            'pc3_var': explained_variance[2] if len(explained_variance) > 2 else 0,
            'loading_pattern': { 'pc1': {model: loading for model, loading in zip(['ETH', 'CNRM', 'CMCC'], loadings[0])},
                                 'pc2': {model: loading for model, loading in zip(['ETH', 'CNRM', 'CMCC'], loadings[1])} if len(loadings) > 1 else None },
            'extreme_pc2_std': np.std(scores[p95_mask, 1]) if len(loadings) > 1 and np.sum(p95_mask) > 0 else None,
            'model_agreement_index': explained_variance[0] / (explained_variance[0] + explained_variance[1]) if len(explained_variance) > 1 else 1.0,
            'divergence_pattern'   : None  # Se asignará más adelante
        }
        
        # Determinar patrón de divergencia (qué modelo se diferencia más)
        if len(loadings) > 1:
            pc2_loadings = loadings[1] # Corresponde al PC2
            # Encontrar el modelo con la carga absoluta más alta en PC2
            max_index    = np.argmax(np.abs(pc2_loadings))
            # Determinar si este modelo tiene carga positiva o negativa (esta sobre estimando o subestimando)
            sign         = np.sign(pc2_loadings[max_index])
            # Asignar un patrón descriptivo basado en el signo del load (guardarlo de foorma que se entienda la divergencia)
            if max_index == 0:
                point_results['divergence_pattern'] = 'ETH_different' if sign > 0 else 'ETH_opposite'
            elif max_index == 1:
                point_results['divergence_pattern'] = 'CNRM_different' if sign > 0 else 'CNRM_opposite'
            elif max_index == 2:
                point_results['divergence_pattern'] = 'CMCC_different' if sign > 0 else 'CMCC_opposite'
        
        # Guardar resultados
        all_pca_results[cat].append(point_results)
    
    print(f"## Finished PCA analysis for category {cat} with {len(all_pca_results[cat])} valid points")

# # Guardar resultados completos
# np.save(f"{bd_out_pca}AllPoints_PCA_results.npy", all_pca_results)

#Leer resultados completos
all_pca_results = np.load(f"{bd_out_pca}AllPoints_PCA_results.npy", allow_pickle=True).item()

########################################################################################
###---------------NIVEL 2: ANÁLISIS POR CATEGORÍA ESPACIAL - PROMEDIOS---------------###
########################################################################################

category_metrics = {}

for cat in fl_cats:
    if cat not in all_pca_results or not all_pca_results[cat]:
        print(f"No results for category {cat}, skipping") ## No es probable que esto pase
        continue
    
    # Extraer resultados para esta categoría
    cat_results = all_pca_results[cat]
    
    # Calcular métricas agregadas
    pc1_values        = [res['pc1_var'] for res in cat_results]
    pc2_values        = [res['pc2_var'] for res in cat_results if 'pc2_var' in res]
    agreement_indices = [res['model_agreement_index'] for res in cat_results if 'model_agreement_index' in res]
    
    # Contar patrones de divergencia
    divergence_patterns = [res['divergence_pattern'] for res in cat_results if res['divergence_pattern'] is not None]
    pattern_counts      = {}
    for pattern in ['ETH_different', 'ETH_opposite', 'CNRM_different', 'CNRM_opposite', 'CMCC_different', 'CMCC_opposite']:
        pattern_counts[pattern] = divergence_patterns.count(pattern)
    
    # Calcular promedio de cargas para PC1 y PC2
    pc1_loadings_eth  = [res['loading_pattern']['pc1']['ETH'] for res in cat_results if 'loading_pattern' in res]
    pc1_loadings_cnrm = [res['loading_pattern']['pc1']['CNRM'] for res in cat_results if 'loading_pattern' in res]
    pc1_loadings_cmcc = [res['loading_pattern']['pc1']['CMCC'] for res in cat_results if 'loading_pattern' in res]
    
    pc2_loadings = {}
    for model in ['ETH', 'CNRM', 'CMCC']:
        pc2_loadings[model] = [res['loading_pattern']['pc2'][model]  for res in cat_results  if 'loading_pattern' in res and res['loading_pattern']['pc2'] is not None]
    
    # Métricas para extremos - No la uso
    extreme_metrics = [res['extreme_pc2_std'] for res in cat_results if res['extreme_pc2_std'] is not None]
    
    # Descomponer categoría para análisis
    cat_details = decompose_spatial_category(cat)
    
    # Guardar métricas ene l diccionario
    category_metrics[cat] = {
        'cat_code'                : cat,
        'climate'                 : cat_details['climate']['description'],
        'climate_code'            : cat_details['climate']['code'],
        'roughness'               : cat_details['roughness']['description'],
        'roughness_code'          : cat_details['roughness']['code'],
        'slope_variance'          : cat_details['slope_variance']['description'],
        'slope_code'              : cat_details['slope_variance']['code'],
        'n_points'                : len(cat_results),
        'pc1_mean'                : np.mean(pc1_values),
        'pc1_std'                 : np.std(pc1_values),
        'pc2_mean'                : np.mean(pc2_values) if pc2_values else np.nan,
        'pc2_std'                 : np.std(pc2_values) if pc2_values else np.nan,
        'agreement_index_mean'    : np.mean(agreement_indices) if agreement_indices else np.nan,
        'agreement_index_std'     : np.std(agreement_indices) if agreement_indices else np.nan,
        'divergence_patterns'     : pattern_counts,
        'dominant_pattern'        : max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None,
        'extreme_discrepancy_mean': np.mean(extreme_metrics) if extreme_metrics else np.nan,
        'pc1_loadings_mean': {'ETH': np.mean(pc1_loadings_eth) if pc1_loadings_eth else np.nan,
                             'CNRM': np.mean(pc1_loadings_cnrm) if pc1_loadings_cnrm else np.nan,
                             'CMCC': np.mean(pc1_loadings_cmcc) if pc1_loadings_cmcc else np.nan },
        'pc2_loadings_mean': {'ETH': np.mean(pc2_loadings['ETH']) if pc2_loadings['ETH'] else np.nan,
                             'CNRM': np.mean(pc2_loadings['CNRM']) if pc2_loadings['CNRM'] else np.nan,
                             'CMCC': np.mean(pc2_loadings['CMCC']) if pc2_loadings['CMCC'] else np.nan }
    }
# Convertir el diccionario a DataFrame para análisis más sencillo
df_categories = pd.DataFrame.from_dict(category_metrics, orient='index')

# # Guardar resultados
# df_categories.to_csv(f"{bd_out_pca}PCA_Category_metrics.csv")


df_categories = pd.read_csv(f"{bd_out_pca}PCA_Category_metrics.csv")
########################################################################################
###------------------------PLOTTING AVERAGED EXPLAINED VARIANCE----------------------###
########################################################################################

# Preparar datos de varianza explicada para cada categoría y calcular valores promedio para esta categoría
variance_data = []
for cat in df_categories.index:
    pc1_var = df_categories.loc[cat, 'pc1_mean']
    pc2_var = df_categories.loc[cat, 'pc2_mean']
    pc3_var = df_categories.loc[cat, 'pc3_mean'] if 'pc3_mean' in df_categories.columns else 1 - pc1_var - pc2_var

    variance_data.append({ 'Category': cat,
                            'PC1': pc1_var * 100,  # Convertir a porcentaje
                            'PC2': pc2_var * 100,
                            'PC3': pc3_var * 100 })

df_variance = pd.DataFrame(variance_data)
avg_pc1     = df_variance['PC1'].mean()
avg_pc2     = df_variance['PC2'].mean()
avg_pc3     = df_variance['PC3'].mean()

colors = ['#3498db', '#e74c3c', '#2ecc71'] # Colores para componentes

# Definir etiquetas simplificadas para categorías climáticas, rugosidad y topografía
# Mapeo de códigos de categoría a etiquetas descriptivas
climate_labels = ['Ar', 'Tm', 'Co', 'Td'] # Las otras se definen abajo
cat_labels     = {}
for cat in fl_cats:
    cat_str = str(cat)    
    if len(cat_str) == 3:  
        clim_digit  = int(cat_str[0])
        rough_digit = int(cat_str[1])
        topo_digit  = int(cat_str[2])

        clim_short     = climate_labels[clim_digit-1]
        cat_labels[cat] = f"{clim_short}"+"$R_{"+f"{rough_digit}"+"}T_{"+f"{topo_digit}"+"}$"
    else:  
        cat_labels[cat] = "$R_{"+f"{str(1)}"+"}$:water"

x_pos    = np.arange(len(fl_cats))
x_labels =  [cat_labels[cat] for cat in fl_cats if cat in cat_labels]

fig, ax = plt.subplots(figsize=(10, 7))
ax      = df_variance.set_index('Category')[['PC1', 'PC2', 'PC3']].plot(kind='bar', stacked=True, color=colors,alpha=0.8, width=0.8, ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axhline(y=avg_pc1, color=colors[0], linestyle='--', lw=1.5, alpha=0.5)         # Añadir línea para promedio general PC1
plt.axhline(y=avg_pc1+avg_pc2, color=colors[1], linestyle='--', lw=1.5, alpha=0.5) # Añadir línea para promedio general PC2
plt.title('Explained variance by Principal Components', fontsize=16, fontweight="bold" , y=1.08)
plt.xlabel('Spatial category', fontsize=13)
plt.ylabel('Variance explained (%)', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=30, ha='center', fontsize=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [f'PC1 (avg: {avg_pc1:.1f}%)', f'PC2 (avg: {avg_pc2:.1f}%)', f'PC3 (avg: {avg_pc3:.1f}%)'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, hspace=0.19, left=0.11, right=0.95, bottom=0.15, top=0.85)
# plt.savefig(f"{bd_out_fig}PCA_VarianceExplained.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()


########################################################################################
###--------------------PLOTTING PC1 LOADS COMMON AVG SIGNAL IQR BANDS----------------###
########################################################################################

# Recopilar datos de loadings para PC1 por categoría
loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
categories    = []
for cat in df_categories.index:
    # Recoge loadings de PC1 para todos los puntos en esta categoría
    cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
    for point_result in all_pca_results[cat]:
        if 'loading_pattern' in point_result and 'pc1' in point_result['loading_pattern']:
            pc1_loadings = point_result['loading_pattern']['pc1']
            for model in ['ETH', 'CNRM', 'CMCC']:
                if model in pc1_loadings:
                    # Asegurar que todos los loadings tengan el mismo signo (positivo)
                    # para facilitar la comparación
                    sign_correction = 1 if np.mean(list(pc1_loadings.values())) >= 0 else -1
                    cat_loadings[model].append(pc1_loadings[model] * sign_correction)
    
    # Si tenemos datos para esta categoría -Siempre va a a suceder por como extraje las series de tiempo
    if all(len(values) > 0 for values in cat_loadings.values()):
        categories.append(cat)
        for model in ['ETH', 'CNRM', 'CMCC']:
            mean_loading = np.mean(cat_loadings[model])
            std_loading  = np.std(cat_loadings[model])
            q1_loading   = np.percentile(cat_loadings[model], 25)
            q3_loading   = np.percentile(cat_loadings[model], 75)
            
            loadings_data[model].append({'category': cat,
                                          'mean'   : mean_loading,
                                          'std'    : std_loading,
                                          'q1'     : q1_loading,
                                          'q3'     : q3_loading,
                                          'min'    : np.min(cat_loadings[model]),
                                          'max'    : np.max(cat_loadings[model]) })

model_colors = {'ETH': '#edae49', 'CNRM': '#00798c', 'CMCC': '#d1495b'}
x_pos     = np.arange(len(fl_cats))
x_labels  =  [cat_labels[cat] for cat in fl_cats if cat in cat_labels]
bar_width = 0.25
offsets   = {'ETH': -bar_width, 'CNRM': 0, 'CMCC': bar_width}
# Para cada modelo, graficar puntos con barras de error, porque no podemos sugerir continuidad (las linas no son apropiadas)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i, (model, color) in enumerate(model_colors.items()):
    # Extraer valores medios y dispersión
    # Calcular longitudes de las barras de error (distancia desde la media)
    means = [data['mean'] for data in loadings_data[model]]
    q1s   = [data['q1'] for data in loadings_data[model]]
    q3s   = [data['q3'] for data in loadings_data[model]]
    
    yerr_low  = [means[j] - q1s[j] for j in range(len(means))]
    yerr_high = [q3s[j] - means[j] for j in range(len(means))]
    yerr      = [yerr_low, yerr_high]
    
    # Graficar puntos con barras de error (IQR)
    ax.errorbar(x_pos + offsets[model], means, yerr=yerr, fmt='o', color=color, label=model,  markersize=6, capsize=4, elinewidth=1.5)

# Añadir línea horizontal para referencia de balance perfecto
ax.axhline(y=1/np.sqrt(3), color='black', linestyle='--', alpha=0.5, label=r'Balanced loading $\left( \frac{1}{\sqrt{3}} \right)$')
plt.title(f'PC1({avg_pc1:.1f}%) loadings by spatial category', fontsize=14, fontweight='bold', y=1.08)
plt.xlabel('Spatial category', fontsize=13)
plt.ylabel('PC1 Loading coefficients', fontsize=13)
plt.xlim(-0.5, len(categories)-0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=30, ha='center', fontsize=10)
ax.tick_params(axis='both', direction='in', which='both')
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=4)
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, hspace=0.19, left=0.11, right=0.95, bottom=0.15, top=0.85)
# plt.savefig(f"{bd_out_fig}PC1_loadings_distribution.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

##############################################################################################
##--------------------------ANALISIS DE DIVERGENCIAS DE LOS MODELOS - PC2 ------------------##
##############################################################################################
"""
Características de los loadings de PCA:
- Rango de valores: Los loadings pueden variar entre -1 y 1.
- Normalización: Los loadings están normalizados, lo que significa que la suma de los cuadrados
 de los loadings para un componente siempre es igual a 1:
 (loading_ETH)² + (loading_CNRM)² + (loading_CMCC)² = 1

Interpretación de magnitud de los laods:
- Valores cercanos a -1 o 1: Fuerte contribución al componente
- Valores cercanos a 0: Contribución débil al componente

Para el caso de 3 feature/variables/modelos:
- Si los tres modelos contribuyeran exactamente igual, cada loading sería ±1/√3 ≈ ±0.577
- Si dos modelos no contribuyeran en absoluto y uno fuera responsable de toda la varianza,
 su loading sería ±1

Para el segundo componente principal (PC2):
- Loading cercano a 0: El modelo está cerca del "consenso" y contribuye poco a la discrepancia principal. 
ETH típicamente muestra este comportamiento.
- Loading con valor alto positivo (ej. 0.8): El modelo tiende a predecir valores consistentemente más 
altos en comparación con el consenso cuando PC2 es positivo.
- Loading con valor alto negativo (ej. -0.8): El modelo tiende a predecir valores consistentemente más 
bajos en comparación con el consenso cuando PC2 es positivo.
- Loadings con signos opuestos: Indican que los modelos correspondientes tienden a divergir en 
direcciones opuestas. Por ejemplo, si CMCC tiene loading +0.8 y CNRM tiene -0.6, estos dos modelos representan 
los extremos opuestos de la principal fuente de discrepancia.

"""

# Definir etiquetas de clima
climate_labels   = ['Ar', 'Tm', 'Co', 'Td']
roughness_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
slope_labels     = ['Very Low', 'Low', 'Moderate', 'High']

# Crear etiquetas para categorías
cat_labels = {}
for cat in fl_cats:
    cat_str = str(cat)
    if len(cat_str) == 3:
        clim_digit = int(cat_str[0])
        rough_digit = int(cat_str[1])
        topo_digit = int(cat_str[2])
        clim_short = climate_labels[clim_digit-1]
        cat_labels[cat] = f"{clim_short}"+"$R_{"+f"{rough_digit}"+"}T_{"+f"{topo_digit}"+"}$"
    else:
        cat_labels[cat] = "$R_{"+f"{str(1)}"+"}$:water"


divergence_data = []
for cat_idx, cat in enumerate(df_categories.index):
    cat_code = cat    
    # Caso especial para agua (cat == 1)
    if cat == 1:
        climate = "Water"
        roughness = "Water"
        slope = "Water"
        climate_code = 0  # Código especial para agua
        roughness_code = 1
        slope_code = 0
    else:
        cat_str = str(cat)
        # Asegúrate de que cat_str tenga al menos 3 dígitos
        if len(cat_str) < 3:
            continue  # Saltarse este caso si no tiene el formato esperado y no es el caso especial de agua
            
        climate_code   = int(cat_str[0])
        roughness_code = int(cat_str[1])
        slope_code     = int(cat_str[2])
        
        # Usar los mismos labels definidos anteriormente para consistencia
        climate   = climate_labels[climate_code-1] if 1 <= climate_code <= len(climate_labels) else f"Unknown ({climate_code})"
        roughness = roughness_labels[roughness_code-1] if 1 <= roughness_code <= len(roughness_labels) else f"Unknown ({roughness_code})"
        slope     = slope_labels[slope_code-1] if 1 <= slope_code <= len(slope_labels) else f"Unknown ({slope_code})"
    
    # Extraer patrones de divergencia
    patterns     = df_categories.loc[cat, 'divergence_patterns']
    total_points = sum(patterns.values())
    
    # Calcular porcentajes por modelo
    eth_total  = patterns.get('ETH_different', 0) + patterns.get('ETH_opposite', 0)
    cnrm_total = patterns.get('CNRM_different', 0) + patterns.get('CNRM_opposite', 0)
    cmcc_total = patterns.get('CMCC_different', 0) + patterns.get('CMCC_opposite', 0)
    
    eth_pct  = eth_total / total_points * 100 if total_points > 0 else 0
    cnrm_pct = cnrm_total / total_points * 100 if total_points > 0 else 0
    cmcc_pct = cmcc_total / total_points * 100 if total_points > 0 else 0
    
    # Calcular ratios de different vs opposite
    eth_diff_ratio = patterns.get('ETH_different', 0) / eth_total * 100 if eth_total > 0 else 0
    eth_opp_ratio  = patterns.get('ETH_opposite', 0) / eth_total * 100 if eth_total > 0 else 0
    
    cnrm_diff_ratio = patterns.get('CNRM_different', 0) / cnrm_total * 100 if cnrm_total > 0 else 0
    cnrm_opp_ratio  = patterns.get('CNRM_opposite', 0) / cnrm_total * 100 if cnrm_total > 0 else 0
    
    cmcc_diff_ratio = patterns.get('CMCC_different', 0) / cmcc_total * 100 if cmcc_total > 0 else 0
    cmcc_opp_ratio  = patterns.get('CMCC_opposite', 0) / cmcc_total * 100 if cmcc_total > 0 else 0
    
    # Guardar resultados con etiqueta de categoría actualizada
    category_label = cat_labels.get(cat, str(cat))  # Usar la etiqueta nueva si existe, o el código como respaldo
    
    divergence_data.append({'Category'       : cat,
                            'Category_Label' : category_label,
                            'Climate'        : climate,
                            'Climate_code'   : climate_code,
                            'Roughness'      : roughness,
                            'Roughness_code' : roughness_code,
                            'Slope'          : slope,
                            'Slope_code'     : slope_code,
                            'ETH_pct'        : eth_pct,
                            'CNRM_pct'       : cnrm_pct,
                            'CMCC_pct'       : cmcc_pct,
                            'ETH_diff_ratio' : eth_diff_ratio,
                            'ETH_opp_ratio'  : eth_opp_ratio,
                            'CNRM_diff_ratio': cnrm_diff_ratio,
                            'CNRM_opp_ratio' : cnrm_opp_ratio,
                            'CMCC_diff_ratio': cmcc_diff_ratio,
                            'CMCC_opp_ratio' : cmcc_opp_ratio })

# Crear DataFrame y ordenar
df_divergence = pd.DataFrame(divergence_data)
# Agregar una clave de ordenación para mantener el orden lógico
# Código especial para agua (valor menor que cualquier otro)
df_divergence['Sort_Key'] = df_divergence.apply(lambda x: -1 if x['Category'] == 1 else x['Climate_code'] * 100 + x['Roughness_code'] * 10 + x['Slope_code'],  axis=1)
df_divergence             = df_divergence.sort_values('Sort_Key')
df_divergence.index       = df_divergence['Category']


########################################################################################
###----------------VISUALIZACIÓN DE RESULTADOS DE DIVERGENCIA -PC2-------------------###
########################################################################################
# Visualización mejorada de patrones de divergencia dominantes
"""Visualiza los patrones de divergencia dominantes por categoría espacial con los neuvos labels"""

climate_labels = ['Ar', 'Tm', 'Co', 'Td']
cat_labels     = {}
for cat in fl_cats:
    cat_str = str(cat)
    if len(cat_str) == 3:
        clim_digit = int(cat_str[0])
        rough_digit = int(cat_str[1])
        topo_digit = int(cat_str[2])
        clim_short = climate_labels[clim_digit-1]
        cat_labels[cat] = f"{clim_short}"+"$R_{"+f"{rough_digit}"+"}T_{"+f"{topo_digit}"+"}$"
    else:
        cat_labels[cat] = "$R_{"+f"{str(1)}"+"}$:water"

pattern_mapping = {'ETH_different' : 'ETH_over', 
                   'ETH_opposite'  : 'ETH_under', 
                   'CNRM_different': 'CNRM_over', 
                   'CNRM_opposite' : 'CNRM_under', 
                   'CMCC_different': 'CMCC_over', 
                   'CMCC_opposite' : 'CMCC_under'}

original_patterns = ['ETH_different', 'ETH_opposite', 'CNRM_different', 'CNRM_opposite', 'CMCC_different', 'CMCC_opposite']
patterns          = [pattern_mapping[p] for p in original_patterns]

pattern_data = []
for cat in df_categories.index:
    cat_patterns = df_categories.loc[cat, 'divergence_patterns']
    total = sum(cat_patterns.values())
    row = {'category': cat}
    
    # Usar los nombres originales para acceder a los datos pero guardar con nombres nuevos
    for old_pattern, new_pattern in pattern_mapping.items():
        row[new_pattern] = cat_patterns.get(old_pattern, 0) / total if total > 0 else 0
    
    pattern_data.append(row)
df_patterns = pd.DataFrame(pattern_data)
df_patterns = df_patterns.set_index('category')

# Ordenar en base al numero de la categoria espacial en fl_cats
df_patterns['order'] = df_categories['climate_code']*100 + df_categories['roughness_code']*10 + df_categories['slope_code']
df_patterns          = df_patterns.sort_values('order')
df_patterns          = df_patterns.drop('order', axis=1)

# Crear un colormap normalizado discreto
bounds = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 0.9, 1.0]
cmap   = plt.cm.get_cmap('magma_r', len(bounds)-1)
norm   = mpl.colors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(14, len(df_categories)*0.5 + 2))
ax = sns.heatmap(df_patterns, cmap=cmap, annot=True, fmt='.2f', linewidths=.5, vmin=0, vmax=1)
# Usar las etiquetas simplificadas del diccionario cat_labels para el eje Y
plt.yticks(np.arange(len(df_patterns.index))+0.5, [cat_labels.get(cat, str(cat)) for cat in df_patterns.index],rotation=0)
plt.ylabel('Spatial categories', fontsize=12)
plt.xlabel('Divergence patterns (over = positive bias, under = negative bias)', fontsize=12)
plt.title('Dominant divergence patterns by spatial category', fontsize=14, fontweight="bold")
plt.xticks(rotation=0)
cbar = ax.collections[0].colorbar
cbar.set_ticks(bounds)  
cbar.set_label('Proportion of total divergence', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.19, left=0.16, right=0.98, bottom=0.15, top=0.90)
# plt.savefig(f"{bd_out_fig}PC2_divergencePatterns.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

########################################################################################
###----------------ANÁLISIS Y EXTRACCION DE DIVERGENCIAS DEL PC3 -------------------###
########################################################################################

"""
Este script analiza las divergencias del tercer componente principal (PC3) 
para cada categoría espacial, extrae sus loadings y genera una visualización
para comparar patrones entre modelos.
"""

# Recopilar loadings de PC3 por categoría
pc3_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
categories     = []

for cat in df_categories.index:
    # Recoge loadings de PC3 para todos los puntos en esta categoría
    cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
    for point_result in all_pca_results[cat]:
        if 'loadings' in point_result and len(point_result['loadings']) >= 3:  # Asegurar que hay PC3
            pc3_loadings = point_result['loadings'][2]  # PC3 es el índice 2
            
            # Asignar loadings de PC3 a cada modelo
            if len(pc3_loadings) >= 3:  # Debe tener 3 valores (ETH, CNRM, CMCC)
                cat_loadings['ETH'].append(pc3_loadings[0])
                cat_loadings['CNRM'].append(pc3_loadings[1])
                cat_loadings['CMCC'].append(pc3_loadings[2])
    
    # Si tenemos datos suficientes para esta categoría
    if all(len(values) > 5 for values in cat_loadings.values()):  # Al menos 5 puntos con PC3
        categories.append(cat)
        for model in ['ETH', 'CNRM', 'CMCC']:
            mean_loading = np.mean(cat_loadings[model])
            std_loading = np.std(cat_loadings[model])
            q1_loading = np.percentile(cat_loadings[model], 25)
            q3_loading = np.percentile(cat_loadings[model], 75)
            
            pc3_loadings_data[model].append({
                'category': cat,
                'mean': mean_loading,
                'std': std_loading,
                'q1': q1_loading,
                'q3': q3_loading,
                'min': np.min(cat_loadings[model]),
                'max': np.max(cat_loadings[model])
            })

########################################################################################
###--------------------- VISUALIZACION DE DIVERGENCIAS DEL PC3 ----------------------###
########################################################################################

climate_labels = ['Ar', 'Tm', 'Co', 'Td']
cat_labels     = {}
for cat in categories:
    cat_str = str(cat)
    if len(cat_str) == 3:
        clim_digit      = int(cat_str[0])
        rough_digit     = int(cat_str[1])
        topo_digit      = int(cat_str[2])
        clim_short      = climate_labels[clim_digit-1]
        cat_labels[cat] = f"{clim_short}"+"$R_{"+f"{rough_digit}"+"}T_{"+f"{topo_digit}"+"}$"
    else:
        cat_labels[cat] = "$R_{"+f"{str(1)}"+"}$:water"

# Calcular la suma absoluta de cargas para cada categoría para ordenamiento descendente
category_abs_loading_sums = []
for i, cat in enumerate(categories):
    eth_mean  = abs(pc3_loadings_data['ETH'][i]['mean'])
    cnrm_mean = abs(pc3_loadings_data['CNRM'][i]['mean'])
    cmcc_mean = abs(pc3_loadings_data['CMCC'][i]['mean'])
    total_abs = eth_mean + cnrm_mean + cmcc_mean
    category_abs_loading_sums.append((cat, total_abs))

# Ordenar categorías por suma absoluta de cargas (descendente)
category_abs_loading_sums.sort(key=lambda x: x[1], reverse=True)
sorted_categories = [item[0] for item in category_abs_loading_sums]
x_pos             = np.arange(len(sorted_categories))
sorted_indices    = [categories.index(cat) for cat in sorted_categories]
bar_width         = 0.25
offsets           = {'ETH': -bar_width, 'CNRM': 0, 'CMCC': bar_width}

fig, ax = plt.figure(figsize=(12, 8), clear=True), plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for model, color in model_colors.items():
    # Extraer valores en el nuevo orden
    means = [pc3_loadings_data[model][idx]['mean'] for idx in sorted_indices]
    stds  = [pc3_loadings_data[model][idx]['std'] for idx in sorted_indices]    
    # Graficar barras
    ax.bar(x_pos + offsets[model], means, bar_width,label=model, color=color, alpha=0.8)
ax.set_xlabel('Spatial category', fontsize=13)
ax.set_ylabel('PC3 Loading coefficient', fontsize=13)
ax.set_title(f'PC3 ({avg_pc3:.1f}%) Loadings by spatial category (ordered by absolute magnitude)', fontsize=15, fontweight='bold')
# Configurar eje X con etiquetas de categoría en roden descendente
ax.set_xticks(x_pos)
ax.set_xticklabels([cat_labels.get(cat, str(cat)) for cat in sorted_categories], rotation=30)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8) # Las barras son al rededor del 0
ax.tick_params(axis='both', direction='in', which='both')
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
y_max = max([abs(data['mean']) for model_data in pc3_loadings_data.values() for data in model_data]) * 1.2
ax.set_ylim(-y_max, y_max)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.tight_layout()
plt.savefig(f"{bd_out_fig}PC3_divergencePatterns_Sorted.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()


##############################################################################################
##---------------------RESUMEN DE ANALISIS DE RESULTADOS Y GARGAS DEL PC3 ------------------##
##############################################################################################

# Análisis estadístico de patrones de PC3
print("\n### ANÁLISIS DE PC3 ###")

# Calcular coherencia de signos entre modelos para PC3
sign_patterns = {cat: {} for cat in categories}
for cat in categories:
    eth_sign = np.sign(pc3_loadings_data['ETH'][categories.index(cat)]['mean'])
    cnrm_sign = np.sign(pc3_loadings_data['CNRM'][categories.index(cat)]['mean'])
    cmcc_sign = np.sign(pc3_loadings_data['CMCC'][categories.index(cat)]['mean'])
    
    pattern = f"{'+' if eth_sign > 0 else '-'}{'+' if cnrm_sign > 0 else '-'}{'+' if cmcc_sign > 0 else '-'}"
    sign_patterns[cat] = {
        'pattern': pattern,
        'description': interpret_pc3_pattern(eth_sign, cnrm_sign, cmcc_sign)
    }

# Mostrar patrones más comunes
patterns_count = {}
for cat, data in sign_patterns.items():
    pattern = data['pattern']
    if pattern not in patterns_count:
        patterns_count[pattern] = 0
    patterns_count[pattern] += 1

print("Patrones de signos predominantes en PC3:")
for pattern, count in sorted(patterns_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  Patrón {pattern}: {count} categorías ({count/len(categories)*100:.1f}%)")

# Identificar categorías con mayor varianza explicada por PC3
pc3_importance = []
for cat in categories:
    # Obtener la varianza explicada por PC3 para esta categoría
    pc3_var = 0
    for point in all_pca_results[cat]:
        if 'pc3_var' in point:
            pc3_var += point['pc3_var']
    if len(all_pca_results[cat]) > 0:
        pc3_var /= len(all_pca_results[cat])
        
    pc3_importance.append((cat, pc3_var))

print("\nCategorías con mayor importancia de PC3:")
for cat, var in sorted(pc3_importance, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {cat_labels.get(cat, str(cat))}: {var*100:.2f}% de varianza explicada")
    
print("\nCategorías con menor importancia de PC3:")
for cat, var in sorted(pc3_importance, key=lambda x: x[1])[:5]:
    print(f"  {cat_labels.get(cat, str(cat))}: {var*100:.2f}% de varianza explicada")

##############################################################################################
##----------------PLOTTING THE FREQUENCIES OF DIVERGENCE AND LOADS FROM PC2 ----------------##
##############################################################################################
"""
Plot 1) Divergence frequency graph: Shows what percentage of points has each model as the ‘most 
divergent’ (the one with the highest absolute load on PC2). ETH with 0% means that it is never 
the model with the highest absolute load.
Plot 2) PC2 loadings graph: Shows the direct contribution of each model to the second principal 
component. ETH with values close to zero indicates that it contributes little to this divergence 
pattern.
"""

# Recopilar loadings de PC2 por categoría
pc2_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
categories        = []
for cat in df_categories.index:
    # Recoge loadings de PC2 para todos los puntos en esta categoría
    cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
    for point_result in all_pca_results[cat]:
        if 'loading_pattern' in point_result and 'pc2' in point_result['loading_pattern']:
            pc2_loadings = point_result['loading_pattern']['pc2']
            for model in ['ETH', 'CNRM', 'CMCC']:
                if model in pc2_loadings:
                    cat_loadings[model].append(pc2_loadings[model])
    
    # Si tenemos datos para esta categoría
    if all(len(values) > 0 for values in cat_loadings.values()):
        categories.append(cat)
        for model in ['ETH', 'CNRM', 'CMCC']:
            mean_loading = np.mean(cat_loadings[model])
            std_loading  = np.std(cat_loadings[model])
            
            pc2_loadings_data[model].append({'category': cat,
                                             'mean'    : mean_loading,
                                             'std'     : std_loading})
# Extraer valores medios para cada modelo
eth_means  = [data['mean'] for data in pc2_loadings_data['ETH']]
cnrm_means = [data['mean'] for data in pc2_loadings_data['CNRM']]
cmcc_means = [data['mean'] for data in pc2_loadings_data['CMCC']]

model_colors = {'ETH': '#edae49', 'CNRM': '#00798c', 'CMCC': '#d1495b'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})
# 1. Gráfico superior: Proporciones de divergencia total por modelo
bar_width = 0.25
indices   = np.arange(len(df_divergence))
ax1.bar(indices - bar_width, df_divergence['ETH_pct'], bar_width, label='ETH', color=model_colors['ETH'], alpha=0.8)
ax1.bar(indices, df_divergence['CNRM_pct'], bar_width, label='CNRM', color=model_colors['CNRM'], alpha=0.8)
ax1.bar(indices + bar_width, df_divergence['CMCC_pct'], bar_width, label='CMCC', color=model_colors['CMCC'], alpha=0.8)
ax1.set_ylabel('Points frequency\n model divergence (%)', fontsize=13)
ax1.set_title('Model divergence frequency by Spatial Category', fontsize=15, fontweight='bold' , y=1.13)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(-10, 100) 
ax1.axhline(y=0, color='grey', linestyle='-', linewidth=1)

# 2. Gráfico inferior: Loadings de PC2
x = np.arange(len(categories))
ax2.bar(x - bar_width, eth_means, bar_width, label='ETH', color=model_colors['ETH'], alpha=0.8)
ax2.bar(x, cnrm_means, bar_width, label='CNRM', color=model_colors['CNRM'], alpha=0.8)
ax2.bar(x + bar_width, cmcc_means, bar_width, label='CMCC', color=model_colors['CMCC'], alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylim(-1, 1) # Fijar los límites entre -1 y 1 de acuerdo a la magnitud normalizada de los loads
ax2.set_xlabel('Spatial Category', fontsize=13)
ax2.set_ylabel('PC2 Loading coefficient', fontsize=13)
ax2.set_title('Divergence patterns (PC2 Loadings) by Spatial Category', fontsize=15, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=0)
ax2.grid(axis='y', alpha=0.3)
ymin, ymax = ax2.get_ylim()
ax2.text(len(categories)/2, ymax*0.82, 
         r"$\uparrow$ Positive values: Model tends to predict higher values", 
         ha='center', fontsize=11, bbox=dict(edgecolor= 'white', facecolor='white', alpha=0.7))
ax2.text(len(categories)/2, ymin*0.85, 
         r"$\downarrow$ Negative values: Model tends to predict lower values", 
         ha='center', fontsize=11, bbox=dict(edgecolor= 'white', facecolor='white', alpha=0.7))

# Añadir líneas divisorias entre categorías de clima en ambos subplots
climate_changes = []
prev_climate = None
for i, cat in enumerate(categories):
    climate = int(str(cat)[0])
    if climate != prev_climate:
        if prev_climate is not None:
            ax1.axvline(i-0.5, color='black', linestyle='--', alpha=0.5)
            ax2.axvline(i-0.5, color='black', linestyle='--', alpha=0.5)
        climate_changes.append((i, climate))
        prev_climate = climate

# Añadir etiquetas para climas en el subplot superior
for i, (pos, climate) in enumerate(climate_changes):
    if i < len(climate_changes) - 1:
        end_pos = climate_changes[i+1][0]
        mid_pos = (pos + end_pos) / 2
    else:
        mid_pos = (pos + len(categories)) / 2
    
    climate_map = {1: 'Arid', 2: 'Temperate', 3: 'Cold', 4: 'Polar'}
    ax1.text(mid_pos, ax1.get_ylim()[1]*1.05, climate_map.get(climate, f'Climate {climate}'), 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, hspace=0.19, left=0.11, right=0.95, bottom=0.15, top=0.85)
plt.savefig(f"{bd_out_fig}Combined_divergence_PC2analysis.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

##############################################################################################
##-------------------PLOTTING DIVERGENCE AND LOADS FROM PC2 AS HEATMAPS------------------##
##############################################################################################

# Crear matriz 3D para almacenar loadings promedio: [modelo, clima, rugosidad, pendiente]
models           = ['ETH', 'CNRM', 'CMCC']
climates         = ['Arid', 'Temperate', 'Cold', 'Polar']
roughness_levels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
slope_levels     = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

# Mapear nombres a códigos
climate_code   = {'Arid': 1, 'Temperate': 2, 'Cold': 3, 'Polar': 4}
roughness_code = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
slope_code     = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}

# Inicializar matriz 4D para almacenar loadings
# [modelo, clima, rugosidad, pendiente]
loading_matrix = np.full((len(models), len(climates), 5, 5), np.nan)

# Llenar la matriz con datos disponibles
for cat in df_categories.index:
    cat_str = str(cat)
    if len(cat_str) < 3:
        continue
        
    c_idx = int(cat_str[0]) - 1  # Índice de clima (0-3)
    r_idx = int(cat_str[1]) - 1  # Índice de rugosidad (0-4)
    s_idx = int(cat_str[2]) - 1  # Índice de pendiente (0-4)
    
    # Verificar índices válidos
    if c_idx < 0 or c_idx >= len(climates) or \
       r_idx < 0 or r_idx >= 5 or \
       s_idx < 0 or s_idx >= 5:
        continue
    
    # Obtener loadings promedio para esta categoría
    for m_idx, model in enumerate(models):
        cat_loadings = []
        for point_result in all_pca_results[cat]:
            if 'loading_pattern' in point_result and 'pc2' in point_result['loading_pattern']:
                pc2_loadings = point_result['loading_pattern']['pc2']
                if model in pc2_loadings:
                    cat_loadings.append(pc2_loadings[model])
                    
        if cat_loadings:
            loading_matrix[m_idx, c_idx, r_idx, s_idx] = np.mean(cat_loadings)

# Definir mapa de color centrado en cero
cmap = plt.cm.RdBu
norm = plt.Normalize(-1, 1)

fig, axs = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
for m_idx, model in enumerate(models):
    for c_idx, climate in enumerate(climates):
        ax = axs[m_idx, c_idx]
        
        # Datos para este modelo y clima
        data = loading_matrix[m_idx, c_idx]
        
        # Crear heatmap
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', origin='lower')
        
        # Configurar títulos y etiquetas
        if m_idx == 0:
            ax.set_title(climate, fontsize=14, fontweight='bold')
        if c_idx == 0:
            ax.set_ylabel(rf"$\bf{{{model}}}$" + "\nRoughness", fontsize=13)
        if m_idx == 2:
            ax.set_xlabel("Slope Variance", fontsize=13)
            
        # Añadir grid
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1, alpha=0.2)
        
        # Añadir valores en las celdas
        for i in range(5):
            for j in range(5):
                value = data[i, j]
                if not np.isnan(value):
                    # Color de texto adaptativo según el valor de fondo
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                            color=text_color, fontsize=11)
        
        # Configurar ticks
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([str(i+1) for i in range(5)] , fontsize=11)
        ax.set_yticklabels([str(i+1) for i in range(5)], fontsize=11)

# Añadir una barra de color horizontal en la parte inferior
cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.6])  # (left, bottom, width, height)
cbar    = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('PC2 Loading Value (Divergence Pattern)', fontsize=12)
cbar.ax.axhline(y=0, color='k', linestyle='--', alpha=0.5) # Añadir línea horizontal en la barra de color

# Leyenda para códigos
legend_text = """Roughness & Slope variance Levels:
1: Very Low  2: Low  3: Moderate  4: High  5: Very High"""
fig.text(0.5, 0.035, legend_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

fig.suptitle('PC2 Loading Distribution by Spatial categories',  fontsize=16, fontweight='bold', y=0.97)
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.19, left=0.06, right=0.91, bottom=0.15, top=0.90)
# plt.savefig(f"{bd_out_fig}PC2_loading_heatmaps.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()


########################################################################################
###-------COLLECTING SCORES FOR ALL THE CATEGORIES (USEFUL IN EXTREME EVENTS)--------###
########################################################################################

# Recopilar scores de todas las categorías
all_scores_data = []

for cat in df_categories.index:
    # Extraer información de categoría
    cat_code       = cat
    climate_code   = int(str(cat)[0])
    roughness_code = int(str(cat)[1])
    slope_code     = int(str(cat)[2])
    
    # Mapear códigos a descripciones
    climate_map   = {1: 'Arid', 2: 'Temperate', 3: 'Cold', 4: 'Polar'}
    roughness_map = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
    slope_map     = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
    
    climate   = climate_map.get(climate_code, f"Unknown ({climate_code})")
    roughness = roughness_map.get(roughness_code, f"Unknown ({roughness_code})")
    slope     = slope_map.get(slope_code, f"Unknown ({slope_code})")
    
    # Para cada punto en esta categoría
    for point_result in all_pca_results[cat]:
        if 'point_idx' not in point_result:
            continue
            
        # Extraer loadings y varianza explicada
        if 'loadings' in point_result and len(point_result['loadings']) >= 2:
            pc1_loading = point_result['loadings'][0]
            pc2_loading = point_result['loadings'][1]
            pc1_var     = point_result.get('pc1_var', 0)
            pc2_var     = point_result.get('pc2_var', 0)
            
            # Calcular scores "promedio" que represente esta categoría-punto
            # Esto es una simplificación, idealmente deberíamos usar scores reales
            eth_contrib_pc1  = pc1_loading[0]
            cnrm_contrib_pc1 = pc1_loading[1]
            cmcc_contrib_pc1 = pc1_loading[2]
            
            eth_contrib_pc2  = pc2_loading[0]
            cnrm_contrib_pc2 = pc2_loading[1]
            cmcc_contrib_pc2 = pc2_loading[2]
            
            # Guardar información
            all_scores_data.append({ 'Category': cat_code,
                                     'Climate'       : climate,
                                     'Climate_code'  : climate_code,
                                     'Roughness'     : roughness,
                                     'Roughness_code': roughness_code,
                                     'Slope': slope,
                                     'Slope_code': slope_code,
                                     'PC1_ETH': eth_contrib_pc1,
                                     'PC1_CNRM': cnrm_contrib_pc1,
                                     'PC1_CMCC': cmcc_contrib_pc1,
                                     'PC2_ETH': eth_contrib_pc2,
                                     'PC2_CNRM': cnrm_contrib_pc2,
                                     'PC2_CMCC': cmcc_contrib_pc2,
                                     'PC1_var': pc1_var,
                                     'PC2_var': pc2_var,
                                     'Point_idx': point_result['point_idx'] })


df_all_scores = pd.DataFrame(all_scores_data)

########################################################################################
###--SCORE PLOTS PER CATEGORIES PC1 Y PC2-MORE FOR THE SAMPLES ANALYSIS ON THE TIME--###
########################################################################################

# Crear score plot por clima
plt.figure(figsize=(10, 8))

# Colores por clima
climate_colors = {'Arid': '#e76f51', 'Temperate': '#f4a261', 
                  'Cold': '#2a9d8f', 'Polar': '#264653'}

# Marcadores por rugosidad
roughness_markers = {'Very Low': 'o', 'Low': 's', 'Moderate': '^', 
                    'High': 'D', 'Very High': '*'}

# Tamaño por pendiente
slope_sizes = {'Very Low': 50, 'Low': 80, 'Moderate': 110, 
              'High': 140, 'Very High': 170}

# Graficar puntos por categoría
for climate in climate_colors:
    for roughness in roughness_markers:
        for slope in slope_sizes:
            # Filtrar datos
            mask = (df_all_scores['Climate'] == climate) & \
                   (df_all_scores['Roughness'] == roughness) & \
                   (df_all_scores['Slope'] == slope)
            
            if not any(mask):
                continue
                
            # Calcular scores promedio para esta combinación
            pc1_avg = df_all_scores.loc[mask, ['PC1_ETH', 'PC1_CNRM', 'PC1_CMCC']].mean(axis=1)
            pc2_avg = df_all_scores.loc[mask, ['PC2_ETH', 'PC2_CNRM', 'PC2_CMCC']].mean(axis=1)
            
            # Graficar
            plt.scatter(pc1_avg, pc2_avg, 
                       c=climate_colors[climate], 
                       marker=roughness_markers[roughness],
                       s=slope_sizes[slope],
                       alpha=0.7,
                       edgecolors='black', linewidth=0.5,
                       label=f"{climate}, {roughness}, {slope}")

# Mejorar estética
plt.title('PCA Score Plot by Spatial Category', fontsize=16)
plt.xlabel('PC1 (Consensus Signal)', fontsize=14)
plt.ylabel('PC2 (Main Divergence Pattern)', fontsize=14)
plt.grid(alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.2)

# Leyendas separadas para clima, rugosidad y pendiente
# Crear leyenda para clima
climate_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color, markersize=10, 
                             label=climate)
                  for climate, color in climate_colors.items()]

# Crear leyenda para rugosidad
roughness_handles = [plt.Line2D([0], [0], marker=marker, color='black', 
                               markerfacecolor='grey', markersize=8, 
                               label=roughness)
                    for roughness, marker in roughness_markers.items()]

# Crear leyenda para pendiente
slope_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor='grey', markersize=np.sqrt(size/np.pi), 
                           label=slope)
                for slope, size in slope_sizes.items()]

# Añadir leyendas
plt.legend(handles=climate_handles, title="Climate", loc='upper left', bbox_to_anchor=(1.01, 1))
plt.legend(handles=roughness_handles, title="Roughness",  loc='upper left', bbox_to_anchor=(1.01, 0.75))
plt.legend(handles=slope_handles, title="Slope Variance",  loc='upper left', bbox_to_anchor=(1.01, 0.5))

plt.tight_layout()
# plt.savefig(f"{bd_out_fig}ScorePlot_all_categories.png", dpi=300, bbox_inches='tight')
plt.show()

########################################################################################
###------------------NIVEL 3: ANÁLISIS COMPARATIVO ENTRE CATEGORÍAS-----------------###
########################################################################################

def analyze_by_dimension(df, dimension):
    """Analizar patrones por dimensión (clima, rugosidad, pendiente)"""
    if dimension  == 'climate':
        dim_col  = 'climate'
        code_col = 'climate_code'
    elif dimension == 'roughness':
        dim_col  = 'roughness'
        code_col = 'roughness_code'
    else:  # slope
        dim_col  = 'slope_variance'
        code_col = 'slope_code'
    
    # Agrupar por la dimensión
    grouped = df.groupby(dim_col)
    
    # Métricas clave
    metrics = ['pc1_mean', 'pc2_mean', 'agreement_index_mean', 'extreme_discrepancy_mean']
    
    # Obtener estadísticas
    stats = grouped[metrics].mean()
    
    # Ordenar por código
    ordered_values = df.groupby(dim_col)[code_col].first().sort_values()
    stats          = stats.loc[ordered_values.index]
    
    return stats

# Analizar patrones por dimensión
climate_stats   = analyze_by_dimension(df_categories, 'climate')
roughness_stats = analyze_by_dimension(df_categories, 'roughness')
slope_stats     = analyze_by_dimension(df_categories, 'slope_variance')

# Identificar categorías con mayor/menor acuerdo
high_agreement_cats = df_categories[df_categories['agreement_index_mean'] > df_categories['agreement_index_mean'].quantile(0.75)]
low_agreement_cats  = df_categories[df_categories['agreement_index_mean'] < df_categories['agreement_index_mean'].quantile(0.25)]

# Identificar categorías con mayor/menor discrepancia en extremos
high_extreme_diff_cats = df_categories[df_categories['extreme_discrepancy_mean'] > df_categories['extreme_discrepancy_mean'].quantile(0.75)]
low_extreme_diff_cats  = df_categories[df_categories['extreme_discrepancy_mean'] < df_categories['extreme_discrepancy_mean'].quantile(0.25)]

########################################################################################
###--------------------------RESUMEN DE RESULTADOS-----------------------------------###
########################################################################################

# Imprimir hallazgos principales
print("\n### KEY FINDINGS ###")
print(f"Categories with highest model agreement: {', '.join(map(str, high_agreement_cats.index.tolist()))}")
print(f"Categories with lowest model agreement: {', '.join(map(str, low_agreement_cats.index.tolist()))}")
print(f"Categories with highest discrepancy in extremes: {', '.join(map(str, high_extreme_diff_cats.index.tolist()))}")
print(f"Categories with lowest discrepancy in extremes: {', '.join(map(str, low_extreme_diff_cats.index.tolist()))}")

# Correlaciones entre métricas PCA y características de categorías
corr_climate = df_categories['climate_code'].corr(df_categories['agreement_index_mean'])
corr_roughness = df_categories['roughness_code'].corr(df_categories['agreement_index_mean'])
corr_slope = df_categories['slope_code'].corr(df_categories['agreement_index_mean'])

print(f"\nCorrelation between agreement index and climate code: {corr_climate:.3f}")
print(f"Correlation between agreement index and roughness code: {corr_roughness:.3f}")
print(f"Correlation between agreement index and slope variance code: {corr_slope:.3f}")


########################################################################################
###------------ VISUALIZING THE THREE PRINCIPAL COMPONENTS IN A SINGLE FIGURE--------###
########################################################################################

bar_width    = 0.25
offsets      = {'ETH': -bar_width, 'CNRM': 0, 'CMCC': bar_width}
model_colors = {'ETH': '#edae49', 'CNRM': '#00798c', 'CMCC': '#d1495b'}

# Definir las categorías para análisis
categories = fl_cats  # Asumiendo que fl_cats contiene las categorías filtradas
x_pos      = np.arange(len(categories))

# Crear etiquetas para categorías
climate_labels = ['Ar', 'Tm', 'Co', 'Td']
cat_labels     = {}
for cat in categories:
    cat_str = str(cat)
    if len(cat_str) == 3:
        clim_digit  = int(cat_str[0])
        rough_digit = int(cat_str[1])
        topo_digit  = int(cat_str[2])
        clim_short  = climate_labels[clim_digit-1]
        cat_labels[cat] = f"{clim_short}"+"$R_{"+f"{rough_digit}"+"}T_{"+f"{topo_digit}"+"}$"
    else:
        cat_labels[cat] = "$R_{"+f"{str(1)}"+"}$:water"


fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)  # Más compacto

# Tamaños de fuente más grandes
TITLE_SIZE  = 16
LABEL_SIZE  = 14
TICK_SIZE   = 13
LEGEND_SIZE = 13

# PANEL 1: PC1 LOADINGS
# Recopilar loadings de PC1 por categoría
pc1_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}

for cat in categories:
    # Recoge loadings de PC1 para todos los puntos en esta categoría
    cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
    for point_result in all_pca_results[cat]:
        if 'loading_pattern' in point_result and 'pc1' in point_result['loading_pattern']:
            pc1_loadings = point_result['loading_pattern']['pc1']
            for model in ['ETH', 'CNRM', 'CMCC']:
                if model in pc1_loadings:
                    # Asegurar que todos los loadings tengan el mismo signo (positivo)
                    sign_correction = 1 if np.mean(list(pc1_loadings.values())) >= 0 else -1
                    cat_loadings[model].append(pc1_loadings[model] * sign_correction)
    
    # Si tenemos datos para esta categoría
    if all(len(values) > 0 for values in cat_loadings.values()):
        for model in ['ETH', 'CNRM', 'CMCC']:
            mean_loading = np.mean(cat_loadings[model])
            std_loading = np.std(cat_loadings[model])
            q1_loading = np.percentile(cat_loadings[model], 25)
            q3_loading = np.percentile(cat_loadings[model], 75)
            
            pc1_loadings_data[model].append({ 'category': cat,
                                               'mean'   : mean_loading,
                                               'std'    : std_loading,
                                               'q1'     : q1_loading,
                                               'q3'     : q3_loading })

# Graficar PC1
ax = axs[0]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for model, color in model_colors.items():
    means = [data['mean'] for data in pc1_loadings_data[model]]
    q1s   = [data['q1'] for data in pc1_loadings_data[model]]
    q3s   = [data['q3'] for data in pc1_loadings_data[model]]
    
    # Crear barras de error (asegurando que son positivas)
    yerr = np.zeros((2, len(means)))
    for j in range(len(means)):
        yerr[0, j] = abs(means[j] - q1s[j])  # Error inferior (positivo)
        yerr[1, j] = abs(q3s[j] - means[j])  # Error superior (positivo)
    ax.errorbar(x_pos + offsets[model], means, yerr=yerr, fmt='o', color=color, label=model, markersize=6, capsize=4, elinewidth=1.5)
ax.axhline(y=1/np.sqrt(3), color='black', linestyle=':', alpha=0.5, label=r'Balanced loading $\left( \frac{1}{\sqrt{3}} \right)$')
ax.set_title(f'a) PC1({avg_pc1:.1f}%)', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_ylabel('PC1 Loadings', fontsize=LABEL_SIZE)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.37), ncol=4, fontsize=LEGEND_SIZE)
ax.tick_params(axis='both', direction='in', which='both', labelsize=TICK_SIZE)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# PANEL 2: PC2 LOADINGS
# Recopilar loadings de PC2 por categoría
pc2_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
for cat in categories:
    # Recoge loadings de PC2 para todos los puntos en esta categoría
    cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
    for point_result in all_pca_results[cat]:
        if 'loading_pattern' in point_result and 'pc2' in point_result['loading_pattern']:
            if point_result['loading_pattern']['pc2'] is not None:
                pc2_loadings = point_result['loading_pattern']['pc2']
                for model in ['ETH', 'CNRM', 'CMCC']:
                    if model in pc2_loadings:
                        cat_loadings[model].append(pc2_loadings[model])
    
    # Si tenemos datos para esta categoría
    if all(len(values) > 0 for values in cat_loadings.values()):
        for model in ['ETH', 'CNRM', 'CMCC']:
            mean_loading = np.mean(cat_loadings[model])
            std_loading  = np.std(cat_loadings[model])
            q1_loading   = np.percentile(cat_loadings[model], 25)
            q3_loading   = np.percentile(cat_loadings[model], 75)
            
            pc2_loadings_data[model].append({'category': cat,
                                             'mean'    : mean_loading,
                                             'std'     : std_loading,
                                             'q1'      : q1_loading,
                                             'q3'      : q3_loading })

# Graficar PC2
ax = axs[1]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for model, color in model_colors.items():
    # Extraer valores medios y dispersión para PC2
    means = [data['mean'] for data in pc2_loadings_data[model]]
    q1s   = [data['q1'] for data in pc2_loadings_data[model]]
    q3s   = [data['q3'] for data in pc2_loadings_data[model]]    
    # Crear barras de error (asegurando que son positivas)
    yerr = np.zeros((2, len(means)))
    for j in range(len(means)):
        yerr[0, j] = abs(means[j] - q1s[j])  # Error inferior (positivo)
        yerr[1, j] = abs(q3s[j] - means[j])  # Error superior (positivo)
    ax.errorbar(x_pos + offsets[model], means, yerr=yerr, fmt='o', color=color, label=model, markersize=6, capsize=4, elinewidth=1.5)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title(f'b) PC2({avg_pc2:.1f}%)', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_ylabel('PC2 Loadings', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', direction='in', which='both', labelsize=TICK_SIZE)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# PANEL 3: PC3 LOADINGS
# Recopilar loadings de PC3 por categoría
pc3_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}

for cat_idx, cat in enumerate(categories):
    cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
    for point_result in all_pca_results[cat]:
        if 'loadings' in point_result and len(point_result['loadings']) >= 3:  # Asegurar que hay PC3
            pc3_loadings = point_result['loadings'][2]  # PC3 es el índice 2
            
            # Asignar loadings de PC3 a cada modelo
            if len(pc3_loadings) >= 3:  # Debe tener 3 valores (ETH, CNRM, CMCC)
                cat_loadings['ETH'].append(pc3_loadings[0])
                cat_loadings['CNRM'].append(pc3_loadings[1])
                cat_loadings['CMCC'].append(pc3_loadings[2])
    
    # Si tenemos datos suficientes para esta categoría
    if all(len(values) > 5 for values in cat_loadings.values()):  # Al menos 5 puntos con PC3
        for model in ['ETH', 'CNRM', 'CMCC']:
            mean_loading = np.mean(cat_loadings[model])
            std_loading  = np.std(cat_loadings[model])
            q1_loading   = np.percentile(cat_loadings[model], 25)
            q3_loading   = np.percentile(cat_loadings[model], 75)
            
            pc3_loadings_data[model].append({'category'    : cat,
                                             'category_idx': cat_idx,  # Guardar el índice original para mantener correspondencia
                                             'mean'        : mean_loading,
                                             'std'         : std_loading,
                                             'q1'          : q1_loading,
                                             'q3'          : q3_loading })

# Graficar PC3 (manteniendo el mismo orden que PC1 y PC2)
ax = axs[2]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for model, color in model_colors.items():
    cat_to_loading = {data['category_idx']: data for data in pc3_loadings_data[model]}    
    # Extraer valores medios y dispersión para PC3 en el mismo orden que PC1 y PC2
    means = []
    q1s   = []
    q3s   = []

    # Mantener solo x_pos para categorías con datos de PC3
    valid_x_pos   = []
    valid_offsets = []    
    for i, cat_idx in enumerate(range(len(categories))):
        if cat_idx in cat_to_loading:
            data = cat_to_loading[cat_idx]
            means.append(data['mean'])
            q1s.append(data['q1'])
            q3s.append(data['q3'])
            valid_x_pos.append(x_pos[i])
            valid_offsets.append(offsets[model])
    
    # Crear barras de error (asegurando que son positivas)
    yerr = np.zeros((2, len(means)))
    for j in range(len(means)):
        yerr[0, j] = abs(means[j] - q1s[j])  # Error inferior (positivo)
        yerr[1, j] = abs(q3s[j] - means[j])  # Error superior (positivo)    
    ax.errorbar(np.array(valid_x_pos) + np.array(valid_offsets), means, yerr=yerr, fmt='o', color=color, label=model, markersize=6, capsize=4, elinewidth=1.5)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title(f'c) PC3({avg_pc3:.1f}%)', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_xlabel('Spatial category', fontsize=LABEL_SIZE)
ax.set_ylabel('PC3 Loadings', fontsize=LABEL_SIZE)
ax.set_xticks(x_pos)
ax.set_xticklabels([cat_labels.get(cat, str(cat)) for cat in categories], rotation=30, ha='center', fontsize=TICK_SIZE) 
ax.tick_params(axis='both', direction='in', which='both', labelsize=TICK_SIZE)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Ocultar las etiquetas del eje x en los primeros dos paneles
plt.setp(axs[0].get_xticklabels(), visible=False)
plt.setp(axs[1].get_xticklabels(), visible=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.1, right=0.95, bottom=0.12, top=0.88)
plt.savefig(f"{bd_out_fig}PCAll_loadings_comparison.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()



# # fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)  # sharex=True para compartir eje X

# # # PANEL 1: PC1 LOADINGS
# # # Recopilar loadings de PC1 por categoría
# # pc1_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}

# # for cat in categories:
# #     # Recoge loadings de PC1 para todos los puntos en esta categoría
# #     cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
# #     for point_result in all_pca_results[cat]:
# #         if 'loading_pattern' in point_result and 'pc1' in point_result['loading_pattern']:
# #             pc1_loadings = point_result['loading_pattern']['pc1']
# #             for model in ['ETH', 'CNRM', 'CMCC']:
# #                 if model in pc1_loadings:
# #                     # Asegurar que todos los loadings tengan el mismo signo (positivo)
# #                     sign_correction = 1 if np.mean(list(pc1_loadings.values())) >= 0 else -1
# #                     cat_loadings[model].append(pc1_loadings[model] * sign_correction)
    
# #     # Si tenemos datos para esta categoría
# #     if all(len(values) > 0 for values in cat_loadings.values()):
# #         for model in ['ETH', 'CNRM', 'CMCC']:
# #             mean_loading = np.mean(cat_loadings[model])
# #             std_loading = np.std(cat_loadings[model])
# #             q1_loading = np.percentile(cat_loadings[model], 25)
# #             q3_loading = np.percentile(cat_loadings[model], 75)
            
# #             pc1_loadings_data[model].append({ 'category': cat,
# #                                                'mean'   : mean_loading,
# #                                                'std'    : std_loading,
# #                                                'q1'     : q1_loading,
# #                                                'q3'     : q3_loading })

# # # Graficar PC1
# # ax = axs[0]
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # for model, color in model_colors.items():
# #     means = [data['mean'] for data in pc1_loadings_data[model]]
# #     q1s   = [data['q1'] for data in pc1_loadings_data[model]]
# #     q3s   = [data['q3'] for data in pc1_loadings_data[model]]
    
# #     # Crear barras de error (asegurando que son positivas)
# #     yerr = np.zeros((2, len(means)))
# #     for j in range(len(means)):
# #         yerr[0, j] = abs(means[j] - q1s[j])  # Error inferior (positivo)
# #         yerr[1, j] = abs(q3s[j] - means[j])  # Error superior (positivo)
# #     ax.errorbar(x_pos + offsets[model], means, yerr=yerr, fmt='o', color=color, label=model, markersize=6, capsize=4, elinewidth=1.5)
# # ax.axhline(y=1/np.sqrt(3), color='black', linestyle=':', alpha=0.5, label=r'Balanced loading $\left( \frac{1}{\sqrt{3}} \right)$')
# # ax.set_title(f'PC1({avg_pc1:.1f}%)', fontsize=13, fontweight='bold')
# # ax.set_ylabel('PC1 Loadings', fontsize=13)
# # ax.set_xlim(-0.5, len(categories)-0.5)
# # ax.grid(True, axis='x', linestyle='--', alpha=0.3)
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=11)
# # ax.tick_params(axis='both', direction='in', which='both')
# # # No establecemos xticks aquí ya que sharex=True los compartirá

# # # PANEL 2: PC2 LOADINGS
# # # Recopilar loadings de PC2 por categoría
# # pc2_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
# # for cat in categories:
# #     # Recoge loadings de PC2 para todos los puntos en esta categoría
# #     cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
# #     for point_result in all_pca_results[cat]:
# #         if 'loading_pattern' in point_result and 'pc2' in point_result['loading_pattern']:
# #             if point_result['loading_pattern']['pc2'] is not None:
# #                 pc2_loadings = point_result['loading_pattern']['pc2']
# #                 for model in ['ETH', 'CNRM', 'CMCC']:
# #                     if model in pc2_loadings:
# #                         cat_loadings[model].append(pc2_loadings[model])
    
# #     # Si tenemos datos para esta categoría
# #     if all(len(values) > 0 for values in cat_loadings.values()):
# #         for model in ['ETH', 'CNRM', 'CMCC']:
# #             mean_loading = np.mean(cat_loadings[model])
# #             std_loading  = np.std(cat_loadings[model])
# #             q1_loading   = np.percentile(cat_loadings[model], 25)
# #             q3_loading   = np.percentile(cat_loadings[model], 75)
            
# #             pc2_loadings_data[model].append({'category': cat,
# #                                              'mean'    : mean_loading,
# #                                              'std'     : std_loading,
# #                                              'q1'      : q1_loading,
# #                                              'q3'      : q3_loading })

# # # Graficar PC2
# # ax = axs[1]
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # for model, color in model_colors.items():
# #     # Extraer valores medios y dispersión para PC2
# #     means = [data['mean'] for data in pc2_loadings_data[model]]
# #     q1s   = [data['q1'] for data in pc2_loadings_data[model]]
# #     q3s   = [data['q3'] for data in pc2_loadings_data[model]]    
# #     # Crear barras de error (asegurando que son positivas)
# #     yerr = np.zeros((2, len(means)))
# #     for j in range(len(means)):
# #         yerr[0, j] = abs(means[j] - q1s[j])  # Error inferior (positivo)
# #         yerr[1, j] = abs(q3s[j] - means[j])  # Error superior (positivo)
# #     ax.errorbar(x_pos + offsets[model], means, yerr=yerr, fmt='o', color=color, label=model, markersize=6, capsize=4, elinewidth=1.5)
# # ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
# # ax.set_title(f'PC2({avg_pc2:.1f}%)', fontsize=13, fontweight='bold')
# # ax.set_ylabel('PC2 Loadings', fontsize=13)
# # ax.set_ylim(-1, 1)  # Límites de -1 a 1 para PC2
# # ax.grid(True, axis='x', linestyle='--', alpha=0.3)
# # ax.tick_params(axis='both', direction='in', which='both')
# # # No establecemos xticks aquí ya que sharex=True los compartirá

# # # PANEL 3: PC3 LOADINGS
# # # Recopilar loadings de PC3 por categoría
# # pc3_loadings_data = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}

# # for cat_idx, cat in enumerate(categories):
# #     cat_loadings = {model: [] for model in ['ETH', 'CNRM', 'CMCC']}
    
# #     for point_result in all_pca_results[cat]:
# #         if 'loadings' in point_result and len(point_result['loadings']) >= 3:  # Asegurar que hay PC3
# #             pc3_loadings = point_result['loadings'][2]  # PC3 es el índice 2
            
# #             # Asignar loadings de PC3 a cada modelo
# #             if len(pc3_loadings) >= 3:  # Debe tener 3 valores (ETH, CNRM, CMCC)
# #                 cat_loadings['ETH'].append(pc3_loadings[0])
# #                 cat_loadings['CNRM'].append(pc3_loadings[1])
# #                 cat_loadings['CMCC'].append(pc3_loadings[2])
    
# #     # Si tenemos datos suficientes para esta categoría
# #     if all(len(values) > 5 for values in cat_loadings.values()):  # Al menos 5 puntos con PC3
# #         for model in ['ETH', 'CNRM', 'CMCC']:
# #             mean_loading = np.mean(cat_loadings[model])
# #             std_loading  = np.std(cat_loadings[model])
# #             q1_loading   = np.percentile(cat_loadings[model], 25)
# #             q3_loading   = np.percentile(cat_loadings[model], 75)
            
# #             pc3_loadings_data[model].append({'category'    : cat,
# #                                              'category_idx': cat_idx,  # Guardar el índice original para mantener correspondencia
# #                                              'mean'        : mean_loading,
# #                                              'std'         : std_loading,
# #                                              'q1'          : q1_loading,
# #                                              'q3'          : q3_loading })

# # # Graficar PC3 (manteniendo el mismo orden que PC1 y PC2)
# # ax = axs[2]
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # for model, color in model_colors.items():
# #     cat_to_loading = {data['category_idx']: data for data in pc3_loadings_data[model]}    
# #     # Extraer valores medios y dispersión para PC3 en el mismo orden que PC1 y PC2
# #     means = []
# #     q1s   = []
# #     q3s   = []

# #     # Mantener solo x_pos para categorías con datos de PC3
# #     valid_x_pos   = []
# #     valid_offsets = []    
# #     for i, cat_idx in enumerate(range(len(categories))):
# #         if cat_idx in cat_to_loading:
# #             data = cat_to_loading[cat_idx]
# #             means.append(data['mean'])
# #             q1s.append(data['q1'])
# #             q3s.append(data['q3'])
# #             valid_x_pos.append(x_pos[i])
# #             valid_offsets.append(offsets[model])
    
# #     # Crear barras de error (asegurando que son positivas)
# #     yerr = np.zeros((2, len(means)))
# #     for j in range(len(means)):
# #         yerr[0, j] = abs(means[j] - q1s[j])  # Error inferior (positivo)
# #         yerr[1, j] = abs(q3s[j] - means[j])  # Error superior (positivo)    
# #     ax.errorbar(np.array(valid_x_pos) + np.array(valid_offsets), means, yerr=yerr, fmt='o', color=color, label=model, markersize=6, capsize=4, elinewidth=1.5)
# # ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
# # ax.set_title(f'PC3({avg_pc3:.1f}%)', fontsize=13, fontweight='bold')
# # ax.set_xlabel('Spatial category', fontsize=13)  # Solo en el último subplot
# # ax.set_ylabel('PC3 Loadings', fontsize=13)
# # ax.set_ylim(-1, 1)  # Límites de -1 a 1 para PC3
# # ax.grid(True, axis='x', linestyle='--', alpha=0.3)
# # ax.set_xticks(x_pos)
# # ax.set_xticklabels([cat_labels.get(cat, str(cat)) for cat in categories], rotation=30, ha='center', fontsize=11)
# # ax.tick_params(axis='both', direction='in', which='both')

# # # Ocultar las etiquetas del eje x en los primeros dos paneles
# # plt.setp(axs[0].get_xticklabels(), visible=False)
# # plt.setp(axs[1].get_xticklabels(), visible=False)

# # plt.tight_layout()
# # plt.subplots_adjust(wspace=0.05, hspace=0.12, left=0.1, right=0.98, bottom=0.15, top=0.90) 
# # # plt.savefig(f"{bd_out_fig}PCAll_loadings_comparison.png", dpi=300, bbox_inches='tight', transparent=True)
# # plt.show()

