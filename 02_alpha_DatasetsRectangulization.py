
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from pathlib import Path

"""
Code to rectangularise and crop the CPM datasets in order to work with them in a comparable grid  
and domain in the spatial categories. This is because some members have a diagonalised domain, so
the cropping and rectagularisaiton helps to avoud edge effects.

Author : Nathalia Correa-Sánchez
Improved with Claude.ai
"""

########################################################################################
##-------------------------------DEFINING IMPORTANT PATHS-----------------------------##
########################################################################################

output_fig = "/Dati/Outputs/Plots/WP3_development/"
bd_in_ws   = "/Dati/Data/WS_CORDEXFPS/"
bd_in_eth  = bd_in_ws + "ETH/wsa100m/"
bd_in_cmcc = bd_in_ws + "CMCC/wsa100m/"
bd_in_cnrm = bd_in_ws + "CNRM/wsa100m/"
output_dir = "/Dati/Data/WS_CORDEXFPS/Rectangularized/"
band1_file = "/Dati/Outputs/Climate_Provinces/Development_Rasters/Combined_RIX_remCPM_WGS84.tif"
bd_in_band = "/Dati/Outputs/Climate_Provinces/Development_Rasters/"

# Definir las carpetas de salida para archivos rectangularizados y creacion de  directorios de salida si no existen
bd_out_eth  = bd_in_ws + "ETH/wsa100m_rect/"
bd_out_cmcc = bd_in_ws + "CMCC/wsa100m_rect/"
bd_out_cnrm = bd_in_ws + "CNRM/wsa100m_rect/"
for directory in [output_dir, bd_out_eth, bd_out_cmcc, bd_out_cnrm]:
    Path(directory).mkdir(exist_ok=True, parents=True)

# Definir las carpetas de salida para archivos cortados y creacion de  directorios de salida si no existen
bd_out_eth_crop  = bd_in_ws + "ETH/wsa100m_crop/"
bd_out_cmcc_crop = bd_in_ws + "CMCC/wsa100m_crop/"
bd_out_cnrm_crop = bd_in_ws + "CNRM/wsa100m_crop/"

# Crear directorios para datos recortados
for directory in [bd_out_eth_crop, bd_out_cmcc_crop, bd_out_cnrm_crop]:
    Path(directory).mkdir(exist_ok=True, parents=True)

########################################################################################
##-----------------------------SETTING RELEVANT PARAMETERS----------------------------##
########################################################################################
sample_datetime = '2009-01-01T01:00:00' # Str con el datetime a buscar en los files para mapear. Example '2009-01-01T01:00:00'
use_band1_grid  = False                 # Cambiar a True para usar band1 como referencia

# Coordenadas específicas para recorte geográfico. No establecemos límites superiores para mantener el resto del dominio. Diferente a lsode abajo
crop_coords = {'lon_min': 0.5,     # Límite occidental   : 0.5°E
               'lat_min': 40.2,    # Límite meridional   : 40.2°N
               'lon_max': 16.3,    # Límite oriental     : 16.3°E 
               'lat_max': 49.7  }  # Límite septentrional: 49.6°N
    
# Directorios de trabajo
input_dir = { 'ETH' : bd_in_eth,
              'CNRM': bd_in_cnrm,
              'CMCC': bd_in_cmcc}

output_rect_dir = {'ETH': bd_out_eth,
                   'CNRM': bd_out_cnrm,
                   'CMCC': bd_out_cmcc }

# Diccionario de directorios de salida para archivos recortados
output_crop_dir = { 'ETH': bd_out_eth_crop,
                    'CNRM': bd_out_cnrm_crop,
                    'CMCC': bd_out_cmcc_crop}


# Archivo de ejemplo por modelo
example_files = { 'ETH': os.path.join(input_dir['ETH'], "wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-ETH-COSMO-crCLIM_fpsconv-x2yn2-vrembil3km_1hr_200901010000-200912312300.nc"),
                  'CNRM': os.path.join(input_dir['CNRM'], "wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc"),
                 'CMCC': os.path.join(input_dir['CMCC'], "wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-CMCC-CCLM5-0-9_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc")}


# Parámetros para recorte de bordes problemáticos por indices - Se ajustarán automáticamente
# basados en el análisis de todos los modelos
crop_params = { 'x_min': 60,    # Valor inicial, se ajustará basado en CNRM
                'y_min': 25,    # Valor inicial, se ajustará basado en CNRM  
                'x_max': -1,    # Valor inicial, se ajustará si es necesario
                'y_max': -1 }   # Valor inicial, se ajustará si es necesario


########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################

def extract_griddes(input_file, output_griddes):
    """Extrae la descripción de la grilla de un archivo netCDF"""
    cmd = f"cdo griddes {input_file} > {output_griddes}"
    print(f"Ejecutando: {cmd}")
    os.system(cmd)
    return output_griddes

def create_reference_grid_from_band1(band1_file, output_grid):
    """Crea una grilla de referencia basada en el raster band1"""
    # Abrir el raster
    with rasterio.open(band1_file) as src:
        # Obtener las transformaciones geoespaciales
        transform = src.transform
        width     = src.width
        height    = src.height
        
        # Calcular coordenadas geográficas
        west  = transform[0]
        north = transform[3]
        east  = west + transform[1] * width
        south = north + transform[5] * height
        
        # Calcular resolución
        res_x = transform[1]
        res_y = abs(transform[5])
    
    # Crear archivo de descripción de grilla para CDO
    with open(output_grid, 'w') as f:
        f.write("gridtype = lonlat\n")
        f.write(f"xsize = {width}\n")
        f.write(f"ysize = {height}\n")
        f.write(f"xfirst = {west}\n")
        f.write(f"yfirst = {south}\n")
        f.write(f"xinc = {res_x}\n")
        f.write(f"yinc = {res_y}\n")
    
    return output_grid

def rectangularize_file(input_file, output_file, griddes_file):
    """Rectangulariza un archivo usando remapeo y fillmiss"""
    # Primero remapear a la grilla de referencia
    temp_file = output_file.replace('.nc', '_temp.nc')
    cmd1      = f"cdo remapcon,{griddes_file} {input_file} {temp_file}"
    print(f"Ejecutando: {cmd1}")
    os.system(cmd1)
    
    # Luego rellenar valores faltantes
    cmd2 = f"cdo fillmiss {temp_file} {output_file}"
    print(f"Ejecutando: {cmd2}")
    os.system(cmd2)
    
    # Eliminar archivo temporal
    os.remove(temp_file)
    
    return output_file

def analyze_missing_data(input_files, output_dir):
    """Analiza patrón de datos faltantes para determinar el recorte óptimo"""
    # Crear figura para visualización
    fig, axes = plt.subplots(1, len(input_files), figsize=(18, 6))
    
    # Recorte óptimo (valores iniciales altos)
    optimal_crop = { 'x_min': 0,
                    'y_min': 0,
                    'x_max': 99999,
                    'y_max': 99999}
    
    for i, (model, file) in enumerate(input_files.items()):
        # Cargar datos
        ds = xr.open_dataset(file)
        data = ds['wsa100m'].isel(time=0).values
        
        # Crear máscara de datos faltantes
        mask = np.isnan(data)
        
        # Analizar bordes
        # Desde izquierda
        for x in range(mask.shape[1]):
            if not np.all(mask[:, x]):
                left_edge = x
                break
        
        # Desde abajo
        for y in range(mask.shape[0]):
            if not np.all(mask[y, :]):
                bottom_edge = y
                break
        
        # Desde derecha
        for x in range(mask.shape[1]-1, -1, -1):
            if not np.all(mask[:, x]):
                right_edge = x
                break
        
        # Desde arriba
        for y in range(mask.shape[0]-1, -1, -1):
            if not np.all(mask[y, :]):
                top_edge = y
                break
        
        # Actualizar recorte óptimo
        optimal_crop['x_min'] = max(optimal_crop['x_min'], left_edge + 10)  # Añadir margen
        optimal_crop['y_min'] = max(optimal_crop['y_min'], bottom_edge + 10)
        optimal_crop['x_max'] = min(optimal_crop['x_max'], right_edge - 10)
        optimal_crop['y_max'] = min(optimal_crop['y_max'], top_edge - 10)
        
        # Visualizar máscara de datos
        im = axes[i].imshow(~mask, cmap='Blues')
        axes[i].set_title(f'Cobertura de datos - {model}')
        axes[i].axvline(left_edge, color='r', linestyle='--')
        axes[i].axvline(right_edge, color='r', linestyle='--')
        axes[i].axhline(bottom_edge, color='r', linestyle='--')
        axes[i].axhline(top_edge, color='r', linestyle='--')
        axes[i].text(5, 5, f"Bordes: L={left_edge}, R={right_edge}, B={bottom_edge}, T={top_edge}", 
                    color='white', backgroundcolor='black', transform=axes[i].transData)
        ds.close()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Rect_missing_data_analysis.png"), dpi=300)
    
    # Imprimir información del recorte óptimo
    print(f"\nRecorte óptimo determinado automáticamente:")
    print(f"  x_min: {optimal_crop['x_min']}")
    print(f"  y_min: {optimal_crop['y_min']}")
    print(f"  x_max: {optimal_crop['x_max']}")
    print(f"  y_max: {optimal_crop['y_max']}")
    
    return optimal_crop

def crop_borders(input_file, output_file, x_min, y_min, x_max=-1, y_max=-1):
    """Recorta los bordes problemáticos del archivo"""
    # Cargar datos
    ds = xr.open_dataset(input_file)
    
    # Determinar dimensiones completas
    dims = ds.dims
    nx   = dims['lon']
    ny   = dims['lat']
    
    # Calcular bordes si son -1
    if x_max == -1:
        x_max = nx
    if y_max == -1:
        y_max = ny
    
    # Crear y guardar la selección
    selector = {'lon': slice(x_min, x_max), 'lat': slice(y_min, y_max)}
    ds_cropped = ds.isel(**selector)
    ds_cropped.to_netcdf(output_file)
    ds.close()
    
    return output_file

def crop_by_coords(input_file, output_file, lon_min, lat_min, lon_max=None, lat_max=None):
    """
    Recorta un archivo netCDF por coordenadas geográficas.
    
    Parameters:
    -----------
    input_file : str
        Ruta del archivo de entrada
    output_file : str
        Ruta del archivo de salida
    lon_min, lat_min : float
        Coordenadas mínimas (límites inferiores)
    lon_max, lat_max : float, optional
        Coordenadas máximas (límites superiores), si no se especifican se mantienen
        hasta el final del dominio
    """
    # Cargar datos
    ds = xr.open_dataset(input_file)
    
    # Crear selección basada en coordenadas
    selection = {}
    
    if 'lon' in ds.dims:
        selection['lon'] = slice(lon_min, lon_max)
    if 'lat' in ds.dims:
        selection['lat'] = slice(lat_min, lat_max)
    
    # Aplicar selección y guardar el archivo recortado
    ds_cropped = ds.sel(**selection)
    ds_cropped.to_netcdf(output_file)
    ds.close()
    
    return output_file

def process_all_model_files(model, input_dir, output_rect_dir, griddes_file, crop_params):
    """
    Procesa todos los archivos de un modelo específico
    
    Parameters:
    -----------
    model : str
        Nombre del modelo ('ETH', 'CNRM', 'CMCC')
    input_dir : dict
        Diccionario con directorios de entrada por modelo
    output_rect_dir : dict
        Diccionario con directorios de salida por modelo
    griddes_file : str
        Ruta al archivo de descripción de grilla
    crop_params : dict
        Parámetros de recorte
    """
    # Listar todos los archivos del modelo
    files = [f for f in os.listdir(input_dir[model]) if f.endswith('.nc')]
    
    if len(files) == 0:
        print(f"No se encontraron archivos .nc en {input_dir[model]}")
        return
    
    print(f"\nProcesando {len(files)} archivos del modelo {model}...")
    
    # Crear directorio temporal para archivos intermedios
    temp_dir = os.path.join(output_dir, "temp")
    Path(temp_dir).mkdir(exist_ok=True)
    
    for i, filename in enumerate(files):
        input_file = os.path.join(input_dir[model], filename)
        
        # Archivo temporal para la fase de rectangularización - CORREGIDO
        rect_temp_file = os.path.join(temp_dir, f"temp_{model}_{i}.nc")
        
        # Archivo final con prefijo "rect_"
        output_file = os.path.join(output_rect_dir[model], f"rect_{filename}")
        
        print(f"  [{i+1}/{len(files)}] Procesando: {filename}")
        
        try:
            # Rectangularizar
            rectangularize_file(input_file, rect_temp_file, griddes_file)
            
            # Recortar bordes
            crop_borders(rect_temp_file, output_file, crop_params['x_min'], crop_params['y_min'], crop_params['x_max'], crop_params['y_max'])
            
            # Limpiar archivo temporal
            if os.path.exists(rect_temp_file):
                os.remove(rect_temp_file)
                
            print(f"    ✓ Guardado como: {output_file}")
            
        except Exception as e:
            print(f"    ✗ Error procesando {filename}: {str(e)}")
            
            # Limpiar archivo temporal en caso de error
            if os.path.exists(rect_temp_file):
                os.remove(rect_temp_file)


def verify_consistency():
    """Verifica que todos los archivos procesados tienen las mismas dimensiones"""
    print("\nVerificando consistencia de dimensiones en los archivos procesados:")
    
    # Diccionario para almacenar las dimensiones de referencia del primer archivo
    ref_dims = {}    
    for model in output_rect_dir.keys():
        files = [f for f in os.listdir(output_rect_dir[model]) if f.startswith("rect_") and f.endswith(".nc")]
        
        if not files:
            print(f"  - {model}: No se encontraron archivos procesados")
            continue
            
        print(f"  - {model}: Analizando {len(files)} archivos...")
        
        for i, file in enumerate(files):
            file_path = os.path.join(output_rect_dir[model], file)
            ds        = xr.open_dataset(file_path)            
            dims      = (ds.dims['lon'], ds.dims['lat'])
            
            # Si es el primer archivo que analizamos, guardamos sus dimensiones como referencia
            if not ref_dims:
                ref_dims['lon'] = dims[0]
                ref_dims['lat'] = dims[1]
                ref_model       = model
                ref_file        = file
            
            # Verificar consistencia con las dimensiones de referencia
            if dims[0] != ref_dims['lon'] or dims[1] != ref_dims['lat']:
                print(f"    ⚠️ INCONSISTENCIA: {file} tiene dimensiones {dims}, " 
                      f"diferentes de la referencia {(ref_dims['lon'], ref_dims['lat'])} "
                      f"de {ref_model}/{ref_file}")
            
            ds.close()
    
    print("  ✓ Verificación completada")

def process_crop_files(model, input_rect_dir, output_crop_dir, crop_coords):
    """
    Recorta todos los archivos rectangularizados según coordenadas geográficas específicas.
    
    Parameters:
    -----------
    model : str
        Nombre del modelo ('ETH', 'CNRM', 'CMCC')
    input_rect_dir : dict
        Diccionario con directorios de entrada (archivos rectangularizados)
    output_crop_dir : dict
        Diccionario con directorios de salida para archivos recortados
    crop_coords : dict
        Coordenadas para recorte (lon_min, lat_min, opcionalmente lon_max, lat_max)
    """
    # Listar todos los archivos rectangularizados
    rect_files = [f for f in os.listdir(input_rect_dir[model]) if f.startswith("rect_") and f.endswith(".nc")]
    
    if len(rect_files) == 0:
        print(f"No se encontraron archivos rectangularizados en {input_rect_dir[model]}")
        return
    
    print(f"\nRecortando {len(rect_files)} archivos del modelo {model} a partir de {crop_coords['lon_min']}°E y {crop_coords['lat_min']}°N...")
    
    for i, filename in enumerate(rect_files):
        input_file = os.path.join(input_rect_dir[model], filename)
        
        # Nombre del archivo de salida (reemplazando 'rect_' con 'crop_')
        output_filename = filename.replace("rect_", "crop_")
        output_file     = os.path.join(output_crop_dir[model], output_filename)
        
        print(f"  [{i+1}/{len(rect_files)}] Recortando: {filename}")
        
        try:
            # Recortar por coordenadas
            crop_by_coords(input_file, output_file, crop_coords['lon_min'], crop_coords['lat_min'], crop_coords['lon_max'], crop_coords['lat_max'])
                
            print(f"    ✓ Guardado como: {output_file}")
            
        except Exception as e:
            print(f"    ✗ Error recortando {filename}: {str(e)}")

def crop_raster_by_coords(input_raster, output_raster, lon_min, lat_min, lon_max=None, lat_max=None):
    """
    Recorta un archivo raster GeoTIFF según coordenadas geográficas.
    
    Parameters:
    -----------
    input_raster : str
        Ruta del raster de entrada
    output_raster : str
        Ruta del raster de salida
    lon_min, lat_min : float
        Coordenadas mínimas (límites inferiores)
    lon_max, lat_max : float, optional
        Coordenadas máximas (límites superiores)
    """
    import rasterio
    from rasterio.windows import from_bounds
    
    with rasterio.open(input_raster) as src:
        # Determinar límites superiores si no se especifican
        if lon_max is None:
            lon_max = src.bounds.right
        if lat_max is None:
            lat_max = src.bounds.top
            
        # Crear ventana para recorte
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        
        # Leer datos de la ventana
        data = src.read(1, window=window)
        
        # Actualizar el transform para la nueva ventana
        transform = rasterio.windows.transform(window, src.transform)
        
        # Crear perfil para el archivo de salida
        profile = src.profile.copy()
        profile.update({ 'height'   : data.shape[0],
                         'width'    : data.shape[1],
                         'transform': transform  })
        
        # Guardar archivo recortado
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(data, 1)
            
    return output_raster

def verify_crop_consistency():
    """Verifica que todos los archivos recortados tienen el mismo dominio geográfico"""
    print("\nVerificando consistencia del dominio geográfico en los archivos recortados:")
    
    # Referencias para comparación
    ref_lon_min = None
    ref_lon_max = None
    ref_lat_min = None
    ref_lat_max = None
    ref_model   = None
    ref_file    = None
    
    for model in output_crop_dir.keys():
        files = [f for f in os.listdir(output_crop_dir[model]) if f.startswith("crop_") and f.endswith(".nc")]
        
        if not files:
            print(f"  - {model}: No se encontraron archivos recortados")
            continue
            
        print(f"  - {model}: Analizando {len(files)} archivos...")
        
        for i, file in enumerate(files):
            file_path = os.path.join(output_crop_dir[model], file)
            ds        = xr.open_dataset(file_path)
            
            # Extraer límites geográficos
            lon_min = float(ds.lon.min())
            lon_max = float(ds.lon.max())
            lat_min = float(ds.lat.min())
            lat_max = float(ds.lat.max())
            
            # Si es el primer archivo, establecer referencia
            if ref_lon_min is None:
                ref_lon_min = lon_min
                ref_lon_max = lon_max
                ref_lat_min = lat_min
                ref_lat_max = lat_max
                ref_model   = model
                ref_file    = file
                continue
            
            # Verificar consistencia
            tolerance = 1e-6  # Tolerancia para comparación de flotantes
            if (abs(lon_min - ref_lon_min) > tolerance or 
                abs(lon_max - ref_lon_max) > tolerance or 
                abs(lat_min - ref_lat_min) > tolerance or 
                abs(lat_max - ref_lat_max) > tolerance):
                print(f"    ⚠️ INCONSISTENCIA: {file} tiene dominio [{lon_min:.4f}E, {lon_max:.4f}E, {lat_min:.4f}N, {lat_max:.4f}N], " 
                      f"diferente de la referencia [{ref_lon_min:.4f}E, {ref_lon_max:.4f}E, {ref_lat_min:.4f}N, {ref_lat_max:.4f}N] "
                      f"de {ref_model}/{ref_file}")
            
            ds.close()
    
    if ref_lon_min is not None:
        print(f"  ✓ Dominio geográfico de referencia: [{ref_lon_min:.4f}E, {ref_lon_max:.4f}E, {ref_lat_min:.4f}N, {ref_lat_max:.4f}N]")
    print("  ✓ Verificación completada")

########################################################################################
### -----------------------MAIN EXECUTION ROUTINE FOR EXAMPLE FILES------------------###
########################################################################################


# 1. Extraer griddes del raster band1 o usar ETH como referencia
if use_band1_grid:
    # Crear grilla de referencia basada en band1
    griddes_file = os.path.join(output_dir, "band1_griddes.txt")
    create_reference_grid_from_band1(band1_file, griddes_file)
else:
    # Usar ETH como referencia
    griddes_file = os.path.join(output_dir, "eth_griddes.txt")
    extract_griddes(example_files['ETH'], griddes_file)

# 2. Rectangularizar cada modelo ejemplo
rect_files = {}

for model, input_file in example_files.items():
    print(f"\nProcesando rectangularización de ejemplo de {model}...")
    
    # Determinar archivos de salida
    rect_file = os.path.join(output_dir, f"{model}_Rect.nc")
    
    # Rectangularizar TODOS los modelos
    rectangularize_file(input_file, rect_file, griddes_file)
    
    rect_files[model] = rect_file

# 3. Analizar datos faltantes para determinar recorte óptimo con los archivos de ejemplo
optimal_crop = analyze_missing_data(rect_files, output_dir)

# Actualizar parámetros de recorte
crop_params.update(optimal_crop)

# 4. Aplicar recorte a los modelos ejemplo
processed_files = {}

for model, rect_file in rect_files.items():
    print(f"\nAplicando recorte a ejemplo de {model}...")
    
    # Determinar archivo final y recortar bordes
    final_file = os.path.join(output_dir, f"{model}_rectangular_cropped.nc")
    crop_borders(rect_file, final_file,  crop_params['x_min'], crop_params['y_min'], crop_params['x_max'], crop_params['y_max']) 
    processed_files[model] = final_file

# 5. Generar mapas comparativos
map_file = os.path.join(output_fig, "Rect_model_comparison_with_band1.png")
# plot_comparison(processed_files, band1_file, map_file)

# Verificar alineación dimensional
print("\nVerificando alineación dimensional de archivos de ejemplo:")
for model, file in processed_files.items():
    ds = xr.open_dataset(file)
    print(f"  - {model}: {ds.dims}")
    ds.close()

# 6. Procesar todos los archivos de cada modelo
print("\n" + "="*80)
print("INICIANDO PROCESAMIENTO DE TODOS LOS ARCHIVOS")
print("="*80)

for model in input_dir.keys():
    process_all_model_files(model, input_dir, output_rect_dir, griddes_file, crop_params)

print("\n" + "="*80)
print("PROCESAMIENTO COMPLETADO")
print("="*80)
print("\nArchivos rectangularizados guardados en:")
for model in input_dir.keys():
    print(f"  - {model}: {output_rect_dir[model]}")

verify_consistency()

########################################################################################
### -----------------GEOGRAPHICAL CROPPING FOR CONSISTENT DOMAIN---------------------###
########################################################################################

print("\n" + "="*80)
print("INICIANDO RECORTE GEOGRÁFICO DE ARCHIVOS")
print("="*80)

# 1. Recortar el raster band1
band1_crop_file = os.path.join(bd_in_band, "band1_cropped.tif")
print(f"\nRecortando raster de categorías espaciales...")
try:
    crop_raster_by_coords(band1_file, band1_crop_file, crop_coords['lon_min'], crop_coords['lat_min'], crop_coords['lon_max'], crop_coords['lat_max'])
    print(f"  ✓ Raster recortado guardado como: {band1_crop_file}")
except Exception as e:
    print(f"  ✗ Error recortando raster: {str(e)}")

# 2. Recortar todos los archivos rectangularizados
for model in input_dir.keys():
    process_crop_files(model, output_rect_dir, output_crop_dir, crop_coords)

print("\n" + "="*80)
print("RECORTE GEOGRÁFICO COMPLETADO")
print("="*80)
print("\nArchivos recortados guardados en:")
for model in input_dir.keys():
    print(f"  - {model}: {output_crop_dir[model]}")

# 3. Verificar consistencia de los archivos recortados
verify_crop_consistency()

########################################################################################
### ----------------PREPARING ORIGINAL AND RECTANGULISED DATA FOR MAPPING------------###
########################################################################################

with rasterio.open(band1_file) as src:
    band1_data                 = src.read(1)
    band1_data[band1_data < 0] = np.nan          ## Reemplazando los negativos con nan o 0(Tener en cuenta NoData= -3.40282e+38) 
    band1_transform            = src.transform
    band1_crs                  = src.crs
    
    # Calcular extent correctamente
    west  = band1_transform[2]  # x_0
    east  = west + band1_transform[0] * src.width  # x_0 + width * pixel_width
    north = band1_transform[5]  # y_0
    south = north + band1_transform[4] * src.height  # y_0 + height * pixel_height
    
    # Corregir si north y south están invertidos (común con coordenadas y negativas)
    if north < south:
        north, south = south, north

unique_vals    = np.unique(band1_data)
unique_vals    = unique_vals[np.isfinite(unique_vals)]
num_categories = len(unique_vals)

# Crear un colormap discreto y normalizar
if num_categories <= 20:
    cmap = plt.get_cmap('tab20', num_categories)
else:
    cmap = plt.get_cmap('nipy_spectral', num_categories)
bounds = np.arange(num_categories + 1)
norm   = BoundaryNorm(bounds, cmap.N) ## Para ax1. Con esto cada valor en bounbds tiene un color

band1_categorized       = np.full(band1_data.shape, np.nan) # Crea un array del mismo tamaño que band1 lleno de NaN
mask                    = np.isfinite(band1_data)           # Mascara de valores no NaN - necesario para la colorbar
band1_categorized[mask] = np.digitize(band1_data[mask], bins=unique_vals, right=True) - 1 #Categoriza solo los valores no NaN

# Archivo de orginales por modelo
original_files = { 'ETH' : os.path.join(input_dir['ETH'], "wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-ETH-COSMO-crCLIM_fpsconv-x2yn2-vrembil3km_1hr_200901010000-200912312300.nc"),
                   'CNRM': os.path.join(input_dir['CNRM'], "wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc"),
                   'CMCC': os.path.join(input_dir['CMCC'], "wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-CMCC-CCLM5-0-9_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc")}

# Archivo de rectangulares por modelo
rectangular_files = { 'ETH' : os.path.join(output_rect_dir['ETH'], "rect_wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-ETH-COSMO-crCLIM_fpsconv-x2yn2-vrembil3km_1hr_200901010000-200912312300.nc"),
                      'CNRM': os.path.join(output_rect_dir['CNRM'], "rect_wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc"),
                      'CMCC': os.path.join(output_rect_dir['CMCC'], "rect_wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-CMCC-CCLM5-0-9_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc")}


########################################################################################
### -------------COMPARATIVE MAPS TO ILUSTRATE THE PROCEDURE-ORIGINALS---------------###
########################################################################################
# Dominio geográfico de referencia para el corte: [0.5046E, 16.2746E, 40.2170N, 49.6940N] 
# (diferente a las coordenas de corte porque los primeros eran valores cercanos y esto los reales)
# obtenidos con verify_crop_consistency()

fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(2, 2)

ax1  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
img1 = ax1.imshow(band1_categorized, extent=[west, east, south, north], origin="upper", cmap=cmap, norm=norm, alpha=0.7, transform=ccrs.PlateCarree())
ax1.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.set_title('Spatial categories', fontsize=12, fontweight ="bold")

rect1 = plt.Rectangle((0.5046, 40.2170), 16.2746-0.5046, 49.6940-40.2170, edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree())
ax1.add_patch(rect1)

cbar = plt.colorbar(img1, ax=ax1, shrink=0.9, ticks=bounds[:-1] + 0.5)
cbar.set_ticks([])  # Elimina los ticks numéricos
cbar.set_label("Spatial category (Total :"+str(num_categories)+")", fontsize=10)

positions = [(0, 1), (1, 0), (1, 1)]
for i, (model, file) in enumerate(original_files.items()):
    ds   = xr.open_dataset(file)
    data = ds['wsa100m'].sel(time=np.datetime64(sample_datetime)).values

    lons               = ds.lon.values
    lats               = ds.lat.values
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    ax  = fig.add_subplot(gs[positions[i]], projection=ccrs.PlateCarree())    
    img = ax.pcolormesh(lon_mesh, lat_mesh, data, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())    
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(f'Model {model} (Original)', fontsize=12, fontweight ="bold")
    rect = plt.Rectangle((0.5046, 40.2170), 16.2746-0.5046, 49.6940-40.2170, edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree())
    ax.add_patch(rect)
    cbar1 = plt.colorbar(img, ax=ax, shrink=0.9) 
    cbar1.set_label("Wind speeds [m/s]", fontsize=10)   
    ds.close()

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.25, left=0.10, right=0.88, bottom=0.10, top=0.89)
plt.savefig(output_fig+ "Org_CORDEXdomain_Maps.png", dpi=300, bbox_inches='tight')
plt.show()

########################################################################################
### ---------COMPARATIVE MAPS TO ILUSTRATE THE PROCEDURE-RECTANGULISED---------------###
########################################################################################

# Dominio geográfico de referencia para el corte: [0.5046E, 16.2746E, 40.2170N, 49.6940N]

fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(2, 2)

ax1  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
img1 = ax1.imshow(band1_categorized, extent=[west, east, south, north], origin="upper", cmap=cmap, norm=norm, alpha=0.7, transform=ccrs.PlateCarree())
ax1.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.set_title('Spatial categories', fontsize=12, fontweight ="bold")

rect1 = plt.Rectangle((0.5046, 40.2170), 16.2746-0.5046, 49.6940-40.2170, edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree())
ax1.add_patch(rect1)

cbar = plt.colorbar(img1, ax=ax1, shrink=0.9, ticks=bounds[:-1] + 0.5)
cbar.set_ticks([])  # Elimina los ticks numéricos
cbar.set_label("Spatial category (Total :"+str(num_categories)+")", fontsize=10)

positions = [(0, 1), (1, 0), (1, 1)]
for i, (model, file) in enumerate(rectangular_files.items()):
    ds   = xr.open_dataset(file)
    data = ds['wsa100m'].sel(time=np.datetime64(sample_datetime)).values

    lons               = ds.lon.values
    lats               = ds.lat.values
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    ax  = fig.add_subplot(gs[positions[i]], projection=ccrs.PlateCarree())    
    img = ax.pcolormesh(lon_mesh, lat_mesh, data, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())    
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(f'Model {model} (Rectangulised)', fontsize=12, fontweight ="bold")
    rect = plt.Rectangle((0.5046, 40.2170), 16.2746-0.5046, 49.6940-40.2170, edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree())
    ax.add_patch(rect)
    cbar1 = plt.colorbar(img, ax=ax, shrink=0.9) 
    cbar1.set_label("Wind speeds [m/s]", fontsize=10)   
    ds.close()

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.25, left=0.10, right=0.88, bottom=0.10, top=0.89)
plt.savefig(output_fig+ "Rect_CORDEXdomain_Maps.png", dpi=300, bbox_inches='tight')
plt.show()

########################################################################################
### -----------------------------PREPARING CROPPED DATA FOR MAPPING------------------###
########################################################################################

# Preparar datos para el raster recortado
with rasterio.open(band1_crop_file) as src:
    band1_crop_data                      = src.read(1)
    band1_crop_data[band1_crop_data < 0] = np.nan
    band1_transform                      = src.transform
    
    # Calcular nuevo extent
    west  = band1_transform[2]
    east  = west + band1_transform[0] * src.width
    north = band1_transform[5]
    south = north + band1_transform[4] * src.height
    
    # Corregir si north y south están invertidos
    if north < south:
        north, south = south, north

# Preparar colormap para categorías
unique_vals    = np.unique(band1_crop_data)
unique_vals    = unique_vals[np.isfinite(unique_vals)]
num_categories = len(unique_vals)

if num_categories <= 20:
    cmap = plt.get_cmap('tab20', num_categories)
else:
    cmap = plt.get_cmap('nipy_spectral', num_categories)
bounds = np.arange(num_categories + 1)
norm   = BoundaryNorm(bounds, cmap.N)

band1_categorized       = np.full(band1_crop_data.shape, np.nan)
mask                    = np.isfinite(band1_crop_data)
band1_categorized[mask] = np.digitize(band1_crop_data[mask], bins=unique_vals, right=True) - 1

# Crear rutas a los archivos recortados
crop_files = { 'ETH' : os.path.join(output_crop_dir['ETH'], "crop_wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-ETH-COSMO-crCLIM_fpsconv-x2yn2-vrembil3km_1hr_200901010000-200912312300.nc"),
               'CNRM': os.path.join(output_crop_dir['CNRM'], "crop_wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc"),
               'CMCC': os.path.join(output_crop_dir['CMCC'], "crop_wsa100m_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-CMCC-CCLM5-0-9_fpsconv-x2yn2-vrembil3km_1hr_200901010030-200912312330.nc")}

########################################################################################
### ----------COMPARATIVE MAPS TO ILUSTRATE THE CROPPED DATASETS---------------------###
########################################################################################

fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(2, 2)

ax1  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
img1 = ax1.imshow(band1_categorized, extent=[west, east, south, north], origin="upper", cmap=cmap, norm=norm, alpha=0.7, transform=ccrs.PlateCarree())
ax1.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.set_title('Spatial categories (Cropped Domain)', fontsize=12, fontweight="bold")
cbar = plt.colorbar(img1, ax=ax1, shrink=0.9, ticks=bounds[:-1] + 0.5)
cbar.set_ticks([])
cbar.set_label(f"Spatial category (Total: {num_categories})", fontsize=10)

positions = [(0, 1), (1, 0), (1, 1)]
for i, (model, file) in enumerate(crop_files.items()):
    ds  = xr.open_dataset(file)
    data = ds['wsa100m'].sel(time=np.datetime64(sample_datetime)).values

    lons               = ds.lon.values
    lats               = ds.lat.values
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    ax  = fig.add_subplot(gs[positions[i]], projection=ccrs.PlateCarree())
    img = ax.pcolormesh(lon_mesh, lat_mesh, data, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(f'Model {model} (Cropped Domain)', fontsize=12, fontweight="bold")        
    cbar1 = plt.colorbar(img, ax=ax, shrink=0.9)
    cbar1.set_label("Wind speeds [m/s]", fontsize=10)
    
    ds.close()

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.25, left=0.10, right=0.88, bottom=0.10, top=0.89)
plt.savefig(output_fig+ "Crop_CORDEXdomain_Maps.png", dpi=300, bbox_inches='tight')
plt.show()



