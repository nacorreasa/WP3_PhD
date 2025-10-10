import xarray as xr
import numpy as np
from pathlib import Path
import gc
import logging
from datetime import datetime

"""
Code to compute the spatial mean, at gid point level, of the CPM members to built a ensemble mean dataset.

Author: Nathalia Correa-Sánchez  - Improved with Claude.ai

"""

#############################################################################
##-------------------------DEFINNING RELEVANT INPUTS-----------------------##
#############################################################################

years_file_list  = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
models_file_list = ["ETH", "CMCC", "CNRM"]

#############################################################################
##-------------------------DEFINNING IMPORTANT PATHS-----------------------##
#############################################################################

bd_in = "/Dati/Data/WS_CORDEXFPS"

#############################################################################
##------------------------CONFIGURING THE LOGGING--------------------------##
#############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#############################################################################
##---------------------DEFINING RELEVANT FUNCITONS-------------------------##
#############################################################################

def setup_paths():
    """Configura las rutas base y crea el directorio de salida si no existe."""
    base_path  = Path(bd_in)
    models     = models_file_list
    output_dir = base_path / "Ensemble_Mean" / "wsa100m"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return base_path, models, output_dir

def get_file_paths(base_path, models):
    """Genera diccionario de rutas de archivos por año para cada modelo."""
    file_paths = {}
    years_list = years_file_list
    # for year in range(2000, 2010):
    for year in years_list :
        file_paths[year] = {
            model: list(Path(base_path / model / "wsa100m").glob(f"*{year}*.nc"))
            for model in models
        }
    return file_paths

def process_year(year, model_files, output_dir):
    """Procesa los archivos de un año específico para todos los modelos."""
    logger.info(f"Procesando año {year}")
    
    # Lista para almacenar los datasets de cada modelo
    datasets = []
    
    # Cargar datos de cada modelo
    for model, files in model_files[year].items():
        if not files:
            logger.warning(f"No se encontraron archivos para {model} en el año {year}")
            continue
            
        try:
            # Cargar datos con dask para procesamiento diferido
            ds = xr.open_mfdataset(
                files,
                combine='by_coords',
                parallel=True,
                chunks={'time': 'auto'}
            )
            datasets.append(ds)
            logger.info(f"Cargado dataset para {model} año {year}")
        except Exception as e:
            logger.error(f"Error al cargar archivos de {model} año {year}: {e}")
            continue
    
    if not datasets:
        logger.error(f"No se pudieron cargar datos para el año {year}")
        return
    
    try:
        # Calcular media del ensemble
        ensemble_mean = xr.concat(datasets, dim='model').mean(dim='model')
        
        # Liberar memoria
        for ds in datasets:
            ds.close()
        datasets.clear()
        gc.collect()
        
        # Guardar resultado
        output_file  = output_dir / f"ensemble_mean_ws100m_{year}.nc"
        ensemble_mean.to_netcdf(
            output_file,
            encoding = {var: {'zlib': True, 'complevel': 5} for var in ensemble_mean.data_vars}
        )
        logger.info(f"Guardado archivo de salida para {year}: {output_file}")
        
        # Liberar memoria del ensemble
        ensemble_mean.close()
        del ensemble_mean
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error al procesar el año {year}: {e}")

#############################################################################
##-------------------EXEUTTING PROCESS MAIN FUNCTION-----------------------##
#############################################################################

def main():
    """Función principal que coordina el procesamiento."""
    start_time = datetime.now()
    logger.info("Iniciando procesamiento")
    
    try:
        # Configurar rutas
        base_path, models, output_dir = setup_paths()
        
        # Obtener rutas de archivos
        file_paths = get_file_paths(base_path, models)
        
        years_list = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
        # Procesar cada año
        for year in years_list:
            process_year(year, file_paths, output_dir)
            
        end_time = datetime.now()
        logger.info(f"Procesamiento completado. Tiempo total: {end_time - start_time}")
        
    except Exception as e:
        logger.error(f"Error en el procesamiento principal: {e}")

if __name__ == "__main__":
    main()