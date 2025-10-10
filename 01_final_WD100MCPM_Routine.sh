#!/bin/bash

# NOTAS de Nathalia: 
# 1) Se necesita tener instalado CDO 
# 2) Hay que darle permisos de ejecición así: chmod +x Alpha_WS100MCPM_DataRetreiveRoutine.sh
# 3) El archivo targetgrid.txt (para el bilinear remapping) debe estar en "${BASE_DIR}/${model}/wda100m/targetgrid.txt"

# Configuración de logging
exec 1> >(tee "process_winddir_data.log")
exec 2>&1

# # echo "=== Iniciando proceso de descarga y procesamiento de datos de viento ==="
echo "Fecha y hora de inicio: $(date)"

# Definir variables globales iniciales
BASE_DIR="/Dati/Data/WS_CORDEXFPS"
MODELS=("ETH")
START_YEAR=2000
END_YEAR=2009

# Función para mostrar el progreso
progress() {
    echo "===== $1 ====="
}

# Función para verificar si un comando se ejecutó correctamente
check_error() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1"
        exit 1
    fi
}

# Función para crear directorios necesarios
create_directories() {
    local model=$1
    mkdir -p "${BASE_DIR}/${model}/ua100m"
    mkdir -p "${BASE_DIR}/${model}/va100m"
    mkdir -p "${BASE_DIR}/${model}/wda100m"
    mkdir -p "${BASE_DIR}/${model}/wda100m_org"
}


# Función para calcular la direccion del viento
calculate_wind_direction() {
    local model=$1
    local year=$2
    
    progress "Calculando direccion del viento para ${model} año ${year}"
    
    local u_file=$(find "${BASE_DIR}/${model}/ua100m" -type f -name "*${year}*.nc")
    local v_file=$(find "${BASE_DIR}/${model}/va100m" -type f -name "*${year}*.nc")
    
    if [ -z "$u_file" ] || [ -z "$v_file" ]; then
        echo "WARNING: Archivos no encontrados para ${model} año ${year}"
        return 1
    fi
    
    local output_file="${BASE_DIR}/${model}/wda100m_org/wda100m_$(basename "$u_file" | sed 's/ua100m/wda100m/')"
    
    echo "Calculando direccion del viento para año $year..."
    cdo -f nc4 -z zip_9 \
        mulc,57.3 -atan2 -mulc,-1 \
        "$u_file" -mulc,-1 "$v_file" \
        "$output_file"
    
    check_error "Error al calcular direccion del viento para ${model} año ${year}"
}

# Función para hacer el remapbil
do_remapbil() {
    local model=$1
    local year=$2
    
    progress "Realizando remapbil para ${model} año ${year}"
    
    local input_file="${BASE_DIR}/${model}/wda100m_org/wda100m_*${year}*.nc"
    local output_file="${BASE_DIR}/${model}/wda100m/wda100m_$(basename "$input_file" | sed 's/_org//')"
    local targetgrid_file="${BASE_DIR}/${model}/wda100m/targetgrid.txt"


    echo "Aplicando remapbil para año $year..."
    cdo remapbil,"$targetgrid_file"  "$input_file" "$output_file"
    
    check_error "Error en remapbil para ${model} año ${year}"
}

# Función principal
main() {

    echo
    
    # Procesar cada modelo
    for model in "${MODELS[@]}"; do
        progress "Procesando modelo $model"
        
        # Crear directorios
        create_directories "$model"
        
        # Procesar cada año
        for year in $(seq $START_YEAR $END_YEAR); do
            calculate_wind_direction "$model" "$year"
            do_remapbil "$model" "$year"
        done
        
        # Limpiar directorios temporales
        progress "Limpiando directorios temporales para $model"
        rm -rf "${BASE_DIR}/${model}/ua100m"
        rm -rf "${BASE_DIR}/${model}/va100m"
        rm -rf "${BASE_DIR}/${model}/wda100m_org"
    done
    
    progress "Proceso completado exitosamente"
    echo "Fecha y hora de finalización: $(date)"
}

# Ejecutar script principal con manejo de errores
main "$@"

