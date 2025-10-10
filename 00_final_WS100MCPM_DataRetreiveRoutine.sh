#!/bin/bash

# NOTAS de Nathalia: 
# 1) Se necesita tener instalado CDO y expect 
# 2) Hay que darle permisos de ejecición así: chmod +x Alpha_WS100MCPM_DataRetreiveRoutine.sh
# 3) El archivo targetgrid.txt (para el bilinear remapping) debe estar en la carpeta de ejecución.
# 4) Los archivos wget_script_XXX_XXX_XX.sh deben existir previamente (Descargados de: https://esgf.nci.org.au/login/)

# Configuración de logging
exec 1> >(tee "process_wind_data.log")
exec 2>&1

echo "=== Iniciando proceso de descarga y procesamiento de datos de viento ==="
echo "Fecha y hora de inicio: $(date)"

# Definir variables globales iniciales
BASE_DIR="/Dati/Data/WS_CORDEXFPS"
MODELS=("CNRM" "CMCC" "ETH")
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
    mkdir -p "${BASE_DIR}/${model}/wsa100m"
    mkdir -p "${BASE_DIR}/${model}/wsa100m_org"
}

# Función para descargar datos
download_data() {
    local model=$1
    local component=$2
    local script_path="${BASE_DIR}/${model}/wget_script_2024-10-22_${component}100m_${model}.sh"
    
    if [ ! -f "$script_path" ]; then
        echo "ERROR: Script de descarga no encontrado: $script_path"
        return 1
    }
    
    progress "Descargando datos ${component}100m para ${model}"
    
    # Hacer ejecutable el script
    chmod +x "$script_path"
    
    # Ejecutar script de descarga con credenciales usando expect
    expect << EOF
        spawn bash $script_path -H
        expect "Enter your OpenID user name:"
        send "$userOpenID\r"
        expect "Enter your OpenID password:"
        send "$passOpenID\r"
        expect eof
EOF


# Función para limpiar archivos fuera del rango temporal
clean_files() {
    local dir=$1
    
    progress "Limpiando archivos fuera del rango temporal en $dir"
    
    find "$dir" -type f -name "*.nc" | while read file; do
        year=$(echo "$file" | grep -o "2[0-9]\{3\}" | head -1)
        if [ ! -z "$year" ] && ([ "$year" -lt $START_YEAR ] || [ "$year" -gt $END_YEAR ]); then
            echo "Eliminando archivo fuera de rango: $file"
            rm -f "$file"
        fi
    done
}

# Función para calcular la velocidad del viento
calculate_wind_speed() {
    local model=$1
    local year=$2
    
    progress "Calculando velocidad del viento para ${model} año ${year}"
    
    local u_file=$(find "${BASE_DIR}/${model}/ua100m" -type f -name "*${year}*.nc")
    local v_file=$(find "${BASE_DIR}/${model}/va100m" -type f -name "*${year}*.nc")
    
    if [ -z "$u_file" ] || [ -z "$v_file" ]; then
        echo "WARNING: Archivos no encontrados para ${model} año ${year}"
        return 1
    fi
    
    local output_file="${BASE_DIR}/${model}/wsa100m_org/wsa100m_$(basename "$u_file" | sed 's/ua100m/wsa100m/')"
    
    echo "Calculando velocidad del viento para año $year..."
    cdo -f nc4 -z zip_9 -expr,'wsa100m=sqrt(ua100m*ua100m + va100m*va100m)' \
        -merge -selname,ua100m "$u_file" -selname,va100m "$v_file" "$output_file"
    
    check_error "Error al calcular velocidad del viento para ${model} año ${year}"
}

# Función para hacer el remapbil
do_remapbil() {
    local model=$1
    local year=$2
    
    progress "Realizando remapbil para ${model} año ${year}"
    
    local input_file="${BASE_DIR}/${model}/wsa100m_org/wsa100m_*${year}*.nc"
    local output_file="${BASE_DIR}/${model}/wsa100m/wsa100m_$(basename "$input_file" | sed 's/_org//')"
    
    echo "Aplicando remapbil para año $year..."
    cdo remapbil,targetgrid.txt "$input_file" "$output_file"
    
    check_error "Error en remapbil para ${model} año ${year}"
}

# Función principal
main() {
    # Solicitar credenciales
    echo "Por favor, ingrese sus credenciales de OpenID:"
    read -p "Usuario: " userOpenID
    read -sp "Contraseña: " passOpenID
    echo
    
    # Procesar cada modelo
    for model in "${MODELS[@]}"; do
        progress "Procesando modelo $model"
        
        # Crear directorios
        create_directories "$model"
        
        # Descargar datos
        download_data "$model" "ua"
        download_data "$model" "va"
        
        # Limpiar archivos fuera del rango temporal
        clean_files "${BASE_DIR}/${model}/ua100m"
        clean_files "${BASE_DIR}/${model}/va100m"
        
        # Procesar cada año
        for year in $(seq $START_YEAR $END_YEAR); do
            calculate_wind_speed "$model" "$year"
            do_remapbil "$model" "$year"
        done
        
        # Limpiar directorios temporales
        progress "Limpiando directorios temporales para $model"
        rm -rf "${BASE_DIR}/${model}/ua100m"
        rm -rf "${BASE_DIR}/${model}/va100m"
        rm -rf "${BASE_DIR}/${model}/wsa100m_org"
    done
    
    progress "Proceso completado exitosamente"
    echo "Fecha y hora de finalización: $(date)"
}

# Ejecutar script principal con manejo de errores
main "$@"


