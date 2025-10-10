import pandas as pd
import numpy as np
import jenkspy
from jenkspy import JenksNaturalBreaks
import time
from tqdm import tqdm  

"""
Code to read and obtain the natural Jenks breaks to set up the cathegories of the variable in the input dataframe.
Initially created for the variance slope  values from ETOPO 2022. The procedure is very slow.

Author: Nathalia Correa-Sánchez
"""
########################################################################################
### ------------------READING ANF FORMATTING INPUT DATAFRAME FILE--------------------###
########################################################################################

df_slvar = pd.read_csv("/Dati/Outputs/Climate_Provinces/VarianceSlope_ETOPO2022csv.csv", sep = ",")
df_slvar = df_slvar["SlopeVaria"].dropna()
df_slvar = pd.DataFrame(df_slvar, columns=["SlopeVaria"])
df_slvar.to_csv("/Dati/Outputs/Climate_Provinces/VarianceSlope_ETOPO2022.csv", sep = ",")

########################################################################################
### ---------------------------DEFINNING RELEVANT FUNCTIONS--------------------------###
########################################################################################

def jenks_breaks(data, num_classes):
    """
    Compute Jenks Natural Breaks classification with progress tracking
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array of values to classify
    num_classes : int
        Number of classes to create
    
    Returns:
    --------
    breaks : numpy.ndarray
        Array of break points defining the classes
    """
    # Medir el tiempo de inicio
    start_time = time.time()
    
    # Sort the data
    sorted_data = np.sort(data)
    
    # Initialize matrices
    mat1 = np.zeros((len(sorted_data) + 1, num_classes + 1))
    mat2 = np.zeros((len(sorted_data) + 1, num_classes + 1))
    
    # Initialize first column
    mat1[1:, 1] = 1
    mat2[1:, 1] = 0
    
    # Compute initial variances
    for j in range(2, num_classes + 1):
        mat1[1, j] = np.inf
        mat2[1, j] = 0
    
    # Crear barra de progreso
    total_iterations = len(sorted_data) * (num_classes - 1)
    with tqdm(total=total_iterations, desc="Calculando Jenks Breaks") as pbar:
        # Main Jenks algorithm
        for i in range(2, len(sorted_data) + 1):
            sum_x = 0
            sum_x2 = 0
            
            for k in range(1, i + 1):
                val = sorted_data[i - k]
                sum_x += val
                sum_x2 += val ** 2
                
                n = k
                
                # Compute variance
                variance = sum_x2 / n - (sum_x / n) ** 2
                
                # Update matrices
                idx = i - 1
                for j in range(2, num_classes + 1):
                    if mat1[i, j] == np.inf:
                        p = mat1[idx, j-1] + variance
                        mat1[i, j] = p
                        mat2[i, j] = idx
                
                # Actualizar barra de progreso
                pbar.update(1)
    
    # Backtrack to find break points
    k = len(sorted_data)
    kclass = np.zeros(num_classes + 1, dtype=int)
    kclass[num_classes] = sorted_data[-1]
    
    count_num = num_classes
    while count_num >= 2:
        idx = int((mat2[k, count_num]) - 1)
        kclass[count_num - 1] = sorted_data[idx]
        k = int(mat2[k, count_num])
        count_num -= 1
    
    kclass[0] = sorted_data[0]
    
    # Calcular tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nTiempo de ejecución: {execution_time:.4f} segundos")
    
    return kclass



########################################################################################
### ----GETTING THE BRAKS FOR DEFFINNING THE BOUNDARY VALUES FO THE CATHERGORIES-----###
########################################################################################

print("Calculando breaks para la variable de pendiente...")

# Ejecutar Jenks Breaks
breaks = jenks_breaks(df_slvar["SlopeVaria"].values, 5)

print("\nPuntos de corte de las categorías:")
for i, brk in enumerate(breaks, 1):
    print(f"Categoría {i}: {brk}")

###----------------Otra Alternativa de Calculo   
jnb    = JenksNaturalBreaks(5) # Asking for 5 clusters
jnb.fit(df_slvar["SlopeVaria"].values)
print(jnb.breaks_)

###----------------Otra Alternativa de Calculo  
breaks = jenkspy.jenks_breaks(df_slvar["SlopeVaria"], nb_class=5)
print(breaks)