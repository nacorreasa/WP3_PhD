import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def classify_distribution(k):
    if np.isclose(k, 0, atol=1e-5):
        return "Gumbel ($\\xi$=0)"
    elif k > 0:
        return "Inversa de Weibull ($\\xi$<0)"
    else:
        return "Fréchet ($\\xi$>0)"
    
def generate_gev_data_scipy(k, loc, scale, size=100):
    return stats.genextreme.rvs(k, loc=loc, scale=scale, size=size)

# Parámetros para las distribuciones
params = [
    (-0.5, 0, 1),  # Fréchet
    (0, 0, 1),     # Gumbel
    (0.5, 0, 1)    # Inversa de Weibull
]

for k, loc, scale in params:
    # Generar datos de la distribución GEV específica
    data_scipy = generate_gev_data_scipy(k, loc, scale)
    
    # Ajustar la distribución GEV
    fitted_params_scipy = stats.genextreme.fit(data_scipy)
    c, loc_fit, scale_fit = fitted_params_scipy
    
    # Clasificar la distribución
    dist_type = classify_distribution(c)
    
    # # Generar puntos para la PDF
    x = np.linspace(min(data_scipy), max(data_scipy), 100)
    pdf = stats.genextreme.pdf(x, c, loc_fit, scale_fit)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.hist(data_scipy, bins=20, density=True, alpha=0.7, color='skyblue')
    plt.plot(x, pdf, 'r-', lw=2, label=f'GEV PDF fit (k ={c:.2f}), ($\\xi$=−k)')
    plt.title(f'GEV Distribution: {dist_type} (k ($\\xi$=−k) synthetic data = {k})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("/Dati/Outputs/Plots/WP3_development/Test_GEVFIT_C"+str(round(c, 2))+".png")
    plt.show()
    
    print(f"Synthetic data shape parameter (k): {k}")
    print(f"GEV fit shape parameter (k): {c:.4f}")
    print(f"Type of the GEV fit distribution: {dist_type}")
    print("---")