import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from iminuit import Minuit
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import inspect
from scipy.stats import norm

def linear(x, m, q):
    return m*x+q
    
def parabola(a, b, c, x):
    return a*x**2+b*x+c

def gaussian(x, amp, mu, sigma):
    # return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)
    return amp * norm.pdf(x, loc=mu, scale=sigma)

def calculate_bins(data):
    bin_width = 3.49 * np.std(data) / len(data)**(1/3)
    bins = int(np.ceil((max(data) - min(data)) / bin_width))
    return max(bins, 1)

def wigner(a, gamma, x0, x):
    f = a * gamma / ((x - x0)**2 + (gamma / 2)**2)
    return f

def res(data, fit):
    return data - fit

#NORMAL DISTRIBUTION
def normal(data, xlabel="X-axis", ylabel="Y-axis", titolo='title', xmin=None, xmax=None, b=None, param_plot=None):
    frame = inspect.currentframe().f_back
    var_name = [name for name, val in frame.f_locals.items() if val is data][0]
    
    #calcolo bin
    if b is not None:
        bins = b
    else:
        bins = calculate_bins(data)
    
    sigma_bins = np.sqrt(bins)  # Errori sulle x
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    sigma_counts = np.sqrt(counts)  # Errori sulle y
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Range per fittare
    if xmin is not None and xmax is not None:
        fit_mask = (bin_centers >= xmin) & (bin_centers <= xmax)
        bin_centers_fit = bin_centers[fit_mask]
        counts_fit = counts[fit_mask]
        sigma_counts_fit = sigma_counts[fit_mask]
    else:
        bin_centers_fit = bin_centers
        counts_fit = counts
        sigma_counts_fit = sigma_counts

    # Fit gaussiano
    initial_guess = [max(counts_fit), np.mean(data), np.std(data)]
    params, cov_matrix = curve_fit(gaussian, bin_centers_fit, counts_fit, p0=initial_guess)
    amp, mu, sigma = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    amp_uncertainty, mu_uncertainty, sigma_uncertainty = uncertainties
    print(f"Parametri ottimizzati:")
    print(f'-----------------------------------------------')
    print(f"Ampiezza = {amp} ± {amp_uncertainty}")
    print(f"Media = {mu} ± {mu_uncertainty}")
    print(f"Sigma = {sigma} ± {sigma_uncertainty}")

    # Calcolo del chi-quadro
    fit_values = gaussian(bin_centers_fit, *params)
    chi_quadro = np.sum(((counts_fit - fit_values) / sigma_counts_fit) ** 2)
    degrees_of_freedom = len(counts_fit) - len(params)
    reduced_chi_quadro = chi_quadro / degrees_of_freedom
    print(f"Chi-quadro = {chi_quadro}")
    print(f"Chi-quadro ridotto = {reduced_chi_quadro}")

    # Residui
    data_residui = res(counts_fit, fit_values)
    globals()[f"{var_name}_residui"] = data_residui
    residui = globals()[f"{var_name}_residui"]

    # Creiamo i dati della Gaussiana sul range X definito
    if xmin is not None and xmax is not None:
        x_fit = np.linspace(xmin, xmax, 10000)
    else:
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 10000)
    y_fit = gaussian(x_fit, *params)

    # Plot dell'istogramma e del fit
    plt.hist(data, bins=bins, density=False, alpha=0.6, label="Data")
    plt.plot(x_fit, y_fit, color='red', label='Gaussian fit', lw=2)
    plt.ylim(np.min(y_fit) * 1.1, np.max(y_fit) * 1.1) # Adattiamo il limite Y per il range X specificato
    if xmin is not None and xmax is not None: #limiti asse x
        plt.xlim(xmin, xmax)
    else:
        plt.xlim(mu - 5*sigma, mu + 5*sigma)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titolo)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Plot dei residui
    plt.errorbar(bin_centers_fit, data_residui, yerr=sigma_counts_fit, alpha=0.6, label="Residuals", fmt='o', markersize=4, capsize=2)
    plt.axhline(0, color='black', linestyle='--', lw=2)
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    else:
        plt.xlim(mu - 5*sigma, mu + 5*sigma)
    plt.xlabel(xlabel)
    plt.ylabel("(data - fit)")
    plt.title('Residuals')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    return amp, mu, sigma, residui, chi_quadro, reduced_chi_quadro

#SOTTRAZIONE BACKGROUND
def background(data, fondo, bins=None, xlabel="X-axis", ylabel="Counts", titolo='Title'):
    # Calcola i bin
    if bins is None:
        bins = max(int(data.max()), int(fondo.max()))

    # Creazione degli istogrammi
    data_hist, bin_edges = np.histogram(data, bins=bins, range=(0, bins))
    background_hist, _ = np.histogram(fondo, bins=bins, range=(0, bins))

    # Normalizzazione del background
    if background_hist.sum() > 0:  # Per evitare divisione per zero
        background_scaled = background_hist * (data_hist.sum() / background_hist.sum())
    else:
        background_scaled = background_hist

    # Sottrazione del background
    corrected_hist = data_hist - background_scaled

    # Evitiamo valori negativi
    corrected_hist[corrected_hist < 0] = 0

    # Centri dei bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Visualizzazione
    #QUI NON HA SENSO PLT.HIST PERCHé QUELLO USA UN ARRAY DI DATI E CREA LUI L'ISTOGRAMMA MENTRE NOI ABBIAMO UN ARRAY GIà CON I COUNTS BIN PER BIN
    plt.figure(figsize=(6.4, 4.8))
    plt.step(bin_centers, corrected_hist, label="Background subtracted", color='blue')
    # plt.bar(bin_centers, corrected_hist, width=np.diff(bin_edges), color='blue', alpha=0.5, label="Background subtracted") questo fa le barre colorate
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titolo)
    plt.grid(True)
    plt.show()

    return bin_centers, corrected_hist

# REGRESSIONE LINEARE
def linear_regression(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis"):
    # Gestione degli errori
    if sx is None or np.all(sx == 0):
        sx = np.zeros_like(x)
    if sy is None or np.all(sy == 0):
        sy = np.zeros_like(y)

    # Gestione dei pesi
    if np.any(sx != 0) and np.any(sy != 0):
        w = 1 / (sy**2 + sx**2)
        sigma_weights = np.sqrt(1 / w)
        fit_with_weights = True
    elif np.any(sx != 0):
        w = 1 / sx**2
        sigma_weights = np.sqrt(1 / w)
        fit_with_weights = True
    elif np.any(sy != 0):
        w = 1 / sy**2
        sigma_weights = np.sqrt(1 / w)
        fit_with_weights = True
    else:
        sigma_weights = None
        fit_with_weights = False

    # Fitting lineare
    initial_guess = [1, np.mean(y)]
    if fit_with_weights:
        params, cov_matrix = curve_fit(
            linear, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True
        )
    else:
        params, cov_matrix = curve_fit(linear, x, y, p0=initial_guess)

    m, q = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    m_uncertainty, q_uncertainty = uncertainties

    # Calcolo dei residui
    fit_values = linear(x, *params)
    residui = y - fit_values

    # Chi quadro
    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    # Gradi di libertà (numero di dati - numero di parametri)
    n_data = len(x)
    n_params = len(params)
    degrees_of_freedom = n_data - n_params
    # Chi quadro ridotto
    chi_squared_reduced = chi_squared / degrees_of_freedom

    # Stampa dei parametri ottimizzati
    print(f"Parametri ottimizzati:")
    print(f'-----------------------------------------------')
    print(f"Inclinazione (m) = {m} ± {m_uncertainty}")
    print(f"Intercetta (q) = {q} ± {q_uncertainty}")
    print(f'Chi-squared $\chi^2$ = {chi_squared}')
    print(f'Reduced chi-squared $\chi^2_r$ = {chi_squared_reduced}')

    # Plot dei dati e del fit
    plt.figure(figsize=(6.4, 4.8))
    if fit_with_weights:
        plt.errorbar(x, y, xerr=sx if np.any(sx != 0) else None,
                     yerr=sy if np.any(sy != 0) else None,
                     fmt='o', color='black', label='Data',
                     markersize=3, capsize=2)
    else:
        plt.scatter(x, y, color='black', label='Data', s=3)
    
    plt.plot(x, fit_values, color='red', label='Linear fit', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Linear Fit")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Plot dei residui
    plt.figure(figsize=(6.4, 4.8))
    if fit_with_weights:
        plt.errorbar(x, residui, xerr=sx if np.any(sx != 0) else None,
                     yerr=sy if np.any(sy != 0) else None,
                     fmt='o', color='blue', alpha=0.6, label='Residuals',
                     markersize=4, capsize=2)
    else:
        plt.scatter(x, residui, color='black', alpha=0.6, label='Residuals', s=10)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(f"(data - fit)")
    plt.title("Residuals")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    return m, q, residui, chi_squared, chi_squared_reduced