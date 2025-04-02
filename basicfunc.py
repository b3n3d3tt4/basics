import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import inspect
from scipy.stats import norm
from iminuit import Minuit
import iminuit
from iminuit.cost import LeastSquares
from scipy.special import erfc
from scipy.optimize import minimize

def gaussian(x, amp, mu, sigma):
    # return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)
    return amp * norm.pdf(x, loc=mu, scale=sigma)

def calculate_bins(data):
    bin_width = 3.49 * np.std(data) / len(data)**(1/3)
    bins = int(np.ceil((max(data) - min(data)) / bin_width))
    return max(bins, 1)

def linear(x, m, q):
    return m*x+q
    
def parabola(a, b, c, x):
    return a*x**2+b*x+c

def exp(x, A, tau, f0):
    return A*np.exp(-x/tau) + f0

def lorentz(x, A, gamma, x0):
        return A * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)

def wigner(x, a, gamma, x0):
    return a * gamma / ((x - x0)**2 + (gamma / 2)**2)

def gauss_exp_conv(x, A, mu, sigma, tau):
    arg = (sigma**2 - tau * (x - mu)) / (np.sqrt(2) * sigma * tau)
    return (A / (2 * tau)) * np.exp((sigma**2 - 2 * tau * (x - mu)) / (2 * tau**2)) * erfc(arg)

def l_norm(x, a, mu, sigma):
    return (a / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))

def res(data, fit):
    return data - fit

def chi2(model, params, x, y, sx=None, sy=None):
    # Calcola il modello y in base ai parametri
    y_model = model(x, *params)
    
    # Calcola il chi-quadro, considerando gli errori sugli assi x e y
    if sx is not None and sy is not None:
        chi2_val = np.sum(((y - y_model) / np.sqrt(sy**2 + sx**2)) ** 2)
    elif sx is not None:
        chi2_val = np.sum(((y - y_model) / sx) ** 2)
    elif sy is not None:
        chi2_val = np.sum(((y - y_model) / sy) ** 2)
    else:
        chi2_val = np.sum((y - y_model) ** 2 / np.var(y))
    
    return chi2_val


#NORMAL DISTRIBUTION
def normal(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', 
           xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    if data is not None:
        # Calcolo bin
        if b is not None:
            bins = b
        else:
            bins = calculate_bins(data)

        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    elif bin_centers is not None and counts is not None:
        var_name = "custom_data"
        bin_edges = None  # Non usiamo bin_edges
    else:
        raise ValueError("Devi fornire o `data`, o `bin_centers` e `counts`.")

    sigma_counts = np.sqrt(counts)  # Errori sulle y

    # Range per il fit
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
    initial_guess = [max(counts_fit), np.mean(bin_centers_fit), np.std(bin_centers_fit)]
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
    residui = res(counts_fit, fit_values)

    # Calcolo dell'integrale dell'istogramma nel range media ± n*sigma
    if n is not None:
        lower_bound = mu - n * sigma
        upper_bound = mu + n * sigma
        bins_to_integrate = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)  # il return è un array booleano con true e false che poi si mette come maskera
        integral = int(np.sum(counts[bins_to_integrate]))
        integral_uncertainty = int(np.sqrt(np.sum(sigma_counts[bins_to_integrate]**2)))
        print(f"Integrale dell'istogramma nel range [{lower_bound}, {upper_bound}] = {integral} ± {integral_uncertainty}")

    # Creiamo i dati della Gaussiana sul range X definito
    if xmin is not None and xmax is not None:
        x_fit = np.linspace(xmin, xmax, 10000)
    else:
        x_fit = np.linspace(bin_centers[0], bin_centers[-1], 10000)
    y_fit = gaussian(x_fit, *params)

    if plot:
        # Plot dell'istogramma e del fit
        plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, label="Data")
        plt.plot(x_fit, y_fit, color='red', label='Gaussian fit', lw=1.5)
        plt.ylim(0, np.max(y_fit) * 1.1)  # Adattiamo il limite Y per il range X specificato
        if x1 is not None and x2 is not None:  # limiti asse x
            plt.xlim(x1, x2)
        else:
            plt.xlim(mu - 3 * sigma, mu + 3 * sigma)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

        # Plot dei residui
        plt.errorbar(bin_centers_fit, residui, yerr=sigma_counts_fit, alpha=0.6, label="Residuals", fmt='o',
                     markersize=4, capsize=2)
        plt.axhline(0, color='black', linestyle='--', lw=2)
        if xmin is not None and xmax is not None:
            plt.xlim(xmin, xmax)
        else:
            plt.xlim(mu - 5 * sigma, mu + 5 * sigma)
        plt.xlabel(xlabel)
        plt.ylabel("(data - fit)")
        plt.title('Residuals')
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    plot = [x_fit, y_fit, bin_centers, counts]
    ints = [integral, integral_uncertainty]

    return params, uncertainties, residui, chi_quadro, reduced_chi_quadro, ints, plot

#GAUSS + EXPONENTIAL
def gauss_exp(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title',
              xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    if data is not None:
        # Calcolo bin
        bins = b if b is not None else int(np.sqrt(len(data)))
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    elif bin_centers is not None and counts is not None:
        bin_edges = None
    else:
        raise ValueError("Devi fornire o `data`, o `bin_centers` e `counts`.") 
    
    sigma_counts = np.sqrt(counts)  # Errori sulle y

    # Range per il fit
    if xmin is not None and xmax is not None:
        fit_mask = (bin_centers >= xmin) & (bin_centers <= xmax)
        bin_centers_fit = bin_centers[fit_mask]
        counts_fit = counts[fit_mask]
        sigma_counts_fit = sigma_counts[fit_mask]
    else:
        bin_centers_fit = bin_centers
        counts_fit = counts
        sigma_counts_fit = sigma_counts

    # Fit con convoluzione gaussiana-esponenziale
    initial_guess = [max(counts_fit), np.mean(bin_centers_fit), np.std(bin_centers_fit), 1.0]
    params, cov_matrix = curve_fit(gauss_exp_conv, bin_centers_fit, counts_fit, sigma=sigma_counts_fit, p0=initial_guess)
    amp, mu, sigma, tau = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    amp_uncertainty, mu_uncertainty, sigma_uncertainty, tau_uncertainty = uncertainties
    
    # Calcolare il massimo numericamente
    def neg_gauss_exp(x):
        return -gauss_exp_conv(x, *params)
    result = minimize(neg_gauss_exp, mu)  # Minimizzare la funzione negativa
    max_x = result.x[0]  # Il valore di x dove la funzione raggiunge il massimo
    
    print(f"Valore di x al massimo: {max_x}")
    
    # Stampa dei parametri ottimizzati
    print(f"Parametri ottimizzati:")
    print(f'-----------------------------------------------')
    print(f"Ampiezza = {amp} ± {amp_uncertainty}")
    print(f"Media = {mu} ± {mu_uncertainty}")
    print(f"Sigma = {sigma} ± {sigma_uncertainty}")
    print(f"Tau = {tau} ± {tau_uncertainty}")
    
    # Calcolo del chi-quadro
    fit_values = gauss_exp_conv(bin_centers_fit, *params)
    chi_quadro = np.sum(((counts_fit - fit_values) / sigma_counts_fit) ** 2)
    degrees_of_freedom = len(counts_fit) - len(params)
    reduced_chi_quadro = chi_quadro / degrees_of_freedom
    print(f"Chi-quadro = {chi_quadro}")
    print(f"Chi-quadro ridotto = {reduced_chi_quadro}")
    
    # Calcolo dell'integrale dell'istogramma nel range media ± n*sigma
    if n is not None:
        lower_bound = mu - n * sigma
        upper_bound = mu + n * sigma
        bins_to_integrate = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
        integral = int(np.sum(counts[bins_to_integrate]))
        integral_uncertainty = int(np.sqrt(np.sum(sigma_counts[bins_to_integrate]**2)))
        print(f"Integrale dell'istogramma nel range [{lower_bound}, {upper_bound}] = {integral} ± {integral_uncertainty}")
    
    # Creiamo i dati della funzione di fit sul range X definito
    x_fit = np.linspace(xmin if xmin is not None else bin_centers[0],
                        xmax if xmax is not None else bin_centers[-1], 10000)
    y_fit = gauss_exp_conv(x_fit, *params)
    
    if plot:
        # Plot dell'istogramma e del fit
        plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, label="Data")
        plt.plot(x_fit, y_fit, color='red', label='Gauss-Exp fit', lw=1.5)
        plt.axvline(max_x, color='blue', linestyle='--', label='Mu')
        # plt.ylim(0, np.max(y_fit) * 1.1)
        plt.xlim(x1 if x1 is not None else mu - 3 * sigma,
                 x2 if x2 is not None else mu + 3 * sigma)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()
    
    plot_data = [x_fit, y_fit, bin_centers, counts]
    integral_results = [integral, integral_uncertainty] if n is not None else None
    
    # Restituisci anche max_x insieme ai parametri
    return params, max_x, uncertainties, chi_quadro, reduced_chi_quadro, integral_results, plot_data

#fit spalla compton
def compton_minuit(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', 
                   xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    if data is not None:
        frame = inspect.currentframe().f_back
        var_name = [name for name, val in frame.f_locals.items() if val is data][0]

        # Calcolo bin
        if b is not None:
            bins = b
        else:
            bins = calculate_bins(data)

        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    elif bin_centers is not None and counts is not None:
        var_name = "custom_data"
        bin_edges = None
    else:
        raise ValueError("Devi fornire o `data`, o `bin_centers` e `counts`.")

    sigma_counts = np.sqrt(counts)  # Errori sulle y

    # Range per il fit
    if xmin is not None and xmax is not None:
        fit_mask = (bin_centers >= xmin) & (bin_centers <= xmax)
        bin_centers_fit = bin_centers[fit_mask]
        counts_fit = counts[fit_mask]
        sigma_counts_fit = sigma_counts[fit_mask]
    else:
        bin_centers_fit = bin_centers
        counts_fit = counts
        sigma_counts_fit = sigma_counts

    # Funzione error function (erfc)
    def fit_function(x, mu, sigma, rate, bkg):
        return rate * erfc((x - mu) / sigma) + bkg

    # Funzione chi-quadro per il fit
    def chi2(y_data, y_model, sigma):
        return np.sum(((y_data - y_model) / sigma) ** 2)

    # Funzione di costo per Minuit
    def cost_function(mu, sigma, rate, bkg):
        y_model = fit_function(bin_centers_fit, mu, sigma, rate, bkg)
        return chi2(counts_fit, y_model, sigma_counts_fit)

    # Parametri iniziali per Minuit
    theta0 = [np.mean(bin_centers_fit), np.std(bin_centers_fit), np.max(counts_fit), np.min(counts_fit)]

    # Inizializzare Minuit
    mfit = iminuit.Minuit(cost_function, *theta0, name=['mu', 'sigma', 'rate', 'bkg'])
    mfit.errordef = mfit.LEAST_SQUARES
    mfit.limits['mu'] = (xmin, xmax)
    mfit.limits['sigma'] = (0, np.max(bin_centers_fit))
    mfit.limits['rate'] = (0, np.max(counts_fit))
    mfit.limits['bkg'] = (0, None)

    # Eseguire il fit
    mfit.migrad()

    # Parametri ottimizzati
    print("Parametri ottimizzati con Minuit:")
    print(f"mu = {mfit.values['mu']} ± {mfit.errors['mu']}")
    print(f"sigma = {mfit.values['sigma']} ± {mfit.errors['sigma']}")
    print(f"rate = {mfit.values['rate']} ± {mfit.errors['rate']}")
    print(f"bkg = {mfit.values['bkg']} ± {mfit.errors['bkg']}")

    # Generare il modello con i parametri ottimizzati
    x_fit = np.linspace(xmin, xmax, 1000)
    y_fit = fit_function(x_fit, *mfit.values)

    # Calcolare l'integrale nell'intervallo mu ± n*sigma
    mu = mfit.values['mu']
    sigma = mfit.values['sigma']
    lower_bound = mu - n * sigma
    upper_bound = mu + n * sigma
    bins_to_integrate = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)  # il return è un array booleano con true e false che poi si mette come maskera
    integral = np.sum(counts[bins_to_integrate])
    integral_uncertainty = np.sqrt(np.sum(sigma_counts[bins_to_integrate]**2))
    print(f"Integrale dell'istogramma nel range [{lower_bound}, {upper_bound}] = {integral} ± {integral_uncertainty}")

    # Plot dei dati e del fit
    if plot:
        plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, label="Data")
        plt.plot(x_fit, y_fit, label='Error function fit', color='red', lw=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
        # Impostare xlim se x1 e x2 sono diversi da None
        if x1 is not None and x2 is not None:
            plt.xlim(x1, x2)
        # Impostare ylim in modo sensato
        plt.ylim(0, np.max(counts)+1)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    int = [integral, integral_uncertainty]

    return mfit.values, mfit.errors, int

#fit spalla compton con curve_fit
def compton_curvefit(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', 
                     xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    if data is not None:
        frame = inspect.currentframe().f_back
        var_name = [name for name, val in frame.f_locals.items() if val is data][0]

        # Calcolo bin
        if b is not None:
            bins = b
        else:
            bins = calculate_bins(data)

        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    elif bin_centers is not None and counts is not None:
        var_name = "custom_data"
        bin_edges = None
    else:
        raise ValueError("Devi fornire o `data`, o `bin_centers` e `counts`.")

    sigma_counts = np.sqrt(counts)  # Errori sulle y

    # Range per il fit
    if xmin is not None and xmax is not None:
        fit_mask = (bin_centers >= xmin) & (bin_centers <= xmax)
        bin_centers_fit = bin_centers[fit_mask]
        counts_fit = counts[fit_mask]
        sigma_counts_fit = sigma_counts[fit_mask]
    else:
        bin_centers_fit = bin_centers
        counts_fit = counts
        sigma_counts_fit = sigma_counts

    # Funzione error function (erfc)
    def fit_function(x, mu, sigma, rate, bkg):
        return rate * erfc((x - mu) / sigma) + bkg

    # Parametri iniziali per curve_fit
    initial_guess = [np.median(bin_centers_fit), 5, np.max(counts_fit), np.min(counts_fit)]
    # initial_guess = [np.mean(bin_centers_fit), np.std(bin_centers_fit), np.max(counts_fit), np.min(counts_fit)]

    # Esegui il fit
    params, cov_matrix = curve_fit(fit_function, bin_centers_fit, counts_fit, p0=initial_guess, sigma=sigma_counts_fit)
    mu, sigma, rate, bkg = params
    uncertainties = np.sqrt(np.diag(cov_matrix))

    # Parametri ottimizzati
    print("Parametri ottimizzati con curve_fit:")
    print(f"mu = {mu} ± {uncertainties[0]}")
    print(f"sigma = {sigma} ± {uncertainties[1]}")
    print(f"rate = {rate} ± {uncertainties[2]}")
    print(f"bkg = {bkg} ± {uncertainties[3]}")

    # Generare il modello con i parametri ottimizzati
    x_fit = np.linspace(xmin, xmax, 1000)
    y_fit = fit_function(x_fit, *params)

    # Calcolare l'integrale nell'intervallo mu ± n*sigma
    lower_bound = mu - n * sigma
    upper_bound = mu + n * sigma
    bins_to_integrate = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)  # il return è un array booleano con true e false che poi si mette come maskera
    integral = np.sum(counts[bins_to_integrate])
    integral_uncertainty = np.sqrt(np.sum(sigma_counts[bins_to_integrate]**2))
    print(f"Integrale dell'istogramma nel range [{lower_bound}, {upper_bound}] = {integral} ± {integral_uncertainty}")

    # Plot dei dati e del fit
    if plot:
        plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, label="Data")
        plt.plot(x_fit, y_fit, label='Error function fit', color='red', lw=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
        if x1 is not None and x2 is not None:
            plt.xlim(x1, x2)
        # Impostare ylim in modo sensato
        plt.ylim(0, np.max(counts)+1)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    int = [integral, integral_uncertainty]

    return params, uncertainties, int

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
def linear_regression(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', plot=False):
    # Gestione degli errori
    if sx is None or np.all(sx == 0):
        sx = np.zeros_like(x)
    if sy is None or np.all(sy == 0):
        sy = np.zeros_like(y)

    # Gestione dei pesi
    if np.any(sx != 0) and np.any(sy != 0):
        w = 1 / (sy**2 + sx**2)
        fit_with_weights = True
    elif np.any(sx != 0):
        w = 1 / sx**2
        fit_with_weights = True
    elif np.any(sy != 0):
        w = 1 / sy**2
        fit_with_weights = True
    else:
        w = np.ones_like(y)
        fit_with_weights = False

    # Cost function per Minuit
    least_squares = LeastSquares(x, y, 1/w, linear)

    # Inizializzazione e fit con Minuit
    m_init, q_init = 1, np.mean(y)  # Stime iniziali
    minuit = Minuit(least_squares, m=m_init, q=q_init)
    minuit.migrad()  # Esegue il fit

    # Estrazione dei risultati
    m, q = minuit.values['m'], minuit.values['q']
    m_uncertainty, q_uncertainty = minuit.errors['m'], minuit.errors['q']

    # Calcolo dei residui
    y_fit = linear(x, m, q)
    residui = res(y, y_fit)

    # Chi quadro calcolato come (expected - observed)^2 / w
    chi_squared = np.sum(((y - y_fit)**2) * w)
    if x.shape[0] > 2:
        dof = len(x) - 2  # Gradi di libertà
        chi_squared_reduced = chi_squared / dof
    else: chi_squared_reduced = 0

    # Stampa dei parametri ottimizzati
    print(f"Parametri ottimizzati:")
    print(f'-----------------------------------------------')
    print(f"Inclinazione (m) = {m} ± {m_uncertainty}")
    print(f"Intercetta (q) = {q} ± {q_uncertainty}")
    print(f'Chi-squared= {chi_squared}')
    if x.shape[0] > 2:
        print(f'Reduced chi-squared= {chi_squared_reduced}')
    else: print(f'Non ha senso calcolare il chi2 ridotto')

    # Plot dei dati e del fit
    if plot:
        plt.figure(figsize=(6.4, 4.8))
        if fit_with_weights:
            plt.errorbar(x, y, xerr=sx if np.any(sx != 0) else None,
                         yerr=sy if np.any(sy != 0) else None,
                         fmt='o', color='black', label='Data',
                         markersize=3, capsize=2)
        else:
            plt.scatter(x, y, color='black', label='Data', s=3)
        
        plt.plot(x, y_fit, color='red', label='Linear fit', lw=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
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
        plt.axhline(0, color='red', linestyle='--', lw=1.5)
        plt.xlabel(xlabel)
        plt.ylabel(f"(data - fit)")
        plt.title("Residuals of the linear fit")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    return m, q, m_uncertainty, q_uncertainty, residui, chi_squared, chi_squared_reduced

# Funzione per il fit esponenziale
def exponential(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', plot=False):
    if sx is None or np.all(sx == 0):
        sx = np.zeros_like(x)
    if sy is None or np.all(sy == 0):
        sy = np.zeros_like(y)
    
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

    A_guess = np.max(y) - np.min(y)
    tau_guess = np.median(x)
    f0_guess = np.min(y)
    initial_guess = [A_guess, tau_guess, f0_guess]
    
    if fit_with_weights:
        params, cov_matrix = curve_fit(exp, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True)
    else:
        params, cov_matrix = curve_fit(exp, x, y, p0=[1, 1, 0])

    A, tau, f0 = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    A_uncertainty, tau_uncertainty, f0_uncertainty = uncertainties

    residui = res(y, exp(x, *params))
    
    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    dof = len(x) - len(params)
    chi_squared_reduced = chi_squared / dof

    print(f"A = {A} ± {A_uncertainty}")
    print(f"Tau = {tau} ± {tau_uncertainty}")
    print(f"f0 = {f0} ± {f0_uncertainty}")
    print(f'Chi-squared = {chi_squared}')
    print(f'Reduced chi-squared = {chi_squared_reduced}')
    
    x_fit = np.linspace(x.min(), x.max(), 1000)

    if plot:
        fig = plt.figure(figsize=(7, 8))  # Aumenta le dimensioni per accomodare la tabella e i grafici
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 0.5, 5, 0.5, 1])  # Griglia con residui più schiacciati

        # Subplot 1: Tabella
        ax_table = fig.add_subplot(gs[:2, 0])
        ax_table.axis('tight')
        ax_table.axis('off')

        # Dati della tabella
        data = [
            ["A", f"{A:.3f} ± {A_uncertainty:.3f}"],
            ["Tau", f"{tau:.3f} ± {tau_uncertainty:.3f}"],
            ["f0", f"{f0:.3f} ± {f0_uncertainty:.3f}"],
            ["Chi²", f"{chi_squared:.8f}"],
            ["Chi² rid.", f"{chi_squared_reduced:.8f}"]
        ]

        table = ax_table.table(
            cellText=data,
            colLabels=["Parametro", "Valore"],
            loc='center',
            cellLoc='center',
            colColours=["#4CAF50", "#4CAF50"],  # Colori per le intestazioni
            bbox=[0, 0, 1, 1]  # Regola la posizione e la dimensione della tabella
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(data[0]))))

        # Personalizza i bordi delle celle
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            if row == 0:  # Intestazioni
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor("lightblue")

        # Subplot 2: Fit esponenziale
        ax1 = fig.add_subplot(gs[2, 0])  # Grafico principale
        ax1.errorbar(x, y, xerr=sx if np.any(sx != 0) else None,
                     yerr=sy if np.any(sy != 0) else None,
                     fmt='*', color='black', label='Data', markersize=6, capsize=2)
        ax1.plot(x_fit, exp(x_fit, *params), color='red', label='Exponential fit', lw=1.2)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(titolo)
        ax1.legend()
        ax1.grid(alpha=0.5)

        # Subplot 3: Residui
        ax2 = fig.add_subplot(gs[3:, 0], sharex=ax1)  # Grafico dei residui con altezza ridotta
        ax2.scatter(x, residui, color='black', label='Residuals', s=10)
        ax2.axhline(0, color='red', linestyle='--', lw=2)  # Linea orizzontale a y=0
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("(data - fit)")
        ax2.grid(alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("grafici/exponential_fit.pdf")
        plt.show()

        parametri = np.array([A, tau, f0])
        incertezze = np.array([A_uncertainty, tau_uncertainty, f0_uncertainty])
    
    return parametri, incertezze, residui, chi_squared, chi_squared_reduced

#Fit parabolico con minuti
def parabolic(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis"):
    # Funzione chi-quadro per Minuit
    def chi2_parabola(a, b, c):
        return chi2(parabola, [a, b, c], x, y, sx, sy)
    
    # Parametri iniziali per il fit parabolico
    initial_guess = [1, 1, 0]
    
    # Creazione dell'oggetto Minuit e settaggio dei parametri
    m = Minuit(chi2_parabola, *initial_guess)
    m.errordef = m.LEAST_SQUARES
    m.migrad(ncall=10000)

    # Estrazione dei parametri ottimizzati e delle incertezze
    a_opt, b_opt, c_opt = m.values['a'], m.values['b'], m.values['c']
    a_err, b_err, c_err = m.errors['a'], m.errors['b'], m.errors['c']
    
    # Calcolo dei residui
    y_model = parabola(x, a_opt, b_opt, c_opt)
    residui = y - y_model
    
    # Calcolo del chi-quadro finale
    chi2_final = m.fval
    dof = len(x) - len([a_opt, b_opt, c_opt])  # gradi di libertà
    chi2_reduced = chi2_final / dof

    # Stampa dei risultati
    print(f"Parametri ottimizzati:")
    print(f"a = {a_opt} ± {a_err}")
    print(f"b = {b_opt} ± {b_err}")
    print(f"c = {c_opt} ± {c_err}")
    print(f"Chi-squared = {chi2_final}")
    print(f"Reduced Chi-squared = {chi2_reduced}")

    # Plot dei dati e del fit
    plt.figure(figsize=(6.4, 4.8))
    if sx is not None or sy is not None:
        plt.errorbar(x, y, xerr=sx, yerr=sy, fmt='o', color='black', label='Data', markersize=3, capsize=2)
    else:
        plt.scatter(x, y, color='black', label='Data', s=3)
    
    plt.plot(x, parabola(x, a_opt, b_opt, c_opt), color='red', label='Parabolic fit', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Parabolic Fit")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Plot dei residui
    plt.figure(figsize=(6.4, 4.8))
    if sx is not None or sy is not None:
        plt.errorbar(x, residui, xerr=sx, yerr=sy, fmt='o', color='blue', alpha=0.6, label='Residuals', markersize=4, capsize=2)
    else:
        plt.scatter(x, residui, color='black', alpha=0.6, label='Residuals', s=10)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel("X-axis")
    plt.ylabel("(data - fit)")
    plt.title("Residuals")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    return a_opt, b_opt, c_opt, residui, chi2_final, chi2_reduced

#Fit Lorentziana
def lorentzian(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis"):
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

    # Fitting Lorentziano
    initial_guess = [1, 1, np.mean(x)]
    if fit_with_weights:
        params, cov_matrix = curve_fit(
            lorentzian, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True
        )
    else:
        params, cov_matrix = curve_fit(lorentzian, x, y, p0=initial_guess)

    A, gamma, x0 = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    A_uncertainty, gamma_uncertainty, x0_uncertainty = uncertainties

    # Calcolo dei residui
    residui = y - lorentzian(x, *params)

    # Calcolo del chi quadro
    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    dof = len(x) - len(params)
    chi_squared_reduced = chi_squared / dof

    # Stampa dei risultati
    print(f"Parametri ottimizzati:")
    print(f"-----------------------------------------------")
    print(f"A = {A} ± {A_uncertainty}")
    print(f"gamma = {gamma} ± {gamma_uncertainty}")
    print(f"x0 = {x0} ± {x0_uncertainty}")
    print(f"Chi-squared = {chi_squared}")
    print(f"Reduced Chi-squared = {chi_squared_reduced}")

    # Plot dei dati e del fit
    plt.figure(figsize=(6.4, 4.8))
    if fit_with_weights:
        plt.errorbar(x, y, xerr=sx if np.any(sx != 0) else None,
                     yerr=sy if np.any(sy != 0) else None,
                     fmt='o', color='black', label='Data',
                     markersize=3, capsize=2)
    else:
        plt.scatter(x, y, color='black', label='Data', s=3)
    
    plt.plot(x, lorentzian(x, *params), color='red', label='Lorentzian fit', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Lorentzian Fit")
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

    return A, gamma, x0, residui, chi_squared, chi_squared_reduced

#FIT BREIT-WIGNER
def breitwigner(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis"):
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

    # Fitting Breit-Wigner
    initial_guess = [1, 1, np.mean(x)]
    if fit_with_weights:
        params, cov_matrix = curve_fit(wigner, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True)
    else:
        params, cov_matrix = curve_fit(wigner, x, y, p0=initial_guess)

    a, gamma, x0 = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    a_uncertainty, gamma_uncertainty, x0_uncertainty = uncertainties

    # Calcolo dei residui
    residui = y - wigner(x, *params)

    # Calcolo del chi quadro
    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    dof = len(x) - len(params)
    chi_squared_reduced = chi_squared / dof

    # Stampa dei risultati
    print(f"Parametri ottimizzati:")
    print(f"-----------------------------------------------")
    print(f"a = {a} ± {a_uncertainty}")
    print(f"gamma = {gamma} ± {gamma_uncertainty}")
    print(f"x0 = {x0} ± {x0_uncertainty}")
    print(f"Chi-squared = {chi_squared}")
    print(f"Reduced Chi-squared = {chi_squared_reduced}")

    # Plot dei dati e del fit
    plt.figure(figsize=(6.4, 4.8))
    if fit_with_weights:
        plt.errorbar(x, y, xerr=sx if np.any(sx != 0) else None, yerr=sy if np.any(sy != 0) else None,
                     fmt='o', color='black', label='Data', markersize=3, capsize=2)
    else:
        plt.scatter(x, y, color='black', label='Data', s=3)
    
    plt.plot(x, wigner(x, *params), color='red', label='Breit-Wigner fit', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Breit-Wigner Fit")
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

    return a, gamma, x0, residui, chi_squared, chi_squared_reduced

#LOGNORMALE
def lognormal(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', 
                  xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    if data is not None:
        bins = b if b is not None else int(np.sqrt(len(data)))
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    elif bin_centers is not None and counts is not None:
        bin_edges = None
    else:
        raise ValueError("Devi fornire o `data`, o `bin_centers` e `counts`.")

    sigma_counts = np.sqrt(counts)
    
    if xmin is not None and xmax is not None:
        fit_mask = (bin_centers >= xmin) & (bin_centers <= xmax)
        bin_centers_fit = bin_centers[fit_mask]
        counts_fit = counts[fit_mask]
        sigma_counts_fit = sigma_counts[fit_mask]
    else:
        bin_centers_fit = bin_centers
        counts_fit = counts
        sigma_counts_fit = sigma_counts

    initial_guess = [max(counts_fit), np.log(np.mean(bin_centers_fit)), np.std(np.log(bin_centers_fit))]
    params, cov_matrix = curve_fit(l_norm, bin_centers_fit, counts_fit, p0=initial_guess)
    amp, mu, sigma = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    amp_uncertainty, mu_uncertainty, sigma_uncertainty = uncertainties

    print(f"Parametri ottimizzati:")
    print(f'-----------------------------------------------')
    print(f"Ampiezza = {amp} ± {amp_uncertainty}")
    print(f"Media = {mu} ± {mu_uncertainty}")
    print(f"Sigma = {sigma} ± {sigma_uncertainty}")

    fit_values = l_norm(bin_centers_fit, *params)
    chi_quadro = np.sum(((counts_fit - fit_values) / sigma_counts_fit) ** 2)
    degrees_of_freedom = len(counts_fit) - len(params)
    reduced_chi_quadro = chi_quadro / degrees_of_freedom

    if n is not None:
        lower_bound, upper_bound = np.exp(mu - n * sigma), np.exp(mu + n * sigma)
        bins_to_integrate = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
        integral = int(np.sum(counts[bins_to_integrate]))
        integral_uncertainty = int(np.sqrt(np.sum(sigma_counts[bins_to_integrate]**2)))
    else:
        integral, integral_uncertainty = None, None

    x_fit = np.linspace(min(bin_centers), max(bin_centers), 10000)
    y_fit = l_norm(x_fit, *params)
    
    if plot:
        plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, label="Data")
        plt.plot(x_fit, y_fit, color='red', label='Lognormal fit', lw=1.5)
        # plt.ylim(0, np.max(y_fit) * 1.1)
        plt.xlim(x1, x2) if x1 is not None and x2 is not None else plt.xlim(min(bin_centers), max(bin_centers))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()
    
    return params, uncertainties, chi_quadro, reduced_chi_quadro, [integral, integral_uncertainty], [x_fit, y_fit, bin_centers, counts]


'''ciao baby'''