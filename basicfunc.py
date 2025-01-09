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
    return A*np.exp(x/tau) + f0

def lorentz(x, A, gamma, x0):
        return A * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)

def wigner(x, a, gamma, x0):
    return a * gamma / ((x - x0)**2 + (gamma / 2)**2)

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
        plt.plot(x_fit, y_fit, color='red', label='Gaussian fit', lw=2)
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
def exponential(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis", plot=False):
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

    # Calcolo di initial_guess in modo sensato
    A_guess = np.max(y) - np.min(y)  # Assumiamo A come differenza tra il massimo e il minimo
    tau_guess = np.median(x)  # Una stima iniziale di tau può essere la mediana di x
    f0_guess = np.min(y)  # Assumiamo f0 come il valore minimo di y

    initial_guess = [A_guess, tau_guess, f0_guess]
    
    # Fitting esponenziale
    if fit_with_weights:
        params, cov_matrix = curve_fit(exp, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True)
    else:
        params, cov_matrix = curve_fit(exp, x, y, p0=initial_guess)

    A, tau, f0 = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    A_uncertainty, tau_uncertainty, f0_uncertainty = uncertainties

    # Calcolo dei residui
    residui = res(y, exp(x, *params))

    # Chi quadro
    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    dof = len(x) - len(params)
    # Chi quadro ridotto
    chi_squared_reduced = chi_squared / dof

    # Stampa dei parametri ottimizzati
    print(f"Parametri ottimizzati:")
    print(f'-----------------------------------------------')
    print(f"A = {A} ± {A_uncertainty}")
    print(f"Tau = {tau} ± {tau_uncertainty}")
    print(f"f0 = {f0} ± {f0_uncertainty}")
    print(f'Chi-squared = {chi_squared}')
    print(f'Reduced chi-squared = {chi_squared_reduced}')

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
        
        plt.plot(x, exp(x, *params), color='red', label='Exponential fit', lw=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Exponential Fit")
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

    # Return dei parametri ottimizzati e delle incertezze
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