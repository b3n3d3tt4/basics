#############################################################################################
##                          UNIVERSITA' DEGLI STUDI DI PADOVA - DFA                        ##
##            corso di laurea magistrale in Physics of the fundamental interactions        ##
##                        Copyright (C) 2023 - 2024, All rights reserved.                  ##
##                             benedetta.rasera@studenti.unipd.it                          ##
#############################################################################################

##############################################################################################
#       This library contains some simple fitting functions for various models               #
#       useful for laboratory courses in the Bachelor's degree in Physics                    #
#       and for early lab work in the Master's degree in Physics (at least at Padua).        #
#       Many of these functions were implemented for data analysis in spectroscopy           #
#       experiments, but they can also be used for similar purposes.                         #
#       The fitting functions included in this library are:                                  #
#           - Gaussian fit                                                                   #
#           - Gaussian-exponential convolution fit                                           #
#           - Compton edge fit (based on the error function)                                 #
#           - Background subtraction from a spectrum                                         #
#           - Linear fit                                                                     #
#           - Exponential fit (increasing and decreasing)                                    #
#           - Parabolic fit                                                                  #
#           - Lorentzian fit                                                                 #
#           - Breit-Wigner fit                                                               #
#           - Log-normal fit                                                                 #
#           - Bode diagram fit (for low-pass, high-pass, and band-pass filters)              #
##############################################################################################


import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.special import erfc
from scipy.optimize import minimize


# DEFINIZIONE DELLE FUNZIONI DI FIT
# funzione gaussiana
def gaussian(x, amp, mu, sigma):
    # return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)
    return amp * norm.pdf(x, loc=mu, scale=sigma)

# funzione per calcolare i bin nel fit con istogrammi
def calculate_bins(data):
    bin_width = 3.49 * np.std(data) / len(data)**(1/3)
    bins = int(np.ceil((max(data) - min(data)) / bin_width))
    return max(bins, 1)

# funzione retta
def retta(x, m, q):
    return m * x + q
    
# funzione parabola
def parabola(a, b, c, x):
    return a * x**2 + b * x + c

# funzione esponenziale
def exp_pos(x, A, tau, f0): #esponenziale crescente
    return A * np.exp(x / tau) + f0
def exp_neg(x, A, tau, f0): ##esponenziale decrescente
    return A * np.exp(-x / tau) + f0

# funzione lorentziana
def lorentz(x, A, gamma, x0):
        return A * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)

# curva di Wigner
def wigner(x, a, gamma, x0):
    return a * gamma / ((x - x0)**2 + (gamma / 2)**2)

# funzione convoluzione gaussiana-esponenziale
def gauss_exp_conv(x, A, mu, sigma, tau):
    arg = (sigma**2 - tau * (x - mu)) / (np.sqrt(2) * sigma * tau)
    return (A / (2 * tau)) * np.exp((sigma**2 - 2 * tau * (x - mu)) / (2 * tau**2)) * erfc(arg)

# funzione log-normale
def l_norm(x, a, mu, sigma):
    return (a / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))

# funzioni di trasferimento per i vari tipi di filtro
def filtro_basso(omega, R, C):
    return 1 / (1 + 1j * omega * R * C)
def filtro_alto(omega, R, C):
    return (1j * omega * R * C) / (1 + 1j * omega * R * C)
def filtro_banda(omega, R, C, omega_0, Q):
    return (1j * omega * R * C) / ((1j * omega) ** 2 + (omega_0 / Q) * (1j * omega) + omega_0**2)

#funzione per i residui di tutte le funzioni che fitta questa libreria
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

#####################################################################################################################
#####                                              FIT FUNCTIONS:                                               #####
#####################################################################################################################

# NORMAL DISTRIBUTION
def normal(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    print("This fit returns a list which contains, in order:\n"
      "- A numpy array with the parameters\n"
      "- A numpy array with the uncertainties\n"
      "- A numpy array with the residuals\n"
      "- The chi squared\n"
      "- The reduced chi squared \n"
      "- The integral of the histogram in the range mu ± n*sigma\n"
      "- The plot data (x_fit, y_fit, bin_centers, counts) if you need to plot other thing\n")
    
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
    amp_unc, mu_unc, sigma_unc = uncertainties

    # Stampa a schermo dei parametri ottimizzati
    print(f"Parametri ottimizzati:")
    print(f'-----------------------------------------------')
    print(f"Ampiezza = {amp} ± {amp_unc}")
    print(f"Media = {mu} ± {mu_unc}")
    print(f"Sigma = {sigma} ± {sigma_unc}")

    # Calcolo del chi-quadro
    fit_values = gaussian(bin_centers_fit, *params)
    chi_quadro = np.sum(((counts_fit - fit_values) / sigma_counts_fit) ** 2)
    degrees_of_freedom = len(counts_fit) - len(params)
    reduced_chi_quadro = chi_quadro / degrees_of_freedom
    print(f"Chi-quadro = {chi_quadro}")
    print(f"Chi-quadro ridotto = {reduced_chi_quadro}")

    # Calcolo dei residui
    residui = res(counts_fit, fit_values)

    # Calcolo dell'integrale dell'istogramma nel range media ± n*sigma
    if n is not None:
        lower_bound = mu - n * sigma
        upper_bound = mu + n * sigma
        bins_to_integrate = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)  # il return è un array booleano con true e false che poi si mette come maskera
        integral = int(np.sum(counts[bins_to_integrate]))
        integral_unc = int(np.sqrt(np.sum(sigma_counts[bins_to_integrate]**2)))
        print(f"Integrale dell'istogramma nel range [{lower_bound}, {upper_bound}] = {integral} ± {integral_unc}")

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
        plt.errorbar(bin_centers_fit, residui, yerr=sigma_counts_fit, alpha=0.6, label="Residuals", fmt='o', markersize=3, capsize=2)
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

    plot = np.array([x_fit, y_fit, bin_centers, counts])
    ints = np.array([integral, integral_unc])

    parametri = np.array([amp, mu, sigma])
    incertezze = np.array([amp_unc, mu_unc, sigma_unc])

    return parametri, incertezze, residui, chi_quadro, reduced_chi_quadro, ints, plot

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

#FIT COMPTON CON ERFC
def compton(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title',
            xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    print("This fit returns a list which contains, in order:\n"
          "- A numpy array with the parameters\n"
          "- A numpy array with the uncertainties\n"
          "- A numpy array with the residuals\n"
          "- The chi squared\n"
          "- The reduced chi squared \n"
          "- The integral of the histogram in the range mu ± n*sigma\n"
          "- The plot data (x_fit, y_fit, bin_centers, counts) if you need to plot other thing\n")

    if data is not None:
        if b is not None:
            bins = b
        else:
            bins = calculate_bins(data)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    elif bin_centers is not None and counts is not None:
        bin_edges = None
    else:
        raise ValueError("Devi fornire o `data`, o `bin_centers` e `counts`.")

    sigma_counts = np.sqrt(counts)

    if xmin is not None and xmax is not None:
        mask = (bin_centers >= xmin) & (bin_centers <= xmax)
        bin_centers_fit = bin_centers[mask]
        counts_fit = counts[mask]
        sigma_counts_fit = sigma_counts[mask]
    else:
        bin_centers_fit = bin_centers
        counts_fit = counts
        sigma_counts_fit = sigma_counts

    def fit_function(x, mu, sigma, rate, bkg):
        return rate * erfc((x - mu) / sigma) + bkg

    initial_guess = [np.median(bin_centers_fit), 5, np.max(counts_fit), np.min(counts_fit)]
    params, cov_matrix = curve_fit(fit_function, bin_centers_fit, counts_fit, p0=initial_guess, sigma=sigma_counts_fit)
    mu, sigma, rate, bkg = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    mu_unc, sigma_unc, rate_unc, bkg_unc = uncertainties

    fit_values = fit_function(bin_centers_fit, *params)
    chi_quadro = np.sum(((counts_fit - fit_values) / sigma_counts_fit) ** 2)
    dof = len(counts_fit) - len(params)
    reduced_chi = chi_quadro / dof
    residui = counts_fit - fit_values

    if n is not None:
        lower_bound = mu - n * sigma
        upper_bound = mu + n * sigma
        integral_mask = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
        integral = int(np.sum(counts[integral_mask]))
        integral_unc = int(np.sqrt(np.sum(sigma_counts[integral_mask]**2)))
        print(f"Integrale dell'istogramma nel range [{lower_bound}, {upper_bound}] = {integral} ± {integral_unc}")
    else:
        integral, integral_unc = 0, 0

    x_fit = np.linspace(xmin if xmin is not None else bin_centers[0],
                        xmax if xmax is not None else bin_centers[-1], 1000)
    y_fit = fit_function(x_fit, *params)

    if plot:
        # Plot principale
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, label='Data')
        plt.plot(x_fit, y_fit, label='Fit con funzione erfc', color='red', lw=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
        plt.grid(alpha=0.5)
        if x1 is not None and x2 is not None:
            plt.xlim(x1, x2)
        else:
            plt.xlim(mu - 3*sigma, mu + 3*sigma)
        plt.ylim(0, np.max(counts)*1.1)
        plt.legend()

        # Tabella dei parametri
        cell_text = [
            [f"{mu:.3f} ± {mu_unc:.3f}"],
            [f"{sigma:.3f} ± {sigma_unc:.3f}"],
            [f"{rate:.1f} ± {rate_unc:.1f}"],
            [f"{bkg:.1f} ± {bkg_unc:.1f}"],
            [f"{chi_quadro:.2f}"],
            [f"{reduced_chi:.2f}"]
        ]
        row_labels = ["μ", "σ", "rate", "bkg", "χ²", "χ² ridotto"]
        table = plt.table(cellText=cell_text, rowLabels=row_labels,
                          loc='upper right', cellLoc='center')
        table.scale(1.2, 1.5)
        plt.tight_layout()
        plt.show()

        # Plot dei residui
        plt.figure(figsize=(8, 4))
        plt.errorbar(bin_centers_fit, residui, yerr=sigma_counts_fit, fmt='o', label='Residui', capsize=2)
        plt.axhline(0, color='black', linestyle='--', lw=1)
        plt.xlabel(xlabel)
        plt.ylabel('Residui (data - fit)')
        plt.title("Residui del fit")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    parametri = np.array([mu, sigma, rate, bkg])
    incertezze = np.array([mu_unc, sigma_unc, rate_unc, bkg_unc])
    ints = np.array([integral, integral_unc])
    plot_data = np.array([x_fit, y_fit, bin_centers, counts])

    return parametri, incertezze, residui, chi_quadro, reduced_chi, ints, plot_data


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

#FIT LINEARE
def linear(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', plot=False):
    print("This fit returns a list which contains, in order:\n"
      "- A numpy array with the parameters\n"
      "- A numpy array with the uncertainties\n"
      "- A numpy array with the residuals\n"
      "- The chi squared\n"
      "- The reduced chi squared \n")
    
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

    m_guess = (y[-1] - y[0]) / (x[-1] - x[0])
    q_guess = np.mean(y)
    initial_guess = [m_guess, q_guess]

    if fit_with_weights:
        params, cov_matrix = curve_fit(retta, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True)
    else:
        params, cov_matrix = curve_fit(retta, x, y, p0=initial_guess)

    m, q = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    m_unc, q_unc = uncertainties

    residui = np.array(y - retta(x, *params))

    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    dof = len(x) - len(params)
    chi_squared_reduced = chi_squared / dof

    print(f"m = {m} ± {m_unc}")
    print(f"q = {q} ± {q_unc}")
    print(f'Chi-squared = {chi_squared}')
    print(f'Reduced chi-squared = {chi_squared_reduced}')

    x_fit = np.linspace(x.min(), x.max(), 1000)

    if plot:
        fig = plt.figure(figsize=(7, 8))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 0.5, 5, 0.5, 1])

        ax_table = fig.add_subplot(gs[:2, 0])
        ax_table.axis('tight')
        ax_table.axis('off')

        data = [
            ["m", f"{m:.3f} ± {m_unc:.3f}"],
            ["q", f"{q:.3f} ± {q_unc:.3f}"],
            ["Chi²", f"{chi_squared:.8f}"],
            ["Chi² rid.", f"{chi_squared_reduced:.8f}"]
        ]

        table = ax_table.table(
            cellText=data,
            colLabels=["Parametro", "Valore"],
            loc='center',
            cellLoc='center',
            colColours=["#4CAF50", "#4CAF50"],
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(data[0]))))

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            if row == 0:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor("lightblue")

        ax1 = fig.add_subplot(gs[2, 0])
        ax1.errorbar(x, y, xerr=sx if np.any(sx != 0) else None,
                     yerr=sy if np.any(sy != 0) else None,
                     fmt='o', color='black', label='Data', markersize=3, capsize=2)
        ax1.plot(x_fit, retta(x_fit, *params), color='red', label='Linear fit', lw=1.2)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(titolo)
        ax1.legend()
        ax1.grid(alpha=0.5)

        ax2 = fig.add_subplot(gs[3:, 0], sharex=ax1)
        ax2.errorbar(x, residui, color='black', label='Residuals', markersize=3, fmt='o')
        ax2.axhline(0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("(data - fit)")
        ax2.grid(alpha=0.5)
        ax2.legend()

    parametri = np.array([m, q])
    incertezze = np.array([m_unc, q_unc])

    return parametri, incertezze, residui, chi_squared, chi_squared_reduced

# FIT ESPONEZIONALE
def exponential(x, y, sx=None, sy=None, tipo="decrescente", xlabel="X-axis", ylabel="Y-axis", titolo='title', plot=False):
    print("This fit returns a list which contains, in order:\n"
      "- A numpy array with the parameters\n"
      "- A numpy array with the uncertainties\n"
      "- A numpy array with the residuals\n"
      "- The chi squared\n"
      "- The reduced chi squared \n")
    
    if tipo == "crescente":
        fit_func = exp_pos
    elif tipo == "decrescente":
        fit_func = exp_neg
    else:
        raise ValueError("Tipo deve essere 'crescente' o 'decrescente'.")

    if sx is None:
        sx = np.zeros_like(x)
    if sy is None:
        sy = np.zeros_like(y)

    # Pesi
    if np.any(sy != 0):
        sigma = sy
    else:
        sigma = None

    # Fit con curve_fit
    p0 = [np.max(y)-np.min(y), (np.max(x)-np.min(x))/2, np.min(y)]
    params, cov = curve_fit(fit_func, x, y, sigma=sigma, absolute_sigma=True, p0=p0)

    A, tau, f0 = params
    perr = np.sqrt(np.diag(cov))
    A_unc, tau_unc, f0_unc = perr

    y_fit = fit_func(x, *params)
    residui = res(y, y_fit)

    # Chi quadro
    if sigma is not None:
        chi_squared = np.sum(((y - y_fit) / sigma) ** 2)
    else:
        chi_squared = np.sum((y - y_fit) ** 2)
    dof = len(x) - 3
    chi_squared_reduced = chi_squared / dof if dof > 0 else 0

    print("Parametri ottimizzati:")
    print(f"A = {A} ± {A_unc}")
    print(f"tau = {tau} ± {tau_unc}")
    print(f"f0 = {f0} ± {f0_unc}")
    print(f"Chi-squared = {chi_squared}")
    print(f"Reduced Chi-squared = {chi_squared_reduced}")

    if plot:
        fig = plt.figure(figsize=(7, 8))
        if sigma is not None:
            plt.errorbar(x, y, yerr=sigma, fmt='o', label='Data', capsize=2)
        else:
            plt.errorbar(x, y, fmt='o', label='Data', capsize=2, markersize=3)
        plt.plot(x, y_fit, color='red', label='Exponential fit')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titolo)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

        # Plot residui
        plt.figure()
        plt.errorbar(x, residui, label='Residuals', alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(xlabel)
        plt.ylabel("Residuals")
        plt.title("Residuals of the exponential fit")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    parametri = np.array([A, tau, f0])
    incertezze = np.array([A_unc, tau_unc, f0_unc])

    return parametri, incertezze, residui, chi_squared, chi_squared_reduced

#FIT PARABOLICO
def parabolic(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', plot=False):
    print("This fit returns a list which contains, in order:\n"
        "- A numpy array with the parameters\n"
        "- A numpy array with the uncertainties\n"
        "- A numpy array with the residuals\n"
        "- The chi squared\n"
        "- The reduced chi squared \n")

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

    # Guess initial parameters for a, b, and c
    a_guess = (y[-1] - 2 * y[-2] + y[-3]) / ((x[-1] - x[-2]) * (x[-2] - x[-3]))  # Second derivative estimate
    b_guess = (y[-1] - y[-2]) / (x[-1] - x[-2]) - a_guess * (x[-1] + x[-2])  # First derivative estimate
    c_guess = y[0]  # Estimate the intercept
    initial_guess = [a_guess, b_guess, c_guess]

    if fit_with_weights:
        params, cov_matrix = curve_fit(parabola, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True)
    else:
        params, cov_matrix = curve_fit(parabola, x, y, p0=initial_guess)

    a, b, c = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    a_unc, b_unc, c_unc = uncertainties

    residui = y - parabola(x, *params)

    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    dof = len(x) - len(params)
    chi_squared_reduced = chi_squared / dof

    print(f"a = {a} ± {a_unc}")
    print(f"b = {b} ± {b_unc}")
    print(f"c = {c} ± {c_unc}")
    print(f'Chi-squared = {chi_squared}')
    print(f'Reduced chi-squared = {chi_squared_reduced}')

    x_fit = np.linspace(x.min(), x.max(), 1000)

    if plot:
        fig = plt.figure(figsize=(7, 8))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 0.5, 5, 0.5, 1])

        ax_table = fig.add_subplot(gs[:2, 0])
        ax_table.axis('tight')
        ax_table.axis('off')

        data = [
            ["a", f"{a:.3f} ± {a_unc:.3f}"],
            ["b", f"{b:.3f} ± {b_unc:.3f}"],
            ["c", f"{c:.3f} ± {c_unc:.3f}"],
            ["Chi²", f"{chi_squared:.8f}"],
            ["Chi² rid.", f"{chi_squared_reduced:.8f}"]
        ]

        table = ax_table.table(
            cellText=data,
            colLabels=["Parametro", "Valore"],
            loc='center',
            cellLoc='center',
            colColours=["#4CAF50", "#4CAF50"],
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(data[0]))))

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            if row == 0:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor("lightblue")

        ax1 = fig.add_subplot(gs[2, 0])
        ax1.errorbar(x, y, xerr=sx if np.any(sx != 0) else None,
                     yerr=sy if np.any(sy != 0) else None,
                     fmt='o', color='black', label='Data', markersize=3, capsize=2)
        ax1.plot(x_fit, parabola(x_fit, *params), color='red', label='Parabolic fit', lw=1.2)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(titolo)
        ax1.legend()
        ax1.grid(alpha=0.5)

        ax2 = fig.add_subplot(gs[3:, 0], sharex=ax1)
        ax2.errorbar(x, residui, color='black', label='Residuals', fmt='o', markersize=3, capsize=2)
        ax2.axhline(0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("(data - fit)")
        ax2.grid(alpha=0.5)
        ax2.legend()

    parametri = np.array([a, b, c])
    incertezze = np.array([a_unc, b_unc, c_unc])

    return parametri, incertezze, residui, chi_squared, chi_squared_reduced

#Fit Lorentziana
def lorentzian(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis"):
    print("This fit returns a list which contains, in order:\n"
        "- A numpy array with the parameters\n"
        "- A numpy array with the uncertainties\n"
        "- A numpy array with the residuals\n"
        "- The chi squared\n"
        "- The reduced chi squared \n")

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

    a, gamma, x0 = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    a_unc, gamma_unc, x0_unc = uncertainties

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
    print(f"A = {a} ± {a_unc}")
    print(f"gamma = {gamma} ± {gamma_unc}")
    print(f"x0 = {x0} ± {x0_unc}")
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
        plt.errorbar(x, y, color='black', label='Data', fmt='o', markersize=3, capsize=2)
    
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
                     markersize=3, capsize=2)
    else:
        plt.erorrbar(x, residui, color='black', alpha=0.6, label='Residuals', fmt='o', markersize=3, capsize=2)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(f"(data - fit)")
    plt.title("Residuals")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    parametri = np.array([a, gamma, x0])
    incertezze = np.array([a_unc, gamma_unc, x0_unc])

    return parametri, incertezze, residui, chi_squared, chi_squared_reduced

#FIT BREIT-WIGNER
def breitwigner(x, y, sx=None, sy=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', plot=False):
    print("This fit returns a list which contains, in order:\n"
      "- A numpy array with the parameters\n"
      "- A numpy array with the uncertainties\n"
      "- A numpy array with the residuals\n"
      "- The chi squared\n"
      "- The reduced chi squared \n")

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

    initial_guess = [1, 1, np.mean(x)]

    if fit_with_weights:
        params, cov_matrix = curve_fit(wigner, x, y, p0=initial_guess, sigma=sigma_weights, absolute_sigma=True)
    else:
        params, cov_matrix = curve_fit(wigner, x, y, p0=initial_guess)

    a, gamma, x0 = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    a_unc, gamma_unc, x0_unc = uncertainties

    residui = y - wigner(x, *params)

    if fit_with_weights:
        chi_squared = np.sum(((residui / sigma_weights) ** 2))
    else:
        chi_squared = np.sum((residui ** 2) / np.var(y))
    dof = len(x) - len(params)
    chi_squared_reduced = chi_squared / dof

    print("Parametri ottimizzati:")
    print("-----------------------------------------------")
    print(f"a = {a} ± {a_unc}")
    print(f"gamma = {gamma} ± {gamma_unc}")
    print(f"x0 = {x0} ± {x0_unc}")
    print(f"Chi-squared = {chi_squared}")
    print(f"Reduced Chi-squared = {chi_squared_reduced}")

    x_fit = np.linspace(x.min(), x.max(), 1000)
    y_fit = wigner(x_fit, *params)

    if plot:
        fig = plt.figure(figsize=(7, 8))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 0.5, 5, 0.5, 1])

        # Tabella con parametri
        ax_table = fig.add_subplot(gs[:2, 0])
        ax_table.axis('tight')
        ax_table.axis('off')

        data = [
            ["a", f"{a:.3f} ± {a_unc:.3f}"],
            [r"$\gamma$", f"{gamma:.3f} ± {gamma_unc:.3f}"],
            [r"$x_0$", f"{x0:.3f} ± {x0_unc:.3f}"],
            [r"$\chi^2$", f"{chi_squared:.8f}"],
            [r"$\chi^2$/dof", f"{chi_squared_reduced:.8f}"]
        ]

        table = ax_table.table(
            cellText=data,
            colLabels=["Parametro", "Valore"],
            loc='center',
            cellLoc='center',
            colColours=["#4CAF50", "#4CAF50"],
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(data[0]))))

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            if row == 0:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor("lightblue")

        # Plot fit
        ax1 = fig.add_subplot(gs[2, 0])
        ax1.errorbar(x, y, xerr=sx if np.any(sx != 0) else None,
                     yerr=sy if np.any(sy != 0) else None,
                     fmt='o', color='black', label='Data', markersize=3, capsize=2)
        ax1.plot(x_fit, y_fit, color='red', label='Breit-Wigner fit', lw=1.5)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(titolo)
        ax1.legend()
        ax1.grid(alpha=0.5)

        # Plot residui
        ax2 = fig.add_subplot(gs[3:, 0], sharex=ax1)
        ax2.errorbar(x, residui, color='black', label='Residuals', fmt='o', markersize=3, capsize=2)
        ax2.axhline(0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("(data - fit)")
        ax2.grid(alpha=0.5)
        ax2.legend()

    parametri = np.array([a, gamma, x0])
    incertezze = np.array([a_unc, gamma_unc, x0_unc])

    return parametri, incertezze, residui, chi_squared, chi_squared_reduced

#LOGNORMALE
def lognormal(data=None, bin_centers=None, counts=None, xlabel="X-axis", ylabel="Y-axis", titolo='title', xmin=None, xmax=None, x1=None, x2=None, b=None, n=None, plot=False):
    print("This fit returns a list which contains, in order:\n"
      "- A numpy array with the parameters\n"
      "- A numpy array with the uncertainties\n"
      "- A numpy array with the residuals\n"
      "- The chi squared\n"
      "- The reduced chi squared \n"
      "- The integral of the histogram in the range mu ± n*sigma\n"
      "- The plot data (x_fit, y_fit, bin_centers, counts) if you need to plot other thing\n")

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
    params, cov_matrix = curve_fit(l_norm, bin_centers_fit, counts_fit, p0=initial_guess, sigma=sigma_counts_fit, absolute_sigma=True)
    amp, mu, sigma = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    amp_unc, mu_unc, sigma_unc = uncertainties

    fit_values = l_norm(bin_centers_fit, *params)
    chi_squared = np.sum(((counts_fit - fit_values) / sigma_counts_fit) ** 2)
    dof = len(counts_fit) - len(params)
    chi_squared_reduced = chi_squared / dof

    if n is not None:
        lower_bound, upper_bound = np.exp(mu - n * sigma), np.exp(mu + n * sigma)
        bins_to_integrate = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
        integral = int(np.sum(counts[bins_to_integrate]))
        integral_unc = int(np.sqrt(np.sum(sigma_counts[bins_to_integrate]**2)))
    else:
        integral, integral_unc = None, None

    x_fit = np.linspace(min(bin_centers), max(bin_centers), 10000)
    y_fit = l_norm(x_fit, *params)

    print(f"Ampiezza = {amp} ± {amp_unc}")
    print(f"Media = {mu} ± {mu_unc}")
    print(f"Sigma = {sigma} ± {sigma_unc}")
    print(f'Chi-squared = {chi_squared}')
    print(f'Reduced chi-squared = {chi_squared_reduced}')
    if integral is not None:
        print(f"Conteggi entro {n}σ: {integral} ± {integral_unc}")

    if plot:
        fig = plt.figure(figsize=(7, 8))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 0.5, 5, 0.5, 1])

        ax_table = fig.add_subplot(gs[:2, 0])
        ax_table.axis('tight')
        ax_table.axis('off')

        data_table = [
            ["Ampiezza", f"{amp:.3f} ± {amp_unc:.3f}"],
            ["Media (μ)", f"{mu:.3f} ± {mu_unc:.3f}"],
            ["Sigma (σ)", f"{sigma:.3f} ± {sigma_unc:.3f}"],
            ["Chi²", f"{chi_squared:.8f}"],
            ["Chi² rid.", f"{chi_squared_reduced:.8f}"]
        ]
        if integral is not None:
            data_table.append([f"Conteggi entro {n}σ", f"{integral} ± {integral_unc}"])

        table = ax_table.table(
            cellText=data_table,
            colLabels=["Parametro", "Valore"],
            loc='center',
            cellLoc='center',
            colColours=["#4CAF50", "#4CAF50"],
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(data_table[0]))))

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            if row == 0:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor("lightblue")

        ax1 = fig.add_subplot(gs[2, 0])
        ax1.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, label="Data", color="gray", edgecolor='black')
        ax1.plot(x_fit, y_fit, color='red', label='Lognormal fit', lw=1.5)
        ax1.set_xlim(x1, x2) if x1 is not None and x2 is not None else ax1.set_xlim(min(bin_centers), max(bin_centers))
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(titolo)
        ax1.legend()
        ax1.grid(alpha=0.5)

        ax2 = fig.add_subplot(gs[3:, 0], sharex=ax1)
        residuals = counts_fit - l_norm(bin_centers_fit, *params)
        ax2.errorbar(bin_centers_fit, residuals, color='black', label='Residuals', fmt='o', markersize=3, capsize=2)
        ax2.axhline(0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("(data - fit)")
        ax2.grid(alpha=0.5)
        ax2.legend()

    parametri = np.array(params)
    incertezze = np.array(uncertainties)

    return parametri, incertezze, chi_squared, chi_squared_reduced, [integral, integral_unc], [x_fit, y_fit, bin_centers, counts]

# FIT DIAGRAMMA DI BODE
def bode(filename, tipo='basso', xlabel="Frequenza (Hz)", ylabel="Guadagno (dB)", titolo='Fit filtro', plot=False):
    # Lettura dati da file
    dati = np.loadtxt(filename)
    frq, vin, vout = dati[:, 0], dati[:, 1], dati[:, 2]

    # Calcolo guadagno in dB
    gain_dB = 20 * np.log10(vout / vin)

    # Definizione modelli
    def low_pass(f, f_cut):
        return 20 * np.log10(1 / np.sqrt(1 + (f / f_cut)**2))

    def high_pass(f, f_cut):
        return 20 * np.log10(1 / np.sqrt(1 + (f_cut**2 / f**2)))

    def band_pass(f, f0, gamma, A):
        return 20 * np.log10(A * (f * gamma) / np.sqrt((f**2 - f0**2)**2 + (f * gamma)**2))

    # Scelta modello in base al tipo di filtro
    if tipo == 'basso':
        model = low_pass
        guess = [1000]
    elif tipo == 'alto':
        model = high_pass
        guess = [10000]
    elif tipo == 'banda':
        model = band_pass
        guess = [1000, 1000, 1]  # f0, gamma, A
    else:
        raise ValueError("Tipo di filtro non valido. Usa 'basso', 'alto' o 'banda'.")

    # Fit
    popt, pcov = curve_fit(model, frq, gain_dB, p0=guess)
    err = np.sqrt(np.diag(pcov))

    # Calcolo residui e chi^2
    fit_vals = model(frq, *popt)
    residui = gain_dB - fit_vals
    chi2 = np.sum(residui**2 / np.var(gain_dB))
    chi2_red = chi2 / (len(frq) - len(popt))

    # Stampa risultati
    print(f"f_cut: {popt[0]:.3f} ± {err[0]:.3f}")
    print(f"Chi² = {chi2:.4f}")
    print(f"Chi² ridotto = {chi2_red:.4f}")

    # Plot
    if plot:
        frq_fit = np.logspace(np.log10(frq.min()), np.log10(frq.max()), 1000)
        fit_curve = model(frq_fit, *popt)

        fig = plt.figure(figsize=(7, 8))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 0.5, 5, 0.5, 1])

        # Tabella
        ax_table = fig.add_subplot(gs[:2, 0])
        ax_table.axis('tight')
        ax_table.axis('off')
        table_data = [['f_cut', f"{popt[0]:.3f} ± {err[0]:.3f}"]]
        table_data += [["Chi²", f"{chi2:.4f}"], ["Chi² rid.", f"{chi2_red:.4f}"]]
        table = ax_table.table(
            cellText=table_data,
            colLabels=["Parametro", "Valore"],
            loc='center',
            cellLoc='center',
            colColours=["#4CAF50", "#4CAF50"],
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(2)))
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            if row == 0:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor("lightblue")

        # Fit
        ax1 = fig.add_subplot(gs[2, 0])
        ax1.errorbar(frq, gain_dB, color='black', label='Dati', fmt='o', markersize=3, capsize=2)
        ax1.plot(frq_fit, fit_curve, color='red', label='Fit')
        ax1.set_xscale('log')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(titolo)
        ax1.grid(alpha=0.5)
        ax1.legend()

        # Residui
        ax2 = fig.add_subplot(gs[3:, 0], sharex=ax1)
        ax2.errorbar(frq, residui, color='black', label='Residui', fmt='o', markersize=3, capsize=2)
        ax2.axhline(0, color='red', linestyle='--', lw=1)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Residui")
        ax2.grid(alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    parametri = np.array(popt)
    incertezze = np.array(err)

    return parametri, incertezze, residui, chi2, chi2_red