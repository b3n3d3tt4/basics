Basic Physics Analysis

SHORT DESCRIPTION: little python library for the data analysis of simple university experiments. 

It is so far provided with 
  - gaussian fitting,
  - gaussian and exponential convolution fitting
  - background subtraction (for example of background spectra from emission spectra)
  - Compton edge fitting (with an errorfunction)
  - linear fitting
  - exponential fitting
  - parabolic fitting
  - Lorentzian fitting
  - Breit-Wigner fitting
  - lognormal fitting
  - Bode diagram fitting.

It is given as a .py file and once imported (import basicfunc as bf, for example) it can be used in your code for the analysis.

For example if you want to use the function called normal to fit a gaussian function over some data with a normal trend you can write in your code what follows: 

  bf.normal(data_to_fit, 'xlabel', 'ylabel', 'title', xmin, xmax, number_of_bins) 
  
  and it will return you:
  
    1. A numpy array with the parameters
    2. A numpy array with the uncertainties
    3. A numpy array with the residuals
    4. The chi squared\n"
    5. The reduced chi squared \n"
    6. The integral of the histogram in the range mu ± n*sigma\n"
    7. The plot data (x_fit, y_fit, bin_centers, counts) if you need to plot other thing\n

If you want to use the linear regression function your code will be:

  bpa.linear_regression(datax, datay, errorx, errory, 'xlabel', 'ylabel')
  
  IMPORTANT: if you don't specify any errorx or errory they will not be taken into consideration while if you specify just one of them this will be taken into account but the other one will not.
  
  it will return you:
  
    1. the parameters of the straight line which best approxximates the data (printed)
    2. the chi squared and the normalized chi squared (printed)
    3. a plot with the data and the fitted line (printed)
    4. a plot with the residuals of the fitted line (printed)
    5. a list with angular coefficient, intercept, residuals, chi-squared and reduced-chi-squared (memorised in a list you decide - i.e. the variable to which you associate the function)
    
