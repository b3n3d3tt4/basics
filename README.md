Basic Physics Analysis

SHORT DESCRIPTION: little python library for the data analysis of simple university experiments. 

It is so far provided with linear regression, gaussian fitting and background subtraction (for example of background spectra from emission spectra).

It is given as a .py file and once imported (import basic_physics_analysis as bpa, for example) it can be used in your code for the analysis.

For example if you want to use the function called normal to fit a gaussian function over some data with a normal trend you can write in your code what follows: 

  bpa.normal(data_to_fit, 'xlabel', 'ylabel', 'title', xmin, xmax, number_of_bins) 
  
  and it will return you:
  
    1. the parameters of the gaussian which best approxximates the data (printed)
    2. the chi squared and the normalized chi squared (printed)
    3. a plot with the data and the fitted gaussian (printed)
    4. a plot with the residuals of the fitted gaussian (printed)
    5. a list with amplitude, mean, standard deviation, chi squared and reduced chi squared (memorised in a list you decide - i.e. the variable to which you associate the function)

If you want to use the linear regression function your code will be:

  bpa.linear_regression(datax, datay, errorx, errory, 'xlabel', 'ylabel')
  
  IMPORTANT: if you don't specify any errorx or errory they will not be taken into consideration while if you specify just one of them this will be taken into account but the other one no.
  
  it will return you:
  
    1. the parameters of the straight line which best approxximates the data (printed)
    2. the chi squared and the normalized chi squared (printed)
    3. a plot with the data and the fitted line (printed)
    4. a plot with the residuals of the fitted line (printed)
    5. a list with angular coefficient, intercept, residuals, chi-squared and reduced-chi-squared (memorised in a list you decide - i.e. the variable to which you associate the function)
    
