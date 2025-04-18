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

  bf.normal(data=data_to_fit, xlabel="X-axis", ylabel="Y-axis", titolo='title', xmin=xmi, xmax=xma, x1=x_1, x2=x_2, b=k, n=n_i, plot=True) 
  
  and it will return you:
  
    1. A numpy array with the parameters
    2. A numpy array with the uncertainties
    3. A numpy array with the residuals
    4. The chi squared 
    5. The reduced chi squared
    6. The integral of the histogram in the range mu ± n*sigma
    7. The plot data (x_fit, y_fit, bin_centers, counts) if you need to plot other thing

Morevoer il will provide you the plot within x1 and x2 with the labels and title you set, with a gaussina fitted on top within xmin and xmax

If you want to use the linear regression function your code will be:

  bf.linear(datax, datay, errorx, errory, 'xlabel', 'ylabel', titolo='title', plot=True)
  
  IMPORTANT: if you don't specify any errorx or errory they will not be taken into consideration while if you specify just one of them this will be taken into account but the other one will not.
  
  it will return you:
  
    1. A numpy array with the parameters
    2. A numpy array with the uncertainties
    3. A numpy array with the residuals
    4. The chi squared
    5. The reduced chi squared

Moreover il will provide you a plot with the labels and title you provided with the data and the fitted function, the residuals and a table with the parameters
