# Basic Physics Analysis library

## 📝 Short description: 
Little python library for the data analysis of simple university experiments (mainly electronics and gamma spectroscopy experiments) 

## 👩‍🔬 Author
Benedetta Rasera 

`benedetta.rasera@studenti.unipd.it`, `raserab3nedetta@gmail.com`

Developed as part of the Master's Degree in Physics at the University of Padua

If you need any clarification or want to annotate possibile errors in the library please write me an email with "BASICFUNC LIBRARY - GitHub" in the object. If you are an UNIPD student please write me with your institutional account.

## ⚠️ WARNING: 
So far some comments are still in Italian, sooner or later I will translate them to English

## ✅ Included functions
It is so far provided with 
  - gaussian fitting,
  - background subtraction (for example of background spectra from emission spectra)
  - Compton edge fitting (with an errorfunction)
  - linear fitting
  - exponential fitting
  - parabolic fitting
  - Lorentzian fitting
  - Breit-Wigner fitting
  - lognormal fitting
  - Bode diagram fitting

A fit of a gaussian and an exponential convoluted will be added soon.

## 🔧 Requirements
- `numpy`
- `matplotlib`
- `scipy`

## 📦 How to Use
After importing the file in your code (e.g. as `import basicfunc as bf`), you can use any function directly.

In the repository you can also find a Jupyter Notebook called `bfexamples.ipynb` where there is an example for each of the functions of the library, please have a look at it.

## ⚙️ An example
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
  
  ⚠️ IMPORTANT: if you don't specify any errorx or errory they will not be taken into consideration while if you specify just one of them this will be taken into account but the other one will not.
  
  it will return you:
  
    1. A numpy array with the parameters
    2. A numpy array with the uncertainties
    3. A numpy array with the residuals
    4. The chi squared
    5. The reduced chi squared

Moreover il will provide you a plot (with the labels and title you wrote) with the data and the fitted function, the residuals and a table with the parameters
