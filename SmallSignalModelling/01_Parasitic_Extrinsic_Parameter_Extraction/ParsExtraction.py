import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
from skrf import media
import skrf as rf
from skrf import Network,network


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rc('axes',  titlesize = 20)
plt.rc('axes',  labelsize = 20)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)




class Open:
    """
    Open Class
    ==========
    A class designed to analyze and extract parasitic capacitances from the S-parameters of open structures in Network format. 
    This class is optimized for use with the scikit-rf library, commonly employed in RF and microwave engineering.

    The class focuses on the analysis of the imaginary part of Y-parameters (admittance) as a function of angular frequency (ω). 
    It utilizes linear curve fitting techniques on the Y-parameters to derive parasitic capacitance values, essential in high-frequency circuit analysis.

    Attributes
    ----------
    ntw : Network
        An instance or a list of instances of scikit-rf's Network class, representing the S-parameters of open structures.
        These instances should contain valid S-parameter data and corresponding frequency information.

    Methods
    -------
    __init__(self, ntw):
        Constructor that initializes the Open class with a single Network instance or a list of them. 
        It calculates the angular frequency based on the provided frequency data and performs linear fitting 
        on the imaginary part of the Y-parameters.

    linear_fit(self, i, j, param_key):
        Performs linear regression fitting on the specified Y-parameter based on its matrix indices (i, j).
        Stores the computed coefficients and the linear fit results for later use. Caching is implemented to 
        avoid redundant calculations.

    Cpg(self):
        Computes and returns the parasitic capacitance at the gate pad (port 1).

    Cpd(self):
        Computes and returns the parasitic capacitance at the drain pad (port 2).

    Cpgd(self):
        Retrieves the cached value of the parasitic capacitance between the gate and drain pads.

    Cpdg(self):
        Retrieves the cached value of the parasitic capacitance between the drain and gate pads.

    Cgd(self):
        Calculates the average of the parasitic capacitances between gate and drain pads.

    parasitic_caps(self):
        A property that returns a list of the computed parasitic capacitances [Cpg, Cpd, Cgd]. 
        It uses a lazy-loading approach to calculate these values only once and cache them.

    help(self):
        Prints the class documentation, providing an overview of the class's functionality and usage instructions.

    Usage
    -----
    To utilize this class, create an instance with a Network or list of Networks, and access the methods to analyze 
    parasitic capacitances:
    
        network = rf.Network(s=s_parameters, z0=50, f=frequencies, name='network_name')
        open_circuit = Open(network)
        capacitances = open_circuit.parasitic_caps  # List of parasitic capacitances

    Note
    ----
    This class assumes the input Network instances contain valid S-parameters and frequency data.
    It is tailored to work seamlessly with the scikit-rf library's Network format.

    References
    ----------
    1. Andreas Alt et.al., "Transistor Modelling", IEEE Microwave Magazine, 2013.
    """
    
    def __init__(self, ntw):
        self.ntw   =  ntw
        self.y     =  ntw.y.imag # Imaginary part of admittance
        self.freq  =  ntw.f  
        self.w     =  2*np.pi*self.freq
        self._cp   = None
        self.coeff = {}  # cache for storing computed coefficients
        self.yLin  = {}  # cahce for storing linear fit to y-parameters 
        
        if not self.coeff:
            # Perform linear fits for specific Y-parameters
            for key in ['y11', 'y12', 'y21', 'y22']:
                i,j = int(key[1])-1, int(key[2])-1
                self.linear_fit(i,j,key)
        
                
    @property
    def yp(self):
        return self.y
    
  
    def linear_fit(self, i,j, param_key):
        
        # check if coefficients have already been computed
        if param_key not in self.coeff:
            y_param = self.y[:,i,j]
            model   = LinearRegression().fit(self.w.reshape((-1,1)),y_param) 
            y_lin   = model.coef_ *self.w + model.intercept_ 
            
            # Store the coefficients and linear Fit in cache
            self.coeff[param_key] = model.coef_
            self.yLin[param_key]  = y_lin
        
    
    
    def Cpg(self):
        return self.coeff['y11'] - self.Cgd()
    
    def Cpd(self):
        return self.coeff['y22'] - self.Cgd()
    
    def Cpgd(self):
        return self.coeff['y21']
    
    def Cpdg(self):
        return self.coeff['y12']
    
    def Cgd(self):
        c_av  = -(self.Cpgd() + self.Cpdg()) /2  # The negative sign comes from Y21=Y12=-jωCpgd
        return c_av
    
    @property 
    def parasitic_caps(self):
        return [self.Cpg()[0], self.Cpd()[0], self.Cgd()[0]]
    
    
    @property
    def help(self):
        print(self.__doc__)

        



    
class Short:
    """
    Short Class
    ===========
    A class designed to extract parasitic inductances from the S-parameters of short structures in Network format. 
    This class is specifically optimized for use with the scikit-rf library, a common tool in RF and microwave engineering.

    The class focuses on the analysis of the imaginary part of Z-parameters (impedance) as a function of angular frequency (ω). 
    It applies linear curve fitting to the imaginary part of Z-parameters to deduce parasitic inductance values, 
    which are crucial in the characterization of high-frequency circuits.

    Attributes
    ----------
    ntw : Network
        An instance or a list of instances of scikit-rf's Network class, representing the S-parameters of short structures.
        These instances should contain valid S-parameter data and corresponding frequency information.

    Methods
    -------
    __init__(self, ntw):
        Constructor for initializing the Short class with a single Network instance or a list of them. 
        It computes angular frequency based on the provided frequency data and performs linear fitting 
        on the imaginary part of the Z-parameters.

    linear_fit(self, i, j, param_key):
        Conducts a linear regression fitting on the specified Z-parameter based on its matrix indices (i, j).
        Stores the computed coefficients and the linear fit results for future use. Caching is implemented to 
        minimize redundant calculations.

    Ls(self):
        Computes and returns the average parasitic series inductance based on Z-parameter linear fits.

    Lg(self):
        Calculates and returns the parasitic inductance at port 1.

    Ld(self):
        Calculates and returns the parasitic inductance at port 2.

    parasitic_inds(self):
        Returns a list of computed parasitic inductances [Lg, Ld, Ls].

    Usage
    -----
    To use this class, create an instance with a Network or list of Networks, and access the methods to analyze 
    parasitic inductances:
    
        network = rf.Network(s=s_parameters, z0=50, f=frequencies, name='network_name')
        short_circuit = Short(network)
        inductances = short_circuit.parasitic_inds

    Note
    ----
    This class assumes that the input Network instances contain valid S-parameters and frequency data.
    It is tailored to work efficiently with the scikit-rf library's Network format.

    References
    ----------
    1. Andreas Alt, et.al., "Transistor Modelling", IEEE Microwave Magazine, 2013.
    """
    
    def __init__(self, ntw):
        self.ntw      = ntw
        self.z        = self.ntw.z.imag                  # Imaginary part of impedance
        self.freq     = self.ntw.f
        self.w        = 2 * np.pi * self.freq
        self._inds    = None
        self.coeff    = {}
        self.zLin     = {}
        
        if not self.coeff:
            # Perform linear fits for specific Z-parameters
            for key in ['z11', 'z12', 'z21', 'z22']:
                i, j = int(key[1])-1, int(key[2])-1
                self.linear_fit(i, j, key)         

    @property
    def zp(self):
        return self.z

    def linear_fit(self, i,j, param_key):
        
        z_param = self.z[:, i, j]
        model = LinearRegression().fit(self.w.reshape((-1, 1)), z_param)
        z_lin = model.coef_ * self.w + model.intercept_
        
        self.coeff[param_key] = model.coef_
        self.zLin[param_key] = z_lin
    
    def Ls(self):
        Ls21  = self.coeff['z21']
        Ls12  = self.coeff['z12']
        return (Ls21 + Ls12) / 2
    
    def Lg(self):
        return self.coeff['z11'] - self.Ls()
    
    def Ld(self):
        return self.coeff['z22'] - self.Ls()
    
    @property
    def parasitic_inds(self):
        return [self.Lg()[0], self.Ld()[0], self.Ls()[0]]
    
    
    @property
    def help(self):
        print(self.__doc__)
        
    








class cold_FET:
    """
    This class enables the extraction of parasitic components from cold FET measurements of FETs.
    
    Cold-FET measurements involve deactivating the transistor's gain mechanism by setting Vds = 0 V and
    varying Vgs. The transistor can exhibit capacitive or inductive behavior depending on the Vgs values.
    
    Capacitive Behavior:
    - Observed when Vgs is reverse-biased, typically below the threshold voltage.
    - S11 and S22 on the Smith chart become capacitive, indicating below the real axis.
    - From this behavior, parasitic pad capacitances can be extracted.

    Inductive Behavior:
    - Occurs when Vgs is forward-biased.
    - S11 and S22 are above the real axis and close to 0 impedance.
    - Parasitic inductances and Rs can be determined.
    - Care is needed to prevent damage due to increasing gate current.

    Bias Conditions:
    - The class focuses on non-destructive bias conditions for Vgs, i.e., above threshold and below 0 V.
    

    Attributes:
        ntw (Network): A Network object representing the SSM parameters.
        y (ndarray): Imaginary part of the admittance matrix of the network.
        z (ndarray): Imaginary part of the impedance matrix of the network.
        freq (ndarray): Frequency array from the network data.
        w (ndarray): Angular frequency (2 * pi * freq).
        bias (str): Bias condition, either 'reversed' or 'forward'.

    Methods:
        yp: Returns the imaginary part of the admittance matrix.
        zp: Returns the imaginary part of the impedance matrix.
        coeff(w, param): 
            Linear regression to find coefficients.
            Params:
                w (ndarray): Angular frequency array.
                param (ndarray): Parameter array for regression.
            Returns:
                ndarray: Coefficient from linear regression.
        parasitic_caps(method='White'): 
            Extracts parasitic capacitances.
            Params:
                method (str): Method for extraction ('White', 'Dambrine', or 'Taryani').
            Returns:
                list: List of parasitic capacitances [cpg, cpd, cb].
        parasitic_inds(method='Taryani'): 
            Extracts parasitic inductances.
            Params:
                method (str): Method for extraction ('Dambrine' or 'Taryani').
            Returns:
                list: List of parasitic inductances [lg, ld, ls].
    """  
    
    
    def __init__(self,ntw, bias='reversed'):
        self.ntw    = ntw
        self.y      = self.ntw.y.imag
        self.z      = self.ntw.z.imag
        self.freq   = self.ntw.f 
        self.w      = 2 * np.pi * self.freq
        self.bias   = bias
        
              
    @property
    def yp(self):
        return self.y
    
    @property
    def zp(self):
        return self.z
        
    def coeff(self, w, param):
        model = LinearRegression().fit(w.reshape((-1, 1)), param) 
        return model.coef_    
    

    def parasitic_caps(self, method='White'):

        if self.bias != 'reversed':
            warnings.warn("Capacitance extraction might be invalid for non-reversed bias.", RuntimeWarning)
        
        def _white(self):
            cb_1    =  - 3* self.coeff(self.w, self.y[:,0,1] ) 
            cb_2    =  - 3* self.coeff(self.w, self.y[:,1,0] ) 
            cb      = ( cb_1 + cb_2 ) / 2
            cpg     =  self.coeff(self.w, self.y[:,0,0])  - cb * 2 / 3 
            cpd     =  self.coeff(self.w, self.y[:,1,1])  - cb * 2 / 3
            return [cpg, cpd, cb]
        
        def _dambrine(self):
            cb_1    =  -  self.coeff(self.w, self.y[:,0,1] ) 
            cb_2    =  -  self.coeff(self.w, self.y[:,1,0] ) 
            cb      = ( cb_1 + cb_2 ) / 2
            cpg     =  self.coeff(self.w, self.y[:,0,0])  - cb * 2  
            cpd     =  self.coeff(self.w, self.y[:,1,1])  - cb 
            return [cpg, cpd, cb]
            
        def _taryani(self):
            cb_1    =  -  self.coeff(self.w, self.y[:,0,1] ) 
            cb_2    =  -  self.coeff(self.w, self.y[:,1,0] ) 
            cb      = ( cb_1 + cb_2 ) / 2
            cpg     =  self.coeff(self.w, self.y[:,0,0])  - cb * 2  
            cpd     = ( self.coeff(self.w, self.y[:,1,1])  - cb ) / 5
            return [cpg, cpd, cb]
        
        method_dict = {
            'White' : _white,
            'Dambrine': _dambrine,
            'Taryani': _taryani
            }        
        if method in method_dict:
            return method_dict[method]()
        else:
            raise ValueError(f"Method '{method}' not implemented in ColdFET class")
            
    
    def parasitic_inds(self, method='Taryani'):
        
        if self.bias != 'forward':
            warnings.warn("Inductance extraction might be invalid for non-forward bias.", RuntimeWarning)
            
        def _dambrine(self):
            ls_1    =  -  self.coeff(self.w, self.z[:,0,1] ) 
            ls_2    =  -  self.coeff(self.w, self.z[:,1,0] ) 
            ls      = ( ls_1 + ls_2 ) / 2
            lg      =  self.coeff(self.w, self.z[:,0,0])  - ls 
            ld      =  self.coeff(self.w, self.z[:,1,1])  - ls 
            return [lg, ld, ls]
            
        def _taryani(self):
            ls_1    =  -  self.coeff(self.w**2, self.w*self.z[:,0,1] ) 
            ls_2    =  -  self.coeff(self.w**2, self.w*self.z[:,1,0] ) 
            ls      = ( ls_1 + ls_2 ) / 2
            lg      =  self.coeff(self.w**2, self.w * self.z[:,0,0])  - ls   
            ld      =  self.coeff(self.w**2, self.w* self.z[:,1,1])   - ls 
            return [lg, ld, ls]        
             
        method_dict = {
            'Dambrine': _dambrine,
            'Taryani': _taryani
            }
        
        if method in method_dict:
            return method_dict[method]()
        else:
            raise ValueError(f"Method '{method}' not implemented in ColdFET class")    

            
  
            
            
            
                
        
        

















    
    